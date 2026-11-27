import csv
import logging
import os
import tempfile
import time
from threading import Lock
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans

from app_featuers.plogic import PExp
from auth.utils import AuthPrincipal, get_current_principal
from schemas.requests import (
    ClusterRequest,
    ExpressionRequest,
    GraphNodeResponse,
    GraphResponse,
    IsEquivalentRequest,
    WhereRequest,
)
from services.storage import storage_client

load_dotenv()

logger = logging.getLogger(__name__)
router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH") or os.path.join(
    BASE_DIR, "models", "yolov8best.pt"
)
READER_TYPE_RAW = os.getenv("READER_TYPE") or "cnn"
ML_SERVICE_URL = (os.getenv("ML_SERVICE_URL") or "").strip().rstrip("/")
ML_SERVICE_API_KEY = (os.getenv("ML_SERVICE_API_KEY") or "").strip()
ENABLE_LOCAL_ML = os.getenv("ENABLE_LOCAL_ML", "true").lower() == "true"
ENABLE_CSV_TIMINGS = os.getenv("ENABLE_CSV_TIMINGS", "false").lower() == "true"
CSV_PATH = os.path.join(BASE_DIR, "graph_timings.csv")

graph_model = None
_model_lock = Lock()
_csv_lock = Lock()


class MLUnavailableError(RuntimeError):
    """Raised when heavy ML dependencies or model weights are not available."""


def get_graph_model():
    """Lazily load GraphModel only when local ML inference is invoked."""
    global graph_model
    if not ENABLE_LOCAL_ML:
        raise MLUnavailableError(
            "Local ML inference is disabled on this instance. "
            "Configure ML_SERVICE_URL to point to an external ML inference service."
        )

    if graph_model is None:
        with _model_lock:
            if graph_model is None:
                try:
                    from ai.graph_model import GraphModel
                    from ai.readers.reader_loader import ReaderType
                except ImportError as exc:
                    raise MLUnavailableError(
                        "ML dependencies (ultralytics, torch, opencv, easyocr) are not "
                        "installed on this lightweight backend instance. "
                        "Configure ML_SERVICE_URL for remote inference or install requirements-ml.txt."
                    ) from exc

                if not os.path.exists(YOLO_MODEL_PATH):
                    raise MLUnavailableError(
                        f"YOLO model weights not found at: {YOLO_MODEL_PATH}"
                    )

                reader_type = ReaderType.from_str(READER_TYPE_RAW)
                graph_model = GraphModel(YOLO_MODEL_PATH, reader_type)
    return graph_model


def log_timing(
    owner_id: str,
    filename: str,
    save_ms: float,
    infer_ms: float,
    total_ms: float,
):
    logger.info(
        "graph_extract owner=%s file=%s save_ms=%.2f infer_ms=%.2f total_ms=%.2f",
        owner_id,
        filename,
        save_ms,
        infer_ms,
        total_ms,
    )
    if not ENABLE_CSV_TIMINGS:
        return

    new_file = not os.path.exists(CSV_PATH)
    with _csv_lock:
        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(
                    [
                        "timestamp",
                        "user_id",
                        "filename",
                        "save_time_ms",
                        "inference_time_ms",
                        "total_time_ms",
                    ]
                )
            writer.writerow(
                [
                    round(time.time(), 3),
                    owner_id,
                    filename,
                    round(save_ms, 2),
                    round(infer_ms, 2),
                    round(total_ms, 2),
                ]
            )


async def _proxy_to_ml_service(
    filename: str, data: bytes, content_type: str
) -> GraphResponse:
    import httpx

    headers = {}
    if ML_SERVICE_API_KEY:
        headers["X-API-Key"] = ML_SERVICE_API_KEY

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{ML_SERVICE_URL}/predict",
                files={"file": (filename, data, content_type)},
                headers=headers,
            )
    except httpx.HTTPError as exc:
        logger.error("Failed to reach ML service at %s: %s", ML_SERVICE_URL, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph extraction ML service is currently unreachable. Please try again later.",
        ) from exc

    if response.status_code != 200:
        logger.error(
            "ML service returned %s: %s", response.status_code, response.text[:300]
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Graph extraction ML service failed to process the image.",
        )

    return GraphResponse(**response.json())


@router.post("/evaluate")
async def evaluate(inp: ExpressionRequest):
    exp = PExp(inp.expression).solve()
    return {
        "headers": exp.df.columns.to_list(),
        "data": exp.df.values.tolist(),
    }


@router.post("/is_equivalent")
async def is_equivalent(inp: IsEquivalentRequest):
    try:
        exp1 = PExp(inp.expression1).solve()
        exp2 = PExp(inp.expression2).solve()
        result = exp1 == exp2
        return {
            "equivalent": bool(result),
            "expression1": inp.expression1,
            "expression2": inp.expression2,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/where")
async def where(inp: WhereRequest):
    try:
        exp = PExp(inp.expression).solve()
        filtered_df = exp.where(**inp.conditions)
        return {
            "headers": filtered_df.columns.to_list(),
            "data": filtered_df.values.tolist(),
            "conditions": inp.conditions,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/cluster")
async def cluster(inp: ClusterRequest):
    points = np.array(inp.points)
    if len(points) == 0:
        raise HTTPException(status_code=400, detail="No points provided")

    if inp.algorithm == "kmeans":
        model = KMeans(n_clusters=inp.k, n_init="auto")
        labels = model.fit_predict(points).tolist()
        centroids = model.cluster_centers_.tolist()

    elif inp.algorithm == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=inp.k)
        labels = model.fit_predict(points).tolist()
        centroids = []
        for cid in set(labels):
            centroids.append(points[np.array(labels) == cid].mean(axis=0).tolist())

    elif inp.algorithm == "DBSCAN":
        model = DBSCAN(eps=inp.eps, min_samples=inp.minSamples)
        labels = model.fit_predict(points).tolist()
        centroids = []

    else:
        raise HTTPException(status_code=400, detail="Invalid algorithm")

    return {
        "labels": labels,
        "centroids": centroids,
        "algorithm": inp.algorithm,
    }


@router.post("/graph/extract", response_model=GraphResponse)
async def extract_graph(
    file: UploadFile = File(...),
    user_id: Optional[int] = Query(default=None),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    t_total_start = time.perf_counter()

    if principal.role == "user" and principal.user_id is not None:
        owner_segment = str(principal.user_id)
    else:
        owner_segment = principal.guest_id or "guest_temp"

    safe_filename = os.path.basename(file.filename)
    timestamp = int(time.time())
    stored_filename = f"graph_{timestamp}_{safe_filename}"

    t_save_start = time.perf_counter()
    data = await file.read()

    # Persist uploaded image to Supabase Storage for registered users (never to Render local disk)
    persisted_image_path: Optional[str] = None
    if principal.role == "user" and principal.user_id is not None:
        object_path = f"users/{principal.user_id}/{stored_filename}"
        persisted_image_path = await storage_client.upload_bytes(
            object_path=object_path,
            data=data,
            content_type=file.content_type,
        )
        if persisted_image_path is None:
            persisted_image_path = object_path

    save_ms = (time.perf_counter() - t_save_start) * 1000

    # Option 1: Proxy to external ML inference service (e.g. Hugging Face Spaces free tier)
    if ML_SERVICE_URL:
        t_infer_start = time.perf_counter()
        result = await _proxy_to_ml_service(safe_filename, data, file.content_type)
        infer_ms = (time.perf_counter() - t_infer_start) * 1000
        total_ms = (time.perf_counter() - t_total_start) * 1000
        await run_in_threadpool(
            log_timing, owner_segment, safe_filename, save_ms, infer_ms, total_ms
        )
        result.image_path = persisted_image_path
        return result

    # Option 2: Run local ML inference using an ephemeral temp file
    try:
        model = get_graph_model()
    except MLUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    suffix = os.path.splitext(safe_filename)[1] or ".png"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            temp_path = tmp.name

        t_infer_start = time.perf_counter()
        graph_nodes = await run_in_threadpool(model.predict_image, temp_path)
        infer_ms = (time.perf_counter() - t_infer_start) * 1000

        edge_count = sum(len(n.neighbors) for n in graph_nodes) // 2
        total_ms = (time.perf_counter() - t_total_start) * 1000

        await run_in_threadpool(
            log_timing,
            owner_segment,
            safe_filename,
            save_ms,
            infer_ms,
            total_ms,
        )

        return GraphResponse(
            nodes=[
                GraphNodeResponse(
                    id=n.id,
                    label=n.label,
                    pos=list(n.pos),
                    neighbors=n.neighbors,
                )
                for n in graph_nodes
            ],
            node_count=len(graph_nodes),
            edge_count=edge_count,
            image_path=persisted_image_path,
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@router.get("/graph/health")
async def graph_health():
    if ML_SERVICE_URL:
        try:
            import httpx

            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(f"{ML_SERVICE_URL}/health")
            return {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "mode": "remote",
                "ml_service_url": ML_SERVICE_URL,
                "remote_status": response.json()
                if response.headers.get("content-type", "").startswith(
                    "application/json"
                )
                else response.text,
            }
        except Exception as exc:
            return {
                "status": "unhealthy",
                "mode": "remote",
                "ml_service_url": ML_SERVICE_URL,
                "error": str(exc),
            }

    try:
        model = get_graph_model()
        return {
            "status": "healthy",
            "mode": "local",
            "model_loaded": model is not None,
            "model_path": YOLO_MODEL_PATH,
        }
    except Exception as e:
        return {
            "status": "unavailable",
            "mode": "local",
            "model_loaded": False,
            "error": str(e),
        }
