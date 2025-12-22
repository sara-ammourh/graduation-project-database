import csv
import logging
import os
import time
from gzip import READ
from threading import Lock

import numpy as np
from dotenv import load_dotenv
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans

from ai.graph_model import GraphModel
from ai.readers.reader_loader import ReaderType
from app_featuers.plogic import PExp
from schemas.requests import (
    ClusterRequest,
    ExpressionRequest,
    GraphNodeResponse,
    GraphResponse,
)

load_dotenv()

logger = logging.getLogger(__name__)
router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH") or ""
READER_TYPE = ReaderType.from_str(os.getenv("READER_TYPE") or "")
GRAPH_IMAGES_DIR = os.path.join(BASE_DIR, "graph_images")
CSV_PATH = os.path.join(BASE_DIR, "graph_timings.csv")

os.makedirs(GRAPH_IMAGES_DIR, exist_ok=True)

graph_model = None
_model_lock = Lock()
_csv_lock = Lock()


def get_graph_model():
    global graph_model
    if graph_model is None:
        with _model_lock:
            if graph_model is None:
                if not os.path.exists(YOLO_MODEL_PATH):
                    raise FileNotFoundError(YOLO_MODEL_PATH)
                graph_model = GraphModel(YOLO_MODEL_PATH, READER_TYPE)
    return graph_model


def save_file(path: str, data: bytes):
    with open(path, "wb") as f:
        f.write(data)


def log_timing(
    user_id: int,
    filename: str,
    save_ms: float,
    infer_ms: float,
    total_ms: float,
):
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
                    user_id,
                    filename,
                    round(save_ms, 2),
                    round(infer_ms, 2),
                    round(total_ms, 2),
                ]
            )


@router.post("/evaluate")
async def evaluate(inp: ExpressionRequest):
    exp = PExp(inp.expression).solve()
    return {
        "headers": exp.df.columns.to_list(),
        "data": exp.df.values.tolist(),
    }


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
async def extract_graph(file: UploadFile = File(...), user_id: int = Query(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    t_total_start = time.perf_counter()
    model = get_graph_model()

    user_dir = os.path.join(GRAPH_IMAGES_DIR, str(user_id))
    os.makedirs(user_dir, exist_ok=True)

    filename = f"graph_{int(time.time())}_{file.filename}"
    path = os.path.join(user_dir, filename)

    t_save_start = time.perf_counter()
    data = await file.read()
    await run_in_threadpool(save_file, path, data)
    save_ms = (time.perf_counter() - t_save_start) * 1000

    try:
        t_infer_start = time.perf_counter()
        graph_nodes = await run_in_threadpool(model.predict_image, path)
        infer_ms = (time.perf_counter() - t_infer_start) * 1000

        edge_count = sum(len(n.neighbors) for n in graph_nodes) // 2
        total_ms = (time.perf_counter() - t_total_start) * 1000

        await run_in_threadpool(
            log_timing,
            user_id,
            file.filename,
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
        )

    finally:
        if os.path.exists(path):
            os.remove(path)


@router.get("/graph/health")
async def graph_health():
    try:
        model = get_graph_model()
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "model_path": YOLO_MODEL_PATH,
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
