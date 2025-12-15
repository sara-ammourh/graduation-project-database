import logging
import os
import time

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile, Query
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans

from ai.graph_model import GraphModel
from schemas.requests import (
    ExpressionRequest,
    ClusterRequest,
    GraphResponse,
    GraphNodeResponse,
)
from plogic import PExp

logger = logging.getLogger(__name__)

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8best.pt")
GRAPH_IMAGES_DIR = os.path.join(BASE_DIR, "graph_images")

os.makedirs(GRAPH_IMAGES_DIR, exist_ok=True)

graph_model = None


def get_graph_model():
    global graph_model
    if graph_model is None:
        model_path = os.getenv("YOLO_MODEL_PATH", YOLO_MODEL_PATH)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found at {model_path}")
        graph_model = GraphModel(model_path)
    return graph_model


@router.post("/evaluate")
async def evaluate(inp: ExpressionRequest):
    try:
        exp0 = PExp(inp.expression).solve()
        response = {
            "headers": exp0.df.columns.to_list(),
            "data": exp0.df.values.tolist(),
        }
        logger.info("Evaluated: %s", inp.expression)
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/cluster")
async def cluster(inp: ClusterRequest):
    logger.info("Clustering: %s for %d points", inp.algorithm, len(inp.points))
    try:
        points = np.array(inp.points)
        if len(points) == 0:
            raise HTTPException(status_code=400, detail="No points provided.")
        if inp.algorithm == "kmeans":
            if inp.k <= 0:
                raise HTTPException(
                    status_code=400, detail="k must be a positive integer."
                )
            model = KMeans(n_clusters=inp.k, n_init="auto")
            model.fit(points)
            centroids = model.cluster_centers_.tolist()
            if model.labels_ is None:
                raise HTTPException(
                    status_code=400, detail="k must be a positive integer."
                )

            labels = model.labels_.tolist()
        elif inp.algorithm == "Agglomerative":
            if inp.k <= 0:
                raise HTTPException(
                    status_code=400, detail="k must be a positive integer."
                )
            model = AgglomerativeClustering(n_clusters=inp.k)
            labels = model.fit_predict(points).tolist()
            centroids = []
            for cluster_id in set(labels):
                cluster_points = points[np.array(labels) == cluster_id]
                centroids.append(cluster_points.mean(axis=0).tolist())
        elif inp.algorithm == "DBSCAN":
            model = DBSCAN(eps=inp.eps, min_samples=inp.minSamples)
            labels = model.fit_predict(points).tolist()
            centroids = []
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid clustering algorithm specified. Choose from: kmeans, Agglomerative, DBSCAN",
            )
        return {
            "labels": labels,
            "centroids": centroids,
            "algorithm": inp.algorithm,
        }
    except Exception as e:
        logger.error(f"Clustering error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")


@router.post("/graph/extract", response_model=GraphResponse)
async def extract_graph(file: UploadFile = File(...), user_id: int = Query(...)):
    try:
        model = get_graph_model()

        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, detail="File must be an image (png, jpg, jpeg)"
            )

        if not file.filename:
            raise HTTPException(status_code=400, detail="File must have a filename")

        # Create user-specific directory
        user_image_dir = os.path.join(GRAPH_IMAGES_DIR, str(user_id))
        os.makedirs(user_image_dir, exist_ok=True)

        # Save file to user directory
        saved_filename = f"graph_{int(time.time())}_{file.filename}"
        saved_filepath = os.path.join(user_image_dir, saved_filename)

        # Save the uploaded file
        content = await file.read()
        with open(saved_filepath, "wb") as f:
            f.write(content)

        try:
            logger.info(f"Processing graph image: {file.filename} for user {user_id}")
            graph_nodes = model.predict_image(saved_filepath)

            edge_count = sum(len(node.neighbors) for node in graph_nodes) // 2

            response = GraphResponse(
                nodes=[
                    GraphNodeResponse(
                        id=node.id,
                        label=node.label,
                        pos=list(node.pos),
                        neighbors=node.neighbors,
                    )
                    for node in graph_nodes
                ],
                node_count=len(graph_nodes),
                edge_count=edge_count,
            )

            logger.info(
                f"Extracted graph: {len(graph_nodes)} nodes, {edge_count} edges from {saved_filepath}"
            )
            return response

        except Exception:
            # Clean up file if processing fails
            if os.path.exists(saved_filepath):
                os.remove(saved_filepath)
            raise

    except FileNotFoundError as e:
        logger.error(f"Model file not found: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Model configuration error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Graph extraction error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Graph extraction failed: {str(e)}"
        )


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
