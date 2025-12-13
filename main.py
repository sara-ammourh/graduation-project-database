import logging
import os
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans

from ai.graph_model import GraphModel
from api.routes import router
from plogic import PExp

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

app = FastAPI(title="Graduation Project API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the model
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8best.pt")
graph_model = None


def get_graph_model():
    global graph_model
    if graph_model is None:
        model_path = os.getenv("YOLO_MODEL_PATH", YOLO_MODEL_PATH)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found at {model_path}")
        graph_model = GraphModel(model_path)
    return graph_model


@app.get("/")
def read_root():
    return {"message": "API is running"}


class ExpressionRequest(BaseModel):
    expression: str


class ClusterRequest(BaseModel):
    algorithm: str
    points: List[Tuple[float, float]]
    k: int = 1
    eps: float = 0.5
    minSamples: int = 5


class GraphNodeResponse(BaseModel):
    id: int
    label: str
    pos: List[float]
    neighbors: List[int]


class GraphResponse(BaseModel):
    nodes: List[GraphNodeResponse]
    node_count: int
    edge_count: int


@app.post("/evaluate")
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


@app.post("/cluster")
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


@app.post("/graph/extract", response_model=GraphResponse)
async def extract_graph(file: UploadFile = File(...)):
    try:
        model = get_graph_model()

        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, detail="File must be an image (png, jpg, jpeg)"
            )

        if not file.filename:
            raise HTTPException(status_code=400, detail="File must have a filename")

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(file.filename).suffix
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            logger.info(f"Processing graph image: {file.filename}")
            graph_nodes = model.predict_image(tmp_path)

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
                f"Extracted graph: {len(graph_nodes)} nodes, {edge_count} edges"
            )
            return response

        finally:
            os.unlink(tmp_path)

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


@app.get("/graph/health")
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


if __name__ == "__main__":
    import uvicorn

    from db.config import create_db_and_tables

    create_db_and_tables()
    uvicorn.run(app, host="0.0.0.0", port=8000)
