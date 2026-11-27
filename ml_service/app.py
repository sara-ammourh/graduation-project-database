"""
Standalone ML Inference Service for GoAlgo Graph Extraction (YOLOv8 + CNN/EasyOCR).

Designed to run on a free high-RAM container host such as Hugging Face Spaces
(16 GB RAM, 2 vCPU free tier) while the main GoAlgo FastAPI backend runs on
Render Free (512 MB RAM) and proxies POST /graph/extract to this service via
ML_SERVICE_URL.
"""

import logging
import os
import sys
import tempfile
from threading import Lock
from typing import Optional

from fastapi import FastAPI, File, Header, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from ai.graph_model import GraphModel
from ai.readers.reader_loader import ReaderType
from schemas.requests import GraphNodeResponse, GraphResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH") or os.path.join(
    BACKEND_DIR, "models", "yolov8best.pt"
)
READER_TYPE_RAW = os.getenv("READER_TYPE") or "cnn"
ML_SERVICE_API_KEY = (os.getenv("ML_SERVICE_API_KEY") or "").strip()

app = FastAPI(title="GoAlgo ML Inference Service")

_model: Optional[GraphModel] = None
_model_lock = Lock()


def get_model() -> GraphModel:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                if not os.path.exists(YOLO_MODEL_PATH):
                    raise FileNotFoundError(f"YOLO weights missing: {YOLO_MODEL_PATH}")
                reader_type = ReaderType.from_str(READER_TYPE_RAW)
                _model = GraphModel(YOLO_MODEL_PATH, reader_type)
    return _model


@app.get("/health")
def health():
    try:
        model = get_model()
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "reader_type": READER_TYPE_RAW,
        }
    except Exception as exc:
        return {"status": "unhealthy", "error": str(exc)}


@app.post("/predict", response_model=GraphResponse)
async def predict(
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(default=None),
):
    if ML_SERVICE_API_KEY and x_api_key != ML_SERVICE_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    data = await file.read()
    suffix = os.path.splitext(file.filename or "graph.png")[1] or ".png"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            temp_path = tmp.name

        model = get_model()
        graph_nodes = await run_in_threadpool(model.predict_image, temp_path)
        edge_count = sum(len(n.neighbors) for n in graph_nodes) // 2

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
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
