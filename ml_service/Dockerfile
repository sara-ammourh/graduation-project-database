# Standalone Docker image for deploying GoAlgo's YOLOv8 + CNN/EasyOCR inference
# on Hugging Face Spaces Free Tier (16 GB RAM, 2 vCPU, port 7860).
# Build context should be the backend/ directory.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    READER_TYPE=cnn \
    YOLO_MODEL_PATH=/app/models/yolov8best.pt

WORKDIR /app

# System libraries required by OpenCV headless and Ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first to minimize image size and memory overhead
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install ML + FastAPI dependencies
RUN pip install --no-cache-dir \
    fastapi>=0.123.10 \
    uvicorn[standard]>=0.24.0 \
    python-multipart>=0.0.6 \
    pydantic>=2.0.0 \
    ultralytics \
    opencv-python-headless \
    easyocr \
    matplotlib

COPY ai ./ai
COPY schemas ./schemas
COPY models ./models
COPY ml_service ./ml_service

EXPOSE 7860

CMD ["uvicorn", "ml_service.app:app", "--host", "0.0.0.0", "--port", "7860"]
