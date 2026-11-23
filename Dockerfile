# EduCore Dockerfile
# Multi-stage build for the FastAPI application + Celery worker

FROM python:3.12-slim AS base

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-ara \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[streamlit]"

# Pre-download the embedding model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code
COPY . .

# Create storage directory
RUN mkdir -p /app/storage

# ── API server ──────────────────────────────────────────────────────
FROM base AS api
EXPOSE 8000
CMD ["uvicorn", "educore.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ── Celery worker ───────────────────────────────────────────────────
FROM base AS worker
CMD ["celery", "-A", "educore.celery_app", "worker", "--loglevel=info", "-Q", "educore.process"]

# ── Streamlit UI ────────────────────────────────────────────────────
FROM base AS streamlit
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
