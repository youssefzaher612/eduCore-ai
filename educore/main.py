"""
FastAPI application entry point.

Creates the app, mounts routes, and runs DB migrations on startup.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from educore.api.routes import router
from educore.config import settings

# ── Logging ─────────────────────────────────────────────────────────
from pathlib import Path
log_dir = Path(settings.STORAGE_DIR) / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

file_handler = logging.FileHandler(log_dir / "api.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(), file_handler]
)
logger = logging.getLogger("educore")


# ── App creation ────────────────────────────────────────────────────

app = FastAPI(
    title="EduCore",
    description="RAG-powered educational assistant with Student and Teacher sections",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow Streamlit and other frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes
app.include_router(router)


# ── Lifecycle events ────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    """Run on application startup."""
    logger.info("=" * 60)
    logger.info("  EduCore starting up")
    vision_model = (
        settings.OLLAMA_VISION_MODEL
        if settings.active_provider == "ollama"
        else settings.GOOGLE_VISION_MODEL
    )
    logger.info("  RAG chat model: %s (via %s)", settings.active_chat_model, settings.active_provider)
    logger.info("  Image-analysis model: %s (via %s)", vision_model, settings.active_provider)
    logger.info("  Embeddings: %s (%d dims)", settings.EMBEDDING_MODEL_NAME, settings.EMBEDDING_DIMENSIONS)
    logger.info("  Database: %s", settings.DATABASE_URL.split("@")[-1] if "@" in settings.DATABASE_URL else "configured")
    logger.info("=" * 60)

    # Ensure storage directory exists
    settings.storage_path.mkdir(parents=True, exist_ok=True)


@app.on_event("shutdown")
async def on_shutdown():
    """Cleanup on application shutdown."""
    from educore.database.connection import async_engine
    await async_engine.dispose()
    logger.info("EduCore shut down")


# ── Root ────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "EduCore",
        "version": "1.0.0",
        "docs": "/docs",
        "description": "RAG-powered educational assistant",
    }
