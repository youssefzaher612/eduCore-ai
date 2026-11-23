
"""
Centralised application settings.

Every setting is loaded from environment variables or .env.
Pydantic validates configuration at startup.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    DATABASE_URL: str = (
        "postgresql+asyncpg://educore:educore_pass@localhost:5432/educore"
    )
    DATABASE_URL_SYNC: str = (
        "postgresql://educore:educore_pass@localhost:5432/educore"
    )

    # Groq API
    GROQ_API_KEY: str = ""
    GROQ_MODEL_ID: str = "allam-2-7b"
    GROQ_MAX_TOKENS: int = Field(default=4096, gt=0)
    GROQ_TEMPERATURE: float = Field(default=0.3, ge=0, le=2)

    # Embeddings
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSIONS: int = Field(default=384, gt=0)

    # RAG Pipeline
    CHUNK_SIZE: int = Field(default=500, gt=0)
    CHUNK_OVERLAP: int = Field(default=100, ge=0)
    MIN_CHUNK_CHARS: int = Field(default=50, ge=0)

    RETRIEVAL_TOP_K: int = Field(default=5, gt=0)
    RETRIEVAL_MIN_SCORE: float = Field(default=0.15, ge=0, le=1)

    # Celery
    CELERY_BROKER_URL: str = "amqp://guest:guest@localhost:5672//"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    # Chat
    CHAT_HISTORY_LIMIT: int = Field(default=10, ge=0)

    # File Storage
    STORAGE_DIR: str = "./storage"

    # FastAPI
    API_HOST: str = "0.0.0.0"
    API_PORT: int = Field(default=8000, gt=0, le=65535)

    @property
    def storage_path(self) -> Path:
        path = Path(self.STORAGE_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
