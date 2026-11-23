
"""
Centralised application settings.

Every setting is loaded from environment variables or .env.
Pydantic validates configuration at startup.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _REPOSITORY_ROOT / ".env"
if not _ENV_FILE.is_file():
    # Backward compatibility for deployments that kept .env beside this package.
    _ENV_FILE = Path(__file__).resolve().parent / ".env"


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
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

    # OpenRouter (OpenAI-compatible API)
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_CHAT_MODEL: str = "baidu/ernie-5.5-flash-preview:free"
    # This is a hard generation ceiling. Feature calls can request less.
    OPENROUTER_MAX_TOKENS: int = Field(default=8192, gt=0, le=8192)
    OPENROUTER_TEMPERATURE: float = Field(default=0.15, ge=0, le=2)
    # Legacy OpenRouter search flag; text web search is controlled by Groq's
    # explicit opt-in flag below.
    OPENROUTER_ENABLE_WEB_SEARCH: bool = False

    # Local Ollama is the default. AI_PROVIDER is the preferred new name;
    # LLM_PROVIDER remains as a backward-compatible alias for existing .env files.
    AI_PROVIDER: str = ""
    LLM_PROVIDER: str = "ollama"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_CHAT_MODEL: str = "qwen3-vl:2b-instruct-q4_K_M"
    OLLAMA_VISION_MODEL: str = "qwen3-vl:2b-instruct-q4_K_M"
    OLLAMA_NUM_CTX: int = Field(default=4096, ge=1024, le=32768)
    OLLAMA_TIMEOUT_SECONDS: float = Field(default=180.0, gt=0, le=900)
    # Keep the small shared model resident between student/teacher requests.
    OLLAMA_KEEP_ALIVE: str = "30m"

    # Number of type-specific teacher-assessment repair calls after generation.
    ASSESSMENT_GENERATION_MAX_CORRECTION_ATTEMPTS: int = Field(default=1, ge=0, le=5)

    # Optional external compatibility. These are not required in local mode.
    GROQ_API_KEY: str = ""
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    GROQ_CHAT_MODEL: str = "openai/gpt-oss-120b"
    GROQ_FAST_MODEL: str = "openai/gpt-oss-20b"
    GROQ_WEB_SEARCH_ENABLED: bool = False

    # Gemini Vision through Google AI Studio.
    GOOGLE_API_KEY: str = ""
    GOOGLE_VISION_MODEL: str = "gemini-flash-latest"
    VISION_MAX_UPLOAD_BYTES: int = Field(default=12 * 1024 * 1024, gt=0)
    VISION_MAX_COMPRESSED_BYTES: int = Field(default=8 * 1024 * 1024, gt=0)

    # Embeddings
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSIONS: int = Field(default=384, gt=0)

    # RAG Pipeline
    CHUNK_SIZE: int = Field(default=500, gt=0)
    CHUNK_OVERLAP: int = Field(default=100, ge=0)
    MIN_CHUNK_CHARS: int = Field(default=50, ge=0)

    # OCR for scanned or image-only PDFs (Tesseract language packs are installed separately).
    OCR_LANGUAGES: str = "eng+ara"
    OCR_DPI: int = Field(default=250, gt=0)
    OCR_MAX_PAGES: int = Field(default=20, ge=0)
    TESSERACT_CMD: str = ""
    TESSDATA_DIR: str = ""

    RETRIEVAL_TOP_K: int = Field(default=3, ge=1, le=5)
    RETRIEVAL_MIN_SCORE: float = Field(default=0.3, ge=0, le=1)
    RETRIEVAL_MAX_CONTEXT_CHARS: int = Field(default=9000, gt=0)

    # Celery
    CELERY_BROKER_URL: str = "amqp://guest:guest@localhost:5672//"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    # Chat
    CHAT_HISTORY_LIMIT: int = Field(default=6, ge=0, le=20)
    CHAT_HISTORY_MESSAGE_CHARS: int = Field(default=2000, gt=0, le=8000)
    DEMO_MODE: bool = False
    AI_MAX_INPUT_CHARS: int = Field(default=30000, gt=0)
    AI_MAX_OUTPUT_TOKENS: int = Field(default=8192, gt=0, le=8192)
    AI_PROVIDER_FAILURE_THRESHOLD: int = Field(default=3, ge=1, le=20)
    AI_PROVIDER_COOLDOWN_SECONDS: int = Field(default=60, ge=1, le=3600)

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

    @property
    def active_chat_model(self) -> str:
        provider = self.active_provider
        if provider == "ollama":
            return self.OLLAMA_CHAT_MODEL
        if provider == "openrouter":
            return self.OPENROUTER_CHAT_MODEL
        return self.GROQ_CHAT_MODEL

    @property
    def active_provider(self) -> str:
        return (self.AI_PROVIDER or self.LLM_PROVIDER or "ollama").strip().lower()

    @property
    def active_web_search_enabled(self) -> bool:
        return self.active_provider == "groq" and self.GROQ_WEB_SEARCH_ENABLED


settings = Settings()
