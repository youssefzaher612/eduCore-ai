"""Shared OpenAI-compatible client configuration for app LLM providers."""

from __future__ import annotations

from openai import AsyncOpenAI, OpenAI

from educore.config import settings


def provider_settings() -> tuple[str, str, str]:
    provider = settings.active_provider
    if provider == "ollama":
        base_url = settings.OLLAMA_BASE_URL.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        return provider, "ollama", base_url
    if provider == "groq":
        return provider, settings.GROQ_API_KEY.strip(), settings.GROQ_BASE_URL
    if provider == "openrouter":
        return provider, settings.OPENROUTER_API_KEY.strip(), settings.OPENROUTER_BASE_URL
    raise ValueError("Unsupported AI provider. Choose ollama, groq, or openrouter.")


def require_api_key() -> str:
    provider, key, _ = provider_settings()
    if provider == "ollama":
        # The OpenAI SDK requires a non-empty value; local Ollama ignores it.
        return "ollama"
    if not key:
        raise ValueError(f"{provider.upper()}_API_KEY is missing or empty in EduCore's .env file.")
    return key


def require_openrouter_api_key() -> str:
    """Backward-compatible alias for older call sites."""
    return require_api_key()


def create_async_client(
    *, api_key: str | None = None, timeout: float | None = None, max_retries: int = 0
) -> AsyncOpenAI:
    provider, configured_key, base_url = provider_settings()
    return AsyncOpenAI(
        api_key=api_key or configured_key or require_api_key(),
        base_url=base_url,
        # Bound provider stalls so a RAG request fails quickly enough for the
        # caller to use its deterministic rewrite fallback or show an error.
        timeout=timeout if timeout is not None else (
            settings.OLLAMA_TIMEOUT_SECONDS if provider == "ollama" else 45.0
        ),
        max_retries=max_retries,
    )


def create_sync_client(
    *, api_key: str | None = None, timeout: float | None = None, max_retries: int = 0
) -> OpenAI:
    provider, configured_key, base_url = provider_settings()
    return OpenAI(
        api_key=api_key or configured_key or require_api_key(),
        base_url=base_url,
        timeout=timeout if timeout is not None else (
            settings.OLLAMA_TIMEOUT_SECONDS if provider == "ollama" else 45.0
        ),
        max_retries=max_retries,
    )
