"""
Groq LLM client for EduCore teacher tools with a Groq-only model fallback.

Public interface
----------------
    llm_client.generate(system_prompt, user_prompt, temperature, max_tokens) -> str
        Backwards-compatible. Returns the text of a COMPLETE response and raises
        a specific LLMError subclass otherwise (never returns truncated text).

    llm_client.generate_detailed(...) -> LLMResponse
        Same call, but returns the text together with finish reason, token usage,
        attempt count and latency.

Behaviour
---------
* Lazy SDK client creation for the configured Groq endpoint.
* Model fallback cascade: primary Groq model -> fast Groq model.
* Safe JSON extraction: handles markdown code fences and cleans JSON text.
* Graceful fallback when response_format={"type": "json_object"} is unsupported.
* Error text and logs are redacted: API keys never appear in them.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

from openai import OpenAI

from educore.teacher_ai.core import config
from educore.config import settings

logger = logging.getLogger(__name__)

_SECRET_PATTERNS = (
    re.compile(r"gsk_[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}"),
)

# Approved Groq fallback model for teacher AI.
_FALLBACK_MODELS = [settings.GROQ_FAST_MODEL] if settings.active_provider == "groq" else []


def redact(text: Any, extra_secrets: tuple = (), limit: int = 600) -> str:
    """Return ``text`` with API keys removed and length capped."""
    value = "" if text is None else str(text)
    for secret in extra_secrets:
        if secret and len(secret) >= 6:
            value = value.replace(secret, "[REDACTED]")
    for pattern in _SECRET_PATTERNS:
        value = pattern.sub("[REDACTED]", value)
    return value if len(value) <= limit else value[:limit] + "..."


def clean_json_text(text: str) -> str:
    """Extract clean JSON substring from model output, stripping markdown fences or commentary."""
    if not isinstance(text, str):
        return ""
    cleaned = text.strip()
    # 1. Try finding balanced { ... } that parses validly
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start : end + 1]
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass

    # 2. Strip code fences
    cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n?```$", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()

    # 3. If balanced braces exist after stripping fences, return candidate
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LLMError(Exception):
    """Base class. ``retryable`` says whether the same call may succeed later."""

    retryable = False

    def __init__(self, message: str, *, status_code: Optional[int] = None,
                 detail: Any = None, retry_after: Optional[float] = None,
                 retryable: Optional[bool] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.detail = redact(detail) if detail else None
        self.retry_after = retry_after
        if retryable is not None:
            self.retryable = retryable


class LLMConfigurationError(LLMError):
    """Missing API key, missing SDK, or invalid client settings."""


class LLMAuthenticationError(LLMError):
    """HTTP 401/403: the API key was rejected."""


class LLMRequestError(LLMError):
    """Permanent client-side failure (HTTP 400/404/413/422 and similar)."""


class LLMRateLimitError(LLMError):
    """HTTP 429. Retryable only when the required wait is short."""

    retryable = True


class LLMTimeoutError(LLMError):
    retryable = True


class LLMConnectionError(LLMError):
    retryable = True


class LLMServerError(LLMError):
    """HTTP 5xx (or 408/409) from the provider."""

    retryable = True


class LLMEmptyResponseError(LLMError):
    """The API answered successfully but returned no text."""

    retryable = True


class LLMResponseError(LLMError):
    """The response had an unexpected shape or finish reason."""


class LLMTruncatedError(LLMError):
    """The output hit the token limit (finish_reason == 'length')."""

    def __init__(self, message: str, *, partial_content: str = "",
                 completion_tokens: Optional[int] = None,
                 max_tokens: Optional[int] = None):
        super().__init__(message)
        self.partial_content = partial_content
        self.completion_tokens = completion_tokens
        self.max_tokens = max_tokens


# ---------------------------------------------------------------------------
# Response object
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    content: str
    finish_reason: Optional[str]
    model: Optional[str]
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    attempts: int = 1
    latency_seconds: float = 0.0
    request_id: Optional[str] = None

    def to_metadata(self) -> dict:
        data = asdict(self)
        data.pop("content", None)
        data["finish_reason_missing"] = self.finish_reason is None
        return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RETRY_IN_PATTERNS = (
    (re.compile(r"try again in\s+(\d+(?:\.\d+)?)\s*h\s*(\d+(?:\.\d+)?)\s*m", re.I), lambda m: float(m[1]) * 3600 + float(m[2]) * 60),
    (re.compile(r"try again in\s+(\d+(?:\.\d+)?)\s*m\s*(\d+(?:\.\d+)?)\s*s", re.I), lambda m: float(m[1]) * 60 + float(m[2])),
    (re.compile(r"try again in\s+(\d+(?:\.\d+)?)\s*ms", re.I), lambda m: float(m[1]) / 1000.0),
    (re.compile(r"try again in\s+(\d+(?:\.\d+)?)\s*s", re.I), lambda m: float(m[1])),
    (re.compile(r"try again in\s+(\d+(?:\.\d+)?)\s*m\b", re.I), lambda m: float(m[1]) * 60),
)


def parse_retry_after(headers: Any, message: str = "") -> Optional[float]:
    """Seconds to wait, from a Retry-After header or the provider's message text."""
    try:
        raw = headers.get("retry-after") if headers is not None else None
    except Exception:
        raw = None
    if raw is not None:
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            pass
    for pattern, convert in _RETRY_IN_PATTERNS:
        match = pattern.search(message or "")
        if match:
            return convert(match)
    return None


def _error_text(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        inner = body.get("error")
        if isinstance(inner, dict) and inner.get("message"):
            return str(inner["message"])
        if body.get("message"):
            return str(body["message"])
    return str(getattr(exc, "message", None) or exc)


def _int_or_none(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


class LLMClient:
    """Thin, well-behaved wrapper around Groq with a Groq model fallback."""

    _groq_blocked_until: float = 0.0

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        sdk_client: Any = None,
        max_retries: Optional[int] = None,
        base_delay: Optional[float] = None,
        max_delay: Optional[float] = None,
        max_retry_after: Optional[float] = None,
        timeout: Optional[float] = None,
        model_max_completion_tokens: Optional[int] = None,
        json_mode: Optional[bool] = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: Callable[[], float] = random.random,
    ):
        self._api_key = api_key
        self.model = model or settings.active_chat_model
        self._sdk = sdk_client
        self.max_retries = config.OPENROUTER_MAX_RETRIES if max_retries is None else max_retries
        self.base_delay = config.OPENROUTER_RETRY_BASE_DELAY if base_delay is None else base_delay
        self.max_delay = config.OPENROUTER_RETRY_MAX_DELAY if max_delay is None else max_delay
        self.max_retry_after = config.OPENROUTER_MAX_RETRY_AFTER if max_retry_after is None else max_retry_after
        self.timeout = config.OPENROUTER_TIMEOUT_SECONDS if timeout is None else timeout
        self.model_max_completion_tokens = (
            config.OPENROUTER_MODEL_MAX_COMPLETION_TOKENS
            if model_max_completion_tokens is None else model_max_completion_tokens
        )
        self.json_mode = config.OPENROUTER_JSON_MODE if json_mode is None else json_mode
        self._sleep = sleep
        self._rng = rng
        self._token_param = (
            "max_tokens" if settings.active_provider == "ollama"
            else "max_completion_tokens"
        )

    # -- SDK access ---------------------------------------------------------
    def _secrets(self) -> tuple:
        return (self._api_key or config.OPENROUTER_API_KEY or "",)

    def _get_sdk(self) -> Any:
        if self._sdk is None:
            key = self._api_key or config.require_openrouter_api_key()
            if not key or not key.strip():
                raise LLMConfigurationError(
                    f"{settings.active_provider.upper()} configuration is missing. Check EduCore's .env file and local AI service."
                )
            try:
                from educore.provider import create_sync_client
            except ImportError:
                raise LLMConfigurationError(
                    "The 'openai' package is not installed. Install EduCore dependencies."
                ) from None
            self._sdk = create_sync_client(
                api_key=key.strip(), timeout=self.timeout, max_retries=0
            )
        return self._sdk

    # -- Public API ---------------------------------------------------------
    def generate(self, system_prompt: str, user_prompt: str,
                 temperature: float = 0.4, max_tokens: Optional[int] = None,
                 *, model: Optional[str] = None,
                 response_format: Optional[dict] = None) -> str:
        """Return the text of a complete response."""
        return self.generate_detailed(
            system_prompt, user_prompt, temperature, max_tokens, model=model,
            response_format=response_format,
        ).content

    def generate_detailed(self, system_prompt: str, user_prompt: str,
                          temperature: float = 0.4, max_tokens: Optional[int] = None,
                          *, model: Optional[str] = None,
                          response_format: Optional[dict] = None) -> LLMResponse:
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt must be a non-empty string.")
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            raise ValueError("user_prompt must be a non-empty string.")

        requested = config.OPENROUTER_GENERATION_MAX_TOKENS if max_tokens is None else int(max_tokens)
        if requested < 1:
            raise ValueError("max_tokens must be a positive integer.")
        limit = min(requested, self.model_max_completion_tokens, 4096)
        temp = max(0.0, min(float(temperature), config.MAX_TEMPERATURE))
        primary_model = model or self.model

        # Try the primary Groq model and then the approved fast Groq model.
        now = time.monotonic()
        active_provider = settings.active_provider
        # A recent primary-model rate limit should skip that model, not disable
        # the whole cascade. The configured fast Groq model remains eligible.
        cascade: list[tuple[str, str]] = []
        if active_provider != "groq" or now >= LLMClient._groq_blocked_until:
            cascade.append((active_provider, primary_model))
        for fb in _FALLBACK_MODELS:
            if fb != primary_model:
                cascade.append((active_provider, fb))

        started = time.monotonic()
        last_error: Optional[Exception] = None

        for attempt, (provider, model_name) in enumerate(cascade, start=1):
            try:
                raw = self._call_provider(
                    provider, model_name, system_prompt, user_prompt, temp, limit,
                    response_format=response_format,
                )
                response = self._to_response(raw, model_name, limit)
                response.attempts = attempt
                response.latency_seconds = round(time.monotonic() - started, 3)
                logger.info(
                    "Teacher LLM call ok: provider=%s model=%s tokens=%s attempts=%s latency=%.1fs",
                    provider, response.model, response.completion_tokens, attempt, response.latency_seconds,
                )
                return response
            except LLMTruncatedError:
                raise
            except (LLMError, Exception) as exc:
                last_error = exc
                classified = self._classify(exc) if not isinstance(exc, LLMError) else exc
                err_text = str(classified.message).lower()
                if "free-models-per-day" in err_text or "daily_limit" in err_text or (isinstance(classified, LLMRateLimitError) and provider == active_provider):
                    LLMClient._groq_blocked_until = time.monotonic() + 300
                logger.warning(
                    "Teacher AI call failed on %s/%s (%s): %s. Trying the next configured model...",
                    provider, model_name, type(classified).__name__, redact(classified.message, self._secrets())
                )
                # Small jitter before next attempt
                if attempt < len(cascade):
                    self._sleep(0.5)

        logger.error("All teacher model providers exhausted: %s", last_error)
        if last_error is None:
            raise LLMServerError("The AI provider is temporarily unavailable. Please try again shortly.")
        raise self._classify(last_error) if not isinstance(last_error, LLMError) else last_error

    # -- Internals ----------------------------------------------------------
    def _call_provider(
        self,
        provider: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        response_format: Optional[dict] = None,
    ) -> Any:
        sdk = self._get_sdk()

        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            self._token_param: max_tokens,
        }
        if provider == "ollama":
            kwargs["extra_body"] = {
                "options": {"num_ctx": settings.OLLAMA_NUM_CTX},
                "keep_alive": settings.OLLAMA_KEEP_ALIVE,
            }

        # Try with JSON mode first, then retry the same model without that
        # constraint when Groq rejects its constrained decoder. Some Groq
        # models/accounts return json_validate_failed with an empty generation.
        if response_format is not None:
            kwargs["response_format"] = response_format
        elif self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        request_kwargs = dict(kwargs)
        for _ in range(3):
            try:
                return sdk.chat.completions.create(**request_kwargs)
            except Exception as exc:
                err_str = str(exc).lower()
                status = getattr(exc, "status_code", None)
                body = getattr(exc, "body", None)
                error_body = body.get("error", {}) if isinstance(body, dict) else {}
                error_code = str(error_body.get("code", "")).casefold() if isinstance(error_body, dict) else ""
                json_format_rejected = (
                    error_code == "json_validate_failed"
                    or "json_validate_failed" in err_str
                    or "response_format" in err_str
                    or "json_object" in err_str
                )
                if "response_format" in request_kwargs and json_format_rejected:
                    logger.warning(
                        "%s rejected structured response formatting (HTTP %s); retrying %s with JSON mode",
                        provider,
                        status or "unknown",
                        model,
                    )
                    current_format = request_kwargs.get("response_format") or {}
                    if current_format.get("type") == "json_schema":
                        request_kwargs["response_format"] = {"type": "json_object"}
                    else:
                        request_kwargs.pop("response_format", None)
                    continue
                if "max_completion_tokens" in request_kwargs and "max_completion_tokens" in err_str:
                    logger.warning("Provider rejected max_completion_tokens; retrying %s with max_tokens", model)
                    self._token_param = "max_tokens"
                    request_kwargs.pop("max_completion_tokens", None)
                    request_kwargs["max_tokens"] = max_tokens
                    continue
                raise

        raise RuntimeError("The provider rejected all compatible request formats.")

    def _to_response(self, raw: Any, requested_model: str, max_tokens: int) -> LLMResponse:
        choices = getattr(raw, "choices", None)
        if not choices:
            raise LLMResponseError("The AI response contained no choices.")
        choice = choices[0]
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None)
        finish = getattr(choice, "finish_reason", None)
        usage = getattr(raw, "usage", None)
        completion_tokens = _int_or_none(getattr(usage, "completion_tokens", None))

        raw_text = content if isinstance(content, str) else ""

        # If JSON mode is enabled, clean the output so json.loads succeeds
        if self.json_mode and raw_text.strip():
            raw_text = clean_json_text(raw_text)

        response = LLMResponse(
            content=raw_text,
            finish_reason=finish if isinstance(finish, str) else None,
            model=getattr(raw, "model", None) or requested_model,
            prompt_tokens=_int_or_none(getattr(usage, "prompt_tokens", None)),
            completion_tokens=completion_tokens,
            total_tokens=_int_or_none(getattr(usage, "total_tokens", None)),
            max_tokens=max_tokens,
            request_id=getattr(raw, "id", None),
        )

        if response.finish_reason == "length":
            raise LLMTruncatedError(
                "The AI response was cut off at the output token limit "
                f"({completion_tokens if completion_tokens is not None else 'unknown'} of "
                f"{max_tokens} tokens used). The output is incomplete.",
                partial_content=response.content,
                completion_tokens=completion_tokens,
                max_tokens=max_tokens,
            )
        if not response.content.strip():
            raise LLMEmptyResponseError(
                f"The AI returned an empty response (finish_reason={response.finish_reason}).")
        if response.finish_reason not in (None, "stop"):
            raise LLMResponseError(
                f"The AI stopped for an unexpected reason: {response.finish_reason!r}.")
        return response

    def _classify(self, exc: Exception) -> LLMError:
        """Map an SDK/transport exception to an LLMError (without leaking secrets)."""
        name = type(exc).__name__
        status = getattr(exc, "status_code", None)
        status = status if isinstance(status, int) else None
        text = redact(_error_text(exc), self._secrets())
        headers = getattr(getattr(exc, "response", None), "headers", None)

        if "Timeout" in name:
            return LLMTimeoutError("The AI service timed out.", detail=text)
        if "Connection" in name:
            if settings.active_provider == "ollama":
                message = "Could not connect to local Ollama. Make sure Ollama is running at OLLAMA_BASE_URL."
            else:
                message = f"Could not connect to {settings.active_provider}. Check its base URL and network, then try again."
            return LLMConnectionError(message, detail=text)
        if status == 429:
            wait = parse_retry_after(headers, text)
            return LLMRateLimitError("Rate limit reached.", status_code=status,
                                     detail=text, retry_after=wait)
        if status in (401, 403):
            return LLMAuthenticationError(
                f"The AI service rejected the configured credentials. Check {settings.active_provider.upper()} settings.",
                status_code=status, detail=text)
        if status == 413:
            return LLMRequestError(
                "The request was too large for the model or your plan's token limit.",
                status_code=status, detail=text)
        if status in (400, 404, 422):
            return LLMRequestError(
                f"The AI service rejected the request (HTTP {status}).",
                status_code=status, detail=text)
        if status is not None and (status >= 500 or status in (408, 409)):
            return LLMServerError(f"The AI service had a temporary problem (HTTP {status}).",
                                  status_code=status, detail=text)
        return LLMError(f"Unexpected error while calling the AI service ({name}).", detail=text)


llm_client = LLMClient()
