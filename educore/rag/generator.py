"""LLM generation through the configured OpenAI-compatible provider."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import AsyncGenerator

from openai import RateLimitError

from educore.config import settings
from educore.provider import create_async_client

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_DELAY_BASE = 2.0  # seconds; doubles each retry


def _groq_reasoning_effort(requested: str | None, default: str | None) -> str:
    effort = (requested or default or "low").strip().lower()
    # "none" is used by non-reasoning providers and is not a Groq GPT-OSS
    # setting. Low keeps these short internal tasks within their token budget.
    return effort if effort in {"low", "medium", "high"} else "low"


def _append_groq_search_citations(message, answer: str) -> str:
    """Expose URLs returned by Groq's browser tool as ordinary Markdown citations."""
    existing_urls = set(re.findall(r"https?://[^)\s]+", answer))
    citations = []
    seen = set(existing_urls)
    for tool in getattr(message, "executed_tools", None) or []:
        payload = tool if isinstance(tool, dict) else {}
        results = (payload.get("search_results") or {}).get("results", [])
        for result in results:
            if not isinstance(result, dict):
                continue
            title = str(result.get("title") or "Web source").replace("[", "(").replace("]", ")").strip()
            url = str(result.get("url") or "").strip()
            if not url.startswith(("https://", "http://")) or url in seen:
                continue
            seen.add(url)
            citations.append(f"- [{title}]({url})")
            if len(citations) == 5:
                break
        if len(citations) == 5:
            break
    if citations:
        answer = answer.rstrip() + "\n\nSources:\n" + "\n".join(citations)
    return answer


def _build_fallback_list(primary: str) -> list[str]:
    """Return ordered list: primary first, then fallbacks excluding primary."""
    models = [primary]
    fallback_models = [settings.GROQ_FAST_MODEL] if settings.active_provider == "groq" else []
    for model in fallback_models:
        if model != primary:
            models.append(model)
    return models


class OpenRouterGenerator:
    """Async wrapper for regular, streaming, and opt-in web-search responses."""

    def __init__(self, model_id: str | None = None, reasoning_effort: str = "medium"):
        self.model_id = model_id or settings.active_chat_model
        self.reasoning_effort = reasoning_effort  # retained for existing call sites
        # Session construction, message history, and other non-generation
        # operations must work when AI credentials are not configured.
        self.client = None

    def _get_client(self):
        """Create the external provider client only for an actual AI request."""
        if self.client is None:
            self.client = create_async_client()
        return self.client

    @staticmethod
    def _max_tokens(value: int | None) -> int:
        # Ordinary chat defaults to a small answer; structured features that
        # explicitly request more receive their task budget up to the hard cap.
        result = 2048 if value is None else value
        if result <= 0:
            raise ValueError("max_tokens must be greater than zero")
        return min(result, settings.OPENROUTER_MAX_TOKENS, settings.AI_MAX_OUTPUT_TOKENS)

    @staticmethod
    def _validate_messages(messages: list[dict]) -> None:
        """Reject oversized prompts before they can trigger unexpectedly large bills."""
        total_chars = sum(
            len(content)
            for message in messages
            if isinstance(message, dict)
            for content in [message.get("content", "")]
            if isinstance(content, str)
        )
        if total_chars > settings.AI_MAX_INPUT_CHARS:
            raise ValueError(
                f"AI input is too large ({total_chars} characters; "
                f"limit {settings.AI_MAX_INPUT_CHARS}). Narrow the question or source material."
            )

    @staticmethod
    def _temperature(value: float | None) -> float:
        result = settings.OPENROUTER_TEMPERATURE if value is None else value
        if not 0 <= result <= 2:
            raise ValueError("temperature must be between 0 and 2")
        return result

    async def generate(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: dict | None = None,
        reasoning_effort: str | None = None,
        num_ctx: int | None = None,
    ) -> str:
        self._validate_messages(messages)
        models = _build_fallback_list(self.model_id)
        last_error: Exception | None = None

        for attempt, model in enumerate(models):
            try:
                request = {
                    "model": model,
                    "messages": messages,
                    ("max_tokens" if settings.active_provider == "ollama" else "max_completion_tokens"):
                        self._max_tokens(max_tokens),
                    "temperature": self._temperature(temperature),
                    "stream": False,
                }
                if settings.active_provider == "ollama":
                    request["extra_body"] = {
                        "options": {
                            "num_ctx": min(
                                32768,
                                max(settings.OLLAMA_NUM_CTX, int(num_ctx or 0)),
                            ),
                        },
                        "keep_alive": settings.OLLAMA_KEEP_ALIVE,
                    }
                if response_format is not None:
                    request["response_format"] = response_format
                response = await self._get_client().chat.completions.create(**request)
                if not response.choices:
                    raise RuntimeError("Groq returned no completion choices")
                content = response.choices[0].message.content or ""
                if not content.strip():
                    raise RuntimeError("Groq returned an empty response")
                if model != self.model_id:
                    logger.info("Succeeded with fallback model %s (primary %s was rate-limited)", model, self.model_id)
                logger.info(
                    "Generated %d chars (model=%s, tokens=%s)",
                    len(content), model, getattr(response.usage, "total_tokens", "?"),
                )
                return content

            except Exception as exc:
                is_rate_limit = isinstance(exc, RateLimitError) or getattr(exc, "status_code", None) == 429

                if not is_rate_limit:
                    raise

                last_error = exc
                delay = _RETRY_DELAY_BASE * (2 ** attempt)
                logger.warning(
                    "Rate-limited (429) on model %s (attempt %d/%d). "
                    "Retrying with next model in %.1fs...",
                    model, attempt + 1, len(models), delay,
                )
                await asyncio.sleep(delay)

        raise RuntimeError(
            f"All models exhausted after {len(models)} attempts due to rate limiting"
        ) from last_error

    async def generate_with_web_search(
        self,
        messages: list[dict],
        max_completion_tokens: int = 2048,
    ) -> str:
        """Search externally only when explicitly enabled; search has separate pricing."""
        if not settings.active_web_search_enabled:
            raise RuntimeError(
                "Web search is disabled for the configured LLM provider."
            )
        self._validate_messages(messages)
        models = _build_fallback_list(self.model_id)
        last_error: Exception | None = None

        for attempt, model in enumerate(models):
            try:
                response = await self._get_client().chat.completions.create(
                    model=model,
                    messages=messages,
                    max_completion_tokens=(
                        max(max_completion_tokens, 2048)
                        if settings.active_provider == "groq"
                        else max_completion_tokens
                    ),
                    temperature=0.2,
                    tools=[{"type": "browser_search"}],
                )
                if not response.choices:
                    raise RuntimeError("Groq web search returned no completion choices")
                message = response.choices[0].message
                content = message.content or ""
                if not content.strip():
                    raise RuntimeError("Groq web search returned an empty answer")
                if settings.active_provider == "groq":
                    content = _append_groq_search_citations(message, content)
                return content

            except Exception as exc:
                is_rate_limit = isinstance(exc, RateLimitError) or getattr(exc, "status_code", None) == 429

                if not is_rate_limit:
                    raise

                last_error = exc
                delay = _RETRY_DELAY_BASE * (2 ** attempt)
                logger.warning(
                    "Rate-limited (429) on web-search model %s (attempt %d/%d). "
                    "Retrying in %.1fs...",
                    model, attempt + 1, len(models), delay,
                )
                await asyncio.sleep(delay)

        raise RuntimeError(
            f"All models exhausted for web search after {len(models)} attempts"
        ) from last_error

    async def generate_stream(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ) -> AsyncGenerator[str, None]:
        self._validate_messages(messages)
        models = _build_fallback_list(self.model_id)
        last_error: Exception | None = None

        for attempt, model in enumerate(models):
            try:
                request = {
                    "model": model,
                    "messages": messages,
                    ("max_tokens" if settings.active_provider == "ollama" else "max_completion_tokens"):
                        self._max_tokens(max_tokens),
                    "temperature": self._temperature(temperature),
                    "stream": True,
                }
                if settings.active_provider == "ollama":
                    request["extra_body"] = {
                        "options": {"num_ctx": settings.OLLAMA_NUM_CTX},
                        "keep_alive": settings.OLLAMA_KEEP_ALIVE,
                    }
                if settings.active_provider == "groq":
                    request["reasoning_effort"] = _groq_reasoning_effort(reasoning_effort, self.reasoning_effort)
                stream = await self._get_client().chat.completions.create(**request)
                yielded = False
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    content = chunk.choices[0].delta.content
                    if content:
                        yielded = True
                        yield content
                if not yielded:
                    logger.warning("OpenRouter stream (%s) returned no visible text; retrying as completion", model)
                    yield await self.generate(
                        messages, max_tokens=max_tokens, temperature=temperature,
                    )
                return  # Success — exit the retry loop

            except Exception as exc:
                is_rate_limit = isinstance(exc, RateLimitError) or getattr(exc, "status_code", None) == 429

                if not is_rate_limit:
                    raise

                last_error = exc
                delay = _RETRY_DELAY_BASE * (2 ** attempt)
                logger.warning(
                    "Rate-limited (429) on streaming model %s (attempt %d/%d). "
                    "Retrying with next model in %.1fs...",
                    model, attempt + 1, len(models), delay,
                )
                await asyncio.sleep(delay)

        raise RuntimeError(
            f"All models exhausted for streaming after {len(models)} attempts"
        ) from last_error
