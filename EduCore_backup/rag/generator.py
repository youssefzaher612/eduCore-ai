
"""
LLM generation via Groq API.

Supports streaming (SSE) and batch generation.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from groq import AsyncGroq

from educore.config import settings

logger = logging.getLogger(__name__)


class GroqGenerator:
    """Async wrapper around the Groq chat completion API."""

    def __init__(self):
        if not settings.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Get a key at https://console.groq.com"
            )

        self.client = AsyncGroq(
            api_key=settings.GROQ_API_KEY
        )

    @staticmethod
    def _get_max_tokens(max_tokens: int | None) -> int:
        """Return a valid output token limit."""

        value = (
            settings.GROQ_MAX_TOKENS
            if max_tokens is None
            else max_tokens
        )

        if value <= 0:
            raise ValueError("max_tokens must be greater than zero")

        return value

    @staticmethod
    def _get_temperature(temperature: float | None) -> float:
        """Return the configured or requested temperature."""

        value = (
            settings.GROQ_TEMPERATURE
            if temperature is None
            else temperature
        )

        if not 0 <= value <= 2:
            raise ValueError("temperature must be between 0 and 2")

        return value

    async def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate a complete response."""

        try:
            response = await self.client.chat.completions.create(
                model=settings.GROQ_MODEL_ID,
                messages=messages,
                max_tokens=self._get_max_tokens(max_tokens),
                temperature=self._get_temperature(temperature),
                stream=False,
            )

            if not response.choices:
                raise RuntimeError(
                    "Groq returned no completion choices"
                )

            content = response.choices[0].message.content or ""

            logger.info(
                "Generated %d chars (model=%s, tokens=%s)",
                len(content),
                settings.GROQ_MODEL_ID,
                getattr(response.usage, "total_tokens", "?"),
            )

            if not content.strip():
                raise RuntimeError(
                    "Groq returned an empty response"
                )

            return content

        except Exception:
            logger.exception("Groq generation failed")
            raise

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream response fragments from Groq."""

        try:
            stream = await self.client.chat.completions.create(
                model=settings.GROQ_MODEL_ID,
                messages=messages,
                max_tokens=self._get_max_tokens(max_tokens),
                temperature=self._get_temperature(temperature),
                stream=True,
            )

            generated_anything = False

            async for chunk in stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                content = delta.content

                if content:
                    generated_anything = True
                    yield content

            if not generated_anything:
                raise RuntimeError(
                    "Groq streaming returned an empty response"
                )

        except Exception:
            logger.exception("Groq streaming failed")
            raise
