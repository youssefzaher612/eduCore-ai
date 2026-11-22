"""Shared parsing helpers for model-produced JSON objects."""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from educore.config import settings

ModelT = TypeVar("ModelT", bound=BaseModel)


class StructuredOutputError(ValueError):
    """Raised when a model response is empty, malformed, or not a JSON object."""


def parse_json_object(raw: Any) -> dict[str, Any]:
    """Parse one complete JSON object, allowing a surrounding code fence or prose."""
    if isinstance(raw, BaseModel):
        raw = raw.model_dump(mode="python")
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        raise StructuredOutputError("The model returned an empty structured response.")

    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, count=1, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text, count=1).strip()
    decoder = json.JSONDecoder()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
        start = text.find("{")
        if start >= 0:
            try:
                parsed, _ = decoder.raw_decode(text, start)
            except json.JSONDecodeError as exc:
                reason = (
                    "truncated"
                    if "Unterminated string" in exc.msg or exc.pos >= len(text) - 1
                    else "invalid"
                )
                raise StructuredOutputError(
                    f"The model returned {reason} JSON; a complete object is required."
                ) from exc
        else:
            raise StructuredOutputError("The model did not return a JSON object.")

    if not isinstance(parsed, dict):
        raise StructuredOutputError("The model response must be a JSON object.")
    return parsed


def validate_json_object(raw: Any, model: type[ModelT]) -> ModelT:
    """Parse a model response and validate its fields using a Pydantic model."""
    return model.model_validate(parse_json_object(raw))


_OLLAMA_SCHEMA_SUPPORTED: set[str] = set()


def _ollama_schema_supported(base_url: str) -> bool:
    """Probe the configured Ollama server; old servers retain JSON-mode fallback."""
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    if base in _OLLAMA_SCHEMA_SUPPORTED:
        return True
    try:
        response = httpx.get(f"{base}/api/version", timeout=1.5)
        response.raise_for_status()
        version = response.json().get("version", "")
        match = re.match(r"^(\d+)\.(\d+)\.(\d+)", str(version))
        if not match:
            return False
        supported = tuple(map(int, match.groups())) >= (0, 5, 0)
        if supported:
            _OLLAMA_SCHEMA_SUPPORTED.add(base)
        return supported
    except (httpx.HTTPError, ValueError, TypeError):
        return False


def structured_response_format(model: type[BaseModel], task_name: str) -> dict | None:
    """Use JSON Schema with capable local Ollama, else the provider JSON-object mode."""
    if settings.active_provider != "ollama" or not _ollama_schema_supported(settings.OLLAMA_BASE_URL):
        return None
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", task_name)[:64] or "structured_output"
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": model.model_json_schema(),
        },
    }
