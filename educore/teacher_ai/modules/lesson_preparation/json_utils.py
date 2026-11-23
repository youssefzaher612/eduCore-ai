"""
Small, dependency-free helpers for reading JSON written by an LLM.

    extract_json(text) -> dict      the first JSON object found in ``text``
    parse_path_text(text) -> tuple  'practice_questions[2].correct_answer' -> ('practice_questions', 2, 'correct_answer')
    get_path(obj, path)             value at a path (KeyError / IndexError / TypeError when absent)
    resolve_prefix(obj, path)       the longest leading part of ``path`` that exists in ``obj``

Tolerances in extract_json (all deterministic, none changes a value):
  * Markdown code fences and prose before/after the object are ignored (the object is the
    first '{' to its matching '}').
  * Raw line breaks / tabs inside strings are accepted (``strict=False``).
  * If - and only if - strict parsing fails, trailing commas before } or ] are removed.
Anything else (unterminated strings, missing braces, a cut-off response) is an error: the
text is never "completed" or guessed.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"^\s*```[A-Za-z0-9_-]*\s*\n?|\n?\s*```\s*$")
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")
_PATH_TOKEN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)\]|\.")


class JSONExtractionError(ValueError):
    """No complete JSON object could be read from the model output."""

    def __init__(self, message: str, *, truncated: bool = False):
        super().__init__(message)
        self.truncated = truncated


def _looks_truncated(text: str) -> bool:
    depth = 0
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
    return in_string or depth > 0


def extract_json(text: Any) -> dict:
    """Return the first JSON object in ``text`` or raise JSONExtractionError."""
    if not isinstance(text, str) or not text.strip():
        raise JSONExtractionError("The AI returned no text to read as JSON.")
    cleaned = text.lstrip("﻿").strip()
    cleaned = _FENCE.sub("", cleaned).strip()
    start = cleaned.find("{")
    if start < 0:
        raise JSONExtractionError("The AI response did not contain a JSON object.")

    # Only the FIRST '{' is considered. Trying later ones could return a nested fragment of a
    # truncated or malformed outer object and present it as the whole answer.
    decoder = json.JSONDecoder(strict=False)
    candidate = cleaned[start:]
    last_error = None
    for variant in (candidate, _TRAILING_COMMA.sub(r"\1", candidate)):
        try:
            value, _ = decoder.raw_decode(variant)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(value, dict):
            return value

    truncated = _looks_truncated(cleaned[start:])
    where = f" near character {last_error.pos}" if last_error is not None else ""
    if truncated:
        raise JSONExtractionError(
            "The AI response looks cut off: the JSON object is not complete" + where + ".",
            truncated=True)
    raise JSONExtractionError("The AI response is not valid JSON" + where + ".")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def parse_path_text(text: Any) -> tuple:
    """'a.b[2].c' -> ('a', 'b', 2, 'c'). Returns () for empty or unreadable text."""
    if not isinstance(text, str):
        return ()
    text = text.strip()
    parts = []
    index = 0
    while index < len(text):
        match = _PATH_TOKEN.match(text, index)
        if not match:
            return ()
        if match.group(1):
            parts.append(match.group(1))
        elif match.group(2) is not None:
            parts.append(int(match.group(2)))
        index = match.end()
    return tuple(parts)


def get_path(obj: Any, path: tuple) -> Any:
    for part in path:
        obj = obj[part]
    return obj


def resolve_prefix(obj: Any, path: tuple) -> tuple:
    """The longest leading part of ``path`` that exists inside ``obj``."""
    good = []
    current = obj
    for part in path:
        try:
            current = current[part]
        except (KeyError, IndexError, TypeError):
            break
        good.append(part)
    return tuple(good)
