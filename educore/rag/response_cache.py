"""Small, process-local exact-response cache with explicit tenant scoping."""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from threading import Lock
from typing import Any


class ExactResponseCache:
    def __init__(self, max_entries: int = 256, ttl_seconds: int = 300):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._items: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._lock = Lock()

    @staticmethod
    def key(
        *,
        user_id: Any,
        project_id: Any,
        role: Any,
        task: str,
        model: str,
        messages: list[dict],
    ) -> str:
        role_value = getattr(role, "value", role)
        payload = json.dumps(
            {
                "user_id": str(user_id),
                "project_id": str(project_id),
                "role": str(role_value),
                "task": task,
                "model": model,
                "messages": messages,
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get(self, key: str) -> str | None:
        now = time.monotonic()
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at <= now:
                self._items.pop(key, None)
                return None
            self._items.move_to_end(key)
            return value

    def put(self, key: str, value: str) -> None:
        if not value:
            return
        with self._lock:
            self._items[key] = (time.monotonic() + self.ttl_seconds, value)
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)


response_cache = ExactResponseCache()


class SemanticResponseCache:
    """Short-lived semantic cache whose scope includes the complete RAG context."""

    def __init__(self, max_entries: int = 256, ttl_seconds: int = 300, threshold: float = 0.965):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.threshold = threshold
        self._items: OrderedDict[str, list[tuple[float, tuple[float, ...], str]]] = OrderedDict()
        self._lock = Lock()

    @staticmethod
    def scope_key(
        *, user_id: Any, project_id: Any, role: Any, task: str, model: str,
        messages: list[dict],
    ) -> tuple[str, str | None]:
        """Return a private-context scope and the final text question, if present."""
        role_value = getattr(role, "value", role)
        safe_messages = list(messages)
        query = None
        if safe_messages:
            last = safe_messages[-1]
            if last.get("role") == "user" and isinstance(last.get("content"), str):
                query = last["content"].strip()
                safe_messages[-1] = {**last, "content": "<semantic-query>"}
        payload = json.dumps(
            {
                "user_id": str(user_id), "project_id": str(project_id),
                "role": str(role_value), "task": task, "model": model,
                "messages": safe_messages,
            },
            sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest(), query or None

    def get(self, scope: str, vector: list[float]) -> str | None:
        norm = sum(value * value for value in vector) ** 0.5
        if not norm:
            return None
        query = tuple(value / norm for value in vector)
        now = time.monotonic()
        best_score = -1.0
        best_value = None
        with self._lock:
            entries = self._items.get(scope)
            if not entries:
                return None
            fresh = []
            for expires_at, candidate, value in entries:
                if expires_at <= now:
                    continue
                fresh.append((expires_at, candidate, value))
                if len(candidate) == len(query):
                    score = sum(left * right for left, right in zip(query, candidate))
                    if score > best_score:
                        best_score, best_value = score, value
            if fresh:
                self._items[scope] = fresh
                self._items.move_to_end(scope)
            else:
                self._items.pop(scope, None)
        return best_value if best_score >= self.threshold else None

    def put(self, scope: str, vector: list[float], value: str) -> None:
        if not value:
            return
        norm = sum(item * item for item in vector) ** 0.5
        if not norm:
            return
        normalized = tuple(item / norm for item in vector)
        with self._lock:
            entries = self._items.setdefault(scope, [])
            entries.append((time.monotonic() + self.ttl_seconds, normalized, value))
            self._items.move_to_end(scope)
            while sum(len(items) for items in self._items.values()) > self.max_entries:
                oldest_scope = next(iter(self._items))
                oldest = self._items[oldest_scope]
                oldest.pop(0)
                if not oldest:
                    self._items.pop(oldest_scope, None)


semantic_response_cache = SemanticResponseCache()
