
"""
Local embedding generation using Sentence Transformers.

Uses all-MiniLM-L6-v2 (384 dimensions).
Runs locally on CPU without requiring an Ollama server.
"""

from __future__ import annotations

import logging
import hashlib
import os
from collections import OrderedDict
from threading import RLock

# Embeddings are a local service in EduCore. Avoid Hub metadata checks and
# exponential network retries during chat; the model is loaded from cache.
# Operators may explicitly set HF_HUB_OFFLINE=0 when they intend to fetch it.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from sentence_transformers import SentenceTransformer

from educore.config import settings

logger = logging.getLogger(__name__)
_QUERY_EMBEDDINGS: OrderedDict[tuple[str, str], list[float]] = OrderedDict()
_QUERY_EMBEDDING_CACHE_SIZE = 2048
_TEXT_EMBEDDINGS: OrderedDict[tuple[str, str], list[float]] = OrderedDict()
_TEXT_EMBEDDING_CACHE_SIZE = 4096
_EMBEDDING_CACHE_LOCK = RLock()


class Embedder:
    """Generate dense vector embeddings locally."""

    MODEL_NAME = settings.EMBEDDING_MODEL_NAME
    DIMENSIONS = settings.EMBEDDING_DIMENSIONS

    _model = None

    @classmethod
    def _get_model(cls):
        """Load the embedding model once and reuse it."""
        if cls._model is None:
            logger.info("Loading embedding model: %s", cls.MODEL_NAME)
            cls._model = SentenceTransformer(cls.MODEL_NAME, device="cpu")
            logger.info("Embedding model loaded successfully")

        return cls._model

    @classmethod
    def embed_texts(cls, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        if not texts:
            return []

        keys = [
            (cls.MODEL_NAME, hashlib.sha256(text.encode("utf-8")).hexdigest())
            for text in texts
        ]
        vectors_by_key: dict[tuple[str, str], list[float]] = {}
        with _EMBEDDING_CACHE_LOCK:
            for key in set(keys):
                cached = _TEXT_EMBEDDINGS.get(key)
                if cached is not None:
                    _TEXT_EMBEDDINGS.move_to_end(key)
                    vectors_by_key[key] = cached

        missing_texts: dict[tuple[str, str], str] = {}
        for key, text in zip(keys, texts):
            if key not in vectors_by_key:
                missing_texts.setdefault(key, text)

        if missing_texts:
            model = cls._get_model()
            vectors = model.encode(
                list(missing_texts.values()),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            if vectors.ndim != 2 or vectors.shape[1] != settings.EMBEDDING_DIMENSIONS:
                raise ValueError(
                    f"Embedding model {settings.EMBEDDING_MODEL_NAME!r} returned "
                    f"{vectors.shape[-1] if vectors.ndim else 0} dimensions; "
                    f"the chunks table expects {settings.EMBEDDING_DIMENSIONS}. "
                    "Use a matching model or migrate and re-index the vector column."
                )
            new_vectors = {
                key: vector.tolist()
                for key, vector in zip(missing_texts, vectors)
            }
            with _EMBEDDING_CACHE_LOCK:
                for key, vector in new_vectors.items():
                    _TEXT_EMBEDDINGS[key] = vector
                    _TEXT_EMBEDDINGS.move_to_end(key)
                    vectors_by_key[key] = vector
                while len(_TEXT_EMBEDDINGS) > _TEXT_EMBEDDING_CACHE_SIZE:
                    _TEXT_EMBEDDINGS.popitem(last=False)
            logger.info("Embedded %d unique texts using %s", len(new_vectors), cls.MODEL_NAME)

        return [vectors_by_key[key].copy() for key in keys]

    @classmethod
    def embed_query(cls, query: str) -> list[float]:
        """Generate embedding for a single query."""
        digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
        key = (cls.MODEL_NAME, digest)
        with _EMBEDDING_CACHE_LOCK:
            cached = _QUERY_EMBEDDINGS.get(key)
            if cached is not None:
                _QUERY_EMBEDDINGS.move_to_end(key)
        if cached is not None:
            return cached.copy()
        vector = cls.embed_texts([query])[0]
        with _EMBEDDING_CACHE_LOCK:
            _QUERY_EMBEDDINGS[key] = vector
            _QUERY_EMBEDDINGS.move_to_end(key)
            while len(_QUERY_EMBEDDINGS) > _QUERY_EMBEDDING_CACHE_SIZE:
                _QUERY_EMBEDDINGS.popitem(last=False)
        return vector

    @classmethod
    def dimensions(cls) -> int:
        """Return embedding dimensionality."""
        return settings.EMBEDDING_DIMENSIONS


