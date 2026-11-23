
"""
Local embedding generation using Sentence Transformers.

Uses all-MiniLM-L6-v2 (384 dimensions).
Runs locally on CPU without requiring an Ollama server.
"""

from __future__ import annotations

import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """Generate dense vector embeddings locally."""

    MODEL_NAME = "all-MiniLM-L6-v2"
    DIMENSIONS = 384

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

        model = cls._get_model()

        vectors = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        logger.info("Embedded %d texts using %s", len(texts), cls.MODEL_NAME)

        return vectors.tolist()

    @classmethod
    def embed_query(cls, query: str) -> list[float]:
        """Generate embedding for a single query."""
        return cls.embed_texts([query])[0]

    @classmethod
    def dimensions(cls) -> int:
        """Return embedding dimensionality."""
        return cls.DIMENSIONS
