
"""
Semantic retriever for EduCore RAG pipeline.

Retrieves relevant chunks from PostgreSQL using pgvector.
Only semantically matched chunks are returned.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from educore.config import settings
from educore.database.repositories import ChunkRepository
from educore.rag.embedder import Embedder

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Represents one retrieved document chunk."""

    document_id: str
    document_name: str
    page_number: int | None
    content: str
    similarity: float


class Retriever:
    """Retrieve semantically relevant chunks for a question."""

    @staticmethod
    async def search(
        session: AsyncSession,
        project_id: uuid.UUID,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[RetrievalResult]:

        # Validate the query.
        if not query or not query.strip():
            logger.warning("Empty retrieval query received")
            return []

        # Use configured defaults only when values are None.
        if top_k is None:
            top_k = settings.RETRIEVAL_TOP_K

        if min_score is None:
            min_score = settings.RETRIEVAL_MIN_SCORE

        # Validate retrieval parameters.
        if top_k <= 0:
            logger.warning("Invalid top_k value: %s", top_k)
            return []

        if not 0 <= min_score <= 1:
            raise ValueError("min_score must be between 0 and 1")

        # Generate query embedding.
        query_embedding = Embedder.embed_query(query.strip())

        # Search using the actual repository implementation.
        chunk_repo = ChunkRepository(session)

        rows = await chunk_repo.search_similar(
            project_id=project_id,
            query_embedding=query_embedding,
            top_k=top_k,
            min_score=min_score,
        )

        results: list[RetrievalResult] = []

        for row in rows:
            similarity = float(row["similarity"])

            # Additional relevance check.
            if similarity < min_score:
                continue

            results.append(
                RetrievalResult(
                    document_id=row["document_id"],
                    document_name=row["document_name"],
                    page_number=row["page_number"],
                    content=row["content"],
                    similarity=similarity,
                )
            )

        logger.info(
            "Retrieved %d relevant chunks for project=%s "
            "(top_k=%d, min_score=%.3f)",
            len(results),
            project_id,
            top_k,
            min_score,
        )

        return results
