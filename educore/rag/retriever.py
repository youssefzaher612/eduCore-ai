
"""
Semantic retriever for EduCore RAG pipeline.

Uses an optional LLM-rewritten query (or deterministic cleanup) to generate
embeddings, then retrieves relevant chunks from PostgreSQL using pgvector.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from educore.config import settings
from educore.database.repositories import ChunkRepository
from educore.rag.embedder import Embedder
from educore.rag.query_processor import process_query

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Represents one retrieved document chunk."""

    document_id: str
    document_name: str
    page_number: int | None
    content: str
    similarity: float
    chunk_id: str = ""


class Retriever:
    """Retrieve semantically relevant chunks for a question."""

    @staticmethod
    async def search(
        session: AsyncSession,
        project_id: uuid.UUID,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
        retrieval_query: str | None = None,
        document_id: uuid.UUID | None = None,
    ) -> list[RetrievalResult]:

        # 1. Validate the original question.
        if not query or not query.strip():
            logger.warning("Empty retrieval query received")
            return []

        # 2. Use the caller's LLM rewrite when available; otherwise apply
        # deterministic cleanup for direct Retriever callers.
        if retrieval_query is None:
            processed = process_query(query)
            retrieval_query = processed.retrieval_query

        if not retrieval_query or not retrieval_query.strip():
            retrieval_query = query.strip()

        logger.info("ORIGINAL QUERY: %s", query)
        logger.info("PROCESSED RETRIEVAL QUERY: %s", retrieval_query)

        # 3. Use configured defaults.
        if top_k is None:
            top_k = settings.RETRIEVAL_TOP_K

        if min_score is None:
            min_score = settings.RETRIEVAL_MIN_SCORE

        # 4. Validate retrieval parameters.
        if top_k <= 0:
            logger.warning("Invalid top_k value: %s", top_k)
            return []
        top_k = max(1, int(top_k))

        if not 0 <= min_score <= 1:
            raise ValueError("min_score must be between 0 and 1")

        # 5. Generate embedding from the processed query.
        query_embedding = Embedder.embed_query(retrieval_query)

        # 6. Search PostgreSQL using pgvector.
        chunk_repo = ChunkRepository(session)

        rows = await chunk_repo.search_similar(
            project_id=project_id,
            query_embedding=query_embedding,
            top_k=top_k,
            min_score=min_score,
            document_id=document_id,
        )

        # 7. Convert database rows into retrieval results.
        results: list[RetrievalResult] = []

        for row in rows:
            similarity = float(row["similarity"])

            if similarity < min_score:
                continue

            results.append(
                RetrievalResult(
                    chunk_id=row["chunk_id"],
                    document_id=row["document_id"],
                    document_name=row["document_name"],
                    page_number=row["page_number"],
                    content=row["content"],
                    similarity=similarity,
                )
            )

        logger.info(
            "RETRIEVAL COMPLETED | PROJECT=%s | "
            "RESULTS=%d | TOP_K=%d | MIN_SCORE=%.3f",
            project_id,
            len(results),
            top_k,
            min_score,
        )

        return results[:top_k]
