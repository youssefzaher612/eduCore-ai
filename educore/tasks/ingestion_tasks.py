"""
Celery tasks for background document processing.

Pipeline: extract → chunk → embed → store vectors → update status.
Runs in a Celery worker with sync DB access (psycopg2).
"""

from __future__ import annotations

import logging
import traceback
import uuid

from educore.celery_app import celery_app
from educore.database.connection import get_sync_session
from educore.database.models import Chunk, DocumentStatus
from educore.database.repositories import SyncChunkRepository, SyncDocumentRepository
from educore.rag.chunker import TextChunker
from educore.rag.embedder import Embedder
from educore.rag.extractor import DocumentExtractor

logger = logging.getLogger(__name__)


@celery_app.task(
    name="educore.process_document",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
    acks_late=True,
)
def process_document_task(self, document_id: str) -> dict:
    """
    Full document processing pipeline (runs in Celery worker).

    Steps:
        1. Load document metadata from DB
        2. Extract text from file
        3. Split into chunks
        4. Generate embeddings
        5. Store chunks + vectors in DB
        6. Update document status

    Args:
        document_id: UUID string of the document to process.

    Returns:
        Summary dict with chunk count and status.
    """
    doc_uuid = uuid.UUID(document_id)
    session = get_sync_session()

    try:
        doc_repo = SyncDocumentRepository(session)
        chunk_repo = SyncChunkRepository(session)

        # ── 1. Load document ────────────────────────────────────────
        doc = doc_repo.get_by_id(doc_uuid)
        if not doc:
            logger.error("Document %s not found", document_id)
            return {"status": "error", "message": "Document not found"}

        logger.info("Processing document: %s (%s)", doc.filename, doc.file_type)

        # Mark as processing
        doc_repo.update_status(doc_uuid, DocumentStatus.PROCESSING)

        # ── 2. Extract text ─────────────────────────────────────────
        pages = DocumentExtractor.extract(
            doc.file_path,
            doc.file_type,
            cache_scope=f"{doc.project_id}:{doc_uuid}",
        )

        if not pages:
            doc_repo.update_status(
                doc_uuid,
                DocumentStatus.FAILED,
                error_message="No text could be extracted from the document.",
            )
            return {"status": "failed", "message": "No text extracted"}

        logger.info("Extracted %d pages from %s", len(pages), doc.filename)

        # ── 3. Chunk text ───────────────────────────────────────────
        chunker = TextChunker()
        text_chunks = chunker.chunk(pages)

        if not text_chunks:
            doc_repo.update_status(
                doc_uuid,
                DocumentStatus.FAILED,
                error_message="Document produced no usable chunks.",
            )
            return {"status": "failed", "message": "No chunks produced"}

        logger.info("Created %d chunks from %s", len(text_chunks), doc.filename)

        # ── 4. Generate embeddings ──────────────────────────────────
        chunk_texts = [c.content for c in text_chunks]
        embeddings = Embedder.embed_texts(chunk_texts)

        logger.info("Generated %d embeddings for %s", len(embeddings), doc.filename)

        # ── 5. Store chunks + vectors ───────────────────────────────
        # Delete any existing chunks (for re-processing)
        chunk_repo.delete_by_document(doc_uuid)

        db_chunks = [
            Chunk(
                document_id=doc_uuid,
                project_id=doc.project_id,
                content=tc.content,
                chunk_order=tc.chunk_order,
                page_number=tc.page_number,
                metadata_=tc.metadata if tc.metadata else None,
                embedding=emb,
            )
            for tc, emb in zip(text_chunks, embeddings)
        ]
        chunk_repo.create_many(db_chunks)

        logger.info("Stored %d chunks for %s", len(db_chunks), doc.filename)

        # ── 6. Update status ────────────────────────────────────────
        doc_repo.update_status(
            doc_uuid,
            DocumentStatus.INDEXED,
            total_chunks=len(db_chunks),
        )

        result = {
            "status": "indexed",
            "document_id": document_id,
            "filename": doc.filename,
            "pages_extracted": len(pages),
            "chunks_created": len(db_chunks),
        }
        logger.info("Document processed successfully: %s", result)
        return result

    except Exception as exc:
        logger.exception("Failed to process document %s: %s", document_id, exc)

        # Update status to FAILED
        try:
            doc_repo.update_status(
                doc_uuid,
                DocumentStatus.FAILED,
                error_message=f"{type(exc).__name__}: {exc}",
            )
        except Exception:
            logger.exception("Failed to update document status after error")

        # Retry on transient errors
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)

        return {
            "status": "failed",
            "document_id": document_id,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

    finally:
        session.close()
