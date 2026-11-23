
"""
Document ingestion service.

Orchestrates: upload -> store -> trigger async processing.
The heavy work (extract -> chunk -> embed -> index) runs in Celery.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from educore.config import settings
from educore.database.models import Document, DocumentStatus
from educore.database.repositories import DocumentRepository

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"pdf", "txt", "md", "csv", "xlsx"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


class IngestionService:
    """Handles document upload, processing status and deletion."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.doc_repo = DocumentRepository(session)

    async def upload_document(
        self,
        project_id: uuid.UUID,
        filename: str,
        file_content: bytes,
    ) -> Document:
        """Save an uploaded file and create its database record."""

        ext = Path(filename).suffix.lower().strip(".")

        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: .{ext}. "
                f"Allowed: {', '.join(f'.{e}' for e in sorted(ALLOWED_EXTENSIONS))}"
            )

        if len(file_content) > MAX_FILE_SIZE:
            raise ValueError(
                f"File too large: {len(file_content) / 1024 / 1024:.1f} MB. "
                f"Maximum: {MAX_FILE_SIZE / 1024 / 1024:.0f} MB."
            )

        doc_id = uuid.uuid4()
        storage_dir = settings.storage_path / "documents"
        storage_dir.mkdir(parents=True, exist_ok=True)

        # Use a generated filename to avoid path traversal and collisions.
        safe_filename = Path(filename).name
        file_path = storage_dir / f"{doc_id}_{safe_filename}"
        file_path.write_bytes(file_content)

        try:
            doc = await self.doc_repo.create(
                project_id=project_id,
                filename=safe_filename,
                file_type=ext,
                file_size=len(file_content),
                file_path=str(file_path),
            )

            doc.id = doc_id
            await self.session.flush()

            logger.info(
                "Document uploaded: %s (%s, %d bytes) -> %s",
                safe_filename,
                ext,
                len(file_content),
                file_path,
            )

            return doc

        except Exception:
            file_path.unlink(missing_ok=True)
            raise

    async def get_document_status(
        self,
        document_id: uuid.UUID,
    ) -> dict:
        """Return document processing status."""

        doc = await self.doc_repo.get_by_id(document_id)

        if not doc:
            return {"error": "Document not found"}

        return {
            "id": str(doc.id),
            "filename": doc.filename,
            "status": doc.status.value,
            "total_chunks": doc.total_chunks,
            "error_message": doc.error_message,
        }

    async def list_documents(
        self,
        project_id: uuid.UUID,
    ) -> list[dict]:
        """List all documents in a project."""

        docs = await self.doc_repo.list_by_project(project_id)

        return [
            {
                "id": str(d.id),
                "filename": d.filename,
                "file_type": d.file_type,
                "file_size": d.file_size,
                "status": d.status.value,
                "total_chunks": d.total_chunks,
                "created_at": (
                    d.created_at.isoformat()
                    if d.created_at else None
                ),
            }
            for d in docs
        ]

    async def delete_document(
        self,
        document_id: uuid.UUID,
    ) -> dict:
        """Delete one document and its associated chunks."""

        doc = await self.doc_repo.get_by_id(document_id)

        if not doc:
            raise ValueError("Document not found")

        # Only block deletion while Celery is actively processing it.
        # Pending and failed documents can be deleted.
        if doc.status == DocumentStatus.PROCESSING:
            raise ValueError(
                "Cannot delete a document while it is processing. "
                "Wait until processing finishes."
            )

        file_path = Path(doc.file_path).resolve()

        documents_dir = (
            settings.storage_path / "documents"
        ).resolve()

        # Safety check: only delete files inside storage/documents.
        if documents_dir not in file_path.parents:
            raise ValueError("Invalid document storage path")

        result = {
            "document_id": str(doc.id),
            "filename": doc.filename,
            "file_path": str(file_path),
        }

        # Delete document and associated chunks.
        # Document.chunks has cascade='all, delete-orphan'.
        await self.session.delete(doc)
        await self.session.flush()

        logger.info(
            "Document scheduled for deletion: %s",
            doc.filename,
        )

        return result
