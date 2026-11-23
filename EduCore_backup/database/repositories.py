"""
Repository layer — data access for every model.

Async methods for FastAPI, sync helpers where Celery needs them.
All raw SQL for pgvector search lives here, nowhere else.
"""

from __future__ import annotations

import uuid
from typing import Sequence

from sqlalchemy import delete, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from educore.database.models import (
    ChatSession,
    Chunk,
    Document,
    DocumentStatus,
    Message,
    Project,
    User,
    UserRole,
)


# ═══════════════════════════════════════════════════════════════════
#  User Repository
# ═══════════════════════════════════════════════════════════════════

class UserRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, name: str, email: str, role: UserRole = UserRole.STUDENT) -> User:
        user = User(name=name, email=email, role=role)
        self.session.add(user)
        await self.session.flush()
        return user

    async def get_by_id(self, user_id: uuid.UUID) -> User | None:
        return await self.session.get(User, user_id)

    async def get_by_email(self, email: str) -> User | None:
        stmt = select(User).where(User.email == email)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_all(self) -> Sequence[User]:
        result = await self.session.execute(select(User))
        return result.scalars().all()


# ═══════════════════════════════════════════════════════════════════
#  Project Repository
# ═══════════════════════════════════════════════════════════════════

class ProjectRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, name: str, owner_id: uuid.UUID, description: str = "") -> Project:
        project = Project(name=name, owner_id=owner_id, description=description)
        self.session.add(project)
        await self.session.flush()
        return project

    async def get_by_id(self, project_id: uuid.UUID) -> Project | None:
        return await self.session.get(Project, project_id)

    async def list_by_owner(self, owner_id: uuid.UUID) -> Sequence[Project]:
        stmt = select(Project).where(Project.owner_id == owner_id)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def delete(self, project_id: uuid.UUID) -> None:
        await self.session.execute(delete(Project).where(Project.id == project_id))


# ═══════════════════════════════════════════════════════════════════
#  Document Repository
# ═══════════════════════════════════════════════════════════════════

class DocumentRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        project_id: uuid.UUID,
        filename: str,
        file_type: str,
        file_size: int,
        file_path: str,
    ) -> Document:
        doc = Document(
            project_id=project_id,
            filename=filename,
            file_type=file_type,
            file_size=file_size,
            file_path=file_path,
        )
        self.session.add(doc)
        await self.session.flush()
        return doc

    async def get_by_id(self, document_id: uuid.UUID) -> Document | None:
        return await self.session.get(Document, document_id)

    async def list_by_project(self, project_id: uuid.UUID) -> Sequence[Document]:
        stmt = select(Document).where(Document.project_id == project_id)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def update_status(
        self,
        document_id: uuid.UUID,
        status: DocumentStatus,
        total_chunks: int = 0,
        error_message: str | None = None,
    ) -> None:
        stmt = (
            update(Document)
            .where(Document.id == document_id)
            .values(
                status=status,
                total_chunks=total_chunks,
                error_message=error_message,
            )
        )
        await self.session.execute(stmt)

    async def delete(self, document_id: uuid.UUID) -> None:
        await self.session.execute(delete(Document).where(Document.id == document_id))


# ═══════════════════════════════════════════════════════════════════
#  Document Repository (Sync — for Celery)
# ═══════════════════════════════════════════════════════════════════

class SyncDocumentRepository:
    """Thin sync wrapper used inside Celery tasks."""

    def __init__(self, session: Session):
        self.session = session

    def get_by_id(self, document_id: uuid.UUID) -> Document | None:
        return self.session.get(Document, document_id)

    def update_status(
        self,
        document_id: uuid.UUID,
        status: DocumentStatus,
        total_chunks: int = 0,
        error_message: str | None = None,
    ) -> None:
        stmt = (
            update(Document)
            .where(Document.id == document_id)
            .values(status=status, total_chunks=total_chunks, error_message=error_message)
        )
        self.session.execute(stmt)
        self.session.commit()


# ═══════════════════════════════════════════════════════════════════
#  Chunk Repository
# ═══════════════════════════════════════════════════════════════════

class ChunkRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_many(self, chunks: list[Chunk]) -> None:
        self.session.add_all(chunks)
        await self.session.flush()

    async def get_by_document(self, document_id: uuid.UUID) -> Sequence[Chunk]:
        stmt = (
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_order)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def delete_by_document(self, document_id: uuid.UUID) -> None:
        await self.session.execute(delete(Chunk).where(Chunk.document_id == document_id))

    async def search_similar(
        self,
        project_id: uuid.UUID,
        query_embedding: list[float],
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> list[dict]:
        """
        Cosine-similarity search via pgvector.

        pgvector's <=> operator returns cosine distance ∈ [0, 2].
        similarity = 1 − distance, so 1.0 = identical, 0.0 = orthogonal.
        """
        sql = text("""
            SELECT
                c.id,
                c.content,
                c.chunk_order,
                c.page_number,
                c.metadata,
                c.document_id,
                d.filename,
                1 - (c.embedding <=> :query_vec) AS similarity
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.project_id = :project_id
              AND c.embedding IS NOT NULL
              AND 1 - (c.embedding <=> :query_vec) >= :min_score
            ORDER BY c.embedding <=> :query_vec
            LIMIT :top_k
        """)

        result = await self.session.execute(
            sql,
            {
                "project_id": project_id,
                "query_vec": str(query_embedding),
                "min_score": min_score,
                "top_k": top_k,
            },
        )

        return [
            {
                "chunk_id": str(row.id),
                "content": row.content,
                "chunk_order": row.chunk_order,
                "page_number": row.page_number,
                "metadata": row.metadata,
                "document_id": str(row.document_id),
                "document_name": row.filename,
                "similarity": round(float(row.similarity), 4),
            }
            for row in result.fetchall()
        ]


# ═══════════════════════════════════════════════════════════════════
#  Chunk Repository (Sync — for Celery)
# ═══════════════════════════════════════════════════════════════════

class SyncChunkRepository:
    """Sync chunk storage used inside Celery tasks."""

    def __init__(self, session: Session):
        self.session = session

    def create_many(self, chunks: list[Chunk]) -> None:
        self.session.add_all(chunks)
        self.session.commit()

    def delete_by_document(self, document_id: uuid.UUID) -> None:
        self.session.execute(delete(Chunk).where(Chunk.document_id == document_id))
        self.session.commit()


# ═══════════════════════════════════════════════════════════════════
#  Chat Session Repository
# ═══════════════════════════════════════════════════════════════════

class ChatSessionRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        user_id: uuid.UUID,
        project_id: uuid.UUID,
        section: UserRole,
        title: str = "محادثة جديدة",
    ) -> ChatSession:
        chat = ChatSession(
            user_id=user_id,
            project_id=project_id,
            section=section,
            title=title,
        )
        self.session.add(chat)
        await self.session.flush()
        return chat

    async def get_by_id(self, session_id: uuid.UUID) -> ChatSession | None:
        return await self.session.get(ChatSession, session_id)

    async def list_by_user_and_project(
        self, user_id: uuid.UUID, project_id: uuid.UUID
    ) -> Sequence[ChatSession]:
        stmt = (
            select(ChatSession)
            .where(ChatSession.user_id == user_id, ChatSession.project_id == project_id)
            .order_by(ChatSession.created_at.desc())
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def delete(self, session_id: uuid.UUID) -> None:
        await self.session.execute(delete(ChatSession).where(ChatSession.id == session_id))


# ═══════════════════════════════════════════════════════════════════
#  Message Repository
# ═══════════════════════════════════════════════════════════════════

class MessageRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        session_id: uuid.UUID,
        role: str,
        content: str,
        citations: list[dict] | None = None,
    ) -> Message:
        msg = Message(
            session_id=session_id,
            role=role,
            content=content,
            citations=citations,
        )
        self.session.add(msg)
        await self.session.flush()
        return msg

    async def list_by_session(
        self, session_id: uuid.UUID, limit: int | None = None
    ) -> Sequence[Message]:
        stmt = (
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.asc())
        )
        if limit:
            stmt = stmt.limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_recent_history(
        self, session_id: uuid.UUID, limit: int = 10
    ) -> list[dict]:
        """Return recent messages formatted for the LLM context."""
        stmt = (
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        messages = list(reversed(result.scalars().all()))
        return [{"role": m.role, "content": m.content} for m in messages]
