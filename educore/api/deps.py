"""
FastAPI dependency injection.

Every service and repository is created here — the single composition
point for the entire API. No route file imports database internals.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from educore.database.connection import get_async_session
from educore.database.repositories import (
    ChunkRepository,
    ProjectRepository,
    UserRepository,
)
from educore.services.chat_service import ChatService
from educore.services.ingestion_service import IngestionService
from educore.services.student_service import StudentLearningService
from educore.services.teacher_service import TeacherWorkspaceService


async def get_session():
    """Yield an async database session."""
    async for session in get_async_session():
        yield session


def get_user_repo(session: AsyncSession) -> UserRepository:
    return UserRepository(session)


def get_project_repo(session: AsyncSession) -> ProjectRepository:
    return ProjectRepository(session)


def get_chunk_repo(session: AsyncSession) -> ChunkRepository:
    return ChunkRepository(session)


def get_ingestion_service(session: AsyncSession) -> IngestionService:
    return IngestionService(session)


def get_chat_service(session: AsyncSession) -> ChatService:
    return ChatService(session)


def get_student_service(session: AsyncSession) -> StudentLearningService:
    return StudentLearningService(session)


def get_teacher_service(session: AsyncSession) -> TeacherWorkspaceService:
    return TeacherWorkspaceService(session)
