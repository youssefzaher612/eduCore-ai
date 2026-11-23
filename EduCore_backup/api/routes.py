
"""
API routes — all FastAPI endpoints.

Organised into:
    /api/users      — user management
    /api/projects   — project CRUD
    /api/documents  — document upload, status and deletion
    /api/chat       — RAG Q&A (streaming SSE)
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession

from educore.api.deps import (
    get_chat_service,
    get_ingestion_service,
    get_session,
)
from educore.database.models import UserRole
from educore.database.repositories import ProjectRepository, UserRepository
from educore.services.chat_service import ChatService
from educore.services.ingestion_service import IngestionService
from educore.tasks.ingestion_tasks import process_document_task

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


# Request / Response schemas

class CreateUserRequest(BaseModel):
    name: str
    email: EmailStr
    role: UserRole = UserRole.STUDENT


class CreateProjectRequest(BaseModel):
    name: str
    description: str = ""


class CreateChatSessionRequest(BaseModel):
    project_id: str
    section: UserRole
    title: str = "محادثة جديدة"


class AskQuestionRequest(BaseModel):
    question: str


# Health

@router.get("/health")
async def health_check():
    return {"status": "ok", "service": "EduCore RAG"}


# Users

@router.post("/users", status_code=201)
async def create_user(
    req: CreateUserRequest,
    session: AsyncSession = Depends(get_session),
):
    repo = UserRepository(session)

    existing = await repo.get_by_email(req.email)

    if existing:
        return {
            "id": str(existing.id),
            "name": existing.name,
            "email": existing.email,
            "role": existing.role.value,
            "message": "User already exists",
        }

    user = await repo.create(
        name=req.name,
        email=req.email,
        role=req.role,
    )

    return {
        "id": str(user.id),
        "name": user.name,
        "email": user.email,
        "role": user.role.value,
    }


@router.get("/users")
async def list_users(session: AsyncSession = Depends(get_session)):
    repo = UserRepository(session)
    users = await repo.list_all()

    return [
        {
            "id": str(u.id),
            "name": u.name,
            "email": u.email,
            "role": u.role.value,
        }
        for u in users
    ]


@router.get("/users/{user_id}")
async def get_user(
    user_id: str,
    session: AsyncSession = Depends(get_session),
):
    try:
        parsed_id = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(400, "Invalid user ID")

    repo = UserRepository(session)
    user = await repo.get_by_id(parsed_id)

    if not user:
        raise HTTPException(404, "User not found")

    return {
        "id": str(user.id),
        "name": user.name,
        "email": user.email,
        "role": user.role.value,
    }


# Projects

@router.post("/projects", status_code=201)
async def create_project(
    req: CreateProjectRequest,
    owner_id: str = Query(..., description="UUID of the project owner"),
    session: AsyncSession = Depends(get_session),
):
    try:
        parsed_owner_id = uuid.UUID(owner_id)
    except ValueError:
        raise HTTPException(400, "Invalid owner ID")

    repo = ProjectRepository(session)

    project = await repo.create(
        name=req.name,
        owner_id=parsed_owner_id,
        description=req.description,
    )

    return {
        "id": str(project.id),
        "name": project.name,
        "description": project.description,
    }


@router.get("/projects")
async def list_projects(
    owner_id: str = Query(..., description="UUID of the owner"),
    session: AsyncSession = Depends(get_session),
):
    try:
        parsed_owner_id = uuid.UUID(owner_id)
    except ValueError:
        raise HTTPException(400, "Invalid owner ID")

    repo = ProjectRepository(session)
    projects = await repo.list_by_owner(parsed_owner_id)

    return [
        {
            "id": str(p.id),
            "name": p.name,
            "description": p.description,
            "created_at": p.created_at.isoformat() if p.created_at else None,
        }
        for p in projects
    ]


@router.delete("/projects/{project_id}", status_code=204)
async def delete_project(
    project_id: str,
    session: AsyncSession = Depends(get_session),
):
    try:
        parsed_id = uuid.UUID(project_id)
    except ValueError:
        raise HTTPException(400, "Invalid project ID")

    repo = ProjectRepository(session)
    await repo.delete(parsed_id)
    await session.commit()


# Documents

@router.post("/documents/upload", status_code=201)
async def upload_document(
    project_id: str = Query(..., description="UUID of the project"),
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    """Upload a document and trigger background processing."""

    try:
        parsed_project_id = uuid.UUID(project_id)
    except ValueError:
        raise HTTPException(400, "Invalid project ID")

    service = get_ingestion_service(session)

    try:
        content = await file.read()

        doc = await service.upload_document(
            project_id=parsed_project_id,
            filename=file.filename or "unnamed",
            file_content=content,
        )

        await session.commit()

        task = process_document_task.delay(str(doc.id))

        return {
            "document_id": str(doc.id),
            "filename": doc.filename,
            "status": doc.status.value,
            "task_id": task.id,
            "message": "Document uploaded. Processing started in background.",
        }

    except ValueError as e:
        await session.rollback()
        raise HTTPException(400, str(e))
    except Exception:
        await session.rollback()
        logger.exception("Document upload failed")
        raise HTTPException(500, "Document upload failed")


@router.get("/documents")
async def list_documents(
    project_id: str = Query(..., description="UUID of the project"),
    session: AsyncSession = Depends(get_session),
):
    try:
        parsed_project_id = uuid.UUID(project_id)
    except ValueError:
        raise HTTPException(400, "Invalid project ID")

    service = get_ingestion_service(session)
    return await service.list_documents(parsed_project_id)


@router.get("/documents/{document_id}/status")
async def document_status(
    document_id: str,
    session: AsyncSession = Depends(get_session),
):
    try:
        parsed_id = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(400, "Invalid document ID")

    service = get_ingestion_service(session)
    result = await service.get_document_status(parsed_id)

    if "error" in result:
        raise HTTPException(404, result["error"])

    return result


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Delete one document and its associated chunks."""

    try:
        parsed_id = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(400, "Invalid document ID")

    service = get_ingestion_service(session)

    try:
        result = await service.delete_document(parsed_id)

        # Commit DB deletion first.
        await session.commit()

        # Remove physical file after successful DB commit.
        file_path = Path(result["file_path"])

        try:
            file_path.unlink(missing_ok=True)
        except OSError:
            logger.exception(
                "Database record deleted but physical file could not be removed: %s",
                file_path,
            )

        return {
            "status": "deleted",
            "document_id": result["document_id"],
            "filename": result["filename"],
        }

    except ValueError as e:
        await session.rollback()
        message = str(e)

        if message == "Document not found":
            raise HTTPException(404, message)

        raise HTTPException(409, message)

    except Exception:
        await session.rollback()
        logger.exception("Failed to delete document %s", document_id)
        raise HTTPException(500, "Failed to delete document")


# Chat Sessions

@router.post("/chat/sessions", status_code=201)
async def create_chat_session(
    req: CreateChatSessionRequest,
    user_id: str = Query(..., description="UUID of the user"),
    session: AsyncSession = Depends(get_session),
):
    try:
        parsed_user_id = uuid.UUID(user_id)
        parsed_project_id = uuid.UUID(req.project_id)
    except ValueError:
        raise HTTPException(400, "Invalid user or project ID")

    service = get_chat_service(session)

    chat = await service.create_session(
        user_id=parsed_user_id,
        project_id=parsed_project_id,
        section=req.section,
        title=req.title,
    )

    return {
        "session_id": str(chat.id),
        "title": chat.title,
        "section": chat.section.value,
    }


@router.get("/chat/sessions")
async def list_chat_sessions(
    user_id: str = Query(...),
    project_id: str = Query(...),
    session: AsyncSession = Depends(get_session),
):
    try:
        parsed_user_id = uuid.UUID(user_id)
        parsed_project_id = uuid.UUID(project_id)
    except ValueError:
        raise HTTPException(400, "Invalid user or project ID")

    service = get_chat_service(session)

    return await service.list_sessions(
        parsed_user_id,
        parsed_project_id,
    )


@router.get("/chat/sessions/{session_id}/messages")
async def get_chat_messages(
    session_id: str,
    session: AsyncSession = Depends(get_session),
):
    try:
        parsed_id = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(400, "Invalid session ID")

    service = get_chat_service(session)
    return await service.get_messages(parsed_id)


# Chat — Ask (Streaming SSE)

@router.post("/chat/sessions/{session_id}/ask")
async def ask_question(
    session_id: str,
    req: AskQuestionRequest,
    session: AsyncSession = Depends(get_session),
):
    """Ask a question and return a streaming SSE response."""

    try:
        parsed_id = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(400, "Invalid session ID")

    service = get_chat_service(session)

    async def event_stream():
        async for event in service.ask(parsed_id, req.question):
            event_type = event["event"]
            data = event["data"]

            if isinstance(data, dict):
                data = json.dumps(data, ensure_ascii=False)

            yield f"event: {event_type}\ndata: {data}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
