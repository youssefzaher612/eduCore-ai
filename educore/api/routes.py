
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
from typing import Literal

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import StreamingResponse
import httpx
from openai import APIError, RateLimitError
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession

from educore.api.deps import (
    get_chat_service,
    get_ingestion_service,
    get_session,
    get_student_service,
    get_teacher_service,
)
from educore.database.models import UserRole
from educore.database.repositories import ProjectRepository, UserRepository
from educore.config import settings
from educore.services.chat_service import ChatService
from educore.services.ingestion_service import IngestionService
from educore.services.student_service import ImageAnalysisProviderError
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
    mode: Literal["study", "hint"] = "study"


class StudentScopeRequest(BaseModel):
    user_id: uuid.UUID
    project_id: uuid.UUID


class StudentPracticeRequest(StudentScopeRequest):
    document_id: uuid.UUID
    topic: str | None = None


class StudentPlanRequest(StudentScopeRequest):
    weekly_hours: int = Field(default=4, ge=1, le=40)
    document_id: uuid.UUID | None = None
    topic: str | None = None
    language: Literal["Arabic", "English"] = "Arabic"


class StudentExamRequest(StudentScopeRequest):
    topic: str
    document_id: uuid.UUID
    difficulty: str = "Mixed"
    question_count: int = Field(default=5, ge=1, le=20)
    language: Literal["Arabic", "English"] = "English"
    question_type: Literal["mcq", "true_false", "essay", "short_answer", "mixed"] = "mcq"


class StudentExamSubmitRequest(StudentScopeRequest):
    answers: list[object | None]


# Health

@router.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "EduCore RAG",
        "ai_provider": settings.active_provider,
        "ai_model": settings.active_chat_model,
    }


@router.get("/ai/status")
async def ai_status():
    """Report local model availability without exposing provider credentials."""
    if settings.active_provider != "ollama":
        configured = bool(
            settings.GROQ_API_KEY if settings.active_provider == "groq"
            else settings.OPENROUTER_API_KEY
        )
        return {
            "provider": settings.active_provider,
            "available": configured,
            "model": settings.active_chat_model,
        }

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/tags")
            response.raise_for_status()
            models = [
                item.get("name", "")
                for item in response.json().get("models", [])
                if item.get("name")
            ]
        return {
            "provider": "ollama",
            "available": True,
            "model": settings.OLLAMA_CHAT_MODEL,
            "vision_model": settings.OLLAMA_VISION_MODEL,
            "chat_model_installed": any(
                name == settings.OLLAMA_CHAT_MODEL
                or name.split(":", 1)[0] == settings.OLLAMA_CHAT_MODEL.split(":", 1)[0]
                for name in models
            ),
            "installed_models": models,
        }
    except (httpx.HTTPError, ValueError):
        return {
            "provider": "ollama",
            "available": False,
            "model": settings.OLLAMA_CHAT_MODEL,
            "vision_model": settings.OLLAMA_VISION_MODEL,
            "installed_models": [],
        }


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
    user_id: str = Query(..., description="UUID of the user"),
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    """Upload a document and trigger background processing."""

    try:
        parsed_project_id = uuid.UUID(project_id)
    except ValueError:
        raise HTTPException(400, "Invalid project ID")
    try:
        parsed_user_id = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(400, "Invalid user ID")

    service = get_ingestion_service(session)

    try:
        content = await file.read()

        doc = await service.upload_document(
            project_id=parsed_project_id,
            user_id=parsed_user_id,
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
    user_id: str = Query(..., description="UUID of the user"),
    session: AsyncSession = Depends(get_session),
):
    """Delete one document and its associated chunks."""

    try:
        parsed_id = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(400, "Invalid document ID")
    try:
        parsed_user_id = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(400, "Invalid user ID")

    service = get_ingestion_service(session)

    try:
        result = await service.delete_document(parsed_id, parsed_user_id)

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


# Student AI — analytics, adaptive practice, and course-grounded exams

@router.get("/student/dashboard")
async def student_dashboard(
    user_id: uuid.UUID = Query(...),
    project_id: uuid.UUID = Query(...),
    document_id: uuid.UUID | None = Query(default=None),
    session: AsyncSession = Depends(get_session),
):
    service = get_student_service(session)
    try:
        return await service.dashboard(user_id, project_id, document_id)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/teacher/materials")
async def teacher_course_materials(
    user_id: uuid.UUID = Query(...),
    project_id: uuid.UUID = Query(...),
    document_id: uuid.UUID = Query(...),
    topic: str = Query(..., min_length=2, max_length=200),
    session: AsyncSession = Depends(get_session),
):
    service = get_teacher_service(session)
    try:
        return await service.course_material(
            user_id, project_id, document_id, topic
        )
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RateLimitError as exc:
        # OpenRouter's free models enforce account/model rate limits. Tell
        # clients when to retry instead of making the failure look like an
        # application error or encouraging rapid repeated submissions.
        raise HTTPException(
            status_code=429,
            detail=(
                "وصل نموذج الذكاء الاصطناعي إلى حد الطلبات الحالي. "
                "انتظر دقيقة ثم أعد المحاولة، أو اختر نموذجًا بحدود أعلى "
                "في GROQ_CHAT_MODEL داخل ملف .env."
            ),
            headers={"Retry-After": "60"},
        ) from exc


@router.get("/teacher/assessment-records")
async def teacher_assessment_records(
    user_id: uuid.UUID = Query(...),
    project_id: uuid.UUID = Query(...),
    document_id: uuid.UUID | None = Query(default=None),
    session: AsyncSession = Depends(get_session),
):
    service = get_teacher_service(session)
    try:
        return await service.assessment_records(user_id, project_id, document_id)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/student/practice-question", status_code=201)
async def student_practice_question(
    req: StudentPracticeRequest,
    session: AsyncSession = Depends(get_session),
):
    service = get_student_service(session)
    try:
        return await service.practice_question(
            req.user_id, req.project_id, req.document_id, req.topic
        )
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RateLimitError as exc:
        raise HTTPException(429, "خدمة الذكاء الاصطناعي مشغولة حاليًا. جرّب بعد دقيقة.") from exc


@router.post("/student/study-plan")
async def student_study_plan(
    req: StudentPlanRequest,
    session: AsyncSession = Depends(get_session),
):
    service = get_student_service(session)
    try:
        return await service.study_plan(
            req.user_id,
            req.project_id,
            req.weekly_hours,
            req.document_id,
            req.topic,
            req.language,
        )
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    except RateLimitError as exc:
        raise HTTPException(429, "خدمة الذكاء الاصطناعي مشغولة حاليًا. جرّب بعد دقيقة.") from exc


@router.post("/student/exams", status_code=201)
async def generate_student_exam(
    req: StudentExamRequest,
    session: AsyncSession = Depends(get_session),
):
    service = get_student_service(session)
    try:
        return await service.generate_exam(
            user_id=req.user_id,
            project_id=req.project_id,
            topic=req.topic,
            document_id=req.document_id,
            difficulty=req.difficulty,
            question_count=req.question_count,
            language=req.language,
            question_type=req.question_type,
        )
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RateLimitError as exc:
        raise HTTPException(
            status_code=429,
            detail=(
                "وصل نموذج الذكاء الاصطناعي إلى حد الطلبات الحالي. "
                "انتظر دقيقة ثم أعد المحاولة، أو اختر نموذجًا بحدود أعلى "
                "في GROQ_CHAT_MODEL داخل ملف .env."
            ),
            headers={"Retry-After": "60"},
        ) from exc
    except APIError as exc:
        logger.warning("AI generation task=student_practice_exam stage=provider_error error_category=%s",
                       type(exc).__name__)
        raise HTTPException(502, "تعذر الاتصال بخدمة إنشاء الامتحان. جرّب مرة أخرى.") from exc


@router.post("/student/analyze-image")
async def analyze_student_image(
    user_id: uuid.UUID = Form(...),
    project_id: uuid.UUID = Form(...),
    question: str = Form(default="اشرح التمرين أو الصورة دي."),
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    try:
        service = get_student_service(session)
        await service.validate_scope(user_id, project_id)
        image_bytes = await file.read(settings.VISION_MAX_UPLOAD_BYTES + 1)
        return await service.analyze_image(
            user_id=user_id,
            project_id=project_id,
            file_name=file.filename or "",
            image_bytes=image_bytes,
            question=question,
        )
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except ImageAnalysisProviderError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/student/exams/{exam_id}/questions/{question_index}/hint")
async def student_exam_question_hint(
    exam_id: uuid.UUID,
    question_index: int,
    req: StudentScopeRequest,
    session: AsyncSession = Depends(get_session),
):
    service = get_student_service(session)
    try:
        return await service.request_exam_question_hint(
            user_id=req.user_id,
            project_id=req.project_id,
            exam_id=exam_id,
            question_index=question_index,
        )
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/student/exams/{exam_id}/submit")
async def submit_student_exam(
    exam_id: uuid.UUID,
    req: StudentExamSubmitRequest,
    session: AsyncSession = Depends(get_session),
):
    service = get_student_service(session)
    try:
        return await service.submit_exam(
            user_id=req.user_id,
            project_id=req.project_id,
            exam_id=exam_id,
            answers=req.answers,
        )
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


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
        async for event in service.ask(parsed_id, req.question, req.mode):
            event_type = event["event"]
            data = json.dumps(event["data"], ensure_ascii=False)

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


@router.post("/chat/sessions/{session_id}/ask-image")
async def ask_about_image(
    session_id: str,
    file: UploadFile = File(...),
    question: str = Form(default="اشرح التمرين في الصورة خطوة بخطوة."),
    mode: str = Form(default="study"),
    session: AsyncSession = Depends(get_session),
):
    """Analyze an image as a turn in the existing chat conversation."""
    try:
        parsed_id = uuid.UUID(session_id)
    except ValueError as exc:
        raise HTTPException(400, "Invalid session ID") from exc

    try:
        image_bytes = await file.read(settings.VISION_MAX_UPLOAD_BYTES + 1)
        service = get_chat_service(session)
        return await service.ask_image(
            session_id=parsed_id,
            file_name=file.filename or "exercise-image.png",
            image_bytes=image_bytes,
            question=question,
            mode=mode,
        )
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except ImageAnalysisProviderError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RateLimitError as exc:
        raise HTTPException(429, "خدمة الذكاء الاصطناعي مشغولة حاليًا. جرّب بعد دقيقة.") from exc
