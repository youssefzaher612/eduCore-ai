"""Teacher workspace access to indexed course material and real exam records."""

from __future__ import annotations

import re
import uuid
from collections import defaultdict
from types import SimpleNamespace

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from educore.config import settings
from educore.database.models import (
    Chunk,
    Document,
    DocumentStatus,
    Project,
    StudentAttempt,
    StudentExam,
    User,
)
from educore.rag.retriever import Retriever


class TeacherWorkspaceService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def _validate_scope(
        self,
        user_id: uuid.UUID,
        project_id: uuid.UUID,
        document_id: uuid.UUID | None = None,
    ) -> tuple[Project, Document | None]:
        user = await self.session.get(User, user_id)
        project = await self.session.get(Project, project_id)
        if user is None or project is None:
            raise ValueError("الحساب أو المقرر مش موجود.")
        if project.owner_id != user_id:
            raise PermissionError("المقرر ده مش تابع للحساب المختار.")
        document = None
        if document_id is not None:
            document = await self.session.get(Document, document_id)
            if document is None or document.project_id != project_id:
                raise ValueError("اختار ملفًا من ملفات المقرر الحالي.")
            if document.status != DocumentStatus.INDEXED:
                raise ValueError("الملف المختار لسه ما اكتملش فهرسته.")
        return project, document

    async def course_material(
        self,
        user_id: uuid.UUID,
        project_id: uuid.UUID,
        document_id: uuid.UUID,
        topic: str,
        top_k: int = 8,
    ) -> dict:
        """Retrieve topic-grounded excerpts only from the selected indexed file."""
        _, document = await self._validate_scope(user_id, project_id, document_id)
        query = (topic or "").strip()
        if len(query) < 2:
            raise ValueError("اختار موضوعًا من موضوعات الملف الأول.")

        retrieved = await Retriever.search(
            self.session,
            project_id,
            query=query,
            top_k=max(4, min(int(top_k), 12)),
            min_score=settings.RETRIEVAL_MIN_SCORE,
            document_id=document_id,
        )

        terms = list(dict.fromkeys(
            term.casefold()
            for term in re.findall(r"[\w\u0600-\u06ff]+", query)
            if len(term) >= 3
        ))
        lexical_sources = []
        if terms:
            term_filter = or_(*[
                Chunk.content.ilike(f"%{term.replace('%', '').replace('_', '')}%")
                for term in terms[:8]
            ])
            result = await self.session.execute(
                select(Chunk.page_number, Chunk.content, Chunk.chunk_order)
                .where(Chunk.document_id == document_id, term_filter)
                .order_by(Chunk.chunk_order.asc())
                .limit(40)
            )
            rows = result.all()
            rows.sort(
                key=lambda row: (
                    sum(term in str(row.content).casefold() for term in terms)
                    + (3 if query.casefold() in str(row.content).casefold() else 0),
                    -int(row.chunk_order or 0),
                ),
                reverse=True,
            )
            lexical_sources = [
                SimpleNamespace(
                    document_id=str(document_id),
                    document_name=document.filename,
                    page_number=row.page_number,
                    content=row.content,
                    similarity=0.0,
                )
                for row in rows[:4]
            ]

        merged = {}
        for item in [*lexical_sources, *retrieved]:
            key = (item.document_name, item.page_number, item.content)
            merged.setdefault(key, item)
        sources = list(merged.values())[:max(4, min(int(top_k), 12))]
        if not sources:
            raise ValueError("ملقتش شرحًا مناسبًا للموضوع ده في الملف المفهرس.")

        excerpts = [
            {
                "document_name": item.document_name,
                "page_number": item.page_number,
                "content": item.content[:2400],
                "similarity": round(float(getattr(item, "similarity", 0) or 0), 4),
            }
            for item in sources
        ]
        context = "\n\n".join(
            f"SOURCE {index} | {item['document_name']} | page {item['page_number'] or 'unknown'}\n"
            f"{item['content']}"
            for index, item in enumerate(excerpts, start=1)
        )
        return {
            "document_id": str(document_id),
            "document_name": document.filename,
            "topic": query,
            "excerpts": excerpts,
            "context": context,
        }

    async def assessment_records(
        self,
        user_id: uuid.UUID,
        project_id: uuid.UUID,
        document_id: uuid.UUID | None = None,
    ) -> dict:
        """Build teacher analytics inputs from submitted EduCore exams, not a local data file."""
        project, selected_document = await self._validate_scope(
            user_id, project_id, document_id
        )
        statement = (
            select(StudentAttempt, StudentExam, User.name)
            .outerjoin(StudentExam, StudentExam.id == StudentAttempt.exam_id)
            .join(User, User.id == StudentAttempt.user_id)
            .where(
                StudentAttempt.project_id == project_id,
            )
            .order_by(StudentExam.submitted_at.asc(), StudentAttempt.created_at.asc())
        )
        result = await self.session.execute(statement)
        rows = result.all()

        by_exam: dict[tuple[uuid.UUID, uuid.UUID], dict] = {}
        for attempt, exam, student_name in rows:
            if exam is None or exam.status != "submitted":
                continue
            if selected_document is not None:
                citation_names = {
                    str(item.get("document_name", "")).casefold()
                    for item in (exam.citations or [])
                    if isinstance(item, dict)
                }
                question_document_ids = {
                    str(item.get("_source_document_id"))
                    for item in (exam.questions or [])
                    if isinstance(item, dict) and item.get("_source_document_id")
                }
                if (
                    selected_document.filename.casefold() not in citation_names
                    and str(document_id) not in question_document_ids
                ):
                    continue

            key = (attempt.user_id, exam.id)
            record = by_exam.setdefault(key, {
                "student_id": str(attempt.user_id),
                "student_name": student_name,
                "assessment_title": exam.title,
                "subject": project.name,
                "grade_level": "غير محدد",
                "awarded_marks": 0.0,
                "max_marks": 0.0,
                "assessment_date": exam.submitted_at.isoformat() if exam.submitted_at else None,
                "topic_scores": defaultdict(lambda: [0.0, 0.0]),
            })
            record["awarded_marks"] += float(bool(attempt.is_correct))
            record["max_marks"] += 1.0
            topic_score = record["topic_scores"][attempt.topic]
            topic_score[0] += float(bool(attempt.is_correct))
            topic_score[1] += 1.0

        records = []
        for record in by_exam.values():
            topics = [
                {
                    "topic": name,
                    "awarded_marks": score[0],
                    "max_marks": score[1],
                }
                for name, score in record.pop("topic_scores").items()
            ]
            records.append({**record, "topic_performance": topics})

        return {
            "project_name": project.name,
            "document_id": str(document_id) if document_id else None,
            "document_name": selected_document.filename if selected_document else None,
            "records": records,
        }
