
"""
Chat service — the heart of the EduCore RAG Q&A pipeline.

Pipeline:
process question -> retrieve context -> validate evidence
-> build prompt -> generate answer -> store messages.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from educore.config import settings
from educore.database.models import DocumentStatus, Project, User, UserRole
from educore.database.repositories import (
    ChatSessionRepository,
    DocumentRepository,
    MessageRepository,
)

from educore.rag.generator import OpenRouterGenerator
from educore.rag.prompts import build_messages
from educore.rag.query_processor import process_query
from educore.rag.query_rewriter import needs_contextual_rewrite, rewrite_for_retrieval
from educore.rag.retriever import RetrievalResult, Retriever
from educore.rag.response_cache import response_cache, semantic_response_cache
from educore.rag.validator import validate_chunk_citations
from educore.services.student_service import StudentLearningService
from educore.services.teacher_service import TeacherWorkspaceService


logger = logging.getLogger(__name__)


EMPTY_RESPONSE_ANSWER = (
    "حصلت مشكلة أثناء إنشاء الإجابة. حاول مرة تانية."
)


def _teacher_performance_intent(question: str) -> bool:
    text = (question or "").casefold().translate(str.maketrans({
        "أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه",
    }))
    return any(term in text for term in (
        "مستوي الطلاب", "مستوي الطالب", "مستوي الطلبه", "مستواهم",
        "اداء الطلاب", "اداء الطلبه", "الاداء",
        "درجات الطلاب", "درجات الطلبه", "درجاتهم", "مستوي الدرجات", "الدرجات",
        "نتائج الطلاب", "نتائج الطلبه", "نتايج الطلاب",
        "تقييم الطلاب", "الطلاب عاملين ايه", "الطلبه عاملين ايه",
        "student performance", "students' performance", "student grades",
        "students grades", "class performance", "student scores", "student level",
    ))


def _format_teacher_performance(records: list[dict], project_name: str) -> str:
    if not records:
        return f"لا توجد امتحانات مُسلَّمة مسجلة حتى الآن في مقرر {project_name}."

    students: dict[str, dict] = {}
    for row in records:
        student_id = row["student_id"]
        student = students.setdefault(student_id, {
            "name": row.get("student_name") or "طالب بدون اسم",
            "scores": [],
            "exams": [],
            "topics": {},
        })
        max_marks = float(row.get("max_marks") or 0)
        awarded = float(row.get("awarded_marks") or 0)
        if max_marks:
            student["scores"].append(100 * awarded / max_marks)
        student["exams"].append((row.get("assessment_title") or "امتحان", awarded, max_marks))
        for topic in row.get("topic_performance", []):
            bucket = student["topics"].setdefault(topic["topic"], [0.0, 0.0])
            bucket[0] += float(topic.get("awarded_marks") or 0)
            bucket[1] += float(topic.get("max_marks") or 0)

    all_scores = [score for student in students.values() for score in student["scores"]]
    average = sum(all_scores) / len(all_scores) if all_scores else 0.0
    lines = [
        f"ملخص أداء الطلاب في مقرر {project_name} (من الامتحانات المُسلَّمة):",
        f"متوسط الدرجات: {average:.1f}% | عدد الطلاب: {len(students)} | عدد المحاولات: {len(all_scores)}",
    ]
    for student in sorted(students.values(), key=lambda item: item["name"].casefold()):
        avg = sum(student["scores"]) / len(student["scores"]) if student["scores"] else 0.0
        lines.append(f"- {student['name']}: المتوسط {avg:.1f}% من {len(student['scores'])} امتحان/امتحانات.")
        for title, awarded, maximum in student["exams"]:
            pct = 100 * awarded / maximum if maximum else 0.0
            lines.append(f"  - {title}: {awarded:g}/{maximum:g} ({pct:.1f}%).")
        if student["topics"]:
            topics = []
            for name, (awarded, maximum) in student["topics"].items():
                if maximum:
                    topics.append(f"{name} {100 * awarded / maximum:.0f}%")
            if topics:
                lines.append("  - حسب الموضوع: " + "، ".join(topics) + ".")
    lines.append("الدرجات محسوبة من إجابات الامتحانات المسلَّمة المسجلة في النظام.")
    return "\n".join(lines)


def _answer_language(question: str, requested_language: str | None) -> str:
    if requested_language:
        return requested_language
    if re.search(r"[\u0600-\u06ff]", question):
        return "Arabic, matching the user's dialect and register"
    return "the same language and register used by the user"


def _greeting_response(question: str) -> str | None:
    """Handle brief social greetings locally instead of misrouting them as progress questions."""
    normalized = re.sub(r"[\u064b-\u065f\u0670\u0640]", "", (question or "").casefold())
    normalized = normalized.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي"}))
    normalized = re.sub(r"[^\w\s\u0600-\u06ff]", " ", normalized)
    normalized = " ".join(normalized.split())
    if len(normalized.split()) > 7:
        return None

    if re.fullmatch(r"(?:ازيك|ازيكم|عامل ايه|عامله ايه|اخبارك ايه|ايه اخبارك|كيف حالك)(?:\s+(?:عامل ايه|عامله ايه))?", normalized):
        return "أهلاً! أنا بخير، شكرًا لسؤالك 😊 اسألني عن محاضراتك أو مذاكرتك أو تقدمك الدراسي."
    if re.fullmatch(r"(?:hi|hello|hey|how are you|how's it going|good morning|good evening)(?:\s+there)?", normalized):
        return "Hi! I’m doing well, thanks for asking. Ask me about your course materials, studying, or progress."
    return None


def _normalize_short_reply(text: str) -> str:
    text = re.sub(r"[\u064b-\u065f\u0670\u0640]", "", text.lower())
    text = text.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه"}))
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


_YES_REPLIES = {
    "اه", "اه ايوه", "ايوه", "ايوة", "اها", "نعم", "تمام", "ماشي",
    "اكيد", "طبعا", "ابحث", "دور", "كمل", "yes", "yeah", "yep",
    "sure", "ok", "okay", "go ahead", "search", "continue",
}
_NO_REPLIES = {
    "لا", "لأ", "لا شكرا", "لا شكرا مش عايز", "مش عايز", "مش عايزه",
    "مش عايزه شكرا", "مش محتاج", "no", "no thanks", "no thank you",
    "nope", "not now", "dont", "do not", "skip",
}


def _confirmation_kind(text: str) -> str | None:
    normalized = _normalize_short_reply(text)
    yes_with_extra = re.fullmatch(
        r"(?:اه|ايوه|اها|نعم|تمام|ماشي|اكيد|طبعا|yes|yeah|yep|sure|ok|okay)"
        r"(?:\s+(?:لو\s+سمحت|من\s+فضلك|ياريت|يا\s+ريت|موافق|موافقه|اكيد|طبعا|please|ابحث|دور|كمل|search(?:\s+(?:the\s+)?web|\s+online)?|continue|go\s+ahead|على\s+النت|على\s+الانترنت))*",
        normalized,
    )
    if normalized in _YES_REPLIES or yes_with_extra:
        return "yes"
    no_with_extra = re.fullmatch(
        r"(?:لا|مش\s+عايز|مش\s+عايزه|مش\s+محتاج|no|nope|not\s+now|dont|do\s+not|skip)"
        r"(?:\s+(?:مش\s+عايزه?|مش\s+محتاج|شكرا|thanks|thank\s+you))*",
        normalized,
    )
    if normalized in _NO_REPLIES or no_with_extra:
        return "no"
    return None


def _explicitly_requests_external_info(question: str) -> bool:
    patterns = (
        r"\b(?:search(?:\s+the\s+web|\s+online)?|look\s+up|google|find\s+online|"
        r"on\s+the\s+internet|from\s+(?:the\s+)?(?:web|internet|online\s+sources?)|"
        r"external\s+sources?)\b",
        r"\boutside\s+(?:the\s+)?(?:files?|lectures?|documents?|course|sources?)\b",
        r"(?:من\s*(?:بره|برا|خارج|النت|الإنترنت|الانترنت|الويب|مصدر خارجي)|"
        r"(?:برا|بره|خارج)\s*(?:المحاضرات|المحاضره|الملفات|الملف|المصادر|المقرر)|"
        r"على\s*(?:النت|الإنترنت|الانترنت|الويب))",
        r"(?:دور|ابحث|هات|جيب)\s*(?:لي|لنا)?\s*(?:على|في|من)?\s*(?:النت|الإنترنت|الانترنت|الويب|مصدر خارجي)",
        r"(?:دور|ابحث|هات|جيب).{0,18}(?:بره|برا|خارج|النت|الإنترنت|الانترنت|الويب)",
        r"(?:معلومات|معلومه|مصدر|مصادر).{0,24}(?:خارجي|من بره|من النت|على النت|على الإنترنت)",
    )
    return any(re.search(pattern, question, re.IGNORECASE) for pattern in patterns)


def _requests_whole_document(question: str) -> bool:
    text = (question or "").casefold()
    whole_scope = bool(re.search(
        r"(?:محتوى\s+(?:ال)?محاضره?|محتوي\s+(?:ال)?محاضره?|"
        r"اشرح\s+(?:لي\s+)?(?:ال)?محاضره?|لخص\s+(?:لي\s+)?(?:ال)?محاضره?|"
        r"content\s+of\s+(?:the\s+)?lecture|lecture\s+contents?)",
        text,
    ))
    has_file = bool(re.search(
        r"(?:الملف|ملف|الفايل|فايل|المحاضره|المحاضرة|المحاضرات|المستند|الكتاب|document|lecture|file|book)",
        text,
    ))
    if not has_file:
        return False
    explicit_scope = bool(re.search(
        r"(?:كله|كلها|بالكامل|كامل|كامله|كاملة|من\s+الأول\s+للآخر|"
        r"entire|whole|full\s+(?:file|document|lecture)|from\s+start\s+to\s+finish)",
        text,
    ))
    short_file_request = len(text.split()) <= 7 and bool(re.search(
        r"(?:\b(?:explain|summari[sz]e|teach)\s+(?:(?:the|my)\s+)?(?:file|document|lecture)(?:\s+in\s+detail)?\b|"
        r"(?:اشرح(?:لي)?|فهمني|لخص(?:لي)?)\s*(?:ال)?(?:ملف|فايل|محاضره|محاضرة|لكتشر)(?:\s+بالتفصيل)?)",
        text,
    ))
    return whole_scope or explicit_scope or short_file_request


def _web_citations(answer: str) -> list[dict]:
    citations = []
    seen = set()
    for title, url in re.findall(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", answer):
        if url in seen:
            continue
        seen.add(url)
        citations.append({
            "source_number": len(citations) + 1,
            "document_name": title.strip(),
            "url": url,
            "snippet": "",
        })
    return citations


def _limit_to_sentences(answer: str, count: int) -> str:
    """Keep a short answer within the exact sentence count requested by the user."""
    parts = re.split(r"(?<=[.!?؟])\s+|\n+", (answer or "").strip())
    sentences = [part.strip() for part in parts if part.strip()]
    return " ".join(sentences[:count])


def _external_messages(
    question: str,
    mode: str,
    language: str,
    chat_history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    mode_instruction = (
        "Give one useful hint that helps the student reason toward the answer. "
        "Do not state the final answer, the correct option, or a complete solution."
        if mode == "hint"
        else "Teach the topic clearly and thoroughly, explaining key ideas and useful examples."
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Answer the user's question using information found through web search and your "
                "general knowledge where it helps explain the topic. Search first, prefer reliable "
                "primary or educational sources, and distinguish established facts from uncertainty. "
                "Respond in " + language + ". " + mode_instruction + " "
                "Cite sources as Markdown links using only real URLs from pages you visited. "
                "Never invent a source or URL. Do not include internal citation markers such as "
                "〖2†L6-L10〗; put a short Sources list of Markdown links at the end."
            ),
        },
    ]
    messages.extend(
        {"role": item["role"], "content": item["content"][:4000]}
        for item in (chat_history or [])[-6:]
        if item.get("role") in {"user", "assistant"} and item.get("content", "").strip()
    )
    messages.append({"role": "user", "content": question})
    return messages


class ChatService:
    """Manages conversations and RAG-grounded Q&A."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.chat_repo = ChatSessionRepository(session)
        self.message_repo = MessageRepository(session)
        self.document_repo = DocumentRepository(session)
        chat_model = settings.active_chat_model
        self.generator = OpenRouterGenerator(
            model_id=chat_model,
            reasoning_effort="medium",
        )

    async def _whole_document_sources(
        self,
        project_id: uuid.UUID,
        question: str,
    ) -> tuple[list[RetrievalResult], str | None]:
        """Load every indexed chunk for the requested file, when it is unambiguous."""
        documents = await self.document_repo.list_by_project(project_id)
        indexed = [doc for doc in documents if doc.status == DocumentStatus.INDEXED]
        if not indexed:
            return [], (
                "مفيش ملفات مفهرسة في المقرر لسه. ارفع المحاضرة واستنى لحد ما الفهرسة تكتمل، "
                "وبعدها هشرحها لك من أولها لآخرها."
                if re.search(r"[\u0600-\u06ff]", question)
                else "There are no indexed course files yet. Upload a lecture and wait for indexing, then I can explain the full file."
            )

        normalized_question = re.sub(r"[^\w\u0600-\u06ff]+", " ", question.casefold()).strip()
        named = []
        for doc in indexed:
            filename = (doc.filename or "").casefold()
            stem = filename.rsplit(".", 1)[0]
            filename = re.sub(r"[^\w\u0600-\u06ff]+", " ", filename).strip()
            stem = re.sub(r"[^\w\u0600-\u06ff]+", " ", stem).strip()
            if any(candidate and candidate in normalized_question for candidate in (filename, stem)):
                named.append(doc)

        if named:
            candidates = named
        else:
            ordinal = re.search(r"(?:المحاضره|المحاضرة|lecture)\s*(?:رقم\s*)?(الأولى|الاولى|الثانيه|الثانية|الثالثه|الثالثة|الرابعه|الرابعة|1|2|3|4)", question, re.IGNORECASE)
            ordinal_map = {"الأولى": "1", "الاولى": "1", "الثانيه": "2", "الثانية": "2", "الثالثه": "3", "الثالثة": "3", "الرابعه": "4", "الرابعة": "4"}
            if ordinal:
                number = ordinal_map.get(ordinal.group(1), ordinal.group(1))
                numbered = [doc for doc in indexed if re.search(
                    rf"(?:lecture|محاضره|محاضرة)[\s_-]*0?{number}(?:\D|$)",
                    doc.filename,
                    re.IGNORECASE,
                )]
                if numbered:
                    candidates = numbered
                else:
                    candidates = []
            else:
                candidates = []
            by_name: dict[str, list] = {}
            for doc in (candidates or indexed):
                by_name.setdefault((doc.filename or "").casefold(), []).append(doc)
            if not candidates and len(by_name) > 1:
                names = list(dict.fromkeys(doc.filename for doc in indexed))
                listed = "، ".join(f"**{name}**" for name in names[:8])
                return [], (
                    f"عندي أكتر من ملف مفهرس: {listed}. تقصد أشرح أنهي ملف؟"
                    if re.search(r"[\u0600-\u06ff]", question)
                    else f"More than one course file is indexed: {', '.join(names[:8])}. Which one should I explain?"
                )
            if not candidates:
                candidates = indexed

        selected = max(
            candidates,
            key=lambda doc: doc.created_at.timestamp() if doc.created_at else 0,
        )
        sources = await Retriever.search(
            session=self.session,
            project_id=project_id,
            query=question,
            retrieval_query=question,
            top_k=100,
            min_score=settings.RETRIEVAL_MIN_SCORE,
            document_id=selected.id,
        )
        if not sources:
            return [], (
                f"ملقتش مقاطع من **{selected.filename}** تتجاوز حد التشابه المطلوب. جرّب سؤالًا أكثر تحديدًا."
                if re.search(r"[\u0600-\u06ff]", question)
                else f"No passages from **{selected.filename}** met the similarity threshold. Try a more specific question."
            )
        return sources, None

    async def _retrieve_query_variants(
        self,
        project_id: uuid.UUID,
        question: str,
        queries: list[str],
        existing_results: list | None = None,
    ) -> list:
        """Search original and rewritten forms, then merge duplicate chunks."""
        merged = {}
        for result in existing_results or []:
            key = (result.document_id, result.page_number, result.content)
            merged[key] = result

        unique_queries = []
        seen_queries = set()
        for query in queries:
            query = (query or "").strip()
            if query and query.casefold() not in seen_queries:
                seen_queries.add(query.casefold())
                unique_queries.append(query)

        for query in unique_queries:
            results = await Retriever.search(
                session=self.session,
                project_id=project_id,
                query=question,
                retrieval_query=query,
                top_k=settings.RETRIEVAL_TOP_K,
                min_score=settings.RETRIEVAL_MIN_SCORE,
            )
            for result in results:
                key = (result.document_id, result.page_number, result.content)
                previous = merged.get(key)
                if previous is None or result.similarity > previous.similarity:
                    merged[key] = result

        return sorted(
            merged.values(), key=lambda item: item.similarity, reverse=True
        )[:settings.RETRIEVAL_TOP_K]

    async def create_session(
        self,
        user_id: uuid.UUID,
        project_id: uuid.UUID,
        section: UserRole,
        title: str = "محادثة جديدة",
    ):
        """Create a new chat session."""

        return await self.chat_repo.create(
            user_id=user_id,
            project_id=project_id,
            section=section,
            title=title,
        )

    async def get_session(self, session_id: uuid.UUID):
        """Get a chat session by ID."""

        return await self.chat_repo.get_by_id(session_id)

    async def list_sessions(
        self,
        user_id: uuid.UUID,
        project_id: uuid.UUID,
    ):
        """List all chat sessions for a user in a project."""

        sessions = await self.chat_repo.list_by_user_and_project(
            user_id,
            project_id,
        )
        first_questions = await self.message_repo.first_user_messages(
            [item.id for item in sessions]
        )

        return [
            {
                "id": str(s.id),
                "title": (
                    first_questions.get(s.id, s.title or "محادثة محفوظة")
                    .replace("\r", " ").replace("\n", " ").strip()[:72]
                ),
                "section": s.section.value,
                "created_at": (
                    s.created_at.isoformat()
                    if s.created_at else None
                ),
            }
            for s in sessions
        ]

    async def get_messages(
        self,
        session_id: uuid.UUID,
    ) -> list[dict]:
        """Get all messages in a chat session."""

        messages = await self.message_repo.list_by_session(session_id)

        return [
            {
                "id": str(m.id),
                "role": m.role,
                "content": m.content,
                "citations": m.citations,
                "created_at": (
                    m.created_at.isoformat()
                    if m.created_at else None
                ),
            }
            for m in messages
        ]

    async def ask_image(
        self,
        session_id: uuid.UUID,
        file_name: str,
        image_bytes: bytes,
        question: str,
        mode: str = "study",
    ) -> dict:
        """Answer an image question and persist both turns in the same chat."""
        chat_session = await self.chat_repo.get_by_id(session_id)
        if chat_session is None:
            raise ValueError("Chat session not found")

        history = await self.message_repo.get_recent_history(
            session_id,
            limit=min(settings.CHAT_HISTORY_LIMIT, 6),
        ) if settings.CHAT_HISTORY_LIMIT else []
        answer = await StudentLearningService(self.session).analyze_image(
            user_id=chat_session.user_id,
            project_id=chat_session.project_id,
            file_name=file_name,
            image_bytes=image_bytes,
            question=question,
            chat_history=history,
            mode=mode,
        )

        user_content = f"📷 {file_name}\n{question.strip() or 'اشرح التمرين في الصورة خطوة بخطوة.'}"
        await self.message_repo.create(
            session_id=session_id,
            role="user",
            content=user_content,
        )
        assistant_msg = await self.message_repo.create(
            session_id=session_id,
            role="assistant",
            content=answer["answer"],
        )
        return {
            "answer": assistant_msg.content,
            "message_id": str(assistant_msg.id),
        }

    async def _external_answer(
        self,
        session_id: uuid.UUID,
        question: str,
        mode: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> AsyncGenerator[dict, None]:
        """Search the web after explicit user intent or consent."""
        if not settings.active_web_search_enabled:
            answer = (
                "البحث على الويب مغلق حاليًا في إعدادات Groq، لذلك لم أرسل أي استعلام خارجي. "
                "أقدر أساعدك من المحاضرات المرفوعة؛ يتطلب البحث تفعيله صراحةً وموافقتك الواضحة."
                if re.search(r"[\u0600-\u06ff]", question)
                else "Web search is currently disabled in the Groq settings, so no external query was sent. "
                "I can help from uploaded lectures; web search requires explicit configuration and your clear consent."
            )
            assistant_msg = await self.message_repo.create(
                session_id=session_id,
                role="assistant",
                content=answer,
                citations=[],
            )
            yield {"event": "meta", "data": {"citations": [], "total_sources": 0}}
            yield {"event": "delta", "data": answer}
            yield {"event": "done", "data": {"message_id": str(assistant_msg.id)}}
            return

        language = _answer_language(
            question,
            process_query(question).requested_language,
        )
        try:
            answer = await self.generator.generate_with_web_search(
                _external_messages(question, mode, language, chat_history)
            )
            citations = _web_citations(answer)
        except Exception:
            logger.exception("External search failed for chat %s", session_id)
            answer = (
                "البحث على الإنترنت مش متاح دلوقتي. جرّب تاني بعد شوية."
                if re.search(r"[\u0600-\u06ff]", question)
                else "Web search is unavailable right now. Please try again shortly."
            )
            citations = []

        assistant_msg = await self.message_repo.create(
            session_id=session_id,
            role="assistant",
            content=answer,
            citations=citations,
        )
        yield {
            "event": "meta",
            "data": {"citations": citations, "total_sources": len(citations)},
        }
        yield {"event": "delta", "data": answer}
        yield {"event": "done", "data": {"message_id": str(assistant_msg.id)}}

    @staticmethod
    def _external_permission_answer(
        question: str,
        requested_language: str | None = None,
    ) -> str:
        if not settings.active_web_search_enabled:
            if requested_language == "English" or not re.search(r"[\u0600-\u06ff]", question):
                return (
                    "I couldn't find enough information in your uploaded lectures. "
                    "Web search is disabled in the Groq settings, so I can only answer from uploaded materials."
                )
            return (
                "ملقتش معلومات كفاية في المحاضرات المرفوعة. البحث على الويب مغلق في إعدادات Groq، "
                "فأقدر أجاوبك من المواد المرفوعة فقط."
            )
        if requested_language == "English":
            return (
                "I couldn't find enough information in your uploaded lectures. "
                "Would you like me to search the web and answer from outside sources?"
            )
        if requested_language == "Arabic" or re.search(r"[\u0600-\u06ff]", question):
            return (
                "ملقتش إجابة كافية في المحاضرات المرفوعة. "
                "تحب أبحث على الإنترنت وأكمل من مصادر خارجية؟"
            )
        return (
            "I couldn't find enough information in your uploaded lectures. "
            "Would you like me to search the web and answer from outside sources?"
        )

    async def _save_and_stream(
        self,
        session_id: uuid.UUID,
        answer: str,
        citations: list[dict] | None = None,
    ) -> AsyncGenerator[dict, None]:
        citations = citations or []
        assistant_msg = await self.message_repo.create(
            session_id=session_id,
            role="assistant",
            content=answer,
            citations=citations,
        )
        yield {
            "event": "meta",
            "data": {"citations": citations, "total_sources": len(citations)},
        }
        yield {"event": "delta", "data": answer}
        yield {"event": "done", "data": {"message_id": str(assistant_msg.id)}}

    async def ask(
        self,
        session_id: uuid.UUID,
        question: str,
        mode: str = "study",
    ) -> AsyncGenerator[dict, None]:
        """
        Full RAG pipeline.

        Events:
            meta: citation metadata
            delta: generated answer text
            done: saved message ID
            error: error message
        """

        try:
            # 1. Validate the original question.
            if not question or not question.strip():
                yield {
                    "event": "error",
                    "data": "Question cannot be empty",
                }
                return

            question = question.strip()
            mode = mode if mode in {"study", "hint"} else "study"

            # 2. Process the question.
            # Keep the original question for validation and generation.
            processed = process_query(question)

            requested_language = processed.requested_language

            logger.info("=" * 60)
            logger.info("RAG REQUEST STARTED")
            logger.info("SESSION ID: %s", session_id)
            logger.info("ORIGINAL QUESTION: %s", question)
            logger.info(
                "RETRIEVAL QUERY: %s",
                processed.retrieval_query,
            )
            logger.info(
                "REQUESTED LANGUAGE: %s",
                requested_language or "Not specified",
            )
            logger.info(
                "ANSWER INSTRUCTIONS: %s",
                processed.answer_instructions or "None",
            )

            # 3. Load chat session.
            chat_session = await self.chat_repo.get_by_id(session_id)

            if not chat_session:
                yield {
                    "event": "error",
                    "data": "Chat session not found",
                }
                return

            logger.info("PROJECT ID: %s", chat_session.project_id)
            logger.info("SECTION: %s", chat_session.section)

            # Keep earlier turns for pronoun resolution, follow-up questions, and natural answers.
            chat_history = await self.message_repo.get_recent_history(
                session_id,
                limit=settings.CHAT_HISTORY_LIMIT,
            ) if settings.CHAT_HISTORY_LIMIT else []

            # 4. Save original user question.
            await self.message_repo.create(
                session_id=session_id,
                role="user",
                content=question,
            )

            explicitly_external = _explicitly_requests_external_info(question)
            pending_question = chat_session.pending_external_question
            confirmation = (
                _confirmation_kind(question) if pending_question else None
            )

            if confirmation == "yes":
                external_question = pending_question
                external_mode = chat_session.pending_response_mode or mode
                chat_session.pending_external_question = None
                chat_session.pending_response_mode = None
                async for event in self._external_answer(
                    session_id, external_question, external_mode, chat_history
                ):
                    yield event
                return

            if confirmation == "no":
                chat_session.pending_external_question = None
                chat_session.pending_response_mode = None
                declined = (
                    "تمام، هكتفي بالمعلومات الموجودة في المحاضرات."
                    if re.search(r"[\u0600-\u06ff]", question)
                    else "Okay, I'll stick to the information in your lectures."
                )
                async for event in self._save_and_stream(session_id, declined):
                    yield event
                return

            # A new question replaces a pending consent prompt. Explicitly
            # external requests also clear it and continue immediately.
            chat_session.pending_external_question = None
            chat_session.pending_response_mode = None

            if explicitly_external:
                async for event in self._external_answer(session_id, question, mode, chat_history):
                    yield event
                return

            greeting = _greeting_response(question)
            if greeting:
                async for event in self._save_and_stream(session_id, greeting):
                    yield event
                return

            # Student progress questions are answered from this student's
            # recorded attempts; course-content questions continue through RAG.
            if chat_session.section == UserRole.STUDENT:
                student_answer = await StudentLearningService(
                    self.session
                ).answer_student_question(
                    user_id=chat_session.user_id,
                    project_id=chat_session.project_id,
                    question=question,
                    chat_history=chat_history,
                    model_id=settings.active_chat_model,
                )
                if student_answer:
                    assistant_msg = await self.message_repo.create(
                        session_id=session_id,
                        role="assistant",
                        content=student_answer,
                        citations=[],
                    )
                    yield {
                        "event": "meta",
                        "data": {"citations": [], "total_sources": 0},
                    }
                    yield {"event": "delta", "data": student_answer}
                    yield {
                        "event": "done",
                        "data": {"message_id": str(assistant_msg.id)},
                    }
                    return

            # Teacher performance questions use submitted assessment records,
            # scoped to the teacher's own project. Never add this data to student chats.
            if (
                chat_session.section == UserRole.TEACHER
                and _teacher_performance_intent(question)
            ):
                teacher = await self.session.get(User, chat_session.user_id)
                project = await self.session.get(Project, chat_session.project_id)
                if (
                    teacher is not None
                    and project is not None
                    and project.owner_id == teacher.id
                ):
                    report = await TeacherWorkspaceService(self.session).assessment_records(
                        user_id=teacher.id,
                        project_id=project.id,
                    )
                    answer = _format_teacher_performance(
                        report["records"], report["project_name"]
                    )
                    assistant_msg = await self.message_repo.create(
                        session_id=session_id,
                        role="assistant",
                        content=answer,
                        citations=[],
                    )
                    yield {
                        "event": "meta",
                        "data": {"citations": [], "total_sources": 0},
                    }
                    yield {"event": "delta", "data": answer}
                    yield {
                        "event": "done",
                        "data": {"message_id": str(assistant_msg.id)},
                    }
                    return

            # 5. Retrieve relevant chunks.
            # Retriever processes the query before embedding.
            logger.info("Starting semantic retrieval...")

            full_document_request = _requests_whole_document(question)
            if full_document_request:
                retrieval_results, clarification = await self._whole_document_sources(
                    chat_session.project_id,
                    question,
                )
                if clarification:
                    async for event in self._save_and_stream(session_id, clarification):
                        yield event
                    return
                rewritten_query = question
            else:
                rewritten_query = (
                    await rewrite_for_retrieval(
                        self.generator,
                        question,
                        processed.retrieval_query,
                        chat_history=chat_history,
                    )
                    if needs_contextual_rewrite(question, chat_history)
                    else processed.retrieval_query
                )
                logger.info("LLM REWRITTEN RETRIEVAL QUERY: %s", rewritten_query)

                retrieval_results = await self._retrieve_query_variants(
                    project_id=chat_session.project_id,
                    question=question,
                    queries=[rewritten_query],
                )

            logger.info(
                "RETRIEVAL COMPLETED: %d sources",
                len(retrieval_results),
            )

            # 6. Log retrieved sources.
            for i, result in enumerate(retrieval_results, start=1):
                logger.info(
                    "SOURCE %d | FILE=%s | PAGE=%s | SCORE=%.4f",
                    i,
                    result.document_name,
                    result.page_number,
                    result.similarity,
                )

                logger.info(
                    "SOURCE %d CONTENT: %s",
                    i,
                    result.content[:1500],
                )

            # 8. Handle empty retrieval.
            if not retrieval_results:
                logger.warning(
                    "NO RELEVANT SOURCES FOUND: %s",
                    question,
                )

                chat_session.pending_external_question = question
                chat_session.pending_response_mode = mode
                permission_answer = self._external_permission_answer(
                    question, requested_language
                )
                async for event in self._save_and_stream(
                    session_id, permission_answer
                ):
                    yield event
                return

            # 9. Build LLM messages with real chunk IDs; citation checks happen locally.
            messages = build_messages(
                section=chat_session.section,
                question=question,
                retrieval_results=retrieval_results,
                chat_history=chat_history,
            )

            # 12. Add processed answer instructions.
            answer_instructions = []

            if processed.answer_instructions:
                answer_instructions.append(
                    processed.answer_instructions
                )

            if requested_language:
                answer_instructions.append(
                    f"Answer entirely in {requested_language}. "
                    "This requirement has priority over the language "
                    "of the question and source documents."
                )
            else:
                answer_instructions.append(
                    "Answer in the same language and register as the user's "
                    "question. For Egyptian Arabic, use natural Egyptian Arabic."
                )

            answer_instructions.append(
                "Interpret brief, colloquial, misspelled, or mixed-language wording by matching it "
                "to the relevant terms and explanations in the retrieved lecture passages. Answer "
                "the request the user meant, while staying within what those passages support."
            )

            if chat_session.section == UserRole.STUDENT:
                if mode == "hint":
                    answer_instructions.append(
                        "HINT MODE: Do not reveal the final answer, correct option, "
                        "or a complete worked solution. Give one useful hint grounded "
                        "in the lecture, then invite the student to try."
                    )
                else:
                    if processed.requested_sentence_count is not None:
                        answer_instructions.append(
                            "STUDY MODE: Explain the central idea clearly and concisely within "
                            "the exact sentence limit requested by the user."
                        )
                    else:
                        answer_instructions.append(
                            "STUDY MODE: Explain the idea clearly and thoroughly, define "
                            "important terms, connect the steps, and use examples only "
                            "when they are supported by the lecture context."
                        )

            if answer_instructions:
                messages[0]["content"] += (
                    "\n\nUSER ANSWER REQUIREMENTS:\n"
                    + "\n".join(answer_instructions)
                )

            # 10. Log prompt roles and bounded content for debugging.
            logger.info("=" * 60)
            logger.info("LLM MESSAGES")

            for i, message in enumerate(messages, start=1):
                logger.info(
                    "MESSAGE %d | ROLE=%s",
                    i,
                    message["role"],
                )

                logger.info(
                    "MESSAGE %d CONTENT:\n%s",
                    i,
                    message["content"][:5000],
                )

            logger.info("=" * 60)

            # 11. Generate once, then validate cited chunk IDs locally.
            cache_key = response_cache.key(
                user_id=chat_session.user_id,
                project_id=chat_session.project_id,
                role=chat_session.section,
                task=f"rag:{mode}:{requested_language or 'auto'}",
                model=settings.active_chat_model,
                messages=messages,
            )
            full_response = response_cache.get(cache_key)
            cache_hit = full_response is not None
            semantic_scope, semantic_query = semantic_response_cache.scope_key(
                user_id=chat_session.user_id,
                project_id=chat_session.project_id,
                role=chat_session.section,
                task=f"rag:{mode}:{requested_language or 'auto'}",
                model=settings.active_chat_model,
                messages=messages,
            )
            semantic_vector = None
            if not cache_hit and semantic_query:
                try:
                    from educore.rag.embedder import Embedder

                    semantic_vector = Embedder.embed_query(semantic_query)
                    full_response = semantic_response_cache.get(
                        semantic_scope, semantic_vector
                    )
                    cache_hit = full_response is not None
                except Exception:
                    # A cache miss or embedding hiccup must never block RAG.
                    logger.debug("Semantic response cache lookup skipped", exc_info=True)
            if cache_hit:
                logger.info("RAG response cache hit for project %s", chat_session.project_id)
            else:
                logger.info("Starting RAG answer generation")
                full_response = ""
                if processed.requested_sentence_count is not None:
                    full_response = await self.generator.generate(
                        messages,
                        max_tokens=min(settings.OPENROUTER_MAX_TOKENS, 1200),
                        temperature=0.2,
                    )
                    full_response = _limit_to_sentences(
                        full_response,
                        processed.requested_sentence_count,
                    )
                else:
                    async for text_chunk in self.generator.generate_stream(
                        messages,
                        temperature=0.2,
                    ):
                        if text_chunk:
                            full_response += text_chunk
                            yield {"event": "delta", "data": text_chunk}

            if not full_response.strip():
                logger.error("RAG answer generation returned an empty response")
                full_response = EMPTY_RESPONSE_ANSWER

            full_response, verified_citations, citation_issue = validate_chunk_citations(
                full_response,
                retrieval_results,
            )
            if citation_issue:
                logger.warning("RAG answer contained missing or unmatched chunk citations")
            citations = verified_citations
            if not cache_hit and full_response.strip() != EMPTY_RESPONSE_ANSWER:
                response_cache.put(cache_key, full_response)
                if semantic_vector is not None:
                    semantic_response_cache.put(
                        semantic_scope, semantic_vector, full_response
                    )

            yield {
                "event": "meta",
                "data": {"citations": citations, "total_sources": len(citations)},
            }
            if processed.requested_sentence_count is not None:
                yield {"event": "delta", "data": full_response}
            else:
                yield {"event": "replace", "data": full_response}

            # 17. Save assistant response.
            assistant_msg = await self.message_repo.create(
                session_id=session_id,
                role="assistant",
                content=full_response,
                citations=citations,
            )

            # 18. Finish response.
            yield {
                "event": "done",
                "data": {
                    "message_id": str(assistant_msg.id),
                },
            }

            logger.info(
                "RAG REQUEST COMPLETED | SESSION=%s | "
                "SOURCES=%d | RESPONSE_CHARS=%d",
                session_id,
                len(retrieval_results),
                len(full_response),
            )

            logger.info("=" * 60)

        except Exception as e:
            logger.exception(
                "ERROR IN CHAT PIPELINE: %s",
                e,
            )

            yield {
                "event": "error",
                "data": str(e),
            }
