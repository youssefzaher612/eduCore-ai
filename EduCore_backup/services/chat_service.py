
"""
Chat service — the heart of the RAG Q&A pipeline.

Orchestrates:
retrieve context -> validate context -> build prompt
-> stream answer -> store messages.

Includes mixed Arabic/English language handling,
retrieval query cleaning, sentence limits, and evidence validation.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from educore.database.models import UserRole
from educore.database.repositories import (
    ChatSessionRepository,
    MessageRepository,
)

from educore.rag.generator import GroqGenerator
from educore.rag.prompts import build_messages
from educore.rag.retriever import Retriever
from educore.rag.validator import EvidenceValidator


logger = logging.getLogger(__name__)


EMPTY_RESPONSE_ANSWER = (
    "حصلت مشكلة أثناء إنشاء الإجابة. حاول مرة تانية."
)


def detect_requested_language(question: str) -> str:
    """
    Detect the requested answer language.

    Explicit language instructions have priority.
    Otherwise, Arabic script means Arabic response,
    including mixed Arabic/English technical questions.
    """

    text = question.lower()

    english_phrases = [
        "answer in english",
        "answer in english only",
        "respond in english",
        "respond in english only",
        "reply in english",
        "reply in english only",
        "english only",
        "use english",
        "use english only",
    ]

    arabic_phrases = [
        "answer in arabic",
        "answer in arabic only",
        "respond in arabic",
        "respond in arabic only",
        "reply in arabic",
        "arabic only",
        "use arabic",
        "بالعربي فقط",
        "جاوب بالعربي",
        "جاوب باللغة العربية",
        "أجب بالعربية",
        "اكتب بالعربي",
        "اكتب باللغة العربية",
    ]

    # Explicit language instructions take priority.
    if any(phrase in text for phrase in english_phrases):
        return "English"

    if any(phrase in text for phrase in arabic_phrases):
        return "Arabic"

    # Detect Arabic even if technical terms are English.
    if re.search(r"[\u0600-\u06FF]", question):
        return "Arabic"

    return "English"


def extract_retrieval_query(question: str) -> str:
    """
    Remove answer-format instructions before semantic retrieval.

    Examples:
        'overfitting اشرحلي في جملتين'
        -> 'overfitting'

        'Explain overfitting in two sentences'
        -> 'overfitting'

        'ما هو overfitting؟ اشرح بالعربي'
        -> 'overfitting'
    """

    query = question.strip()

    # Remove explicit language instructions.
    language_patterns = [
        r"\banswer\s+in\s+english(?:\s+only)?\b",
        r"\brespond\s+in\s+english(?:\s+only)?\b",
        r"\breply\s+in\s+english(?:\s+only)?\b",
        r"\benglish\s+only\b",
        r"\buse\s+english(?:\s+only)?\b",
        r"\banswer\s+in\s+arabic(?:\s+only)?\b",
        r"\brespond\s+in\s+arabic(?:\s+only)?\b",
        r"\breply\s+in\s+arabic(?:\s+only)?\b",
        r"\barabic\s+only\b",
        r"\buse\s+arabic(?:\s+only)?\b",
        r"بالعربي\s*فقط",
        r"باللغة\s+العربية",
        r"جاوب\s+بالعربي",
        r"اكتب\s+بالعربي",
        r"أجب\s+بالعربية",
    ]

    for pattern in language_patterns:
        query = re.sub(
            pattern,
            " ",
            query,
            flags=re.IGNORECASE,
        )

    # Remove sentence-count instructions.
    sentence_patterns = [
        r"\b(?:in|within|using)\s+\d+\s+sentences?\b",
        r"\bno\s+more\s+than\s+\d+\s+sentences?\b",
        r"\bexactly\s+\d+\s+sentences?\b",
        r"\bin\s+two\s+sentences?\b",
        r"\bin\s+three\s+sentences?\b",
        r"في\s+جملتين",
        r"ف\s+جملتين",
        r"في\s+ثلاث\s+جمل",
        r"ف\s+ثلاث\s+جمل",
        r"في\s+\d+\s+جمل",
        r"ف\s+\d+\s+جمل",
        r"في\s+\d+\s+جمل\s+فقط",
        r"في\s+\d+\s+سطور",
    ]

    for pattern in sentence_patterns:
        query = re.sub(
            pattern,
            " ",
            query,
            flags=re.IGNORECASE,
        )

    # Remove common explanation instructions.
    instruction_patterns = [
        r"\bplease\s+explain\b",
        r"\bexplain\b",
        r"\bdescribe\b",
        r"\bdefine\b",
        r"\bwhat\s+is\b",
        r"\bwhat\s+are\b",
        r"\btell\s+me\s+about\b",
        r"اشرحلي",
        r"اشرح\s+لي",
        r"اشرح",
        r"وضحلي",
        r"وضح\s+لي",
        r"وضح",
        r"عرفلي",
        r"عرف\s+لي",
        r"ما\s+هو",
        r"ما\s+هي",
        r"ايه\s+هو",
        r"ايه\s+هي",
        r"ما\s+معنى",
        r"يعني\s+ايه",
        r"بالتفصيل",
        r"باختصار",
    ]

    for pattern in instruction_patterns:
        query = re.sub(
            pattern,
            " ",
            query,
            flags=re.IGNORECASE,
        )

    # Remove punctuation and normalize whitespace.
    query = re.sub(
        r"[؟?!،,:;.!]",
        " ",
        query,
    )

    query = re.sub(
        r"\s+",
        " ",
        query,
    ).strip()

    # Avoid empty retrieval queries.
    if not query:
        return question.strip()

    return query


def detect_sentence_limit(question: str) -> int | None:
    """
    Detect requested sentence count in Arabic or English.
    """

    text = question.lower()

    # Arabic word-based counts.
    arabic_counts = {
        "جملة واحدة": 1,
        "جمله واحده": 1,
        "جملة": 1,
        "جملتين": 2,
        "ثلاث جمل": 3,
        "ثلاثة جمل": 3,
        "أربع جمل": 4,
        "اربعة جمل": 4,
        "خمس جمل": 5,
    }

    for phrase, count in arabic_counts.items():
        if phrase in text:
            return count

    # Arabic numeric sentence counts.
    match = re.search(
        r"(?:في|ف)\s*(\d+)\s*جمل",
        text,
    )

    if match:
        return int(match.group(1))

    # English numeric sentence counts.
    match = re.search(
        r"(?:in|within|exactly)\s+(\d+)\s+sentences?",
        text,
    )

    if match:
        return int(match.group(1))

    # English word-based counts.
    english_counts = {
        "in one sentence": 1,
        "in two sentences": 2,
        "in three sentences": 3,
        "in four sentences": 4,
        "in five sentences": 5,
    }

    for phrase, count in english_counts.items():
        if phrase in text:
            return count

    # No sentence limit requested.
    return None


def get_no_context_answer(language: str | None) -> str:
    """Return a no-context message in the requested language."""

    if language == "English":
        return (
            "The requested information was not found "
            "in the available project documents. "
            "Please upload a document containing this information."
        )

    if language == "Arabic":
        return (
            "المعلومة المطلوبة غير موجودة في المستندات "
            "المتاحة للمشروع. يُرجى رفع مستند يحتوي على هذه المعلومة."
        )

    return (
        "ملقتش المعلومة المطلوبة في المستندات المرفوعة للمشروع. "
        "ممكن تراجع الملف أو ترفع مستند يحتوي على المعلومة المطلوبة."
    )


class ChatService:
    """Manages conversations and RAG-grounded Q&A."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.chat_repo = ChatSessionRepository(session)
        self.message_repo = MessageRepository(session)
        self.generator = GroqGenerator()
        self.validator = EvidenceValidator()

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

        return [
            {
                "id": str(s.id),
                "title": s.title,
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

    async def ask(
        self,
        session_id: uuid.UUID,
        question: str,
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
            # 1. Validate question.
            if not question or not question.strip():
                yield {
                    "event": "error",
                    "data": "Question cannot be empty",
                }
                return

            question = question.strip()

            # 2. Detect language, topic query, and sentence limit.
            requested_language = detect_requested_language(question)

            retrieval_query = extract_retrieval_query(question)

            sentence_limit = detect_sentence_limit(question)

            logger.info("=" * 60)
            logger.info("RAG REQUEST STARTED")
            logger.info("SESSION ID: %s", session_id)
            logger.info("ORIGINAL QUESTION: %s", question)
            logger.info("RETRIEVAL QUERY: %s", retrieval_query)
            logger.info("REQUESTED LANGUAGE: %s", requested_language)
            logger.info("REQUESTED SENTENCE LIMIT: %s", sentence_limit)

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

            # 4. Save original user message.
            await self.message_repo.create(
                session_id=session_id,
                role="user",
                content=question,
            )

            # 5. Retrieve using the cleaned topic only.
            logger.info("Starting semantic retrieval...")

            retrieval_results = await Retriever.search(
                session=self.session,
                project_id=chat_session.project_id,
                query=retrieval_query,
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

            # 7. Build citations from actual retrieved results.
            citations = [
                {
                    "source_number": i + 1,
                    "document_name": result.document_name,
                    "page_number": result.page_number,
                    "similarity": result.similarity,
                    "snippet": (
                        result.content[:200] + "..."
                        if len(result.content) > 200
                        else result.content
                    ),
                }
                for i, result in enumerate(retrieval_results)
            ]

            # 8. Handle empty retrieval.
            if not retrieval_results:
                logger.warning(
                    "NO RELEVANT SOURCES FOUND: %s",
                    retrieval_query,
                )

                no_context_answer = get_no_context_answer(
                    requested_language
                )

                assistant_msg = await self.message_repo.create(
                    session_id=session_id,
                    role="assistant",
                    content=no_context_answer,
                    citations=[],
                )

                yield {
                    "event": "meta",
                    "data": {
                        "citations": [],
                        "total_sources": 0,
                    },
                }

                yield {
                    "event": "delta",
                    "data": no_context_answer,
                }

                yield {
                    "event": "done",
                    "data": {
                        "message_id": str(assistant_msg.id),
                    },
                }

                return

            # 9. Validate evidence against the cleaned topic.
            logger.info("Starting evidence validation...")

            validation_status = await self.validator.validate(
                question=retrieval_query,
                sources=retrieval_results,
            )

            logger.info(
                "EVIDENCE VALIDATION RESULT: %s",
                validation_status,
            )

            # 10. Reject unsupported answers.
            if validation_status == "NOT_SUPPORTED":
                no_context_answer = get_no_context_answer(
                    requested_language
                )

                assistant_msg = await self.message_repo.create(
                    session_id=session_id,
                    role="assistant",
                    content=no_context_answer,
                    citations=[],
                )

                yield {
                    "event": "meta",
                    "data": {
                        "citations": [],
                        "total_sources": 0,
                    },
                }

                yield {
                    "event": "delta",
                    "data": no_context_answer,
                }

                yield {
                    "event": "done",
                    "data": {
                        "message_id": str(assistant_msg.id),
                    },
                }

                return

            # 11. Build LLM messages using the original question.
            messages = build_messages(
                section=chat_session.section,
                question=question,
                retrieval_results=retrieval_results,
                chat_history=None,
            )

            # 12. Enforce the detected answer language.
            messages[0]["content"] += (
                f"\n\nIMPORTANT LANGUAGE REQUIREMENT: "
                f"Answer entirely in {requested_language}. "
                "This instruction has priority over the "
                "language of the question and source documents. "
                "Keep technical terms in English when appropriate."
            )

            # 13. Enforce requested sentence count.
            if sentence_limit is not None:
                messages[0]["content"] += (
                    f"\n\nIMPORTANT LENGTH REQUIREMENT: "
                    f"Answer in exactly {sentence_limit} sentences. "
                    "Do not add an introduction, greeting, "
                    "separate conclusion, or extra bullet points. "
                    "Keep the answer concise and evidence-based."
                )

            # 14. Handle partially supported questions.
            if validation_status == "PARTIALLY_SUPPORTED":
                messages[0]["content"] += (
                    "\n\nIMPORTANT: The evidence supports only "
                    "part of the question. Answer only the "
                    "supported parts. Clearly state which "
                    "requested information is not available "
                    "in the documents. Never fill gaps using "
                    "general knowledge."
                )

            # 15. Log exact messages sent to Groq.
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

            # 16. Generate answer.
            full_response = ""

            logger.info("Starting Groq streaming generation...")

            async for text_chunk in self.generator.generate_stream(
                messages
            ):
                if not text_chunk:
                    continue

                full_response += text_chunk

                yield {
                    "event": "delta",
                    "data": text_chunk,
                }

            # 17. Handle empty response.
            if not full_response.strip():
                logger.error("Groq returned an empty response")

                full_response = EMPTY_RESPONSE_ANSWER

                yield {
                    "event": "delta",
                    "data": full_response,
                }

            # 18. Save assistant response.
            assistant_msg = await self.message_repo.create(
                session_id=session_id,
                role="assistant",
                content=full_response,
                citations=citations,
            )

            # 19. Finish response.
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
