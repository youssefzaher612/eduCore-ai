"""
SQLAlchemy ORM models for EduCore.

All tables live in a single PostgreSQL database with pgvector
for embedding storage and similarity search.
"""

import enum
import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship

from educore.config import settings


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Base ────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── Enums ───────────────────────────────────────────────────────────

class UserRole(str, enum.Enum):
    STUDENT = "student"
    TEACHER = "teacher"


class DocumentStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


# ── Models ──────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    role = Column(
        SAEnum(UserRole, name="user_role", values_callable=lambda obj: [e.value for e in obj]),
        nullable=False,
        default=UserRole.STUDENT,
    )
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    projects = relationship("Project", back_populates="owner", lazy="selectin")
    chat_sessions = relationship("ChatSession", back_populates="user", lazy="selectin")

    def __repr__(self) -> str:
        return f"<User {self.name} ({self.role.value})>"


class Project(Base):
    __tablename__ = "projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    owner_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    owner = relationship("User", back_populates="projects")
    documents = relationship(
        "Document", back_populates="project", cascade="all, delete-orphan", lazy="selectin"
    )
    chat_sessions = relationship("ChatSession", back_populates="project", lazy="selectin")

    def __repr__(self) -> str:
        return f"<Project {self.name}>"


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    filename = Column(String(500), nullable=False)
    file_type = Column(String(50), nullable=False)  # pdf, txt, md, csv, xlsx
    file_size = Column(Integer, nullable=False)
    file_path = Column(String(1000), nullable=False)  # path in storage/
    status = Column(
        SAEnum(DocumentStatus, name="document_status", values_callable=lambda obj: [e.value for e in obj]),
        default=DocumentStatus.PENDING,
    )
    total_chunks = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    project = relationship("Project", back_populates="documents")
    chunks = relationship(
        "Chunk", back_populates="document", cascade="all, delete-orphan", lazy="selectin"
    )

    def __repr__(self) -> str:
        return f"<Document {self.filename} [{self.status.value}]>"


class Chunk(Base):
    """
    A text chunk from a document, with its vector embedding.

    The embedding column uses pgvector's Vector type for efficient
    cosine-similarity search via HNSW or IVFFlat indexes.
    """

    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    project_id = Column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    content = Column(Text, nullable=False)
    chunk_order = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=True)
    metadata_ = Column("metadata", JSON, nullable=True)
    embedding = Column(Vector(settings.EMBEDDING_DIMENSIONS), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    document = relationship("Document", back_populates="chunks")

    def __repr__(self) -> str:
        return f"<Chunk #{self.chunk_order} of doc={self.document_id}>"


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    project_id = Column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    section = Column(
        SAEnum(UserRole, name="user_role", create_type=False, values_callable=lambda obj: [e.value for e in obj]),
        nullable=False,
    )
    title = Column(String(255), default="محادثة جديدة")
    pending_external_question = Column(Text, nullable=True)
    pending_response_mode = Column(String(20), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    user = relationship("User", back_populates="chat_sessions")
    project = relationship("Project", back_populates="chat_sessions")
    messages = relationship(
        "Message",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<ChatSession {self.title} ({self.section.value})>"


class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    role = Column(String(20), nullable=False)  # "user" | "assistant"
    content = Column(Text, nullable=False)
    citations = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    session = relationship("ChatSession", back_populates="messages")

    def __repr__(self) -> str:
        return f"<Message {self.role} in session={self.session_id}>"


class StudentExam(Base):
    """A generated course exam and its server-side answer key."""

    __tablename__ = "student_exams"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    project_id = Column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    topic = Column(String(100), nullable=False)
    difficulty = Column(String(30), nullable=False)
    title = Column(String(255), nullable=False)
    questions = Column(JSON, nullable=False)
    citations = Column(JSON, nullable=False, default=list)
    status = Column(String(20), nullable=False, default="open")
    score_pct = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    submitted_at = Column(DateTime(timezone=True), nullable=True)

    attempts = relationship("StudentAttempt", back_populates="exam", lazy="selectin")


class StudentAttempt(Base):
    """A graded answer used for progress and BKT mastery calculations."""

    __tablename__ = "student_attempts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    project_id = Column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    exam_id = Column(
        UUID(as_uuid=True),
        ForeignKey("student_exams.id", ondelete="SET NULL"),
        nullable=True,
    )
    question_id = Column(String(100), nullable=False)
    topic = Column(String(100), nullable=False)
    difficulty = Column(String(30), nullable=False)
    selected_index = Column(Integer, nullable=True)
    is_correct = Column(Boolean, nullable=False)
    time_taken_sec = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    exam = relationship("StudentExam", back_populates="attempts")
