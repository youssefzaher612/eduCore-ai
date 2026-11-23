"""Student AI learning records and generated exams.

Revision ID: 002_student_learning
Revises: 001_initial_schema
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON, UUID


revision = "002_student_learning"
down_revision = "001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "student_exams",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "project_id",
            UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("topic", sa.String(100), nullable=False),
        sa.Column("difficulty", sa.String(30), nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("questions", JSON, nullable=False),
        sa.Column("citations", JSON, nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="open"),
        sa.Column("score_pct", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("idx_student_exams_user_project", "student_exams", ["user_id", "project_id"])

    op.create_table(
        "student_attempts",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "project_id",
            UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "exam_id",
            UUID(as_uuid=True),
            sa.ForeignKey("student_exams.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("question_id", sa.String(100), nullable=False),
        sa.Column("topic", sa.String(100), nullable=False),
        sa.Column("difficulty", sa.String(30), nullable=False),
        sa.Column("selected_index", sa.Integer(), nullable=True),
        sa.Column("is_correct", sa.Boolean(), nullable=False),
        sa.Column("time_taken_sec", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_student_attempts_user_project", "student_attempts", ["user_id", "project_id"])
    op.create_index("idx_student_attempts_exam", "student_attempts", ["exam_id"])


def downgrade() -> None:
    op.drop_index("idx_student_attempts_exam", table_name="student_attempts")
    op.drop_index("idx_student_attempts_user_project", table_name="student_attempts")
    op.drop_table("student_attempts")
    op.drop_index("idx_student_exams_user_project", table_name="student_exams")
    op.drop_table("student_exams")
