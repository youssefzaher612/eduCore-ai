"""Remember pending permission before external answers.

Revision ID: 003_chat_external_consent
Revises: 002_student_learning
"""

from alembic import op
import sqlalchemy as sa


revision = "003_chat_external_consent"
down_revision = "002_student_learning"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chat_sessions", sa.Column("pending_external_question", sa.Text(), nullable=True))
    op.add_column("chat_sessions", sa.Column("pending_response_mode", sa.String(20), nullable=True))


def downgrade() -> None:
    op.drop_column("chat_sessions", "pending_response_mode")
    op.drop_column("chat_sessions", "pending_external_question")
