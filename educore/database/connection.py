"""
Database connection management.

Provides both async (for FastAPI) and sync (for Celery workers) sessions
pointing at the same PostgreSQL + pgvector database.
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import sessionmaker, Session

from educore.config import settings

# ── Async engine (FastAPI) ──────────────────────────────────────────
async_engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# ── Sync engine (Celery workers) ────────────────────────────────────
sync_database_url = settings.DATABASE_URL_SYNC
# SQLAlchemy 2.1 defaults ``postgresql://`` to psycopg 3, while this project
# declares psycopg2-binary as its synchronous PostgreSQL driver.
if sync_database_url.startswith("postgresql://"):
    sync_database_url = sync_database_url.replace(
        "postgresql://", "postgresql+psycopg2://", 1
    )

sync_engine = create_engine(
    sync_database_url,
    echo=False,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)

SyncSessionLocal = sessionmaker(
    sync_engine,
    class_=Session,
    expire_on_commit=False,
)


async def get_async_session() -> AsyncSession:
    """FastAPI dependency — yields an async session with auto-commit/rollback."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def get_sync_session() -> Session:
    """Celery helper — returns a sync session (use as context manager)."""
    return SyncSessionLocal()
