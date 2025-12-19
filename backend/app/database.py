"""Database connection and session management."""

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


# Async engine for FastAPI endpoints
engine = create_async_engine(
    settings.database_url,
    echo=settings.database_echo,
    future=True,
)

# Sync engine for Celery tasks (convert async URL to sync)
# postgresql+asyncpg:// -> postgresql://, sqlite+aiosqlite:// -> sqlite://
_sync_url = settings.database_url.replace("+asyncpg", "").replace("+aiosqlite", "")
sync_engine = create_engine(
    _sync_url,
    echo=settings.database_echo,
    future=True,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncSession:
    """Dependency that provides a database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        else:
            await session.commit()
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
