"""Database engines and session scopes. The only module in maverick/ that
talks to SQLAlchemy engine/pool/session plumbing.

Preserves the legacy `maverick_mcp/data/models.py` and
`maverick_mcp/data/session_management.py` semantics: SQLite always gets
NullPool (SQLite has no real connection pool to speak of and
``check_same_thread=False`` lets it work across the async loop/thread
boundaries used by the MCP server); Postgres gets a tuned QueuePool unless
pooling is explicitly disabled; schema creation is lazy, locked, and
memoized per engine; and session scopes commit on success, roll back on
exception, and always close in a ``finally``.
"""

import threading
import weakref
from collections.abc import AsyncGenerator, Callable, Generator
from contextlib import asynccontextmanager, contextmanager

from sqlalchemy import Engine, MetaData, create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool, QueuePool

from maverick.platform.config import DatabaseSettings

# Legacy default: seconds to wait for a new TCP connection before giving up.
_POSTGRES_CONNECT_TIMEOUT_SECONDS = 10


def _is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


def _is_postgres(url: str) -> bool:
    return url.startswith("postgresql")


def _async_url(url: str) -> str:
    """Rewrite a sync database URL to its async driver equivalent."""
    if url.startswith("sqlite://"):
        return url.replace("sqlite://", "sqlite+aiosqlite://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


def create_engine_from_settings(settings: DatabaseSettings) -> Engine:
    """Build a sync SQLAlchemy engine from platform database settings.

    SQLite always uses NullPool with ``check_same_thread=False`` regardless
    of ``use_pooling`` -- SQLite doesn't benefit from connection pooling and
    the legacy engine never pooled it either. Postgres uses a QueuePool
    tuned from ``settings`` unless ``use_pooling`` is False, in which case
    every backend falls back to NullPool.
    """
    url = settings.url

    if _is_sqlite(url):
        return create_engine(
            url,
            poolclass=NullPool,
            echo=settings.echo,
            connect_args={"check_same_thread": False},
        )

    if not settings.use_pooling:
        return create_engine(url, poolclass=NullPool, echo=settings.echo)

    connect_args: dict[str, object] = {}
    if _is_postgres(url):
        connect_args = {
            "connect_timeout": _POSTGRES_CONNECT_TIMEOUT_SECONDS,
            "options": f"-c statement_timeout={settings.statement_timeout_ms}",
        }

    return create_engine(
        url,
        poolclass=QueuePool,
        pool_size=settings.pool_size,
        max_overflow=settings.pool_max_overflow,
        pool_timeout=settings.pool_timeout,
        pool_recycle=settings.pool_recycle,
        pool_pre_ping=settings.pool_pre_ping,
        echo=settings.echo,
        connect_args=connect_args,
    )


def create_async_engine_from_settings(settings: DatabaseSettings) -> AsyncEngine:
    """Build an async SQLAlchemy engine from platform database settings.

    Mirrors :func:`create_engine_from_settings`, rewriting the URL to the
    async driver (``sqlite+aiosqlite`` / ``postgresql+asyncpg``) first.
    """
    url = settings.url
    async_url = _async_url(url)

    if _is_sqlite(url):
        return create_async_engine(
            async_url,
            poolclass=NullPool,
            echo=settings.echo,
            connect_args={"check_same_thread": False},
        )

    if not settings.use_pooling:
        return create_async_engine(async_url, poolclass=NullPool, echo=settings.echo)

    connect_args: dict[str, object] = {}
    if _is_postgres(url):
        connect_args = {
            "server_settings": {
                "statement_timeout": str(settings.statement_timeout_ms),
            }
        }

    return create_async_engine(
        async_url,
        pool_size=settings.pool_size,
        max_overflow=settings.pool_max_overflow,
        pool_timeout=settings.pool_timeout,
        pool_recycle=settings.pool_recycle,
        pool_pre_ping=settings.pool_pre_ping,
        echo=settings.echo,
        connect_args=connect_args,
    )


_schema_lock = threading.Lock()
# WeakKeyDictionary so memoization never keeps an otherwise-unreferenced
# engine (and its pool/connections) alive.
_schema_created: "weakref.WeakKeyDictionary[Engine, bool]" = weakref.WeakKeyDictionary()


def ensure_schema(engine: Engine, metadata: MetaData, *, force: bool = False) -> bool:
    """Ensure ``metadata``'s tables exist on ``engine``, lazily and once.

    Memoizes per engine so repeat calls after the first successful check are
    a dict lookup, not a schema inspection. Pass ``force=True`` to bypass
    the memoized fast path and re-run ``create_all`` unconditionally.

    Returns:
        ``True`` if table creation was executed, ``False`` if the schema
        was already known to be present.
    """
    if not force and _schema_created.get(engine):
        return False

    with _schema_lock:
        if not force and _schema_created.get(engine):
            return False

        try:
            inspector = inspect(engine)
            existing_tables = set(inspector.get_table_names())
        except SQLAlchemyError:
            existing_tables = set()

        defined_tables = set(metadata.tables.keys())
        missing_tables = defined_tables - existing_tables

        should_create = force or bool(missing_tables)
        if should_create:
            metadata.create_all(bind=engine)
            _schema_created[engine] = True
            return True

        _schema_created[engine] = True
        return False


@contextmanager
def session_scope(
    factory: Callable[[], Session],
) -> Generator[Session, None, None]:
    """Sync session scope: commit on success, rollback on exception, always close."""
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def read_only_session_scope(
    factory: Callable[[], Session],
) -> Generator[Session, None, None]:
    """Sync read-only session scope: never commits, rolls back on exception, always closes."""
    session = factory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@asynccontextmanager
async def async_session_scope(
    factory: Callable[[], AsyncSession],
) -> AsyncGenerator[AsyncSession, None]:
    """Async session scope: commit on success, rollback on exception, always close."""
    session = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
