"""Database engines and session scopes. The only module in maverick/ that
talks to SQLAlchemy engine/pool/session plumbing.

Preserves the legacy `maverick_mcp/data/models.py` and
`maverick_mcp/data/session_management.py` semantics: file SQLite gets
NullPool; memory SQLite retains one connection and serializes its checkouts,
including schema operations and reads.
``check_same_thread=False`` supports the server's worker threads.
Postgres gets a tuned QueuePool unless
pooling is explicitly disabled; schema creation is lazy, locked, and
memoized per engine; and session scopes commit on success, roll back on
exception, and always close in a ``finally``.

Policy: SQLite foreign-key enforcement is turned on for every engine this
module creates (``PRAGMA foreign_keys=ON`` per connection -- SQLite parses
``ForeignKey(..., ondelete=...)`` but ignores it by default). The listener
is registered per-engine, scoped only to engines built by
``create_engine_from_settings``/``create_async_engine_from_settings`` --
never at the ``Engine`` class level -- so it has no effect on engines built
elsewhere (e.g. the legacy ``maverick_mcp`` engines, whose FK write paths
this policy has not audited).
"""

import asyncio
import threading
import weakref
from collections.abc import AsyncGenerator, Callable, Generator
from contextlib import asynccontextmanager, contextmanager
from typing import Any
from urllib.parse import unquote, urlsplit

from sqlalchemy import Engine, MetaData, create_engine, event, inspect
from sqlalchemy.engine import make_url
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import ConnectionPoolEntry, NullPool, QueuePool, StaticPool
from sqlalchemy.schema import CreateColumn
from sqlalchemy.util import asbool
from sqlalchemy.util.concurrency import await_only, greenlet_spawn, in_greenlet

from maverick.platform.config import DatabaseSettings
from maverick.platform.telemetry import get_logger

logger = get_logger(__name__)

# Legacy default: seconds to wait for a new TCP connection before giving up.
_POSTGRES_CONNECT_TIMEOUT_SECONDS = 10


class _SerializedStaticPool(StaticPool):
    """Retain a memory database with exclusive checkout through rollback/return.

    Unlike ordinary StaticPool, never lend the same DBAPI connection to two
    transactions. Ownership belongs to the worker/connection, not its caller.
    As with other single-connection pools, do not nest independent checkouts
    or dispose the engine while it is in use.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._checkout_lock = threading.Lock()
        self._checked_out: ConnectionPoolEntry | None = None

    def _acquire(self) -> None:
        self._checkout_lock.acquire()

    def _release(self) -> None:
        self._checkout_lock.release()

    def _do_get(self) -> ConnectionPoolEntry:
        self._acquire()
        try:
            record = super()._do_get()
            self._checked_out = record
            return record
        except BaseException:
            self._release()
            raise

    def _do_return_conn(self, record: ConnectionPoolEntry) -> None:
        if self._checked_out is record:
            self._checked_out = None
            self._release()

    def _close_connection(
        self, connection: DBAPIConnection, *, terminate: bool = False
    ) -> None:
        record = self._checked_out
        try:
            super()._close_connection(connection, terminate=terminate)
        finally:
            if record is not None and record.dbapi_connection is connection:
                # Cancellation during reset can skip SQLAlchemy's checkin.
                # Retire this record so a later stale checkin cannot release
                # the lock belonging to a replacement connection.
                record.dbapi_connection = None
                if self.__dict__.get("connection") is record:
                    del self.__dict__["connection"]
                self._do_return_conn(record)


class _AsyncSerializedStaticPool(_SerializedStaticPool):
    """Await checkout without blocking the async engine's event loop."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._async_checkout_lock = asyncio.Lock()

    def _acquire(self) -> None:
        await_only(self._async_checkout_lock.acquire())

    def _release(self) -> None:
        self._async_checkout_lock.release()

    def _close_connection(
        self, connection: DBAPIConnection, *, terminate: bool = False
    ) -> None:
        if not in_greenlet():  # SQLAlchemy's GC termination path cannot await.
            super()._close_connection(connection, terminate=terminate)
            return
        close = super()._close_connection

        async def finish_close() -> None:
            task = asyncio.create_task(
                greenlet_spawn(close, connection, terminate=terminate)
            )
            cancelled = False
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    cancelled = True
            task.result()
            if cancelled:
                raise asyncio.CancelledError

        # Keep exclusive ownership until driver termination actually finishes,
        # even if the caller is cancelled again while cleanup is in progress.
        await_only(finish_close())


def _is_sqlite(url: str) -> bool:
    return make_url(url).get_backend_name() == "sqlite"


def _is_memory_sqlite(url: str) -> bool:
    parsed = make_url(url)
    if parsed.get_backend_name() != "sqlite":
        return False
    if parsed.database in (None, "", ":memory:"):
        return True
    return (
        asbool(parsed.query.get("uri", False))
        and parsed.database.startswith("file:")
        and (
            parsed.query.get("mode") == "memory"
            or unquote(urlsplit(parsed.database).path) == ":memory:"
        )
    )


def _forget_memory_schema(engine: Engine) -> None:
    # Disposing the sole connection also destroys its database.
    _schema_created.pop(engine, None)


def _is_postgres(url: str) -> bool:
    return url.startswith("postgresql")


def _async_url(url: str) -> str:
    """Rewrite a sync database URL to its async driver equivalent."""
    if url.startswith("sqlite://"):
        return url.replace("sqlite://", "sqlite+aiosqlite://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


def _enable_sqlite_foreign_keys(dbapi_connection, connection_record) -> None:  # noqa: ANN001
    """``connect``-event handler: turn on FK enforcement for a SQLite connection.

    Registered per-engine (via ``event.listen(engine, "connect", ...)``) on
    engines this module builds, never at the ``Engine`` class level -- so it
    never touches engines constructed outside this module.
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def create_engine_from_settings(settings: DatabaseSettings) -> Engine:
    """Build a sync SQLAlchemy engine from platform database settings.

    SQLite uses NullPool for files and a serialized StaticPool for memory,
    regardless of ``use_pooling``, with ``check_same_thread=False``.
    Postgres uses a QueuePool tuned from ``settings`` unless ``use_pooling``
    is False. Non-SQLite backends use NullPool when pooling is disabled.
    """
    url = settings.url

    if _is_sqlite(url):
        engine = create_engine(
            url,
            poolclass=_SerializedStaticPool if _is_memory_sqlite(url) else NullPool,
            echo=settings.echo,
            connect_args={"check_same_thread": False},
        )
        event.listen(engine, "connect", _enable_sqlite_foreign_keys)
        if _is_memory_sqlite(url):
            event.listen(engine, "engine_disposed", _forget_memory_schema)
        return engine

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
        engine = create_async_engine(
            async_url,
            poolclass=_AsyncSerializedStaticPool
            if _is_memory_sqlite(url)
            else NullPool,
            echo=settings.echo,
            connect_args={"check_same_thread": False},
        )
        event.listen(engine.sync_engine, "connect", _enable_sqlite_foreign_keys)
        if _is_memory_sqlite(url):
            event.listen(engine.sync_engine, "engine_disposed", _forget_memory_schema)
        return engine

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
# engine (and its pool/connections) alive. Keyed by engine, valued by the
# set of table names already confirmed present on it -- not a bare bool --
# so that a second, different `MetaData` (e.g. another domain sharing the
# same physical engine) still gets its own tables created instead of being
# short-circuited by an earlier, unrelated metadata's success.
_schema_created: "weakref.WeakKeyDictionary[Engine, set[str]]" = (
    weakref.WeakKeyDictionary()
)


def ensure_schema(engine: Engine, metadata: MetaData, *, force: bool = False) -> bool:
    """Ensure ``metadata``'s tables and nullable columns exist, lazily and once.

    Memoizes per ``(engine, table name)`` so repeat calls for a metadata
    whose tables are already known-present are a set-membership check, not a
    schema inspection -- while a *different* metadata sharing the same
    engine (multiple domains against one physical DB) still gets its own
    tables created on its first call. Pass ``force=True`` to bypass the
    memoized fast path and re-run ``create_all`` unconditionally.

    Existing tables gain plain nullable columns defined by ``metadata``
    through portable SQLAlchemy column compilation. Missing non-nullable or
    constrained columns are skipped with a warning because they require a
    data-migration policy. Each ALTER is isolated; a concurrent migration
    race logs a warning without preventing startup or later column attempts.

    Returns:
        ``True`` if schema DDL was executed, ``False`` if the schema was
        already known to be present.
    """
    defined_tables = set(metadata.tables.keys())
    known_tables = _schema_created.get(engine, set())
    if not force and defined_tables <= known_tables:
        return False

    with _schema_lock:
        known_tables = _schema_created.get(engine, set())
        if not force and defined_tables <= known_tables:
            return False

        try:
            inspector = inspect(engine)
            existing_tables = set(inspector.get_table_names())
            existing_columns = {
                table_name: {
                    column["name"] for column in inspector.get_columns(table_name)
                }
                for table_name in defined_tables & existing_tables
            }
        except SQLAlchemyError:
            existing_tables = set()
            existing_columns = {}

        missing_tables = defined_tables - existing_tables
        missing_columns = [
            (metadata.tables[table_name], column)
            for table_name, column_names in existing_columns.items()
            for column in metadata.tables[table_name].columns
            if column.name not in column_names
        ]
        columns_to_add = []
        for table, column in missing_columns:
            if (
                not column.nullable
                or column.foreign_keys
                or column.unique
                or column.index
                or column.primary_key
            ):
                logger.warning(
                    "database: skipping automatic add for missing column %s.%s; "
                    "column is non-nullable or constrained",
                    table.fullname,
                    column.name,
                )
                continue
            columns_to_add.append((table, column))

        should_create = force or bool(missing_tables)
        if should_create:
            metadata.create_all(bind=engine)

        columns_added = False
        if columns_to_add:
            preparer = engine.dialect.identifier_preparer
            for table, column in columns_to_add:
                try:
                    table_ddl = preparer.format_table(table)
                    column_ddl = str(
                        CreateColumn(column).compile(dialect=engine.dialect)
                    )
                    with engine.begin() as connection:
                        connection.exec_driver_sql(
                            f"ALTER TABLE {table_ddl} ADD COLUMN {column_ddl}"
                        )
                except SQLAlchemyError:
                    logger.warning(
                        "database: failed to add missing column %s.%s",
                        table.fullname,
                        column.name,
                        exc_info=True,
                    )
                    continue
                columns_added = True

        _schema_created[engine] = known_tables | defined_tables
        return should_create or columns_added


@contextmanager
def session_scope(
    factory: Callable[[], Session],
    *,
    sqlite_immediate: bool = False,
) -> Generator[Session, None, None]:
    """Commit on success, roll back on error, and always close.

    ``sqlite_immediate`` reserves the SQLite writer before any reads, so a
    read/modify/write transaction cannot overwrite another writer's result.
    Other backends use their normal transaction and domain-level row locks.
    """
    session = factory()
    try:
        if sqlite_immediate and session.get_bind().dialect.name == "sqlite":
            session.connection().exec_driver_sql("BEGIN IMMEDIATE")
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
