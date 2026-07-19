"""Tests for maverick.platform.db."""

import gc

import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table, insert, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool, QueuePool

from maverick.platform import db as db_module
from maverick.platform.config import DatabaseSettings
from maverick.platform.db import (
    create_async_engine_from_settings,
    create_engine_from_settings,
    ensure_schema,
    read_only_session_scope,
    session_scope,
)

METADATA = MetaData()
ITEMS = Table(
    "items",
    METADATA,
    Column("id", Integer, primary_key=True),
    Column("name", String(50)),
)


def _settings(tmp_path, **overrides) -> DatabaseSettings:
    base = {"url": f"sqlite:///{tmp_path}/t.db", "use_pooling": True}
    base.update(overrides)
    return DatabaseSettings(**base)


def test_sqlite_engine_uses_nullpool_even_when_pooling_enabled(tmp_path):
    engine = create_engine_from_settings(_settings(tmp_path))
    assert isinstance(engine.pool, NullPool)


def test_postgres_engine_would_use_queuepool():
    settings = DatabaseSettings(url="postgresql://u:p@localhost/x", use_pooling=True)
    engine = create_engine_from_settings(settings)
    assert isinstance(engine.pool, QueuePool)
    assert engine.pool.size() == settings.pool_size


def test_ensure_schema_is_lazy_and_idempotent(tmp_path):
    engine = create_engine_from_settings(_settings(tmp_path))
    assert ensure_schema(engine, METADATA) is True
    assert ensure_schema(engine, METADATA) is False
    assert ensure_schema(engine, METADATA, force=True) is True


def test_session_scope_commits_on_success(tmp_path):
    engine = create_engine_from_settings(_settings(tmp_path))
    ensure_schema(engine, METADATA)
    factory = sessionmaker(bind=engine)
    with session_scope(factory) as session:
        session.execute(insert(ITEMS).values(name="kept"))
    with read_only_session_scope(factory) as session:
        assert session.execute(select(ITEMS.c.name)).scalar_one() == "kept"


def test_session_scope_rolls_back_on_error(tmp_path):
    engine = create_engine_from_settings(_settings(tmp_path))
    ensure_schema(engine, METADATA)
    factory = sessionmaker(bind=engine)
    with pytest.raises(RuntimeError):
        with session_scope(factory) as session:
            session.execute(insert(ITEMS).values(name="lost"))
            raise RuntimeError("abort")
    with read_only_session_scope(factory) as session:
        assert session.execute(select(ITEMS.c.name)).first() is None


def test_async_engine_url_rewrite(tmp_path):
    engine = create_async_engine_from_settings(_settings(tmp_path))
    assert engine.url.drivername == "sqlite+aiosqlite"


def test_ensure_schema_memoization_does_not_leak_engines(tmp_path):
    """Regression test for the schema-memoization leak: `_schema_created`
    must not hold a strong reference to engines, or every engine ever
    passed to `ensure_schema` would live for the lifetime of the process.

    Accesses `maverick.platform.db._schema_created` directly -- it's a
    private, module-internal mapping -- to confirm it drops its entry once
    nothing else references the engine.
    """
    # Earlier tests' engines may be reference cycles awaiting a GC pass
    # rather than already-freed objects; collect first for a clean baseline.
    gc.collect()

    engine = create_engine_from_settings(_settings(tmp_path))
    ensure_schema(engine, METADATA)
    assert len(db_module._schema_created) == 1

    engine.dispose()
    del engine
    gc.collect()

    assert len(db_module._schema_created) == 0
