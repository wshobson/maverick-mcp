"""Persistence and exclusive connection ownership for SQLite memory engines."""

import asyncio
import threading
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import event, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool, StaticPool

from maverick.platform.config import DatabaseSettings, PlatformSettings
from maverick.platform.db import (
    _AsyncSerializedStaticPool,
    _SerializedStaticPool,
    async_session_scope,
    create_async_engine_from_settings,
    create_engine_from_settings,
    ensure_schema,
    session_scope,
)
from maverick.portfolio import service as service_module
from maverick.portfolio.data import METADATA
from maverick.portfolio.service import PortfolioService


def test_ci_selected_memory_database_survives_sessions(monkeypatch):
    monkeypatch.setenv("CI", "true")
    engine = create_engine_from_settings(PlatformSettings().database)
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql("CREATE TABLE retained (id INTEGER)")
            connection.exec_driver_sql("INSERT INTO retained VALUES (1)")
        with engine.connect() as connection:
            assert connection.exec_driver_sql("SELECT * FROM retained").all() == [(1,)]
    finally:
        engine.dispose()


@pytest.mark.parametrize("use_pooling", [True, False])
@pytest.mark.parametrize(
    "url,memory",
    [
        ("sqlite:///:memory:", True),
        ("sqlite://", True),
        ("sqlite:///file:task11?mode=memory&cache=shared&uri=true", True),
        ("sqlite:///file::memory:?cache=shared&uri=true", True),
        ("sqlite:///ordinary.db", False),
        ("sqlite:///contains:memory:.db", False),
        ("sqlite:///ordinary.db?mode=memory", False),
    ],
)
def test_memory_pool_selection(url, memory, use_pooling):
    settings = DatabaseSettings(url=url, use_pooling=use_pooling)
    for engine in (
        create_engine_from_settings(settings),
        create_async_engine_from_settings(settings).sync_engine,
    ):
        assert isinstance(engine.pool, StaticPool if memory else NullPool)
        engine.dispose()


def test_memory_engines_are_isolated_and_disposal_resets_schema():
    settings = DatabaseSettings(url="sqlite://")
    first = create_engine_from_settings(settings)
    second = create_engine_from_settings(settings)
    try:
        ensure_schema(first, METADATA)
        with second.connect() as connection:
            assert (
                connection.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).all()
                == []
            )
        first.dispose()
        assert ensure_schema(first, METADATA) is True
        with first.connect() as connection:
            assert connection.exec_driver_sql("SELECT * FROM pf_portfolios").all() == []
    finally:
        first.dispose()
        second.dispose()


def test_memory_sync_foreign_keys_are_enforced():
    engine = create_engine_from_settings(DatabaseSettings(url="sqlite://"))
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql("CREATE TABLE parent (id INTEGER PRIMARY KEY)")
            connection.exec_driver_sql(
                "CREATE TABLE child (parent_id INTEGER REFERENCES parent(id))"
            )
        with pytest.raises(IntegrityError):
            with engine.begin() as connection:
                connection.exec_driver_sql("INSERT INTO child VALUES (999)")
    finally:
        engine.dispose()


async def test_memory_async_foreign_keys_and_engine_isolation():
    settings = DatabaseSettings(url="sqlite://")
    first = create_async_engine_from_settings(settings)
    second = create_async_engine_from_settings(settings)
    try:
        async with first.begin() as connection:
            await connection.exec_driver_sql(
                "CREATE TABLE parent (id INTEGER PRIMARY KEY)"
            )
            await connection.exec_driver_sql(
                "CREATE TABLE child (parent_id INTEGER REFERENCES parent(id))"
            )
        with pytest.raises(IntegrityError):
            async with first.begin() as connection:
                await connection.exec_driver_sql("INSERT INTO child VALUES (999)")
        async with second.connect() as connection:
            assert (
                await connection.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            ).all() == []
    finally:
        await first.dispose()
        await second.dispose()


async def test_memory_async_transactions_wait_and_cancel_safely():
    engine = create_async_engine_from_settings(DatabaseSettings(url="sqlite://"))
    factory = async_sessionmaker(engine)
    acquired = asyncio.Event()
    release = asyncio.Event()
    waiting = asyncio.Event()

    async def holder():
        async with async_session_scope(factory) as session:
            await session.execute(text("INSERT INTO retained VALUES (1)"))
            acquired.set()
            await release.wait()
            raise RuntimeError("roll back holder")

    async def follower():
        waiting.set()
        async with async_session_scope(factory) as session:
            await session.execute(text("INSERT INTO retained VALUES (2)"))

    async with engine.begin() as connection:
        await connection.exec_driver_sql("CREATE TABLE retained (id INTEGER)")
    first = asyncio.create_task(holder())
    second = None
    try:
        await asyncio.wait_for(acquired.wait(), 5)
        second = asyncio.create_task(follower())
        await waiting.wait()
        await asyncio.sleep(0.05)
        assert not second.done()
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        # Cancelling a queued borrower must not unlock the active transaction.
        waiting.clear()
        second = asyncio.create_task(follower())
        await waiting.wait()
        await asyncio.sleep(0.05)
        assert not second.done()
        release.set()
        with pytest.raises(RuntimeError, match="roll back holder"):
            await first
        await asyncio.wait_for(second, 5)
        async with engine.connect() as connection:
            assert (
                await connection.exec_driver_sql("SELECT * FROM retained")
            ).all() == [(2,)]
    finally:
        release.set()
        await asyncio.gather(
            first, *([second] if second else []), return_exceptions=True
        )
        await engine.dispose()


async def test_memory_service_worker_keeps_connection_until_cancelled_write_finishes(
    monkeypatch,
):
    engine = create_engine_from_settings(DatabaseSettings(url="sqlite://"))
    monkeypatch.setattr(
        service_module.service_risk, "resolve_sector", AsyncMock(return_value="Tech")
    )
    first, second = [PortfolioService(engine, AsyncMock()) for _ in range(2)]
    await first.add_position("u", "p", "AAPL", Decimal("10"), Decimal("100"))
    entered, waiting, release = (threading.Event() for _ in range(3))
    original_upsert = service_module.upsert_position
    assert isinstance(engine.pool, _SerializedStaticPool)
    original_acquire = engine.pool._acquire

    def held_upsert(session, portfolio_id, position):
        if position.shares == Decimal("11"):
            entered.set()
            assert release.wait(5)
        original_upsert(session, portfolio_id, position)

    def observed_acquire():
        if entered.is_set():
            waiting.set()
        original_acquire()

    monkeypatch.setattr(service_module, "upsert_position", held_upsert)
    monkeypatch.setattr(engine.pool, "_acquire", observed_acquire)
    task = asyncio.create_task(
        first.add_position("u", "p", "AAPL", Decimal("1"), Decimal("100"))
    )
    follower = None
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        follower = asyncio.create_task(
            second.add_position("u", "p", "AAPL", Decimal("2"), Decimal("100"))
        )
        assert await asyncio.to_thread(waiting.wait, 5)
        await asyncio.sleep(0.05)
        assert not follower.done()
    finally:
        release.set()
        if follower is not None:
            await follower
    [position] = await first._read_positions("u", "p")
    assert position.shares == Decimal("13")
    assert position.total_cost == Decimal("1300")
    engine.dispose()


async def test_memory_raw_reader_waits_for_rollback():
    engine = create_engine_from_settings(DatabaseSettings(url="sqlite://"))
    factory = sessionmaker(engine)
    with engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE retained (id INTEGER)")
    entered, release, waiting = (threading.Event() for _ in range(3))

    def writer():
        with pytest.raises(RuntimeError):
            with session_scope(factory) as session:
                session.execute(text("INSERT INTO retained VALUES (1)"))
                entered.set()
                assert release.wait(5)
                raise RuntimeError("rollback")

    def reader():
        waiting.set()
        with engine.connect() as connection:
            return connection.exec_driver_sql("SELECT * FROM retained").all()

    task = asyncio.create_task(asyncio.to_thread(writer))
    follower = None
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        follower = asyncio.create_task(asyncio.to_thread(reader))
        assert await asyncio.to_thread(waiting.wait, 5)
        await asyncio.sleep(0.05)
        assert not follower.done()
    finally:
        release.set()
        await task
        if follower is not None:
            assert await follower == []
        engine.dispose()


def test_memory_failed_initial_connect_does_not_leak_checkout():
    engine = create_engine_from_settings(DatabaseSettings(url="sqlite://"))

    def fail(*args):
        raise RuntimeError("connect failed")

    event.listen(engine, "connect", fail)
    try:
        with pytest.raises(RuntimeError, match="connect failed"):
            engine.connect()
        event.remove(engine, "connect", fail)
        with engine.connect() as connection:
            assert connection.exec_driver_sql("SELECT 1").scalar_one() == 1
    finally:
        engine.dispose()


async def test_memory_async_cancelled_holder_rolls_back_before_follower():
    engine = create_async_engine_from_settings(DatabaseSettings(url="sqlite://"))
    factory = async_sessionmaker(engine)
    entered = asyncio.Event()
    async with engine.begin() as connection:
        await connection.exec_driver_sql("CREATE TABLE retained (id INTEGER)")

    async def holder():
        async with async_session_scope(factory) as session:
            await session.execute(text("INSERT INTO retained VALUES (1)"))
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(holder())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        async with async_session_scope(factory) as session:
            await session.execute(text("INSERT INTO retained VALUES (2)"))
        async with engine.connect() as connection:
            assert (
                await connection.exec_driver_sql("SELECT * FROM retained")
            ).all() == [(2,)]
    finally:
        await engine.dispose()


@pytest.mark.parametrize(
    "url,memory",
    [
        ("sqlite:///file:task11-review?mode=memory&cache=shared&uri=1", True),
        ("sqlite:///file::memory:?cache=shared&uri=yes", True),
        ("sqlite:///file:%3Amemory%3A?uri=on", True),
        ("sqlite:///ordinary.db?mode=memory&uri=true", False),
        ("sqlite:///file::memory:ordinary.db?uri=true", False),
    ],
)
async def test_memory_uri_matches_sqlite_filename_rules(url, memory):
    settings = DatabaseSettings(url=url)
    sync = create_engine_from_settings(settings)
    async_engine = create_async_engine_from_settings(settings)
    try:
        for engine in (sync, async_engine.sync_engine):
            assert isinstance(engine.pool, StaticPool if memory else NullPool)
        if memory:
            with sync.begin() as connection:
                connection.exec_driver_sql("CREATE TABLE retained (id INTEGER)")
                connection.exec_driver_sql("INSERT INTO retained VALUES (1)")
            with sync.connect() as connection:
                assert connection.exec_driver_sql("SELECT * FROM retained").all() == [
                    (1,)
                ]
            # Dispose first: named shared-memory URI may share SQLite state.
            sync.dispose()
            async with async_engine.begin() as connection:
                await connection.exec_driver_sql("CREATE TABLE retained (id INTEGER)")
                await connection.exec_driver_sql("INSERT INTO retained VALUES (1)")
            async with async_engine.connect() as connection:
                assert (
                    await connection.exec_driver_sql("SELECT * FROM retained")
                ).all() == [(1,)]
    finally:
        sync.dispose()
        await async_engine.dispose()


@pytest.mark.parametrize("cancel_cleanup", [False, True])
async def test_memory_async_cancelled_reset_releases_only_after_termination(
    monkeypatch, cancel_cleanup
):
    engine = create_async_engine_from_settings(DatabaseSettings(url="sqlite://"))
    connection = await engine.connect()
    driver = (await connection.get_raw_connection()).driver_connection
    rollback_entered, close_entered, release_close = (asyncio.Event() for _ in range(3))
    assert driver is not None
    original_close = driver.close

    async def held_rollback():
        rollback_entered.set()
        await asyncio.Event().wait()

    async def held_close():
        close_entered.set()
        await release_close.wait()
        await original_close()

    monkeypatch.setattr(driver, "rollback", held_rollback)
    monkeypatch.setattr(driver, "close", held_close)
    closing = asyncio.create_task(connection.close())
    follower = None
    try:
        await asyncio.wait_for(rollback_entered.wait(), 5)
        closing.cancel()
        await asyncio.wait_for(close_entered.wait(), 5)
        follower = asyncio.create_task(engine.connect().start())
        await asyncio.sleep(0.05)
        assert not follower.done(), "released ownership while close still ran"
        if cancel_cleanup:
            closing.cancel()
            await asyncio.sleep(0.05)
            assert not closing.done()
            assert not follower.done()
        release_close.set()
        with pytest.raises(asyncio.CancelledError):
            await closing
        replacement = await asyncio.wait_for(follower, 0.5)
        assert (await replacement.get_raw_connection()).driver_connection is not driver
        # A stale close/checkin must not unlock the replacement borrower's lock.
        await connection.close()
        assert isinstance(engine.pool, _AsyncSerializedStaticPool)
        assert engine.pool._async_checkout_lock.locked()
        await replacement.close()
    finally:
        release_close.set()
        await asyncio.gather(closing, return_exceptions=True)
        if follower is not None and not follower.done():
            follower.cancel()
        if follower is not None:
            await asyncio.gather(follower, return_exceptions=True)
        await connection.close()
        await engine.dispose()
