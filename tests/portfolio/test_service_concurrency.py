"""Database-backed serialization across independent portfolio service instances."""

import asyncio
import multiprocessing
import threading
import time
import uuid
from contextlib import contextmanager
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select

from maverick.platform.config import DatabaseSettings
from maverick.platform.db import create_engine_from_settings
from maverick.portfolio import service as service_module
from maverick.portfolio.data import PF_PORTFOLIOS
from maverick.portfolio.service import PortfolioService


@pytest.fixture(
    params=["sqlite", pytest.param("postgresql", marks=pytest.mark.integration)]
)
async def services(tmp_path, monkeypatch, request):
    url = f"sqlite:///{tmp_path}/concurrent.db"
    admin = None
    if request.param == "postgresql":
        admin_url = request.getfixturevalue("portfolio_postgres_url")
        admin = create_engine_from_settings(DatabaseSettings(url=admin_url))
        database = "portfolio_" + uuid.uuid4().hex
        with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.exec_driver_sql(f'CREATE DATABASE "{database}"')
        url = admin_url.rsplit("/", 1)[0] + "/" + database
    engines = [create_engine_from_settings(DatabaseSettings(url=url)) for _ in range(2)]
    monkeypatch.setattr(
        service_module.service_risk, "resolve_sector", AsyncMock(return_value="Tech")
    )
    pair = [PortfolioService(engine, AsyncMock()) for engine in engines]
    for service in pair:
        await service._ensure_schema()
    yield pair
    for engine in engines:
        engine.dispose()
    if admin is not None:
        with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.exec_driver_sql(f'DROP DATABASE "{database}"')
        admin.dispose()


def coordinate_writers(monkeypatch):
    """Rendezvous before entering transactions; never wait for a locked peer."""
    barrier = threading.Barrier(2, timeout=5)
    original_scope = service_module.session_scope
    original_add = service_module.add_shares

    @contextmanager
    def coordinated_scope(*args, **kwargs):
        barrier.wait()
        with original_scope(*args, **kwargs) as session:
            yield session

    def slow_add(*args, **kwargs):
        # Widen the original lost-update window without a barrier inside the
        # transaction, which would deadlock a correctly serialized writer.
        time.sleep(0.1)
        return original_add(*args, **kwargs)

    monkeypatch.setattr(service_module, "session_scope", coordinated_scope)
    monkeypatch.setattr(service_module, "add_shares", slow_add)


async def test_concurrent_adds_preserve_every_acknowledged_share(services, monkeypatch):
    first, second = services
    await first.add_position("u", "p", "AAPL", Decimal("10"), Decimal("100"))
    coordinate_writers(monkeypatch)
    results = await asyncio.gather(
        first.add_position("u", "p", "AAPL", Decimal("1"), Decimal("100")),
        second.add_position("u", "p", "AAPL", Decimal("2"), Decimal("100")),
    )
    assert len(results) == 2
    [final] = await first._read_positions("u", "p")
    assert final.shares == Decimal("13")
    assert final.total_cost == Decimal("1300")


async def test_concurrent_first_creation_preserves_both_adds(services, monkeypatch):
    first, second = services
    coordinate_writers(monkeypatch)
    await asyncio.gather(
        first.add_position("u", "p", "AAPL", Decimal("1"), Decimal("100")),
        second.add_position("u", "p", "AAPL", Decimal("2"), Decimal("100")),
    )
    [final] = await first._read_positions("u", "p")
    assert final.shares == Decimal("3")
    with first._engine.connect() as connection:
        assert len(connection.execute(select(PF_PORTFOLIOS)).all()) == 1


async def test_concurrent_add_and_remove_preserve_both_mutations(services, monkeypatch):
    first, second = services
    await first.add_position("u", "p", "AAPL", Decimal("10"), Decimal("100"))
    coordinate_writers(monkeypatch)
    await asyncio.gather(
        first.add_position("u", "p", "AAPL", Decimal("2"), Decimal("100")),
        second.remove_position("u", "p", "AAPL", Decimal("3")),
    )
    [final] = await first._read_positions("u", "p")
    assert final.shares == Decimal("9")
    assert final.total_cost == Decimal("900")


async def test_concurrent_clear_and_add_match_transaction_order(services, monkeypatch):
    first, second = services
    await first.add_position("u", "p", "AAPL", Decimal("10"), Decimal("100"))
    coordinate_writers(monkeypatch)
    updated, cleared = await asyncio.gather(
        first.add_position("u", "p", "AAPL", Decimal("2"), Decimal("100")),
        second.clear_portfolio("u", "p"),
    )
    positions = await first._read_positions("u", "p")
    assert cleared == 1
    if updated.shares == Decimal("12"):
        assert positions == []  # add committed before clear
    else:
        assert updated.shares == Decimal("2")  # clear committed before add
        assert positions == [updated]


async def test_failed_write_rolls_back_new_portfolio_and_position(
    services, monkeypatch
):
    first, second = services
    original = service_module.upsert_position

    def fail_after_write(*args):
        original(*args)
        raise RuntimeError("injected transaction failure")

    with monkeypatch.context() as patch:
        patch.setattr(service_module, "upsert_position", fail_after_write)
        with pytest.raises(RuntimeError, match="injected transaction failure"):
            await first.add_position("u", "p", "AAPL", Decimal("2"), Decimal("100"))
    with first._engine.connect() as connection:
        assert connection.execute(select(PF_PORTFOLIOS)).all() == []
    added = await second.add_position("u", "p", "AAPL", Decimal("3"), Decimal("100"))
    assert added.shares == Decimal("3")


async def test_reads_of_missing_portfolio_do_not_create_it(services):
    first, _ = services
    assert await first._read_positions("u", "missing") == []
    with first._engine.connect() as connection:
        assert connection.execute(select(PF_PORTFOLIOS)).all() == []


async def test_cancelled_writer_keeps_database_lock_until_worker_finishes(
    services, monkeypatch
):
    first, second = services
    await first.add_position("u", "p", "AAPL", Decimal("10"), Decimal("100"))
    entered = threading.Event()
    follower_entering = threading.Event()
    release = threading.Event()
    original = service_module.upsert_position

    def hold_first(session, portfolio_id, position):
        if position.shares == Decimal("11"):
            entered.set()
            assert release.wait(5), "test did not release first writer"
        original(session, portfolio_id, position)

    original_scope = service_module.session_scope

    @contextmanager
    def observe_entry(*args, **kwargs):
        if entered.is_set():
            follower_entering.set()
        with original_scope(*args, **kwargs) as session:
            yield session

    monkeypatch.setattr(service_module, "session_scope", observe_entry)
    monkeypatch.setattr(service_module, "upsert_position", hold_first)
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
        assert await asyncio.to_thread(follower_entering.wait, 5)
        await asyncio.sleep(0.1)
        assert not follower.done()
    finally:
        release.set()
        if follower is not None:
            await follower
    [final] = await first._read_positions("u", "p")
    assert final.shares == Decimal("13")


def _process_add(url, shares, ready, results):
    """Each process has its own engine, session factory and async event loop."""
    engine = create_engine_from_settings(DatabaseSettings(url=url))
    service_module.service_risk.resolve_sector = AsyncMock(return_value="Tech")
    original_add = service_module.add_shares

    def slow_add(*args, **kwargs):
        time.sleep(0.1)
        return original_add(*args, **kwargs)

    async def run():
        service = PortfolioService(engine, AsyncMock())
        await service._ensure_schema()
        ready.wait(timeout=15)
        await service.add_position("u", "p", "AAPL", Decimal(shares), Decimal("100"))

    try:
        with patch.object(service_module, "add_shares", slow_add):
            asyncio.run(run())
        results.put("ok")
    except Exception as error:
        results.put(repr(error))
    finally:
        engine.dispose()


async def test_concurrent_processes_preserve_every_acknowledged_share(services):
    first, _ = services
    await first.add_position("u", "p", "AAPL", Decimal("10"), Decimal("100"))
    context = multiprocessing.get_context("spawn")
    ready = context.Barrier(3)
    results = context.Queue()
    url = first._engine.url.render_as_string(hide_password=False)
    processes = [
        context.Process(target=_process_add, args=(url, shares, ready, results))
        for shares in ("1", "2")
    ]
    try:
        for process in processes:
            process.start()
        await asyncio.to_thread(ready.wait, 15)
        for _ in processes:
            assert await asyncio.to_thread(results.get, True, 15) == "ok"
        for process in processes:
            await asyncio.to_thread(process.join, 15)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        results.close()
        results.join_thread()
    [final] = await first._read_positions("u", "p")
    assert final.shares == Decimal("13")
    assert final.total_cost == Decimal("1300")
