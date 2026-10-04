"""Real database races and additive legacy-schema migration for price history."""

import asyncio
import threading
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import (
    Column,
    Integer,
    MetaData,
    String,
    Table,
    event,
    insert,
    inspect,
    select,
    text,
)
from sqlalchemy.orm import sessionmaker

from maverick.market_data import data as data_module
from maverick.market_data.data import (
    MD_PRICE_BARS,
    MD_STOCKS,
    METADATA,
    get_or_create_stock,
    write_price_bars,
)
from maverick.platform.config import DatabaseSettings
from maverick.platform.db import (
    create_engine_from_settings,
    ensure_schema,
    session_scope,
)
from tests.market_data.test_history_refresh import DAYS, bars, freshness, service
from tests.portfolio.conftest import (
    portfolio_postgres_url as history_postgres_url,  # noqa: F401
)


@pytest.fixture(
    params=["sqlite", pytest.param("postgresql", marks=pytest.mark.integration)]
)
def engines(tmp_path, request):
    url = f"sqlite:///{tmp_path}/concurrent.db"
    admin = None
    if request.param == "postgresql":
        admin_url = request.getfixturevalue("history_postgres_url")
        admin = create_engine_from_settings(DatabaseSettings(url=admin_url))
        database = "history_" + uuid.uuid4().hex
        with admin.connect().execution_options(
            isolation_level="AUTOCOMMIT"
        ) as connection:
            connection.exec_driver_sql(f'CREATE DATABASE "{database}"')
        url = admin_url.rsplit("/", 1)[0] + "/" + database
    pair = [create_engine_from_settings(DatabaseSettings(url=url)) for _ in range(2)]
    yield pair
    for engine in pair:
        engine.dispose()
    if admin is not None:
        with admin.connect().execution_options(
            isolation_level="AUTOCOMMIT"
        ) as connection:
            connection.exec_driver_sql(f'DROP DATABASE "{database}"')
        admin.dispose()


async def test_simultaneous_bar_upserts_have_one_row_per_date(engines):
    first, second = engines
    ensure_schema(first, METADATA)
    with session_scope(sessionmaker(first)) as session:
        get_or_create_stock(session, "AAPL")
    barrier = threading.Barrier(2, timeout=5)

    def before_insert(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("INSERT INTO md_price_bars"):
            barrier.wait()

    for engine in engines:
        event.listen(engine, "before_cursor_execute", before_insert)

    def write(engine, value):
        with session_scope(sessionmaker(engine)) as session:
            return write_price_bars(
                session,
                "AAPL",
                bars(value=value).iloc[::-1] if value == 50 else bars(value=value),
            )

    try:
        assert await asyncio.gather(
            asyncio.to_thread(write, first, 100), asyncio.to_thread(write, second, 50)
        ) == [5, 5]
    finally:
        for engine in engines:
            event.remove(engine, "before_cursor_execute", before_insert)
    with first.connect() as connection:
        rows = connection.execute(
            select(MD_PRICE_BARS.c.date, MD_PRICE_BARS.c.close)
        ).all()
    assert len(rows) == 5
    assert {row.date for row in rows} == set(DAYS)
    assert {row.close for row in rows} in ({50}, {100})


async def test_simultaneous_initial_services_reject_older_response(engines):
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def history(symbol, start, end):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await release.wait()
            return bars(value=100)
        return bars(value=50)

    provider = AsyncMock()
    provider.history.side_effect = history
    now = [datetime(2026, 7, 18, 12, tzinfo=UTC)]
    first, second = [service(engine, provider, now) for engine in engines]
    barrier = threading.Barrier(2, timeout=5)

    def coordinate(original):
        def prepare(*args):
            barrier.wait()
            return original(*args)

        return prepare

    first._prepare_history = coordinate(first._prepare_history)
    second._prepare_history = coordinate(second._prepare_history)
    tasks = [
        asyncio.create_task(instance.get_price_history("AAPL", DAYS[0], DAYS[-1]))
        for instance in (first, second)
    ]
    try:
        await asyncio.wait_for(entered.wait(), 5)
        finished, _ = await asyncio.wait(
            tasks, timeout=5, return_when=asyncio.FIRST_COMPLETED
        )
        assert len(finished) == 1
        assert next(iter(finished)).result().Close.tolist() == [50] * 5
    finally:
        release.set()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    errors = [result for result in results if isinstance(result, ValueError)]
    assert len(errors) == 1 and "superseded" in str(errors[0])
    with engines[0].connect() as connection:
        assert (
            connection.execute(text("SELECT COUNT(*) FROM md_stocks")).scalar_one() == 1
        )
        assert (
            connection.execute(text("SELECT COUNT(*) FROM md_price_bars")).scalar_one()
            == 5
        )
        assert (
            connection.execute(select(MD_STOCKS.c.history_generation)).scalar_one() == 2
        )
    assert first._read_range("AAPL", DAYS[0], DAYS[-1]).Close.tolist() == [50] * 5


@pytest.mark.parametrize("newer_fails", [False, True])
async def test_different_range_generation_rejects_old_response(engines, newer_fails):
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def history(symbol, start, end):
        nonlocal calls
        calls += 1
        if calls == 1:
            return bars(DAYS[:2], 100)
        if calls == 2:
            entered.set()
            await release.wait()
            return bars(DAYS[:3], 75)
        if calls == 3 and newer_fails:
            return bars(DAYS[1:4], 50)  # Omits a previously stored date.
        return bars(DAYS[:4], 50)

    provider = AsyncMock()
    provider.history.side_effect = history
    now = [datetime(2026, 7, 18, 12, tzinfo=UTC)]
    first, second = [service(engine, provider, now) for engine in engines]
    await first.get_price_history("AAPL", DAYS[0], DAYS[1])
    timestamp = freshness(engines[0])
    older = asyncio.create_task(first.get_price_history("AAPL", DAYS[0], DAYS[2]))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        if newer_fails:
            with pytest.raises(ValueError, match="Incomplete"):
                await second.get_price_history("AAPL", DAYS[1], DAYS[3])
        else:
            assert (
                await second.get_price_history("AAPL", DAYS[1], DAYS[3])
            ).Close.tolist() == [50] * 3
    finally:
        release.set()
    with pytest.raises(ValueError, match="superseded"):
        await older
    expected = [100] * 2 if newer_fails else [50] * 4
    assert first._read_range("AAPL", DAYS[0], DAYS[-1]).Close.tolist() == expected
    if newer_fails:
        assert freshness(engines[0]) == timestamp
    assert (
        await second.get_price_history("AAPL", DAYS[0], DAYS[3])
    ).Close.tolist() == [50] * 4


async def test_failed_commit_rolls_back_bars_and_freshness_together(
    engines, monkeypatch
):
    provider = AsyncMock()
    provider.history.return_value = bars(DAYS[:2], 100)
    now = [datetime(2026, 7, 18, 12, tzinfo=UTC)]
    instance = service(engines[0], provider, now)
    await instance.get_price_history("AAPL", DAYS[0], DAYS[1])
    timestamp = freshness(engines[0])
    provider.history.return_value = bars(DAYS[:3], 50)
    original = data_module.write_price_bars

    def fail_after_upsert(*args):
        original(*args)
        raise RuntimeError("injected commit failure")

    with monkeypatch.context() as patch:
        patch.setattr(data_module, "write_price_bars", fail_after_upsert)
        with pytest.raises(RuntimeError, match="injected commit failure"):
            await instance.get_price_history("AAPL", DAYS[0], DAYS[2])
    assert freshness(engines[0]) == timestamp
    assert instance._read_range("AAPL", DAYS[0], DAYS[2]).Close.tolist() == [100] * 2
    assert (
        await instance.get_price_history("AAPL", DAYS[0], DAYS[2])
    ).Close.tolist() == [50] * 3


async def test_legacy_schema_adds_only_nullable_history_metadata(engines):
    engine = engines[0]
    legacy = MetaData()
    stocks = Table(
        "md_stocks",
        legacy,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("symbol", String(20), nullable=False, unique=True, index=True),
        Column("company_name", String(255), nullable=True),
    )
    price_bars = MD_PRICE_BARS.to_metadata(legacy)
    sentinel = Table("user_keeps_this", legacy, Column("value", String(20)))
    legacy.create_all(engine)
    with engine.begin() as connection:
        stock_id = connection.execute(
            insert(stocks)
            .values(symbol="AAPL", company_name="Kept")
            .returning(stocks.c.id)
        ).scalar_one()
        connection.execute(
            insert(price_bars).values(
                stock_id=stock_id,
                date=DAYS[0],
                open=100,
                high=101,
                low=99,
                close=100,
                volume=1000,
            )
        )
        connection.execute(insert(sentinel).values(value="untouched"))
    provider = AsyncMock()
    provider.history.return_value = bars(DAYS[:1], 50)
    instance = service(engine, provider, [datetime(2026, 7, 18, 12, tzinfo=UTC)])
    columns = {
        column["name"]: column for column in inspect(engine).get_columns("md_stocks")
    }
    assert set(columns) == {
        "id",
        "symbol",
        "company_name",
        "history_refreshed_at",
        "history_generation",
    }
    assert columns["history_refreshed_at"]["nullable"]
    assert columns["history_generation"]["nullable"]
    assert freshness(engine) is None
    assert instance._read_range("AAPL", DAYS[0], DAYS[0]).Close.tolist() == [100]
    assert (
        await instance.get_price_history("AAPL", DAYS[0], DAYS[0])
    ).Close.tolist() == [50]
    with engine.connect() as connection:
        assert connection.execute(select(sentinel.c.value)).scalar_one() == "untouched"
        assert connection.execute(select(stocks.c.company_name)).scalar_one() == "Kept"
