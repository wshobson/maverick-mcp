"""Tests for maverick.portfolio.service_journal.

Uses a tmp SQLite engine (matching `tests/portfolio/test_service.py`'s
pattern). `JournalService` is a standalone domain service (not composed
inside `PortfolioService` -- see `service_journal.py`'s module docstring),
so it is constructed directly here rather than via `PortfolioService`.
"""

from decimal import Decimal

import pytest

from maverick.platform.config import DatabaseSettings
from maverick.platform.db import create_engine_from_settings
from maverick.portfolio.service_journal import JournalService
from maverick.portfolio.types import JournalEntryPayload


def _engine(tmp_path):
    settings = DatabaseSettings(
        url=f"sqlite:///{tmp_path}/journal.db", use_pooling=True
    )
    return create_engine_from_settings(settings)


def _service(tmp_path) -> JournalService:
    return JournalService(_engine(tmp_path))


# -- add_trade / close_trade round trips ------------------------------


async def test_add_trade_creates_open_entry(tmp_path):
    service = _service(tmp_path)

    entry = await service.add_trade(
        symbol="aapl",
        side="LONG",
        entry_price=Decimal("150.0"),
        shares=Decimal("10"),
        rationale="Momentum breakout",
        tags=["momentum"],
        notes="watching for continuation",
    )

    assert entry.symbol == "AAPL"
    assert entry.side == "long"
    assert entry.entry_price == 150.0
    assert entry.shares == 10.0
    assert entry.status == "open"
    assert entry.exit_price is None
    assert entry.pnl is None
    assert entry.tags == ["momentum"]
    assert entry.notes == "watching for continuation"


async def test_add_trade_quantizes_entry_price_to_cents(tmp_path):
    service = _service(tmp_path)

    entry = await service.add_trade(
        symbol="AAPL",
        side="long",
        entry_price=Decimal("150.005"),
        shares=Decimal("1"),
    )

    assert entry.entry_price == 150.01  # ROUND_HALF_UP


async def test_close_trade_long_computes_pnl(tmp_path):
    service = _service(tmp_path)
    entry = await service.add_trade(
        symbol="AAPL", side="long", entry_price=Decimal("100"), shares=Decimal("10")
    )

    closed = await service.close_trade(entry.id, exit_price=Decimal("120"))

    assert closed.status == "closed"
    assert closed.exit_price == 120.0
    assert closed.pnl == 200.0  # (120-100)*10


async def test_close_trade_short_computes_pnl(tmp_path):
    service = _service(tmp_path)
    entry = await service.add_trade(
        symbol="TSLA", side="short", entry_price=Decimal("300"), shares=Decimal("5")
    )

    closed = await service.close_trade(entry.id, exit_price=Decimal("270"))

    assert closed.pnl == 150.0  # (300-270)*5


async def test_close_trade_short_loss_computes_negative_pnl(tmp_path):
    service = _service(tmp_path)
    entry = await service.add_trade(
        symbol="TSLA", side="short", entry_price=Decimal("300"), shares=Decimal("5")
    )

    closed = await service.close_trade(entry.id, exit_price=Decimal("320"))

    assert closed.pnl == -100.0


async def test_close_trade_not_found_raises(tmp_path):
    service = _service(tmp_path)

    with pytest.raises(ValueError, match="not found"):
        await service.close_trade(9999, exit_price=Decimal("100"))


async def test_close_trade_already_closed_raises(tmp_path):
    service = _service(tmp_path)
    entry = await service.add_trade(
        symbol="AAPL", side="long", entry_price=Decimal("100"), shares=Decimal("1")
    )
    await service.close_trade(entry.id, exit_price=Decimal("110"))

    with pytest.raises(ValueError, match="already closed"):
        await service.close_trade(entry.id, exit_price=Decimal("120"))


async def test_close_trade_appends_notes_when_provided(tmp_path):
    service = _service(tmp_path)
    entry = await service.add_trade(
        symbol="AAPL",
        side="long",
        entry_price=Decimal("100"),
        shares=Decimal("1"),
        notes="Initial note",
    )

    closed = await service.close_trade(
        entry.id, exit_price=Decimal("110"), notes="Exit note"
    )

    assert closed.notes == "Initial note\nExit note"


async def test_close_trade_preserves_notes_when_none_provided(tmp_path):
    service = _service(tmp_path)
    entry = await service.add_trade(
        symbol="AAPL",
        side="long",
        entry_price=Decimal("100"),
        shares=Decimal("1"),
        notes="Initial note",
    )

    closed = await service.close_trade(entry.id, exit_price=Decimal("110"))

    assert closed.notes == "Initial note"


# -- list_trades ------------------------------------------------------


async def test_list_trades_filters_by_status(tmp_path):
    service = _service(tmp_path)
    await service.add_trade(
        symbol="AAPL", side="long", entry_price=Decimal("100"), shares=Decimal("1")
    )
    entry_two = await service.add_trade(
        symbol="GOOG", side="long", entry_price=Decimal("200"), shares=Decimal("1")
    )
    await service.close_trade(entry_two.id, exit_price=Decimal("210"))

    open_trades = await service.list_trades(status="open")
    closed_trades = await service.list_trades(status="closed")

    assert len(open_trades) == 1
    assert open_trades[0].symbol == "AAPL"
    assert len(closed_trades) == 1
    assert closed_trades[0].symbol == "GOOG"


async def test_list_trades_filters_by_symbol(tmp_path):
    service = _service(tmp_path)
    await service.add_trade(
        symbol="AAPL", side="long", entry_price=Decimal("100"), shares=Decimal("1")
    )
    await service.add_trade(
        symbol="AAPL", side="short", entry_price=Decimal("150"), shares=Decimal("2")
    )
    await service.add_trade(
        symbol="GOOG", side="long", entry_price=Decimal("200"), shares=Decimal("1")
    )

    trades = await service.list_trades(symbol="aapl")

    assert len(trades) == 2
    assert all(t.symbol == "AAPL" for t in trades)


async def test_list_trades_filters_by_strategy_tag(tmp_path):
    service = _service(tmp_path)
    await service.add_trade(
        symbol="AAPL",
        side="long",
        entry_price=Decimal("100"),
        shares=Decimal("1"),
        tags=["momentum"],
    )
    await service.add_trade(
        symbol="GOOG",
        side="long",
        entry_price=Decimal("200"),
        shares=Decimal("1"),
        tags=["value"],
    )

    trades = await service.list_trades(strategy_tag="momentum")

    assert len(trades) == 1
    assert trades[0].symbol == "AAPL"


async def test_list_trades_respects_limit_after_tag_filter(tmp_path):
    service = _service(tmp_path)
    for i in range(3):
        await service.add_trade(
            symbol=f"T{i}",
            side="long",
            entry_price=Decimal("100"),
            shares=Decimal("1"),
            tags=["momentum"],
        )

    trades = await service.list_trades(strategy_tag="momentum", limit=2)

    assert len(trades) == 2


async def test_list_trades_defaults_limit_from_settings(tmp_path):
    service = _service(tmp_path)
    assert service.settings.journal_list_default_limit == 50


# -- get_trade / review_trade -------------------------------------------


async def test_get_trade_returns_none_for_unknown_id(tmp_path):
    service = _service(tmp_path)
    assert await service.get_trade(9999) is None


async def test_review_trade_returns_none_for_unknown_id(tmp_path):
    service = _service(tmp_path)
    assert await service.review_trade(9999) is None


async def test_review_trade_long_computes_pnl_pct(tmp_path):
    service = _service(tmp_path)
    entry = await service.add_trade(
        symbol="AAPL", side="long", entry_price=Decimal("100"), shares=Decimal("10")
    )
    await service.close_trade(entry.id, exit_price=Decimal("120"))

    review = await service.review_trade(entry.id)

    assert review.pnl_pct == 20.0  # (120-100)/100*100


async def test_review_trade_short_computes_pnl_pct(tmp_path):
    service = _service(tmp_path)
    entry = await service.add_trade(
        symbol="TSLA", side="short", entry_price=Decimal("300"), shares=Decimal("5")
    )
    await service.close_trade(entry.id, exit_price=Decimal("270"))

    review = await service.review_trade(entry.id)

    assert review.pnl_pct == 10.0  # (300-270)/300*100


async def test_review_trade_open_trade_has_no_pnl_pct(tmp_path):
    service = _service(tmp_path)
    entry = await service.add_trade(
        symbol="AAPL", side="long", entry_price=Decimal("100"), shares=Decimal("10")
    )

    review = await service.review_trade(entry.id)

    assert review.pnl_pct is None


# -- strategy performance: hand-computable fixture -----------------------


async def test_close_trade_recomputes_strategy_performance(tmp_path):
    """Three closed trades tagged "momentum": +200, +100, -50.

    win_count=2, loss_count=1, total_pnl=250.0, avg_win=150.0,
    avg_loss=50.0. expectancy = (2/3 * 150.00) - (1/3 * 50.00), quantized
    to 0.01 with ROUND_HALF_UP -> 83.33. profit_factor = 300.00/50.00 =
    6.00. Hand-verified via the exact Decimal path this module uses."""
    service = _service(tmp_path)

    aapl = await service.add_trade(
        symbol="AAPL",
        side="long",
        entry_price=Decimal("100"),
        shares=Decimal("10"),
        tags=["momentum"],
    )
    goog = await service.add_trade(
        symbol="GOOG",
        side="long",
        entry_price=Decimal("200"),
        shares=Decimal("5"),
        tags=["momentum"],
    )
    tsla = await service.add_trade(
        symbol="TSLA",
        side="short",
        entry_price=Decimal("300"),
        shares=Decimal("2"),
        tags=["momentum"],
    )

    await service.close_trade(aapl.id, exit_price=Decimal("120"))  # +200
    await service.close_trade(goog.id, exit_price=Decimal("220"))  # +100
    await service.close_trade(tsla.id, exit_price=Decimal("325"))  # -50

    perf = await service.get_strategy_performance("momentum")

    assert perf is not None
    assert perf.win_count == 2
    assert perf.loss_count == 1
    assert perf.total_pnl == 250.0
    assert perf.avg_win == 150.0
    assert perf.avg_loss == 50.0
    assert perf.expectancy == 83.33
    assert perf.profit_factor == 6.0
    assert perf.period == "all_time"


async def test_get_strategy_performance_returns_none_before_any_close(tmp_path):
    """Legacy semantics: performance is only persisted after a close_trade
    recompute for that tag -- an untouched tag has no row at all."""
    service = _service(tmp_path)
    await service.add_trade(
        symbol="AAPL",
        side="long",
        entry_price=Decimal("100"),
        shares=Decimal("1"),
        tags=["untouched"],
    )

    assert await service.get_strategy_performance("untouched") is None


async def test_strategy_performance_all_wins_gives_infinite_profit_factor(tmp_path):
    """No losing trades: profit_factor is `float("inf")`, matching legacy
    `StrategyTracker.recompute` exactly (its own "capped for serialization"
    comment does not reflect what its code actually does)."""
    service = _service(tmp_path)
    entry = await service.add_trade(
        symbol="AAPL",
        side="long",
        entry_price=Decimal("100"),
        shares=Decimal("1"),
        tags=["solo"],
    )
    await service.close_trade(entry.id, exit_price=Decimal("110"))

    perf = await service.get_strategy_performance("solo")

    assert perf.profit_factor == float("inf")


async def test_close_trade_recomputes_every_tag_on_the_entry(tmp_path):
    service = _service(tmp_path)
    entry = await service.add_trade(
        symbol="AAPL",
        side="long",
        entry_price=Decimal("100"),
        shares=Decimal("1"),
        tags=["momentum", "breakout"],
    )
    await service.close_trade(entry.id, exit_price=Decimal("110"))

    momentum = await service.get_strategy_performance("momentum")
    breakout = await service.get_strategy_performance("breakout")

    assert momentum is not None
    assert breakout is not None
    assert momentum.total_pnl == breakout.total_pnl == 10.0


async def test_compare_strategies_ranks_by_expectancy_descending(tmp_path):
    service = _service(tmp_path)
    low = await service.add_trade(
        symbol="AAPL",
        side="long",
        entry_price=Decimal("100"),
        shares=Decimal("1"),
        tags=["low"],
    )
    high = await service.add_trade(
        symbol="GOOG",
        side="long",
        entry_price=Decimal("100"),
        shares=Decimal("1"),
        tags=["high"],
    )
    await service.close_trade(low.id, exit_price=Decimal("101"))  # expectancy 1.00
    await service.close_trade(high.id, exit_price=Decimal("200"))  # expectancy 100.00

    ranked = await service.compare_strategies()

    assert [r.strategy_tag for r in ranked] == ["high", "low"]


async def test_compare_strategies_empty_when_no_closed_tagged_trades(tmp_path):
    service = _service(tmp_path)
    assert await service.compare_strategies() == []


# -- carry-over: real legacy tables ---------------------------------------


async def test_journal_operations_carry_over_against_a_preexisting_legacy_database(
    tmp_path,
):
    """The real carry-over scenario named in `journal.py`'s module
    docstring: a pre-existing database already has `journal_entries`/
    `strategy_performance` created by the now-deleted legacy
    `maverick_mcp.services.journal.models` declarative models
    (`JournalEntry` -> `TimestampMixin` -> NOT NULL `created_at`/
    `updated_at`; `StrategyPerformance` -> no timestamp columns at all).
    `ensure_schema`'s checkfirst behavior means the new stack sees the
    tables already present and never re-creates them -- it has to write
    into the legacy shape as-is. Exercises add/close/list/review/
    get_strategy_performance/compare_strategies end to end against that
    exact shape. The legacy package is gone, so the shape is reconstructed
    inline here (plain `Table` objects, not ORM classes) rather than
    imported."""
    from sqlalchemy import (
        JSON,
        Column,
        DateTime,
        Float,
        Integer,
        MetaData,
        String,
        Table,
        Text,
    )

    legacy_metadata = MetaData()
    journal_entries_table = Table(
        "journal_entries",
        legacy_metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("symbol", String(20), nullable=False, index=True),
        Column("side", String(10), nullable=False),
        Column("entry_price", Float, nullable=False),
        Column("exit_price", Float, nullable=True),
        Column("shares", Float, nullable=False),
        Column("entry_date", DateTime(timezone=True), nullable=False),
        Column("exit_date", DateTime(timezone=True), nullable=True),
        Column("rationale", Text, nullable=True),
        Column("tags", JSON, nullable=True),
        Column("pnl", Float, nullable=True),
        Column("r_multiple", Float, nullable=True),
        Column("notes", Text, nullable=True),
        Column("status", String(10), nullable=False),
        Column("created_at", DateTime(timezone=True), nullable=False),
        Column("updated_at", DateTime(timezone=True), nullable=False),
    )
    strategy_performance_table = Table(
        "strategy_performance",
        legacy_metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("strategy_tag", String(100), nullable=False, index=True, unique=True),
        Column("period", String(20), nullable=False),
        Column("win_count", Integer, nullable=False),
        Column("loss_count", Integer, nullable=False),
        Column("total_pnl", Float, nullable=False),
        Column("avg_win", Float, nullable=False),
        Column("avg_loss", Float, nullable=False),
        Column("expectancy", Float, nullable=False),
        Column("profit_factor", Float, nullable=False),
    )

    engine = _engine(tmp_path)
    legacy_metadata.create_all(
        engine, tables=[journal_entries_table, strategy_performance_table]
    )

    service = JournalService(engine)

    entry = await service.add_trade(
        symbol="aapl",
        side="LONG",
        entry_price=Decimal("100"),
        shares=Decimal("10"),
        tags=["momentum"],
    )
    assert entry.symbol == "AAPL"

    closed = await service.close_trade(entry.id, exit_price=Decimal("120"))
    assert closed.status == "closed"
    assert closed.pnl == 200.0

    trades = await service.list_trades(symbol="AAPL")
    assert len(trades) == 1
    assert isinstance(trades[0], JournalEntryPayload)

    review = await service.review_trade(entry.id)
    assert review.pnl_pct == 20.0

    perf = await service.get_strategy_performance("momentum")
    assert perf is not None
    assert perf.win_count == 1
    assert perf.total_pnl == 200.0

    ranked = await service.compare_strategies()
    assert [r.strategy_tag for r in ranked] == ["momentum"]
