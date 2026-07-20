"""Tests for maverick.portfolio.journal.

Persistence layer for the `journal_entries`/`strategy_performance` tables
(names and columns ported verbatim from the legacy schema). Uses a tmp
SQLite database per test, matching `tests/portfolio/test_watchlist.py`'s
pattern.
"""

from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import sessionmaker

from maverick.platform.config import DatabaseSettings
from maverick.platform.db import (
    create_engine_from_settings,
    ensure_schema,
    session_scope,
)
from maverick.portfolio.journal import (
    JOURNAL_ENTRIES,
    METADATA,
    STRATEGY_PERFORMANCE,
    insert_trade,
    read_closed_trades,
    read_strategy_performance,
    read_strategy_performance_ranked,
    read_trade,
    read_trades,
    update_trade_close,
    upsert_strategy_performance,
)


@pytest.fixture
def factory(tmp_path):
    settings = DatabaseSettings(
        url=f"sqlite:///{tmp_path}/journal.db", use_pooling=True
    )
    engine = create_engine_from_settings(settings)
    ensure_schema(engine, METADATA)
    return sessionmaker(bind=engine)


def _entry_date() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


# -- insert_trade -------------------------------------------------------


def test_insert_trade_returns_open_payload(factory):
    with session_scope(factory) as session:
        entry = insert_trade(
            session,
            symbol="aapl",
            side="LONG",
            entry_price=150.0,
            shares=10.0,
            entry_date=_entry_date(),
            rationale="Momentum breakout",
            tags=["momentum"],
            notes=None,
        )

    assert entry.id is not None
    assert entry.symbol == "AAPL"
    assert entry.side == "long"
    assert entry.entry_price == 150.0
    assert entry.shares == 10.0
    assert entry.status == "open"
    assert entry.exit_price is None
    assert entry.pnl is None
    assert entry.tags == ["momentum"]


def test_insert_trade_defaults_tags_to_empty_list(factory):
    with session_scope(factory) as session:
        entry = insert_trade(
            session,
            symbol="AAPL",
            side="long",
            entry_price=150.0,
            shares=10.0,
            entry_date=_entry_date(),
            rationale=None,
            tags=[],
            notes=None,
        )
    assert entry.tags == []


# -- created_at/updated_at: legacy TimestampMixin carry-over --------------


def test_journal_entries_columns_are_a_superset_of_legacy_timestamp_columns():
    """`created_at`/`updated_at` must be present on `journal_entries` --
    legacy's `TimestampMixin` puts NOT NULL versions of both there. Omitting
    them would break inserts against a pre-existing legacy-shaped database
    (see the module docstring and
    `test_journal_operations_carry_over_against_a_preexisting_legacy_database`
    in `tests/portfolio/test_service_journal.py`)."""
    columns = {c.name for c in JOURNAL_ENTRIES.columns}
    assert {"created_at", "updated_at"} <= columns


def test_strategy_performance_has_no_timestamp_columns():
    """Legacy's `StrategyPerformance` model extends only `Base`, not
    `TimestampMixin` -- adding `created_at`/`updated_at` here would break
    inserts against a pre-existing legacy-shaped `strategy_performance`
    table that lacks them."""
    columns = {c.name for c in STRATEGY_PERFORMANCE.columns}
    assert "created_at" not in columns
    assert "updated_at" not in columns


def test_insert_trade_populates_created_at_and_updated_at(factory):
    with session_scope(factory) as session:
        entry_id = insert_trade(
            session,
            symbol="AAPL",
            side="long",
            entry_price=150.0,
            shares=10.0,
            entry_date=_entry_date(),
            rationale=None,
            tags=[],
            notes=None,
        ).id

    with session_scope(factory) as session:
        row = session.execute(
            JOURNAL_ENTRIES.select().where(JOURNAL_ENTRIES.c.id == entry_id)
        ).one()
    assert row.created_at is not None
    assert row.updated_at is not None


# -- read_trade / read_trades ---------------------------------------------


def test_read_trade_returns_none_for_unknown_id(factory):
    with session_scope(factory) as session:
        assert read_trade(session, 999999) is None


def test_read_trade_round_trips(factory):
    with session_scope(factory) as session:
        entry_id = insert_trade(
            session,
            symbol="AAPL",
            side="long",
            entry_price=150.0,
            shares=10.0,
            entry_date=_entry_date(),
            rationale=None,
            tags=[],
            notes=None,
        ).id

    with session_scope(factory) as session:
        fetched = read_trade(session, entry_id)

    assert fetched is not None
    assert fetched.id == entry_id
    assert fetched.symbol == "AAPL"


def test_read_trades_filters_by_symbol(factory):
    with session_scope(factory) as session:
        insert_trade(
            session,
            symbol="AAPL",
            side="long",
            entry_price=100.0,
            shares=1.0,
            entry_date=_entry_date(),
            rationale=None,
            tags=[],
            notes=None,
        )
        insert_trade(
            session,
            symbol="GOOG",
            side="long",
            entry_price=200.0,
            shares=1.0,
            entry_date=_entry_date(),
            rationale=None,
            tags=[],
            notes=None,
        )

    with session_scope(factory) as session:
        entries = read_trades(session, "aapl", None)

    assert len(entries) == 1
    assert entries[0].symbol == "AAPL"


def test_read_trades_filters_by_status(factory):
    with session_scope(factory) as session:
        open_id = insert_trade(
            session,
            symbol="AAPL",
            side="long",
            entry_price=100.0,
            shares=1.0,
            entry_date=_entry_date(),
            rationale=None,
            tags=[],
            notes=None,
        ).id
        closed_id = insert_trade(
            session,
            symbol="GOOG",
            side="long",
            entry_price=200.0,
            shares=1.0,
            entry_date=_entry_date(),
            rationale=None,
            tags=[],
            notes=None,
        ).id
        update_trade_close(
            session,
            closed_id,
            exit_price=210.0,
            exit_date=_entry_date(),
            pnl=10.0,
            notes=None,
        )

    with session_scope(factory) as session:
        open_entries = read_trades(session, None, "open")
        closed_entries = read_trades(session, None, "closed")

    assert [e.id for e in open_entries] == [open_id]
    assert [e.id for e in closed_entries] == [closed_id]


def test_read_trades_orders_by_entry_date_descending(factory):
    with session_scope(factory) as session:
        insert_trade(
            session,
            symbol="AAPL",
            side="long",
            entry_price=100.0,
            shares=1.0,
            entry_date=datetime(2026, 1, 1, tzinfo=UTC),
            rationale=None,
            tags=[],
            notes=None,
        )
        insert_trade(
            session,
            symbol="MSFT",
            side="long",
            entry_price=100.0,
            shares=1.0,
            entry_date=datetime(2026, 2, 1, tzinfo=UTC),
            rationale=None,
            tags=[],
            notes=None,
        )

    with session_scope(factory) as session:
        entries = read_trades(session, None, None)

    assert [e.symbol for e in entries] == ["MSFT", "AAPL"]


# -- update_trade_close -----------------------------------------------


def test_update_trade_close_marks_closed_with_pnl(factory):
    with session_scope(factory) as session:
        entry_id = insert_trade(
            session,
            symbol="AAPL",
            side="long",
            entry_price=100.0,
            shares=10.0,
            entry_date=_entry_date(),
            rationale=None,
            tags=[],
            notes=None,
        ).id

    with session_scope(factory) as session:
        closed = update_trade_close(
            session,
            entry_id,
            exit_price=120.0,
            exit_date=_entry_date(),
            pnl=200.0,
            notes="closed note",
        )

    assert closed.status == "closed"
    assert closed.exit_price == 120.0
    assert closed.pnl == 200.0
    assert closed.notes == "closed note"


# -- strategy_performance -------------------------------------------------


def test_upsert_strategy_performance_inserts_new_row(factory):
    with session_scope(factory) as session:
        perf = upsert_strategy_performance(
            session,
            "momentum",
            period="all_time",
            win_count=2,
            loss_count=1,
            total_pnl=250.0,
            avg_win=150.0,
            avg_loss=50.0,
            expectancy=83.33,
            profit_factor=6.0,
        )
    assert perf.strategy_tag == "momentum"
    assert perf.win_count == 2
    assert perf.expectancy == 83.33


def test_upsert_strategy_performance_updates_existing_row(factory):
    with session_scope(factory) as session:
        upsert_strategy_performance(
            session,
            "momentum",
            period="all_time",
            win_count=1,
            loss_count=0,
            total_pnl=100.0,
            avg_win=100.0,
            avg_loss=0.0,
            expectancy=100.0,
            profit_factor=0.0,
        )
        updated = upsert_strategy_performance(
            session,
            "momentum",
            period="all_time",
            win_count=2,
            loss_count=1,
            total_pnl=250.0,
            avg_win=150.0,
            avg_loss=50.0,
            expectancy=83.33,
            profit_factor=6.0,
        )

    assert updated.win_count == 2

    with session_scope(factory) as session:
        rows = session.execute(STRATEGY_PERFORMANCE.select()).all()
    assert len(rows) == 1


def test_read_strategy_performance_returns_none_when_missing(factory):
    with session_scope(factory) as session:
        assert read_strategy_performance(session, "unknown") is None


def test_read_strategy_performance_ranked_orders_by_expectancy_descending(factory):
    with session_scope(factory) as session:
        upsert_strategy_performance(
            session,
            "low",
            period="all_time",
            win_count=1,
            loss_count=1,
            total_pnl=0.0,
            avg_win=10.0,
            avg_loss=10.0,
            expectancy=1.0,
            profit_factor=1.0,
        )
        upsert_strategy_performance(
            session,
            "high",
            period="all_time",
            win_count=2,
            loss_count=0,
            total_pnl=200.0,
            avg_win=100.0,
            avg_loss=0.0,
            expectancy=100.0,
            profit_factor=0.0,
        )

    with session_scope(factory) as session:
        ranked = read_strategy_performance_ranked(session)

    assert [r.strategy_tag for r in ranked] == ["high", "low"]


def test_read_closed_trades_excludes_open(factory):
    with session_scope(factory) as session:
        open_id = insert_trade(
            session,
            symbol="AAPL",
            side="long",
            entry_price=100.0,
            shares=1.0,
            entry_date=_entry_date(),
            rationale=None,
            tags=[],
            notes=None,
        ).id
        closed_id = insert_trade(
            session,
            symbol="MSFT",
            side="long",
            entry_price=100.0,
            shares=1.0,
            entry_date=_entry_date(),
            rationale=None,
            tags=[],
            notes=None,
        ).id
        update_trade_close(
            session,
            closed_id,
            exit_price=110.0,
            exit_date=_entry_date(),
            pnl=10.0,
            notes=None,
        )

    with session_scope(factory) as session:
        closed = read_closed_trades(session)

    assert [e.id for e in closed] == [closed_id]
    assert open_id not in [e.id for e in closed]
