"""Tests for maverick.portfolio.data.

Persistence layer: stores what the ledger computed, no math. Uses a tmp
SQLite database per test (via `platform.db.create_engine_from_settings` +
`ensure_schema`), matching the pattern in `tests/market_data/test_data.py`.
"""

import uuid
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import pytest
from sqlalchemy import delete as sa_delete
from sqlalchemy import func, insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from maverick.platform.config import DatabaseSettings
from maverick.platform.db import (
    create_engine_from_settings,
    ensure_schema,
    session_scope,
)
from maverick.portfolio.data import (
    METADATA,
    PF_PORTFOLIOS,
    PF_POSITIONS,
    clear_positions,
    delete_position,
    get_or_create_portfolio,
    read_positions,
    upsert_position,
)
from maverick.portfolio.types import PositionPayload


@pytest.fixture
def factory(tmp_path):
    settings = DatabaseSettings(
        url=f"sqlite:///{tmp_path}/portfolio.db", use_pooling=True
    )
    engine = create_engine_from_settings(settings)
    ensure_schema(engine, METADATA)
    return sessionmaker(bind=engine)


def _position(**overrides) -> PositionPayload:
    fields = {
        "ticker": "AAPL",
        "shares": Decimal("10.12345678"),
        "average_cost_basis": Decimal("105.0001"),
        "total_cost": Decimal("1062.7259"),
        "purchase_date": "2026-01-15T00:00:00+00:00",
        "notes": "Long-term hold",
    }
    fields.update(overrides)
    return PositionPayload(**fields)


# -- get_or_create_portfolio -------------------------------------------------


def test_get_or_create_portfolio_creates_new_row(factory):
    with session_scope(factory) as session:
        portfolio_id = get_or_create_portfolio(session, "default", "My Portfolio")

    assert isinstance(portfolio_id, uuid.UUID)

    with session_scope(factory) as session:
        count = session.execute(
            select(func.count()).select_from(PF_PORTFOLIOS)
        ).scalar_one()
    assert count == 1


def test_get_or_create_portfolio_is_idempotent(factory):
    with session_scope(factory) as session:
        first_id = get_or_create_portfolio(session, "default", "My Portfolio")
    with session_scope(factory) as session:
        second_id = get_or_create_portfolio(session, "default", "My Portfolio")

    assert first_id == second_id

    with session_scope(factory) as session:
        count = session.execute(
            select(func.count()).select_from(PF_PORTFOLIOS)
        ).scalar_one()
    assert count == 1


def test_get_or_create_portfolio_distinguishes_by_user_id(factory):
    with session_scope(factory) as session:
        alice_id = get_or_create_portfolio(session, "alice", "My Portfolio")
    with session_scope(factory) as session:
        bob_id = get_or_create_portfolio(session, "bob", "My Portfolio")

    assert alice_id != bob_id


def test_get_or_create_portfolio_distinguishes_by_name(factory):
    with session_scope(factory) as session:
        default_id = get_or_create_portfolio(session, "default", "My Portfolio")
    with session_scope(factory) as session:
        trading_id = get_or_create_portfolio(session, "default", "Trading Account")

    assert default_id != trading_id


def test_get_or_create_portfolio_enforces_user_name_uniqueness(factory):
    """The (user_id, name) unique constraint holds even bypassing get_or_create."""
    with session_scope(factory) as session:
        session.execute(insert(PF_PORTFOLIOS).values(user_id="default", name="Dup"))

    with pytest.raises(IntegrityError):
        with session_scope(factory) as session:
            session.execute(insert(PF_PORTFOLIOS).values(user_id="default", name="Dup"))


# -- Decimal round-trip exactness --------------------------------------------


def test_upsert_then_read_preserves_decimal_shares_exactly(factory):
    with session_scope(factory) as session:
        portfolio_id = get_or_create_portfolio(session, "default", "My Portfolio")
        upsert_position(
            session,
            portfolio_id,
            _position(
                shares=Decimal("10.12345678"), average_cost_basis=Decimal("105.0001")
            ),
        )

    with session_scope(factory) as session:
        positions = read_positions(session, portfolio_id)

    assert len(positions) == 1
    assert positions[0].shares == Decimal("10.12345678")
    assert str(positions[0].shares) == "10.12345678"
    assert positions[0].average_cost_basis == Decimal("105.0001")
    assert str(positions[0].average_cost_basis) == "105.0001"


def test_upsert_then_read_preserves_total_cost_exactly(factory):
    with session_scope(factory) as session:
        portfolio_id = get_or_create_portfolio(session, "default", "My Portfolio")
        upsert_position(
            session, portfolio_id, _position(total_cost=Decimal("1062.7259"))
        )

    with session_scope(factory) as session:
        positions = read_positions(session, portfolio_id)

    assert positions[0].total_cost == Decimal("1062.7259")


# -- read_positions -----------------------------------------------------------


def test_read_positions_returns_empty_list_for_new_portfolio(factory):
    with session_scope(factory) as session:
        portfolio_id = get_or_create_portfolio(session, "default", "My Portfolio")

    with session_scope(factory) as session:
        positions = read_positions(session, portfolio_id)

    assert positions == []


def test_read_positions_returns_all_positions_for_portfolio(factory):
    with session_scope(factory) as session:
        portfolio_id = get_or_create_portfolio(session, "default", "My Portfolio")
        upsert_position(session, portfolio_id, _position(ticker="AAPL"))
        upsert_position(session, portfolio_id, _position(ticker="MSFT"))

    with session_scope(factory) as session:
        positions = read_positions(session, portfolio_id)

    tickers = {p.ticker for p in positions}
    assert tickers == {"AAPL", "MSFT"}


def test_read_positions_only_returns_positions_for_that_portfolio(factory):
    with session_scope(factory) as session:
        portfolio_a = get_or_create_portfolio(session, "default", "Portfolio A")
        portfolio_b = get_or_create_portfolio(session, "default", "Portfolio B")
        upsert_position(session, portfolio_a, _position(ticker="AAPL"))
        upsert_position(session, portfolio_b, _position(ticker="MSFT"))

    with session_scope(factory) as session:
        positions_a = read_positions(session, portfolio_a)

    assert len(positions_a) == 1
    assert positions_a[0].ticker == "AAPL"


def test_read_positions_preserves_notes(factory):
    with session_scope(factory) as session:
        portfolio_id = get_or_create_portfolio(session, "default", "My Portfolio")
        upsert_position(session, portfolio_id, _position(notes="Core holding"))

    with session_scope(factory) as session:
        positions = read_positions(session, portfolio_id)

    assert positions[0].notes == "Core holding"


def test_read_positions_preserves_none_notes(factory):
    with session_scope(factory) as session:
        portfolio_id = get_or_create_portfolio(session, "default", "My Portfolio")
        upsert_position(session, portfolio_id, _position(notes=None))

    with session_scope(factory) as session:
        positions = read_positions(session, portfolio_id)

    assert positions[0].notes is None


# -- upsert_position: insert then update, no dupes ---------------------------


def test_upsert_position_inserts_new_ticker(factory):
    with session_scope(factory) as session:
        portfolio_id = get_or_create_portfolio(session, "default", "My Portfolio")
        upsert_position(session, portfolio_id, _position(ticker="AAPL"))

    with session_scope(factory) as session:
        count = session.execute(
            select(func.count()).select_from(PF_POSITIONS)
        ).scalar_one()
    assert count == 1


def test_upsert_position_updates_existing_ticker_without_duplicating(factory):
    with session_scope(factory) as session:
        portfolio_id = get_or_create_portfolio(session, "default", "My Portfolio")
        upsert_position(
            session, portfolio_id, _position(ticker="AAPL", shares=Decimal("10"))
        )

    with session_scope(factory) as session:
        upsert_position(
            session, portfolio_id, _position(ticker="AAPL", shares=Decimal("20"))
        )

    with session_scope(factory) as session:
        count = session.execute(
            select(func.count()).select_from(PF_POSITIONS)
        ).scalar_one()
        positions = read_positions(session, portfolio_id)

    assert count == 1
    assert len(positions) == 1
    assert positions[0].shares == Decimal("20")


def test_upsert_position_distinguishes_tickers_across_portfolios(factory):
    with session_scope(factory) as session:
        portfolio_a = get_or_create_portfolio(session, "default", "Portfolio A")
        portfolio_b = get_or_create_portfolio(session, "default", "Portfolio B")
        upsert_position(
            session, portfolio_a, _position(ticker="AAPL", shares=Decimal("10"))
        )
        upsert_position(
            session, portfolio_b, _position(ticker="AAPL", shares=Decimal("99"))
        )

    with session_scope(factory) as session:
        positions_a = read_positions(session, portfolio_a)
        positions_b = read_positions(session, portfolio_b)

    assert positions_a[0].shares == Decimal("10")
    assert positions_b[0].shares == Decimal("99")


# -- delete_position ------------------------------------------------------


def test_delete_position_returns_true_and_removes_row(factory):
    with session_scope(factory) as session:
        portfolio_id = get_or_create_portfolio(session, "default", "My Portfolio")
        upsert_position(session, portfolio_id, _position(ticker="AAPL"))

    with session_scope(factory) as session:
        deleted = delete_position(session, portfolio_id, "AAPL")

    assert deleted is True

    with session_scope(factory) as session:
        positions = read_positions(session, portfolio_id)
    assert positions == []


def test_delete_position_returns_false_when_not_found(factory):
    with session_scope(factory) as session:
        portfolio_id = get_or_create_portfolio(session, "default", "My Portfolio")

    with session_scope(factory) as session:
        deleted = delete_position(session, portfolio_id, "NOPE")

    assert deleted is False


def test_delete_position_only_removes_from_target_portfolio(factory):
    with session_scope(factory) as session:
        portfolio_a = get_or_create_portfolio(session, "default", "Portfolio A")
        portfolio_b = get_or_create_portfolio(session, "default", "Portfolio B")
        upsert_position(session, portfolio_a, _position(ticker="AAPL"))
        upsert_position(session, portfolio_b, _position(ticker="AAPL"))

    with session_scope(factory) as session:
        delete_position(session, portfolio_a, "AAPL")

    with session_scope(factory) as session:
        positions_a = read_positions(session, portfolio_a)
        positions_b = read_positions(session, portfolio_b)

    assert positions_a == []
    assert len(positions_b) == 1


# -- clear_positions --------------------------------------------------------


def test_clear_positions_returns_count_and_removes_all(factory):
    with session_scope(factory) as session:
        portfolio_id = get_or_create_portfolio(session, "default", "My Portfolio")
        upsert_position(session, portfolio_id, _position(ticker="AAPL"))
        upsert_position(session, portfolio_id, _position(ticker="MSFT"))

    with session_scope(factory) as session:
        cleared = clear_positions(session, portfolio_id)

    assert cleared == 2

    with session_scope(factory) as session:
        positions = read_positions(session, portfolio_id)
    assert positions == []


def test_clear_positions_returns_zero_for_empty_portfolio(factory):
    with session_scope(factory) as session:
        portfolio_id = get_or_create_portfolio(session, "default", "My Portfolio")

    with session_scope(factory) as session:
        cleared = clear_positions(session, portfolio_id)

    assert cleared == 0


def test_clear_positions_only_clears_target_portfolio(factory):
    with session_scope(factory) as session:
        portfolio_a = get_or_create_portfolio(session, "default", "Portfolio A")
        portfolio_b = get_or_create_portfolio(session, "default", "Portfolio B")
        upsert_position(session, portfolio_a, _position(ticker="AAPL"))
        upsert_position(session, portfolio_b, _position(ticker="MSFT"))

    with session_scope(factory) as session:
        clear_positions(session, portfolio_a)

    with session_scope(factory) as session:
        positions_b = read_positions(session, portfolio_b)
    assert len(positions_b) == 1


# -- cascade delete -----------------------------------------------------------


def test_deleting_portfolio_row_cascades_to_positions(factory):
    with session_scope(factory) as session:
        portfolio_id = get_or_create_portfolio(session, "default", "My Portfolio")
        upsert_position(session, portfolio_id, _position(ticker="AAPL"))
        upsert_position(session, portfolio_id, _position(ticker="MSFT"))

    with session_scope(factory) as session:
        session.execute(
            sa_delete(PF_PORTFOLIOS).where(PF_PORTFOLIOS.c.id == portfolio_id)
        )

    with session_scope(factory) as session:
        remaining = session.execute(
            select(func.count())
            .select_from(PF_POSITIONS)
            .where(PF_POSITIONS.c.portfolio_id == portfolio_id)
        ).scalar_one()

    assert remaining == 0


# -- tz-aware purchase_date round-trip ----------------------------------------


def test_purchase_date_round_trips_as_the_same_instant_utc(factory):
    original = datetime(2026, 1, 15, 12, 30, 0, tzinfo=UTC)
    with session_scope(factory) as session:
        portfolio_id = get_or_create_portfolio(session, "default", "My Portfolio")
        upsert_position(
            session, portfolio_id, _position(purchase_date=original.isoformat())
        )

    with session_scope(factory) as session:
        positions = read_positions(session, portfolio_id)

    round_tripped = datetime.fromisoformat(positions[0].purchase_date)
    assert round_tripped.tzinfo is not None
    assert round_tripped == original


def test_purchase_date_round_trips_the_same_instant_for_non_utc_offset(factory):
    # -04:00 offset: exercises actual tz normalization, not just a UTC passthrough.
    original = datetime(2026, 7, 19, 12, 30, 0, tzinfo=timezone(timedelta(hours=-4)))
    with session_scope(factory) as session:
        portfolio_id = get_or_create_portfolio(session, "default", "My Portfolio")
        upsert_position(
            session, portfolio_id, _position(purchase_date=original.isoformat())
        )

    with session_scope(factory) as session:
        positions = read_positions(session, portfolio_id)

    round_tripped = datetime.fromisoformat(positions[0].purchase_date)
    assert round_tripped.tzinfo is not None
    assert round_tripped == original


# -- empty states ---------------------------------------------------------


def test_delete_position_on_portfolio_with_no_positions_returns_false(factory):
    with session_scope(factory) as session:
        portfolio_id = get_or_create_portfolio(session, "default", "Empty")

    with session_scope(factory) as session:
        assert delete_position(session, portfolio_id, "AAPL") is False
