"""Unit tests for StrategyTracker analytics."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from maverick_mcp.database.base import Base
from maverick_mcp.services.event_bus import EventBus
from maverick_mcp.services.journal.analytics import StrategyTracker
from maverick_mcp.services.journal.service import JournalService

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    _Session = sessionmaker(bind=engine)
    session = _Session()
    yield session
    session.close()


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def service(db_session, event_bus):
    return JournalService(db_session=db_session, event_bus=event_bus)


@pytest.fixture
def tracker(db_session):
    return StrategyTracker(db_session=db_session)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _add_closed(
    service: JournalService,
    symbol: str,
    entry: float,
    exit: float,
    shares: float,
    tags: list[str],
) -> None:
    """Add a trade and immediately close it."""
    e = service.add_trade(
        symbol=symbol, side="long", entry_price=entry, shares=shares, tags=tags
    )
    service.close_trade(e.id, exit_price=exit)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_recompute_with_mixed_results(service, tracker):
    """2 wins and 1 loss — verify all metrics are computed correctly."""
    # Win 1: PnL = (110 - 100) * 10 = 100
    _add_closed(
        service, "AAPL", entry=100.0, exit=110.0, shares=10.0, tags=["momentum"]
    )
    # Win 2: PnL = (120 - 100) * 5 = 100
    _add_closed(service, "GOOG", entry=100.0, exit=120.0, shares=5.0, tags=["momentum"])
    # Loss: PnL = (90 - 100) * 10 = -100
    _add_closed(service, "TSLA", entry=100.0, exit=90.0, shares=10.0, tags=["momentum"])

    perf = tracker.recompute("momentum")

    assert perf.win_count == 2
    assert perf.loss_count == 1
    assert perf.total_pnl == pytest.approx(100.0)  # 100 + 100 - 100
    assert perf.avg_win == pytest.approx(100.0)  # (100 + 100) / 2
    assert perf.avg_loss == pytest.approx(100.0)  # abs(-100 / 1)

    # expectancy = (2/3 * 100) - (1/3 * 100) = 66.67 - 33.33 = 33.33
    assert perf.expectancy == pytest.approx(100.0 / 3, rel=1e-4)

    # profit_factor = 200 / 100 = 2.0
    assert perf.profit_factor == pytest.approx(2.0)


def test_recompute_all_wins(service, tracker):
    _add_closed(service, "AAPL", entry=100.0, exit=110.0, shares=1.0, tags=["breakout"])
    _add_closed(service, "GOOG", entry=200.0, exit=220.0, shares=1.0, tags=["breakout"])

    perf = tracker.recompute("breakout")

    assert perf.win_count == 2
    assert perf.loss_count == 0
    assert perf.total_pnl == pytest.approx(30.0)
    assert perf.profit_factor == float("inf")  # No losses → infinite profit factor
    # expectancy = (1.0 * 15.0) - (0.0 * 0.0) = 15.0
    assert perf.expectancy == pytest.approx(15.0)


def test_recompute_no_trades(tracker):
    perf = tracker.recompute("nonexistent")
    assert perf.win_count == 0
    assert perf.loss_count == 0
    assert perf.total_pnl == pytest.approx(0.0)
    assert perf.expectancy == pytest.approx(0.0)
    assert perf.profit_factor == pytest.approx(0.0)


def test_compare_strategies(service, tracker):
    """Multiple strategies ranked by expectancy descending."""
    # High expectancy strategy: 2 wins, 0 losses
    _add_closed(service, "AAPL", entry=100.0, exit=150.0, shares=1.0, tags=["trend"])
    _add_closed(service, "GOOG", entry=100.0, exit=130.0, shares=1.0, tags=["trend"])

    # Low expectancy strategy: 1 win, 1 big loss
    _add_closed(service, "TSLA", entry=100.0, exit=101.0, shares=1.0, tags=["scalp"])
    _add_closed(service, "MSFT", entry=100.0, exit=50.0, shares=1.0, tags=["scalp"])

    tracker.recompute("trend")
    tracker.recompute("scalp")

    results = tracker.compare_strategies()

    assert len(results) == 2
    assert results[0].strategy_tag == "trend"
    assert results[1].strategy_tag == "scalp"
    assert results[0].expectancy > results[1].expectancy


def test_get_performance_not_found(tracker):
    result = tracker.get_performance("ghost_strategy")
    assert result is None


def test_get_performance_after_recompute(service, tracker):
    _add_closed(service, "AAPL", entry=100.0, exit=110.0, shares=1.0, tags=["value"])
    tracker.recompute("value")

    perf = tracker.get_performance("value")
    assert perf is not None
    assert perf.strategy_tag == "value"
    assert perf.win_count == 1


def test_recompute_excludes_open_trades(service, tracker):
    """Open trades should not affect strategy metrics."""
    # Closed win
    _add_closed(service, "AAPL", entry=100.0, exit=110.0, shares=1.0, tags=["swing"])
    # Open trade (not closed)
    service.add_trade(
        symbol="GOOG", side="long", entry_price=200.0, shares=1.0, tags=["swing"]
    )

    perf = tracker.recompute("swing")

    assert perf.win_count == 1
    assert perf.loss_count == 0


def test_recompute_is_idempotent(service, tracker):
    """Calling recompute twice yields the same result."""
    _add_closed(service, "AAPL", entry=100.0, exit=110.0, shares=1.0, tags=["dip"])

    perf1 = tracker.recompute("dip")
    perf2 = tracker.recompute("dip")

    assert perf1.win_count == perf2.win_count
    assert perf1.total_pnl == pytest.approx(perf2.total_pnl)
