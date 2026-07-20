"""Tests for maverick.portfolio.service.

Uses a tmp SQLite engine (matching `tests/screening/test_service.py`'s
pattern) plus a stub `MarketDataService` with deterministic quotes/frames,
including one symbol that always fails, to exercise the "never fatal"
partial-failure paths.
"""

import math
from datetime import date
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from maverick.market_data.types import Quote
from maverick.platform.config import DatabaseSettings
from maverick.platform.db import create_engine_from_settings
from maverick.portfolio.config import PortfolioSettings
from maverick.portfolio.ledger import position_value
from maverick.portfolio.service import PortfolioService


def _engine(tmp_path):
    settings = DatabaseSettings(
        url=f"sqlite:///{tmp_path}/portfolio.db", use_pooling=True
    )
    return create_engine_from_settings(settings)


def _close_series(
    n: int, base: float = 100.0, trend: float = 0.05, amp: float = 5.0
) -> list[float]:
    return [base + trend * i + amp * math.sin(i / 15) for i in range(n)]


def _ohlcv_frame(
    closes: list[float], volumes: list[float] | None = None
) -> pd.DataFrame:
    n = len(closes)
    index = pd.date_range("2024-01-01", periods=n, freq="B")
    close_arr = np.array(closes, dtype=float)
    volume_arr = (
        np.array(volumes, dtype=float)
        if volumes is not None
        else np.full(n, 1_000_000.0)
    )
    return pd.DataFrame(
        {
            "Open": close_arr - 0.1,
            "High": close_arr + 0.5,
            "Low": close_arr - 0.5,
            "Close": close_arr,
            "Volume": volume_arr,
        },
        index=index,
    )


class StubMarketData:
    """Async fake matching `MarketDataService`'s `get_quote`/`get_price_history` surface."""

    def __init__(
        self,
        quotes: dict[str, float] | None = None,
        frames: dict[str, pd.DataFrame] | None = None,
        quote_errors: set[str] | None = None,
        history_errors: set[str] | None = None,
    ) -> None:
        self._quotes = quotes or {}
        self._frames = frames or {}
        self._quote_errors = quote_errors or set()
        self._history_errors = history_errors or set()
        self.quote_calls: list[str] = []
        self.history_calls: list[tuple[str, date | None, date | None]] = []

    async def get_quote(self, symbol: str) -> Quote:
        self.quote_calls.append(symbol)
        if symbol in self._quote_errors:
            raise RuntimeError(f"quote fetch failed for {symbol}")
        return Quote(
            symbol=symbol,
            price=self._quotes[symbol],
            change=0.0,
            change_percent=0.0,
            volume=1_000_000,
            timestamp="2026-07-19T00:00:00+00:00",
        )

    async def get_price_history(
        self, symbol: str, start: date | None, end: date | None
    ) -> pd.DataFrame:
        self.history_calls.append((symbol, start, end))
        if symbol in self._history_errors:
            raise RuntimeError(f"history fetch failed for {symbol}")
        return self._frames.get(symbol, pd.DataFrame())


def _service(tmp_path, market_data=None, settings=None) -> PortfolioService:
    engine = _engine(tmp_path)
    return PortfolioService(
        engine,
        market_data if market_data is not None else StubMarketData(),
        settings=settings,
    )


# ---------------------------------------------------------------------------
# add_position: averaging is visible through the service (via the ledger)
# ---------------------------------------------------------------------------


async def test_add_position_creates_new_position(tmp_path):
    service = _service(tmp_path)

    position = await service.add_position(
        "default",
        "My Portfolio",
        "aapl",
        Decimal("10"),
        Decimal("100"),
        "2026-01-01",
        "note",
    )

    assert position.ticker == "AAPL"
    assert position.shares == Decimal("10")
    assert position.average_cost_basis == Decimal("100")
    assert position.total_cost == Decimal("1000")
    assert position.notes == "note"


async def test_add_position_second_lot_averages_cost_basis_via_ledger(tmp_path):
    service = _service(tmp_path)
    await service.add_position(
        "default", "My Portfolio", "AAPL", Decimal("10"), Decimal("100"), "2026-01-01"
    )

    position = await service.add_position(
        "default", "My Portfolio", "AAPL", Decimal("10"), Decimal("120"), "2026-01-05"
    )

    assert position.shares == Decimal("20")
    assert position.total_cost == Decimal("2200")
    assert position.average_cost_basis == Decimal("110.0000")


async def test_add_position_second_lot_with_naive_date_against_stored_aware_date(
    tmp_path,
):
    """Regression: the first lot's purchase_date round-trips through
    storage picking up UTC tzinfo; a second, differently-formatted (naive)
    date must not blow up the ledger's earliest-date-wins comparison."""
    service = _service(tmp_path)
    await service.add_position(
        "default", "My Portfolio", "AAPL", Decimal("10"), Decimal("100"), "2026-01-01"
    )

    position = await service.add_position(
        "default", "My Portfolio", "AAPL", Decimal("10"), Decimal("120"), "2026-01-05"
    )

    assert position.shares == Decimal("20")
    # The earliest (normalized-to-UTC) purchase_date must win: the first
    # lot's "2026-01-01" predates the second lot's "2026-01-05".
    assert position.purchase_date == "2026-01-01T00:00:00+00:00"


async def test_add_position_rejects_invalid_ticker(tmp_path):
    service = _service(tmp_path)

    with pytest.raises(ValueError, match="Invalid ticker"):
        await service.add_position(
            "default",
            "My Portfolio",
            "TOOLONGTICKER",
            Decimal("1"),
            Decimal("1"),
            "2026-01-01",
        )


async def test_add_position_rejects_non_positive_shares(tmp_path):
    service = _service(tmp_path)

    with pytest.raises(ValueError, match="Shares must be positive"):
        await service.add_position(
            "default",
            "My Portfolio",
            "AAPL",
            Decimal("0"),
            Decimal("100"),
            "2026-01-01",
        )


async def test_add_position_rejects_shares_over_settings_max(tmp_path):
    service = _service(tmp_path, settings=PortfolioSettings(max_shares=100))

    with pytest.raises(ValueError, match="too large"):
        await service.add_position(
            "default",
            "My Portfolio",
            "AAPL",
            Decimal("101"),
            Decimal("100"),
            "2026-01-01",
        )


async def test_add_position_rejects_price_over_settings_max(tmp_path):
    service = _service(tmp_path, settings=PortfolioSettings(max_price=50))

    with pytest.raises(ValueError, match="too large"):
        await service.add_position(
            "default", "My Portfolio", "AAPL", Decimal("1"), Decimal("51"), "2026-01-01"
        )


async def test_add_position_defaults_purchase_date_to_today(tmp_path):
    service = _service(tmp_path)

    position = await service.add_position(
        "default", "My Portfolio", "AAPL", Decimal("1"), Decimal("100")
    )

    assert position.purchase_date.startswith(date.today().isoformat())


# ---------------------------------------------------------------------------
# remove_position / clear_portfolio: also flow through the ledger
# ---------------------------------------------------------------------------


async def test_remove_position_partial_keeps_basis_via_ledger(tmp_path):
    service = _service(tmp_path)
    await service.add_position(
        "default", "My Portfolio", "AAPL", Decimal("20"), Decimal("160"), "2026-01-01"
    )

    result = await service.remove_position(
        "default", "My Portfolio", "AAPL", Decimal("10")
    )

    assert result.shares_removed == Decimal("10")
    assert result.position_fully_closed is False
    snapshot = await service.get_portfolio("default", "My Portfolio")
    assert snapshot.positions[0].shares == Decimal("10")
    assert snapshot.positions[0].average_cost_basis == Decimal("160")


async def test_remove_position_full_close_removes_row(tmp_path):
    service = _service(tmp_path)
    await service.add_position(
        "default", "My Portfolio", "AAPL", Decimal("10"), Decimal("100"), "2026-01-01"
    )

    result = await service.remove_position("default", "My Portfolio", "AAPL")

    assert result.position_fully_closed is True
    assert result.shares_removed == Decimal("10")
    snapshot = await service.get_portfolio("default", "My Portfolio")
    assert snapshot.positions == []


async def test_remove_position_not_found_raises(tmp_path):
    service = _service(tmp_path)

    with pytest.raises(ValueError, match="not found"):
        await service.remove_position("default", "My Portfolio", "AAPL")


async def test_clear_portfolio_returns_count_and_empties(tmp_path):
    service = _service(tmp_path)
    await service.add_position(
        "default", "My Portfolio", "AAPL", Decimal("1"), Decimal("100"), "2026-01-01"
    )
    await service.add_position(
        "default", "My Portfolio", "MSFT", Decimal("1"), Decimal("100"), "2026-01-01"
    )

    count = await service.clear_portfolio("default", "My Portfolio")

    assert count == 2
    snapshot = await service.get_portfolio("default", "My Portfolio")
    assert snapshot.positions == []


async def test_clear_empty_portfolio_returns_zero(tmp_path):
    service = _service(tmp_path)

    assert await service.clear_portfolio("default", "My Portfolio") == 0


# ---------------------------------------------------------------------------
# get_portfolio: live prices under Semaphore(4), a failed quote never fatal
# ---------------------------------------------------------------------------


async def test_get_portfolio_empty_returns_zero_metrics(tmp_path):
    service = _service(tmp_path)

    snapshot = await service.get_portfolio("default", "My Portfolio")

    assert snapshot.positions == []
    assert snapshot.metrics.position_count == 0
    assert snapshot.metrics.total_invested == Decimal("0")


async def test_get_portfolio_failed_quote_leaves_price_fields_none_but_metrics_fallback(
    tmp_path,
):
    market_data = StubMarketData(quotes={"AAPL": 150.0}, quote_errors={"MSFT"})
    service = _service(tmp_path, market_data=market_data)
    await service.add_position(
        "default", "My Portfolio", "AAPL", Decimal("10"), Decimal("100"), "2026-01-01"
    )
    await service.add_position(
        "default", "My Portfolio", "MSFT", Decimal("5"), Decimal("200"), "2026-01-01"
    )

    snapshot = await service.get_portfolio("default", "My Portfolio")

    by_ticker = {p.ticker: p for p in snapshot.positions}
    aapl = by_ticker["AAPL"]
    assert aapl.current_price == 150.0
    assert aapl.current_value == 1500.0
    assert aapl.unrealized_pnl == 500.0
    assert aapl.unrealized_pnl_percent == 50.0

    msft = by_ticker["MSFT"]
    assert msft.current_price is None
    assert msft.current_value is None
    assert msft.unrealized_pnl is None
    assert msft.unrealized_pnl_percent is None

    # Aggregate metrics still fall back to MSFT's average cost basis (the
    # ledger's built-in fallback), so the whole read never fails.
    assert snapshot.metrics.total_invested == Decimal("2000")
    assert snapshot.metrics.total_value == 2500.0
    assert snapshot.metrics.total_pnl == 500.0
    assert snapshot.metrics.total_pnl_percent == 25.0
    assert snapshot.metrics.position_count == 2
    assert set(market_data.quote_calls) == {"AAPL", "MSFT"}


# ---------------------------------------------------------------------------
# correlation_analysis: auto-fill, calendar pad, correlation_min_rows guard
# ---------------------------------------------------------------------------


async def _seeded_correlation_service(
    tmp_path, n: int = 60, settings=None
) -> PortfolioService:
    base = _close_series(n)
    aapl = base
    msft = [c * 1.5 for c in base]  # perfectly positively correlated returns
    tlt = [400.0 - c for c in base]  # strongly negatively correlated returns
    market_data = StubMarketData(
        frames={
            "AAPL": _ohlcv_frame(aapl),
            "MSFT": _ohlcv_frame(msft),
            "TLT": _ohlcv_frame(tlt),
        }
    )
    return _service(tmp_path, market_data=market_data, settings=settings)


async def test_correlation_analysis_explicit_tickers_finds_high_pair_and_hedge(
    tmp_path,
):
    service = await _seeded_correlation_service(tmp_path)

    result = await service.correlation_analysis(
        "default", "My Portfolio", tickers=["AAPL", "MSFT", "TLT"], days=50
    )

    assert result.portfolio_context is None
    assert result.period_days == 50
    assert result.data_points >= service.settings.correlation_min_rows
    pairs = {tuple(sorted(p["pair"])) for p in result.high_correlation_pairs}
    assert ("AAPL", "MSFT") in pairs
    hedge_pairs = {tuple(sorted(p["pair"])) for p in result.hedges}
    assert ("AAPL", "TLT") in hedge_pairs
    assert result.matrix["AAPL"]["AAPL"] == pytest.approx(1.0)
    assert result.recommendation in {
        "Well diversified",
        "Moderately diversified",
        "Consider adding uncorrelated assets",
    }


async def test_correlation_analysis_autofills_from_portfolio_and_sets_context(tmp_path):
    service = await _seeded_correlation_service(tmp_path)
    await service.add_position(
        "default", "My Portfolio", "AAPL", Decimal("1"), Decimal("100"), "2026-01-01"
    )
    await service.add_position(
        "default", "My Portfolio", "MSFT", Decimal("1"), Decimal("100"), "2026-01-01"
    )

    result = await service.correlation_analysis(
        "default", "My Portfolio", tickers=None, days=50
    )

    assert result.portfolio_context == {
        "using_portfolio": True,
        "portfolio_name": "My Portfolio",
        "position_count": 2,
    }


async def test_correlation_analysis_raises_when_fewer_than_two_holdings(tmp_path):
    service = await _seeded_correlation_service(tmp_path)
    await service.add_position(
        "default", "My Portfolio", "AAPL", Decimal("1"), Decimal("100"), "2026-01-01"
    )

    with pytest.raises(ValueError, match="at least 2 tickers"):
        await service.correlation_analysis(
            "default", "My Portfolio", tickers=None, days=50
        )


async def test_correlation_analysis_raises_when_single_explicit_ticker(tmp_path):
    service = await _seeded_correlation_service(tmp_path)

    with pytest.raises(ValueError, match="At least two tickers"):
        await service.correlation_analysis(
            "default", "My Portfolio", tickers=["AAPL"], days=50
        )


async def test_correlation_analysis_guard_raises_on_insufficient_rows(tmp_path):
    short_frame = _ohlcv_frame(_close_series(10))
    market_data = StubMarketData(
        frames={"AAPL": short_frame, "MSFT": _ohlcv_frame(_close_series(10, base=90.0))}
    )
    service = _service(tmp_path, market_data=market_data)

    with pytest.raises(ValueError, match="Insufficient data points"):
        await service.correlation_analysis(
            "default", "My Portfolio", tickers=["AAPL", "MSFT"], days=10
        )


# ---------------------------------------------------------------------------
# compare_tickers: ranks, portfolio_context
# ---------------------------------------------------------------------------


async def test_compare_tickers_ranks_best_performer_and_strongest_trend(tmp_path):
    n = 60
    winner = [100.0 + i * 2.0 for i in range(n)]  # strong steady uptrend
    loser = [200.0 - i * 1.0 for i in range(n)]  # steady downtrend
    market_data = StubMarketData(
        frames={"WIN": _ohlcv_frame(winner), "LOSE": _ohlcv_frame(loser)}
    )
    service = _service(tmp_path, market_data=market_data)

    result = await service.compare_tickers(
        "default", "My Portfolio", tickers=["WIN", "LOSE"], days=40
    )

    assert result.best_performer == "WIN"
    assert result.strongest_trend == "WIN"
    assert result.portfolio_context is None
    assert result.period_days == 40
    assert result.comparison["WIN"]["rankings"]["performance_rank"] == 1
    assert result.comparison["LOSE"]["rankings"]["performance_rank"] == 2


async def test_compare_tickers_autofills_from_portfolio_and_sets_context(tmp_path):
    n = 60
    market_data = StubMarketData(
        frames={
            "AAPL": _ohlcv_frame(_close_series(n)),
            "MSFT": _ohlcv_frame(_close_series(n, base=200.0)),
        }
    )
    service = _service(tmp_path, market_data=market_data)
    await service.add_position(
        "default", "My Portfolio", "AAPL", Decimal("1"), Decimal("100"), "2026-01-01"
    )
    await service.add_position(
        "default", "My Portfolio", "MSFT", Decimal("1"), Decimal("100"), "2026-01-01"
    )

    result = await service.compare_tickers(
        "default", "My Portfolio", tickers=None, days=40
    )

    assert result.portfolio_context == {
        "using_portfolio": True,
        "portfolio_name": "My Portfolio",
        "position_count": 2,
    }


async def test_compare_tickers_raises_when_single_explicit_ticker(tmp_path):
    service = _service(tmp_path)

    with pytest.raises(ValueError, match="At least two tickers"):
        await service.compare_tickers(
            "default", "My Portfolio", tickers=["AAPL"], days=40
        )


async def test_compare_tickers_raises_on_missing_price_data(tmp_path):
    market_data = StubMarketData(frames={"AAPL": _ohlcv_frame(_close_series(60))})
    service = _service(tmp_path, market_data=market_data)

    with pytest.raises(ValueError, match="Insufficient price data"):
        await service.compare_tickers(
            "default", "My Portfolio", tickers=["AAPL", "NODATA"], days=40
        )


# ---------------------------------------------------------------------------
# risk_adjusted_analysis: ATR sizing + existing-position block via the ledger
# ---------------------------------------------------------------------------


def _risk_frame() -> pd.DataFrame:
    n = 25
    closes = [100.0 + i * (50.0 / (n - 1)) for i in range(n)]
    return _ohlcv_frame(closes)


async def test_risk_adjusted_analysis_basic_fields(tmp_path):
    market_data = StubMarketData(frames={"AAPL": _risk_frame()})
    service = _service(tmp_path, market_data=market_data)

    result = await service.risk_adjusted_analysis(
        "default", "My Portfolio", "aapl", 50.0
    )

    assert result.ticker == "AAPL"
    assert result.current_price == 150.0
    assert result.atr > 0
    assert result.risk_level == 50.0
    assert set(result.position_sizing) == {
        "suggested_position_size",
        "max_shares",
        "position_value",
        "percent_of_portfolio",
    }
    assert set(result.stop_loss) == {
        "stop_loss",
        "stop_loss_percent",
        "max_risk_amount",
    }
    assert set(result.entry_strategy) == {"immediate_entry", "scale_in_levels"}
    assert set(result.targets) == {
        "price_target",
        "profit_potential",
        "risk_reward_ratio",
    }
    assert result.analysis is not None
    assert result.analysis["strategy_type"] == "moderate"
    assert result.existing_position is None


async def test_risk_adjusted_analysis_existing_position_computed_via_ledger(tmp_path):
    market_data = StubMarketData(frames={"AAPL": _risk_frame()})
    service = _service(tmp_path, market_data=market_data)
    await service.add_position(
        "default", "My Portfolio", "AAPL", Decimal("10"), Decimal("100"), "2026-01-01"
    )

    result = await service.risk_adjusted_analysis(
        "default", "My Portfolio", "AAPL", 50.0
    )

    # Cross-check against the ledger directly: the service must not
    # recompute this with float math of its own.
    from maverick.portfolio.types import PositionPayload

    position = PositionPayload(
        ticker="AAPL",
        shares=Decimal("10"),
        average_cost_basis=Decimal("100"),
        total_cost=Decimal("1000"),
        purchase_date="2026-01-01",
    )
    value, pnl, pnl_percent = position_value(
        position, Decimal(str(result.current_price))
    )

    assert result.existing_position is not None
    assert result.existing_position["shares_owned"] == 10.0
    assert result.existing_position["average_cost_basis"] == 100.0
    assert result.existing_position["total_invested"] == 1000.0
    assert result.existing_position["current_value"] == float(value)
    assert result.existing_position["unrealized_pnl"] == float(pnl)
    assert result.existing_position["unrealized_pnl_pct"] == float(pnl_percent)
    assert (
        result.existing_position["position_recommendation"]
        == "Consider taking partial profits"
    )


async def test_risk_adjusted_analysis_raises_on_insufficient_data(tmp_path):
    market_data = StubMarketData(frames={"AAPL": pd.DataFrame()})
    service = _service(tmp_path, market_data=market_data)

    with pytest.raises(ValueError, match="Insufficient data"):
        await service.risk_adjusted_analysis("default", "My Portfolio", "AAPL", 50.0)


async def test_risk_adjusted_analysis_rejects_invalid_ticker(tmp_path):
    service = _service(tmp_path)

    with pytest.raises(ValueError, match="Invalid ticker"):
        await service.risk_adjusted_analysis(
            "default", "My Portfolio", "TOOLONGTICKER", 50.0
        )


# ---------------------------------------------------------------------------
# risk dashboard: get_risk_dashboard / check_position_risk /
# get_regime_adjusted_sizing / get_risk_alerts
# ---------------------------------------------------------------------------


def _spy_frame(n: int = 60, start: float = 100.0) -> pd.DataFrame:
    """A mild uptrend SPY series -- long enough (>= 51 rows) for the regime
    detector's trend factor to engage."""
    return _ohlcv_frame([start + i * 0.5 for i in range(n)])


async def test_get_risk_dashboard_seeded_positions_via_service_reads(tmp_path):
    market_data = StubMarketData(quotes={"AAPL": 150.0, "MSFT": 100.0})
    service = _service(tmp_path, market_data=market_data)
    await service.add_position(
        "default", "My Portfolio", "AAPL", Decimal("10"), Decimal("100"), "2026-01-01"
    )
    await service.add_position(
        "default", "My Portfolio", "MSFT", Decimal("10"), Decimal("100"), "2026-01-01"
    )

    result = await service.get_risk_dashboard("default", "My Portfolio")

    # value = 10*150 + 10*100 = 2500; sector "Unknown" 100% concentrated.
    assert result.total_value == 2500.0
    assert result.sector_concentration == {"Unknown": 1.0}
    assert result.position_count == 2
    assert set(market_data.quote_calls) == {"AAPL", "MSFT"}


async def test_get_risk_dashboard_empty_portfolio_returns_zero_dashboard(tmp_path):
    service = _service(tmp_path)

    result = await service.get_risk_dashboard("default", "My Portfolio")

    assert result.total_value == 0.0
    assert result.position_count == 0


async def test_get_risk_dashboard_failed_quote_falls_back_to_cost_basis(tmp_path):
    market_data = StubMarketData(quotes={}, quote_errors={"AAPL"})
    service = _service(tmp_path, market_data=market_data)
    await service.add_position(
        "default", "My Portfolio", "AAPL", Decimal("10"), Decimal("100"), "2026-01-01"
    )

    result = await service.get_risk_dashboard("default", "My Portfolio")

    # Unlike get_portfolio (None on failure), risk exposures fall back to
    # cost basis -- value = 10 * 100 = 1000, zero P&L.
    assert result.total_value == 1000.0
    assert result.total_pnl == 0.0


async def test_check_position_risk_projects_against_seeded_holdings(tmp_path):
    market_data = StubMarketData(quotes={"AAPL": 150.0})
    service = _service(tmp_path, market_data=market_data)
    await service.add_position(
        "default", "My Portfolio", "AAPL", Decimal("10"), Decimal("100"), "2026-01-01"
    )

    result = await service.check_position_risk(
        "default", "My Portfolio", "msft", 10, 200.0
    )

    assert result.current.total_value == 1500.0
    assert result.projected.total_value == 1500.0 + 2000.0
    assert result.new_position.ticker == "MSFT"


async def test_check_position_risk_rejects_invalid_ticker(tmp_path):
    service = _service(tmp_path)

    with pytest.raises(ValueError, match="Invalid ticker"):
        await service.check_position_risk(
            "default", "My Portfolio", "TOOLONGTICKER", 1, 100.0
        )


async def test_get_regime_adjusted_sizing_detects_regime_from_spy_history(tmp_path):
    market_data = StubMarketData(frames={"SPY": _spy_frame()})
    service = _service(tmp_path, market_data=market_data)

    result = await service.get_regime_adjusted_sizing(100000, 50, 45, 2.0)

    assert result.regime in {"bull", "choppy", "transitional", "bear"}
    assert market_data.history_calls[0][0] == "SPY"
    assert market_data.history_calls[0][1] is not None  # start date always set


async def test_get_regime_adjusted_sizing_falls_back_to_default_regime_on_fetch_failure(
    tmp_path,
):
    market_data = StubMarketData(history_errors={"SPY"})
    service = _service(tmp_path, market_data=market_data)

    result = await service.get_regime_adjusted_sizing(100000, 50, 45, 2.0)

    assert result.regime == service.settings.risk_regime_default_fallback
    assert result.regime_multiplier == 1.0  # bull multiplier


async def test_get_regime_adjusted_sizing_falls_back_when_spy_history_empty(tmp_path):
    market_data = StubMarketData(frames={"SPY": pd.DataFrame()})
    service = _service(tmp_path, market_data=market_data)

    result = await service.get_regime_adjusted_sizing(100000, 50, 45, 2.0)

    assert result.regime == service.settings.risk_regime_default_fallback


async def test_get_risk_alerts_reports_position_count_and_alerts(tmp_path):
    market_data = StubMarketData(quotes={"AAPL": 80.0})
    service = _service(tmp_path, market_data=market_data)
    await service.add_position(
        "default", "My Portfolio", "AAPL", Decimal("10"), Decimal("100"), "2026-01-01"
    )

    result = await service.get_risk_alerts("default", "My Portfolio")

    assert result.position_count == 1
    assert result.alert_count == len(result.alerts)
    assert any(a.alert_type == "drawdown" for a in result.alerts)


async def test_get_risk_alerts_empty_portfolio_reports_zero_position_count(tmp_path):
    service = _service(tmp_path)

    result = await service.get_risk_alerts("default", "My Portfolio")

    assert result.position_count == 0
    assert result.alert_count == 0
    assert result.alerts == []
