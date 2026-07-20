"""Tests for maverick.portfolio.tools."""

from decimal import Decimal
from typing import Any

import pytest
from fastmcp import Client, FastMCP

from maverick.portfolio import tools
from maverick.portfolio.config import PortfolioSettings
from maverick.portfolio.types import (
    ComparisonResult,
    CorrelationResult,
    JournalEntryPayload,
    JournalTradeReview,
    PortfolioMetrics,
    PortfolioSnapshot,
    PositionPayload,
    PositionRiskCheck,
    PositionRiskImpact,
    PositionWithPrice,
    RegimeAdjustedSizing,
    RemoveResult,
    RiskAlert,
    RiskAlertsResult,
    RiskAnalysis,
    RiskDashboard,
    StrategyPerformancePayload,
    WatchlistBrief,
    WatchlistBriefItem,
    WatchlistItemPayload,
    WatchlistPayload,
    WatchlistRemoveResult,
)


def _position_payload(ticker: str = "AAPL") -> PositionPayload:
    return PositionPayload(
        ticker=ticker,
        shares=Decimal("10"),
        average_cost_basis=Decimal("100"),
        total_cost=Decimal("1000"),
        purchase_date="2026-01-01T00:00:00+00:00",
        notes=None,
    )


def _snapshot() -> PortfolioSnapshot:
    position = PositionWithPrice(
        **_position_payload().model_dump(),
        current_price=150.0,
        current_value=1500.0,
        unrealized_pnl=500.0,
        unrealized_pnl_percent=50.0,
    )
    metrics = PortfolioMetrics(
        total_invested=Decimal("1000"),
        total_value=1500.0,
        total_pnl=500.0,
        total_pnl_percent=50.0,
        position_count=1,
    )
    return PortfolioSnapshot(
        user_id="default",
        name="My Portfolio",
        positions=[position],
        metrics=metrics,
        as_of="2026-07-19T00:00:00+00:00",
    )


def _comparison_result() -> ComparisonResult:
    return ComparisonResult(
        comparison={"AAPL": {"current_price": 175.50}},
        best_performer="AAPL",
        strongest_trend="AAPL",
        period_days=90,
        as_of="2026-07-19T00:00:00+00:00",
        portfolio_context=None,
    )


def _correlation_result() -> CorrelationResult:
    return CorrelationResult(
        matrix={"AAPL": {"AAPL": 1.0}},
        high_correlation_pairs=[],
        hedges=[],
        average_correlation=0.1,
        diversification_score=90.0,
        recommendation="Well diversified",
        period_days=252,
        data_points=200,
        portfolio_context=None,
    )


def _risk_result() -> RiskAnalysis:
    return RiskAnalysis(
        ticker="AAPL",
        current_price=175.50,
        atr=3.25,
        risk_level=50.0,
        position_sizing={"suggested_position_size": 500.0},
        stop_loss={"stop_loss": 170.0},
        entry_strategy={"immediate_entry": 175.50},
        targets={"price_target": 190.0},
        analysis={"confidence_score": 35.0},
        existing_position=None,
    )


def _risk_dashboard() -> RiskDashboard:
    return RiskDashboard(
        total_value=2500.0,
        sector_concentration={"Unknown": 1.0},
        max_sector_pct=1.0,
        portfolio_var_95=49.35,
        portfolio_var_99=69.78,
        total_pnl=500.0,
        position_count=1,
    )


def _empty_risk_dashboard() -> RiskDashboard:
    return RiskDashboard(
        total_value=0.0,
        sector_concentration={},
        max_sector_pct=0.0,
        portfolio_var_95=0.0,
        portfolio_var_99=0.0,
        total_pnl=0.0,
        position_count=0,
    )


def _position_risk_check() -> PositionRiskCheck:
    return PositionRiskCheck(
        current=_risk_dashboard(),
        projected=_risk_dashboard(),
        new_position=PositionRiskImpact(
            ticker="MSFT",
            shares=10,
            price=200.0,
            position_value=2000.0,
            pct_of_projected_portfolio=0.5,
        ),
    )


def _regime_adjusted_sizing() -> RegimeAdjustedSizing:
    return RegimeAdjustedSizing(
        shares=400,
        position_value=20000.0,
        risk_amount=2000.0,
        regime_multiplier=1.0,
        adjusted_risk_pct=2.0,
        regime="bull",
    )


def _risk_alerts_result() -> RiskAlertsResult:
    return RiskAlertsResult(
        alert_count=1,
        alerts=[
            RiskAlert(
                alert_type="drawdown",
                severity="warning",
                message="Portfolio is down 20.0% from cost basis (threshold: 10%)",
                details={"loss_pct": 0.2, "total_cost": 1000.0},
            )
        ],
        position_count=1,
    )


def _empty_risk_alerts_result() -> RiskAlertsResult:
    return RiskAlertsResult(alert_count=0, alerts=[], position_count=0)


def _watchlist_payload() -> WatchlistPayload:
    return WatchlistPayload(id=1, name="Tech Movers", description="High-beta names")


def _watchlist_item_payload() -> WatchlistItemPayload:
    return WatchlistItemPayload(
        id=1,
        watchlist_id=1,
        symbol="AAPL",
        added_at="2026-07-19T00:00:00+00:00",
        notes="Watching",
    )


def _watchlist_remove_result() -> WatchlistRemoveResult:
    return WatchlistRemoveResult(watchlist_id=1, symbol="AAPL", removed=True)


def _watchlist_brief() -> WatchlistBrief:
    return WatchlistBrief(
        watchlist_id=1,
        count=1,
        items=[
            WatchlistBriefItem(
                symbol="AAPL",
                days_on_watchlist=5,
                notes="Watching",
                current_price=175.50,
            )
        ],
    )


def _journal_entry_payload(**overrides) -> JournalEntryPayload:
    fields = {
        "id": 1,
        "symbol": "AAPL",
        "side": "long",
        "entry_price": 150.0,
        "exit_price": None,
        "shares": 10.0,
        "entry_date": "2026-07-19T00:00:00+00:00",
        "exit_date": None,
        "rationale": "Momentum breakout",
        "tags": ["momentum"],
        "pnl": None,
        "r_multiple": None,
        "notes": None,
        "status": "open",
    }
    fields.update(overrides)
    return JournalEntryPayload(**fields)


def _journal_trade_review(**overrides) -> JournalTradeReview:
    fields = _journal_entry_payload().model_dump()
    fields["pnl_pct"] = 20.0
    fields.update(overrides)
    return JournalTradeReview(**fields)


def _strategy_performance_payload(**overrides) -> StrategyPerformancePayload:
    fields = {
        "strategy_tag": "momentum",
        "period": "all_time",
        "win_count": 2,
        "loss_count": 1,
        "total_pnl": 250.0,
        "avg_win": 150.0,
        "avg_loss": 50.0,
        "expectancy": 83.33,
        "profit_factor": 6.0,
    }
    fields.update(overrides)
    return StrategyPerformancePayload(**fields)


class StubJournalService:
    """Async fakes matching `JournalService`'s public surface."""

    def __init__(self) -> None:
        self.add_trade_calls: list[tuple] = []
        self.close_trade_calls: list[tuple] = []
        self.list_trades_calls: list[tuple] = []
        self.review_trade_calls: list[int] = []
        self.get_strategy_performance_calls: list[str] = []
        self.compare_strategies_calls = 0

        self.add_trade_result = _journal_entry_payload()
        self.close_trade_result = _journal_entry_payload(
            status="closed", exit_price=180.0, pnl=300.0
        )
        self.list_trades_result = [_journal_entry_payload()]
        self.review_trade_result: JournalTradeReview | None = _journal_trade_review()
        self.get_strategy_performance_result: StrategyPerformancePayload | None = (
            _strategy_performance_payload()
        )
        self.compare_strategies_result = [_strategy_performance_payload()]

        self.raise_on_add_trade: Exception | None = None
        self.raise_on_close_trade: Exception | None = None
        self.raise_on_list_trades: Exception | None = None
        self.raise_on_review_trade: Exception | None = None
        self.raise_on_get_strategy_performance: Exception | None = None
        self.raise_on_compare_strategies: Exception | None = None

    async def add_trade(
        self, symbol, side, entry_price, shares, rationale=None, tags=None, notes=None
    ) -> JournalEntryPayload:
        self.add_trade_calls.append(
            (symbol, side, entry_price, shares, rationale, tags, notes)
        )
        if self.raise_on_add_trade is not None:
            raise self.raise_on_add_trade
        return self.add_trade_result

    async def close_trade(
        self, entry_id, exit_price, notes=None
    ) -> JournalEntryPayload:
        self.close_trade_calls.append((entry_id, exit_price, notes))
        if self.raise_on_close_trade is not None:
            raise self.raise_on_close_trade
        return self.close_trade_result

    async def list_trades(
        self, symbol=None, status=None, strategy_tag=None, limit=50
    ) -> list[JournalEntryPayload]:
        self.list_trades_calls.append((symbol, status, strategy_tag, limit))
        if self.raise_on_list_trades is not None:
            raise self.raise_on_list_trades
        return self.list_trades_result

    async def review_trade(self, entry_id) -> JournalTradeReview | None:
        self.review_trade_calls.append(entry_id)
        if self.raise_on_review_trade is not None:
            raise self.raise_on_review_trade
        return self.review_trade_result

    async def get_strategy_performance(
        self, strategy_tag
    ) -> StrategyPerformancePayload | None:
        self.get_strategy_performance_calls.append(strategy_tag)
        if self.raise_on_get_strategy_performance is not None:
            raise self.raise_on_get_strategy_performance
        return self.get_strategy_performance_result

    async def compare_strategies(self) -> list[StrategyPerformancePayload]:
        self.compare_strategies_calls += 1
        if self.raise_on_compare_strategies is not None:
            raise self.raise_on_compare_strategies
        return self.compare_strategies_result


class StubService:
    """Async fakes matching `PortfolioService`'s public surface."""

    def __init__(self) -> None:
        self.settings = PortfolioSettings()
        self.add_calls: list[tuple] = []
        self.get_portfolio_calls: list[tuple[str, str]] = []
        self.remove_calls: list[tuple] = []
        self.clear_calls: list[tuple[str, str]] = []
        self.risk_calls: list[tuple] = []
        self.compare_calls: list[tuple] = []
        self.correlation_calls: list[tuple] = []
        self.risk_dashboard_calls: list[tuple] = []
        self.check_position_risk_calls: list[tuple] = []
        self.regime_adjusted_sizing_calls: list[tuple] = []
        self.risk_alerts_calls: list[tuple] = []
        self.create_watchlist_calls: list[tuple] = []
        self.add_watchlist_item_calls: list[tuple] = []
        self.remove_watchlist_item_calls: list[tuple] = []
        self.watchlist_brief_calls: list[int] = []

        self.add_result = _position_payload()
        self.get_portfolio_result = _snapshot()
        self.remove_result = RemoveResult(
            ticker="AAPL", shares_removed=Decimal("5"), position_fully_closed=False
        )
        self.clear_result = 3
        self.risk_result = _risk_result()
        self.compare_result = _comparison_result()
        self.correlation_result = _correlation_result()
        self.risk_dashboard_result = _risk_dashboard()
        self.check_position_risk_result = _position_risk_check()
        self.regime_adjusted_sizing_result = _regime_adjusted_sizing()
        self.risk_alerts_result = _risk_alerts_result()
        self.create_watchlist_result = _watchlist_payload()
        self.add_watchlist_item_result = _watchlist_item_payload()
        self.remove_watchlist_item_result = _watchlist_remove_result()
        self.watchlist_brief_result = _watchlist_brief()

        self.raise_on_add: Exception | None = None
        self.raise_on_get_portfolio: Exception | None = None
        self.raise_on_remove: Exception | None = None
        self.raise_on_clear: Exception | None = None
        self.raise_on_risk: Exception | None = None
        self.raise_on_compare: Exception | None = None
        self.raise_on_correlation: Exception | None = None
        self.raise_on_risk_dashboard: Exception | None = None
        self.raise_on_check_position_risk: Exception | None = None
        self.raise_on_regime_adjusted_sizing: Exception | None = None
        self.raise_on_risk_alerts: Exception | None = None
        self.raise_on_create_watchlist: Exception | None = None
        self.raise_on_add_watchlist_item: Exception | None = None
        self.raise_on_remove_watchlist_item: Exception | None = None
        self.raise_on_watchlist_brief: Exception | None = None

    async def add_position(
        self, user_id, portfolio_name, ticker, shares, price, purchase_date, notes
    ) -> PositionPayload:
        self.add_calls.append(
            (user_id, portfolio_name, ticker, shares, price, purchase_date, notes)
        )
        if self.raise_on_add is not None:
            raise self.raise_on_add
        return self.add_result

    async def get_portfolio(self, user_id, portfolio_name) -> PortfolioSnapshot:
        self.get_portfolio_calls.append((user_id, portfolio_name))
        if self.raise_on_get_portfolio is not None:
            raise self.raise_on_get_portfolio
        return self.get_portfolio_result

    async def remove_position(
        self, user_id, portfolio_name, ticker, shares
    ) -> RemoveResult:
        self.remove_calls.append((user_id, portfolio_name, ticker, shares))
        if self.raise_on_remove is not None:
            raise self.raise_on_remove
        return self.remove_result

    async def clear_portfolio(self, user_id, portfolio_name) -> int:
        self.clear_calls.append((user_id, portfolio_name))
        if self.raise_on_clear is not None:
            raise self.raise_on_clear
        return self.clear_result

    async def risk_adjusted_analysis(
        self, user_id, portfolio_name, ticker, risk_level
    ) -> RiskAnalysis:
        self.risk_calls.append((user_id, portfolio_name, ticker, risk_level))
        if self.raise_on_risk is not None:
            raise self.raise_on_risk
        return self.risk_result

    async def compare_tickers(
        self, user_id, portfolio_name, tickers, days
    ) -> ComparisonResult:
        self.compare_calls.append((user_id, portfolio_name, tickers, days))
        if self.raise_on_compare is not None:
            raise self.raise_on_compare
        return self.compare_result

    async def correlation_analysis(
        self, user_id, portfolio_name, tickers, days
    ) -> CorrelationResult:
        self.correlation_calls.append((user_id, portfolio_name, tickers, days))
        if self.raise_on_correlation is not None:
            raise self.raise_on_correlation
        return self.correlation_result

    async def get_risk_dashboard(self, user_id, portfolio_name) -> RiskDashboard:
        self.risk_dashboard_calls.append((user_id, portfolio_name))
        if self.raise_on_risk_dashboard is not None:
            raise self.raise_on_risk_dashboard
        return self.risk_dashboard_result

    async def check_position_risk(
        self, user_id, portfolio_name, ticker, shares, entry_price
    ) -> PositionRiskCheck:
        self.check_position_risk_calls.append(
            (user_id, portfolio_name, ticker, shares, entry_price)
        )
        if self.raise_on_check_position_risk is not None:
            raise self.raise_on_check_position_risk
        return self.check_position_risk_result

    async def get_regime_adjusted_sizing(
        self, account_size, entry_price, stop_loss, risk_pct
    ) -> RegimeAdjustedSizing:
        self.regime_adjusted_sizing_calls.append(
            (account_size, entry_price, stop_loss, risk_pct)
        )
        if self.raise_on_regime_adjusted_sizing is not None:
            raise self.raise_on_regime_adjusted_sizing
        return self.regime_adjusted_sizing_result

    async def get_risk_alerts(self, user_id, portfolio_name) -> RiskAlertsResult:
        self.risk_alerts_calls.append((user_id, portfolio_name))
        if self.raise_on_risk_alerts is not None:
            raise self.raise_on_risk_alerts
        return self.risk_alerts_result

    async def create_watchlist(self, name, description=None) -> WatchlistPayload:
        self.create_watchlist_calls.append((name, description))
        if self.raise_on_create_watchlist is not None:
            raise self.raise_on_create_watchlist
        return self.create_watchlist_result

    async def add_watchlist_item(
        self, watchlist_id, symbol, notes=None
    ) -> WatchlistItemPayload:
        self.add_watchlist_item_calls.append((watchlist_id, symbol, notes))
        if self.raise_on_add_watchlist_item is not None:
            raise self.raise_on_add_watchlist_item
        return self.add_watchlist_item_result

    async def remove_watchlist_item(
        self, watchlist_id, symbol
    ) -> WatchlistRemoveResult:
        self.remove_watchlist_item_calls.append((watchlist_id, symbol))
        if self.raise_on_remove_watchlist_item is not None:
            raise self.raise_on_remove_watchlist_item
        return self.remove_watchlist_item_result

    async def watchlist_brief(self, watchlist_id) -> WatchlistBrief:
        self.watchlist_brief_calls.append(watchlist_id)
        if self.raise_on_watchlist_brief is not None:
            raise self.raise_on_watchlist_brief
        return self.watchlist_brief_result


@pytest.fixture
def stub_service() -> Any:
    stub = StubService()
    tools.configure(stub)
    yield stub


@pytest.fixture
def stub_journal_service(stub_service) -> Any:
    stub = StubJournalService()
    tools.configure(stub_service, stub)
    yield stub


# ---------------------------------------------------------------------------
# unconfigured service
# ---------------------------------------------------------------------------


async def test_unconfigured_service_returns_configure_error_payload():
    tools.configure(None)  # type: ignore[arg-type]

    result = await tools.portfolio_get_my_portfolio()

    assert result == {
        "status": "error",
        "error": "portfolio.tools: configure(service) was not called",
    }


# ---------------------------------------------------------------------------
# portfolio_add_position
# ---------------------------------------------------------------------------


async def test_add_position_str_mediated_decimal_ingress(stub_service):
    result = await tools.portfolio_add_position(
        ticker="aapl",
        shares=10.5,
        purchase_price=150.25,
        purchase_date="2026-01-01",
        notes="note",
    )

    assert result["status"] == "success"
    assert result["position"]["ticker"] == "AAPL"
    call = stub_service.add_calls[0]
    assert call[0] == "default"
    assert call[1] == "My Portfolio"
    assert call[2] == "aapl"
    assert call[3] == Decimal("10.5")
    assert isinstance(call[3], Decimal)
    assert call[4] == Decimal("150.25")
    assert isinstance(call[4], Decimal)


async def test_add_position_service_exception_returns_error_payload(stub_service):
    stub_service.raise_on_add = ValueError("Invalid ticker")

    result = await tools.portfolio_add_position(
        ticker="TOOLONG", shares=1, purchase_price=1
    )

    assert result == {"status": "error", "error": "Invalid ticker"}


# ---------------------------------------------------------------------------
# portfolio_get_my_portfolio
# ---------------------------------------------------------------------------


async def test_get_my_portfolio_returns_model_dump_plus_status(stub_service):
    result = await tools.portfolio_get_my_portfolio(portfolio_name="Trading")

    assert result["status"] == "success"
    assert result["positions"][0]["ticker"] == "AAPL"
    assert result["metrics"]["total_invested"] == "1000"
    assert stub_service.get_portfolio_calls == [("default", "Trading")]


# ---------------------------------------------------------------------------
# portfolio_remove_position
# ---------------------------------------------------------------------------


async def test_remove_position_full_close_when_shares_omitted(stub_service):
    result = await tools.portfolio_remove_position(ticker="AAPL")

    assert result["status"] == "success"
    assert stub_service.remove_calls == [("default", "My Portfolio", "AAPL", None)]


async def test_remove_position_partial_shares_decimal_ingress(stub_service):
    await tools.portfolio_remove_position(ticker="AAPL", shares=5.0)

    call = stub_service.remove_calls[0]
    assert call[3] == Decimal("5.0")
    assert isinstance(call[3], Decimal)


async def test_remove_position_service_exception_returns_error_payload(stub_service):
    stub_service.raise_on_remove = ValueError("not found")

    result = await tools.portfolio_remove_position(ticker="AAPL")

    assert result == {"status": "error", "error": "not found"}


# ---------------------------------------------------------------------------
# portfolio_clear_portfolio: confirm=True required
# ---------------------------------------------------------------------------


async def test_clear_portfolio_rejects_without_confirm(stub_service):
    result = await tools.portfolio_clear_portfolio(confirm=False)

    assert result == {
        "status": "error",
        "error": "Must set confirm=True to clear portfolio",
    }
    assert stub_service.clear_calls == []


async def test_clear_portfolio_confirm_true_calls_service(stub_service):
    result = await tools.portfolio_clear_portfolio(confirm=True)

    assert result == {"status": "success", "positions_cleared": 3}
    assert stub_service.clear_calls == [("default", "My Portfolio")]


async def test_clear_portfolio_service_exception_returns_error_payload(stub_service):
    stub_service.raise_on_clear = RuntimeError("boom")

    result = await tools.portfolio_clear_portfolio(confirm=True)

    assert result == {"status": "error", "error": "boom"}


# ---------------------------------------------------------------------------
# portfolio_risk_adjusted_analysis
# ---------------------------------------------------------------------------


async def test_risk_adjusted_analysis_returns_model_dump_plus_status(stub_service):
    result = await tools.portfolio_risk_adjusted_analysis(
        ticker="AAPL", risk_level=75.0
    )

    assert result["status"] == "success"
    assert result["ticker"] == "AAPL"
    assert result["risk_level"] == 50.0  # from the stub's fixed result payload
    assert stub_service.risk_calls == [("default", "My Portfolio", "AAPL", 75.0)]


async def test_risk_adjusted_analysis_service_exception_returns_error_payload(
    stub_service,
):
    stub_service.raise_on_risk = ValueError("Insufficient data")

    result = await tools.portfolio_risk_adjusted_analysis(ticker="AAPL")

    assert result == {"status": "error", "error": "Insufficient data"}


# ---------------------------------------------------------------------------
# portfolio_compare_tickers
# ---------------------------------------------------------------------------


async def test_compare_tickers_returns_model_dump_plus_status(stub_service):
    result = await tools.portfolio_compare_tickers(tickers=["AAPL", "MSFT"], days=30)

    assert result["status"] == "success"
    assert result["best_performer"] == "AAPL"
    assert stub_service.compare_calls == [
        ("default", "My Portfolio", ["AAPL", "MSFT"], 30)
    ]


async def test_compare_tickers_defaults_to_none_tickers_and_days(stub_service):
    await tools.portfolio_compare_tickers()

    assert stub_service.compare_calls == [("default", "My Portfolio", None, None)]


async def test_compare_tickers_service_exception_returns_error_payload(stub_service):
    stub_service.raise_on_compare = ValueError("At least two tickers are required")

    result = await tools.portfolio_compare_tickers(tickers=["AAPL"])

    assert result == {"status": "error", "error": "At least two tickers are required"}


# ---------------------------------------------------------------------------
# portfolio_correlation_analysis
# ---------------------------------------------------------------------------


async def test_correlation_analysis_returns_model_dump_plus_status(stub_service):
    result = await tools.portfolio_correlation_analysis(
        tickers=["AAPL", "MSFT"], days=100
    )

    assert result["status"] == "success"
    assert result["recommendation"] == "Well diversified"
    assert stub_service.correlation_calls == [
        ("default", "My Portfolio", ["AAPL", "MSFT"], 100)
    ]


async def test_correlation_analysis_service_exception_returns_error_payload(
    stub_service,
):
    stub_service.raise_on_correlation = ValueError("Insufficient data points")

    result = await tools.portfolio_correlation_analysis(tickers=["AAPL", "MSFT"])

    assert result == {"status": "error", "error": "Insufficient data points"}


# ---------------------------------------------------------------------------
# portfolio_get_risk_dashboard
# ---------------------------------------------------------------------------


async def test_get_risk_dashboard_returns_model_dump_plus_status(stub_service):
    result = await tools.portfolio_get_risk_dashboard(portfolio_name="Trading")

    assert result["status"] == "success"
    assert result["total_value"] == 2500.0
    assert result["portfolio_name"] == "Trading"
    assert stub_service.risk_dashboard_calls == [("default", "Trading")]


async def test_get_risk_dashboard_empty_positions_returns_empty_status(stub_service):
    stub_service.risk_dashboard_result = _empty_risk_dashboard()

    result = await tools.portfolio_get_risk_dashboard(portfolio_name="Empty")

    assert result == {
        "status": "empty",
        "message": "No positions found in portfolio 'Empty'",
        "portfolio_name": "Empty",
    }


async def test_get_risk_dashboard_service_exception_returns_error_payload(
    stub_service,
):
    stub_service.raise_on_risk_dashboard = ValueError("boom")

    result = await tools.portfolio_get_risk_dashboard()

    assert result == {"status": "error", "error": "boom"}


# ---------------------------------------------------------------------------
# portfolio_check_position_risk
# ---------------------------------------------------------------------------


async def test_check_position_risk_returns_model_dump_plus_status(stub_service):
    result = await tools.portfolio_check_position_risk(
        ticker="msft", shares=10, entry_price=200.0
    )

    assert result["status"] == "success"
    assert result["new_position"]["ticker"] == "MSFT"
    assert result["portfolio_name"] == "My Portfolio"
    assert stub_service.check_position_risk_calls == [
        ("default", "My Portfolio", "msft", 10, 200.0)
    ]


async def test_check_position_risk_service_exception_returns_error_payload(
    stub_service,
):
    stub_service.raise_on_check_position_risk = ValueError("Invalid ticker")

    result = await tools.portfolio_check_position_risk(
        ticker="TOOLONG", shares=1, entry_price=1.0
    )

    assert result == {"status": "error", "error": "Invalid ticker"}


# ---------------------------------------------------------------------------
# portfolio_get_regime_adjusted_sizing
# ---------------------------------------------------------------------------


async def test_get_regime_adjusted_sizing_returns_model_dump_plus_status(
    stub_service,
):
    result = await tools.portfolio_get_regime_adjusted_sizing(
        account_size=100000, entry_price=50, stop_loss=45, risk_pct=2.0
    )

    assert result["status"] == "success"
    assert result["shares"] == 400
    assert result["regime"] == "bull"
    assert stub_service.regime_adjusted_sizing_calls == [(100000, 50, 45, 2.0)]


async def test_get_regime_adjusted_sizing_defaults_risk_pct_to_two(stub_service):
    await tools.portfolio_get_regime_adjusted_sizing(
        account_size=100000, entry_price=50, stop_loss=45
    )

    assert stub_service.regime_adjusted_sizing_calls == [(100000, 50, 45, 2.0)]


async def test_get_regime_adjusted_sizing_service_exception_returns_error_payload(
    stub_service,
):
    stub_service.raise_on_regime_adjusted_sizing = RuntimeError("boom")

    result = await tools.portfolio_get_regime_adjusted_sizing(
        account_size=100000, entry_price=50, stop_loss=45
    )

    assert result == {"status": "error", "error": "boom"}


# ---------------------------------------------------------------------------
# portfolio_get_risk_alerts
# ---------------------------------------------------------------------------


async def test_get_risk_alerts_returns_alerts_plus_status(stub_service):
    result = await tools.portfolio_get_risk_alerts(portfolio_name="Trading")

    assert result["status"] == "success"
    assert result["portfolio_name"] == "Trading"
    assert result["alert_count"] == 1
    assert result["alerts"][0]["alert_type"] == "drawdown"
    assert stub_service.risk_alerts_calls == [("default", "Trading")]


async def test_get_risk_alerts_empty_positions_returns_empty_status(stub_service):
    stub_service.risk_alerts_result = _empty_risk_alerts_result()

    result = await tools.portfolio_get_risk_alerts(portfolio_name="Empty")

    assert result == {
        "status": "empty",
        "message": "No positions found in portfolio 'Empty'",
        "portfolio_name": "Empty",
        "alerts": [],
    }


async def test_get_risk_alerts_service_exception_returns_error_payload(stub_service):
    stub_service.raise_on_risk_alerts = ValueError("boom")

    result = await tools.portfolio_get_risk_alerts()

    assert result == {"status": "error", "error": "boom"}


# ---------------------------------------------------------------------------
# portfolio_watchlist_create
# ---------------------------------------------------------------------------


async def test_watchlist_create_returns_model_dump_plus_status(stub_service):
    result = await tools.portfolio_watchlist_create(
        name="Tech Movers", description="High-beta names"
    )

    assert result["status"] == "success"
    assert result["name"] == "Tech Movers"
    assert stub_service.create_watchlist_calls == [("Tech Movers", "High-beta names")]


async def test_watchlist_create_defaults_description_to_none(stub_service):
    await tools.portfolio_watchlist_create(name="Tech Movers")

    assert stub_service.create_watchlist_calls == [("Tech Movers", None)]


async def test_watchlist_create_service_exception_returns_error_payload(stub_service):
    stub_service.raise_on_create_watchlist = ValueError("Watchlist name already exists")

    result = await tools.portfolio_watchlist_create(name="Dup")

    assert result == {"status": "error", "error": "Watchlist name already exists"}


# ---------------------------------------------------------------------------
# portfolio_watchlist_add
# ---------------------------------------------------------------------------


async def test_watchlist_add_returns_model_dump_plus_status(stub_service):
    result = await tools.portfolio_watchlist_add(
        watchlist_id=1, symbol="aapl", notes="Watching"
    )

    assert result["status"] == "success"
    assert result["symbol"] == "AAPL"
    assert stub_service.add_watchlist_item_calls == [(1, "aapl", "Watching")]


async def test_watchlist_add_defaults_notes_to_none(stub_service):
    await tools.portfolio_watchlist_add(watchlist_id=1, symbol="AAPL")

    assert stub_service.add_watchlist_item_calls == [(1, "AAPL", None)]


async def test_watchlist_add_service_exception_returns_error_payload(stub_service):
    stub_service.raise_on_add_watchlist_item = ValueError("boom")

    result = await tools.portfolio_watchlist_add(watchlist_id=1, symbol="AAPL")

    assert result == {"status": "error", "error": "boom"}


# ---------------------------------------------------------------------------
# portfolio_watchlist_remove
# ---------------------------------------------------------------------------


async def test_watchlist_remove_returns_model_dump_plus_status(stub_service):
    result = await tools.portfolio_watchlist_remove(watchlist_id=1, symbol="aapl")

    assert result["status"] == "success"
    assert result["removed"] is True
    assert stub_service.remove_watchlist_item_calls == [(1, "aapl")]


async def test_watchlist_remove_reports_removed_false(stub_service):
    stub_service.remove_watchlist_item_result = WatchlistRemoveResult(
        watchlist_id=1, symbol="AAPL", removed=False
    )

    result = await tools.portfolio_watchlist_remove(watchlist_id=1, symbol="AAPL")

    assert result["status"] == "success"
    assert result["removed"] is False


async def test_watchlist_remove_service_exception_returns_error_payload(stub_service):
    stub_service.raise_on_remove_watchlist_item = ValueError("boom")

    result = await tools.portfolio_watchlist_remove(watchlist_id=1, symbol="AAPL")

    assert result == {"status": "error", "error": "boom"}


# ---------------------------------------------------------------------------
# portfolio_watchlist_brief
# ---------------------------------------------------------------------------


async def test_watchlist_brief_returns_model_dump_plus_status(stub_service):
    result = await tools.portfolio_watchlist_brief(watchlist_id=1)

    assert result["status"] == "success"
    assert result["count"] == 1
    assert result["items"][0]["symbol"] == "AAPL"
    assert result["items"][0]["current_price"] == 175.50
    assert stub_service.watchlist_brief_calls == [1]


async def test_watchlist_brief_service_exception_returns_error_payload(stub_service):
    stub_service.raise_on_watchlist_brief = ValueError("boom")

    result = await tools.portfolio_watchlist_brief(watchlist_id=1)

    assert result == {"status": "error", "error": "boom"}


# ---------------------------------------------------------------------------
# unconfigured journal service
# ---------------------------------------------------------------------------


async def test_unconfigured_journal_service_returns_configure_error_payload(
    stub_service,
):
    """`stub_service` configures the portfolio service but leaves the
    journal service unset -- journal tools must fail honestly rather than
    raising `AttributeError`/`TypeError` from a `None` service."""
    result = await tools.portfolio_journal_list_trades()

    assert result == {
        "status": "error",
        "error": (
            "portfolio.tools: configure(service, journal_service) was not "
            "called with a journal_service"
        ),
    }


# ---------------------------------------------------------------------------
# portfolio_journal_add_trade
# ---------------------------------------------------------------------------


async def test_journal_add_trade_str_mediated_decimal_ingress(stub_journal_service):
    result = await tools.portfolio_journal_add_trade(
        symbol="aapl",
        side="long",
        entry_price=150.25,
        shares=10.5,
        rationale="Momentum breakout",
        tags=["momentum"],
        notes="note",
    )

    assert result["status"] == "success"
    assert result["symbol"] == "AAPL"
    call = stub_journal_service.add_trade_calls[0]
    assert call[0] == "aapl"
    assert call[1] == "long"
    assert call[2] == Decimal("150.25")
    assert isinstance(call[2], Decimal)
    assert call[3] == Decimal("10.5")
    assert isinstance(call[3], Decimal)
    assert call[4] == "Momentum breakout"
    assert call[5] == ["momentum"]
    assert call[6] == "note"


async def test_journal_add_trade_service_exception_returns_error_payload(
    stub_journal_service,
):
    stub_journal_service.raise_on_add_trade = ValueError("boom")

    result = await tools.portfolio_journal_add_trade(
        symbol="AAPL", side="long", entry_price=150.0, shares=10.0
    )

    assert result == {"status": "error", "error": "boom"}


# ---------------------------------------------------------------------------
# portfolio_journal_close_trade
# ---------------------------------------------------------------------------


async def test_journal_close_trade_str_mediated_decimal_ingress(stub_journal_service):
    result = await tools.portfolio_journal_close_trade(
        entry_id=1, exit_price=180.0, notes="closed on target"
    )

    assert result["status"] == "success"
    assert result["status"] == "success"
    assert result["pnl"] == 300.0
    call = stub_journal_service.close_trade_calls[0]
    assert call[0] == 1
    assert call[1] == Decimal("180.0")
    assert isinstance(call[1], Decimal)
    assert call[2] == "closed on target"


async def test_journal_close_trade_service_exception_returns_error_payload(
    stub_journal_service,
):
    stub_journal_service.raise_on_close_trade = ValueError("already closed")

    result = await tools.portfolio_journal_close_trade(entry_id=1, exit_price=180.0)

    assert result == {"status": "error", "error": "already closed"}


# ---------------------------------------------------------------------------
# portfolio_journal_list_trades
# ---------------------------------------------------------------------------


async def test_journal_list_trades_returns_trades_plus_count(stub_journal_service):
    result = await tools.portfolio_journal_list_trades(symbol="AAPL", status="open")

    assert result["status"] == "success"
    assert result["count"] == 1
    assert result["trades"][0]["symbol"] == "AAPL"
    assert stub_journal_service.list_trades_calls == [("AAPL", "open", None, 50)]


async def test_journal_list_trades_service_exception_returns_error_payload(
    stub_journal_service,
):
    stub_journal_service.raise_on_list_trades = ValueError("boom")

    result = await tools.portfolio_journal_list_trades()

    assert result == {"status": "error", "error": "boom"}


# ---------------------------------------------------------------------------
# portfolio_journal_review
# ---------------------------------------------------------------------------


async def test_journal_review_returns_model_dump_plus_status(stub_journal_service):
    result = await tools.portfolio_journal_review(entry_id=1)

    assert result["status"] == "success"
    assert result["found"] is True
    assert result["pnl_pct"] == 20.0
    assert stub_journal_service.review_trade_calls == [1]


async def test_journal_review_returns_found_false_for_unknown_id(stub_journal_service):
    stub_journal_service.review_trade_result = None

    result = await tools.portfolio_journal_review(entry_id=999)

    assert result == {"status": "success", "found": False, "entry_id": 999}


async def test_journal_review_service_exception_returns_error_payload(
    stub_journal_service,
):
    stub_journal_service.raise_on_review_trade = ValueError("boom")

    result = await tools.portfolio_journal_review(entry_id=1)

    assert result == {"status": "error", "error": "boom"}


# ---------------------------------------------------------------------------
# portfolio_get_strategy_performance (single + comparison consolidation)
# ---------------------------------------------------------------------------


async def test_get_strategy_performance_single_mode_returns_model_dump(
    stub_journal_service,
):
    result = await tools.portfolio_get_strategy_performance(strategy_tag="momentum")

    assert result["status"] == "success"
    assert result["mode"] == "single"
    assert result["found"] is True
    assert result["total_trades"] == 3
    assert result["expectancy"] == 83.33
    assert stub_journal_service.get_strategy_performance_calls == ["momentum"]
    assert stub_journal_service.compare_strategies_calls == 0


async def test_get_strategy_performance_single_mode_requires_strategy_tag(
    stub_journal_service,
):
    result = await tools.portfolio_get_strategy_performance()

    assert result == {
        "status": "error",
        "error": "strategy_tag is required when compare is False",
    }


async def test_get_strategy_performance_single_mode_not_found(stub_journal_service):
    stub_journal_service.get_strategy_performance_result = None

    result = await tools.portfolio_get_strategy_performance(strategy_tag="unknown")

    assert result == {
        "status": "success",
        "mode": "single",
        "found": False,
        "strategy_tag": "unknown",
        "message": "No performance data found. Close some trades with this tag first.",
    }


async def test_get_strategy_performance_comparison_mode_returns_ranked_list(
    stub_journal_service,
):
    result = await tools.portfolio_get_strategy_performance(compare=True)

    assert result["status"] == "success"
    assert result["mode"] == "comparison"
    assert result["count"] == 1
    assert result["strategies"][0]["rank"] == 1
    assert result["strategies"][0]["strategy_tag"] == "momentum"
    assert stub_journal_service.compare_strategies_calls == 1
    assert stub_journal_service.get_strategy_performance_calls == []


async def test_get_strategy_performance_comparison_mode_ignores_strategy_tag(
    stub_journal_service,
):
    await tools.portfolio_get_strategy_performance(strategy_tag="ignored", compare=True)

    assert stub_journal_service.get_strategy_performance_calls == []


async def test_get_strategy_performance_service_exception_returns_error_payload(
    stub_journal_service,
):
    stub_journal_service.raise_on_get_strategy_performance = ValueError("boom")

    result = await tools.portfolio_get_strategy_performance(strategy_tag="momentum")

    assert result == {"status": "error", "error": "boom"}


# ---------------------------------------------------------------------------
# register: twenty tools + resource, honest annotations
# ---------------------------------------------------------------------------


_EXPECTED_TOOL_NAMES = {
    "portfolio_add_position",
    "portfolio_get_my_portfolio",
    "portfolio_remove_position",
    "portfolio_clear_portfolio",
    "portfolio_risk_adjusted_analysis",
    "portfolio_compare_tickers",
    "portfolio_correlation_analysis",
    "portfolio_get_risk_dashboard",
    "portfolio_check_position_risk",
    "portfolio_get_regime_adjusted_sizing",
    "portfolio_get_risk_alerts",
    "portfolio_watchlist_create",
    "portfolio_watchlist_add",
    "portfolio_watchlist_remove",
    "portfolio_watchlist_brief",
    "portfolio_journal_add_trade",
    "portfolio_journal_close_trade",
    "portfolio_journal_list_trades",
    "portfolio_journal_review",
    "portfolio_get_strategy_performance",
}

_READ_ONLY_NAMES = {
    "portfolio_get_my_portfolio",
    "portfolio_risk_adjusted_analysis",
    "portfolio_compare_tickers",
    "portfolio_correlation_analysis",
    "portfolio_get_risk_dashboard",
    "portfolio_check_position_risk",
    "portfolio_get_regime_adjusted_sizing",
    "portfolio_get_risk_alerts",
    "portfolio_watchlist_brief",
    "portfolio_journal_list_trades",
    "portfolio_journal_review",
    "portfolio_get_strategy_performance",
}


async def test_register_attaches_twenty_tools(stub_service):
    mcp = FastMCP("test")
    tools.register(mcp)

    registered = await mcp.list_tools()

    assert {tool.name for tool in registered} == _EXPECTED_TOOL_NAMES


async def test_register_marks_reads_read_only(stub_service):
    mcp = FastMCP("test")
    tools.register(mcp)

    for name in _READ_ONLY_NAMES:
        tool = await mcp.get_tool(name)
        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is True


async def test_register_marks_add_honestly(stub_service):
    mcp = FastMCP("test")
    tools.register(mcp)

    tool = await mcp.get_tool("portfolio_add_position")

    assert tool.annotations is not None
    assert tool.annotations.readOnlyHint is False
    assert tool.annotations.destructiveHint is False
    assert tool.annotations.idempotentHint is False


async def test_register_marks_remove_honestly(stub_service):
    mcp = FastMCP("test")
    tools.register(mcp)

    tool = await mcp.get_tool("portfolio_remove_position")

    assert tool.annotations is not None
    assert tool.annotations.readOnlyHint is False
    assert tool.annotations.destructiveHint is True
    assert tool.annotations.idempotentHint is False


async def test_register_marks_clear_honestly(stub_service):
    mcp = FastMCP("test")
    tools.register(mcp)

    tool = await mcp.get_tool("portfolio_clear_portfolio")

    assert tool.annotations is not None
    assert tool.annotations.readOnlyHint is False
    assert tool.annotations.destructiveHint is True
    assert tool.annotations.idempotentHint is True


async def test_register_marks_watchlist_create_honestly(stub_service):
    mcp = FastMCP("test")
    tools.register(mcp)

    tool = await mcp.get_tool("portfolio_watchlist_create")

    assert tool.annotations is not None
    assert tool.annotations.readOnlyHint is False
    assert tool.annotations.destructiveHint is False
    assert tool.annotations.idempotentHint is False


async def test_register_marks_watchlist_add_honestly(stub_service):
    mcp = FastMCP("test")
    tools.register(mcp)

    tool = await mcp.get_tool("portfolio_watchlist_add")

    assert tool.annotations is not None
    assert tool.annotations.readOnlyHint is False
    assert tool.annotations.destructiveHint is False
    assert tool.annotations.idempotentHint is False


async def test_register_marks_watchlist_remove_honestly(stub_service):
    mcp = FastMCP("test")
    tools.register(mcp)

    tool = await mcp.get_tool("portfolio_watchlist_remove")

    assert tool.annotations is not None
    assert tool.annotations.readOnlyHint is False
    assert tool.annotations.destructiveHint is True
    assert tool.annotations.idempotentHint is True


async def test_register_marks_journal_add_trade_honestly(stub_service):
    mcp = FastMCP("test")
    tools.register(mcp)

    tool = await mcp.get_tool("portfolio_journal_add_trade")

    assert tool.annotations is not None
    assert tool.annotations.readOnlyHint is False
    assert tool.annotations.destructiveHint is False
    assert tool.annotations.idempotentHint is False


async def test_register_marks_journal_close_trade_honestly(stub_service):
    mcp = FastMCP("test")
    tools.register(mcp)

    tool = await mcp.get_tool("portfolio_journal_close_trade")

    assert tool.annotations is not None
    assert tool.annotations.readOnlyHint is False
    assert tool.annotations.destructiveHint is False
    assert tool.annotations.idempotentHint is False


async def test_register_attaches_my_holdings_resource(stub_service):
    mcp = FastMCP("test")
    tools.register(mcp)

    resources = await mcp.list_resources()

    assert any(str(r.uri) == "portfolio://my-holdings" for r in resources)


async def test_register_in_memory_client_round_trips_get_my_portfolio(stub_service):
    mcp = FastMCP("test")
    tools.register(mcp)

    async with Client(mcp) as client:
        result = await client.call_tool("portfolio_get_my_portfolio", {})

    assert result.data["status"] == "success"
    assert result.data["positions"][0]["ticker"] == "AAPL"
    assert stub_service.get_portfolio_calls == [("default", "My Portfolio")]


async def test_register_in_memory_client_round_trips_get_risk_dashboard(stub_service):
    mcp = FastMCP("test")
    tools.register(mcp)

    async with Client(mcp) as client:
        result = await client.call_tool("portfolio_get_risk_dashboard", {})

    assert result.data["status"] == "success"
    assert result.data["total_value"] == 2500.0
    assert stub_service.risk_dashboard_calls == [("default", "My Portfolio")]


async def test_register_in_memory_client_round_trips_watchlist_brief(stub_service):
    mcp = FastMCP("test")
    tools.register(mcp)

    async with Client(mcp) as client:
        result = await client.call_tool(
            "portfolio_watchlist_brief", {"watchlist_id": 1}
        )

    assert result.data["status"] == "success"
    assert result.data["count"] == 1
    assert stub_service.watchlist_brief_calls == [1]


async def test_register_in_memory_client_round_trips_journal_list_trades(
    stub_journal_service,
):
    mcp = FastMCP("test")
    tools.register(mcp)

    async with Client(mcp) as client:
        result = await client.call_tool("portfolio_journal_list_trades", {})

    assert result.data["status"] == "success"
    assert result.data["trades"][0]["symbol"] == "AAPL"
    assert stub_journal_service.list_trades_calls == [(None, None, None, 50)]


async def test_register_in_memory_client_reads_my_holdings_resource(stub_service):
    import json

    mcp = FastMCP("test")
    tools.register(mcp)

    async with Client(mcp) as client:
        result = await client.read_resource("portfolio://my-holdings")

    payload = json.loads(result[0].text)
    assert payload["status"] == "success"
    assert payload["uri"] == "portfolio://my-holdings"
    assert payload["positions"][0]["ticker"] == "AAPL"
