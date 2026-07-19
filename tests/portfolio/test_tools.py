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
    PortfolioMetrics,
    PortfolioSnapshot,
    PositionPayload,
    PositionWithPrice,
    RemoveResult,
    RiskAnalysis,
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

        self.add_result = _position_payload()
        self.get_portfolio_result = _snapshot()
        self.remove_result = RemoveResult(
            ticker="AAPL", shares_removed=Decimal("5"), position_fully_closed=False
        )
        self.clear_result = 3
        self.risk_result = _risk_result()
        self.compare_result = _comparison_result()
        self.correlation_result = _correlation_result()

        self.raise_on_add: Exception | None = None
        self.raise_on_get_portfolio: Exception | None = None
        self.raise_on_remove: Exception | None = None
        self.raise_on_clear: Exception | None = None
        self.raise_on_risk: Exception | None = None
        self.raise_on_compare: Exception | None = None
        self.raise_on_correlation: Exception | None = None

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


@pytest.fixture
def stub_service() -> Any:
    stub = StubService()
    tools.configure(stub)
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
# register: seven tools + resource, honest annotations
# ---------------------------------------------------------------------------


_EXPECTED_TOOL_NAMES = {
    "portfolio_add_position",
    "portfolio_get_my_portfolio",
    "portfolio_remove_position",
    "portfolio_clear_portfolio",
    "portfolio_risk_adjusted_analysis",
    "portfolio_compare_tickers",
    "portfolio_correlation_analysis",
}

_READ_ONLY_NAMES = {
    "portfolio_get_my_portfolio",
    "portfolio_risk_adjusted_analysis",
    "portfolio_compare_tickers",
    "portfolio_correlation_analysis",
}


async def test_register_attaches_seven_tools(stub_service):
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
