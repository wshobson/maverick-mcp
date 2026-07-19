"""Tests for maverick.market_data.tools."""

from datetime import date
from typing import Any

import pandas as pd
import pytest
from fastmcp import Client, FastMCP

from maverick.market_data import tools
from maverick.market_data.config import MarketDataSettings
from maverick.market_data.types import (
    CompanyInfo,
    Fundamentals,
    IndexQuote,
    MarketNumbers,
    MarketOverview,
    Quote,
    TradingStats,
    Volatility,
)


def _frame(dates: list[date]) -> pd.DataFrame:
    index = pd.DatetimeIndex(dates, name="Date")
    n = len(index)
    return pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(n)],
            "High": [101.0 + i for i in range(n)],
            "Low": [99.0 + i for i in range(n)],
            "Close": [100.5 + i for i in range(n)],
            "Volume": [1_000_000 + i for i in range(n)],
        },
        index=index,
    )


def _quote(symbol: str = "AAPL") -> Quote:
    return Quote(
        symbol=symbol,
        price=190.5,
        change=2.5,
        change_percent=1.3,
        volume=55_000_000,
        timestamp="2026-07-18T00:00:00+00:00",
    )


def _fundamentals(symbol: str = "AAPL") -> Fundamentals:
    return Fundamentals(
        symbol=symbol,
        company=CompanyInfo(
            name="Apple Inc.",
            sector="Technology",
            industry="Consumer Electronics",
            website="https://apple.com",
            description="Makes phones.",
        ),
        market_data=MarketNumbers(
            current_price=190.5,
            market_cap=3_000_000_000_000,
            enterprise_value=3_100_000_000_000,
            shares_outstanding=15_000_000_000,
            float_shares=14_900_000_000,
        ),
        valuation={"pe_ratio": 30.0},
        financials={"revenue": 400_000_000_000},
        trading=TradingStats(
            avg_volume=50_000_000,
            avg_volume_10d=55_000_000,
            beta=1.2,
            week_52_high=200.0,
            week_52_low=150.0,
        ),
    )


def _overview() -> MarketOverview:
    return MarketOverview(
        indices={
            "^GSPC": IndexQuote(
                name="S&P 500",
                symbol="^GSPC",
                price=6100.0,
                change=5.0,
                change_percent=0.1,
            )
        },
        sectors={"Technology": 1.5},
        top_gainers=[],
        top_losers=[],
        volatility=Volatility(vix=15.0, vix_change_percent=-1.0, fear_level="low"),
        last_updated="2026-07-18T00:00:00+00:00",
    )


class StubService:
    """Async fakes matching `MarketDataService`'s public surface."""

    def __init__(self) -> None:
        self.settings = MarketDataSettings()
        self.price_history_calls: list[tuple[str, date | None, date | None]] = []
        self.quote_calls: list[str] = []
        self.fundamentals_calls: list[str] = []
        self.market_overview_calls = 0
        self.clear_cache_calls: list[str | None] = []
        self.price_history_result = _frame([date(2026, 7, 13), date(2026, 7, 14)])
        self.quote_result = _quote()
        self.fundamentals_result = _fundamentals()
        self.overview_result = _overview()
        self.clear_cache_result = 3
        self.raise_on_price_history: Exception | None = None
        self.raise_on_quote: Exception | None = None

    async def get_price_history(
        self, symbol: str, start: date | None, end: date | None
    ) -> pd.DataFrame:
        self.price_history_calls.append((symbol, start, end))
        if self.raise_on_price_history is not None:
            raise self.raise_on_price_history
        return self.price_history_result

    async def get_quote(self, symbol: str) -> Quote:
        self.quote_calls.append(symbol)
        if self.raise_on_quote is not None:
            raise self.raise_on_quote
        return self.quote_result

    async def get_fundamentals(self, symbol: str) -> Fundamentals:
        self.fundamentals_calls.append(symbol)
        return self.fundamentals_result

    async def get_market_overview(self) -> MarketOverview:
        self.market_overview_calls += 1
        return self.overview_result

    async def clear_cache(self, symbol: str | None = None) -> int:
        self.clear_cache_calls.append(symbol)
        return self.clear_cache_result


@pytest.fixture(autouse=True)
def stub_service() -> Any:
    stub = StubService()
    tools.configure(stub)
    yield stub


# ---------------------------------------------------------------------------
# unconfigured service: _require_service() raises before any service call
# ---------------------------------------------------------------------------


async def test_unconfigured_service_returns_configure_error_payload(stub_service):
    """Reset the module-level service the autouse fixture just configured.

    The next test's autouse fixture reconfigures a fresh stub, so no
    explicit restore is needed here.
    """
    tools.configure(None)  # type: ignore[arg-type]

    result = await tools.get_quote("AAPL")

    assert result == {
        "status": "error",
        "error": "market_data.tools: configure(service) was not called",
    }


# ---------------------------------------------------------------------------
# get_price_history
# ---------------------------------------------------------------------------


async def test_get_price_history_returns_documented_shape(stub_service):
    result = await tools.get_price_history("AAPL", "2026-07-13", "2026-07-14")

    assert result["status"] == "success"
    assert result["ticker"] == "AAPL"
    assert result["record_count"] == 2
    assert "columns" in result
    assert "data" in result
    assert stub_service.price_history_calls == [
        ("AAPL", date(2026, 7, 13), date(2026, 7, 14))
    ]


async def test_get_price_history_defaults_dates_to_none(stub_service):
    await tools.get_price_history("AAPL")

    assert stub_service.price_history_calls == [("AAPL", None, None)]


async def test_get_price_history_bad_start_date_returns_error_payload_not_raise(
    stub_service,
):
    result = await tools.get_price_history("AAPL", "not-a-date")

    assert result["status"] == "error"
    assert "error" in result
    assert stub_service.price_history_calls == []


async def test_get_price_history_bad_end_date_returns_error_payload_not_raise(
    stub_service,
):
    result = await tools.get_price_history("AAPL", "2026-07-13", "also-not-a-date")

    assert result["status"] == "error"
    assert "error" in result


async def test_get_price_history_service_exception_returns_error_payload(
    stub_service,
):
    stub_service.raise_on_price_history = RuntimeError("boom")

    result = await tools.get_price_history("AAPL")

    assert result == {"status": "error", "error": "boom"}


# ---------------------------------------------------------------------------
# get_price_history_batch
# ---------------------------------------------------------------------------


async def test_get_price_history_batch_returns_per_ticker_results(stub_service):
    result = await tools.get_price_history_batch(["AAPL", "MSFT"])

    assert result["status"] == "success"
    assert result["tickers"] == ["AAPL", "MSFT"]
    assert result["success_count"] == 2
    assert result["error_count"] == 0
    assert set(result["results"]) == {"AAPL", "MSFT"}
    assert result["results"]["AAPL"]["status"] == "success"
    assert result["results"]["AAPL"]["record_count"] == 2


async def test_get_price_history_batch_partial_failure_does_not_abort_batch(
    stub_service,
):
    calls = {"n": 0}

    async def flaky_get_price_history(symbol, start, end):
        calls["n"] += 1
        if symbol == "BAD":
            raise RuntimeError("no such ticker")
        return stub_service.price_history_result

    stub_service.get_price_history = flaky_get_price_history

    result = await tools.get_price_history_batch(["AAPL", "BAD"])

    assert result["status"] == "success"
    assert result["success_count"] == 1
    assert result["error_count"] == 1
    assert result["results"]["AAPL"]["status"] == "success"
    assert result["results"]["BAD"] == {"status": "error", "error": "no such ticker"}


async def test_get_price_history_batch_bad_date_returns_error_payload_not_raise(
    stub_service,
):
    result = await tools.get_price_history_batch(["AAPL"], start_date="nope")

    assert result["status"] == "error"
    assert "error" in result
    assert stub_service.price_history_calls == []


async def test_get_price_history_batch_over_limit_returns_error_naming_limit(
    stub_service,
):
    stub_service.settings = MarketDataSettings(history_batch_max=2)

    result = await tools.get_price_history_batch(["AAPL", "MSFT", "GOOG"])

    assert result["status"] == "error"
    assert "2" in result["error"]
    assert stub_service.price_history_calls == []


async def test_get_price_history_batch_at_limit_succeeds(stub_service):
    stub_service.settings = MarketDataSettings(history_batch_max=2)

    result = await tools.get_price_history_batch(["AAPL", "MSFT"])

    assert result["status"] == "success"
    assert result["tickers"] == ["AAPL", "MSFT"]


# ---------------------------------------------------------------------------
# get_quote
# ---------------------------------------------------------------------------


async def test_get_quote_returns_model_dump_plus_status(stub_service):
    result = await tools.get_quote("aapl")

    assert result["status"] == "success"
    assert result["symbol"] == "AAPL"
    assert result["price"] == 190.5
    assert stub_service.quote_calls == ["aapl"]


async def test_get_quote_service_exception_returns_error_payload(stub_service):
    stub_service.raise_on_quote = ValueError("no such ticker")

    result = await tools.get_quote("ZZZZ")

    assert result == {"status": "error", "error": "no such ticker"}


# ---------------------------------------------------------------------------
# get_stock_fundamentals
# ---------------------------------------------------------------------------


async def test_get_stock_fundamentals_returns_model_dump_plus_status(stub_service):
    result = await tools.get_stock_fundamentals("aapl")

    assert result["status"] == "success"
    assert result["symbol"] == "AAPL"
    assert result["company"]["name"] == "Apple Inc."
    assert result["market_data"]["market_cap"] == 3_000_000_000_000
    assert stub_service.fundamentals_calls == ["aapl"]


# ---------------------------------------------------------------------------
# get_market_overview
# ---------------------------------------------------------------------------


async def test_get_market_overview_returns_model_dump_plus_status(stub_service):
    result = await tools.get_market_overview()

    assert result["status"] == "success"
    assert result["volatility"]["fear_level"] == "low"
    assert "^GSPC" in result["indices"]
    assert stub_service.market_overview_calls == 1


# ---------------------------------------------------------------------------
# get_chart_links: static, no service call
# ---------------------------------------------------------------------------


async def test_get_chart_links_returns_static_urls_without_calling_service(
    stub_service,
):
    result = await tools.get_chart_links("AAPL")

    assert result["status"] == "success"
    assert result["ticker"] == "AAPL"
    charts = result["charts"]
    assert charts["trading_view"] == "https://www.tradingview.com/symbols/AAPL"
    assert charts["finviz"] == "https://finviz.com/quote.ashx?t=AAPL"
    assert charts["yahoo_finance"] == "https://finance.yahoo.com/quote/AAPL/chart"
    assert charts["stock_charts"] == "https://stockcharts.com/h-sc/ui?s=AAPL"
    assert stub_service.quote_calls == []
    assert stub_service.fundamentals_calls == []
    assert stub_service.price_history_calls == []
    assert stub_service.market_overview_calls == 0


# ---------------------------------------------------------------------------
# clear_market_cache
# ---------------------------------------------------------------------------


async def test_clear_market_cache_with_ticker_calls_service(stub_service):
    result = await tools.clear_market_cache("AAPL")

    assert result["status"] == "success"
    assert result["ticker"] == "AAPL"
    assert result["entries_cleared"] == 3
    assert stub_service.clear_cache_calls == ["AAPL"]


async def test_clear_market_cache_without_ticker_calls_service_with_none(
    stub_service,
):
    result = await tools.clear_market_cache()

    assert result["status"] == "success"
    assert result["ticker"] is None
    assert stub_service.clear_cache_calls == [None]


async def test_clear_market_cache_service_exception_returns_error_payload(
    stub_service,
):
    async def raise_clear_cache(symbol=None):
        raise RuntimeError("cache backend down")

    stub_service.clear_cache = raise_clear_cache

    result = await tools.clear_market_cache("AAPL")

    assert result == {"status": "error", "error": "cache backend down"}


# ---------------------------------------------------------------------------
# register: attaches seven tools with honest annotations
# ---------------------------------------------------------------------------


_EXPECTED_TOOL_NAMES = {
    "market_data_get_price_history",
    "market_data_get_price_history_batch",
    "market_data_get_quote",
    "market_data_get_stock_fundamentals",
    "market_data_get_market_overview",
    "market_data_get_chart_links",
    "market_data_clear_market_cache",
}


async def test_register_attaches_seven_tools_with_market_data_prefix(stub_service):
    mcp = FastMCP("test")
    tools.register(mcp)

    registered = await mcp.list_tools()

    assert {tool.name for tool in registered} == _EXPECTED_TOOL_NAMES


async def test_register_marks_all_but_clear_cache_read_only(stub_service):
    mcp = FastMCP("test")
    tools.register(mcp)

    for name in _EXPECTED_TOOL_NAMES - {"market_data_clear_market_cache"}:
        tool = await mcp.get_tool(name)
        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is True


async def test_register_marks_clear_cache_honestly_non_read_only(stub_service):
    mcp = FastMCP("test")
    tools.register(mcp)

    tool = await mcp.get_tool("market_data_clear_market_cache")

    assert tool.annotations is not None
    assert tool.annotations.readOnlyHint is False
    assert tool.annotations.destructiveHint is False
    assert tool.annotations.idempotentHint is True


async def test_register_in_memory_client_round_trips_get_quote(stub_service):
    mcp = FastMCP("test")
    tools.register(mcp)

    async with Client(mcp) as client:
        result = await client.call_tool("market_data_get_quote", {"ticker": "AAPL"})

    assert result.data["status"] == "success"
    assert result.data["symbol"] == "AAPL"
    assert stub_service.quote_calls == ["AAPL"]
