"""Tests for maverick.technical.tools."""

from typing import Any

import pytest
from fastmcp import Client, FastMCP

from maverick.technical import tools
from maverick.technical.types import (
    BollingerAnalysis,
    FullTechnicalAnalysis,
    LevelsResult,
    MACDAnalysis,
    RSIAnalysis,
    StochasticAnalysis,
    TrendAnalysis,
    VolumeAnalysis,
)


def _rsi() -> RSIAnalysis:
    return RSIAnalysis(
        current=69.91, period=14, signal="bullish", description="RSI is 69.91."
    )


def _macd() -> MACDAnalysis:
    return MACDAnalysis(
        macd=1.08,
        signal_line=1.05,
        histogram=0.02,
        indicator_signal="bullish",
        crossover="bullish crossover detected",
        description="MACD is bullish.",
    )


def _levels() -> LevelsResult:
    return LevelsResult(support=[123.66, 130.53, 132.0], resistance=[138.0, 144.27])


def _full() -> FullTechnicalAnalysis:
    return FullTechnicalAnalysis(
        ticker="AAPL",
        current_price=137.4,
        trend=TrendAnalysis(score=7, direction="bullish", adx=33.78),
        outlook="strongly bullish",
        rsi=_rsi(),
        macd=_macd(),
        stochastic=StochasticAnalysis(
            k=81.15,
            d=77.56,
            signal="bullish",
            crossover="bullish crossover detected",
            description="Stochastic is bullish.",
        ),
        bollinger=BollingerAnalysis(
            upper=137.57,
            middle=135.6,
            lower=133.63,
            current_price=137.4,
            position="above middle band",
            volatility="stable",
            description="Price is above middle band.",
        ),
        volume=VolumeAnalysis(
            current=1_800_000.0,
            average=1_080_000.0,
            ratio=1.67,
            description="above average",
            signal="bullish (high volume on up move)",
        ),
        levels=_levels(),
        analysis_metadata={"bars_analyzed": 247, "as_of": "2020-12-10"},
    )


class StubService:
    """Async fake matching `TechnicalService`'s public surface used by tools."""

    def __init__(self) -> None:
        self.rsi_calls: list[tuple[str, int | None, int | None]] = []
        self.macd_calls: list[
            tuple[str, int | None, int | None, int | None, int | None]
        ] = []
        self.support_resistance_calls: list[tuple[str, int | None]] = []
        self.full_analysis_calls: list[tuple[str, int | None]] = []

        self.rsi_result = _rsi()
        self.macd_result = _macd()
        self.levels_result = _levels()
        self.full_result = _full()

        self.raise_on_rsi: Exception | None = None
        self.raise_on_macd: Exception | None = None
        self.raise_on_support_resistance: Exception | None = None
        self.raise_on_full_analysis: Exception | None = None

    async def get_rsi(
        self, ticker: str, days: int | None = None, period: int | None = None
    ) -> RSIAnalysis:
        self.rsi_calls.append((ticker, days, period))
        if self.raise_on_rsi is not None:
            raise self.raise_on_rsi
        return self.rsi_result

    async def get_macd(
        self,
        ticker: str,
        days: int | None = None,
        fast_period: int | None = None,
        slow_period: int | None = None,
        signal_period: int | None = None,
    ) -> MACDAnalysis:
        self.macd_calls.append((ticker, days, fast_period, slow_period, signal_period))
        if self.raise_on_macd is not None:
            raise self.raise_on_macd
        return self.macd_result

    async def get_support_resistance(
        self, ticker: str, days: int | None = None
    ) -> LevelsResult:
        self.support_resistance_calls.append((ticker, days))
        if self.raise_on_support_resistance is not None:
            raise self.raise_on_support_resistance
        return self.levels_result

    async def get_full_analysis(
        self, ticker: str, days: int | None = None
    ) -> FullTechnicalAnalysis:
        self.full_analysis_calls.append((ticker, days))
        if self.raise_on_full_analysis is not None:
            raise self.raise_on_full_analysis
        return self.full_result


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

    result = await tools.technical_get_rsi_analysis("AAPL")

    assert result == {
        "status": "error",
        "error": "technical.tools: configure(service) was not called",
    }


# ---------------------------------------------------------------------------
# technical_get_rsi_analysis
# ---------------------------------------------------------------------------


async def test_get_rsi_analysis_returns_model_dump_plus_ticker_and_status(
    stub_service,
):
    result = await tools.technical_get_rsi_analysis("aapl", period=10, days=100)

    assert result["status"] == "success"
    assert result["ticker"] == "AAPL"
    assert result["current"] == 69.91
    assert result["signal"] == "bullish"
    assert stub_service.rsi_calls == [("aapl", 100, 10)]


async def test_get_rsi_analysis_defaults_pass_none_through_to_service(stub_service):
    await tools.technical_get_rsi_analysis("AAPL")

    assert stub_service.rsi_calls == [("AAPL", None, None)]


async def test_get_rsi_analysis_service_exception_returns_error_payload(stub_service):
    stub_service.raise_on_rsi = ValueError("insufficient history for 'AAPL'")

    result = await tools.technical_get_rsi_analysis("AAPL")

    assert result == {
        "status": "error",
        "error": "insufficient history for 'AAPL'",
    }


# ---------------------------------------------------------------------------
# technical_get_macd_analysis
# ---------------------------------------------------------------------------


async def test_get_macd_analysis_returns_model_dump_plus_ticker_and_status(
    stub_service,
):
    result = await tools.technical_get_macd_analysis(
        "aapl", fast_period=5, slow_period=15, signal_period=4, days=200
    )

    assert result["status"] == "success"
    assert result["ticker"] == "AAPL"
    assert result["indicator_signal"] == "bullish"
    assert result["crossover"] == "bullish crossover detected"
    assert stub_service.macd_calls == [("aapl", 200, 5, 15, 4)]


async def test_get_macd_analysis_defaults_pass_none_through_to_service(stub_service):
    await tools.technical_get_macd_analysis("AAPL")

    assert stub_service.macd_calls == [("AAPL", None, None, None, None)]


async def test_get_macd_analysis_service_exception_returns_error_payload(
    stub_service,
):
    stub_service.raise_on_macd = ValueError("boom")

    result = await tools.technical_get_macd_analysis("AAPL")

    assert result == {"status": "error", "error": "boom"}


# ---------------------------------------------------------------------------
# technical_get_support_resistance
# ---------------------------------------------------------------------------


async def test_get_support_resistance_returns_model_dump_plus_ticker_and_status(
    stub_service,
):
    result = await tools.technical_get_support_resistance("aapl", days=100)

    assert result["status"] == "success"
    assert result["ticker"] == "AAPL"
    assert result["support"] == [123.66, 130.53, 132.0]
    assert result["resistance"] == [138.0, 144.27]
    assert stub_service.support_resistance_calls == [("aapl", 100)]


async def test_get_support_resistance_service_exception_returns_error_payload(
    stub_service,
):
    stub_service.raise_on_support_resistance = ValueError("boom")

    result = await tools.technical_get_support_resistance("AAPL")

    assert result == {"status": "error", "error": "boom"}


# ---------------------------------------------------------------------------
# technical_get_full_technical_analysis
# ---------------------------------------------------------------------------


async def test_get_full_technical_analysis_returns_model_dump_plus_status(
    stub_service,
):
    result = await tools.technical_get_full_technical_analysis("AAPL", days=300)

    assert result["status"] == "success"
    assert result["ticker"] == "AAPL"
    assert result["current_price"] == 137.4
    assert result["trend"]["score"] == 7
    assert result["outlook"] == "strongly bullish"
    assert result["analysis_metadata"]["bars_analyzed"] == 247
    assert stub_service.full_analysis_calls == [("AAPL", 300)]


async def test_get_full_technical_analysis_service_exception_returns_error_payload(
    stub_service,
):
    stub_service.raise_on_full_analysis = ValueError("boom")

    result = await tools.technical_get_full_technical_analysis("AAPL")

    assert result == {"status": "error", "error": "boom"}


# ---------------------------------------------------------------------------
# register: attaches four tools, all honestly read-only
# ---------------------------------------------------------------------------


_EXPECTED_TOOL_NAMES = {
    "technical_get_rsi_analysis",
    "technical_get_macd_analysis",
    "technical_get_support_resistance",
    "technical_get_full_technical_analysis",
}


async def test_register_attaches_four_tools_with_technical_names(stub_service):
    mcp = FastMCP("test")
    tools.register(mcp)

    registered = await mcp.list_tools()

    assert {tool.name for tool in registered} == _EXPECTED_TOOL_NAMES


async def test_register_marks_every_tool_read_only(stub_service):
    mcp = FastMCP("test")
    tools.register(mcp)

    for name in _EXPECTED_TOOL_NAMES:
        tool = await mcp.get_tool(name)
        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is True


async def test_register_in_memory_client_round_trips_get_rsi(stub_service):
    mcp = FastMCP("test")
    tools.register(mcp)

    async with Client(mcp) as client:
        result = await client.call_tool(
            "technical_get_rsi_analysis", {"ticker": "AAPL", "period": 10}
        )

    assert result.data["status"] == "success"
    assert result.data["ticker"] == "AAPL"
    assert result.data["signal"] == "bullish"
    assert stub_service.rsi_calls == [("AAPL", None, 10)]
