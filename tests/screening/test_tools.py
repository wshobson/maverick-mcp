"""Tests for maverick.screening.tools."""

from typing import Any

import pytest
from fastmcp import Client, FastMCP

from maverick.screening import tools
from maverick.screening.config import ScreeningSettings
from maverick.screening.types import ScreeningCriteria, ScreeningResult, ScreenRun


def _result(
    symbol: str, screen: str = "bullish", combined_score: int = 90
) -> ScreeningResult:
    return ScreeningResult(
        symbol=symbol,
        screen=screen,  # type: ignore[arg-type]
        date_analyzed="2026-07-19",
        close=190.5,
        combined_score=combined_score,
        momentum_score=None,
        indicators={"rsi14": 65.0},
        flags={"close_above_sma50": True},
        reason="stub reason",
    )


def _run(
    screen: str, screened: int = 4, qualified: int = 1, failed: int = 0
) -> ScreenRun:
    return ScreenRun(
        screen=screen,  # type: ignore[arg-type]
        symbols_screened=screened,
        symbols_qualified=qualified,
        symbols_failed=failed,
        date_analyzed="2026-07-19",
        duration_seconds=0.42,
    )


class StubService:
    """Async fakes matching `ScreeningService`'s public surface."""

    def __init__(self) -> None:
        self.settings = ScreeningSettings()
        self.bullish_calls: list[tuple[int | None, int | None]] = []
        self.bearish_calls: list[tuple[int | None, int | None]] = []
        self.supply_demand_calls: list[tuple[int | None, float | None]] = []
        self.all_calls = 0
        self.criteria_calls: list[tuple[ScreeningCriteria, int | None]] = []
        self.run_screen_calls: list[str] = []
        self.run_all_calls = 0

        self.bullish_result = [_result("AAPL", "bullish", 90)]
        self.bearish_result = [_result("XOM", "bearish", 80)]
        self.supply_demand_result = [_result("MSFT", "supply_demand", 75)]
        self.criteria_result = [_result("GOOG", "bullish", 95)]
        self.run_screen_result = _run("bullish")
        self.run_all_result = {
            "bullish": _run("bullish"),
            "bearish": _run("bearish"),
            "supply_demand": _run("supply_demand"),
        }

        self.raise_on_bullish: Exception | None = None
        self.raise_on_run_screen: Exception | None = None

    async def get_bullish(
        self, limit: int | None = None, min_score: int | None = None
    ) -> list[ScreeningResult]:
        self.bullish_calls.append((limit, min_score))
        if self.raise_on_bullish is not None:
            raise self.raise_on_bullish
        return self.bullish_result

    async def get_bearish(
        self, limit: int | None = None, min_score: int | None = None
    ) -> list[ScreeningResult]:
        self.bearish_calls.append((limit, min_score))
        return self.bearish_result

    async def get_supply_demand(
        self, limit: int | None = None, min_momentum_score: float | None = None
    ) -> list[ScreeningResult]:
        self.supply_demand_calls.append((limit, min_momentum_score))
        return self.supply_demand_result

    async def get_all(self):
        self.all_calls += 1
        from maverick.screening.types import AllScreeningResults

        return AllScreeningResults(
            bullish=self.bullish_result,
            bearish=self.bearish_result,
            supply_demand=self.supply_demand_result,
        )

    async def get_by_criteria(
        self, criteria: ScreeningCriteria, limit: int | None = None
    ) -> list[ScreeningResult]:
        self.criteria_calls.append((criteria, limit))
        return self.criteria_result

    async def run_screen(self, screen: str) -> ScreenRun:
        self.run_screen_calls.append(screen)
        if self.raise_on_run_screen is not None:
            raise self.raise_on_run_screen
        return self.run_screen_result

    async def run_all_screens(self) -> dict[str, ScreenRun]:
        self.run_all_calls += 1
        return self.run_all_result


@pytest.fixture
def stub_service() -> Any:
    stub = StubService()
    tools.configure(stub)
    yield stub


# ---------------------------------------------------------------------------
# unconfigured service: _require_service() raises before any service call
# ---------------------------------------------------------------------------


async def test_unconfigured_service_returns_configure_error_payload():
    tools.configure(None)  # type: ignore[arg-type]

    result = await tools.screening_get_bullish()

    assert result == {
        "status": "error",
        "error": "screening.tools: configure(service) was not called",
    }


# ---------------------------------------------------------------------------
# screening_get_bullish
# ---------------------------------------------------------------------------


async def test_get_bullish_returns_model_dump_list_plus_status_and_count(stub_service):
    result = await tools.screening_get_bullish(limit=10, min_score=50)

    assert result["status"] == "success"
    assert result["count"] == 1
    assert result["results"][0]["symbol"] == "AAPL"
    assert stub_service.bullish_calls == [(10, 50)]


async def test_get_bullish_defaults(stub_service):
    await tools.screening_get_bullish()

    assert stub_service.bullish_calls == [(20, None)]


async def test_get_bullish_service_exception_returns_error_payload(stub_service):
    stub_service.raise_on_bullish = RuntimeError("boom")

    result = await tools.screening_get_bullish()

    assert result == {"status": "error", "error": "boom"}


# ---------------------------------------------------------------------------
# screening_get_bearish
# ---------------------------------------------------------------------------


async def test_get_bearish_returns_model_dump_list_plus_status_and_count(stub_service):
    result = await tools.screening_get_bearish(limit=5, min_score=40)

    assert result["status"] == "success"
    assert result["count"] == 1
    assert result["results"][0]["symbol"] == "XOM"
    assert stub_service.bearish_calls == [(5, 40)]


# ---------------------------------------------------------------------------
# screening_get_supply_demand
# ---------------------------------------------------------------------------


async def test_get_supply_demand_returns_model_dump_list_plus_status_and_count(
    stub_service,
):
    result = await tools.screening_get_supply_demand(limit=15, min_momentum_score=70.0)

    assert result["status"] == "success"
    assert result["count"] == 1
    assert result["results"][0]["symbol"] == "MSFT"
    assert stub_service.supply_demand_calls == [(15, 70.0)]


# ---------------------------------------------------------------------------
# screening_get_all
# ---------------------------------------------------------------------------


async def test_get_all_returns_all_screening_results_dump_plus_status(stub_service):
    result = await tools.screening_get_all()

    assert result["status"] == "success"
    assert result["bullish"][0]["symbol"] == "AAPL"
    assert result["bearish"][0]["symbol"] == "XOM"
    assert result["supply_demand"][0]["symbol"] == "MSFT"
    assert stub_service.all_calls == 1


# ---------------------------------------------------------------------------
# screening_get_by_criteria
# ---------------------------------------------------------------------------


async def test_get_by_criteria_builds_criteria_and_returns_payload(stub_service):
    result = await tools.screening_get_by_criteria(
        min_momentum_score=60.0,
        min_volume=1_000_000,
        max_price=200.0,
        min_combined_score=70,
        limit=25,
    )

    assert result["status"] == "success"
    assert result["count"] == 1
    assert result["results"][0]["symbol"] == "GOOG"
    assert len(stub_service.criteria_calls) == 1
    criteria, limit = stub_service.criteria_calls[0]
    assert criteria == ScreeningCriteria(
        min_momentum_score=60.0,
        min_volume=1_000_000,
        max_price=200.0,
        min_combined_score=70,
    )
    assert limit == 25


async def test_get_by_criteria_defaults(stub_service):
    await tools.screening_get_by_criteria()

    criteria, limit = stub_service.criteria_calls[0]
    assert criteria == ScreeningCriteria()
    assert limit == 20


# ---------------------------------------------------------------------------
# screening_run_screens
# ---------------------------------------------------------------------------


async def test_run_screens_with_no_screen_runs_all(stub_service):
    result = await tools.screening_run_screens()

    assert result["status"] == "success"
    assert result["count"] == 3
    assert set(result["results"]) == {"bullish", "bearish", "supply_demand"}
    assert result["results"]["bullish"]["symbols_screened"] == 4
    assert stub_service.run_all_calls == 1
    assert stub_service.run_screen_calls == []


async def test_run_screens_with_explicit_screen_runs_only_that_one(stub_service):
    result = await tools.screening_run_screens(screen="bullish")

    assert result["status"] == "success"
    assert result["count"] == 1
    assert set(result["results"]) == {"bullish"}
    assert stub_service.run_screen_calls == ["bullish"]
    assert stub_service.run_all_calls == 0


async def test_run_screens_service_exception_returns_error_payload(stub_service):
    stub_service.raise_on_run_screen = ValueError("Unknown screen: 'bogus'")

    result = await tools.screening_run_screens(screen="bogus")

    assert result == {"status": "error", "error": "Unknown screen: 'bogus'"}


# ---------------------------------------------------------------------------
# register: attaches six tools with honest annotations
# ---------------------------------------------------------------------------


_EXPECTED_TOOL_NAMES = {
    "screening_get_bullish",
    "screening_get_bearish",
    "screening_get_supply_demand",
    "screening_get_all",
    "screening_get_by_criteria",
    "screening_run_screens",
}


async def test_register_attaches_six_tools_with_screening_names(stub_service):
    mcp = FastMCP("test")
    tools.register(mcp)

    registered = await mcp.list_tools()

    assert {tool.name for tool in registered} == _EXPECTED_TOOL_NAMES


async def test_register_marks_all_but_run_screens_read_only(stub_service):
    mcp = FastMCP("test")
    tools.register(mcp)

    for name in _EXPECTED_TOOL_NAMES - {"screening_run_screens"}:
        tool = await mcp.get_tool(name)
        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is True


async def test_register_marks_run_screens_honestly_non_read_only(stub_service):
    mcp = FastMCP("test")
    tools.register(mcp)

    tool = await mcp.get_tool("screening_run_screens")

    assert tool.annotations is not None
    assert tool.annotations.readOnlyHint is False
    assert tool.annotations.destructiveHint is False
    assert tool.annotations.idempotentHint is True


async def test_register_in_memory_client_round_trips_get_bullish(stub_service):
    mcp = FastMCP("test")
    tools.register(mcp)

    async with Client(mcp) as client:
        result = await client.call_tool("screening_get_bullish", {"limit": 10})

    assert result.data["status"] == "success"
    assert result.data["results"][0]["symbol"] == "AAPL"
    assert stub_service.bullish_calls == [(10, None)]
