"""Tests for `maverick.research.tools`. No `importorskip`: `tools.py` never imports
langgraph/langchain/exa-py (see its module docstring), so this file collects and runs identically
on a base install. The stubbed service below is a plain duck-typed double -- it never imports
`maverick.research.service` (which DOES require the `[research]` extra), so this whole file stays
extra-independent.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastmcp import Client, FastMCP

from maverick.research import tools
from maverick.research.types import (
    CompanyAnalysis,
    CompanyAnalysisMetadata,
    CompanyResearchResult,
    ComprehensiveResearchResult,
    ParallelProcessingInfo,
    ResearchError,
    ResearchMetadata,
    ResearchResultSummary,
    SentimentAnalysis,
    SentimentAnalysisMetadata,
    SentimentAnalysisResult,
)


def _metadata(**overrides: Any) -> ResearchMetadata:
    fields: dict[str, Any] = {
        "persona": "moderate",
        "scope": "standard",
        "timeframe": "1m",
        "max_sources_requested": 10,
        "max_sources_optimized": 10,
        "sources_actually_used": 3,
        "execution_mode": "sequential_graph",
        "is_partial_result": False,
        "timeout_warning": False,
        "elapsed_time": 1.5,
        "completion_percentage": 100,
        "optimization_features": [],
        "parallel_processing": ParallelProcessingInfo(
            enabled=False, max_concurrent_requests=1, batch_processing=False
        ),
    }
    fields.update(overrides)
    return ResearchMetadata(**fields)


def _comprehensive_result(**overrides: Any) -> ComprehensiveResearchResult:
    fields: dict[str, Any] = {
        "query": "AAPL outlook",
        "research_results": ResearchResultSummary(
            summary="Strong fundamentals.",
            confidence_score=0.75,
            sources_analyzed=3,
            key_insights=["Insight A"],
            sentiment={"direction": "bullish", "confidence": 0.8},
            key_themes=["Expansion"],
        ),
        "research_metadata": _metadata(),
        "request_id": "req-1",
        "timestamp": "2026-07-19T00:00:00+00:00",
    }
    fields.update(overrides)
    return ComprehensiveResearchResult(**fields)


def _company_result() -> CompanyResearchResult:
    return CompanyResearchResult(
        symbol="AAPL",
        company_analysis=CompanyAnalysis(
            investment_summary="Strong fundamentals.",
            confidence_score=0.75,
            key_insights=["Insight A"],
            financial_sentiment={"direction": "bullish"},
            analysis_themes=["Expansion"],
            sources_analyzed=3,
        ),
        analysis_metadata=CompanyAnalysisMetadata(
            **_metadata().model_dump(),
            symbol="AAPL",
            competitive_analysis_included=False,
        ),
        request_id="req-2",
        timestamp="2026-07-19T00:00:00+00:00",
    )


def _sentiment_result() -> SentimentAnalysisResult:
    return SentimentAnalysisResult(
        topic="semiconductors",
        sentiment_analysis=SentimentAnalysis(
            overall_sentiment={"direction": "bullish"},
            sentiment_confidence=0.7,
            key_themes=["Expansion"],
            market_insights=["Insight A"],
            sources_analyzed=3,
        ),
        analysis_metadata=SentimentAnalysisMetadata(
            **_metadata(
                scope="basic", max_sources_requested=8, max_sources_optimized=8
            ).model_dump(),
            topic="semiconductors",
        ),
        request_id="req-3",
        timestamp="2026-07-19T00:00:00+00:00",
    )


def _not_configured_error(**extra: Any) -> ResearchError:
    return ResearchError(
        error="Research functionality unavailable - Exa search provider not configured",
        error_type="not_configured",
        request_id="req-err",
        timestamp="2026-07-19T00:00:00+00:00",
        details={"exa_api_key": "Missing"},
        **extra,
    )


class StubService:
    """Fake matching `ResearchService`'s three-method public surface -- never imports the real
    `ResearchService` (which requires the `[research]` extra)."""

    def __init__(self) -> None:
        self.calls: dict[str, list[tuple]] = {}
        self.results: dict[str, Any] = {
            "run_comprehensive": _comprehensive_result(),
            "analyze_company": _company_result(),
            "analyze_sentiment": _sentiment_result(),
        }
        self.raise_on: dict[str, Exception] = {}

    async def _call(self, name: str, args: tuple) -> Any:
        self.calls.setdefault(name, []).append(args)
        if name in self.raise_on:
            raise self.raise_on[name]
        return self.results[name]

    async def run_comprehensive(self, query, **kwargs):
        return await self._call("run_comprehensive", (query, kwargs))

    async def analyze_company(self, symbol, **kwargs):
        return await self._call("analyze_company", (symbol, kwargs))

    async def analyze_sentiment(self, topic, **kwargs):
        return await self._call("analyze_sentiment", (topic, kwargs))


@pytest.fixture
def stub_service():
    stub = StubService()
    tools.configure(stub)
    yield stub


_EXPECTED_TOOL_NAMES = {
    "research_run_comprehensive",
    "research_analyze_company",
    "research_analyze_sentiment",
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


async def test_register_attaches_three_tools(stub_service):
    mcp = FastMCP("test")
    tools.register(mcp)

    registered = await mcp.list_tools()

    assert {tool.name for tool in registered} == _EXPECTED_TOOL_NAMES


async def test_register_marks_every_tool_read_only_and_open_world(stub_service):
    mcp = FastMCP("test")
    tools.register(mcp)

    for name in _EXPECTED_TOOL_NAMES:
        tool = await mcp.get_tool(name)
        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is True
        assert tool.annotations.openWorldHint is True


# ---------------------------------------------------------------------------
# Unconfigured service
# ---------------------------------------------------------------------------


async def test_research_run_comprehensive_unconfigured_returns_error_payload():
    # Explicit reset: another test's `stub_service` fixture may have already configured the
    # module-level service, and it is not torn down automatically (matches
    # tests/backtesting/test_tools.py's `test_unconfigured_service_returns_configure_error_payload`).
    tools.configure(None)  # type: ignore[arg-type]

    payload = await tools.research_run_comprehensive("AAPL outlook")

    assert payload == {
        "status": "error",
        "error": "research.tools: configure(service) was not called",
    }


async def test_research_analyze_company_unconfigured_returns_error_payload():
    tools.configure(None)  # type: ignore[arg-type]

    payload = await tools.research_analyze_company("AAPL")

    assert payload["status"] == "error"
    assert "configure(service)" in payload["error"]


async def test_research_analyze_sentiment_unconfigured_returns_error_payload():
    tools.configure(None)  # type: ignore[arg-type]

    payload = await tools.research_analyze_sentiment("semiconductors")

    assert payload["status"] == "error"
    assert "configure(service)" in payload["error"]


# ---------------------------------------------------------------------------
# Success payloads
# ---------------------------------------------------------------------------


async def test_research_run_comprehensive_success_payload(stub_service):
    payload = await tools.research_run_comprehensive(
        "AAPL outlook", persona="aggressive", research_scope="standard"
    )

    assert payload["status"] == "success"
    assert payload["success"] is True
    assert payload["query"] == "AAPL outlook"
    assert payload["research_results"]["confidence_score"] == 0.75
    assert stub_service.calls["run_comprehensive"][0][0] == "AAPL outlook"
    assert stub_service.calls["run_comprehensive"][0][1]["persona"] == "aggressive"


async def test_research_analyze_company_success_payload(stub_service):
    payload = await tools.research_analyze_company(
        "AAPL", include_competitive_analysis=True
    )

    assert payload["status"] == "success"
    assert payload["symbol"] == "AAPL"
    assert payload["company_analysis"]["confidence_score"] == 0.75
    assert stub_service.calls["analyze_company"][0][0] == "AAPL"
    assert (
        stub_service.calls["analyze_company"][0][1]["include_competitive_analysis"]
        is True
    )


async def test_research_analyze_sentiment_success_payload(stub_service):
    payload = await tools.research_analyze_sentiment("semiconductors", timeframe="1w")

    assert payload["status"] == "success"
    assert payload["topic"] == "semiconductors"
    assert payload["sentiment_analysis"]["sentiment_confidence"] == 0.7
    assert stub_service.calls["analyze_sentiment"][0][0] == "semiconductors"


# ---------------------------------------------------------------------------
# Error payloads: typed ResearchError from the service, and a raised exception
# ---------------------------------------------------------------------------


async def test_research_run_comprehensive_typed_service_error_becomes_error_payload(
    stub_service,
):
    stub_service.results["run_comprehensive"] = _not_configured_error(
        query="AAPL outlook"
    )

    payload = await tools.research_run_comprehensive("AAPL outlook")

    assert payload["status"] == "error"
    assert payload["success"] is False
    assert "not configured" in payload["error"]


async def test_research_analyze_company_raised_exception_becomes_error_payload(
    stub_service,
):
    stub_service.raise_on["analyze_company"] = ValueError("boom")

    payload = await tools.research_analyze_company("AAPL")

    assert payload == {"status": "error", "error": "boom"}


# ---------------------------------------------------------------------------
# In-memory fastmcp.Client round-trip
# ---------------------------------------------------------------------------


async def test_register_in_memory_client_round_trips_run_comprehensive(stub_service):
    mcp = FastMCP("test")
    tools.register(mcp)

    async with Client(mcp) as client:
        result = await client.call_tool(
            "research_run_comprehensive",
            {"query": "AAPL outlook", "persona": "moderate"},
        )

    assert result.data["status"] == "success"
    assert result.data["query"] == "AAPL outlook"
    assert stub_service.calls["run_comprehensive"][0][0] == "AAPL outlook"
