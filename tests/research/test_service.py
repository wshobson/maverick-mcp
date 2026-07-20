"""Tests for `maverick.research.service`/`service_support`.

`service.py` imports `maverick.research.agents.graph.DeepResearchAgent` (which imports
`langgraph`/`langchain_core`) and `maverick.research.providers.exa.ExaSearchProvider` at module
top level, so importing `maverick.research.service` requires the `[research]` extra -- this whole
module is guarded, mirroring `tests/backtesting/test_service.py`'s guard for vectorbt/sklearn.

Fully mocked: `FakeAgent` is a plain async double satisfying `service_support.ResearchRunner`'s
structural shape, injected via `ResearchService(agent_factory=...)`. No network, no real API keys,
no LLM/Exa calls -- the injectable-agent seam is exactly what makes that possible (see
`service.py`'s module docstring, "Injectable seam" section).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytest.importorskip("langgraph")

from pydantic import SecretStr  # noqa: E402

from maverick.platform.llm import reset_llm_settings  # noqa: E402
from maverick.research.config import ResearchSettings  # noqa: E402
from maverick.research.service import ResearchService  # noqa: E402
from maverick.research.types import (  # noqa: E402
    CompanyResearchResult,
    ComprehensiveResearchResult,
    ResearchError,
    ResearchReport,
    SentimentAnalysisResult,
)


class FakeAgent:
    """Async double for `service_support.ResearchRunner`. Records every call and can be scripted
    to return a fixed `ResearchReport`, raise, or hang past a timeout."""

    def __init__(
        self,
        *,
        report: ResearchReport | None = None,
        raise_exc: Exception | None = None,
        delay: float = 0.0,
    ) -> None:
        self.report = report
        self.raise_exc = raise_exc
        self.delay = delay
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def _respond(self, method: str, **kwargs: Any) -> ResearchReport:
        self.calls.append((method, kwargs))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.raise_exc is not None:
            raise self.raise_exc
        assert self.report is not None
        return self.report

    async def research_topic(
        self,
        query: str,
        session_id: str,
        *,
        focus_areas: list[str] | None = None,
        timeframe: str = "30d",
        depth: str | None = None,
    ) -> ResearchReport:
        return await self._respond(
            "research_topic",
            query=query,
            session_id=session_id,
            focus_areas=focus_areas,
            timeframe=timeframe,
            depth=depth,
        )

    async def research_company_comprehensive(
        self,
        symbol: str,
        session_id: str,
        *,
        include_competitive_analysis: bool = False,
        depth: str | None = None,
    ) -> ResearchReport:
        return await self._respond(
            "research_company_comprehensive",
            symbol=symbol,
            session_id=session_id,
            include_competitive_analysis=include_competitive_analysis,
            depth=depth,
        )

    async def analyze_market_sentiment(
        self, topic: str, session_id: str, *, timeframe: str = "7d"
    ) -> ResearchReport:
        return await self._respond(
            "analyze_market_sentiment",
            topic=topic,
            session_id=session_id,
            timeframe=timeframe,
        )


def _fixture_report(**overrides: Any) -> ResearchReport:
    fields: dict[str, Any] = {
        "status": "success",
        "agent_type": "deep_research",
        "persona": "moderate",
        "research_topic": "AAPL outlook",
        "research_depth": "standard",
        "findings": {
            "synthesis": "Strong fundamentals with moderate near-term risk.",
            "key_insights": ["Revenue growing", "Margins expanding", "Guidance raised"],
            "overall_sentiment": {
                "direction": "bullish",
                "confidence": 0.8,
                "consensus": 0.7,
                "source_count": 3,
            },
            "risk_assessment": ["Macro headwinds"],
            "investment_implications": {
                "opportunities": ["Expansion", "New product line", "Buyback"],
                "threats": ["Competition"],
                "recommended_action": "Consider accumulating",
                "time_horizon": "medium-term",
            },
            "confidence_score": 0.75,
        },
        "sources_analyzed": 3,
        "confidence_score": 0.75,
        "citations": [],
        "execution_time_ms": 1500.0,
        "search_queries_used": ["AAPL financial analysis"],
        "source_diversity": 0.6,
    }
    fields.update(overrides)
    return ResearchReport(**fields)


def _configured_settings(**overrides: Any) -> ResearchSettings:
    fields: dict[str, Any] = {"exa_api_key": SecretStr("test-exa-key")}
    fields.update(overrides)
    return ResearchSettings(**fields)


@pytest.fixture
def configured_llm(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key")
    monkeypatch.setenv("LLM_MODEL", "gpt-test")
    reset_llm_settings()
    yield
    reset_llm_settings()


def _service(agent: FakeAgent, **settings_overrides: Any) -> ResearchService:
    return ResearchService(
        settings=_configured_settings(**settings_overrides),
        agent_factory=lambda **_kw: agent,
    )


# ---------------------------------------------------------------------------
# Not-configured errors
# ---------------------------------------------------------------------------


async def test_run_comprehensive_errors_when_exa_not_configured(configured_llm):
    service = ResearchService(settings=ResearchSettings())  # no exa_api_key

    result = await service.run_comprehensive("AAPL outlook")

    assert isinstance(result, ResearchError)
    assert result.success is False
    assert result.error_type == "not_configured"
    assert "Exa search provider not configured" in result.error


async def test_run_comprehensive_errors_when_llm_not_configured():
    # No configured_llm fixture: LLM_PROVIDER etc. stay unset (conftest resets the cache).
    service = _service(FakeAgent())

    result = await service.run_comprehensive("AAPL outlook")

    assert isinstance(result, ResearchError)
    assert result.error_type == "not_configured"
    assert "no LLM configured" in result.error


async def test_run_comprehensive_errors_when_llm_config_incomplete(
    monkeypatch: pytest.MonkeyPatch,
):
    # LLM_PROVIDER set but LLM_API_KEY/LLM_MODEL missing: `LLMSettings()` raises a
    # `ValueError` at construction (inside `get_llm_settings()`) rather than returning a
    # settings object with `provider=None` -- this must still route through the same
    # typed `ResearchError` shape as the other two not-configured cases, not escape as a
    # raw pydantic-formatted message (see `service.py`'s `_configuration_error`).
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    reset_llm_settings()
    service = _service(FakeAgent())

    result = await service.run_comprehensive("AAPL outlook")

    assert isinstance(result, ResearchError)
    assert result.success is False
    assert result.error_type == "not_configured"
    assert "LLM_API_KEY" in result.error
    assert "LLM_PROVIDER=anthropic" in result.error
    assert result.model_extra is not None
    assert "LLM_API_KEY" in result.model_extra["details"]["required_configuration"]


async def test_analyze_company_not_configured_error_names_symbol():
    service = ResearchService(settings=ResearchSettings())

    result = await service.analyze_company("AAPL")

    assert isinstance(result, ResearchError)
    assert result.model_extra is not None
    assert result.model_extra["symbol"] == "AAPL"
    assert result.model_extra["analysis_type"] == "company_comprehensive"


async def test_analyze_sentiment_not_configured_error_names_topic():
    service = ResearchService(settings=ResearchSettings())

    result = await service.analyze_sentiment("semiconductors")

    assert isinstance(result, ResearchError)
    assert result.model_extra is not None
    assert result.model_extra["topic"] == "semiconductors"


# ---------------------------------------------------------------------------
# run_comprehensive: typed success, pinned values from the fixture report
# ---------------------------------------------------------------------------


async def test_run_comprehensive_success_pins_real_report_fields(configured_llm):
    agent = FakeAgent(report=_fixture_report())
    service = _service(agent)

    result = await service.run_comprehensive("AAPL outlook", persona="aggressive")

    assert isinstance(result, ComprehensiveResearchResult)
    assert result.success is True
    assert result.query == "AAPL outlook"
    assert (
        result.research_results.summary
        == "Strong fundamentals with moderate near-term risk."
    )
    assert result.research_results.confidence_score == 0.75
    assert result.research_results.sources_analyzed == 3
    assert result.research_results.key_insights == [
        "Revenue growing",
        "Margins expanding",
        "Guidance raised",
    ]
    assert result.research_results.sentiment == {
        "direction": "bullish",
        "confidence": 0.8,
        "consensus": 0.7,
        "source_count": 3,
    }
    assert result.research_results.key_themes == [
        "Expansion",
        "New product line",
        "Buyback",
    ]
    assert result.research_metadata.persona == "aggressive"
    assert result.research_metadata.scope == "standard"
    assert result.research_metadata.sources_actually_used == 3
    assert result.research_metadata.execution_mode == "sequential_graph"
    assert result.research_metadata.optimization_features == []
    assert result.research_metadata.parallel_processing.enabled is False
    assert result.research_metadata.is_partial_result is False
    assert result.research_metadata.elapsed_time == pytest.approx(1.5)

    # The agent was called with the resolved depth/timeframe, a generated session id.
    method, kwargs = agent.calls[0]
    assert method == "research_topic"
    assert kwargs["query"] == "AAPL outlook"
    assert kwargs["depth"] == "standard"
    assert kwargs["timeframe"] == "1m"  # ResearchSettings.default_timeframe


async def test_run_comprehensive_defaults_persona_and_depth_from_settings(
    configured_llm,
):
    agent = FakeAgent(report=_fixture_report())
    service = _service(agent)

    result = await service.run_comprehensive("semiconductors outlook")

    assert isinstance(result, ComprehensiveResearchResult)
    assert result.research_metadata.persona == "moderate"
    assert result.research_metadata.scope == "standard"


async def test_run_comprehensive_invalid_persona_and_scope_fall_back_to_defaults(
    configured_llm,
):
    agent = FakeAgent(report=_fixture_report())
    service = _service(agent)

    result = await service.run_comprehensive(
        "semiconductors outlook", persona="nonsense", research_scope="nonsense"
    )

    assert isinstance(result, ComprehensiveResearchResult)
    assert result.research_metadata.persona == "moderate"
    assert result.research_metadata.scope == "standard"


async def test_run_comprehensive_max_sources_defaults_from_settings(configured_llm):
    agent = FakeAgent(report=_fixture_report())
    service = _service(agent)

    result = await service.run_comprehensive("AAPL outlook")

    assert isinstance(result, ComprehensiveResearchResult)
    assert result.research_metadata.max_sources_requested == 10
    assert result.research_metadata.max_sources_optimized == 10


# ---------------------------------------------------------------------------
# analyze_company: typed success, fixed depth/timeframe/max_sources overrides
# ---------------------------------------------------------------------------


async def test_analyze_company_success_pins_real_report_fields(configured_llm):
    agent = FakeAgent(report=_fixture_report())
    service = _service(agent)

    result = await service.analyze_company(
        "AAPL", include_competitive_analysis=True, persona="conservative"
    )

    assert isinstance(result, CompanyResearchResult)
    assert result.symbol == "AAPL"
    assert result.company_analysis.investment_summary == (
        "Strong fundamentals with moderate near-term risk."
    )
    assert result.company_analysis.confidence_score == 0.75
    assert result.company_analysis.sources_analyzed == 3
    assert result.company_analysis.analysis_themes == [
        "Expansion",
        "New product line",
        "Buyback",
    ]
    assert result.analysis_metadata.symbol == "AAPL"
    assert result.analysis_metadata.competitive_analysis_included is True
    assert result.analysis_metadata.analysis_type == "company_comprehensive"
    # Fixed overrides from settings, not caller input.
    assert result.analysis_metadata.scope == "standard"
    assert result.analysis_metadata.timeframe == "1m"
    assert result.analysis_metadata.max_sources_requested == 10

    method, kwargs = agent.calls[0]
    assert method == "research_company_comprehensive"
    assert kwargs["symbol"] == "AAPL"
    assert kwargs["include_competitive_analysis"] is True
    assert kwargs["depth"] == "standard"


# ---------------------------------------------------------------------------
# analyze_sentiment: typed success, fixed depth/max_sources overrides
# ---------------------------------------------------------------------------


async def test_analyze_sentiment_success_pins_real_report_fields(configured_llm):
    agent = FakeAgent(report=_fixture_report())
    service = _service(agent)

    result = await service.analyze_sentiment("semiconductors", timeframe="1w")

    assert isinstance(result, SentimentAnalysisResult)
    assert result.topic == "semiconductors"
    assert result.sentiment_analysis.overall_sentiment == {
        "direction": "bullish",
        "confidence": 0.8,
        "consensus": 0.7,
        "source_count": 3,
    }
    assert result.sentiment_analysis.sentiment_confidence == 0.75
    assert result.sentiment_analysis.market_insights == [
        "Revenue growing",
        "Margins expanding",
        "Guidance raised",
    ]
    assert result.analysis_metadata.topic == "semiconductors"
    assert result.analysis_metadata.analysis_type == "market_sentiment"
    # Fixed depth override from settings.
    assert result.analysis_metadata.scope == "basic"
    assert result.analysis_metadata.max_sources_requested == 8

    method, kwargs = agent.calls[0]
    assert method == "analyze_market_sentiment"
    assert kwargs["topic"] == "semiconductors"
    assert kwargs["timeframe"] == "1w"


async def test_analyze_sentiment_defaults_timeframe_from_settings(configured_llm):
    agent = FakeAgent(report=_fixture_report())
    service = _service(agent)

    await service.analyze_sentiment("semiconductors")

    method, kwargs = agent.calls[0]
    assert kwargs["timeframe"] == "1w"  # ResearchSettings.sentiment_default_timeframe


# ---------------------------------------------------------------------------
# Timeout -> typed ResearchError
# ---------------------------------------------------------------------------


async def test_run_comprehensive_timeout_returns_typed_error(configured_llm):
    agent = FakeAgent(report=_fixture_report(), delay=10.0)
    service = _service(agent, depth_timeout_seconds={"standard": 0.01})

    result = await service.run_comprehensive("AAPL outlook")

    assert isinstance(result, ResearchError)
    assert result.error_type == "timeout"
    assert "timed out" in result.error
    assert result.model_extra is not None
    assert result.model_extra["research_scope"] == "standard"
    assert "reduce_scope" in result.model_extra["suggestions"]
    assert "reduce_sources" not in result.model_extra["suggestions"]


async def test_analyze_company_timeout_returns_typed_error(configured_llm):
    agent = FakeAgent(report=_fixture_report(), delay=10.0)
    service = _service(agent, depth_timeout_seconds={"standard": 0.01})

    result = await service.analyze_company("AAPL")

    assert isinstance(result, ResearchError)
    assert result.error_type == "timeout"
    assert result.model_extra is not None
    assert result.model_extra["symbol"] == "AAPL"


async def test_analyze_sentiment_timeout_returns_typed_error(configured_llm):
    agent = FakeAgent(report=_fixture_report(), delay=10.0)
    service = _service(agent, depth_timeout_seconds={"basic": 0.01})

    result = await service.analyze_sentiment("semiconductors")

    assert isinstance(result, ResearchError)
    assert result.error_type == "timeout"
    assert result.model_extra is not None
    assert result.model_extra["topic"] == "semiconductors"


# ---------------------------------------------------------------------------
# Agent failure -> typed ResearchError (not a raised exception)
# ---------------------------------------------------------------------------


async def test_run_comprehensive_agent_failure_returns_typed_error(configured_llm):
    agent = FakeAgent(raise_exc=RuntimeError("graph exploded"))
    service = _service(agent)

    result = await service.run_comprehensive("AAPL outlook")

    assert isinstance(result, ResearchError)
    assert result.error_type == "RuntimeError"
    assert "graph exploded" in result.error


# ---------------------------------------------------------------------------
# timeout_warning: real computation from execution_time_ms, not a legacy fallback
# ---------------------------------------------------------------------------


async def test_run_comprehensive_flags_timeout_warning_near_budget(configured_llm):
    # 240s standard timeout; 80% threshold is 192s = 192000ms.
    agent = FakeAgent(report=_fixture_report(execution_time_ms=200_000.0))
    service = _service(agent)

    result = await service.run_comprehensive("AAPL outlook")

    assert isinstance(result, ComprehensiveResearchResult)
    assert result.research_metadata.timeout_warning is True
    assert result.warning is not None
    assert result.warning.type == "timeout_warning"


async def test_run_comprehensive_no_warning_when_well_under_budget(configured_llm):
    agent = FakeAgent(report=_fixture_report(execution_time_ms=1000.0))
    service = _service(agent)

    result = await service.run_comprehensive("AAPL outlook")

    assert isinstance(result, ComprehensiveResearchResult)
    assert result.research_metadata.timeout_warning is False
    assert result.warning is None
