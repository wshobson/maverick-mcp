"""Tests for `maverick.research.agents.graph.DeepResearchAgent`.

Fully mocked: fake search clients (no network), a deterministic fake chat
model (no real LLM calls, no API keys). Pins the behaviors this task's
brief calls for: the graph runs to completion and produces a typed
`ResearchReport` from fixture search results; persona conditioning
appears in the prompts the fake model receives; error propagation
(no configured search client, and an all-providers-failure run, both
resolve cleanly rather than hanging or raising an unhandled exception).
"""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from maverick.research.agents.graph import (  # noqa: E402
    DeepResearchAgent,
    ResearchAgentError,
)
from maverick.research.types import ResearchReport  # noqa: E402

from ._fakes import FakeChatModel, FakeSearchClient, make_source  # noqa: E402


def _analysis_json(direction: str = "bullish", confidence: float = 0.8) -> str:
    return json.dumps(
        {
            "KEY_INSIGHTS": ["Key insight one"],
            "SENTIMENT": {"direction": direction, "confidence": confidence},
            "RISK_FACTORS": ["A risk"],
            "OPPORTUNITIES": ["An opportunity"],
            "CREDIBILITY": 0.8,
            "RELEVANCE": 0.7,
            "SUMMARY": "A summary.",
        }
    )


def _dispatching_responder(messages: list) -> str:
    """Route by which node's system prompt is present -- content analysis
    gets JSON, synthesis gets free text -- deterministic regardless of
    call ordering under concurrent batches."""
    system = next((m.content for m in messages if isinstance(m, SystemMessage)), "")
    if "financial research synthesizer" in system:
        return "Executive summary: strong fundamentals with moderate risk."
    return _analysis_json()


def _fixture_search_client(count: int = 3) -> FakeSearchClient:
    return FakeSearchClient(
        results=[
            make_source(
                url=f"https://sec.gov/filing-{i}",
                title=f"Filing {i}",
                content=f"Company reports strong growth and revenue in period {i}.",
                published_date="2026-06-01T00:00:00+00:00",
            )
            for i in range(count)
        ]
    )


def _agent(**kwargs) -> DeepResearchAgent:
    llm = FakeChatModel(responder=_dispatching_responder)
    client = _fixture_search_client()
    return DeepResearchAgent(llm=llm, search_clients=[client], **kwargs)


class TestGraphReachesTerminalStateWithTypedReport:
    def test_research_comprehensive_returns_research_report(self) -> None:
        agent = _agent(persona="moderate")

        report = asyncio.run(
            agent.research_comprehensive(topic="AAPL", session_id="s-1")
        )

        assert isinstance(report, ResearchReport)
        assert report.status == "success"
        assert report.agent_type == "deep_research"
        assert report.persona == "moderate"
        assert report.research_topic == "AAPL"
        assert report.sources_analyzed > 0
        assert report.confidence_score > 0.0
        assert len(report.citations) == report.sources_analyzed
        assert report.execution_time_ms >= 0.0

    def test_research_topic_wrapper_delegates(self) -> None:
        agent = _agent()
        report = asyncio.run(agent.research_topic("AAPL outlook", session_id="s-2"))
        assert report.research_topic == "AAPL outlook"

    def test_research_company_comprehensive_builds_topic_from_symbol(self) -> None:
        agent = _agent()
        report = asyncio.run(
            agent.research_company_comprehensive("MSFT", session_id="s-3")
        )
        assert "MSFT" in report.research_topic

    def test_analyze_market_sentiment_selects_sentiment_branch_and_completes(
        self,
    ) -> None:
        """`focus_areas=["sentiment", ...]` drives `_route_specialized_analysis`
        to the "sentiment" branch, exercising the fixed single-dispatch
        routing (see graph.py's module docstring) end to end -- this would
        raise `InvalidUpdateError` under the legacy dual-dispatch bug."""
        agent = _agent()
        report = asyncio.run(agent.analyze_market_sentiment("AAPL", session_id="s-4"))
        assert isinstance(report, ResearchReport)
        assert report.status == "success"


class TestPersonaConditioning:
    def test_persona_appears_in_captured_synthesis_prompt(self) -> None:
        llm = FakeChatModel(responder=_dispatching_responder)
        client = _fixture_search_client()
        agent = DeepResearchAgent(
            llm=llm, search_clients=[client], persona="aggressive"
        )

        asyncio.run(agent.research_comprehensive(topic="TSLA", session_id="s-5"))

        synthesis_prompts = [
            m
            for messages in llm.captured_prompts
            for m in messages
            if isinstance(m, HumanMessage) and "Synthesize comprehensive" in m.content
        ]
        assert len(synthesis_prompts) == 1
        assert "aggressive investor" in synthesis_prompts[0].content

    def test_persona_appears_in_content_analysis_prompts(self) -> None:
        llm = FakeChatModel(responder=_dispatching_responder)
        client = _fixture_search_client()
        agent = DeepResearchAgent(
            llm=llm, search_clients=[client], persona="conservative"
        )

        asyncio.run(agent.research_comprehensive(topic="KO", session_id="s-6"))

        analysis_prompts = [
            m
            for messages in llm.captured_prompts
            for m in messages
            if isinstance(m, HumanMessage) and "conservative investor" in m.content
        ]
        assert len(analysis_prompts) > 0


class TestErrorPropagation:
    def test_no_search_clients_raises_clean_error(self) -> None:
        llm = FakeChatModel(responder=_dispatching_responder)
        agent = DeepResearchAgent(llm=llm, search_clients=[])

        with pytest.raises(ResearchAgentError):
            asyncio.run(agent.research_comprehensive(topic="AAPL", session_id="s-7"))

    def test_all_providers_failing_completes_without_hanging(self) -> None:
        """A search client that always raises must not hang or crash the
        graph -- it degrades to zero sources, matching `_safe_search`'s
        swallow-and-continue behavior."""
        llm = FakeChatModel(responder=_dispatching_responder)
        failing_client = FakeSearchClient(fail=True)
        agent = DeepResearchAgent(llm=llm, search_clients=[failing_client])

        report = asyncio.run(
            agent.research_comprehensive(topic="AAPL", session_id="s-8")
        )
        assert report.sources_analyzed == 0
        assert report.confidence_score == 0.0

    def test_llm_failure_during_synthesis_raises_research_agent_error(self) -> None:
        def failing_synth_responder(messages: list) -> str:
            system = next(
                (m.content for m in messages if isinstance(m, SystemMessage)), ""
            )
            if "financial research synthesizer" in system:
                raise RuntimeError("simulated LLM outage")
            return _analysis_json()

        llm = FakeChatModel(responder=failing_synth_responder)
        client = _fixture_search_client()
        agent = DeepResearchAgent(llm=llm, search_clients=[client])

        with pytest.raises(ResearchAgentError):
            asyncio.run(agent.research_comprehensive(topic="AAPL", session_id="s-9"))


class TestDefaultLlmSeam:
    def test_no_llm_injected_and_unconfigured_raises_clear_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Constructing without an injected `llm` falls back to
        `platform.llm.get_llm()` -- with no `LLM_PROVIDER` configured that
        raises a clear, typed error rather than hanging or silently
        defaulting to some provider."""
        monkeypatch.delenv("LLM_PROVIDER", raising=False)

        with pytest.raises(ValueError, match="No LLM configured"):
            DeepResearchAgent(search_clients=[_fixture_search_client()])
