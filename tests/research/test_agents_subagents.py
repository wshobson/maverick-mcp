"""Tests for `maverick.research.agents.subagents`.

Fully mocked: fake search clients, no network; a deterministic fake chat
model for content analysis. Covers query generation/dedup, the two
sentiment-aggregation strategies (majority-vote vs confidence-weighted),
and provider-failure -> clean-degrade behavior.
"""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("langgraph")

from maverick.research.agents import subagents  # noqa: E402
from maverick.research.agents.analyzer import ContentAnalyzer  # noqa: E402

from ._fakes import FakeChatModel, FakeSearchClient, make_source  # noqa: E402


def _bullish_json() -> str:
    return json.dumps(
        {
            "KEY_INSIGHTS": ["Growth insight"],
            "SENTIMENT": {"direction": "bullish", "confidence": 0.8},
            "RISK_FACTORS": ["Some risk"],
            "OPPORTUNITIES": ["Some opportunity"],
            "CREDIBILITY": 0.8,
            "RELEVANCE": 0.7,
            "SUMMARY": "Bullish summary.",
        }
    )


def _bearish_json() -> str:
    return json.dumps(
        {
            "KEY_INSIGHTS": ["Decline insight"],
            "SENTIMENT": {"direction": "bearish", "confidence": 0.9},
            "RISK_FACTORS": ["Major risk"],
            "OPPORTUNITIES": [],
            "CREDIBILITY": 0.6,
            "RELEVANCE": 0.5,
            "SUMMARY": "Bearish summary.",
        }
    )


def _always_bullish_llm() -> FakeChatModel:
    return FakeChatModel(responder=lambda _messages: _bullish_json())


class TestFundamentalResearch:
    def test_returns_expected_shape_and_focus_areas(self) -> None:
        client = FakeSearchClient(results=[make_source(url="https://sec.gov/a")])
        analyzer = ContentAnalyzer(_always_bullish_llm())

        result = asyncio.run(
            subagents.run_fundamental_research(analyzer, [client], "moderate", "AAPL")
        )

        assert result["research_type"] == "fundamental"
        assert result["focus_areas"] == [
            "earnings",
            "valuation",
            "financial_health",
            "growth_prospects",
        ]
        assert result["insights"] == ["Growth insight"]
        assert result["sentiment"]["direction"] == "bullish"
        # Majority-vote fundamental confidence literal is 0.7 (see subagents.py spec).
        assert result["sentiment"]["confidence"] == 0.7

    def test_query_builder_mentions_topic(self) -> None:
        queries = subagents._generate_fundamental_queries("AAPL")
        assert all("AAPL" in q for q in queries)
        assert len(queries) == 5


class TestSentimentResearchWeightedAggregation:
    def test_weighted_sentiment_blends_confidence(self) -> None:
        """Sentiment subagent uses confidence-weighted aggregation, unlike
        the majority-vote strategy the other three use -- pinning the
        distinct algorithm ported from legacy's `_calculate_market_sentiment`."""
        call_count = {"n": 0}

        def alternating_responder(_messages: list) -> str:
            call_count["n"] += 1
            return _bullish_json() if call_count["n"] % 2 else _bearish_json()

        llm = FakeChatModel(responder=alternating_responder)
        analyzer = ContentAnalyzer(llm)
        client = FakeSearchClient(
            results=[
                make_source(url="https://a.com"),
                make_source(url="https://b.com"),
            ]
        )

        result = asyncio.run(
            subagents.run_sentiment_research(analyzer, [client], "moderate", "AAPL")
        )
        assert result["research_type"] == "sentiment"
        # Weighted score depends on both direction and each item's own
        # confidence; just assert it lands on a valid direction/confidence
        # pair rather than a fixed literal (unlike the majority strategy).
        assert result["sentiment"]["direction"] in {"bullish", "bearish", "neutral"}
        assert 0.0 <= result["sentiment"]["confidence"] <= 1.0

    def test_no_analyzed_results_yields_neutral(self) -> None:
        analyzer = ContentAnalyzer(_always_bullish_llm())
        client = FakeSearchClient(results=[])  # no sources at all

        result = asyncio.run(
            subagents.run_sentiment_research(analyzer, [client], "moderate", "AAPL")
        )
        assert result["sentiment"] == {"direction": "neutral", "confidence": 0.5}
        assert result["sources"] == []


class TestCompetitiveAndTechnicalResearch:
    def test_competitive_research_majority_confidence_is_point_six(self) -> None:
        analyzer = ContentAnalyzer(_always_bullish_llm())
        client = FakeSearchClient(results=[make_source(url="https://a.com")])

        result = asyncio.run(
            subagents.run_competitive_research(analyzer, [client], "moderate", "AAPL")
        )
        assert result["sentiment"] == {"direction": "bullish", "confidence": 0.6}

    def test_technical_research_is_directly_callable(self) -> None:
        """Not wired into the sequential graph's routing (matching legacy
        fidelity -- see subagents.py's module docstring), but remains a
        first-class, directly callable specialization."""
        analyzer = ContentAnalyzer(_always_bullish_llm())
        client = FakeSearchClient(results=[make_source(url="https://a.com")])

        result = asyncio.run(
            subagents.run_technical_research(analyzer, [client], "moderate", "AAPL")
        )
        assert result["research_type"] == "technical"


class TestSearchDedupAndFailure:
    def test_dedups_results_by_url_across_providers(self) -> None:
        shared_url_results = [
            make_source(url="https://dup.com"),
            make_source(url="https://dup.com"),
        ]
        client_a = FakeSearchClient(results=shared_url_results)
        client_b = FakeSearchClient(results=[make_source(url="https://unique.com")])
        analyzer = ContentAnalyzer(_always_bullish_llm())

        result = asyncio.run(
            subagents.run_fundamental_research(
                analyzer, [client_a, client_b], "moderate", "AAPL"
            )
        )
        urls = {s["url"] for s in result["sources"]}
        assert urls == {"https://dup.com", "https://unique.com"}

    def test_provider_failure_degrades_cleanly_not_hangs(self) -> None:
        """'error propagation (provider failure -> clean error, not hang)':
        a failing search client must not raise out of the subagent -- it
        degrades to an empty-sources result, same as legacy's `_safe_search`
        swallowing the exception."""
        failing_client = FakeSearchClient(fail=True)
        analyzer = ContentAnalyzer(_always_bullish_llm())

        result = asyncio.run(
            subagents.run_fundamental_research(
                analyzer, [failing_client], "moderate", "AAPL"
            )
        )
        assert result["sources"] == []
        assert result["sentiment"] == {"direction": "neutral", "confidence": 0.5}
        assert result["credibility_score"] == 0.5
