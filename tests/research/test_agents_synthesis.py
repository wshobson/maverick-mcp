"""Tests for `maverick.research.agents.synthesis` (pure functions).

Fully mocked: no network, no LLM calls -- everything here is
deterministic pure-function scoring/dedup logic operating on fixture
content, per this task's brief ("ContentAnalyzer scoring/dedup logic
pinned on fixture content").
"""

from __future__ import annotations

import pytest

pytest.importorskip("langgraph")

from maverick.research.agents import synthesis  # noqa: E402


def _analyzed_source(
    *,
    url: str,
    title: str = "Title",
    published_date: str | None = None,
    insights: list[str] | None = None,
    risk_factors: list[str] | None = None,
    opportunities: list[str] | None = None,
    sentiment_direction: str = "bullish",
    sentiment_confidence: float = 0.8,
    credibility_score: float = 0.7,
    relevance_score: float = 0.6,
) -> dict:
    return {
        "url": url,
        "title": title,
        "published_date": published_date,
        "credibility_score": credibility_score,
        "analysis": {
            "insights": insights or [],
            "risk_factors": risk_factors or [],
            "opportunities": opportunities or [],
            "sentiment": {
                "direction": sentiment_direction,
                "confidence": sentiment_confidence,
            },
            "relevance_score": relevance_score,
        },
    }


class TestSourceCredibility:
    def test_gov_domain_scores_higher_than_baseline(self) -> None:
        baseline = synthesis.calculate_source_credibility(
            {"url": "https://blog.example.com"}
        )
        gov = synthesis.calculate_source_credibility({"url": "https://sec.gov/filing"})
        assert gov > baseline

    def test_recent_publication_date_boosts_score(self) -> None:
        import datetime

        recent = (
            datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=5)
        ).isoformat()
        old = "2015-01-01T00:00:00+00:00"

        recent_score = synthesis.calculate_source_credibility(
            {"url": "https://example.com", "published_date": recent}
        )
        old_score = synthesis.calculate_source_credibility(
            {"url": "https://example.com", "published_date": old}
        )
        assert recent_score > old_score

    def test_malformed_date_does_not_raise(self) -> None:
        score = synthesis.calculate_source_credibility(
            {"url": "https://example.com", "published_date": "not-a-date"}
        )
        assert 0.0 <= score <= 1.0

    def test_meets_credibility_threshold(self) -> None:
        assert synthesis.meets_credibility_threshold(0.6) is True
        assert synthesis.meets_credibility_threshold(0.59) is False


class TestSourceDiversity:
    def test_empty_sources_zero(self) -> None:
        assert synthesis.calculate_source_diversity([]) == 0.0

    def test_all_same_domain_low_diversity(self) -> None:
        sources = [{"url": f"https://example.com/{i}"} for i in range(3)]
        assert synthesis.calculate_source_diversity(sources) == pytest.approx(1 / 3)

    def test_all_distinct_domains_full_diversity(self) -> None:
        sources = [
            {"url": "https://a.com"},
            {"url": "https://b.com"},
            {"url": "https://c.com"},
        ]
        assert synthesis.calculate_source_diversity(sources) == 1.0


class TestExtractKeyInsightsAndRisks:
    def test_dedupes_and_caps_at_ten(self) -> None:
        sources = [
            _analyzed_source(
                url=f"https://s{i}.com", insights=["Same insight", f"Unique {i}"]
            )
            for i in range(12)
        ]
        insights = synthesis.extract_key_insights(sources)
        assert len(insights) == 10
        assert (
            insights[0] == "Same insight"
        )  # first occurrence kept, dedup preserves order

    def test_assess_risks_caps_at_eight(self) -> None:
        sources = [
            _analyzed_source(url=f"https://s{i}.com", risk_factors=[f"Risk {i}"])
            for i in range(10)
        ]
        risks = synthesis.assess_risks(sources)
        assert len(risks) == 8


class TestOverallSentiment:
    def test_no_sources_neutral(self) -> None:
        sentiment = synthesis.calculate_overall_sentiment([])
        assert sentiment.direction == "neutral"
        assert sentiment.source_count is None

    def test_majority_bullish_sources_yield_bullish_direction(self) -> None:
        sources = [
            _analyzed_source(
                url="https://a.com",
                sentiment_direction="bullish",
                sentiment_confidence=0.9,
            ),
            _analyzed_source(
                url="https://b.com",
                sentiment_direction="bullish",
                sentiment_confidence=0.9,
            ),
            _analyzed_source(
                url="https://c.com",
                sentiment_direction="bearish",
                sentiment_confidence=0.3,
            ),
        ]
        sentiment = synthesis.calculate_overall_sentiment(sources)
        assert sentiment.direction == "bullish"
        assert sentiment.source_count == 3

    def test_all_bearish_yields_bearish_direction(self) -> None:
        sources = [
            _analyzed_source(
                url="https://a.com",
                sentiment_direction="bearish",
                sentiment_confidence=0.8,
            ),
        ]
        sentiment = synthesis.calculate_overall_sentiment(sources)
        assert sentiment.direction == "bearish"


class TestRecommendAction:
    def test_high_confidence_bullish_conservative_recommends_gradual(self) -> None:
        sentiment = synthesis.OverallSentiment(
            direction="bullish", confidence=0.8, consensus=0.9
        )
        action = synthesis.recommend_action(sentiment, "conservative")
        assert "gradual" in action.lower()

    def test_high_confidence_bullish_aggressive_recommends_initiating(self) -> None:
        sentiment = synthesis.OverallSentiment(
            direction="bullish", confidence=0.8, consensus=0.9
        )
        action = synthesis.recommend_action(sentiment, "aggressive")
        assert "initiating" in action.lower()

    def test_high_confidence_bearish_recommends_caution(self) -> None:
        sentiment = synthesis.OverallSentiment(
            direction="bearish", confidence=0.8, consensus=0.9
        )
        action = synthesis.recommend_action(sentiment, "moderate")
        assert "caution" in action.lower()

    def test_low_confidence_recommends_monitoring(self) -> None:
        sentiment = synthesis.OverallSentiment(
            direction="bullish", confidence=0.3, consensus=0.5
        )
        action = synthesis.recommend_action(sentiment, "moderate")
        assert "monitor" in action.lower()


class TestResearchConfidence:
    def test_no_sources_zero_confidence(self) -> None:
        assert synthesis.calculate_research_confidence([]) == 0.0

    def test_more_diverse_credible_sources_increase_confidence(self) -> None:
        few = [_analyzed_source(url="https://a.com", credibility_score=0.5)]
        many_diverse = [
            _analyzed_source(url=f"https://s{i}.com", credibility_score=0.9)
            for i in range(10)
        ]
        assert synthesis.calculate_research_confidence(
            many_diverse
        ) > synthesis.calculate_research_confidence(few)


class TestBuildResearchFindingsAndMerge:
    def test_build_research_findings_shape(self) -> None:
        sources = [
            _analyzed_source(
                url="https://a.com",
                insights=["Insight A"],
                risk_factors=["Risk A"],
                opportunities=["Opportunity A"],
            )
        ]
        findings = synthesis.build_research_findings(
            "synthesis text", sources, "moderate"
        )
        assert findings.synthesis == "synthesis text"
        assert findings.key_insights == ["Insight A"]
        assert findings.risk_assessment == ["Risk A"]
        assert findings.investment_implications.opportunities == ["Opportunity A"]
        assert findings.investment_implications.time_horizon == "medium-term"

    def test_merge_specialized_findings_unions_and_caps(self) -> None:
        sources = [_analyzed_source(url="https://a.com", insights=["Base insight"])]
        findings = synthesis.build_research_findings("text", sources, "moderate")

        specialized = {
            "insights": ["Base insight", "Specialized insight"],
            "risk_factors": ["Specialized risk"],
            "opportunities": ["Specialized opportunity"],
        }
        merged = synthesis.merge_specialized_findings(findings, specialized)

        assert merged.key_insights == ["Base insight", "Specialized insight"]
        assert "Specialized risk" in merged.risk_assessment
        assert "Specialized opportunity" in merged.investment_implications.opportunities
        # overall_sentiment/confidence_score are left untouched (see module docstring)
        assert merged.overall_sentiment == findings.overall_sentiment
        assert merged.confidence_score == findings.confidence_score


class TestGenerateCitationsAndReport:
    def test_generate_citations_numbers_sequentially(self) -> None:
        sources = [
            {
                "url": "https://a.com",
                "title": "A",
                "published_date": None,
                "author": None,
            },
            {
                "url": "https://b.com",
                "title": "B",
                "published_date": "2024-01-01",
                "author": "X",
            },
        ]
        citations = synthesis.generate_citations(
            sources, {"https://a.com": 0.9, "https://b.com": 0.4}
        )
        assert [c.id for c in citations] == [1, 2]
        assert citations[0].credibility_score == 0.9
        assert citations[1].author == "X"

    def test_format_research_report_round_trip(self) -> None:
        sources = [_analyzed_source(url="https://a.com")]
        findings = synthesis.build_research_findings("text", sources, "moderate")
        citations = synthesis.generate_citations(sources, {"https://a.com": 0.7})

        report = synthesis.format_research_report(
            persona="moderate",
            research_topic="AAPL",
            research_depth="standard",
            findings=findings,
            validated_sources=sources,
            citations=citations,
            execution_time_ms=123.0,
            search_queries_used=["AAPL financial analysis"],
            source_diversity=1.0,
        )
        assert report.status == "success"
        assert report.sources_analyzed == 1
        assert report.confidence_score == findings.confidence_score
        assert report.source_diversity == 1.0


class TestGenerateSearchQueries:
    def test_capped_at_max_searches(self) -> None:
        queries = synthesis.generate_search_queries("AAPL", "moderate", max_searches=2)
        assert len(queries) == 2

    def test_includes_topic_in_every_query(self) -> None:
        queries = synthesis.generate_search_queries(
            "AAPL", "aggressive", max_searches=8
        )
        assert all("AAPL" in q for q in queries)
