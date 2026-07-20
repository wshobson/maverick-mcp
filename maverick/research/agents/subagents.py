"""Specialized research subagents: fundamental, technical, sentiment, competitive.

Ported from `maverick_mcp/agents/deep_research.py`'s `BaseSubagent` and its
four subclasses (`FundamentalResearchAgent`, `TechnicalResearchAgent`,
`SentimentResearchAgent`, `CompetitiveResearchAgent`, lines 2472-3012),
which the Phase 7 decision log names explicitly as part of what ports
("`DeepResearchAgent`'s StateGraph, `ContentAnalyzer`, the subagent
specializations, and the Exa provider").

Two changes from the legacy shape:

- The four ~100-line subclasses were ~95% duplicated boilerplate (search,
  analyze, extract insights/risks/opportunities/sources, dedupe, cap list
  lengths) differing only in: query templates, max result count, analysis
  focus label, output `focus_areas`, and sentiment-aggregation strategy.
  Consolidated here into one `_run_specialized_research` function
  parameterized by a `_SubagentSpec`, with the two distinct sentiment
  algorithms legacy used (`_majority_sentiment` for
  fundamental/technical/competitive, `_weighted_sentiment` for sentiment)
  kept as separate small functions rather than duplicated four times.
  Every query template, magic number (`max_results`, confidence literals,
  dedup caps), and the weighted-sentiment threshold (`+/-0.3`) is carried
  over verbatim.
- `execute_research(self, task)` took a `ResearchTask` (from
  `maverick_mcp.utils.parallel_research`, the parallel multi-agent
  orchestrator -- see `graph.py`'s module docstring for why that
  orchestrator does not port) and only ever read `task.target_topic` off
  it. The functions here take a plain `topic: str` instead, so a subagent
  can run without the dropped orchestrator's task-queue machinery.
  `BaseSubagent.__init__(self, parent_agent)` pulled `llm` /
  `search_providers` / `content_analyzer` / `persona` off a back-reference
  to the parent `DeepResearchAgent`; those are passed as plain arguments
  here instead, since `graph.py`'s nodes already hold them directly.

`graph.py`'s specialized analysis nodes wire `run_sentiment_research`,
`run_fundamental_research`, and `run_competitive_research` into the
sequential graph in place of the legacy sequential nodes' placeholder
behavior (mutate `focus_areas`, then re-run generic content analysis --
see `graph.py`'s docstring for why that placeholder is not preserved).
`run_technical_research` remains directly callable but, matching legacy
fidelity, is not wired into `_route_specialized_analysis`'s routing table
-- that routing function never selected "technical" in legacy either
(only sentiment/fundamental/competitive/validation), so this is not a
regression.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from maverick.research.agents.analyzer import ContentAnalyzer
from maverick.research.agents.state import SearchClient
from maverick.research.types import Persona

logger = logging.getLogger(__name__)


def _generate_fundamental_queries(topic: str) -> list[str]:
    return [
        f"{topic} earnings report financial results",
        f"{topic} revenue growth profit margins",
        f"{topic} balance sheet debt ratio financial health",
        f"{topic} valuation PE ratio price earnings",
        f"{topic} cash flow dividend payout",
    ]


def _generate_technical_queries(topic: str) -> list[str]:
    return [
        f"{topic} technical analysis chart pattern",
        f"{topic} price target support resistance",
        f"{topic} RSI MACD technical indicators",
        f"{topic} breakout trend analysis",
        f"{topic} volume analysis price movement",
    ]


def _generate_sentiment_queries(topic: str) -> list[str]:
    return [
        f"{topic} analyst rating recommendation upgrade downgrade",
        f"{topic} market sentiment investor opinion",
        f"{topic} news sentiment positive negative",
        f"{topic} social sentiment reddit twitter discussion",
        f"{topic} institutional investor sentiment",
    ]


def _generate_competitive_queries(topic: str) -> list[str]:
    return [
        f"{topic} market share competitive position industry",
        f"{topic} competitors comparison competitive advantage",
        f"{topic} industry analysis market trends",
        f"{topic} competitive landscape market dynamics",
        f"{topic} industry outlook sector performance",
    ]


def _majority_sentiment(
    results: list[dict[str, Any]], confidence: float
) -> dict[str, Any]:
    """Majority-vote sentiment, used by fundamental/technical/competitive."""
    sentiments = [
        r.get("analysis", {}).get("sentiment", {}) for r in results if r.get("analysis")
    ]
    sentiments = [s for s in sentiments if s]
    if not sentiments:
        return {"direction": "neutral", "confidence": 0.5}

    bullish = sum(1 for s in sentiments if s.get("direction") == "bullish")
    bearish = sum(1 for s in sentiments if s.get("direction") == "bearish")

    if bullish > bearish:
        return {"direction": "bullish", "confidence": confidence}
    elif bearish > bullish:
        return {"direction": "bearish", "confidence": confidence}
    return {"direction": "neutral", "confidence": 0.5}


def _weighted_sentiment(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Confidence-weighted sentiment, used by the sentiment subagent."""
    sentiments = [
        r.get("analysis", {}).get("sentiment", {}) for r in results if r.get("analysis")
    ]
    sentiments = [s for s in sentiments if s]
    if not sentiments:
        return {"direction": "neutral", "confidence": 0.5}

    weighted_scores = []
    total_confidence = 0.0
    for sentiment in sentiments:
        direction = sentiment.get("direction", "neutral")
        confidence = sentiment.get("confidence", 0.5)
        if direction == "bullish":
            weighted_scores.append(confidence)
        elif direction == "bearish":
            weighted_scores.append(-confidence)
        else:
            weighted_scores.append(0.0)
        total_confidence += confidence

    avg_score = sum(weighted_scores) / len(weighted_scores)
    avg_confidence = total_confidence / len(sentiments)

    if avg_score > 0.3:
        return {"direction": "bullish", "confidence": avg_confidence}
    elif avg_score < -0.3:
        return {"direction": "bearish", "confidence": avg_confidence}
    return {"direction": "neutral", "confidence": avg_confidence}


def _average_credibility(results: list[dict[str, Any]]) -> float:
    if not results:
        return 0.5
    scores = [r.get("credibility_score", 0.5) for r in results]
    return sum(scores) / len(scores)


@dataclass(frozen=True)
class _SubagentSpec:
    research_type: str
    query_builder: Callable[[str], list[str]]
    max_results: int
    analysis_focus: str
    focus_areas: list[str]
    sentiment_fn: Callable[[list[dict[str, Any]]], dict[str, Any]]


async def _safe_search(
    client: SearchClient,
    query: str,
    num_results: int,
    timeout_budget: float | None,
) -> list[dict[str, Any]]:
    try:
        return await client.search(
            query, num_results=num_results, timeout_budget=timeout_budget
        )
    except Exception as e:
        logger.warning(f"Specialized search failed for '{query}': {e}")
        return []


async def _perform_specialized_search(
    search_clients: list[SearchClient],
    queries: list[str],
    max_results: int,
    timeout_budget: float | None,
) -> list[dict[str, Any]]:
    results_per_query = max_results // len(queries) if queries else max_results

    timeout_per_search = None
    if timeout_budget:
        total_searches = len(queries) * len(search_clients)
        timeout_per_search = timeout_budget / max(total_searches, 1)

    tasks: list[Awaitable[list[dict[str, Any]]]] = [
        _safe_search(client, query, results_per_query, timeout_per_search)
        for query in queries
        for client in search_clients
    ]

    all_results: list[dict[str, Any]] = []
    if tasks:
        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        for result in gathered:
            if isinstance(result, BaseException):
                logger.warning(f"Specialized search task failed: {result}")
            elif result:
                all_results.extend(result)

    seen_urls: set[str] = set()
    unique_results = []
    for result in all_results:
        url = result.get("url")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(result)

    return unique_results[:max_results]


async def _analyze_specialized_results(
    content_analyzer: ContentAnalyzer,
    persona: Persona,
    results: list[dict[str, Any]],
    analysis_focus: str,
) -> list[dict[str, Any]]:
    analyzed = []
    for result in results:
        if not result.get("content"):
            continue
        try:
            analysis = await content_analyzer.analyze_content(
                content=result["content"],
                persona=persona,
                analysis_focus=analysis_focus,
            )
            credibility_score = analysis.get("credibility_score", 0.5)
            analyzed.append(
                {**result, "analysis": analysis, "credibility_score": credibility_score}
            )
        except Exception as e:
            logger.warning(
                f"Specialized analysis failed for {result.get('url', 'unknown')}: {e}"
            )
    return analyzed


async def _run_specialized_research(
    *,
    content_analyzer: ContentAnalyzer,
    search_clients: list[SearchClient],
    persona: Persona,
    topic: str,
    spec: _SubagentSpec,
    timeout_budget: float | None = None,
) -> dict[str, Any]:
    queries = spec.query_builder(topic)
    search_results = await _perform_specialized_search(
        search_clients, queries, spec.max_results, timeout_budget
    )
    analyzed = await _analyze_specialized_results(
        content_analyzer, persona, search_results, spec.analysis_focus
    )

    insights: list[str] = []
    risks: list[str] = []
    opportunities: list[str] = []
    sources: list[dict[str, Any]] = []
    for result in analyzed:
        analysis = result.get("analysis", {})
        insights.extend(analysis.get("insights", []))
        risks.extend(analysis.get("risk_factors", []))
        opportunities.extend(analysis.get("opportunities", []))
        sources.append(
            {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "credibility_score": result.get("credibility_score", 0.5),
                "published_date": result.get("published_date"),
                "author": result.get("author"),
            }
        )

    return {
        "research_type": spec.research_type,
        "insights": list(dict.fromkeys(insights))[:8],
        "risk_factors": list(dict.fromkeys(risks))[:6],
        "opportunities": list(dict.fromkeys(opportunities))[:6],
        "sentiment": spec.sentiment_fn(analyzed),
        "credibility_score": _average_credibility(analyzed),
        "sources": sources,
        "focus_areas": spec.focus_areas,
    }


_FUNDAMENTAL_SPEC = _SubagentSpec(
    research_type="fundamental",
    query_builder=_generate_fundamental_queries,
    max_results=8,
    analysis_focus="fundamental_analysis",
    focus_areas=["earnings", "valuation", "financial_health", "growth_prospects"],
    sentiment_fn=lambda results: _majority_sentiment(results, 0.7),
)

_TECHNICAL_SPEC = _SubagentSpec(
    research_type="technical",
    query_builder=_generate_technical_queries,
    max_results=6,
    analysis_focus="technical_analysis",
    focus_areas=[
        "price_action",
        "chart_patterns",
        "technical_indicators",
        "support_resistance",
    ],
    sentiment_fn=lambda results: _majority_sentiment(results, 0.6),
)

_SENTIMENT_SPEC = _SubagentSpec(
    research_type="sentiment",
    query_builder=_generate_sentiment_queries,
    max_results=10,
    analysis_focus="sentiment_analysis",
    focus_areas=[
        "market_sentiment",
        "analyst_opinions",
        "news_sentiment",
        "social_sentiment",
    ],
    sentiment_fn=_weighted_sentiment,
)

_COMPETITIVE_SPEC = _SubagentSpec(
    research_type="competitive",
    query_builder=_generate_competitive_queries,
    max_results=8,
    analysis_focus="competitive_analysis",
    focus_areas=[
        "competitive_position",
        "market_share",
        "industry_trends",
        "competitive_advantages",
    ],
    sentiment_fn=lambda results: _majority_sentiment(results, 0.6),
)


async def run_fundamental_research(
    content_analyzer: ContentAnalyzer,
    search_clients: list[SearchClient],
    persona: Persona,
    topic: str,
    timeout_budget: float | None = None,
) -> dict[str, Any]:
    """Specialized fundamental-analysis research (earnings, valuation, financial health)."""
    return await _run_specialized_research(
        content_analyzer=content_analyzer,
        search_clients=search_clients,
        persona=persona,
        topic=topic,
        spec=_FUNDAMENTAL_SPEC,
        timeout_budget=timeout_budget,
    )


async def run_technical_research(
    content_analyzer: ContentAnalyzer,
    search_clients: list[SearchClient],
    persona: Persona,
    topic: str,
    timeout_budget: float | None = None,
) -> dict[str, Any]:
    """Specialized technical-analysis research (chart patterns, indicators)."""
    return await _run_specialized_research(
        content_analyzer=content_analyzer,
        search_clients=search_clients,
        persona=persona,
        topic=topic,
        spec=_TECHNICAL_SPEC,
        timeout_budget=timeout_budget,
    )


async def run_sentiment_research(
    content_analyzer: ContentAnalyzer,
    search_clients: list[SearchClient],
    persona: Persona,
    topic: str,
    timeout_budget: float | None = None,
) -> dict[str, Any]:
    """Specialized market-sentiment research (analyst ratings, news, social)."""
    return await _run_specialized_research(
        content_analyzer=content_analyzer,
        search_clients=search_clients,
        persona=persona,
        topic=topic,
        spec=_SENTIMENT_SPEC,
        timeout_budget=timeout_budget,
    )


async def run_competitive_research(
    content_analyzer: ContentAnalyzer,
    search_clients: list[SearchClient],
    persona: Persona,
    topic: str,
    timeout_budget: float | None = None,
) -> dict[str, Any]:
    """Specialized competitive/industry-landscape research."""
    return await _run_specialized_research(
        content_analyzer=content_analyzer,
        search_clients=search_clients,
        persona=persona,
        topic=topic,
        spec=_COMPETITIVE_SPEC,
        timeout_budget=timeout_budget,
    )
