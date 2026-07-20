"""Pure-function research synthesis: query generation, scoring, citations.

Ported field-for-field from `maverick_mcp/agents/deep_research.py`'s
`DeepResearchAgent` helper methods: `_generate_search_queries` (1594),
`_calculate_source_credibility` (1618), `_build_synthesis_prompt` (1657),
`_extract_key_insights` (1692), `_calculate_overall_sentiment` (1704),
`_assess_risks` (1757), `_derive_investment_implications` (1769),
`_recommend_action` (1793), `_calculate_research_confidence` (1813),
`_generate_citations` (1566-1590 of the node body).

Moved from instance methods to plain functions (nothing here reads
`self` beyond the constants already available at module scope) so
`graph.py` stays focused on graph wiring and node orchestration.

One behavioral addition, not a legacy port: `calculate_source_diversity`.
In legacy, `source_diversity_score` is initialized to `0.0` in
`research_comprehensive`'s `initial_state` and is written by exactly one
place in the whole module -- `_calculate_aggregated_sentiment` (line
2341), which only runs on the parallel multi-agent path (does not port,
see `graph.py`). The sequential graph's `_format_research_response`
(1844-1859) reads `source_diversity_score` back out, so on every
sequential-path run (the only path here) it is always `0.0` -- a second
latent bug in the same family as the router timeout-wrapper key mismatch
this task's brief calls out. Rather than perpetuate an always-zero field
in the real typed `ResearchReport`, `calculate_source_diversity` computes
it for real: fraction of validated sources with a distinct domain,
mirroring `_calculate_aggregated_sentiment`'s own diversity definition
(`len({source domain}) / len(sources)`, line 2341, adapted from URL to
domain-only since a full URL is very rarely a source's *only*
distinguishing feature but the parallel path used full URLs -- kept
close to legacy: same denominator/numerator shape, domain instead of URL
so two articles on the same authoritative site don't count as "diverse").

`_recommend_action` recomputes `_calculate_overall_sentiment(sources)`
in legacy even though `_synthesize_findings` already computed it one line
earlier for `research_findings["overall_sentiment"]`. `recommend_action`
here takes the already-computed `OverallSentiment` instead of
recomputing it -- same inputs, same result, one fewer redundant pass over
`sources`.

`merge_specialized_findings` has no legacy equivalent: it folds a
specialized subagent's pooled insights/risks/opportunities (`subagents.py`,
run when `graph.py`'s routing selects the sentiment/fundamental/
competitive branch) into the findings already built from validated
sources, by union+dedup+cap -- the same shape `extract_key_insights`/
`assess_risks`/`derive_investment_implications` already produce. It
deliberately leaves `overall_sentiment`/`confidence_score` untouched:
blending two independently-scaled confidence numbers has no single
correct answer, so the general-research sentiment (drawn from the full
validated-source set) is kept as the report's sentiment of record rather
than invent a blending formula the legacy code never had.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

from maverick.research.agents.constants import PERSONA_RESEARCH_FOCUS
from maverick.research.types import (
    InvestmentImplications,
    OverallSentiment,
    Persona,
    ResearchDepth,
    ResearchFindings,
    ResearchReport,
    SourceCitation,
)

_CREDIBILITY_THRESHOLD = 0.6


def generate_search_queries(
    topic: str, persona: Persona, max_searches: int
) -> list[str]:
    """Generate search queries optimized for the research topic and persona."""
    persona_focus = PERSONA_RESEARCH_FOCUS.get(
        persona, PERSONA_RESEARCH_FOCUS["moderate"]
    )

    base_queries = [
        f"{topic} financial analysis",
        f"{topic} investment research",
        f"{topic} market outlook",
    ]
    persona_queries = [f"{topic} {kw}" for kw in persona_focus["keywords"][:3]]
    source_queries = [f"{topic} {src}" for src in persona_focus["sources"][:2]]

    all_queries = base_queries + persona_queries + source_queries
    return all_queries[:max_searches]


def calculate_source_credibility(content: dict[str, Any]) -> float:
    """Calculate credibility score for a single analyzed source."""
    score = 0.5

    url = content.get("url", "")
    if any(domain in url for domain in [".gov", ".edu", ".org"]):
        score += 0.2
    elif any(
        domain in url
        for domain in ["sec.gov", "investopedia.com", "bloomberg.com", "reuters.com"]
    ):
        score += 0.3

    pub_date = content.get("published_date")
    if pub_date:
        try:
            date_obj = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
            now = datetime.now(date_obj.tzinfo) if date_obj.tzinfo else datetime.now()
            days_old = (now - date_obj).days
            if days_old < 30:
                score += 0.1
            elif days_old < 90:
                score += 0.05
        except (ValueError, TypeError, AttributeError):
            pass

    if "analysis" in content:
        analysis_cred = content["analysis"].get("credibility_score", 0.5)
        score = (score + analysis_cred) / 2

    return min(score, 1.0)


def meets_credibility_threshold(score: float) -> bool:
    return score >= _CREDIBILITY_THRESHOLD


def calculate_source_diversity(sources: list[dict[str, Any]]) -> float:
    """Fraction of validated sources drawn from distinct domains. See module docstring."""
    if not sources:
        return 0.0
    domains = {urlparse(s.get("url", "")).netloc for s in sources if s.get("url")}
    return len(domains) / len(sources)


def build_synthesis_prompt(
    topic: str,
    persona: Persona,
    sources: list[dict[str, Any]],
    credibility_scores: dict[str, float],
) -> str:
    """Build the synthesis prompt for the final research narrative."""
    prompt = f"""
        Synthesize comprehensive research findings on '{topic}' for a {persona} investor.

        Research Sources ({len(sources)} validated sources):
        """

    for i, source in enumerate(sources, 1):
        analysis = source.get("analysis", {})
        prompt += f"\n{i}. {source.get('title', 'Unknown Title')}"
        prompt += f"   - Insights: {', '.join(analysis.get('insights', [])[:2])}"
        prompt += f"   - Sentiment: {analysis.get('sentiment', {}).get('direction', 'neutral')}"
        prompt += f"   - Credibility: {credibility_scores.get(source.get('url', ''), 0.5):.2f}"

    prompt += f"""

        Please provide a comprehensive synthesis that includes:
        1. Executive Summary (2-3 sentences)
        2. Key Findings (5-7 bullet points)
        3. Investment Implications for {persona} investors
        4. Risk Considerations
        5. Recommended Actions
        6. Confidence Level and reasoning

        Tailor the analysis specifically for {persona} investment characteristics and risk tolerance.
        """
    return prompt


def extract_key_insights(sources: list[dict[str, Any]]) -> list[str]:
    """Extract and dedupe key insights from all analyzed sources."""
    all_insights: list[str] = []
    for source in sources:
        all_insights.extend(source.get("analysis", {}).get("insights", []))
    return list(dict.fromkeys(all_insights))[:10]


def calculate_overall_sentiment(sources: list[dict[str, Any]]) -> OverallSentiment:
    """Confidence*credibility-weighted overall sentiment across sources."""
    sentiments: list[int] = []
    weights: list[float] = []

    for source in sources:
        sentiment = source.get("analysis", {}).get("sentiment", {})
        direction = sentiment.get("direction", "neutral")
        sentiment_value = {"bullish": 1, "bearish": -1}.get(direction, 0)

        confidence = sentiment.get("confidence", 0.5)
        credibility = source.get("credibility_score", 0.5)

        sentiments.append(sentiment_value)
        weights.append(confidence * credibility)

    if not sentiments:
        return OverallSentiment(direction="neutral", confidence=0.5, consensus=0.5)

    total_weight = sum(weights)
    weighted_sentiment = (
        sum(s * w for s, w in zip(sentiments, weights, strict=False)) / total_weight
        if total_weight
        else 0.0
    )

    if weighted_sentiment > 0.2:
        overall_direction = "bullish"
    elif weighted_sentiment < -0.2:
        overall_direction = "bearish"
    else:
        overall_direction = "neutral"

    sentiment_variance = total_weight / len(sentiments) if sentiments else 0
    consensus = 1 - sentiment_variance if sentiment_variance < 1 else 0

    return OverallSentiment(
        direction=overall_direction,
        confidence=abs(weighted_sentiment),
        consensus=consensus,
        source_count=len(sentiments),
    )


def assess_risks(sources: list[dict[str, Any]]) -> list[str]:
    """Dedupe and cap risk factors from all analyzed sources."""
    all_risks: list[str] = []
    for source in sources:
        all_risks.extend(source.get("analysis", {}).get("risk_factors", []))
    return list(dict.fromkeys(all_risks))[:8]


def recommend_action(overall_sentiment: OverallSentiment, persona: Persona) -> str:
    """Recommend an investment action given the already-computed overall sentiment."""
    if overall_sentiment.direction == "bullish" and overall_sentiment.confidence > 0.7:
        if persona == "conservative":
            return "Consider gradual position building with proper risk management"
        return "Consider initiating position with appropriate position sizing"
    elif (
        overall_sentiment.direction == "bearish" and overall_sentiment.confidence > 0.7
    ):
        return "Exercise caution - consider waiting for better entry or avoiding"
    return "Monitor closely - mixed signals suggest waiting for clarity"


def derive_investment_implications(
    sources: list[dict[str, Any]],
    persona: Persona,
    overall_sentiment: OverallSentiment,
) -> InvestmentImplications:
    """Derive investment implications from research findings."""
    opportunities: list[str] = []
    threats: list[str] = []
    for source in sources:
        analysis = source.get("analysis", {})
        opportunities.extend(analysis.get("opportunities", []))
        threats.extend(analysis.get("risk_factors", []))

    persona_focus = PERSONA_RESEARCH_FOCUS.get(
        persona, PERSONA_RESEARCH_FOCUS["moderate"]
    )
    return InvestmentImplications(
        opportunities=list(dict.fromkeys(opportunities))[:5],
        threats=list(dict.fromkeys(threats))[:5],
        recommended_action=recommend_action(overall_sentiment, persona),
        time_horizon=persona_focus["time_horizon"],
    )


def calculate_research_confidence(sources: list[dict[str, Any]]) -> float:
    """Overall confidence in the research findings."""
    if not sources:
        return 0.0

    source_count_factor = min(len(sources) / 10, 1.0)
    avg_credibility = sum(s.get("credibility_score", 0.5) for s in sources) / len(
        sources
    )
    avg_relevance = sum(
        s.get("analysis", {}).get("relevance_score", 0.5) for s in sources
    ) / len(sources)

    unique_domains = len({urlparse(s["url"]).netloc for s in sources if s.get("url")})
    diversity_factor = min(unique_domains / 5, 1.0)

    confidence = (
        source_count_factor + avg_credibility + avg_relevance + diversity_factor
    ) / 4
    return round(confidence, 2)


def generate_citations(
    sources: list[dict[str, Any]], credibility_scores: dict[str, float]
) -> list[SourceCitation]:
    """Build numbered citations for all validated sources."""
    citations = []
    for i, source in enumerate(sources, 1):
        citations.append(
            SourceCitation(
                id=i,
                title=source.get("title", "Untitled"),
                url=source["url"],
                published_date=source.get("published_date"),
                author=source.get("author"),
                credibility_score=credibility_scores.get(source["url"], 0.5),
                relevance_score=source.get("analysis", {}).get("relevance_score", 0.5),
            )
        )
    return citations


def build_research_findings(
    synthesis_text: str, validated_sources: list[dict[str, Any]], persona: Persona
) -> ResearchFindings:
    """Assemble the typed `ResearchFindings` from validated sources.

    Mirrors `_synthesize_findings`'s `research_findings` dict (legacy
    lines 1546-1555); `overall_sentiment` is computed once and reused by
    `derive_investment_implications` -- see module docstring.
    """
    overall_sentiment = calculate_overall_sentiment(validated_sources)
    return ResearchFindings(
        synthesis=synthesis_text,
        key_insights=extract_key_insights(validated_sources),
        overall_sentiment=overall_sentiment,
        risk_assessment=assess_risks(validated_sources),
        investment_implications=derive_investment_implications(
            validated_sources, persona, overall_sentiment
        ),
        confidence_score=calculate_research_confidence(validated_sources),
    )


def merge_specialized_findings(
    findings: ResearchFindings, specialized: dict[str, Any]
) -> ResearchFindings:
    """Fold a specialized subagent's pooled contribution into findings. See module docstring."""
    merged_insights = list(
        dict.fromkeys([*findings.key_insights, *specialized.get("insights", [])])
    )[:10]
    merged_risks = list(
        dict.fromkeys([*findings.risk_assessment, *specialized.get("risk_factors", [])])
    )[:8]
    merged_opportunities = list(
        dict.fromkeys(
            [
                *findings.investment_implications.opportunities,
                *specialized.get("opportunities", []),
            ]
        )
    )[:5]

    return findings.model_copy(
        update={
            "key_insights": merged_insights,
            "risk_assessment": merged_risks,
            "investment_implications": findings.investment_implications.model_copy(
                update={"opportunities": merged_opportunities}
            ),
        }
    )


def format_research_report(
    *,
    persona: Persona,
    research_topic: str,
    research_depth: ResearchDepth,
    findings: ResearchFindings,
    validated_sources: list[dict[str, Any]],
    citations: list[SourceCitation],
    execution_time_ms: float,
    search_queries_used: list[str],
    source_diversity: float,
) -> ResearchReport:
    """Assemble the final typed `ResearchReport`."""
    return ResearchReport(
        status="success",
        agent_type="deep_research",
        persona=persona,
        research_topic=research_topic,
        research_depth=research_depth,
        findings=findings.model_dump(),
        sources_analyzed=len(validated_sources),
        confidence_score=findings.confidence_score,
        citations=list(citations),
        execution_time_ms=execution_time_ms,
        search_queries_used=search_queries_used,
        source_diversity=source_diversity,
    )


def utc_now() -> datetime:
    return datetime.now(UTC)
