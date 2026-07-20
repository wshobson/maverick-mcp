"""AI-powered content analysis for research results.

Ported from `maverick_mcp/agents/deep_research.py`'s `ContentAnalyzer`
(lines 134-351). `analyze_content`, `_fallback_analysis`,
`_coerce_message_content`, and `analyze_content_batch` port verbatim
(batching logic, fallback heuristics, and prompt text unchanged).

Three legacy methods do NOT port: `analyze_content_items`,
`_analyze_single_content`, `_extract_themes`. Each is explicitly
comment-marked in the legacy source as existing "for test compatibility"
/ "used by tests" and has zero callers among `DeepResearchAgent`'s graph
nodes or public research methods -- they were internal-API surface kept
alive only for old unit tests poking at `ContentAnalyzer` directly, not
production code paths. Dropped as dead weight per this task's directive
to port only what the surviving code paths reach.

Unlike legacy (where `_analyze_content` calls `analyze_content` once per
source in a plain `for` loop, leaving `analyze_content_batch` unused by
the production graph), `graph.py`'s `_analyze_content` node calls
`analyze_content_batch` directly -- turning previously dead code into the
live concurrent-analysis path (batches of 4, `asyncio.gather`), a
same-shape, strictly-better-performing substitution disclosed here rather
than left as an unreachable method.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from maverick.research.agents.constants import PERSONA_RESEARCH_FOCUS
from maverick.research.types import Persona

logger = logging.getLogger(__name__)

_BATCH_SIZE = 4
_CONTENT_CHARS = 3000


class ContentAnalyzer:
    """AI-powered content analysis with batch processing capability."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self._batch_size = _BATCH_SIZE

    @staticmethod
    def _coerce_message_content(raw_content: Any) -> str:
        """Convert LLM response content to a string for JSON parsing."""
        if isinstance(raw_content, str):
            return raw_content

        if isinstance(raw_content, list):
            parts: list[str] = []
            for item in raw_content:
                if isinstance(item, dict):
                    text_value = item.get("text")
                    parts.append(
                        text_value if isinstance(text_value, str) else str(text_value)
                    )
                else:
                    parts.append(str(item))
            return "".join(parts)

        return str(raw_content)

    async def analyze_content(
        self, content: str, persona: Persona, analysis_focus: str = "general"
    ) -> dict[str, Any]:
        """Analyze content with AI for insights, sentiment, and relevance."""
        persona_focus = PERSONA_RESEARCH_FOCUS.get(
            persona, PERSONA_RESEARCH_FOCUS["moderate"]
        )

        analysis_prompt = f"""
        Analyze this financial content from the perspective of a {persona} investor.

        Content to analyze:
        {content[:_CONTENT_CHARS]}

        Focus Areas: {", ".join(persona_focus["keywords"])}
        Risk Focus: {persona_focus["risk_focus"]}
        Time Horizon: {persona_focus["time_horizon"]}

        Provide analysis in the following structure:

        1. KEY_INSIGHTS: 3-5 bullet points of most important insights
        2. SENTIMENT: Overall sentiment (bullish/bearish/neutral) with confidence (0-1)
        3. RISK_FACTORS: Key risks identified relevant to {persona} investors
        4. OPPORTUNITIES: Investment opportunities or catalysts identified
        5. CREDIBILITY: Assessment of source credibility (0-1 score)
        6. RELEVANCE: How relevant is this to {persona} investment strategy (0-1 score)
        7. SUMMARY: 2-3 sentence summary for {persona} investors

        Format as JSON with clear structure.
        """

        try:
            response = await self.llm.ainvoke(
                [
                    SystemMessage(
                        content="You are a financial content analyst. Return only valid JSON."
                    ),
                    HumanMessage(content=analysis_prompt),
                ]
            )

            raw_content = self._coerce_message_content(response.content).strip()
            analysis = json.loads(raw_content)

            return {
                "insights": analysis.get("KEY_INSIGHTS", []),
                "sentiment": {
                    "direction": analysis.get("SENTIMENT", {}).get(
                        "direction", "neutral"
                    ),
                    "confidence": analysis.get("SENTIMENT", {}).get("confidence", 0.5),
                },
                "risk_factors": analysis.get("RISK_FACTORS", []),
                "opportunities": analysis.get("OPPORTUNITIES", []),
                "credibility_score": analysis.get("CREDIBILITY", 0.5),
                "relevance_score": analysis.get("RELEVANCE", 0.5),
                "summary": analysis.get("SUMMARY", ""),
                "analysis_timestamp": datetime.now(UTC),
            }

        except Exception as e:
            logger.warning(f"AI content analysis failed: {e}, using fallback")
            return self._fallback_analysis(content, persona)

    def _fallback_analysis(self, content: str, persona: Persona) -> dict[str, Any]:
        """Fallback analysis using keyword matching."""
        persona_focus = PERSONA_RESEARCH_FOCUS.get(
            persona, PERSONA_RESEARCH_FOCUS["moderate"]
        )

        content_lower = content.lower()

        positive_words = [
            "growth",
            "increase",
            "profit",
            "success",
            "opportunity",
            "strong",
        ]
        negative_words = ["decline", "loss", "risk", "problem", "concern", "weak"]

        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)

        if positive_count > negative_count:
            sentiment = "bullish"
            confidence = 0.6
        elif negative_count > positive_count:
            sentiment = "bearish"
            confidence = 0.6
        else:
            sentiment = "neutral"
            confidence = 0.5

        keywords = persona_focus["keywords"]
        keyword_matches = sum(1 for keyword in keywords if keyword in content_lower)
        relevance_score = min(keyword_matches / len(keywords), 1.0)

        return {
            "insights": [f"Fallback analysis for {persona} investor perspective"],
            "sentiment": {"direction": sentiment, "confidence": confidence},
            "risk_factors": ["Unable to perform detailed risk analysis"],
            "opportunities": ["Unable to identify specific opportunities"],
            "credibility_score": 0.5,
            "relevance_score": relevance_score,
            "summary": f"Content analysis for {persona} investor using fallback method",
            "analysis_timestamp": datetime.now(UTC),
            "fallback_used": True,
        }

    async def analyze_content_batch(
        self,
        content_items: list[tuple[str, str]],
        persona: Persona,
        analysis_focus: str = "general",
    ) -> list[dict[str, Any]]:
        """Analyze multiple content items in parallel batches.

        Args:
            content_items: List of (content, source_identifier) tuples.
            persona: Investor persona for analysis perspective.
            analysis_focus: Focus area for analysis.

        Returns:
            List of analysis results in the same order as the input.
        """
        if not content_items:
            return []

        results: list[dict[str, Any]] = []
        for i in range(0, len(content_items), self._batch_size):
            batch = content_items[i : i + self._batch_size]
            tasks = [
                self.analyze_content(content, persona, analysis_focus)
                for content, _ in batch
            ]

            try:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                for j, result in enumerate(batch_results):
                    content, source_id = batch[j]
                    if isinstance(result, BaseException):
                        logger.warning(
                            f"Batch analysis failed for item {i + j}: {result}"
                        )
                        fallback_result = self._fallback_analysis(content, persona)
                        fallback_result["source_identifier"] = source_id
                        fallback_result["batch_processed"] = True
                        results.append(fallback_result)
                    else:
                        enriched_result = dict(result)
                        enriched_result["source_identifier"] = source_id
                        enriched_result["batch_processed"] = True
                        results.append(enriched_result)

            except Exception as e:
                logger.error(f"Batch analysis completely failed: {e}")
                for content, source_id in batch:
                    fallback_result = self._fallback_analysis(content, persona)
                    fallback_result["source_identifier"] = source_id
                    fallback_result["batch_processed"] = True
                    fallback_result["batch_error"] = str(e)
                    results.append(fallback_result)

        logger.info(
            f"Batch content analysis completed: {len(content_items)} items processed "
            f"in {(len(content_items) + self._batch_size - 1) // self._batch_size} batches"
        )

        return results
