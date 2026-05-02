"""Bayesian-style confidence tracker with early-termination heuristics."""

from __future__ import annotations

from datetime import datetime
from typing import Any


class ConfidenceTracker:
    """Tracks research confidence and triggers early termination when appropriate."""

    def __init__(
        self,
        target_confidence: float = 0.75,
        min_sources: int = 3,
        max_sources: int = 15,
    ):
        self.target_confidence = target_confidence
        self.min_sources = min_sources
        self.max_sources = max_sources
        self.confidence_history = []
        self.evidence_history = []
        self.source_count = 0
        self.sources_analyzed = 0  # For backward compatibility
        self.last_significant_improvement = 0
        self.sentiment_votes = {"bullish": 0, "bearish": 0, "neutral": 0}

    def update_confidence(
        self,
        new_evidence: dict,
        source_credibility: float | None = None,
        credibility_score: float | None = None,
    ) -> dict[str, Any]:
        """Update confidence based on new evidence and return continuation decision."""

        # Handle both parameter names for backward compatibility
        if source_credibility is None and credibility_score is not None:
            source_credibility = credibility_score
        elif source_credibility is None and credibility_score is None:
            source_credibility = 0.5  # Default value

        self.source_count += 1
        self.sources_analyzed += 1  # Keep both for compatibility

        # Store evidence
        self.evidence_history.append(
            {
                "evidence": new_evidence,
                "credibility": source_credibility,
                "timestamp": datetime.now(),
            }
        )

        # Update sentiment voting
        sentiment = new_evidence.get("sentiment", {})
        direction = sentiment.get("direction", "neutral")
        confidence = sentiment.get("confidence", 0.5)

        # Weight vote by source credibility and sentiment confidence
        vote_weight = source_credibility * confidence
        self.sentiment_votes[direction] += vote_weight

        # Calculate evidence strength
        evidence_strength = self._calculate_evidence_strength(
            new_evidence, source_credibility
        )

        # Update confidence using Bayesian-style updating
        current_confidence = self._update_bayesian_confidence(evidence_strength)
        self.confidence_history.append(current_confidence)

        # Check for significant improvement
        if len(self.confidence_history) >= 2:
            improvement = current_confidence - self.confidence_history[-2]
            if improvement > 0.1:  # 10% improvement
                self.last_significant_improvement = self.source_count

        # Make continuation decision
        should_continue = self._should_continue_research(current_confidence)

        return {
            "current_confidence": current_confidence,
            "should_continue": should_continue,
            "sources_processed": self.source_count,
            "sources_analyzed": self.source_count,  # For backward compatibility
            "confidence_trend": self._calculate_confidence_trend(),
            "early_termination_reason": None
            if should_continue
            else self._get_termination_reason(current_confidence),
            "sentiment_consensus": self._calculate_sentiment_consensus(),
        }

    def _calculate_evidence_strength(self, evidence: dict, credibility: float) -> float:
        """Calculate the strength of new evidence."""

        # Base strength from sentiment confidence
        sentiment = evidence.get("sentiment", {})
        sentiment_confidence = sentiment.get("confidence", 0.5)

        # Adjust for source credibility
        credibility_adjusted = sentiment_confidence * credibility

        # Factor in evidence richness
        insights_count = len(evidence.get("insights", []))
        risk_factors_count = len(evidence.get("risk_factors", []))
        opportunities_count = len(evidence.get("opportunities", []))

        # Evidence richness score (0-1)
        evidence_richness = min(
            (insights_count + risk_factors_count + opportunities_count) / 12, 1.0
        )

        # Relevance factor
        relevance_score = evidence.get("relevance_score", 0.5)

        # Final evidence strength calculation
        final_strength = credibility_adjusted * (
            0.5 + 0.3 * evidence_richness + 0.2 * relevance_score
        )

        return min(final_strength, 1.0)

    def _update_bayesian_confidence(self, evidence_strength: float) -> float:
        """Update confidence using Bayesian approach."""

        if not self.confidence_history:
            # First evidence - base confidence
            return evidence_strength

        # Current prior
        prior = self.confidence_history[-1]

        # Bayesian update with evidence strength as likelihood
        # Simple approximation: weighted average with decay
        decay_factor = 0.9 ** (self.source_count - 1)  # Diminishing returns

        updated = prior * decay_factor + evidence_strength * (1 - decay_factor)

        # Ensure within bounds
        return max(0.1, min(updated, 0.95))

    def _should_continue_research(self, current_confidence: float) -> bool:
        """Determine if research should continue based on multiple factors."""

        # Always process minimum sources
        if self.source_count < self.min_sources:
            return True

        # Stop at maximum sources
        if self.source_count >= self.max_sources:
            return False

        # High confidence reached
        if current_confidence >= self.target_confidence:
            return False

        # Check for diminishing returns
        if self.source_count - self.last_significant_improvement > 4:
            # No significant improvement in last 4 sources
            return False

        # Check sentiment consensus
        consensus_score = self._calculate_sentiment_consensus()
        if consensus_score > 0.8 and self.source_count >= 5:
            # Strong consensus with adequate sample
            return False

        # Check confidence plateau
        if len(self.confidence_history) >= 3:
            recent_change = abs(current_confidence - self.confidence_history[-3])
            if recent_change < 0.03:  # Less than 3% change in last 3 sources
                return False

        return True

    def _calculate_confidence_trend(self) -> str:
        """Calculate the trend in confidence over recent sources."""

        if len(self.confidence_history) < 3:
            return "insufficient_data"

        recent = self.confidence_history[-3:]

        # Calculate trend
        if recent[-1] > recent[0] + 0.05:
            return "increasing"
        elif recent[-1] < recent[0] - 0.05:
            return "decreasing"
        else:
            return "stable"

    def _calculate_sentiment_consensus(self) -> float:
        """Calculate how much sources agree on sentiment."""

        total_votes = sum(self.sentiment_votes.values())
        if total_votes == 0:
            return 0.0

        # Calculate consensus as max vote share
        max_votes = max(self.sentiment_votes.values())
        consensus = max_votes / total_votes

        return consensus

    def _get_termination_reason(self, current_confidence: float) -> str:
        """Get reason for early termination."""

        if current_confidence >= self.target_confidence:
            return "target_confidence_reached"
        elif self.source_count >= self.max_sources:
            return "max_sources_reached"
        elif self._calculate_sentiment_consensus() > 0.8:
            return "strong_consensus"
        elif self.source_count - self.last_significant_improvement > 4:
            return "diminishing_returns"
        else:
            return "confidence_plateau"
