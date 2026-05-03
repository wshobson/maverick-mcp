"""Progressive token budgeting across research phases."""

from __future__ import annotations

import time
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ResearchPhase(StrEnum):
    """Research phases for token allocation."""

    SEARCH = "search"
    CONTENT_ANALYSIS = "content_analysis"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"


class TokenAllocation(BaseModel):
    """Token allocation for a research phase."""

    input_tokens: int = Field(description="Maximum input tokens")
    output_tokens: int = Field(description="Maximum output tokens")
    per_source_tokens: int = Field(description="Tokens per source")
    emergency_reserve: int = Field(description="Emergency reserve tokens")
    timeout_seconds: float = Field(description="Processing timeout")


class ProgressiveTokenBudgeter:
    """Manages token budgets across research phases with time awareness."""

    def __init__(
        self, total_time_budget_seconds: float, confidence_target: float = 0.75
    ):
        self.total_time_budget = total_time_budget_seconds
        self.confidence_target = confidence_target
        self.phase_budgets = self._calculate_base_phase_budgets()
        self.time_started = time.time()

    def _calculate_base_phase_budgets(self) -> dict[ResearchPhase, int]:
        """Calculate base token budgets for each research phase."""
        # Allocate tokens based on typical phase requirements
        if self.total_time_budget < 30:
            # Emergency mode - minimal tokens
            return {
                ResearchPhase.SEARCH: 500,
                ResearchPhase.CONTENT_ANALYSIS: 2000,
                ResearchPhase.SYNTHESIS: 800,
                ResearchPhase.VALIDATION: 300,
            }
        elif self.total_time_budget < 60:
            # Fast mode
            return {
                ResearchPhase.SEARCH: 1000,
                ResearchPhase.CONTENT_ANALYSIS: 4000,
                ResearchPhase.SYNTHESIS: 1500,
                ResearchPhase.VALIDATION: 500,
            }
        else:
            # Standard mode
            return {
                ResearchPhase.SEARCH: 1500,
                ResearchPhase.CONTENT_ANALYSIS: 6000,
                ResearchPhase.SYNTHESIS: 2500,
                ResearchPhase.VALIDATION: 1000,
            }

    def allocate_tokens_for_phase(
        self,
        phase: ResearchPhase,
        sources_count: int,
        current_confidence: float,
        complexity_score: float = 0.5,
    ) -> TokenAllocation:
        """Allocate tokens for a research phase based on current state."""

        time_elapsed = time.time() - self.time_started
        time_remaining = max(0, self.total_time_budget - time_elapsed)

        base_budget = self.phase_budgets[phase]

        # Confidence-based scaling
        if current_confidence > self.confidence_target:
            # High confidence - focus on validation with fewer tokens
            confidence_multiplier = 0.7
        elif current_confidence < 0.4:
            # Low confidence - increase token usage if time allows
            confidence_multiplier = 1.3 if time_remaining > 30 else 0.9
        else:
            confidence_multiplier = 1.0

        # Time pressure scaling
        time_multiplier = self._calculate_time_multiplier(time_remaining)

        # Complexity scaling
        complexity_multiplier = 0.8 + (complexity_score * 0.4)  # Range: 0.8 to 1.2

        # Source count scaling (diminishing returns)
        if sources_count > 0:
            source_multiplier = min(1.0 + (sources_count - 3) * 0.05, 1.3)
        else:
            source_multiplier = 1.0

        # Calculate final budget
        final_budget = int(
            base_budget
            * confidence_multiplier
            * time_multiplier
            * complexity_multiplier
            * source_multiplier
        )

        # Calculate timeout based on available time and token budget
        base_timeout = min(time_remaining * 0.8, 45)  # Max 45 seconds per phase
        adjusted_timeout = base_timeout * (final_budget / base_budget) ** 0.5

        return TokenAllocation(
            input_tokens=min(int(final_budget * 0.75), 15000),  # Cap input tokens
            output_tokens=min(int(final_budget * 0.25), 3000),  # Cap output tokens
            per_source_tokens=final_budget // max(sources_count, 1)
            if sources_count > 0
            else final_budget,
            emergency_reserve=200,  # Always keep emergency reserve
            timeout_seconds=max(adjusted_timeout, 5),  # Minimum 5 seconds
        )

    def get_next_allocation(
        self,
        sources_remaining: int,
        current_confidence: float,
        time_elapsed_seconds: float,
    ) -> dict[str, Any]:
        """Get the next token allocation for processing sources."""
        time_remaining = max(0, self.total_time_budget - time_elapsed_seconds)

        # Determine priority based on confidence and time pressure
        if current_confidence < 0.4 and time_remaining > 30:
            priority = "high"
        elif current_confidence < 0.6 and time_remaining > 15:
            priority = "medium"
        else:
            priority = "low"

        # Calculate time budget per remaining source
        if sources_remaining > 0:
            time_per_source = time_remaining / sources_remaining
        else:
            time_per_source = 0

        # Calculate token budget
        base_tokens = self.phase_budgets.get(ResearchPhase.CONTENT_ANALYSIS, 2000)

        if priority == "high":
            max_tokens = min(int(base_tokens * 1.2), 4000)
        elif priority == "medium":
            max_tokens = base_tokens
        else:
            max_tokens = int(base_tokens * 0.8)

        return {
            "time_budget": min(time_per_source, 30.0),  # Cap at 30 seconds
            "max_tokens": max_tokens,
            "priority": priority,
            "sources_remaining": sources_remaining,
        }

    def _calculate_time_multiplier(self, time_remaining: float) -> float:
        """Scale token budget based on time pressure."""
        if time_remaining < 5:
            return 0.2  # Extreme emergency mode
        elif time_remaining < 15:
            return 0.4  # Emergency mode
        elif time_remaining < 30:
            return 0.7  # Time-constrained
        elif time_remaining < 60:
            return 0.9  # Slightly reduced
        else:
            return 1.0  # Full budget available
