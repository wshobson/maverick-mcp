"""Cost tracking and budget enforcement for LLM calls.

This module provides a CostAccumulator that tracks estimated and actual costs
per LLM call, enforces per-request and daily spending limits, and raises
BudgetExceededError when limits are breached.
"""

import asyncio
import logging
import os
import time
from datetime import date

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BudgetExceededError(Exception):
    """Raised when an LLM call would exceed the configured budget.

    Attributes:
        estimated_cost: The estimated cost of the blocked call.
        budget_type: Whether the per-request or daily budget was exceeded.
        current_total: The current accumulated total for that budget scope.
        limit: The configured limit that would be exceeded.
    """

    def __init__(
        self,
        estimated_cost: float,
        budget_type: str,
        current_total: float,
        limit: float,
    ) -> None:
        self.estimated_cost = estimated_cost
        self.budget_type = budget_type
        self.current_total = current_total
        self.limit = limit
        super().__init__(
            f"LLM call blocked: estimated ${estimated_cost:.4f} would exceed "
            f"{budget_type} budget (current: ${current_total:.4f}, limit: ${limit:.2f})"
        )


class CostRecord(BaseModel):
    """Record of cost for a single LLM call."""

    model_id: str = Field(description="Model identifier")
    input_tokens: int = Field(default=0, description="Input tokens used")
    output_tokens: int = Field(default=0, description="Output tokens used")
    estimated_cost: float = Field(default=0.0, description="Estimated cost in USD")
    actual_cost: float = Field(default=0.0, description="Actual cost in USD")
    timestamp: float = Field(
        default_factory=time.time, description="Unix timestamp of the call"
    )
    request_id: str | None = Field(
        default=None, description="Request ID for grouping calls"
    )


class CostAccumulator:
    """Thread-safe cost accumulator with per-request and daily budget enforcement.

    Tracks estimated cost before each LLM call and actual cost after,
    maintaining running totals per request_id and per calendar day.

    Budget limits are read from environment variables:
        MAVERICK_MAX_COST_PER_REQUEST  (default: $1.00)
        MAVERICK_MAX_DAILY_COST        (default: $50.00)
    """

    def __init__(
        self,
        max_cost_per_request: float | None = None,
        max_daily_cost: float | None = None,
    ) -> None:
        """Initialize the cost accumulator.

        Args:
            max_cost_per_request: Maximum cost per logical request in USD.
                Falls back to MAVERICK_MAX_COST_PER_REQUEST env var, then $1.00.
            max_daily_cost: Maximum daily cost in USD.
                Falls back to MAVERICK_MAX_DAILY_COST env var, then $50.00.
        """
        self.max_cost_per_request = max_cost_per_request or float(
            os.getenv("MAVERICK_MAX_COST_PER_REQUEST", "1.00")
        )
        self.max_daily_cost = max_daily_cost or float(
            os.getenv("MAVERICK_MAX_DAILY_COST", "50.00")
        )

        # Running totals
        self._request_totals: dict[str, float] = {}
        self._daily_total: float = 0.0
        self._current_day: date = date.today()

        # Full history for analytics
        self._records: list[CostRecord] = []

        # asyncio lock for thread safety in async context
        self._lock = asyncio.Lock()

        logger.info(
            f"CostAccumulator initialized: max_per_request=${self.max_cost_per_request:.2f}, "
            f"max_daily=${self.max_daily_cost:.2f}"
        )

    def _rotate_day_if_needed(self) -> None:
        """Reset daily total if the calendar day has changed."""
        today = date.today()
        if today != self._current_day:
            logger.info(
                f"Day rotated from {self._current_day} to {today}. "
                f"Previous day total: ${self._daily_total:.4f}"
            )
            self._daily_total = 0.0
            self._current_day = today

    def estimate_cost(
        self,
        model_id: str,
        cost_per_million_input: float,
        cost_per_million_output: float,
        estimated_input_tokens: int,
        estimated_output_tokens: int,
    ) -> float:
        """Estimate the cost of an LLM call before executing it.

        Args:
            model_id: The model identifier.
            cost_per_million_input: Model's cost per million input tokens.
            cost_per_million_output: Model's cost per million output tokens.
            estimated_input_tokens: Estimated number of input tokens.
            estimated_output_tokens: Estimated number of output tokens.

        Returns:
            Estimated cost in USD.
        """
        input_cost = (estimated_input_tokens / 1_000_000) * cost_per_million_input
        output_cost = (estimated_output_tokens / 1_000_000) * cost_per_million_output
        total = input_cost + output_cost
        logger.debug(
            f"Cost estimate for {model_id}: "
            f"{estimated_input_tokens} input + {estimated_output_tokens} output = ${total:.6f}"
        )
        return total

    async def check_budget(
        self,
        estimated_cost: float,
        request_id: str | None = None,
    ) -> bool:
        """Check whether a call with the estimated cost should proceed.

        Args:
            estimated_cost: The estimated cost of the upcoming call.
            request_id: Optional request ID to check per-request budget.

        Returns:
            True if the call should proceed.

        Raises:
            BudgetExceededError: If the call would exceed either budget limit.
        """
        async with self._lock:
            self._rotate_day_if_needed()

            # Check daily budget
            if self._daily_total + estimated_cost > self.max_daily_cost:
                raise BudgetExceededError(
                    estimated_cost=estimated_cost,
                    budget_type="daily",
                    current_total=self._daily_total,
                    limit=self.max_daily_cost,
                )

            # Check per-request budget
            if request_id is not None:
                request_total = self._request_totals.get(request_id, 0.0)
                if request_total + estimated_cost > self.max_cost_per_request:
                    raise BudgetExceededError(
                        estimated_cost=estimated_cost,
                        budget_type="per-request",
                        current_total=request_total,
                        limit=self.max_cost_per_request,
                    )

        return True

    async def record_cost(
        self,
        model_id: str,
        cost_per_million_input: float,
        cost_per_million_output: float,
        input_tokens: int,
        output_tokens: int,
        request_id: str | None = None,
    ) -> CostRecord:
        """Record actual token usage and cost after an LLM call completes.

        Args:
            model_id: The model identifier used.
            cost_per_million_input: Model's cost per million input tokens.
            cost_per_million_output: Model's cost per million output tokens.
            input_tokens: Actual input tokens consumed.
            output_tokens: Actual output tokens consumed.
            request_id: Optional request ID for grouping.

        Returns:
            The CostRecord created for this call.
        """
        actual_cost = self.estimate_cost(
            model_id,
            cost_per_million_input,
            cost_per_million_output,
            input_tokens,
            output_tokens,
        )

        record = CostRecord(
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=actual_cost,
            actual_cost=actual_cost,
            request_id=request_id,
        )

        async with self._lock:
            self._rotate_day_if_needed()
            self._daily_total += actual_cost
            if request_id is not None:
                self._request_totals[request_id] = (
                    self._request_totals.get(request_id, 0.0) + actual_cost
                )
            self._records.append(record)

        logger.info(
            f"Cost recorded: {model_id} - ${actual_cost:.6f} "
            f"({input_tokens} in / {output_tokens} out) "
            f"[daily: ${self._daily_total:.4f}]"
        )

        return record

    async def get_request_total(self, request_id: str) -> float:
        """Get the total cost accumulated for a specific request.

        Args:
            request_id: The request identifier.

        Returns:
            Total cost in USD for this request.
        """
        async with self._lock:
            return self._request_totals.get(request_id, 0.0)

    async def get_daily_total(self) -> float:
        """Get the total cost accumulated for the current day.

        Returns:
            Total cost in USD for today.
        """
        async with self._lock:
            self._rotate_day_if_needed()
            return self._daily_total

    async def get_summary(self) -> dict:
        """Get a summary of cost tracking state.

        Returns:
            Dictionary with daily total, request totals, limits, and record count.
        """
        async with self._lock:
            self._rotate_day_if_needed()
            return {
                "daily_total": round(self._daily_total, 6),
                "daily_limit": self.max_daily_cost,
                "daily_remaining": round(
                    max(0.0, self.max_daily_cost - self._daily_total), 6
                ),
                "request_limit": self.max_cost_per_request,
                "active_requests": len(self._request_totals),
                "total_records": len(self._records),
                "current_day": self._current_day.isoformat(),
            }

    async def clear_request(self, request_id: str) -> None:
        """Clear the accumulated cost for a completed request.

        Call this when a request finishes to free up memory.

        Args:
            request_id: The request identifier to clear.
        """
        async with self._lock:
            self._request_totals.pop(request_id, None)
