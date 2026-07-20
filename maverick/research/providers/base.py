"""Web search provider base: adaptive timeout policy and health tracking.

Ported from `maverick_mcp/agents/research/providers/base.py`
(`WebSearchProvider`). Two resilience mechanisms are layered here, distinct
from each other and from the circuit breaker `exa.py` wraps around the
actual network call:

- `_calculate_timeout`: a per-search timeout formula (legacy lines 25-55),
  not a single overridable literal -- it tiers the timeout by query word
  count (30s/45s/60s) and optionally shrinks to a fraction of a caller-
  supplied budget. The legacy code has no environment override for any of
  these literals (no `SEARCH_TIMEOUT_*` env var reads them), so they stay as
  module-level constants here rather than moving into `ResearchSettings`,
  matching the legacy shape exactly.
- `_record_failure`/`_record_success`/`is_healthy`: a provider-level health
  gate (legacy lines 57-98) that disables a provider after repeated
  failures until a success resets it -- no recovery timeout, unlike a
  circuit breaker's half-open probe. `search_timeout_failure_threshold`
  (12) is the one literal here that legacy reads from settings
  (`settings.performance.search_timeout_failure_threshold`,
  `maverick_mcp/config/settings.py:777-782`) rather than hardcoding; it is
  now `ResearchSettings.search_timeout_failure_threshold` (see
  `maverick/research/config.py`'s module docstring). The non-timeout
  threshold (`_max_failures * 2` = 6) was a plain hardcoded instance
  attribute in legacy too, so it stays a module constant here.
"""

from __future__ import annotations

import logging
from typing import Any

from maverick.research.config import ResearchSettings, get_research_settings

logger = logging.getLogger(__name__)

# `_calculate_timeout` tiers, verbatim from legacy (no settings/env knobs there).
_SHORT_QUERY_WORDS = 3
_MEDIUM_QUERY_WORDS = 8
_SHORT_QUERY_TIMEOUT_SECONDS = 30.0
_MEDIUM_QUERY_TIMEOUT_SECONDS = 45.0
_LONG_QUERY_TIMEOUT_SECONDS = 60.0
_MIN_TIMEOUT_SECONDS = 30.0
_BUDGET_FRACTION = 0.6

# Non-timeout failures get double the timeout-failure leniency, same ratio
# as legacy's `_max_failures * 2` (legacy `_max_failures` is a hardcoded 3).
_MAX_NON_TIMEOUT_FAILURES = 6


class WebSearchError(Exception):
    """Raised when a web search provider call fails or is disabled.

    A plain `Exception` rather than an import of the legacy
    `maverick_mcp.exceptions.WebSearchError` hierarchy: `maverick/` may
    never import `maverick_mcp` (see the "new package never imports the
    legacy package" import-linter contract), so this domain defines its own.
    """


class WebSearchProvider:
    """Base class for web search providers with early abort mechanism."""

    def __init__(self, api_key: str, *, settings: ResearchSettings | None = None):
        self.api_key = api_key
        self._failure_count = 0
        self._is_healthy = True
        self._settings = settings or get_research_settings()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _calculate_timeout(
        self, query: str, timeout_budget: float | None = None
    ) -> float:
        """Calculate generous timeout for thorough research operations."""
        query_words = len(query.split())

        # Generous timeout calculation for thorough search operations
        if query_words <= _SHORT_QUERY_WORDS:
            base_timeout = _SHORT_QUERY_TIMEOUT_SECONDS
        elif query_words <= _MEDIUM_QUERY_WORDS:
            base_timeout = _MEDIUM_QUERY_TIMEOUT_SECONDS
        else:
            base_timeout = _LONG_QUERY_TIMEOUT_SECONDS

        # Apply budget constraints if available
        if timeout_budget and timeout_budget > 0:
            # Use generous portion of available budget per search operation
            budget_timeout = max(
                timeout_budget * _BUDGET_FRACTION, _MIN_TIMEOUT_SECONDS
            )
            calculated_timeout = min(base_timeout, budget_timeout)
            calculated_timeout = max(calculated_timeout, _MIN_TIMEOUT_SECONDS)
        else:
            calculated_timeout = base_timeout

        return max(calculated_timeout, _MIN_TIMEOUT_SECONDS)

    def _record_failure(self, error_type: str = "unknown") -> None:
        """Record a search failure and check if provider should be disabled."""
        self._failure_count += 1

        timeout_threshold = self._settings.search_timeout_failure_threshold

        # Much more tolerant of timeout failures - they may be due to network/complexity
        if error_type == "timeout" and self._failure_count >= timeout_threshold:
            self._is_healthy = False
            logger.warning(
                f"Search provider {self.__class__.__name__} disabled after "
                f"{self._failure_count} consecutive timeout failures (threshold: {timeout_threshold})"
            )
        elif (
            error_type != "timeout" and self._failure_count >= _MAX_NON_TIMEOUT_FAILURES
        ):
            self._is_healthy = False
            logger.warning(
                f"Search provider {self.__class__.__name__} disabled after "
                f"{self._failure_count} total non-timeout failures"
            )

        logger.debug(
            f"Provider {self.__class__.__name__} failure recorded: "
            f"type={error_type}, count={self._failure_count}, healthy={self._is_healthy}"
        )

    def _record_success(self) -> None:
        """Record a successful search and reset failure count."""
        if self._failure_count > 0:
            logger.info(
                f"Search provider {self.__class__.__name__} recovered after "
                f"{self._failure_count} failures"
            )
        self._failure_count = 0
        self._is_healthy = True

    def is_healthy(self) -> bool:
        """Check if provider is healthy and should be used."""
        return self._is_healthy

    async def search(
        self, query: str, num_results: int = 10, timeout_budget: float | None = None
    ) -> list[dict[str, Any]]:
        """Perform web search and return results."""
        raise NotImplementedError

    async def get_content(self, url: str) -> dict[str, Any]:
        """Extract content from URL."""
        raise NotImplementedError

    async def search_multiple_providers(
        self,
        queries: list[str],
        providers: list[str] | None = None,
        max_results_per_query: int = 5,
    ) -> dict[str, list[dict[str, Any]]]:
        """Search using multiple providers and return aggregated results."""
        providers = providers or ["exa"]  # Default to available providers
        results = {}

        for provider_name in providers:
            provider_results = []
            for query in queries:
                try:
                    query_results = await self.search(query, max_results_per_query)

                    provider_results.extend(query_results or [])
                except Exception as e:
                    self.logger.warning(
                        f"Search failed for provider {provider_name}, query '{query}': {e}"
                    )
                    continue

            results[provider_name] = provider_results

        return results

    def _timeframe_to_date(self, timeframe: str) -> str | None:
        """Convert timeframe string to date string."""
        from datetime import datetime, timedelta

        now = datetime.now()

        if timeframe == "1d":
            date = now - timedelta(days=1)
        elif timeframe == "1w":
            date = now - timedelta(weeks=1)
        elif timeframe == "1m":
            date = now - timedelta(days=30)
        else:
            # Invalid or unsupported timeframe, return None
            return None

        return date.strftime("%Y-%m-%d")
