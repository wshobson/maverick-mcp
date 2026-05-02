"""Web search provider base class with health tracking and timeout policy."""

from __future__ import annotations

import logging
from typing import Any

from maverick_mcp.config.settings import get_settings

logger = logging.getLogger(__name__)


class WebSearchProvider:
    """Base class for web search providers with early abort mechanism."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.rate_limiter = None  # Implement rate limiting
        self._failure_count = 0
        self._max_failures = 3  # Abort after 3 consecutive failures
        self._is_healthy = True
        self.settings = get_settings()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _calculate_timeout(
        self, query: str, timeout_budget: float | None = None
    ) -> float:
        """Calculate generous timeout for thorough research operations."""
        query_words = len(query.split())

        # Generous timeout calculation for thorough search operations
        if query_words <= 3:
            base_timeout = 30.0  # Simple queries - 30s for thorough results
        elif query_words <= 8:
            base_timeout = 45.0  # Standard queries - 45s for comprehensive search
        else:
            base_timeout = 60.0  # Complex queries - 60s for exhaustive search

        # Apply budget constraints if available
        if timeout_budget and timeout_budget > 0:
            # Use generous portion of available budget per search operation
            budget_timeout = max(
                timeout_budget * 0.6, 30.0
            )  # At least 30s, use 60% of budget
            calculated_timeout = min(base_timeout, budget_timeout)

            # Ensure minimum timeout (at least 30s for thorough search)
            calculated_timeout = max(calculated_timeout, 30.0)
        else:
            calculated_timeout = base_timeout

        # Final timeout with generous minimum for thorough search
        final_timeout = max(calculated_timeout, 30.0)

        return final_timeout

    def _record_failure(self, error_type: str = "unknown") -> None:
        """Record a search failure and check if provider should be disabled."""
        self._failure_count += 1

        # Use separate thresholds for timeout vs other failures
        timeout_threshold = getattr(
            self.settings.performance, "search_timeout_failure_threshold", 12
        )

        # Much more tolerant of timeout failures - they may be due to network/complexity
        if error_type == "timeout" and self._failure_count >= timeout_threshold:
            self._is_healthy = False
            logger.warning(
                f"Search provider {self.__class__.__name__} disabled after "
                f"{self._failure_count} consecutive timeout failures (threshold: {timeout_threshold})"
            )
        elif error_type != "timeout" and self._failure_count >= self._max_failures * 2:
            # Be more lenient for non-timeout failures (2x threshold)
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
