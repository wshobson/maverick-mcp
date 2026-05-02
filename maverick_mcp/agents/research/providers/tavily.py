"""Tavily-based web search provider."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from typing import Any

from maverick_mcp.agents.circuit_breaker import circuit_manager
from maverick_mcp.agents.research.providers.base import WebSearchProvider
from maverick_mcp.exceptions import WebSearchError

try:  # pragma: no cover - optional dependency
    from tavily import TavilyClient  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    TavilyClient = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class TavilySearchProvider(WebSearchProvider):
    """Tavily search provider with sensible filtering for financial research."""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.excluded_domains = {
            "facebook.com",
            "twitter.com",
            "x.com",
            "instagram.com",
            "reddit.com",
        }

    async def search(
        self, query: str, num_results: int = 10, timeout_budget: float | None = None
    ) -> list[dict[str, Any]]:
        if not self.is_healthy():
            raise WebSearchError("Tavily provider disabled due to repeated failures")

        timeout = self._calculate_timeout(query, timeout_budget)
        circuit_breaker = await circuit_manager.get_or_create(
            "tavily_search",
            failure_threshold=8,
            recovery_timeout=30,
        )

        async def _search() -> list[dict[str, Any]]:
            if TavilyClient is None:
                raise ImportError("tavily package is required for TavilySearchProvider")

            client = TavilyClient(api_key=self.api_key)
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.search(query=query, max_results=num_results),
            )
            return self._process_results(response.get("results", []))

        return await circuit_breaker.call(_search, timeout=timeout)

    def _process_results(
        self, results: Iterable[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        processed: list[dict[str, Any]] = []
        for item in results:
            url = item.get("url", "")
            if any(domain in url for domain in self.excluded_domains):
                continue
            processed.append(
                {
                    "url": url,
                    "title": item.get("title"),
                    "content": item.get("content") or item.get("raw_content", ""),
                    "raw_content": item.get("raw_content"),
                    "published_date": item.get("published_date"),
                    "score": item.get("score", 0.0),
                    "provider": "tavily",
                }
            )
        return processed
