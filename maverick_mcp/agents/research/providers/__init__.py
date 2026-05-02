"""Search-provider strategies extracted from `agents/deep_research.py`.

Public surface — both ``maverick_mcp.agents.research.providers`` and the
legacy ``maverick_mcp.agents.deep_research`` re-export these symbols so
callers (and tests that patch
``maverick_mcp.agents.deep_research.get_cached_search_provider``)
continue to work unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

from maverick_mcp.agents.research.providers.base import WebSearchProvider
from maverick_mcp.agents.research.providers.exa import ExaSearchProvider
from maverick_mcp.agents.research.providers.tavily import TavilySearchProvider

logger = logging.getLogger(__name__)


# Cache for instantiated providers to avoid repeated initialization.
_search_provider_cache: dict[str, Any] = {}


async def get_cached_search_provider(exa_api_key: str | None = None) -> Any | None:
    """Get cached Exa search provider to avoid repeated initialization delays."""
    cache_key = f"exa:{exa_api_key is not None}"

    if cache_key in _search_provider_cache:
        return _search_provider_cache[cache_key]

    logger.info("Initializing Exa search provider")
    provider = None

    # Initialize Exa provider with caching
    if exa_api_key:
        try:
            provider = ExaSearchProvider(exa_api_key)
            logger.info("Initialized Exa search provider")
            # Cache the provider
            _search_provider_cache[cache_key] = provider
        except ImportError as e:
            logger.warning(f"Failed to initialize Exa provider: {e}")

    return provider


__all__ = [
    "ExaSearchProvider",
    "TavilySearchProvider",
    "WebSearchProvider",
    "_search_provider_cache",
    "get_cached_search_provider",
]
