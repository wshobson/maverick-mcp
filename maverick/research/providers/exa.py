"""Exa-based web search provider with financial domain optimization.

Ported from `maverick_mcp/agents/research/providers/exa.py`
(`ExaSearchProvider`). Result normalization (URL/title/content truncation,
score, financial relevance, domain, authoritativeness) and the search-
strategy parameter builder are ported field-for-field.

`exa_py` import: kept lazy (inside `_search_with_strategy`'s inner
`_search` closure), matching the legacy module exactly -- legacy already
imports it there rather than at module top level. This module's top level
therefore never touches `exa_py`, so `maverick.research.providers` (whose
`__init__.py` re-exports from this module and is imported whenever any
sibling submodule is, including by the service tier) stays importable on a
base install without the `research` extra. Only calling `.search(...)` (or
`.search_financial(...)`) requires `exa_py` to be installed.

Circuit breaker: legacy wraps the network call in
`circuit_manager.get_or_create("exa_search", failure_threshold=8,
recovery_timeout=30).call(...)` -- a process-global named registry
(`maverick_mcp/agents/circuit_breaker.py`'s `CircuitBreakerManager`) with
CLOSED/OPEN/HALF_OPEN states, near-identical in shape to
`maverick.platform.http`'s `CircuitBreaker`/`get_breaker`. Comparison:

- Failure threshold / recovery timeout: both configurable per breaker name.
  Legacy's `failure_threshold=8`/`recovery_timeout=30` (search-specific,
  more tolerant than the general 5/60 default) map onto
  `HttpSettings.breaker_failure_threshold`/`breaker_recovery_seconds`,
  sourced from `ResearchSettings.search_circuit_breaker_failure_threshold`/
  `search_circuit_breaker_recovery_seconds` (see `config.py`'s docstring).
- Half-open probe: legacy re-checks `state == OPEN` only, so once one
  caller flips it to HALF_OPEN, concurrent callers arriving before that
  probe resolves see HALF_OPEN and are let through uncontrolled (no
  fail-fast). The platform breaker explicitly fail-fasts (raises
  `CircuitOpenError`) for any caller that arrives while a probe is already
  in flight. Strictly safer, not a missing behavior.
- Success accounting while closed: legacy decrements the failure counter by
  one per success (a "leaky bucket"); the platform breaker resets it to
  zero. Both converge to the same CLOSED/OPEN boundary for a run of
  consecutive failures, which is the only case the threshold cares about;
  the platform's harder reset does not weaken failure detection.

No semantics found in the legacy manager that the platform breaker lacks
(no per-provider fallback chain, no additional breaker states) -> the
platform breaker is behaviorally adequate. `CircuitBreakerManager` /
`CircuitBreaker` / `circuit_manager` were therefore NOT ported; this module
calls `maverick.platform.http.get_breaker` directly.

`WebSearchProvider`'s own health gate (`is_healthy`/`_record_failure`) is a
separate, provider-level mechanism (see `base.py`'s docstring) and is
unaffected by this decision -- it still runs alongside the breaker, exactly
as in legacy.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from maverick.platform.config import HttpSettings
from maverick.platform.http import CircuitOpenError, get_breaker
from maverick.research.config import ResearchSettings
from maverick.research.providers.base import WebSearchError, WebSearchProvider

if TYPE_CHECKING:
    from exa_py.api import Result as ExaResult

logger = logging.getLogger(__name__)

_BREAKER_NAME = "exa_search"

# Financial-specific domain preferences for better results.
_FINANCIAL_DOMAINS = [
    "sec.gov",
    "edgar.sec.gov",
    "investor.gov",
    "bloomberg.com",
    "reuters.com",
    "wsj.com",
    "ft.com",
    "marketwatch.com",
    "yahoo.com/finance",
    "finance.yahoo.com",
    "morningstar.com",
    "fool.com",
    "seekingalpha.com",
    "investopedia.com",
    "barrons.com",
    "cnbc.com",
    "nasdaq.com",
    "nyse.com",
    "finra.org",
    "federalreserve.gov",
    "treasury.gov",
    "bls.gov",
]

# Domains to exclude for financial searches.
_EXCLUDED_DOMAINS = [
    "facebook.com",
    "twitter.com",
    "x.com",
    "instagram.com",
    "tiktok.com",
    "reddit.com",
    "pinterest.com",
    "linkedin.com",
    "youtube.com",
    "wikipedia.org",
]

_AUTHORITATIVE_DOMAINS = [
    "sec.gov",
    "edgar.sec.gov",
    "federalreserve.gov",
    "treasury.gov",
    "bloomberg.com",
    "reuters.com",
    "wsj.com",
    "ft.com",
]

_FINANCIAL_KEYWORDS = [
    "earnings",
    "revenue",
    "profit",
    "financial",
    "quarterly",
    "annual",
    "sec filing",
    "10-k",
    "10-q",
    "balance sheet",
    "income statement",
    "cash flow",
    "dividend",
    "market cap",
    "valuation",
    "analyst",
    "forecast",
    "guidance",
    "ebitda",
    "eps",
    "pe ratio",
]

_FINANCIAL_QUERY_TERMS = {
    "earnings",
    "revenue",
    "profit",
    "loss",
    "financial",
    "quarterly",
    "annual",
    "SEC",
    "10-K",
    "10-Q",
    "balance sheet",
    "income statement",
    "cash flow",
    "dividend",
    "stock",
    "share",
    "market cap",
    "valuation",
}

# Content truncation boundaries, verbatim from legacy.
_CONTENT_CHARS = 2000
_RAW_CONTENT_CHARS = 5000
_DEFAULT_SCORE = 0.7

# Financial-relevance scoring weights, verbatim from legacy.
_DOMAIN_SCORE_TOP_TIER = 0.4
_DOMAIN_SCORE_HIGH_QUALITY = 0.3
_DOMAIN_SCORE_OTHER = 0.2
_KEYWORD_SCORE_PER_MATCH = 0.05
_KEYWORD_SCORE_MAX = 0.3
_TITLE_SCORE = 0.1
_RECENCY_SCORE_30D = 0.1
_RECENCY_SCORE_90D = 0.05


class ExaSearchProvider(WebSearchProvider):
    """Exa search provider for comprehensive web search using MCP tools with financial optimization."""

    def __init__(self, api_key: str, *, settings: ResearchSettings | None = None):
        super().__init__(api_key, settings=settings)
        self._api_key_verified = bool(api_key)
        self.financial_domains = list(_FINANCIAL_DOMAINS)
        self.excluded_domains = list(_EXCLUDED_DOMAINS)
        logger.info("Initialized ExaSearchProvider with financial optimization")

    async def search(
        self, query: str, num_results: int = 10, timeout_budget: float | None = None
    ) -> list[dict[str, Any]]:
        """Search using Exa via async client for comprehensive web results with adaptive timeout."""
        return await self._search_with_strategy(
            query, num_results, timeout_budget, "auto"
        )

    async def search_financial(
        self,
        query: str,
        num_results: int = 10,
        timeout_budget: float | None = None,
        strategy: str = "hybrid",
    ) -> list[dict[str, Any]]:
        """Enhanced financial search with optimized queries and domain targeting.

        Args:
            query: Search query
            num_results: Number of results to return
            timeout_budget: Timeout budget in seconds
            strategy: Search strategy - 'hybrid', 'authoritative', 'comprehensive', or 'auto'
        """
        return await self._search_with_strategy(
            query, num_results, timeout_budget, strategy
        )

    async def _search_with_strategy(
        self, query: str, num_results: int, timeout_budget: float | None, strategy: str
    ) -> list[dict[str, Any]]:
        """Internal method to handle different search strategies."""
        if not self.is_healthy():
            logger.warning("Exa provider is unhealthy - skipping search")
            raise WebSearchError("Exa provider disabled due to repeated failures")

        search_timeout = self._calculate_timeout(query, timeout_budget)

        breaker = get_breaker(
            _BREAKER_NAME,
            HttpSettings(
                breaker_failure_threshold=self._settings.search_circuit_breaker_failure_threshold,
                breaker_recovery_seconds=self._settings.search_circuit_breaker_recovery_seconds,
            ),
        )

        async def _search() -> list[dict[str, Any]]:
            try:
                from exa_py import AsyncExa
            except ImportError as exc:
                logger.error("exa-py library not available - cannot perform search")
                raise WebSearchError(
                    "exa-py library required for ExaSearchProvider"
                ) from exc

            try:
                async_exa_client = AsyncExa(api_key=self.api_key)
                search_params = self._get_search_params(query, num_results, strategy)
                exa_response = await async_exa_client.search_and_contents(
                    **search_params
                )
            except Exception as e:
                logger.error(f"Error calling Exa API: {e}")
                raise

            return self._normalize_response(exa_response)

        try:
            result = await asyncio.wait_for(
                breaker.call(_search), timeout=search_timeout
            )
            self._record_success()
            logger.debug(
                f"Exa search completed in {search_timeout:.1f}s timeout window"
            )
            return result

        except TimeoutError:
            self._record_failure("timeout")
            query_snippet = query[:100] + ("..." if len(query) > 100 else "")
            logger.error(
                f"Exa search timeout after {search_timeout:.1f} seconds (failure #{self._failure_count}) "
                f"for query: '{query_snippet}'"
            )
            raise WebSearchError(
                f"Exa search timed out after {search_timeout:.1f} seconds"
            )
        except CircuitOpenError as e:
            self._record_failure("error")
            logger.error(
                f"Exa search circuit open (failure #{self._failure_count}): {e}"
            )
            raise WebSearchError(f"Exa search failed: {e}") from e
        except Exception as e:
            self._record_failure("error")
            logger.error(f"Exa search error (failure #{self._failure_count}): {e}")
            raise WebSearchError(f"Exa search failed: {str(e)}") from e

    def _normalize_response(self, exa_response: Any) -> list[dict[str, Any]]:
        """Convert an Exa `search_and_contents` response into the legacy result shape."""
        results = []
        if exa_response and hasattr(exa_response, "results"):
            for result in exa_response.results:
                financial_relevance = self._calculate_financial_relevance(result)
                results.append(
                    {
                        "url": result.url or "",
                        "title": result.title or "No Title",
                        "content": (result.text or "")[:_CONTENT_CHARS],
                        "raw_content": (result.text or "")[:_RAW_CONTENT_CHARS],
                        "published_date": result.published_date or "",
                        "score": result.score
                        if hasattr(result, "score") and result.score is not None
                        else _DEFAULT_SCORE,
                        "financial_relevance": financial_relevance,
                        "provider": "exa",
                        "author": result.author
                        if hasattr(result, "author") and result.author is not None
                        else "",
                        "domain": self._extract_domain(result.url or ""),
                        "is_authoritative": self._is_authoritative_source(
                            result.url or ""
                        ),
                    }
                )

        results.sort(key=lambda x: (x["financial_relevance"], x["score"]), reverse=True)
        return results

    def _get_search_params(
        self, query: str, num_results: int, strategy: str
    ) -> dict[str, Any]:
        """Generate optimized search parameters based on strategy and query type."""
        params: dict[str, Any] = {
            "query": query,
            "num_results": num_results,
            "text": {"max_characters": 5000},
        }

        if strategy == "authoritative":
            params.update(
                {
                    "include_domains": self.financial_domains[:10],
                    "type": "auto",
                    "start_published_date": "2020-01-01",
                }
            )
        elif strategy == "comprehensive":
            params.update(
                {
                    "exclude_domains": self.excluded_domains,
                    "type": "neural",
                    "start_published_date": "2018-01-01",
                }
            )
        elif strategy == "hybrid":
            params.update(
                {
                    "exclude_domains": self.excluded_domains,
                    "type": "auto",
                    "start_published_date": "2019-01-01",
                }
            )
        else:  # "auto" or default
            params.update(
                {
                    "exclude_domains": self.excluded_domains[:5],
                    "type": "auto",
                }
            )

        enhanced_query = self._enhance_financial_query(query)
        if enhanced_query != query:
            params["query"] = enhanced_query

        return params

    def _enhance_financial_query(self, query: str) -> str:
        """Enhance queries with financial context and terminology."""
        query_lower = query.lower()
        has_financial_context = any(
            term in query_lower for term in _FINANCIAL_QUERY_TERMS
        )

        if not has_financial_context:
            if any(
                indicator in query_lower
                for indicator in ["company", "corp", "inc", "$", "stock"]
            ):
                return f"{query} financial analysis earnings revenue"
            elif len(query.split()) <= 3 and query.isupper():
                return f"{query} stock financial performance earnings"
            elif "analysis" in query_lower or "research" in query_lower:
                return f"{query} financial data SEC filings"

        return query

    def _calculate_financial_relevance(self, result: ExaResult | Any) -> float:
        """Calculate financial relevance score for a search result (0.0 to 1.0)."""
        score = 0.0

        domain = self._extract_domain(result.url)
        if domain in self.financial_domains:
            if domain in ["sec.gov", "edgar.sec.gov", "federalreserve.gov"]:
                score += _DOMAIN_SCORE_TOP_TIER
            elif domain in ["bloomberg.com", "reuters.com", "wsj.com", "ft.com"]:
                score += _DOMAIN_SCORE_HIGH_QUALITY
            else:
                score += _DOMAIN_SCORE_OTHER

        if hasattr(result, "text") and result.text:
            text_lower = result.text.lower()
            keyword_matches = sum(
                1 for keyword in _FINANCIAL_KEYWORDS if keyword in text_lower
            )
            score += min(keyword_matches * _KEYWORD_SCORE_PER_MATCH, _KEYWORD_SCORE_MAX)

        if hasattr(result, "title") and result.title:
            title_lower = result.title.lower()
            if any(
                term in title_lower
                for term in ["financial", "earnings", "quarterly", "annual", "sec"]
            ):
                score += _TITLE_SCORE

        if hasattr(result, "published_date") and result.published_date:
            try:
                from datetime import datetime

                date_str = str(result.published_date)
                if date_str and date_str != "":
                    if date_str.endswith("Z"):
                        date_str = date_str.replace("Z", "+00:00")

                    pub_date = datetime.fromisoformat(date_str)
                    days_old = (datetime.now(UTC) - pub_date).days

                    if days_old <= 30:
                        score += _RECENCY_SCORE_30D
                    elif days_old <= 90:
                        score += _RECENCY_SCORE_90D
            except (ValueError, AttributeError, TypeError):
                pass

        return min(score, 1.0)

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            return urlparse(url).netloc.lower().replace("www.", "")
        except Exception:
            return ""

    def _is_authoritative_source(self, url: str) -> bool:
        """Check if URL is from an authoritative financial source."""
        domain = self._extract_domain(url)
        return domain in _AUTHORITATIVE_DOMAINS
