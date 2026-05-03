"""Exa-based web search provider with financial domain optimization."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC
from typing import Any

from maverick_mcp.agents.circuit_breaker import circuit_manager
from maverick_mcp.agents.research.providers.base import WebSearchProvider
from maverick_mcp.exceptions import WebSearchError

logger = logging.getLogger(__name__)


class ExaSearchProvider(WebSearchProvider):
    """Exa search provider for comprehensive web search using MCP tools with financial optimization."""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        # Store the API key for verification
        self._api_key_verified = bool(api_key)

        # Financial-specific domain preferences for better results
        self.financial_domains = [
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

        # Domains to exclude for financial searches
        self.excluded_domains = [
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
        """
        Enhanced financial search with optimized queries and domain targeting.

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

        # Check provider health before attempting search
        if not self.is_healthy():
            logger.warning("Exa provider is unhealthy - skipping search")
            raise WebSearchError("Exa provider disabled due to repeated failures")

        # Calculate adaptive timeout
        search_timeout = self._calculate_timeout(query, timeout_budget)

        try:
            # Use search-specific circuit breaker settings (more tolerant)
            circuit_breaker = await circuit_manager.get_or_create(
                "exa_search",
                failure_threshold=getattr(
                    self.settings.performance,
                    "search_circuit_breaker_failure_threshold",
                    8,
                ),
                recovery_timeout=getattr(
                    self.settings.performance,
                    "search_circuit_breaker_recovery_timeout",
                    30,
                ),
            )

            async def _search():
                # Use the async exa-py library for web search
                try:
                    from exa_py import AsyncExa

                    # Initialize AsyncExa client with API key
                    async_exa_client = AsyncExa(api_key=self.api_key)

                    # Configure search parameters based on strategy
                    search_params = self._get_search_params(
                        query, num_results, strategy
                    )

                    # Call Exa search with optimized parameters
                    exa_response = await async_exa_client.search_and_contents(
                        **search_params
                    )

                    # Convert Exa response to standard format with enhanced metadata
                    results = []
                    if exa_response and hasattr(exa_response, "results"):
                        for result in exa_response.results:
                            # Enhanced result processing with financial relevance scoring
                            financial_relevance = self._calculate_financial_relevance(
                                result
                            )

                            results.append(
                                {
                                    "url": result.url or "",
                                    "title": result.title or "No Title",
                                    "content": (result.text or "")[:2000],
                                    "raw_content": (result.text or "")[
                                        :5000
                                    ],  # Increased for financial content
                                    "published_date": result.published_date or "",
                                    "score": result.score
                                    if hasattr(result, "score")
                                    and result.score is not None
                                    else 0.7,
                                    "financial_relevance": financial_relevance,
                                    "provider": "exa",
                                    "author": result.author
                                    if hasattr(result, "author")
                                    and result.author is not None
                                    else "",
                                    "domain": self._extract_domain(result.url or ""),
                                    "is_authoritative": self._is_authoritative_source(
                                        result.url or ""
                                    ),
                                }
                            )

                    # Sort results by financial relevance and score
                    results.sort(
                        key=lambda x: (x["financial_relevance"], x["score"]),
                        reverse=True,
                    )
                    return results

                except ImportError:
                    logger.error("exa-py library not available - cannot perform search")
                    raise WebSearchError(
                        "exa-py library required for ExaSearchProvider"
                    )
                except Exception as e:
                    logger.error(f"Error calling Exa API: {e}")
                    raise e

            # Use adaptive timeout based on query complexity and budget
            result = await asyncio.wait_for(
                circuit_breaker.call(_search), timeout=search_timeout
            )
            self._record_success()  # Record successful search
            logger.debug(
                f"Exa search completed in {search_timeout:.1f}s timeout window"
            )
            return result

        except TimeoutError:
            self._record_failure("timeout")  # Record timeout as specific failure type
            query_snippet = query[:100] + ("..." if len(query) > 100 else "")
            logger.error(
                f"Exa search timeout after {search_timeout:.1f} seconds (failure #{self._failure_count}) "
                f"for query: '{query_snippet}'"
            )
            raise WebSearchError(
                f"Exa search timed out after {search_timeout:.1f} seconds"
            )
        except Exception as e:
            self._record_failure("error")  # Record non-timeout failure
            logger.error(f"Exa search error (failure #{self._failure_count}): {e}")
            raise WebSearchError(f"Exa search failed: {str(e)}")

    def _get_search_params(
        self, query: str, num_results: int, strategy: str
    ) -> dict[str, Any]:
        """
        Generate optimized search parameters based on strategy and query type.

        Args:
            query: Search query
            num_results: Number of results
            strategy: Search strategy

        Returns:
            Dictionary of search parameters for Exa API
        """
        # Base parameters
        params = {
            "query": query,
            "num_results": num_results,
            "text": {"max_characters": 5000},  # Increased for financial content
        }

        # Strategy-specific optimizations
        if strategy == "authoritative":
            # Focus on authoritative financial sources
            # Note: Exa API doesn't allow both include_domains and exclude_domains with content
            params.update(
                {
                    "include_domains": self.financial_domains[
                        :10
                    ],  # Top authoritative sources
                    "type": "auto",  # Let Exa decide neural vs keyword
                    "start_published_date": "2020-01-01",  # Recent financial data
                }
            )

        elif strategy == "comprehensive":
            # Broad search across all financial sources
            params.update(
                {
                    "exclude_domains": self.excluded_domains,
                    "type": "neural",  # Better for comprehensive understanding
                    "start_published_date": "2018-01-01",  # Broader historical context
                }
            )

        elif strategy == "hybrid":
            # Balanced approach with domain preferences
            params.update(
                {
                    "exclude_domains": self.excluded_domains,
                    "type": "auto",  # Hybrid neural/keyword approach
                    "start_published_date": "2019-01-01",
                    # Use domain weighting rather than strict inclusion
                }
            )

        else:  # "auto" or default
            # Standard search with basic optimizations
            params.update(
                {
                    "exclude_domains": self.excluded_domains[:5],  # Basic exclusions
                    "type": "auto",
                }
            )

        # Add financial-specific query enhancements
        enhanced_query = self._enhance_financial_query(query)
        if enhanced_query != query:
            params["query"] = enhanced_query

        return params

    def _enhance_financial_query(self, query: str) -> str:
        """
        Enhance queries with financial context and terminology.

        Args:
            query: Original search query

        Returns:
            Enhanced query with financial context
        """
        # Financial keywords that improve search quality
        financial_terms = {
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

        query_lower = query.lower()

        # Check if query already contains financial terms
        has_financial_context = any(term in query_lower for term in financial_terms)

        # Add context for company/stock queries
        if not has_financial_context:
            # Detect if it's a company or stock symbol query
            if any(
                indicator in query_lower
                for indicator in ["company", "corp", "inc", "$", "stock"]
            ):
                return f"{query} financial analysis earnings revenue"
            elif len(query.split()) <= 3 and query.isupper():  # Likely stock symbol
                return f"{query} stock financial performance earnings"
            elif "analysis" in query_lower or "research" in query_lower:
                return f"{query} financial data SEC filings"

        return query

    def _calculate_financial_relevance(self, result) -> float:
        """
        Calculate financial relevance score for a search result.

        Args:
            result: Exa search result object

        Returns:
            Financial relevance score (0.0 to 1.0)
        """
        score = 0.0

        # Domain-based scoring
        domain = self._extract_domain(result.url)
        if domain in self.financial_domains:
            if domain in ["sec.gov", "edgar.sec.gov", "federalreserve.gov"]:
                score += 0.4  # Highest authority
            elif domain in ["bloomberg.com", "reuters.com", "wsj.com", "ft.com"]:
                score += 0.3  # High-quality financial news
            else:
                score += 0.2  # Other financial sources

        # Content-based scoring
        if hasattr(result, "text") and result.text:
            text_lower = result.text.lower()

            # Financial terminology scoring
            financial_keywords = [
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

            keyword_matches = sum(
                1 for keyword in financial_keywords if keyword in text_lower
            )
            score += min(keyword_matches * 0.05, 0.3)  # Max 0.3 from keywords

        # Title-based scoring
        if hasattr(result, "title") and result.title:
            title_lower = result.title.lower()
            if any(
                term in title_lower
                for term in ["financial", "earnings", "quarterly", "annual", "sec"]
            ):
                score += 0.1

        # Recency bonus for financial data
        if hasattr(result, "published_date") and result.published_date:
            try:
                from datetime import datetime

                # Handle different date formats
                date_str = str(result.published_date)
                if date_str and date_str != "":
                    # Handle ISO format with Z
                    if date_str.endswith("Z"):
                        date_str = date_str.replace("Z", "+00:00")

                    pub_date = datetime.fromisoformat(date_str)
                    days_old = (datetime.now(UTC) - pub_date).days

                    if days_old <= 30:
                        score += 0.1  # Recent data bonus
                    elif days_old <= 90:
                        score += 0.05  # Somewhat recent bonus
            except (ValueError, AttributeError, TypeError):
                pass  # Skip if date parsing fails

        return min(score, 1.0)  # Cap at 1.0

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse

            return urlparse(url).netloc.lower().replace("www.", "")
        except Exception:
            return ""

    def _is_authoritative_source(self, url: str) -> bool:
        """Check if URL is from an authoritative financial source."""
        domain = self._extract_domain(url)
        authoritative_domains = [
            "sec.gov",
            "edgar.sec.gov",
            "federalreserve.gov",
            "treasury.gov",
            "bloomberg.com",
            "reuters.com",
            "wsj.com",
            "ft.com",
        ]
        return domain in authoritative_domains
