"""
Finnhub alternative data router.

Provides MCP tools for accessing Finnhub's alternative data:
- Company news
- Earnings calendar and surprises
- Analyst recommendation trends
- Institutional ownership (13F filings)
- Company peers
- Economic calendar
- Market news
- Backup quotes

All tools degrade gracefully when no API key is configured.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from maverick_mcp.providers.finnhub_data import FinnhubDataProvider
from maverick_mcp.utils.rate_limiters import finnhub_limiter

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4)
atexit.register(_executor.shutdown, wait=False)

_provider: FinnhubDataProvider | None = None


def _get_provider() -> FinnhubDataProvider:
    """Lazy-initialise the Finnhub provider singleton."""
    global _provider
    if _provider is None:
        _provider = FinnhubDataProvider()
    return _provider


# ------------------------------------------------------------------ #
# Tool functions
# ------------------------------------------------------------------ #


async def get_finnhub_company_news(
    ticker: str,
    from_date: str | None = None,
    to_date: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Get company-specific news articles from Finnhub.

    Provides recent news articles about a specific company including
    headline, source, URL, publication time, and summary.

    Args:
        ticker: Stock ticker symbol (e.g., AAPL, TSLA)
        from_date: Start date YYYY-MM-DD (defaults to 7 days ago)
        to_date: End date YYYY-MM-DD (defaults to today)
        limit: Maximum number of articles to return (default 20)

    Returns:
        Dictionary containing company news articles
    """
    try:
        await finnhub_limiter.acquire()
        loop = asyncio.get_event_loop()
        news = await loop.run_in_executor(
            _executor,
            lambda: _get_provider().get_company_news(ticker, from_date, to_date),
        )
        articles = news[:limit]
        return {
            "status": "success",
            "ticker": ticker,
            "count": len(articles),
            "articles": articles,
        }
    except Exception as e:
        logger.error("finnhub_company_news failed for %s: %s", ticker, e)
        return {"status": "error", "ticker": ticker, "error": str(e)}


async def get_finnhub_earnings_calendar(
    from_date: str | None = None,
    to_date: str | None = None,
    ticker: str | None = None,
) -> dict[str, Any]:
    """Get upcoming and recent earnings dates with EPS estimates.

    Shows scheduled earnings releases with consensus EPS estimates,
    actual EPS when reported, and revenue estimates. Filter by ticker
    or view the full calendar.

    Args:
        from_date: Start date YYYY-MM-DD (defaults to today)
        to_date: End date YYYY-MM-DD (defaults to 14 days from now)
        ticker: Optional ticker symbol to filter results

    Returns:
        Dictionary containing earnings calendar data
    """
    try:
        await finnhub_limiter.acquire()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            lambda: _get_provider().get_earnings_calendar(from_date, to_date, ticker),
        )
        calendar = result.get("earningsCalendar", [])
        return {
            "status": "success",
            "ticker": ticker,
            "count": len(calendar),
            "earnings_calendar": calendar,
        }
    except Exception as e:
        logger.error("finnhub_earnings_calendar failed: %s", e)
        return {"status": "error", "ticker": ticker, "error": str(e)}


async def get_finnhub_earnings_surprises(
    ticker: str,
    limit: int = 4,
) -> dict[str, Any]:
    """Get historical earnings beat/miss data with surprise percentages.

    Shows actual vs estimated EPS for recent quarters, with the
    surprise amount and percentage. Useful for evaluating a company's
    earnings track record.

    Args:
        ticker: Stock ticker symbol
        limit: Number of quarters to return (default 4, max 20)

    Returns:
        Dictionary containing earnings surprise data
    """
    try:
        await finnhub_limiter.acquire()
        loop = asyncio.get_event_loop()
        surprises = await loop.run_in_executor(
            _executor,
            lambda: _get_provider().get_earnings_surprises(ticker, limit),
        )
        return {
            "status": "success",
            "ticker": ticker,
            "count": len(surprises),
            "earnings_surprises": surprises,
        }
    except Exception as e:
        logger.error("finnhub_earnings_surprises failed for %s: %s", ticker, e)
        return {"status": "error", "ticker": ticker, "error": str(e)}


async def get_finnhub_analyst_recommendations(
    ticker: str,
) -> dict[str, Any]:
    """Get analyst recommendation trends (buy/hold/sell consensus).

    Shows the monthly history of analyst recommendations broken down
    by strongBuy, buy, hold, sell, and strongSell counts. Useful for
    gauging Wall Street sentiment on a stock.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary containing analyst recommendation trends
    """
    try:
        await finnhub_limiter.acquire()
        loop = asyncio.get_event_loop()
        trends = await loop.run_in_executor(
            _executor,
            lambda: _get_provider().get_recommendation_trends(ticker),
        )
        # Summarise latest consensus if available
        latest = trends[0] if trends else {}
        return {
            "status": "success",
            "ticker": ticker,
            "latest_consensus": latest,
            "trend_history": trends,
        }
    except Exception as e:
        logger.error("finnhub_analyst_recommendations failed for %s: %s", ticker, e)
        return {"status": "error", "ticker": ticker, "error": str(e)}


async def get_finnhub_institutional_ownership(
    ticker: str,
    limit: int = 20,
) -> dict[str, Any]:
    """Get institutional ownership data from 13F filings.

    Shows top institutional holders with their share counts,
    changes since last filing, and filing dates. Useful for
    tracking smart money positions.

    Args:
        ticker: Stock ticker symbol
        limit: Maximum number of holders to return (default 20, max 50)

    Returns:
        Dictionary containing institutional ownership data
    """
    try:
        await finnhub_limiter.acquire()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            lambda: _get_provider().get_institutional_ownership(ticker, limit),
        )
        ownership = result.get("ownership", [])
        return {
            "status": "success",
            "ticker": ticker,
            "count": len(ownership),
            "holders": ownership,
        }
    except Exception as e:
        logger.error("finnhub_institutional_ownership failed for %s: %s", ticker, e)
        return {"status": "error", "ticker": ticker, "error": str(e)}


async def get_finnhub_company_peers(
    ticker: str,
) -> dict[str, Any]:
    """Get peer/comparable company tickers.

    Returns a list of ticker symbols that Finnhub considers
    peers or comparable companies. Useful for competitive
    analysis and relative valuation.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary containing peer ticker symbols
    """
    try:
        await finnhub_limiter.acquire()
        loop = asyncio.get_event_loop()
        peers = await loop.run_in_executor(
            _executor,
            lambda: _get_provider().get_company_peers(ticker),
        )
        return {
            "status": "success",
            "ticker": ticker,
            "count": len(peers),
            "peers": peers,
        }
    except Exception as e:
        logger.error("finnhub_company_peers failed for %s: %s", ticker, e)
        return {"status": "error", "ticker": ticker, "error": str(e)}


async def get_finnhub_economic_calendar(
    from_date: str | None = None,
    to_date: str | None = None,
) -> dict[str, Any]:
    """Get upcoming economic events and data releases.

    Shows scheduled economic indicators (GDP, CPI, FOMC, jobs, etc.)
    with expected and actual values. Useful for macro-aware trading.

    Args:
        from_date: Start date YYYY-MM-DD (defaults to today)
        to_date: End date YYYY-MM-DD (defaults to 7 days from now)

    Returns:
        Dictionary containing economic calendar events
    """
    try:
        await finnhub_limiter.acquire()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            lambda: _get_provider().get_economic_calendar(from_date, to_date),
        )
        events = result.get("economicCalendar", [])
        return {
            "status": "success",
            "count": len(events),
            "economic_events": events,
        }
    except Exception as e:
        logger.error("finnhub_economic_calendar failed: %s", e)
        return {"status": "error", "error": str(e)}


async def get_finnhub_market_news(
    category: str = "general",
    min_id: int = 0,
) -> dict[str, Any]:
    """Get broad market news feed.

    Provides a stream of market-wide news articles across categories
    including general market, forex, crypto, and M&A news.

    Args:
        category: News category (general, forex, crypto, merger)
        min_id: Minimum article ID for pagination (default 0)

    Returns:
        Dictionary containing market news articles
    """
    try:
        await finnhub_limiter.acquire()
        loop = asyncio.get_event_loop()
        news = await loop.run_in_executor(
            _executor,
            lambda: _get_provider().get_market_news(category, min_id),
        )
        return {
            "status": "success",
            "category": category,
            "count": len(news),
            "articles": news,
        }
    except Exception as e:
        logger.error("finnhub_market_news failed: %s", e)
        return {"status": "error", "category": category, "error": str(e)}
