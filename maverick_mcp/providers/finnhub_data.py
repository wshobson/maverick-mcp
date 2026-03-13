"""Finnhub alternative data provider with in-memory TTL caching.

Wraps the finnhub-python client to provide earnings calendars,
analyst recommendations, institutional ownership, company news,
economic calendar, and backup quote data.

Free tier: 60 API calls per minute.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Any

import finnhub

from maverick_mcp.config.settings import get_settings

logger = logging.getLogger("maverick_mcp.providers.finnhub_data")

settings = get_settings()


class FinnhubDataProvider:
    """Provider for Finnhub alternative data with TTL caching and rate limiting."""

    def __init__(self, cache_ttl_seconds: int | None = None) -> None:
        self._cache: dict[str, tuple[float, Any]] = {}
        self._cache_ttl = cache_ttl_seconds or settings.finnhub.cache_ttl_seconds

        # Rate limiter (token bucket)
        self._rate_lock = threading.Lock()
        self._tokens = float(settings.finnhub.rate_limit_per_minute)
        self._max_tokens = float(settings.finnhub.rate_limit_per_minute)
        self._last_refill = time.monotonic()
        self._refill_rate = self._max_tokens / 60.0  # tokens per second

        # Initialise client
        api_key = settings.finnhub.api_key
        if api_key:
            self._client: finnhub.Client | None = finnhub.Client(api_key=api_key)
            logger.info("Finnhub provider initialised (API key configured)")
        else:
            self._client = None
            logger.warning(
                "Finnhub API key not configured — "
                "set FINNHUB_API_KEY for earnings, analyst, and news data"
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_company_news(
        self,
        ticker: str,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch company-specific news articles from Finnhub.

        Args:
            ticker: Stock ticker symbol.
            from_date: Start date YYYY-MM-DD (defaults to 7 days ago).
            to_date: End date YYYY-MM-DD (defaults to today).

        Returns:
            List of news article dicts with headline, source, url, datetime, summary.
        """
        if not self._api_available():
            return []

        if not from_date:
            from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        if not to_date:
            to_date = datetime.now().strftime("%Y-%m-%d")

        cache_key = f"news:{ticker}:{from_date}:{to_date}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            if not self._check_rate_limit():
                logger.debug("Finnhub rate limit reached, returning cached/empty")
                return []
            result = self._client.company_news(ticker, _from=from_date, to=to_date)
            self._set_cached(cache_key, result)
            return result
        except Exception as e:
            logger.error("Finnhub company_news failed for %s: %s", ticker, e)
            return []

    def get_earnings_calendar(
        self,
        from_date: str | None = None,
        to_date: str | None = None,
        ticker: str | None = None,
    ) -> dict[str, Any]:
        """Fetch upcoming/recent earnings with EPS estimates.

        Args:
            from_date: Start date YYYY-MM-DD (defaults to today).
            to_date: End date YYYY-MM-DD (defaults to 14 days from now).
            ticker: Optional ticker to filter results.

        Returns:
            Dict with earningsCalendar list.
        """
        if not self._api_available():
            return {"earningsCalendar": []}

        if not from_date:
            from_date = datetime.now().strftime("%Y-%m-%d")
        if not to_date:
            to_date = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")

        cache_key = f"earnings_cal:{ticker or 'all'}:{from_date}:{to_date}"
        cached = self._get_cached(cache_key, ttl=3600)  # 1 hour
        if cached is not None:
            return cached

        try:
            if not self._check_rate_limit():
                return {"earningsCalendar": []}
            kwargs: dict[str, Any] = {"_from": from_date, "to": to_date}
            if ticker:
                kwargs["symbol"] = ticker
            result = self._client.earnings_calendar(**kwargs)
            self._set_cached(cache_key, result, ttl=3600)
            return result
        except Exception as e:
            logger.error("Finnhub earnings_calendar failed: %s", e)
            return {"earningsCalendar": []}

    def get_earnings_surprises(
        self, ticker: str, limit: int = 4
    ) -> list[dict[str, Any]]:
        """Fetch historical earnings surprises (beat/miss data).

        Args:
            ticker: Stock ticker symbol.
            limit: Number of quarters to return (default 4).

        Returns:
            List of dicts with actual, estimate, period, surprise, surprisePercent.
        """
        if not self._api_available():
            return []

        cache_key = f"earnings_surprise:{ticker}:{limit}"
        cached = self._get_cached(cache_key, ttl=3600)
        if cached is not None:
            return cached

        try:
            if not self._check_rate_limit():
                return []
            result = self._client.company_earnings(ticker, limit=limit)
            self._set_cached(cache_key, result, ttl=3600)
            return result
        except Exception as e:
            logger.error("Finnhub company_earnings failed for %s: %s", ticker, e)
            return []

    def get_recommendation_trends(self, ticker: str) -> list[dict[str, Any]]:
        """Fetch analyst recommendation trends (buy/hold/sell consensus).

        Args:
            ticker: Stock ticker symbol.

        Returns:
            List of dicts with buy, hold, sell, strongBuy, strongSell, period.
        """
        if not self._api_available():
            return []

        cache_key = f"recommendations:{ticker}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            if not self._check_rate_limit():
                return []
            result = self._client.recommendation_trends(ticker)
            self._set_cached(cache_key, result)
            return result
        except Exception as e:
            logger.error("Finnhub recommendation_trends failed for %s: %s", ticker, e)
            return []

    def get_institutional_ownership(
        self, ticker: str, limit: int = 20
    ) -> dict[str, Any]:
        """Fetch institutional ownership from 13F filings.

        Args:
            ticker: Stock ticker symbol.
            limit: Max number of holders to return.

        Returns:
            Dict with ownership list containing name, share, change, filingDate.
        """
        if not self._api_available():
            return {"ownership": []}

        cache_key = f"ownership:{ticker}:{limit}"
        cached = self._get_cached(cache_key, ttl=3600)
        if cached is not None:
            return cached

        try:
            if not self._check_rate_limit():
                return {"ownership": []}
            result = self._client.ownership(ticker, limit=limit)
            self._set_cached(cache_key, result, ttl=3600)
            return result
        except Exception as e:
            logger.error("Finnhub ownership failed for %s: %s", ticker, e)
            return {"ownership": []}

    def get_company_peers(self, ticker: str) -> list[str]:
        """Fetch peer/comparable company tickers.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            List of peer ticker symbols.
        """
        if not self._api_available():
            return []

        cache_key = f"peers:{ticker}"
        cached = self._get_cached(cache_key, ttl=86400)  # 24 hours
        if cached is not None:
            return cached

        try:
            if not self._check_rate_limit():
                return []
            result = self._client.company_peers(ticker)
            self._set_cached(cache_key, result, ttl=86400)
            return result
        except Exception as e:
            logger.error("Finnhub company_peers failed for %s: %s", ticker, e)
            return []

    def get_economic_calendar(
        self,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> dict[str, Any]:
        """Fetch upcoming economic events and releases.

        Args:
            from_date: Start date YYYY-MM-DD (defaults to today).
            to_date: End date YYYY-MM-DD (defaults to 7 days from now).

        Returns:
            Dict with economicCalendar list of events.
        """
        if not self._api_available():
            return {"economicCalendar": []}

        if not from_date:
            from_date = datetime.now().strftime("%Y-%m-%d")
        if not to_date:
            to_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")

        cache_key = f"econ_cal:{from_date}:{to_date}"
        cached = self._get_cached(cache_key, ttl=3600)
        if cached is not None:
            return cached

        try:
            if not self._check_rate_limit():
                return {"economicCalendar": []}
            result = self._client.economic_calendar(
                _from=from_date, to=to_date
            )
            self._set_cached(cache_key, result, ttl=3600)
            return result
        except Exception as e:
            logger.error("Finnhub economic_calendar failed: %s", e)
            return {"economicCalendar": []}

    def get_market_news(
        self, category: str = "general", min_id: int = 0
    ) -> list[dict[str, Any]]:
        """Fetch broad market news.

        Args:
            category: News category (general, forex, crypto, merger).
            min_id: Minimum article ID for pagination.

        Returns:
            List of news article dicts.
        """
        if not self._api_available():
            return []

        cache_key = f"market_news:{category}:{min_id}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            if not self._check_rate_limit():
                return []
            result = self._client.general_news(category, min_id=min_id)
            self._set_cached(cache_key, result)
            return result
        except Exception as e:
            logger.error("Finnhub general_news failed: %s", e)
            return []

    def get_quote(self, ticker: str) -> dict[str, Any]:
        """Fetch current quote (backup source when Tiingo/yfinance fail).

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Dict with c (current), h (high), l (low), o (open), pc (prev close).
        """
        if not self._api_available():
            return {}

        cache_key = f"quote:{ticker}"
        cached = self._get_cached(cache_key, ttl=60)  # 1 minute
        if cached is not None:
            return cached

        try:
            if not self._check_rate_limit():
                return {}
            result = self._client.quote(ticker)
            self._set_cached(cache_key, result, ttl=60)
            return result
        except Exception as e:
            logger.error("Finnhub quote failed for %s: %s", ticker, e)
            return {}

    # ------------------------------------------------------------------ #
    # Cache helpers
    # ------------------------------------------------------------------ #

    def _get_cached(
        self, key: str, ttl: int | None = None
    ) -> Any | None:
        """Return cached value if still valid, else None."""
        if key not in self._cache:
            return None
        ts, value = self._cache[key]
        effective_ttl = ttl if ttl is not None else self._cache_ttl
        if (time.monotonic() - ts) < effective_ttl:
            return value
        return None

    def _set_cached(
        self, key: str, value: Any, ttl: int | None = None  # noqa: ARG002
    ) -> None:
        """Store value in cache with current timestamp."""
        self._cache[key] = (time.monotonic(), value)

    # ------------------------------------------------------------------ #
    # Rate limiter (token bucket)
    # ------------------------------------------------------------------ #

    def _check_rate_limit(self) -> bool:
        """Check and consume a rate-limit token. Returns True if allowed."""
        with self._rate_lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(
                self._max_tokens,
                self._tokens + elapsed * self._refill_rate,
            )
            self._last_refill = now

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False

    def _api_available(self) -> bool:
        """Check if the Finnhub client is configured."""
        return self._client is not None
