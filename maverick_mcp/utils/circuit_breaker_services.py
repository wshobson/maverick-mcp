"""
Service-specific circuit breakers — backward-compatible facade.

The actual circuit breaker implementation lives in ``circuit_breaker_adapter.py``.
This module preserves the service-specific classes and global instances that
the rest of the codebase depends on, while delegating to the adapter.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pandas as pd
import requests
from requests.exceptions import ConnectionError, HTTPError, RequestException, Timeout

from maverick_mcp.utils.circuit_breaker import (
    CircuitBreakerConfig,
    FailureDetectionStrategy,
)
from maverick_mcp.utils.circuit_breaker_adapter import (
    MaverickCircuitBreaker,
    get_or_create_breaker,
)
from maverick_mcp.utils.fallback_strategies import (
    ECONOMIC_DATA_FALLBACK,
    MARKET_DATA_FALLBACK,
    NEWS_FALLBACK,
    STOCK_DATA_FALLBACK_CHAIN,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Service-specific configurations (kept for backward compat)
# ---------------------------------------------------------------------------

YFINANCE_CONFIG = CircuitBreakerConfig(
    name="yfinance",
    failure_threshold=3,
    failure_rate_threshold=0.5,
    timeout_threshold=30.0,
    recovery_timeout=120,
    success_threshold=2,
    window_size=300,
    detection_strategy=FailureDetectionStrategy.COMBINED,
    expected_exceptions=(Exception,),
)

FINVIZ_CONFIG = CircuitBreakerConfig(
    name="finviz",
    failure_threshold=5,
    failure_rate_threshold=0.6,
    timeout_threshold=20.0,
    recovery_timeout=180,
    success_threshold=3,
    window_size=300,
    detection_strategy=FailureDetectionStrategy.COMBINED,
    expected_exceptions=(Exception,),
)

FRED_CONFIG = CircuitBreakerConfig(
    name="fred_api",
    failure_threshold=5,
    failure_rate_threshold=0.5,
    timeout_threshold=15.0,
    recovery_timeout=300,
    success_threshold=3,
    window_size=600,
    detection_strategy=FailureDetectionStrategy.COMBINED,
    expected_exceptions=(Exception,),
)

EXTERNAL_API_CONFIG = CircuitBreakerConfig(
    name="external_api",
    failure_threshold=3,
    failure_rate_threshold=0.4,
    timeout_threshold=10.0,
    recovery_timeout=60,
    success_threshold=2,
    window_size=300,
    detection_strategy=FailureDetectionStrategy.COMBINED,
    expected_exceptions=(RequestException, HTTPError, Timeout, ConnectionError),
)

TIINGO_CONFIG = CircuitBreakerConfig(
    name="tiingo",
    failure_threshold=3,
    failure_rate_threshold=0.5,
    timeout_threshold=15.0,
    recovery_timeout=120,
    success_threshold=2,
    window_size=300,
    detection_strategy=FailureDetectionStrategy.COMBINED,
    expected_exceptions=(Exception,),
)

FINNHUB_CONFIG = CircuitBreakerConfig(
    name="finnhub",
    failure_threshold=5,
    failure_rate_threshold=0.5,
    timeout_threshold=15.0,
    recovery_timeout=120,
    success_threshold=3,
    window_size=300,
    detection_strategy=FailureDetectionStrategy.COMBINED,
    expected_exceptions=(Exception,),
)

HTTP_CONFIG = CircuitBreakerConfig(
    name="http_general",
    failure_threshold=5,
    failure_rate_threshold=0.6,
    timeout_threshold=30.0,
    recovery_timeout=60,
    success_threshold=3,
    window_size=300,
    detection_strategy=FailureDetectionStrategy.FAILURE_RATE,
    expected_exceptions=(RequestException, HTTPError, Timeout, ConnectionError),
)


# ---------------------------------------------------------------------------
# Service wrapper classes — delegate to get_or_create_breaker()
# ---------------------------------------------------------------------------


class StockDataCircuitBreaker(MaverickCircuitBreaker):
    """Circuit breaker for stock data APIs (yfinance)."""

    def __init__(self) -> None:
        breaker = get_or_create_breaker("yfinance")
        self.__dict__ = breaker.__dict__
        self.fallback_chain = STOCK_DATA_FALLBACK_CHAIN

    def fetch_with_fallback(
        self,
        fetch_func: Callable,
        symbol: str,
        start_date: str,
        end_date: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fetch stock data with circuit breaker and fallback."""
        try:
            return self.call_sync(fetch_func, symbol, start_date, end_date, **kwargs)
        except Exception as e:
            logger.warning(
                "Primary stock data fetch failed for %s: %s. "
                "Attempting fallback strategies.",
                symbol,
                e,
            )
            return self.fallback_chain.execute_sync(
                symbol, start_date, end_date, **kwargs
            )

    async def fetch_with_fallback_async(
        self,
        fetch_func: Callable,
        symbol: str,
        start_date: str,
        end_date: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Async version of fetch_with_fallback."""
        try:
            return await self.call_async(
                fetch_func, symbol, start_date, end_date, **kwargs
            )
        except Exception as e:
            logger.warning(
                "Primary stock data fetch failed for %s: %s. "
                "Attempting fallback strategies.",
                symbol,
                e,
            )
            return await self.fallback_chain.execute_async(
                symbol, start_date, end_date, **kwargs
            )


class MarketDataCircuitBreaker(MaverickCircuitBreaker):
    """Circuit breaker for market data APIs (finviz, External API)."""

    def __init__(self, service_name: str = "market_data") -> None:
        if service_name == "finviz":
            name = "finviz"
        elif service_name == "external_api":
            name = "external_api"
        else:
            name = "finviz"

        breaker = get_or_create_breaker(name)
        self.__dict__ = breaker.__dict__
        self.fallback = MARKET_DATA_FALLBACK

    def fetch_with_fallback(
        self, fetch_func: Callable, mover_type: str = "gainers", **kwargs: Any
    ) -> dict:
        """Fetch market data with circuit breaker and fallback."""
        try:
            return self.call_sync(fetch_func, mover_type, **kwargs)
        except Exception as e:
            logger.warning(
                "Market data fetch failed for %s: %s. Returning fallback data.",
                mover_type,
                e,
            )
            return self.fallback.execute_sync(mover_type, **kwargs)


class EconomicDataCircuitBreaker(MaverickCircuitBreaker):
    """Circuit breaker for economic data APIs (FRED)."""

    def __init__(self) -> None:
        breaker = get_or_create_breaker("fred_api")
        self.__dict__ = breaker.__dict__
        self.fallback = ECONOMIC_DATA_FALLBACK

    def fetch_with_fallback(
        self,
        fetch_func: Callable,
        series_id: str,
        start_date: str,
        end_date: str,
        **kwargs: Any,
    ) -> pd.Series:
        """Fetch economic data with circuit breaker and fallback."""
        try:
            return self.call_sync(fetch_func, series_id, start_date, end_date, **kwargs)
        except Exception as e:
            logger.warning(
                "Economic data fetch failed for %s: %s. Using fallback values.",
                series_id,
                e,
            )
            return self.fallback.execute_sync(series_id, start_date, end_date, **kwargs)


class NewsDataCircuitBreaker(MaverickCircuitBreaker):
    """Circuit breaker for news/sentiment APIs."""

    def __init__(self) -> None:
        breaker = get_or_create_breaker("news_api")
        self.__dict__ = breaker.__dict__
        self.fallback = NEWS_FALLBACK

    def fetch_with_fallback(
        self, fetch_func: Callable, symbol: str, **kwargs: Any
    ) -> dict:
        """Fetch news data with circuit breaker and fallback."""
        try:
            return self.call_sync(fetch_func, symbol, **kwargs)
        except Exception as e:
            logger.warning(
                "News data fetch failed for %s: %s. Returning empty news data.",
                symbol,
                e,
            )
            return self.fallback.execute_sync(symbol, **kwargs)


class HttpCircuitBreaker(MaverickCircuitBreaker):
    """General circuit breaker for HTTP requests."""

    def __init__(self) -> None:
        breaker = get_or_create_breaker("http_general")
        self.__dict__ = breaker.__dict__

    def request_with_circuit_breaker(
        self,
        method: str,
        url: str,
        session: requests.Session | None = None,
        **kwargs: Any,
    ) -> requests.Response:
        """Make HTTP request with circuit breaker protection."""

        def make_request() -> requests.Response:
            if "timeout" not in kwargs:
                kwargs["timeout"] = self.config.timeout_threshold
            if session:
                return session.request(method, url, **kwargs)
            return requests.request(method, url, **kwargs)

        return self.call_sync(make_request)


class FinnhubCircuitBreaker(MaverickCircuitBreaker):
    """Circuit breaker for Finnhub API calls."""

    def __init__(self) -> None:
        breaker = get_or_create_breaker("finnhub")
        self.__dict__ = breaker.__dict__


# ---------------------------------------------------------------------------
# Global instances for reuse
# ---------------------------------------------------------------------------

stock_data_breaker = StockDataCircuitBreaker()
market_data_breaker = MarketDataCircuitBreaker()
economic_data_breaker = EconomicDataCircuitBreaker()
news_data_breaker = NewsDataCircuitBreaker()
http_breaker = HttpCircuitBreaker()
finnhub_breaker = FinnhubCircuitBreaker()


def get_service_circuit_breaker(service: str) -> MaverickCircuitBreaker:
    """Get a circuit breaker for a specific service."""
    service_breakers: dict[str, MaverickCircuitBreaker] = {
        "yfinance": stock_data_breaker,
        "finviz": market_data_breaker,
        "fred": economic_data_breaker,
        "external_api": MarketDataCircuitBreaker("external_api"),
        "tiingo": get_or_create_breaker("tiingo"),
        "finnhub": finnhub_breaker,
        "news": news_data_breaker,
        "http": http_breaker,
    }

    breaker = service_breakers.get(service)
    if not breaker:
        logger.warning(
            "No specific circuit breaker for service '%s', using HTTP breaker",
            service,
        )
        return http_breaker

    return breaker
