"""
Service-specific circuit breakers for external APIs.
Provides pre-configured circuit breakers for different external services.
"""

import logging

import pandas as pd
import requests
from requests.exceptions import ConnectionError, HTTPError, RequestException, Timeout

from maverick_mcp.config.settings import get_settings
from maverick_mcp.utils.circuit_breaker_enhanced import (
    CircuitBreakerConfig,
    EnhancedCircuitBreaker,
    FailureDetectionStrategy,
)
from maverick_mcp.utils.fallback_strategies import (
    ECONOMIC_DATA_FALLBACK,
    MARKET_DATA_FALLBACK,
    NEWS_FALLBACK,
    STOCK_DATA_FALLBACK_CHAIN,
)

logger = logging.getLogger(__name__)
settings = get_settings()


# Service-specific configurations
YFINANCE_CONFIG = CircuitBreakerConfig(
    name="yfinance",
    failure_threshold=3,
    failure_rate_threshold=0.5,
    timeout_threshold=30.0,
    recovery_timeout=120,  # 2 minutes
    success_threshold=2,
    window_size=300,  # 5 minutes
    detection_strategy=FailureDetectionStrategy.COMBINED,
    expected_exceptions=(Exception,),  # yfinance can throw various exceptions
)

FINVIZ_CONFIG = CircuitBreakerConfig(
    name="finviz",
    failure_threshold=5,
    failure_rate_threshold=0.6,
    timeout_threshold=20.0,
    recovery_timeout=180,  # 3 minutes
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
    recovery_timeout=300,  # 5 minutes
    success_threshold=3,
    window_size=600,  # 10 minutes
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


class StockDataCircuitBreaker(EnhancedCircuitBreaker):
    """Circuit breaker for stock data APIs (yfinance)."""

    def __init__(self):
        """Initialize with yfinance configuration."""
        super().__init__(YFINANCE_CONFIG)
        self.fallback_chain = STOCK_DATA_FALLBACK_CHAIN

    def fetch_with_fallback(
        self,
        fetch_func: callable,
        symbol: str,
        start_date: str,
        end_date: str,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fetch stock data with circuit breaker and fallback.

        Args:
            fetch_func: The function to fetch data (e.g., yfinance call)
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            **kwargs: Additional arguments for fetch_func

        Returns:
            DataFrame with stock data
        """
        try:
            # Try primary fetch through circuit breaker
            return self.call_sync(fetch_func, symbol, start_date, end_date, **kwargs)
        except Exception as e:
            logger.warning(
                f"Primary stock data fetch failed for {symbol}: {e}. "
                f"Attempting fallback strategies."
            )
            # Execute fallback chain
            return self.fallback_chain.execute_sync(
                symbol, start_date, end_date, **kwargs
            )

    async def fetch_with_fallback_async(
        self,
        fetch_func: callable,
        symbol: str,
        start_date: str,
        end_date: str,
        **kwargs,
    ) -> pd.DataFrame:
        """Async version of fetch_with_fallback."""
        try:
            return await self.call_async(
                fetch_func, symbol, start_date, end_date, **kwargs
            )
        except Exception as e:
            logger.warning(
                f"Primary stock data fetch failed for {symbol}: {e}. "
                f"Attempting fallback strategies."
            )
            return await self.fallback_chain.execute_async(
                symbol, start_date, end_date, **kwargs
            )


class MarketDataCircuitBreaker(EnhancedCircuitBreaker):
    """Circuit breaker for market data APIs (finviz, External API)."""

    def __init__(self, service_name: str = "market_data"):
        """Initialize with market data configuration."""
        if service_name == "finviz":
            config = FINVIZ_CONFIG
        elif service_name == "external_api":
            config = EXTERNAL_API_CONFIG
        else:
            config = FINVIZ_CONFIG  # Default

        super().__init__(config)
        self.fallback = MARKET_DATA_FALLBACK

    def fetch_with_fallback(
        self, fetch_func: callable, mover_type: str = "gainers", **kwargs
    ) -> dict:
        """Fetch market data with circuit breaker and fallback."""
        try:
            return self.call_sync(fetch_func, mover_type, **kwargs)
        except Exception as e:
            logger.warning(
                f"Market data fetch failed for {mover_type}: {e}. "
                f"Returning fallback data."
            )
            return self.fallback.execute_sync(mover_type, **kwargs)


class EconomicDataCircuitBreaker(EnhancedCircuitBreaker):
    """Circuit breaker for economic data APIs (FRED)."""

    def __init__(self):
        """Initialize with FRED configuration."""
        super().__init__(FRED_CONFIG)
        self.fallback = ECONOMIC_DATA_FALLBACK

    def fetch_with_fallback(
        self,
        fetch_func: callable,
        series_id: str,
        start_date: str,
        end_date: str,
        **kwargs,
    ) -> pd.Series:
        """Fetch economic data with circuit breaker and fallback."""
        try:
            return self.call_sync(fetch_func, series_id, start_date, end_date, **kwargs)
        except Exception as e:
            logger.warning(
                f"Economic data fetch failed for {series_id}: {e}. "
                f"Using fallback values."
            )
            return self.fallback.execute_sync(series_id, start_date, end_date, **kwargs)


class NewsDataCircuitBreaker(EnhancedCircuitBreaker):
    """Circuit breaker for news/sentiment APIs."""

    def __init__(self):
        """Initialize with news API configuration."""
        # Use a generic config for news APIs
        config = CircuitBreakerConfig(
            name="news_api",
            failure_threshold=3,
            failure_rate_threshold=0.6,
            timeout_threshold=10.0,
            recovery_timeout=300,
            success_threshold=2,
            window_size=600,
            detection_strategy=FailureDetectionStrategy.COMBINED,
            expected_exceptions=(Exception,),
        )
        super().__init__(config)
        self.fallback = NEWS_FALLBACK

    def fetch_with_fallback(self, fetch_func: callable, symbol: str, **kwargs) -> dict:
        """Fetch news data with circuit breaker and fallback."""
        try:
            return self.call_sync(fetch_func, symbol, **kwargs)
        except Exception as e:
            logger.warning(
                f"News data fetch failed for {symbol}: {e}. Returning empty news data."
            )
            return self.fallback.execute_sync(symbol, **kwargs)


class HttpCircuitBreaker(EnhancedCircuitBreaker):
    """General circuit breaker for HTTP requests."""

    def __init__(self):
        """Initialize with HTTP configuration."""
        super().__init__(HTTP_CONFIG)

    def request_with_circuit_breaker(
        self, method: str, url: str, session: requests.Session | None = None, **kwargs
    ) -> requests.Response:
        """
        Make HTTP request with circuit breaker protection.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: URL to request
            session: Optional requests session
            **kwargs: Additional arguments for requests

        Returns:
            Response object
        """

        def make_request():
            # Ensure timeout is set
            if "timeout" not in kwargs:
                kwargs["timeout"] = self.config.timeout_threshold

            if session:
                return session.request(method, url, **kwargs)
            else:
                return requests.request(method, url, **kwargs)

        return self.call_sync(make_request)


# Global instances for reuse
stock_data_breaker = StockDataCircuitBreaker()
market_data_breaker = MarketDataCircuitBreaker()
economic_data_breaker = EconomicDataCircuitBreaker()
news_data_breaker = NewsDataCircuitBreaker()
http_breaker = HttpCircuitBreaker()


def get_service_circuit_breaker(service: str) -> EnhancedCircuitBreaker:
    """
    Get a circuit breaker for a specific service.

    Args:
        service: Service name (yfinance, finviz, fred, news, http)

    Returns:
        Configured circuit breaker for the service
    """
    service_breakers = {
        "yfinance": stock_data_breaker,
        "finviz": market_data_breaker,
        "fred": economic_data_breaker,
        "external_api": MarketDataCircuitBreaker("external_api"),
        "tiingo": EnhancedCircuitBreaker(TIINGO_CONFIG),
        "news": news_data_breaker,
        "http": http_breaker,
    }

    breaker = service_breakers.get(service)
    if not breaker:
        logger.warning(
            f"No specific circuit breaker for service '{service}', using HTTP breaker"
        )
        return http_breaker

    return breaker
