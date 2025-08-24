"""
Decorators for easy circuit breaker integration.
Provides convenient decorators for common external service patterns.
"""

import asyncio
import functools
import logging
from collections.abc import Callable
from typing import TypeVar, cast

from maverick_mcp.config.settings import get_settings
from maverick_mcp.utils.circuit_breaker_services import (
    economic_data_breaker,
    http_breaker,
    market_data_breaker,
    news_data_breaker,
    stock_data_breaker,
)

logger = logging.getLogger(__name__)
settings = get_settings()

T = TypeVar("T")


def with_stock_data_circuit_breaker(
    use_fallback: bool = True, fallback_on_open: bool = True
) -> Callable:
    """
    Decorator for stock data fetching functions.

    Args:
        use_fallback: Whether to use fallback strategies on failure
        fallback_on_open: Whether to use fallback when circuit is open

    Example:
        @with_stock_data_circuit_breaker()
        def get_stock_data(symbol: str, start: str, end: str) -> pd.DataFrame:
            return yf.download(symbol, start=start, end=end)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if use_fallback and len(args) >= 3:
                    # Extract symbol, start, end from args
                    symbol = args[0] if args else kwargs.get("symbol", "UNKNOWN")
                    start_date = (
                        args[1] if len(args) > 1 else kwargs.get("start_date", "")
                    )
                    end_date = args[2] if len(args) > 2 else kwargs.get("end_date", "")

                    return await stock_data_breaker.fetch_with_fallback_async(
                        func, symbol, start_date, end_date, **kwargs
                    )
                else:
                    return await stock_data_breaker.call_async(func, *args, **kwargs)

            return cast(Callable[..., T], async_wrapper)
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                if use_fallback and len(args) >= 3:
                    # Extract symbol, start, end from args
                    symbol = args[0] if args else kwargs.get("symbol", "UNKNOWN")
                    start_date = (
                        args[1] if len(args) > 1 else kwargs.get("start_date", "")
                    )
                    end_date = args[2] if len(args) > 2 else kwargs.get("end_date", "")

                    return stock_data_breaker.fetch_with_fallback(
                        func, symbol, start_date, end_date, **kwargs
                    )
                else:
                    return stock_data_breaker.call_sync(func, *args, **kwargs)

            return cast(Callable[..., T], sync_wrapper)

    return decorator


def with_market_data_circuit_breaker(
    use_fallback: bool = True, service: str = "finviz"
) -> Callable:
    """
    Decorator for market data fetching functions.

    Args:
        use_fallback: Whether to use fallback strategies on failure
        service: Service name (finviz, external_api)

    Example:
        @with_market_data_circuit_breaker(service="finviz")
        def get_top_gainers() -> dict:
            return fetch_finviz_gainers()
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Get appropriate breaker
        if service == "external_api":
            from maverick_mcp.utils.circuit_breaker_services import (
                MarketDataCircuitBreaker,
            )

            breaker = MarketDataCircuitBreaker("external_api")
        else:
            breaker = market_data_breaker

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if use_fallback:
                    # Try to extract mover_type
                    mover_type = kwargs.get("mover_type", "market_data")
                    try:
                        return await breaker.call_async(func, *args, **kwargs)
                    except Exception as e:
                        logger.warning(f"Market data fetch failed: {e}, using fallback")
                        return breaker.fallback.execute_sync(mover_type)
                else:
                    return await breaker.call_async(func, *args, **kwargs)

            return cast(Callable[..., T], async_wrapper)
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                if use_fallback:
                    # Try to extract mover_type
                    mover_type = kwargs.get("mover_type", "market_data")
                    return breaker.fetch_with_fallback(func, mover_type, **kwargs)
                else:
                    return breaker.call_sync(func, *args, **kwargs)

            return cast(Callable[..., T], sync_wrapper)

    return decorator


def with_economic_data_circuit_breaker(use_fallback: bool = True) -> Callable:
    """
    Decorator for economic data fetching functions.

    Args:
        use_fallback: Whether to use fallback strategies on failure

    Example:
        @with_economic_data_circuit_breaker()
        def get_gdp_data(start: str, end: str) -> pd.Series:
            return fred.get_series("GDP", start, end)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if use_fallback and (args or "series_id" in kwargs):
                    # Extract series_id and dates
                    series_id = args[0] if args else kwargs.get("series_id", "UNKNOWN")
                    start_date = (
                        args[1] if len(args) > 1 else kwargs.get("start_date", "")
                    )
                    end_date = args[2] if len(args) > 2 else kwargs.get("end_date", "")

                    try:
                        return await economic_data_breaker.call_async(
                            func, *args, **kwargs
                        )
                    except Exception as e:
                        logger.warning(
                            f"Economic data fetch failed: {e}, using fallback"
                        )
                        return economic_data_breaker.fallback.execute_sync(
                            series_id, start_date, end_date
                        )
                else:
                    return await economic_data_breaker.call_async(func, *args, **kwargs)

            return cast(Callable[..., T], async_wrapper)
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                if use_fallback and (args or "series_id" in kwargs):
                    # Extract series_id and dates
                    series_id = args[0] if args else kwargs.get("series_id", "UNKNOWN")
                    start_date = (
                        args[1] if len(args) > 1 else kwargs.get("start_date", "")
                    )
                    end_date = args[2] if len(args) > 2 else kwargs.get("end_date", "")

                    return economic_data_breaker.fetch_with_fallback(
                        func, series_id, start_date, end_date, **kwargs
                    )
                else:
                    return economic_data_breaker.call_sync(func, *args, **kwargs)

            return cast(Callable[..., T], sync_wrapper)

    return decorator


def with_news_circuit_breaker(use_fallback: bool = True) -> Callable:
    """
    Decorator for news/sentiment API calls.

    Args:
        use_fallback: Whether to use fallback strategies on failure

    Example:
        @with_news_circuit_breaker()
        def get_stock_news(symbol: str) -> dict:
            return fetch_news_api(symbol)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if use_fallback and (args or "symbol" in kwargs):
                    symbol = args[0] if args else kwargs.get("symbol", "UNKNOWN")
                    try:
                        return await news_data_breaker.call_async(func, *args, **kwargs)
                    except Exception as e:
                        logger.warning(f"News data fetch failed: {e}, using fallback")
                        return news_data_breaker.fallback.execute_sync(symbol)
                else:
                    return await news_data_breaker.call_async(func, *args, **kwargs)

            return cast(Callable[..., T], async_wrapper)
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                if use_fallback and (args or "symbol" in kwargs):
                    symbol = args[0] if args else kwargs.get("symbol", "UNKNOWN")
                    return news_data_breaker.fetch_with_fallback(func, symbol, **kwargs)
                else:
                    return news_data_breaker.call_sync(func, *args, **kwargs)

            return cast(Callable[..., T], sync_wrapper)

    return decorator


def with_http_circuit_breaker(
    timeout: float | None = None, use_session: bool = False
) -> Callable:
    """
    Decorator for general HTTP requests.

    Args:
        timeout: Override default timeout
        use_session: Whether the function uses a requests Session

    Example:
        @with_http_circuit_breaker(timeout=10.0)
        def fetch_api_data(url: str) -> dict:
            response = requests.get(url)
            return response.json()
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Override timeout if specified
                if timeout is not None:
                    kwargs["timeout"] = timeout
                return await http_breaker.call_async(func, *args, **kwargs)

            return cast(Callable[..., T], async_wrapper)
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Override timeout if specified
                if timeout is not None:
                    kwargs["timeout"] = timeout
                return http_breaker.call_sync(func, *args, **kwargs)

            return cast(Callable[..., T], sync_wrapper)

    return decorator


def circuit_breaker_method(
    service: str = "http", use_fallback: bool = True, **breaker_kwargs
) -> Callable:
    """
    Generic circuit breaker decorator for class methods.

    Args:
        service: Service type (yfinance, finviz, fred, news, http)
        use_fallback: Whether to use fallback strategies
        **breaker_kwargs: Additional arguments for the circuit breaker

    Example:
        class DataProvider:
            @circuit_breaker_method(service="yfinance")
            def get_stock_data(self, symbol: str) -> pd.DataFrame:
                return yf.download(symbol)
    """
    # Map service names to decorators
    service_decorators = {
        "yfinance": with_stock_data_circuit_breaker,
        "stock": with_stock_data_circuit_breaker,
        "finviz": lambda **kw: with_market_data_circuit_breaker(service="finviz", **kw),
        "external_api": lambda **kw: with_market_data_circuit_breaker(
            service="external_api", **kw
        ),
        "market": with_market_data_circuit_breaker,
        "fred": with_economic_data_circuit_breaker,
        "economic": with_economic_data_circuit_breaker,
        "news": with_news_circuit_breaker,
        "sentiment": with_news_circuit_breaker,
        "http": with_http_circuit_breaker,
    }

    decorator_func = service_decorators.get(service, with_http_circuit_breaker)
    return decorator_func(use_fallback=use_fallback, **breaker_kwargs)
