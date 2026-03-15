"""
Circuit breaker decorators — backward-compatible facade.

Simplified decorators backed by the ``circuitbreaker`` library.
All production call sites use ``use_fallback=False``, so the complex
fallback-in-decorator logic is removed (fallback chains are handled
at a higher level in fallback_strategies.py).
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Callable
from typing import TypeVar, cast

from maverick_mcp.utils.circuit_breaker_adapter import get_or_create_breaker

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Generic factory
# ---------------------------------------------------------------------------


def _service_decorator(service_name: str) -> Callable:
    """Create a decorator that routes calls through the named circuit breaker."""
    breaker = get_or_create_breaker(service_name)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await breaker.call_async(func, *args, **kwargs)

            return cast(Callable[..., T], async_wrapper)
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return breaker.call_sync(func, *args, **kwargs)

            return cast(Callable[..., T], sync_wrapper)

    return decorator


# ---------------------------------------------------------------------------
# Backward-compatible decorator factories
# ---------------------------------------------------------------------------


def with_stock_data_circuit_breaker(
    use_fallback: bool = True, fallback_on_open: bool = True
) -> Callable:
    """Decorator for stock data fetching functions."""
    return _service_decorator("yfinance")


def with_market_data_circuit_breaker(
    use_fallback: bool = True, service: str = "finviz"
) -> Callable:
    """Decorator for market data fetching functions."""
    return _service_decorator(service)


def with_economic_data_circuit_breaker(use_fallback: bool = True) -> Callable:
    """Decorator for economic data fetching functions."""
    return _service_decorator("fred_api")


def with_news_circuit_breaker(use_fallback: bool = True) -> Callable:
    """Decorator for news/sentiment API calls."""
    return _service_decorator("news_api")


def with_http_circuit_breaker(
    timeout: float | None = None, use_session: bool = False
) -> Callable:
    """Decorator for general HTTP requests."""
    return _service_decorator("http_general")


def with_finnhub_circuit_breaker(use_fallback: bool = False) -> Callable:
    """Decorator for Finnhub API calls."""
    return _service_decorator("finnhub")


# ---------------------------------------------------------------------------
# Generic class-method decorator
# ---------------------------------------------------------------------------

_SERVICE_MAP = {
    "yfinance": "yfinance",
    "stock": "yfinance",
    "finviz": "finviz",
    "external_api": "external_api",
    "market": "finviz",
    "fred": "fred_api",
    "economic": "fred_api",
    "news": "news_api",
    "sentiment": "news_api",
    "finnhub": "finnhub",
    "http": "http_general",
}


def circuit_breaker_method(
    service: str = "http", use_fallback: bool = True, **kwargs
) -> Callable:
    """Generic circuit breaker decorator for class methods."""
    resolved = _SERVICE_MAP.get(service, "http_general")
    return _service_decorator(resolved)
