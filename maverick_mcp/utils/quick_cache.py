"""
Quick in-memory cache decorator for development.

This module provides a simple LRU cache decorator with TTL support
to avoid repeated API calls during development and testing.
"""

import asyncio
import functools
import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, TypeVar

from maverick_mcp.config.settings import settings
from maverick_mcp.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class QuickCache:
    """Simple in-memory LRU cache with TTL support."""

    def __init__(self, max_size: int = 1000):
        self.cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self._lock = asyncio.Lock()

    def make_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate a cache key from function name and arguments."""
        # Convert args and kwargs to a stable string representation
        key_data = {
            "func": func_name,
            "args": args,
            "kwargs": sorted(kwargs.items()),
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        # Use hash for shorter keys
        return hashlib.md5(key_str.encode()).hexdigest()

    async def get(self, key: str) -> Any | None:
        """Get value from cache if not expired."""
        async with self._lock:
            if key in self.cache:
                value, expiry = self.cache[key]
                if time.time() < expiry:
                    # Move to end (LRU)
                    self.cache.move_to_end(key)
                    self.hits += 1
                    return value
                else:
                    # Expired, remove it
                    del self.cache[key]

            self.misses += 1
            return None

    async def set(self, key: str, value: Any, ttl_seconds: float):
        """Set value in cache with TTL."""
        async with self._lock:
            expiry = time.time() + ttl_seconds

            # Remove oldest if at capacity
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)

            self.cache[key] = (value, expiry)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": round(hit_rate, 2),
            "size": len(self.cache),
            "max_size": self.max_size,
        }

    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0


# Global cache instance
_cache = QuickCache()


def quick_cache(
    ttl_seconds: float = 300,  # 5 minutes default
    max_size: int = 1000,
    key_prefix: str = "",
    log_stats: bool | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for in-memory caching with TTL.

    Args:
        ttl_seconds: Time to live in seconds (default: 300)
        max_size: Maximum cache size (default: 1000)
        key_prefix: Optional prefix for cache keys
        log_stats: Whether to log cache statistics (default: settings.api.debug)

    Usage:
        @quick_cache(ttl_seconds=60)
        async def expensive_api_call(symbol: str):
            return await fetch_data(symbol)

        @quick_cache(ttl_seconds=300, key_prefix="stock_data")
        def get_stock_info(symbol: str, period: str):
            return fetch_stock_data(symbol, period)
    """
    if log_stats is None:
        log_stats = settings.api.debug

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Update global cache size if specified
        if max_size != _cache.max_size:
            _cache.max_size = max_size

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            # Generate cache key
            cache_key = _cache.make_key(
                f"{key_prefix}:{func.__name__}" if key_prefix else func.__name__,
                args,
                kwargs,
            )

            # Try to get from cache
            cached_value = await _cache.get(cache_key)
            if cached_value is not None:
                if log_stats:
                    stats = _cache.get_stats()
                    logger.debug(
                        f"Cache HIT for {func.__name__}",
                        extra={
                            "function": func.__name__,
                            "cache_key": cache_key[:8] + "...",
                            "hit_rate": stats["hit_rate"],
                            "cache_size": stats["size"],
                        },
                    )
                return cached_value

            # Cache miss - execute function
            if log_stats:
                logger.debug(
                    f"Cache MISS for {func.__name__}",
                    extra={
                        "function": func.__name__,
                        "cache_key": cache_key[:8] + "...",
                    },
                )

            # Execute the function
            start_time = time.time()
            # func is guaranteed to be async since we're in async_wrapper
            result = await func(*args, **kwargs)  # type: ignore[misc]
            execution_time = time.time() - start_time

            # Cache the result
            await _cache.set(cache_key, result, ttl_seconds)

            if log_stats:
                stats = _cache.get_stats()
                logger.debug(
                    f"Cached result for {func.__name__}",
                    extra={
                        "function": func.__name__,
                        "execution_time": round(execution_time, 3),
                        "ttl_seconds": ttl_seconds,
                        "cache_stats": stats,
                    },
                )

            return result

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            # For sync functions, we need to run the async cache operations
            # in a thread to avoid blocking
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                cache_key = _cache.make_key(
                    f"{key_prefix}:{func.__name__}" if key_prefix else func.__name__,
                    args,
                    kwargs,
                )

                # Try to get from cache (sync version)
                cached_value = loop.run_until_complete(_cache.get(cache_key))
                if cached_value is not None:
                    if log_stats:
                        stats = _cache.get_stats()
                        logger.debug(
                            f"Cache HIT for {func.__name__}",
                            extra={
                                "function": func.__name__,
                                "hit_rate": stats["hit_rate"],
                            },
                        )
                    return cached_value

                # Cache miss
                result = func(*args, **kwargs)

                # Cache the result
                loop.run_until_complete(_cache.set(cache_key, result, ttl_seconds))

                return result
            finally:
                loop.close()

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        else:
            return sync_wrapper

    return decorator


def get_cache_stats() -> dict[str, Any]:
    """Get global cache statistics."""
    return _cache.get_stats()


def clear_cache():
    """Clear the global cache."""
    _cache.clear()
    logger.info("Cache cleared")


# Convenience decorators with common TTLs
cache_1min = functools.partial(quick_cache, ttl_seconds=60)
cache_5min = functools.partial(quick_cache, ttl_seconds=300)
cache_15min = functools.partial(quick_cache, ttl_seconds=900)
cache_1hour = functools.partial(quick_cache, ttl_seconds=3600)


# Example usage for API calls
@quick_cache(ttl_seconds=300, key_prefix="stock")
async def cached_stock_data(symbol: str, start_date: str, end_date: str) -> dict:
    """Example of caching stock data API calls."""
    # This would normally make an expensive API call
    logger.info(f"Fetching stock data for {symbol}")
    # Simulate API call
    await asyncio.sleep(0.1)
    return {
        "symbol": symbol,
        "start": start_date,
        "end": end_date,
        "data": "mock_data",
    }


# Cache management commands for development
if settings.api.debug:

    @quick_cache(ttl_seconds=1)  # Very short TTL for testing
    def test_cache_function(value: str) -> str:
        """Test function for cache debugging."""
        return f"processed_{value}_{time.time()}"
