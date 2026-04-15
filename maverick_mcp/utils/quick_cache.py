"""
Quick in-memory cache decorator for development.

This module provides a simple LRU cache decorator with TTL support
to avoid repeated API calls during development and testing.
"""

import asyncio
import functools
import hashlib
import inspect
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
        return hashlib.sha256(key_str.encode()).hexdigest()

    # ── sync accessors (no event-loop overhead) ──────────────────────
    def get_sync(self, key: str) -> Any | None:
        """Get value from cache (sync path — no event-loop needed)."""
        if key in self.cache:
            value, expiry = self.cache[key]
            if time.time() < expiry:
                self.cache.move_to_end(key)
                self.hits += 1
                return value
            else:
                del self.cache[key]
        self.misses += 1
        return None

    def set_sync(self, key: str, value: Any, ttl_seconds: float) -> None:
        """Set value in cache (sync path — no event-loop needed)."""
        expiry = time.time() + ttl_seconds
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        self.cache[key] = (value, expiry)

    # ── async accessors (lock-protected for concurrent coroutines) ──
    async def get(self, key: str) -> Any | None:
        """Get value from cache if not expired."""
        async with self._lock:
            return self.get_sync(key)

    async def set(self, key: str, value: Any, ttl_seconds: float):
        """Set value in cache with TTL."""
        async with self._lock:
            self.set_sync(key, value, ttl_seconds)

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
            _maybe_emit_stats()
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
            # Use sync cache accessors — no event-loop creation overhead.
            cache_key = _cache.make_key(
                f"{key_prefix}:{func.__name__}" if key_prefix else func.__name__,
                args,
                kwargs,
            )

            # Try to get from cache
            cached_value = _cache.get_sync(cache_key)
            _maybe_emit_stats()
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

            # Cache miss — execute the sync function, then cache
            result = func(*args, **kwargs)
            _cache.set_sync(cache_key, result, ttl_seconds)
            return result

        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
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


# ── observability: automatic hit-rate emission ──────────────────────
#
# The audit flagged that ``hits`` / ``misses`` were being tracked but never
# exposed: operators had no way to tell whether the cache was pulling its
# weight. The original design required opt-in via ``log_stats=True`` on
# each decorator call, which nobody set.
#
# We now emit a consolidated snapshot every ``_STATS_EMIT_INTERVAL_S`` or
# every ``_STATS_EMIT_REQUEST_STEP`` requests, whichever comes first, at
# INFO level. No new threads, no event loop — the hook piggybacks on the
# hot path via ``_record_access``.

_STATS_EMIT_INTERVAL_S = 300.0  # 5 minutes
_STATS_EMIT_REQUEST_STEP = 500
_last_stats_emit_t = time.time()
_requests_since_emit = 0


def _maybe_emit_stats() -> None:
    """Emit a consolidated hit-rate snapshot if the interval has elapsed.

    Cheap by design: two int reads, one wall-clock read, at most one log
    line per interval. Called from the decorator wrappers on every
    request — must not do anything heavy. Hidden behind ``settings.api.debug
    or MAVERICK_QUICK_CACHE_STATS=1`` so a production STDIO transport does
    not spam stdout.
    """
    import os as _os

    enabled = settings.api.debug or _os.getenv("MAVERICK_QUICK_CACHE_STATS") == "1"
    if not enabled:
        return

    global _last_stats_emit_t, _requests_since_emit
    _requests_since_emit += 1

    now = time.time()
    if (
        _requests_since_emit < _STATS_EMIT_REQUEST_STEP
        and (now - _last_stats_emit_t) < _STATS_EMIT_INTERVAL_S
    ):
        return

    stats = _cache.get_stats()
    _last_stats_emit_t = now
    _requests_since_emit = 0
    logger.info(
        "quick_cache stats",
        extra={
            "quick_cache_hits": stats["hits"],
            "quick_cache_misses": stats["misses"],
            "quick_cache_hit_rate_pct": stats["hit_rate"],
            "quick_cache_size": stats["size"],
        },
    )


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
