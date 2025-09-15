"""
Cache manager module.

This module provides a bridge to import CacheManager and related cache utilities.
The actual implementation is in cache.py but this module provides the expected import path.
"""

from .cache import (
    CacheManager,
    cleanup_redis_pool,
    clear_cache,
    ensure_timezone_naive,
    generate_cache_key,
    get_cache_stats,
    get_from_cache,
    get_redis_client,
    normalize_timezone,
    reset_cache_stats,
    save_to_cache,
)

__all__ = [
    "CacheManager",
    "get_cache_stats",
    "reset_cache_stats",
    "get_from_cache",
    "save_to_cache",
    "clear_cache",
    "generate_cache_key",
    "ensure_timezone_naive",
    "normalize_timezone",
    "get_redis_client",
    "cleanup_redis_pool",
]
