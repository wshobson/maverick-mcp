"""
Cache manager adapter.

This module provides adapters that make the existing cache system
compatible with the new ICacheManager interface.
"""

import asyncio
import logging
from typing import Any

from maverick_mcp.data.cache import (
    CacheManager as ExistingCacheManager,
)
from maverick_mcp.data.cache import (
    clear_cache,
    get_from_cache,
    save_to_cache,
)
from maverick_mcp.providers.interfaces.cache import CacheConfig, ICacheManager

logger = logging.getLogger(__name__)


class RedisCacheAdapter(ICacheManager):
    """
    Adapter that makes the existing cache system compatible with ICacheManager interface.

    This adapter wraps the existing cache functions and CacheManager class
    to provide the new interface while maintaining all existing functionality.
    """

    def __init__(self, config: CacheConfig | None = None):
        """
        Initialize the cache adapter.

        Args:
            config: Cache configuration (optional, defaults to environment)
        """
        self._config = config
        self._cache_manager = ExistingCacheManager()

        logger.debug("RedisCacheAdapter initialized")

    async def get(self, key: str) -> Any:
        """
        Get data from cache (async wrapper).

        Args:
            key: Cache key to retrieve

        Returns:
            Cached data or None if not found or expired
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, get_from_cache, key)

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """
        Store data in cache (async wrapper).

        Args:
            key: Cache key
            value: Data to cache (must be JSON serializable)
            ttl: Time-to-live in seconds (None for default TTL)

        Returns:
            True if successfully cached, False otherwise
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, save_to_cache, key, value, ttl)

    async def delete(self, key: str) -> bool:
        """
        Delete a key from cache.

        Args:
            key: Cache key to delete

        Returns:
            True if key was deleted, False if key didn't exist
        """
        return await self._cache_manager.delete(key)

    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in cache.

        Args:
            key: Cache key to check

        Returns:
            True if key exists and hasn't expired, False otherwise
        """
        return await self._cache_manager.exists(key)

    async def clear(self, pattern: str | None = None) -> int:
        """
        Clear cache entries.

        Args:
            pattern: Pattern to match keys (e.g., "stock:*")
                    If None, clears all cache entries

        Returns:
            Number of entries cleared
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, clear_cache, pattern)

    async def get_many(self, keys: list[str]) -> dict[str, Any]:
        """
        Get multiple values at once for better performance.

        Args:
            keys: List of cache keys to retrieve

        Returns:
            Dictionary mapping keys to their cached values
            (missing keys will not be in the result)
        """
        return await self._cache_manager.get_many(keys)

    async def set_many(self, items: list[tuple[str, Any, int | None]]) -> int:
        """
        Set multiple values at once for better performance.

        Args:
            items: List of tuples (key, value, ttl)

        Returns:
            Number of items successfully cached
        """
        return await self._cache_manager.batch_save(items)

    async def delete_many(self, keys: list[str]) -> int:
        """
        Delete multiple keys for better performance.

        Args:
            keys: List of keys to delete

        Returns:
            Number of keys successfully deleted
        """
        return await self._cache_manager.batch_delete(keys)

    async def exists_many(self, keys: list[str]) -> dict[str, bool]:
        """
        Check existence of multiple keys for better performance.

        Args:
            keys: List of keys to check

        Returns:
            Dictionary mapping keys to their existence status
        """
        return await self._cache_manager.batch_exists(keys)

    async def count_keys(self, pattern: str) -> int:
        """
        Count keys matching a pattern.

        Args:
            pattern: Pattern to match (e.g., "stock:*")

        Returns:
            Number of matching keys
        """
        return await self._cache_manager.count_keys(pattern)

    async def get_or_set(
        self, key: str, default_value: Any, ttl: int | None = None
    ) -> Any:
        """
        Get value from cache, setting it if it doesn't exist.

        Args:
            key: Cache key
            default_value: Value to set if key doesn't exist
            ttl: Time-to-live for the default value

        Returns:
            Either the existing cached value or the default value
        """
        # Check if key exists
        existing_value = await self.get(key)
        if existing_value is not None:
            return existing_value

        # Set default value and return it
        await self.set(key, default_value, ttl)
        return default_value

    async def increment(self, key: str, amount: int = 1) -> int:
        """
        Increment a numeric value in cache.

        Args:
            key: Cache key
            amount: Amount to increment by

        Returns:
            New value after increment

        Raises:
            ValueError: If the key exists but doesn't contain a numeric value
        """
        # Get current value
        current = await self.get(key)

        if current is None:
            # Key doesn't exist, start from 0
            new_value = amount
        else:
            # Try to convert to int and increment
            try:
                current_int = int(current)
                new_value = current_int + amount
            except (ValueError, TypeError):
                raise ValueError(f"Key {key} contains non-numeric value: {current}")

        # Set the new value
        await self.set(key, new_value)
        return new_value

    async def set_if_not_exists(
        self, key: str, value: Any, ttl: int | None = None
    ) -> bool:
        """
        Set a value only if the key doesn't already exist.

        Args:
            key: Cache key
            value: Value to set
            ttl: Time-to-live in seconds

        Returns:
            True if the value was set, False if key already existed
        """
        # Check if key already exists
        if await self.exists(key):
            return False

        # Key doesn't exist, set the value
        return await self.set(key, value, ttl)

    async def get_ttl(self, key: str) -> int | None:
        """
        Get the remaining time-to-live for a key.

        Args:
            key: Cache key

        Returns:
            Remaining TTL in seconds, None if key doesn't exist or has no TTL
        """
        # This would need to be implemented in the underlying cache manager
        # For now, return None as we don't have TTL introspection in the existing system
        logger.warning(f"TTL introspection not implemented for key: {key}")
        return None

    async def expire(self, key: str, ttl: int) -> bool:
        """
        Set expiration time for an existing key.

        Args:
            key: Cache key
            ttl: Time-to-live in seconds

        Returns:
            True if expiration was set, False if key doesn't exist
        """
        # Check if key exists
        if not await self.exists(key):
            return False

        # Get current value and re-set with new TTL
        current_value = await self.get(key)
        if current_value is not None:
            return await self.set(key, current_value, ttl)

        return False

    def get_sync_cache_manager(self) -> ExistingCacheManager:
        """
        Get the underlying synchronous cache manager for backward compatibility.

        Returns:
            The wrapped CacheManager instance
        """
        return self._cache_manager
