"""
Cache manager interface.

This module defines the abstract interface for caching operations,
enabling different caching implementations (Redis, in-memory, etc.)
to be used interchangeably throughout the application.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ICacheManager(Protocol):
    """
    Interface for cache management operations.

    This interface abstracts caching operations to enable different
    implementations (Redis, in-memory, etc.) to be used interchangeably.
    All methods should be async-compatible to support non-blocking operations.
    """

    async def get(self, key: str) -> Any:
        """
        Get data from cache.

        Args:
            key: Cache key to retrieve

        Returns:
            Cached data or None if not found or expired
        """
        ...

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """
        Store data in cache.

        Args:
            key: Cache key
            value: Data to cache (must be JSON serializable)
            ttl: Time-to-live in seconds (None for default TTL)

        Returns:
            True if successfully cached, False otherwise
        """
        ...

    async def delete(self, key: str) -> bool:
        """
        Delete a key from cache.

        Args:
            key: Cache key to delete

        Returns:
            True if key was deleted, False if key didn't exist
        """
        ...

    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in cache.

        Args:
            key: Cache key to check

        Returns:
            True if key exists and hasn't expired, False otherwise
        """
        ...

    async def clear(self, pattern: str | None = None) -> int:
        """
        Clear cache entries.

        Args:
            pattern: Pattern to match keys (e.g., "stock:*")
                    If None, clears all cache entries

        Returns:
            Number of entries cleared
        """
        ...

    async def get_many(self, keys: list[str]) -> dict[str, Any]:
        """
        Get multiple values at once for better performance.

        Args:
            keys: List of cache keys to retrieve

        Returns:
            Dictionary mapping keys to their cached values
            (missing keys will not be in the result)
        """
        ...

    async def set_many(self, items: list[tuple[str, Any, int | None]]) -> int:
        """
        Set multiple values at once for better performance.

        Args:
            items: List of tuples (key, value, ttl)

        Returns:
            Number of items successfully cached
        """
        ...

    async def delete_many(self, keys: list[str]) -> int:
        """
        Delete multiple keys for better performance.

        Args:
            keys: List of keys to delete

        Returns:
            Number of keys successfully deleted
        """
        ...

    async def exists_many(self, keys: list[str]) -> dict[str, bool]:
        """
        Check existence of multiple keys for better performance.

        Args:
            keys: List of keys to check

        Returns:
            Dictionary mapping keys to their existence status
        """
        ...

    async def count_keys(self, pattern: str) -> int:
        """
        Count keys matching a pattern.

        Args:
            pattern: Pattern to match (e.g., "stock:*")

        Returns:
            Number of matching keys
        """
        ...

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
        ...

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
        ...

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
        ...

    async def get_ttl(self, key: str) -> int | None:
        """
        Get the remaining time-to-live for a key.

        Args:
            key: Cache key

        Returns:
            Remaining TTL in seconds, None if key doesn't exist or has no TTL
        """
        ...

    async def expire(self, key: str, ttl: int) -> bool:
        """
        Set expiration time for an existing key.

        Args:
            key: Cache key
            ttl: Time-to-live in seconds

        Returns:
            True if expiration was set, False if key doesn't exist
        """
        ...


class CacheConfig:
    """
    Configuration class for cache implementations.

    This class encapsulates cache-related configuration parameters
    to reduce coupling between cache implementations and configuration sources.
    """

    def __init__(
        self,
        enabled: bool = True,
        default_ttl: int = 3600,
        max_memory_size: int = 1000,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: str | None = None,
        redis_ssl: bool = False,
        connection_pool_size: int = 20,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5,
    ):
        """
        Initialize cache configuration.

        Args:
            enabled: Whether caching is enabled
            default_ttl: Default time-to-live in seconds
            max_memory_size: Maximum in-memory cache size
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            redis_password: Redis password (if required)
            redis_ssl: Whether to use SSL for Redis connection
            connection_pool_size: Redis connection pool size
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Socket connection timeout in seconds
        """
        self.enabled = enabled
        self.default_ttl = default_ttl
        self.max_memory_size = max_memory_size
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.redis_password = redis_password
        self.redis_ssl = redis_ssl
        self.connection_pool_size = connection_pool_size
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout

    def get_redis_url(self) -> str:
        """
        Get Redis connection URL.

        Returns:
            Redis connection URL string
        """
        scheme = "rediss" if self.redis_ssl else "redis"
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"{scheme}://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"
