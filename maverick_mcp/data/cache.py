"""
Cache utilities for Maverick-MCP.
Implements Redis-based caching with fallback to in-memory caching.
Now uses centralized Redis connection pooling for improved performance.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any

import redis
from dotenv import load_dotenv

from maverick_mcp.config.settings import get_settings

# Import the new performance module for Redis connection pooling

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("maverick_mcp.cache")

settings = get_settings()

# Redis configuration (kept for backwards compatibility)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_SSL = os.getenv("REDIS_SSL", "False").lower() == "true"

# Cache configuration
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "True").lower() == "true"
CACHE_TTL_SECONDS = settings.performance.cache_ttl_seconds

# In-memory cache as fallback
_memory_cache: dict[str, dict[str, Any]] = {}
_memory_cache_max_size = 1000  # Will be updated to use config


def _cleanup_expired_memory_cache():
    """Clean up expired entries from memory cache and enforce size limit."""
    current_time = time.time()

    # Remove expired entries
    expired_keys = [
        k
        for k, v in _memory_cache.items()
        if "expiry" in v and v["expiry"] < current_time
    ]
    for k in expired_keys:
        del _memory_cache[k]

    # Enforce size limit - remove oldest entries if over limit
    if len(_memory_cache) > _memory_cache_max_size:
        # Sort by expiry time (oldest first)
        sorted_items = sorted(
            _memory_cache.items(), key=lambda x: x[1].get("expiry", float("inf"))
        )
        # Remove oldest entries
        num_to_remove = len(_memory_cache) - _memory_cache_max_size
        for k, _ in sorted_items[:num_to_remove]:
            del _memory_cache[k]
        logger.debug(
            f"Removed {num_to_remove} entries from memory cache to enforce size limit"
        )


# Global Redis connection pool - created once and reused
_redis_pool: redis.ConnectionPool | None = None


def _get_or_create_redis_pool() -> redis.ConnectionPool | None:
    """Create or return existing Redis connection pool."""
    global _redis_pool
    
    if _redis_pool is not None:
        return _redis_pool
        
    try:
        # Build connection pool parameters
        pool_params = {
            'host': REDIS_HOST,
            'port': REDIS_PORT,
            'db': REDIS_DB,
            'max_connections': settings.db.redis_max_connections,
            'retry_on_timeout': settings.db.redis_retry_on_timeout,
            'socket_timeout': settings.db.redis_socket_timeout,
            'socket_connect_timeout': settings.db.redis_socket_connect_timeout,
            'health_check_interval': 30,  # Check connection health every 30 seconds
        }
        
        # Only add password if provided
        if REDIS_PASSWORD:
            pool_params['password'] = REDIS_PASSWORD
            
        # Only add SSL params if SSL is enabled
        if REDIS_SSL:
            pool_params['ssl'] = True
            pool_params['ssl_check_hostname'] = False
            
        _redis_pool = redis.ConnectionPool(**pool_params)
        logger.debug(f"Created Redis connection pool with {settings.db.redis_max_connections} max connections")
        return _redis_pool
    except Exception as e:
        logger.warning(f"Failed to create Redis connection pool: {e}")
        return None


def get_redis_client() -> redis.Redis | None:
    """
    Get a Redis client using the centralized connection pool.
    
    This function uses a singleton connection pool to avoid pool exhaustion
    and provides robust error handling with graceful fallback.
    """
    if not CACHE_ENABLED:
        return None

    try:
        # Get or create the connection pool
        pool = _get_or_create_redis_pool()
        if pool is None:
            return None
            
        # Create client using the shared pool
        client = redis.Redis(
            connection_pool=pool,
            decode_responses=False,
        )
        
        # Test connection with a timeout to avoid hanging
        client.ping()
        return client  # type: ignore[no-any-return]
        
    except redis.ConnectionError as e:
        logger.warning(f"Redis connection failed: {e}. Using in-memory cache.")
        return None
    except redis.TimeoutError as e:
        logger.warning(f"Redis connection timeout: {e}. Using in-memory cache.")
        return None
    except Exception as e:
        # Handle the IndexError: pop from empty list and other unexpected errors
        logger.warning(f"Redis client error: {e}. Using in-memory cache.")
        # Reset the pool if we encounter unexpected errors
        global _redis_pool
        _redis_pool = None
        return None


def get_from_cache(key: str) -> Any | None:
    """
    Get data from the cache.

    Args:
        key: Cache key

    Returns:
        Cached data or None if not found
    """
    if not CACHE_ENABLED:
        return None

    # Try Redis first
    redis_client = get_redis_client()
    if redis_client:
        try:
            data = redis_client.get(key)
            if data:
                logger.debug(f"Cache hit for {key} (Redis)")
                return json.loads(data)  # type: ignore[arg-type]
        except Exception as e:
            logger.warning(f"Error reading from Redis cache: {e}")

    # Fall back to in-memory cache
    if key in _memory_cache:
        entry = _memory_cache[key]
        if "expiry" not in entry or entry["expiry"] > time.time():
            logger.debug(f"Cache hit for {key} (memory)")
            return entry["data"]
        else:
            # Clean up expired entry
            del _memory_cache[key]

    logger.debug(f"Cache miss for {key}")
    return None


def save_to_cache(key: str, data: Any, ttl: int | None = None) -> bool:
    """
    Save data to the cache.

    Args:
        key: Cache key
        data: Data to cache
        ttl: Time-to-live in seconds (default: CACHE_TTL_SECONDS)

    Returns:
        True if saved successfully, False otherwise
    """
    if not CACHE_ENABLED:
        return False

    if ttl is None:
        ttl = CACHE_TTL_SECONDS

    # Convert data to JSON
    json_data = json.dumps(data)

    # Try Redis first
    redis_client = get_redis_client()
    if redis_client:
        try:
            redis_client.setex(key, ttl, json_data)
            logger.debug(f"Saved to Redis cache: {key}")
            return True
        except Exception as e:
            logger.warning(f"Error saving to Redis cache: {e}")

    # Fall back to in-memory cache
    _memory_cache[key] = {"data": data, "expiry": time.time() + ttl}
    logger.debug(f"Saved to memory cache: {key}")

    # Clean up memory cache if needed
    if len(_memory_cache) > _memory_cache_max_size:
        _cleanup_expired_memory_cache()

    return True


def cleanup_redis_pool() -> None:
    """Cleanup Redis connection pool."""
    global _redis_pool
    if _redis_pool:
        try:
            _redis_pool.disconnect()
            logger.debug("Redis connection pool disconnected")
        except Exception as e:
            logger.warning(f"Error disconnecting Redis pool: {e}")
        finally:
            _redis_pool = None


def clear_cache(pattern: str | None = None) -> int:
    """
    Clear cache entries matching the pattern.

    Args:
        pattern: Pattern to match keys (e.g., "stock:*")
                If None, clears all cache

    Returns:
        Number of entries cleared
    """
    count = 0

    # Clear from Redis
    redis_client = get_redis_client()
    if redis_client:
        try:
            if pattern:
                keys = redis_client.keys(pattern)
                if keys:
                    count += redis_client.delete(*keys)  # type: ignore[operator,misc]
            else:
                count += redis_client.flushdb()  # type: ignore[operator]
            logger.info(f"Cleared {count} entries from Redis cache")
        except Exception as e:
            logger.warning(f"Error clearing Redis cache: {e}")

    # Clear from memory cache
    if pattern:
        # Simple pattern matching for memory cache (only supports prefix*)
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            memory_keys = [k for k in _memory_cache.keys() if k.startswith(prefix)]
        else:
            memory_keys = [k for k in _memory_cache.keys() if k == pattern]

        for k in memory_keys:
            del _memory_cache[k]
        count += len(memory_keys)
    else:
        count += len(_memory_cache)
        _memory_cache.clear()

    logger.info(f"Cleared {count} total cache entries")
    return count


class CacheManager:
    """
    Enhanced cache manager with async support and additional methods.

    This manager now integrates with the centralized Redis connection pool
    for improved performance and resource management.
    """

    def __init__(self):
        """Initialize the cache manager."""
        self._redis_client = None
        self._initialized = False
        self._use_performance_redis = True  # Flag to use new performance module

    def _ensure_client(self) -> redis.Redis | None:
        """Ensure Redis client is initialized with connection pooling."""
        if not self._initialized:
            # Always use the new robust connection pooling approach
            self._redis_client = get_redis_client()
            self._initialized = True
        return self._redis_client

    async def get(self, key: str) -> Any | None:
        """Async wrapper for get_from_cache."""
        return await asyncio.get_event_loop().run_in_executor(None, get_from_cache, key)

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Async wrapper for save_to_cache."""
        return await asyncio.get_event_loop().run_in_executor(
            None, save_to_cache, key, value, ttl
        )

    async def set_with_ttl(self, key: str, value: str, ttl: int) -> bool:
        """Set a value with specific TTL."""
        if not CACHE_ENABLED:
            return False

        client = self._ensure_client()
        if client:
            try:
                client.setex(key, ttl, value)
                return True
            except Exception as e:
                logger.warning(f"Error setting value with TTL: {e}")

        # Fallback to memory cache
        _memory_cache[key] = {"data": value, "expiry": time.time() + ttl}
        return True

    async def set_many_with_ttl(self, items: list[tuple[str, str, int]]) -> bool:
        """Set multiple values with TTL in a batch."""
        if not CACHE_ENABLED:
            return False

        client = self._ensure_client()
        if client:
            try:
                pipe = client.pipeline()
                for key, value, ttl in items:
                    pipe.setex(key, ttl, value)
                pipe.execute()
                return True
            except Exception as e:
                logger.warning(f"Error in batch set with TTL: {e}")

        # Fallback to memory cache
        for key, value, ttl in items:
            _memory_cache[key] = {"data": value, "expiry": time.time() + ttl}
        return True

    async def get_many(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values at once using pipeline for better performance."""
        results: dict[str, Any] = {}

        if not CACHE_ENABLED:
            return results

        client = self._ensure_client()
        if client:
            try:
                # Use pipeline for better performance with multiple operations
                pipe = client.pipeline(transaction=False)
                for key in keys:
                    pipe.get(key)
                values = pipe.execute()

                for key, value in zip(keys, values, strict=False):  # type: ignore[arg-type]
                    if value:
                        try:
                            # Try to decode JSON if it's stored as JSON
                            decoded_value = (
                                value.decode() if isinstance(value, bytes) else value
                            )
                            results[key] = json.loads(decoded_value)
                        except (json.JSONDecodeError, AttributeError):
                            # If not JSON, store as-is
                            results[key] = decoded_value
            except Exception as e:
                logger.warning(f"Error in batch get: {e}")

        # Fallback to memory cache for missing keys
        for key in keys:
            if key not in results and key in _memory_cache:
                entry = _memory_cache[key]
                if "expiry" not in entry or entry["expiry"] > time.time():
                    results[key] = entry["data"]

        return results

    async def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        if not CACHE_ENABLED:
            return False

        deleted = False
        client = self._ensure_client()
        if client:
            try:
                deleted = bool(client.delete(key))
            except Exception as e:
                logger.warning(f"Error deleting key: {e}")

        # Also delete from memory cache
        if key in _memory_cache:
            del _memory_cache[key]
            deleted = True

        return deleted

    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching a pattern."""
        count = 0

        if not CACHE_ENABLED:
            return count

        client = self._ensure_client()
        if client:
            try:
                keys = client.keys(pattern)
                if keys:
                    count = client.delete(*keys)  # type: ignore[assignment,misc]
            except Exception as e:
                logger.warning(f"Error deleting pattern: {e}")

        # Also delete from memory cache
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            memory_keys = [k for k in _memory_cache.keys() if k.startswith(prefix)]
            for k in memory_keys:
                del _memory_cache[k]
                count += 1

        return count

    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        if not CACHE_ENABLED:
            return False

        client = self._ensure_client()
        if client:
            try:
                return bool(client.exists(key))
            except Exception as e:
                logger.warning(f"Error checking key existence: {e}")

        # Fallback to memory cache
        if key in _memory_cache:
            entry = _memory_cache[key]
            return "expiry" not in entry or entry["expiry"] > time.time()

        return False

    async def count_keys(self, pattern: str) -> int:
        """Count keys matching a pattern."""
        if not CACHE_ENABLED:
            return 0

        count = 0
        client = self._ensure_client()
        if client:
            try:
                cursor = 0
                while True:
                    cursor, keys = client.scan(cursor, match=pattern, count=1000)  # type: ignore[misc]
                    count += len(keys)
                    if cursor == 0:
                        break
            except Exception as e:
                logger.warning(f"Error counting keys: {e}")

        # Add memory cache count
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            count += sum(1 for k in _memory_cache.keys() if k.startswith(prefix))

        return count

    async def batch_save(self, items: list[tuple[str, Any, int | None]]) -> int:
        """
        Save multiple items to cache using pipeline for better performance.

        Args:
            items: List of tuples (key, data, ttl)

        Returns:
            Number of items successfully saved
        """
        if not CACHE_ENABLED:
            return 0

        saved_count = 0
        client = self._ensure_client()

        if client:
            try:
                pipe = client.pipeline(transaction=False)

                for key, data, ttl in items:
                    if ttl is None:
                        ttl = CACHE_TTL_SECONDS

                    # Convert data to JSON
                    json_data = json.dumps(data)
                    pipe.setex(key, ttl, json_data)

                results = pipe.execute()
                saved_count = sum(1 for r in results if r)
                logger.debug(f"Batch saved {saved_count} items to Redis cache")
            except Exception as e:
                logger.warning(f"Error in batch save to Redis: {e}")

        # Fallback to memory cache for failed items
        if saved_count < len(items):
            for key, data, ttl in items:
                if ttl is None:
                    ttl = CACHE_TTL_SECONDS
                _memory_cache[key] = {"data": data, "expiry": time.time() + ttl}
                saved_count += 1

        return saved_count

    async def batch_delete(self, keys: list[str]) -> int:
        """
        Delete multiple keys from cache using pipeline for better performance.

        Args:
            keys: List of keys to delete

        Returns:
            Number of keys deleted
        """
        if not CACHE_ENABLED:
            return 0

        deleted_count = 0
        client = self._ensure_client()

        if client and keys:
            try:
                # Use single delete command for multiple keys
                deleted_count = client.delete(*keys)
                logger.debug(f"Batch deleted {deleted_count} keys from Redis cache")
            except Exception as e:
                logger.warning(f"Error in batch delete from Redis: {e}")

        # Also delete from memory cache
        for key in keys:
            if key in _memory_cache:
                del _memory_cache[key]
                deleted_count += 1

        return deleted_count

    async def batch_exists(self, keys: list[str]) -> dict[str, bool]:
        """
        Check existence of multiple keys using pipeline for better performance.

        Args:
            keys: List of keys to check

        Returns:
            Dictionary mapping key to existence boolean
        """
        results: dict[str, bool] = {}

        if not CACHE_ENABLED:
            return dict.fromkeys(keys, False)

        client = self._ensure_client()

        if client:
            try:
                pipe = client.pipeline(transaction=False)
                for key in keys:
                    pipe.exists(key)

                existence_results = pipe.execute()
                for key, exists in zip(keys, existence_results, strict=False):
                    results[key] = bool(exists)
            except Exception as e:
                logger.warning(f"Error in batch exists check: {e}")

        # Check memory cache for missing keys
        for key in keys:
            if key not in results and key in _memory_cache:
                entry = _memory_cache[key]
                results[key] = "expiry" not in entry or entry["expiry"] > time.time()
            elif key not in results:
                results[key] = False

        return results

    async def batch_get_or_set(
        self, items: list[tuple[str, Any, int | None]]
    ) -> dict[str, Any]:
        """
        Get multiple values, setting missing ones atomically using pipeline.

        Args:
            items: List of tuples (key, default_value, ttl)

        Returns:
            Dictionary of key-value pairs
        """
        if not CACHE_ENABLED:
            return {key: default for key, default, _ in items}

        results: dict[str, Any] = {}
        keys = [item[0] for item in items]

        # First, try to get all values
        existing = await self.get_many(keys)

        # Identify missing keys
        missing_items = [item for item in items if item[0] not in existing]

        # Set missing values if any
        if missing_items:
            await self.batch_save(missing_items)

            # Add default values to results
            for key, default_value, _ in missing_items:
                results[key] = default_value

        # Add existing values to results
        results.update(existing)

        return results
