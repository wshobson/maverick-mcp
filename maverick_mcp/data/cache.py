"""
Cache utilities for Maverick-MCP.
Implements Redis-based caching with fallback to in-memory caching.
Now uses centralized Redis connection pooling for improved performance.
Includes timezone handling, smart invalidation, and performance monitoring.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
import zlib
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, date, datetime
from typing import Any, cast

import msgpack
import pandas as pd
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
CACHE_VERSION = os.getenv("CACHE_VERSION", "v1")

# Cache statistics
CacheStatMap = defaultdict[str, float]
_cache_stats: CacheStatMap = defaultdict(float)
_cache_stats["hits"] = 0.0
_cache_stats["misses"] = 0.0
_cache_stats["sets"] = 0.0
_cache_stats["errors"] = 0.0
_cache_stats["serialization_time"] = 0.0
_cache_stats["deserialization_time"] = 0.0

# In-memory cache as fallback with memory management
_memory_cache: dict[str, dict[str, Any]] = {}
_memory_cache_max_size = 1000  # Will be updated to use config

# Cache metadata for version tracking
_cache_metadata: dict[str, dict[str, Any]] = {}

# Memory monitoring
_cache_memory_stats: dict[str, float] = {
    "memory_cache_bytes": 0.0,
    "redis_connection_count": 0.0,
    "large_object_count": 0.0,
    "compression_savings_bytes": 0.0,
}


def _dataframe_to_payload(df: pd.DataFrame) -> dict[str, Any]:
    """Convert a DataFrame to a JSON-serializable payload."""

    normalized = ensure_timezone_naive(df)
    json_payload = cast(
        str,
        normalized.to_json(orient="split", date_format="iso", default_handler=str),
    )
    payload = json.loads(json_payload)
    payload["index_type"] = (
        "datetime" if isinstance(normalized.index, pd.DatetimeIndex) else "other"
    )
    payload["index_name"] = normalized.index.name
    return payload


def _payload_to_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
    """Reconstruct a DataFrame from a serialized payload."""

    data = payload.get("data", {})
    columns = data.get("columns", [])
    frame = pd.DataFrame(data.get("data", []), columns=columns)
    index_values = data.get("index", [])

    if payload.get("index_type") == "datetime":
        index_values = pd.to_datetime(index_values)
        index = normalize_timezone(pd.DatetimeIndex(index_values))
    else:
        index = index_values

    frame.index = index
    frame.index.name = payload.get("index_name")
    return ensure_timezone_naive(frame)


def _json_default(value: Any) -> Any:
    """JSON serializer for unsupported types."""

    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, set):
        return list(value)
    raise TypeError(f"Unsupported type {type(value)!r} for cache serialization")


def _decode_json_payload(raw_data: str) -> Any:
    """Decode JSON payloads with DataFrame support."""

    payload = json.loads(raw_data)
    if isinstance(payload, dict) and payload.get("__cache_type__") == "dataframe":
        return _payload_to_dataframe(payload)
    if isinstance(payload, dict) and payload.get("__cache_type__") == "dict":
        result: dict[str, Any] = {}
        for key, value in payload.get("data", {}).items():
            if isinstance(value, dict) and value.get("__cache_type__") == "dataframe":
                result[key] = _payload_to_dataframe(value)
            else:
                result[key] = value
        return result
    return payload


def normalize_timezone(index: pd.Index | Sequence[Any]) -> pd.DatetimeIndex:
    """Return a timezone-naive :class:`~pandas.DatetimeIndex` in UTC."""

    dt_index = index if isinstance(index, pd.DatetimeIndex) else pd.DatetimeIndex(index)

    if dt_index.tz is not None:
        dt_index = dt_index.tz_convert("UTC").tz_localize(None)

    return dt_index


def ensure_timezone_naive(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has timezone-naive datetime index.

    Args:
        df: DataFrame with potentially timezone-aware index

    Returns:
        DataFrame with timezone-naive index
    """
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = normalize_timezone(df.index)
    return df


def get_cache_stats() -> dict[str, Any]:
    """Get current cache statistics with memory information.

    Returns:
        Dictionary containing cache performance metrics
    """
    stats: dict[str, float | int] = cast(dict[str, float | int], dict(_cache_stats))

    # Calculate hit rate
    total_requests = stats["hits"] + stats["misses"]
    hit_rate = (stats["hits"] / total_requests * 100) if total_requests > 0 else 0

    stats["hit_rate_percent"] = round(hit_rate, 2)
    stats["total_requests"] = total_requests

    # Memory cache stats
    stats["memory_cache_size"] = len(_memory_cache)
    stats["memory_cache_max_size"] = _memory_cache_max_size

    # Add memory statistics
    stats.update(_cache_memory_stats)

    # Calculate memory cache size in bytes
    memory_size_bytes = 0
    for entry in _memory_cache.values():
        if "data" in entry:
            try:
                if hasattr(entry["data"], "__sizeof__"):
                    memory_size_bytes += entry["data"].__sizeof__()
                elif isinstance(entry["data"], str | bytes):
                    memory_size_bytes += len(entry["data"])
                elif isinstance(entry["data"], pd.DataFrame):
                    memory_size_bytes += entry["data"].memory_usage(deep=True).sum()
            except Exception:
                pass  # Skip if size calculation fails

    stats["memory_cache_bytes"] = memory_size_bytes
    stats["memory_cache_mb"] = memory_size_bytes / (1024**2)

    return stats


def reset_cache_stats() -> None:
    """Reset cache statistics."""
    global _cache_stats
    _cache_stats.clear()
    _cache_stats.update(
        {
            "hits": 0.0,
            "misses": 0.0,
            "sets": 0.0,
            "errors": 0.0,
            "serialization_time": 0.0,
            "deserialization_time": 0.0,
        }
    )


def generate_cache_key(base_key: str, **kwargs) -> str:
    """Generate versioned cache key with consistent hashing.

    Args:
        base_key: Base cache key
        **kwargs: Additional parameters to include in key

    Returns:
        Versioned and hashed cache key
    """
    # Include cache version and sorted parameters
    key_parts = [CACHE_VERSION, base_key]

    # Sort kwargs for consistent key generation
    if kwargs:
        sorted_params = sorted(kwargs.items())
        param_str = ":".join(f"{k}={v}" for k, v in sorted_params)
        key_parts.append(param_str)

    full_key = ":".join(str(part) for part in key_parts)

    # Hash long keys to prevent Redis key length limits
    if len(full_key) > 250:
        key_hash = hashlib.sha256(full_key.encode()).hexdigest()
        return f"{CACHE_VERSION}:hashed:{key_hash}"

    return full_key


def _cleanup_expired_memory_cache():
    """Clean up expired entries from memory cache and enforce size limit with memory tracking."""
    current_time = time.time()
    bytes_freed = 0

    # Remove expired entries
    expired_keys = [
        k
        for k, v in _memory_cache.items()
        if "expiry" in v and v["expiry"] < current_time
    ]
    for k in expired_keys:
        entry = _memory_cache[k]
        if "data" in entry and isinstance(entry["data"], pd.DataFrame):
            bytes_freed += entry["data"].memory_usage(deep=True).sum()
        del _memory_cache[k]

    # Calculate current memory usage
    current_memory_bytes = 0
    for entry in _memory_cache.values():
        if "data" in entry and isinstance(entry["data"], pd.DataFrame):
            current_memory_bytes += entry["data"].memory_usage(deep=True).sum()

    # Enforce memory-based size limit (100MB default)
    memory_limit_bytes = 100 * 1024 * 1024  # 100MB

    # Enforce size limit - remove oldest entries if over limit
    if (
        len(_memory_cache) > _memory_cache_max_size
        or current_memory_bytes > memory_limit_bytes
    ):
        # Sort by expiry time (oldest first)
        sorted_items = sorted(
            _memory_cache.items(), key=lambda x: x[1].get("expiry", float("inf"))
        )

        # Calculate how many to remove
        num_to_remove = max(len(_memory_cache) - _memory_cache_max_size, 0)

        # Remove by memory if over memory limit
        if current_memory_bytes > memory_limit_bytes:
            removed_memory = 0
            for k, v in sorted_items:
                if "data" in v and isinstance(v["data"], pd.DataFrame):
                    entry_size = v["data"].memory_usage(deep=True).sum()
                    removed_memory += entry_size
                    bytes_freed += entry_size
                del _memory_cache[k]
                num_to_remove = max(num_to_remove, 1)

                if removed_memory >= (current_memory_bytes - memory_limit_bytes):
                    break
        else:
            # Remove by count
            for k, v in sorted_items[:num_to_remove]:
                if "data" in v and isinstance(v["data"], pd.DataFrame):
                    bytes_freed += v["data"].memory_usage(deep=True).sum()
                del _memory_cache[k]

        if num_to_remove > 0:
            logger.debug(
                f"Removed {num_to_remove} entries from memory cache "
                f"(freed {bytes_freed / (1024**2):.2f}MB)"
            )

    # Update memory stats
    _cache_memory_stats["memory_cache_bytes"] = current_memory_bytes - bytes_freed


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
            "host": REDIS_HOST,
            "port": REDIS_PORT,
            "db": REDIS_DB,
            "max_connections": settings.db.redis_max_connections,
            "retry_on_timeout": settings.db.redis_retry_on_timeout,
            "socket_timeout": settings.db.redis_socket_timeout,
            "socket_connect_timeout": settings.db.redis_socket_connect_timeout,
            "health_check_interval": 30,  # Check connection health every 30 seconds
        }

        # Only add password if provided
        if REDIS_PASSWORD:
            pool_params["password"] = REDIS_PASSWORD

        # Only add SSL params if SSL is enabled
        if REDIS_SSL:
            pool_params["ssl"] = True
            pool_params["ssl_check_hostname"] = False

        _redis_pool = redis.ConnectionPool(**pool_params)
        logger.debug(
            f"Created Redis connection pool with {settings.db.redis_max_connections} max connections"
        )
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


def _deserialize_cached_data(data: bytes, key: str) -> Any:
    """Deserialize cached data with multiple format support and timezone handling."""
    start_time = time.time()

    try:
        # Try msgpack with zlib compression first (most efficient for DataFrames)
        if data[:2] == b"\x78\x9c":  # zlib magic bytes
            try:
                decompressed = zlib.decompress(data)
                # Try msgpack first
                try:
                    result = msgpack.loads(decompressed, raw=False)
                    # Handle DataFrame reconstruction with timezone normalization
                    if isinstance(result, dict) and result.get("_type") == "dataframe":
                        df = pd.DataFrame.from_dict(result["data"], orient="index")

                        # Restore proper index
                        if result.get("index_data"):
                            if result.get("index_type") == "datetime":
                                df.index = pd.to_datetime(result["index_data"])
                                df.index = normalize_timezone(df.index)
                            else:
                                df.index = result["index_data"]
                        elif result.get("index_type") == "datetime":
                            df.index = pd.to_datetime(df.index)
                            df.index = normalize_timezone(df.index)

                        # Restore column order
                        if result.get("columns"):
                            df = df[result["columns"]]

                        return df
                    return result
                except Exception as e:
                    logger.debug(f"Msgpack decompressed failed for {key}: {e}")
                    try:
                        return _decode_json_payload(decompressed.decode("utf-8"))
                    except Exception as e2:
                        logger.debug(f"JSON decompressed failed for {key}: {e2}")
                        pass
            except Exception:
                pass

        # Try msgpack uncompressed
        try:
            result = msgpack.loads(data, raw=False)
            if isinstance(result, dict) and result.get("_type") == "dataframe":
                df = pd.DataFrame.from_dict(result["data"], orient="index")

                # Restore proper index
                if result.get("index_data"):
                    if result.get("index_type") == "datetime":
                        df.index = pd.to_datetime(result["index_data"])
                        df.index = normalize_timezone(df.index)
                    else:
                        df.index = result["index_data"]
                elif result.get("index_type") == "datetime":
                    df.index = pd.to_datetime(df.index)
                    df.index = normalize_timezone(df.index)

                # Restore column order
                if result.get("columns"):
                    df = df[result["columns"]]

                return df
            return result
        except Exception:
            pass

        # Fall back to JSON
        try:
            decoded = data.decode() if isinstance(data, bytes) else data
            return _decode_json_payload(decoded)
        except Exception:
            pass

    except Exception as e:
        _cache_stats["errors"] += 1
        logger.warning(f"Failed to deserialize cache data for key {key}: {e}")
        return None
    finally:
        _cache_stats["deserialization_time"] += time.time() - start_time

    _cache_stats["errors"] += 1
    logger.warning(f"Failed to deserialize cache data for key {key} - no format worked")
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
                _cache_stats["hits"] += 1
                logger.debug(f"Cache hit for {key} (Redis)")
                result = _deserialize_cached_data(data, key)  # type: ignore[arg-type]
                return result
        except Exception as e:
            _cache_stats["errors"] += 1
            logger.warning(f"Error reading from Redis cache: {e}")

    # Fall back to in-memory cache
    if key in _memory_cache:
        entry = _memory_cache[key]
        if "expiry" not in entry or entry["expiry"] > time.time():
            _cache_stats["hits"] += 1
            logger.debug(f"Cache hit for {key} (memory)")
            return entry["data"]
        else:
            # Clean up expired entry
            del _memory_cache[key]

    _cache_stats["misses"] += 1
    logger.debug(f"Cache miss for {key}")
    return None


def _serialize_data(data: Any, key: str) -> bytes:
    """Serialize data efficiently based on type with optimized formats and memory tracking."""
    start_time = time.time()
    original_size = 0
    compressed_size = 0

    try:
        # Special handling for DataFrames - use msgpack with timezone normalization
        if isinstance(data, pd.DataFrame):
            original_size = data.memory_usage(deep=True).sum()

            # Track large objects
            if original_size > 10 * 1024 * 1024:  # 10MB threshold
                _cache_memory_stats["large_object_count"] += 1
                logger.debug(
                    f"Serializing large DataFrame for {key}: {original_size / (1024**2):.2f}MB"
                )

            # Ensure timezone-naive DataFrame
            df = ensure_timezone_naive(data)

            # Try msgpack first (most efficient for DataFrames)
            try:
                # Convert to msgpack-serializable format with proper index handling
                df_dict = {
                    "_type": "dataframe",
                    "data": df.to_dict("index"),
                    "index_type": (
                        "datetime"
                        if isinstance(df.index, pd.DatetimeIndex)
                        else "other"
                    ),
                    "columns": list(df.columns),
                    "index_data": [str(idx) for idx in df.index],
                }
                msgpack_data = cast(bytes, msgpack.packb(df_dict))
                compressed = zlib.compress(msgpack_data, level=1)
                compressed_size = len(compressed)

                # Track compression savings
                if original_size > compressed_size:
                    _cache_memory_stats["compression_savings_bytes"] += (
                        original_size - compressed_size
                    )

                return compressed
            except Exception as e:
                logger.debug(f"Msgpack DataFrame serialization failed for {key}: {e}")
                json_payload = {
                    "__cache_type__": "dataframe",
                    "data": _dataframe_to_payload(df),
                }
                compressed = zlib.compress(
                    json.dumps(json_payload).encode("utf-8"), level=1
                )
                compressed_size = len(compressed)

                if original_size > compressed_size:
                    _cache_memory_stats["compression_savings_bytes"] += (
                        original_size - compressed_size
                    )

                return compressed

        # For dictionaries with DataFrames (like backtest results)
        if isinstance(data, dict) and any(
            isinstance(v, pd.DataFrame) for v in data.values()
        ):
            # Ensure all DataFrames are timezone-naive
            processed_data = {}
            for k, v in data.items():
                if isinstance(v, pd.DataFrame):
                    processed_data[k] = ensure_timezone_naive(v)
                else:
                    processed_data[k] = v

            try:
                # Try msgpack for mixed dict with DataFrames
                serializable_data = {}
                for k, v in processed_data.items():
                    if isinstance(v, pd.DataFrame):
                        serializable_data[k] = {
                            "_type": "dataframe",
                            "data": v.to_dict("index"),
                            "index_type": (
                                "datetime"
                                if isinstance(v.index, pd.DatetimeIndex)
                                else "other"
                            ),
                        }
                    else:
                        serializable_data[k] = v

                msgpack_data = cast(bytes, msgpack.packb(serializable_data))
                compressed = zlib.compress(msgpack_data, level=1)
                return compressed
            except Exception:
                payload = {
                    "__cache_type__": "dict",
                    "data": {
                        key: (
                            {
                                "__cache_type__": "dataframe",
                                "data": _dataframe_to_payload(value),
                            }
                            if isinstance(value, pd.DataFrame)
                            else value
                        )
                        for key, value in processed_data.items()
                    },
                }
                compressed = zlib.compress(
                    json.dumps(payload, default=_json_default).encode("utf-8"),
                    level=1,
                )
                return compressed

        # For simple data types, try msgpack first (more efficient than JSON)
        if isinstance(data, dict | list | str | int | float | bool | type(None)):
            try:
                return cast(bytes, msgpack.packb(data))
            except Exception:
                # Fall back to JSON
                return json.dumps(data, default=_json_default).encode("utf-8")

        raise TypeError(f"Unsupported cache data type {type(data)!r} for key {key}")

    except TypeError as exc:
        _cache_stats["errors"] += 1
        logger.warning(f"Unsupported data type for cache key {key}: {exc}")
        raise
    except Exception as e:
        _cache_stats["errors"] += 1
        logger.warning(f"Failed to serialize data for key {key}: {e}")
        # Fall back to JSON string representation
        try:
            return json.dumps(str(data)).encode("utf-8")
        except Exception:
            return b"null"
    finally:
        _cache_stats["serialization_time"] += time.time() - start_time


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

    resolved_ttl = CACHE_TTL_SECONDS if ttl is None else ttl

    # Serialize data efficiently
    try:
        serialized_data = _serialize_data(data, key)
    except TypeError as exc:
        logger.warning(f"Skipping cache for {key}: {exc}")
        return False

    # Store cache metadata
    _cache_metadata[key] = {
        "created_at": datetime.now(UTC).isoformat(),
        "ttl": resolved_ttl,
        "size_bytes": len(serialized_data),
        "version": CACHE_VERSION,
    }

    success = False

    # Try Redis first
    redis_client = get_redis_client()
    if redis_client:
        try:
            redis_client.setex(key, resolved_ttl, serialized_data)
            logger.debug(f"Saved to Redis cache: {key}")
            success = True
        except Exception as e:
            _cache_stats["errors"] += 1
            logger.warning(f"Error saving to Redis cache: {e}")

    if not success:
        # Fall back to in-memory cache
        _memory_cache[key] = {"data": data, "expiry": time.time() + resolved_ttl}
        logger.debug(f"Saved to memory cache: {key}")
        success = True

        # Clean up memory cache if needed
        if len(_memory_cache) > _memory_cache_max_size:
            _cleanup_expired_memory_cache()

    if success:
        _cache_stats["sets"] += 1

    return success


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
                keys = cast(list[bytes], redis_client.keys(pattern))
                if keys:
                    delete_result = cast(int, redis_client.delete(*keys))
                    count += delete_result
            else:
                flush_result = cast(int, redis_client.flushdb())
                count += flush_result
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
                        decoded_value: Any
                        if isinstance(value, bytes):
                            decoded_value = value.decode()
                        else:
                            decoded_value = value

                        if isinstance(decoded_value, str):
                            try:
                                # Try to decode JSON if it's stored as JSON
                                results[key] = json.loads(decoded_value)
                                continue
                            except json.JSONDecodeError:
                                pass

                        # If not JSON or decoding fails, store as-is
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
                keys = cast(list[bytes], client.keys(pattern))
                if keys:
                    delete_result = cast(int, client.delete(*keys))
                    count += delete_result
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
                deleted_result = client.delete(*keys)
                deleted_count = cast(int, deleted_result)
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
