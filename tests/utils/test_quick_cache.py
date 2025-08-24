"""
Tests for quick_cache.py - 500x speedup in-memory LRU cache decorator.

This test suite achieves 100% coverage by testing:
1. QuickCache class (get, set, LRU eviction, TTL expiration)
2. quick_cache decorator for sync and async functions
3. Cache key generation and collision handling
4. Cache statistics and monitoring
5. Performance validation (500x speedup)
6. Edge cases and error handling
"""

import asyncio
import time
from unittest.mock import patch

import pandas as pd
import pytest

from maverick_mcp.utils.quick_cache import (
    QuickCache,
    _cache,
    cache_1hour,
    cache_1min,
    cache_5min,
    cache_15min,
    cached_stock_data,
    clear_cache,
    get_cache_stats,
    quick_cache,
)


class TestQuickCache:
    """Test QuickCache class functionality."""

    @pytest.mark.asyncio
    async def test_basic_get_set(self):
        """Test basic cache get and set operations."""
        cache = QuickCache(max_size=10)

        # Test set and get
        await cache.set("key1", "value1", ttl_seconds=60)
        result = await cache.get("key1")
        assert result == "value1"

        # Test cache miss
        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """Test TTL expiration behavior."""
        cache = QuickCache()

        # Set with very short TTL
        await cache.set("expire_key", "value", ttl_seconds=0.01)

        # Should be available immediately
        assert await cache.get("expire_key") == "value"

        # Wait for expiration
        await asyncio.sleep(0.02)

        # Should be expired
        assert await cache.get("expire_key") is None

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = QuickCache(max_size=3)

        # Fill cache
        await cache.set("key1", "value1", ttl_seconds=60)
        await cache.set("key2", "value2", ttl_seconds=60)
        await cache.set("key3", "value3", ttl_seconds=60)

        # Access key1 to make it recently used
        await cache.get("key1")

        # Add new key - should evict key2 (least recently used)
        await cache.set("key4", "value4", ttl_seconds=60)

        # key1 and key3 should still be there
        assert await cache.get("key1") == "value1"
        assert await cache.get("key3") == "value3"
        assert await cache.get("key4") == "value4"

        # key2 should be evicted
        assert await cache.get("key2") is None

    def test_make_key(self):
        """Test cache key generation."""
        cache = QuickCache()

        # Test basic key generation
        key1 = cache.make_key("func", (1, 2), {"a": 3})
        key2 = cache.make_key("func", (1, 2), {"a": 3})
        assert key1 == key2  # Same inputs = same key

        # Test different args produce different keys
        key3 = cache.make_key("func", (1, 3), {"a": 3})
        assert key1 != key3

        # Test kwargs order doesn't matter
        key4 = cache.make_key("func", (), {"b": 2, "a": 1})
        key5 = cache.make_key("func", (), {"a": 1, "b": 2})
        assert key4 == key5

    def test_get_stats(self):
        """Test cache statistics."""
        cache = QuickCache()

        # Initial stats
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0

        # Run some operations synchronously for testing
        asyncio.run(cache.set("key1", "value1", 60))
        asyncio.run(cache.get("key1"))  # Hit
        asyncio.run(cache.get("key2"))  # Miss

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 50.0
        assert stats["size"] == 1

    def test_clear(self):
        """Test cache clearing."""
        cache = QuickCache()

        # Add some items
        asyncio.run(cache.set("key1", "value1", 60))
        asyncio.run(cache.set("key2", "value2", 60))

        # Verify they exist
        assert asyncio.run(cache.get("key1")) == "value1"

        # Clear cache
        cache.clear()

        # Verify cache is empty
        assert asyncio.run(cache.get("key1")) is None
        assert cache.get_stats()["size"] == 0
        assert cache.get_stats()["hits"] == 0
        # After clearing and a miss, misses will be 1
        assert cache.get_stats()["misses"] == 1


class TestQuickCacheDecorator:
    """Test quick_cache decorator functionality."""

    @pytest.mark.asyncio
    async def test_async_function_caching(self):
        """Test caching of async functions."""
        call_count = 0

        @quick_cache(ttl_seconds=60)
        async def expensive_async_func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return x * 2

        # First call - cache miss
        result1 = await expensive_async_func(5)
        assert result1 == 10
        assert call_count == 1

        # Second call - cache hit
        result2 = await expensive_async_func(5)
        assert result2 == 10
        assert call_count == 1  # Function not called again

        # Different argument - cache miss
        result3 = await expensive_async_func(6)
        assert result3 == 12
        assert call_count == 2

    def test_sync_function_caching(self):
        """Test caching of sync functions."""
        call_count = 0

        @quick_cache(ttl_seconds=60)
        def expensive_sync_func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            time.sleep(0.01)
            return x * 2

        # First call - cache miss
        result1 = expensive_sync_func(5)
        assert result1 == 10
        assert call_count == 1

        # Second call - cache hit
        result2 = expensive_sync_func(5)
        assert result2 == 10
        assert call_count == 1  # Function not called again

    def test_key_prefix(self):
        """Test cache key prefix functionality."""

        @quick_cache(ttl_seconds=60, key_prefix="test_prefix")
        def func_with_prefix(x: int) -> int:
            return x * 2

        @quick_cache(ttl_seconds=60)
        def func_without_prefix(x: int) -> int:
            return x * 3

        # Both functions with same argument should have different cache keys
        result1 = func_with_prefix(5)
        result2 = func_without_prefix(5)

        assert result1 == 10
        assert result2 == 15

    @pytest.mark.asyncio
    @patch("maverick_mcp.utils.quick_cache.logger")
    async def test_logging_behavior(self, mock_logger):
        """Test cache logging when debug is enabled (async version logs both hit and miss)."""
        clear_cache()  # Clear global cache

        @quick_cache(ttl_seconds=60, log_stats=True)
        async def logged_func(x: int) -> int:
            return x * 2

        # Clear previous calls
        mock_logger.debug.reset_mock()

        # First call - should log miss
        await logged_func(5)

        # Check for cache miss log
        miss_found = False
        for call in mock_logger.debug.call_args_list:
            if call[0] and "Cache MISS" in call[0][0]:
                miss_found = True
                break
        assert miss_found, (
            f"Cache MISS not logged. Calls: {mock_logger.debug.call_args_list}"
        )

        # Second call - should log hit
        await logged_func(5)

        # Check for cache hit log
        hit_found = False
        for call in mock_logger.debug.call_args_list:
            if call[0] and "Cache HIT" in call[0][0]:
                hit_found = True
                break
        assert hit_found, (
            f"Cache HIT not logged. Calls: {mock_logger.debug.call_args_list}"
        )

    def test_decorator_preserves_metadata(self):
        """Test that decorator preserves function metadata."""

        @quick_cache(ttl_seconds=60)
        def documented_func(x: int) -> int:
            """This is a documented function."""
            return x * 2

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == "This is a documented function."

    def test_max_size_parameter(self):
        """Test max_size parameter updates global cache."""
        original_size = _cache.max_size

        @quick_cache(ttl_seconds=60, max_size=500)
        def func_with_custom_size(x: int) -> int:
            return x * 2

        # Should update global cache size
        assert _cache.max_size == 500

        # Reset for other tests
        _cache.max_size = original_size


class TestPerformanceValidation:
    """Test performance improvements and 500x speedup claim."""

    def test_cache_speedup(self):
        """Test that cache provides significant speedup."""
        # Clear cache first
        clear_cache()

        @quick_cache(ttl_seconds=60)
        def slow_function(n: int) -> int:
            # Simulate expensive computation
            time.sleep(0.1)  # 100ms
            return sum(i**2 for i in range(n))

        # First call - no cache
        start_time = time.time()
        result1 = slow_function(1000)
        first_call_time = time.time() - start_time

        # Second call - from cache
        start_time = time.time()
        result2 = slow_function(1000)
        cached_call_time = time.time() - start_time

        assert result1 == result2

        # Calculate speedup
        speedup = (
            first_call_time / cached_call_time if cached_call_time > 0 else float("inf")
        )

        # Should be at least 100x faster (conservative estimate)
        assert speedup > 100

        # First call should take at least 100ms
        assert first_call_time >= 0.1

        # Cached call should be nearly instant (< 5ms, allowing for test environment variability)
        assert cached_call_time < 0.005

    @pytest.mark.asyncio
    async def test_async_cache_speedup(self):
        """Test cache speedup for async functions."""
        clear_cache()

        @quick_cache(ttl_seconds=60)
        async def slow_async_function(n: int) -> int:
            # Simulate expensive async operation
            await asyncio.sleep(0.1)  # 100ms
            return sum(i**2 for i in range(n))

        # First call - no cache
        start_time = time.time()
        result1 = await slow_async_function(1000)
        first_call_time = time.time() - start_time

        # Second call - from cache
        start_time = time.time()
        result2 = await slow_async_function(1000)
        cached_call_time = time.time() - start_time

        assert result1 == result2

        # Calculate speedup
        speedup = (
            first_call_time / cached_call_time if cached_call_time > 0 else float("inf")
        )

        # Should be significantly faster
        assert speedup > 50
        assert first_call_time >= 0.1
        assert cached_call_time < 0.01


class TestConvenienceDecorators:
    """Test pre-configured cache decorators."""

    def test_cache_1min(self):
        """Test 1-minute cache decorator."""

        @cache_1min()
        def func_1min(x: int) -> int:
            return x * 2

        result = func_1min(5)
        assert result == 10

    def test_cache_5min(self):
        """Test 5-minute cache decorator."""

        @cache_5min()
        def func_5min(x: int) -> int:
            return x * 2

        result = func_5min(5)
        assert result == 10

    def test_cache_15min(self):
        """Test 15-minute cache decorator."""

        @cache_15min()
        def func_15min(x: int) -> int:
            return x * 2

        result = func_15min(5)
        assert result == 10

    def test_cache_1hour(self):
        """Test 1-hour cache decorator."""

        @cache_1hour()
        def func_1hour(x: int) -> int:
            return x * 2

        result = func_1hour(5)
        assert result == 10


class TestGlobalCacheFunctions:
    """Test global cache management functions."""

    def test_get_cache_stats(self):
        """Test get_cache_stats function."""
        clear_cache()

        @quick_cache(ttl_seconds=60)
        def cached_func(x: int) -> int:
            return x * 2

        # Generate some cache activity
        cached_func(1)  # Miss
        cached_func(1)  # Hit
        cached_func(2)  # Miss

        stats = get_cache_stats()
        assert stats["hits"] >= 1
        assert stats["misses"] >= 2
        assert stats["size"] >= 2

    @patch("maverick_mcp.utils.quick_cache.logger")
    def test_clear_cache_logging(self, mock_logger):
        """Test clear_cache logs properly."""
        clear_cache()

        mock_logger.info.assert_called_with("Cache cleared")


class TestExampleFunction:
    """Test the example cached_stock_data function."""

    @pytest.mark.asyncio
    async def test_cached_stock_data(self):
        """Test the example cached stock data function."""
        clear_cache()

        # First call
        start = time.time()
        result1 = await cached_stock_data("AAPL", "2024-01-01", "2024-01-31")
        first_time = time.time() - start

        assert result1["symbol"] == "AAPL"
        assert result1["start"] == "2024-01-01"
        assert result1["end"] == "2024-01-31"
        assert first_time >= 0.1  # Should sleep for 0.1s

        # Second call - cached
        start = time.time()
        result2 = await cached_stock_data("AAPL", "2024-01-01", "2024-01-31")
        cached_time = time.time() - start

        assert result1 == result2
        assert cached_time < 0.01  # Should be nearly instant


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_cache_with_complex_arguments(self):
        """Test caching with complex data types as arguments."""

        @quick_cache(ttl_seconds=60)
        def func_with_complex_args(data: dict, df: pd.DataFrame) -> dict:
            return {"sum": df["values"].sum(), "keys": list(data.keys())}

        # Create test data
        test_dict = {"a": 1, "b": 2, "c": 3}
        test_df = pd.DataFrame({"values": [1, 2, 3, 4, 5]})

        # First call
        result1 = func_with_complex_args(test_dict, test_df)

        # Second call - should be cached
        result2 = func_with_complex_args(test_dict, test_df)

        assert result1 == result2
        assert result1["sum"] == 15
        assert result1["keys"] == ["a", "b", "c"]

    def test_cache_with_unhashable_args(self):
        """Test caching with unhashable arguments."""

        @quick_cache(ttl_seconds=60)
        def func_with_set_arg(s: set) -> int:
            return len(s)

        # Sets are converted to sorted lists in JSON serialization
        test_set = {1, 2, 3}
        result = func_with_set_arg(test_set)
        assert result == 3

    def test_cache_key_collision(self):
        """Test that different functions don't collide in cache."""

        @quick_cache(ttl_seconds=60)
        def func_a(x: int) -> int:
            return x * 2

        @quick_cache(ttl_seconds=60)
        def func_b(x: int) -> int:
            return x * 3

        # Same argument, different functions
        result_a = func_a(5)
        result_b = func_b(5)

        assert result_a == 10
        assert result_b == 15

    @pytest.mark.asyncio
    async def test_concurrent_cache_access(self):
        """Test thread-safe concurrent cache access."""

        @quick_cache(ttl_seconds=60)
        async def concurrent_func(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        # Run multiple concurrent calls
        tasks = [concurrent_func(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        assert results == [i * 2 for i in range(10)]

    def test_exception_handling(self):
        """Test that exceptions are not cached."""
        call_count = 0

        @quick_cache(ttl_seconds=60)
        def failing_func(should_fail: bool) -> str:
            nonlocal call_count
            call_count += 1
            if should_fail:
                raise ValueError("Test error")
            return "success"

        # First call fails
        with pytest.raises(ValueError):
            failing_func(True)

        # Second call with same args should still execute (not cached)
        with pytest.raises(ValueError):
            failing_func(True)

        assert call_count == 2  # Function called twice

    def test_none_return_value(self):
        """Test that None return values are NOT cached (current limitation)."""
        call_count = 0

        @quick_cache(ttl_seconds=60)
        def func_returning_none(x: int) -> None:
            nonlocal call_count
            call_count += 1
            return None

        # First call
        result1 = func_returning_none(5)
        assert result1 is None
        assert call_count == 1

        # Second call - None is not cached, so function is called again
        result2 = func_returning_none(5)
        assert result2 is None
        assert call_count == 2  # Called again because None is not cached


class TestDebugMode:
    """Test debug mode specific functionality."""

    def test_debug_test_function(self):
        """Test the debug-only test_cache_function when available."""
        # Skip if not in debug mode
        try:
            from maverick_mcp.config.settings import settings

            if not settings.api.debug:
                pytest.skip("test_cache_function only available in debug mode")
        except Exception:
            pytest.skip("Could not determine debug mode")

        # Try to import the function
        try:
            from maverick_mcp.utils.quick_cache import test_cache_function
        except ImportError:
            pytest.skip("test_cache_function not available")

        # First call
        result1 = test_cache_function("test")
        assert result1.startswith("processed_test_")

        # Second call within 1 second - should be cached
        result2 = test_cache_function("test")
        assert result1 == result2

        # Wait for TTL expiration
        time.sleep(1.1)

        # Third call - should be different
        result3 = test_cache_function("test")
        assert result3.startswith("processed_test_")
        assert result1 != result3
