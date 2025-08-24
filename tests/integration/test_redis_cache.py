"""
Integration tests for Redis caching functionality.
"""

import asyncio
import json
from datetime import datetime

import pytest

from tests.integration.base import RedisIntegrationTest


@pytest.mark.integration
@pytest.mark.redis
class TestRedisCache(RedisIntegrationTest):
    """Test Redis caching with real Redis instance."""

    async def test_basic_cache_operations(self):
        """Test basic cache set/get/delete operations."""
        # Set value
        key = "test:basic:key"
        value = {"data": "test value", "timestamp": datetime.now().isoformat()}

        await self.redis_client.setex(
            key,
            300,  # 5 minutes TTL
            json.dumps(value),
        )

        # Get value
        cached = await self.redis_client.get(key)
        assert cached is not None

        cached_data = json.loads(cached)
        assert cached_data["data"] == value["data"]
        assert cached_data["timestamp"] == value["timestamp"]

        # Delete value
        deleted = await self.redis_client.delete(key)
        assert deleted == 1

        # Verify deleted
        await self.assert_cache_not_exists(key)

    async def test_cache_expiration(self):
        """Test cache TTL and expiration."""
        key = "test:expiry:key"
        value = "expires soon"

        # Set with 1 second TTL
        await self.redis_client.setex(key, 1, value)

        # Should exist immediately
        await self.assert_cache_exists(key)

        # Wait for expiration
        await asyncio.sleep(1.5)

        # Should be expired
        await self.assert_cache_not_exists(key)

    async def test_stock_data_caching(self):
        """Test caching of stock data."""
        from maverick_mcp.data.cache import CacheManager

        cache_manager = CacheManager()

        # Create sample stock data
        stock_data = {
            "symbol": "AAPL",
            "data": {"sample": "data"},  # Simplified for test
            "timestamp": datetime.now().isoformat(),
        }

        # Cache stock data
        cache_key = "stock:AAPL:1d"
        await cache_manager.set(
            cache_key,
            stock_data,
            ttl=3600,  # 1 hour
        )

        # Retrieve from cache
        cached = await cache_manager.get(cache_key)
        assert cached is not None
        assert cached["symbol"] == "AAPL"
        assert "data" in cached

        # Test cache invalidation
        await cache_manager.delete(cache_key)

        # Should be removed
        cached = await cache_manager.get(cache_key)
        assert cached is None

    # Test commented out - rate_limiter module not available
    # async def test_rate_limiting_cache(self):
    #     """Test rate limiting using Redis."""
    #     from maverick_mcp.auth.rate_limiter import RateLimiter
    #
    #     rate_limiter = RateLimiter(self.redis_client)
    #
    #     # Configure rate limit: 5 requests per minute
    #     user_id = "test_user_123"
    #     limit = 5
    #     window = 60  # seconds
    #
    #     # Make requests up to limit
    #     for _ in range(limit):
    #         allowed = await rate_limiter.check_rate_limit(user_id, limit, window)
    #         assert allowed is True
    #
    #     # Next request should be blocked
    #     allowed = await rate_limiter.check_rate_limit(user_id, limit, window)
    #     assert allowed is False
    #
    #     # Check remaining
    #     remaining = await rate_limiter.get_remaining_requests(user_id, limit, window)
    #     assert remaining == 0

    async def test_distributed_locking(self):
        """Test distributed locking with Redis."""
        import uuid

        lock_key = "test:lock:resource"
        lock_value = str(uuid.uuid4())
        lock_ttl = 5  # seconds

        # Acquire lock
        acquired = await self.redis_client.set(
            lock_key,
            lock_value,
            nx=True,  # Only set if not exists
            ex=lock_ttl,
        )
        assert acquired is not None  # Redis returns 'OK' string on success

        # Try to acquire again (should fail)
        acquired2 = await self.redis_client.set(
            lock_key, "different_value", nx=True, ex=lock_ttl
        )
        assert acquired2 is None  # Redis returns None when nx fails

        # Release lock (only if we own it)
        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """

        released = await self.redis_client.eval(
            lua_script, keys=[lock_key], args=[lock_value]
        )
        assert released == 1

        # Lock should be available now
        acquired3 = await self.redis_client.set(
            lock_key, "new_value", nx=True, ex=lock_ttl
        )
        assert acquired3 is not None  # Redis returns 'OK' string on success

    async def test_cache_patterns(self):
        """Test various cache key patterns and operations."""
        # Set multiple keys with pattern
        base_pattern = "test:pattern"
        for i in range(10):
            key = f"{base_pattern}:{i}"
            await self.redis_client.set(key, f"value_{i}")

        # Scan for keys matching pattern
        keys = []
        cursor = 0
        while True:
            cursor, batch = await self.redis_client.scan(
                cursor, match=f"{base_pattern}:*", count=100
            )
            keys.extend(batch)
            if cursor == 0:
                break

        assert len(keys) == 10

        # Bulk get
        values = await self.redis_client.mget(keys)
        assert len(values) == 10
        assert all(v is not None for v in values)

        # Bulk delete
        deleted = await self.redis_client.delete(*keys)
        assert deleted == 10

    async def test_cache_statistics(self):
        """Test cache hit/miss statistics."""
        stats_key = "cache:stats"

        # Initialize stats
        await self.redis_client.hset(
            stats_key,
            mapping={
                "hits": 0,
                "misses": 0,
                "total": 0,
            },
        )

        # Simulate cache operations
        async def record_hit():
            await self.redis_client.hincrby(stats_key, "hits", 1)
            await self.redis_client.hincrby(stats_key, "total", 1)

        async def record_miss():
            await self.redis_client.hincrby(stats_key, "misses", 1)
            await self.redis_client.hincrby(stats_key, "total", 1)

        # Simulate 70% hit rate
        for i in range(100):
            if i % 10 < 7:
                await record_hit()
            else:
                await record_miss()

        # Get stats
        stats = await self.redis_client.hgetall(stats_key)

        hits = int(stats[b"hits"])
        misses = int(stats[b"misses"])
        total = int(stats[b"total"])

        assert total == 100
        assert hits == 70
        assert misses == 30

        hit_rate = hits / total
        assert hit_rate == 0.7

    async def test_pub_sub_messaging(self):
        """Test Redis pub/sub for real-time updates."""
        channel = "test:updates"
        message = {"type": "price_update", "symbol": "AAPL", "price": 150.50}

        # Create pubsub
        pubsub = self.redis_client.pubsub()
        await pubsub.subscribe(channel)

        # Publish message
        await self.redis_client.publish(channel, json.dumps(message))

        # Receive message
        received = None
        async for msg in pubsub.listen():
            if msg["type"] == "message":
                received = json.loads(msg["data"])
                break

        assert received is not None
        assert received["type"] == "price_update"
        assert received["symbol"] == "AAPL"
        assert received["price"] == 150.50

        # Cleanup
        await pubsub.unsubscribe(channel)
        await pubsub.close()

    async def test_cache_warming(self):
        """Test cache warming strategies."""
        # Simulate warming cache with frequently accessed data
        frequent_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

        # Warm cache
        for symbol in frequent_symbols:
            cache_key = f"stock:quote:{symbol}"
            quote_data = {
                "symbol": symbol,
                "price": 100.0 + hash(symbol) % 100,
                "volume": 1000000,
                "timestamp": datetime.now().isoformat(),
            }

            await self.redis_client.setex(
                cache_key,
                3600,  # 1 hour
                json.dumps(quote_data),
            )

        # Verify all cached
        for symbol in frequent_symbols:
            cache_key = f"stock:quote:{symbol}"
            await self.assert_cache_exists(cache_key)

        # Test batch retrieval
        keys = [f"stock:quote:{symbol}" for symbol in frequent_symbols]
        values = await self.redis_client.mget(keys)

        assert len(values) == len(frequent_symbols)
        assert all(v is not None for v in values)

        # Parse and verify
        for value, symbol in zip(values, frequent_symbols, strict=False):
            data = json.loads(value)
            assert data["symbol"] == symbol

    async def test_cache_memory_optimization(self):
        """Test memory optimization strategies."""
        # Test different serialization formats
        import pickle
        import zlib

        large_data = {
            "symbol": "TEST",
            "historical_data": [
                {"date": f"2024-01-{i:02d}", "price": 100 + i} for i in range(1, 32)
            ]
            * 10,  # Replicate to make it larger
        }

        # JSON serialization
        json_data = json.dumps(large_data)
        json_size = len(json_data.encode())

        # Pickle serialization
        pickle_data = pickle.dumps(large_data)
        pickle_size = len(pickle_data)

        # Compressed JSON
        compressed_data = zlib.compress(json_data.encode())
        compressed_size = len(compressed_data)

        # Store all versions
        await self.redis_client.set("test:json", json_data)
        await self.redis_client.set("test:pickle", pickle_data)
        await self.redis_client.set("test:compressed", compressed_data)

        # Compare sizes
        assert compressed_size < json_size
        assert compressed_size < pickle_size

        # Verify decompression works
        retrieved = await self.redis_client.get("test:compressed")
        decompressed = zlib.decompress(retrieved)
        restored_data = json.loads(decompressed)

        assert restored_data["symbol"] == "TEST"
        assert len(restored_data["historical_data"]) == 310
