"""Base classes and utilities for integration testing."""

from __future__ import annotations

import asyncio
import fnmatch
import time
from collections import defaultdict
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


class InMemoryPubSub:
    """Lightweight pub/sub implementation for the in-memory Redis stub."""

    def __init__(self, redis: InMemoryRedis) -> None:
        self._redis = redis
        self._queues: dict[str, asyncio.Queue[dict[str, Any]]] = {}
        self._active = True

    async def subscribe(self, channel: str) -> None:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._queues[channel] = queue
        self._redis.register_subscriber(channel, queue)

    async def unsubscribe(self, channel: str) -> None:
        queue = self._queues.pop(channel, None)
        if queue is not None:
            self._redis.unregister_subscriber(channel, queue)

    async def close(self) -> None:
        self._active = False
        for channel, _queue in list(self._queues.items()):
            await self.unsubscribe(channel)

    async def listen(self):  # pragma: no cover - simple async generator
        while self._active:
            tasks = [
                asyncio.create_task(queue.get()) for queue in self._queues.values()
            ]
            if not tasks:
                await asyncio.sleep(0.01)
                continue
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            for task in done:
                message = task.result()
                yield message


class InMemoryRedis:
    """A minimal asynchronous Redis replacement used in tests."""

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}
        self._hashes: dict[str, dict[str, str]] = defaultdict(dict)
        self._expiry: dict[str, float] = {}
        self._pubsub_channels: dict[str, list[asyncio.Queue[dict[str, Any]]]] = (
            defaultdict(list)
        )

    def _is_expired(self, key: str) -> bool:
        expiry = self._expiry.get(key)
        if expiry is None:
            return False
        if expiry < time.time():
            self._data.pop(key, None)
            self._hashes.pop(key, None)
            self._expiry.pop(key, None)
            return True
        return False

    def register_subscriber(
        self, channel: str, queue: asyncio.Queue[dict[str, Any]]
    ) -> None:
        self._pubsub_channels[channel].append(queue)

    def unregister_subscriber(
        self, channel: str, queue: asyncio.Queue[dict[str, Any]]
    ) -> None:
        if channel in self._pubsub_channels:
            try:
                self._pubsub_channels[channel].remove(queue)
            except ValueError:
                pass
            if not self._pubsub_channels[channel]:
                del self._pubsub_channels[channel]

    async def setex(self, key: str, ttl: int, value: Any) -> None:
        self._data[key] = self._encode(value)
        self._expiry[key] = time.time() + ttl

    async def set(
        self,
        key: str,
        value: Any,
        *,
        nx: bool = False,
        ex: int | None = None,
    ) -> str | None:
        if nx and key in self._data and not self._is_expired(key):
            return None
        self._data[key] = self._encode(value)
        if ex is not None:
            self._expiry[key] = time.time() + ex
        return "OK"

    async def get(self, key: str) -> bytes | None:
        if self._is_expired(key):
            return None
        return self._data.get(key)

    async def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            if key in self._data and not self._is_expired(key):
                removed += 1
            self._data.pop(key, None)
            self._hashes.pop(key, None)
            self._expiry.pop(key, None)
        return removed

    async def scan(
        self, cursor: int, match: str | None = None, count: int = 100
    ) -> tuple[int, list[str]]:
        keys = [key for key in self._data.keys() if not self._is_expired(key)]
        if match:
            keys = [key for key in keys if fnmatch.fnmatch(key, match)]
        return 0, keys[:count]

    async def mget(self, keys: list[str]) -> list[bytes | None]:
        return [await self.get(key) for key in keys]

    async def hincrby(self, key: str, field: str, amount: int) -> int:
        current = int(self._hashes[key].get(field, "0"))
        current += amount
        self._hashes[key][field] = str(current)
        return current

    async def hgetall(self, key: str) -> dict[bytes, bytes]:
        if self._is_expired(key):
            return {}
        mapping = self._hashes.get(key, {})
        return {
            field.encode("utf-8"): value.encode("utf-8")
            for field, value in mapping.items()
        }

    async def hset(self, key: str, mapping: dict[str, Any]) -> None:
        for field, value in mapping.items():
            self._hashes[key][field] = str(value)

    async def eval(self, script: str, keys: list[str], args: list[str]) -> int:
        if not keys:
            return 0
        key = keys[0]
        expected = args[0] if args else ""
        stored = await self.get(key)
        if stored is not None and stored.decode("utf-8") == expected:
            await self.delete(key)
            return 1
        return 0

    async def publish(self, channel: str, message: Any) -> None:
        encoded = self._encode(message)
        for queue in self._pubsub_channels.get(channel, []):
            await queue.put(
                {"type": "message", "channel": channel, "data": encoded.decode("utf-8")}
            )

    def pubsub(self) -> InMemoryPubSub:
        return InMemoryPubSub(self)

    def _encode(self, value: Any) -> bytes:
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            return value.encode("utf-8")
        return str(value).encode("utf-8")

    async def close(self) -> None:
        self._data.clear()
        self._hashes.clear()
        self._expiry.clear()
        self._pubsub_channels.clear()


class BaseIntegrationTest:
    """Base class for integration tests with common utilities."""

    def setup_test(self):
        """Set up test environment for each test."""
        return None

    def assert_response_success(self, response, expected_status: int = 200):
        """Assert that a response is successful."""
        if hasattr(response, "status_code"):
            assert response.status_code == expected_status, (
                f"Expected status {expected_status}, got {response.status_code}. "
                f"Response: {response.json() if hasattr(response, 'content') and response.content else 'No content'}"
            )


class RedisIntegrationTest(BaseIntegrationTest):
    """Integration tests that rely on a Redis-like backend."""

    redis_client: InMemoryRedis

    @pytest.fixture(autouse=True)
    async def _setup_redis(self):
        self.redis_client = InMemoryRedis()
        yield
        await self.redis_client.close()

    async def assert_cache_exists(self, key: str) -> None:
        value = await self.redis_client.get(key)
        assert value is not None, f"Expected cache key {key} to exist"

    async def assert_cache_not_exists(self, key: str) -> None:
        value = await self.redis_client.get(key)
        assert value is None, f"Expected cache key {key} to be absent"


class MockLLMBase:
    """Base mock LLM for consistent testing."""

    def __init__(self):
        self.ainvoke = AsyncMock()
        self.bind_tools = MagicMock(return_value=self)
        self.invoke = MagicMock()

        mock_response = MagicMock()
        mock_response.content = '{"insights": ["Test insight"], "sentiment": {"direction": "neutral", "confidence": 0.5}}'
        self.ainvoke.return_value = mock_response


class MockCacheManager:
    """Mock cache manager for testing."""

    def __init__(self):
        self.get = AsyncMock(return_value=None)
        self.set = AsyncMock()
        self._cache: dict[str, Any] = {}

    async def get_cached(self, key: str) -> Any:
        """Get value from mock cache."""
        return self._cache.get(key)

    async def set_cached(self, key: str, value: Any) -> None:
        """Set value in mock cache."""
        self._cache[key] = value
