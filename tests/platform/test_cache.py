"""Tests for maverick.platform.cache."""

import asyncio

import pandas as pd

from maverick.platform.cache import (
    Cache,
    MemoryTier,
    SqliteTier,
    generate_cache_key,
)
from maverick.platform.config import CacheSettings


def _settings(tmp_path, **overrides) -> CacheSettings:
    base = dict(sqlite_path=str(tmp_path / "cache.db"))  # noqa: C408
    base.update(overrides)
    return CacheSettings(**base)


def test_cache_key_is_deterministic_and_versioned():
    a = generate_cache_key("quotes", ticker="AAPL", days=30)
    b = generate_cache_key("quotes", days=30, ticker="AAPL")
    assert a == b
    assert a.startswith("v1:quotes:")


def test_cache_key_long_inputs_hashed():
    key = generate_cache_key("x", blob="A" * 500)
    assert len(key) <= 250


async def test_memory_tier_roundtrip_and_expiry():
    tier = MemoryTier(max_items=10, max_bytes=10_000_000)
    await tier.set("k", b"payload", ttl=100)
    assert await tier.get("k") == b"payload"
    await tier.set("gone", b"x", ttl=-1)
    assert await tier.get("gone") is None


async def test_memory_tier_evicts_at_item_limit():
    tier = MemoryTier(max_items=3, max_bytes=10_000_000)
    for i in range(5):
        await tier.set(f"k{i}", b"x", ttl=100 + i)
    stored = [k for k in (f"k{i}" for i in range(5)) if await tier.get(k)]
    assert len(stored) == 3
    assert "k4" in stored


async def test_sqlite_tier_persists_across_instances(tmp_path):
    path = str(tmp_path / "cache.db")
    tier = SqliteTier(path)
    await tier.set("stay", b"here", ttl=100)
    fresh = SqliteTier(path)
    assert await fresh.get("stay") == b"here"


async def test_sqlite_tier_expires(tmp_path):
    tier = SqliteTier(str(tmp_path / "cache.db"))
    await tier.set("gone", b"x", ttl=-1)
    assert await tier.get("gone") is None


async def test_cache_facade_dataframe_roundtrip(tmp_path):
    cache = Cache(settings=_settings(tmp_path))
    df = pd.DataFrame({"Close": [1.0, 2.0, 3.0]})
    await cache.set("df", df, ttl=100)
    result = await cache.get("df")
    pd.testing.assert_frame_equal(result, df)


async def test_cache_facade_backfills_memory_from_sqlite(tmp_path):
    settings = _settings(tmp_path)
    first = Cache(settings=settings)
    await first.set("warm", {"a": 1}, ttl=100)
    second = Cache(settings=settings)
    assert await second.get("warm") == {"a": 1}
    assert await second.memory.get(second._versioned("warm")) is not None


async def test_cache_facade_get_many_and_delete_pattern(tmp_path):
    cache = Cache(settings=_settings(tmp_path))
    await cache.set("q:AAPL", 1, ttl=100)
    await cache.set("q:MSFT", 2, ttl=100)
    await cache.set("other", 3, ttl=100)
    many = await cache.get_many(["q:AAPL", "q:MSFT", "missing"])
    assert many == {"q:AAPL": 1, "q:MSFT": 2}
    removed = await cache.delete_pattern("q:*")
    assert removed == 2
    assert await cache.get("other") == 3


async def test_cache_facade_backfill_uses_entry_ttl_not_default(tmp_path):
    settings = _settings(tmp_path)
    first = Cache(settings=settings)
    await first.set("short", "value", ttl=1)
    second = Cache(settings=settings)
    assert await second.get("short") == "value"  # backfills memory
    await asyncio.sleep(1.1)
    assert await second.get("short") is None


async def test_cache_facade_delete_pattern_counts_across_tiers(tmp_path):
    settings = _settings(tmp_path, memory_max_items=1)
    cache = Cache(settings=settings)
    await cache.set("q:AAPL", 1, ttl=100)
    await cache.set("q:MSFT", 2, ttl=100)  # evicts q:AAPL from memory
    removed = await cache.delete_pattern("q:*")
    assert removed == 2


class FakeRedis:
    def __init__(self):
        self.store = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value

    async def delete(self, *keys):
        return sum(1 for k in keys if self.store.pop(k, None) is not None)

    async def exists(self, key):
        return 1 if key in self.store else 0


async def test_cache_facade_uses_injected_redis(tmp_path):
    settings = _settings(tmp_path)
    cache = Cache(settings=settings, redis_client=FakeRedis())
    await cache.set("r", "value", ttl=100)
    assert await cache.get("r") == "value"
    assert cache.sqlite is None
