"""Redis expiry must not turn a short quote into a long-lived memory entry."""

from types import SimpleNamespace

import pytest

from maverick.platform import cache as cache_module
from maverick.platform.cache import Cache, RedisTier
from maverick.platform.config import CacheSettings
from maverick.platform.serde import serialize


class RedisWithExpiry:
    def __init__(self, remaining):
        self.remaining = remaining
        self.payload = serialize("quote")

    async def get(self, key):
        return self.payload

    async def pttl(self, key):
        return self.remaining


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(cache_module, "time", SimpleNamespace(time=lambda: now[0]))
    return now


async def test_subsecond_redis_expiry_is_preserved_in_memory(clock):
    cache = Cache(settings=CacheSettings(), redis_client=RedisWithExpiry(400))
    assert await cache.get("quote") == "quote"
    clock[0] += 0.39
    assert await cache.memory.get("v1:quote") is not None
    clock[0] += 0.02
    assert await cache.memory.get("v1:quote") is None


@pytest.mark.parametrize("remaining", [0, -2])
async def test_expired_between_get_and_pttl_is_a_miss(remaining, clock):
    cache = Cache(settings=CacheSettings(), redis_client=RedisWithExpiry(remaining))
    assert await cache.get("quote") is None
    assert await cache.memory.get("v1:quote") is None


async def test_nonexpiring_redis_key_is_served_without_inventing_expiry(clock):
    cache = Cache(settings=CacheSettings(), redis_client=RedisWithExpiry(-1))
    assert await cache.get("quote") == "quote"
    assert await cache.memory.get("v1:quote") is None


async def test_pttl_is_preferred_over_rounded_ttl(clock):
    class BothExpiryMethods(RedisWithExpiry):
        async def ttl(self, key):
            return 0

    tier = RedisTier(BothExpiryMethods(400), default_ttl_seconds=604800)
    assert await tier.get_with_expiry("quote") == (serialize("quote"), 1000.4)


@pytest.mark.parametrize(
    "seconds,expected", [(2, 1002.0), (0, None), (-2, None), (-1, 0)]
)
async def test_seconds_only_clients_keep_expiry_semantics(seconds, expected, clock):
    class SecondsClient:
        async def get(self, key):
            return serialize("quote")

        async def ttl(self, key):
            return seconds

    entry = await RedisTier(
        SecondsClient(), default_ttl_seconds=604800
    ).get_with_expiry("quote")
    assert entry == (None if expected is None else (serialize("quote"), expected))


async def test_pttl_roundtrip_does_not_extend_lifetime(clock):
    class SlowExpiry(RedisWithExpiry):
        async def pttl(self, key):
            clock[0] += 0.25
            return 400

    cache = Cache(settings=CacheSettings(), redis_client=SlowExpiry(400))
    assert await cache.get("quote") == "quote"
    clock[0] = 1000.41
    assert await cache.memory.get("v1:quote") is None
