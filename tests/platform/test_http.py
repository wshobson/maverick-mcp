"""Tests for maverick.platform.http."""

import asyncio

import httpx
import pytest

from maverick.platform.config import HttpSettings
from maverick.platform.http import (
    CircuitBreaker,
    CircuitOpenError,
    RateLimiter,
    create_client,
    get_breaker,
    request_with_retry,
)


def _settings(**overrides) -> HttpSettings:
    base = dict(  # noqa: C408
        timeout_seconds=1.0,
        retries=2,
        backoff_base_seconds=0.0,
        rate_limit_per_second=1000.0,
        breaker_failure_threshold=2,
        breaker_recovery_seconds=0.05,
    )
    base.update(overrides)
    return HttpSettings(**base)


async def test_retry_then_success():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(503)
        return httpx.Response(200, json={"ok": True})

    client = create_client(_settings(), transport=httpx.MockTransport(handler))
    response = await request_with_retry(
        client, "GET", "https://api.example.com/x", retries=2, backoff_base=0.0
    )
    assert response.status_code == 200
    assert calls == 3


async def test_retries_exhausted_returns_last_response():
    client = create_client(
        _settings(), transport=httpx.MockTransport(lambda r: httpx.Response(503))
    )
    response = await request_with_retry(
        client, "GET", "https://api.example.com/x", retries=1, backoff_base=0.0
    )
    assert response.status_code == 503


async def test_breaker_opens_after_threshold_and_recovers():
    breaker = CircuitBreaker("svc", _settings())

    async def failing():
        raise RuntimeError("down")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            await breaker.call(failing)
    assert breaker.state == "open"
    with pytest.raises(CircuitOpenError):
        await breaker.call(failing)

    await asyncio.sleep(0.06)

    async def healthy():
        return "up"

    assert await breaker.call(healthy) == "up"
    assert breaker.state == "closed"


def test_breaker_registry_returns_same_instance():
    a = get_breaker("tiingo", _settings())
    assert get_breaker("tiingo") is a
    assert get_breaker("fred") is not a


async def test_rate_limiter_spaces_calls():
    limiter = RateLimiter(rate_per_second=50.0, burst=1)
    loop = asyncio.get_running_loop()
    start = loop.time()
    for _ in range(3):
        await limiter.acquire()
    elapsed = loop.time() - start
    assert elapsed >= 0.03
