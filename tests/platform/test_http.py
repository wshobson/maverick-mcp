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
    request_resilient,
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


async def test_half_open_admits_single_probe():
    breaker = CircuitBreaker("svc", _settings(breaker_failure_threshold=1))

    async def failing():
        raise RuntimeError("down")

    with pytest.raises(RuntimeError):
        await breaker.call(failing)
    assert breaker.state == "open"

    await asyncio.sleep(0.06)

    calls = 0

    async def slow_probe():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return "up"

    results = await asyncio.gather(
        *(breaker.call(slow_probe) for _ in range(5)), return_exceptions=True
    )

    successes = [r for r in results if r == "up"]
    open_errors = [r for r in results if isinstance(r, CircuitOpenError)]
    assert calls == 1
    assert len(successes) == 1
    assert len(open_errors) == 4
    assert breaker.state == "closed"

    assert await breaker.call(slow_probe) == "up"
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


def test_rate_limiter_rejects_non_positive_rate():
    with pytest.raises(ValueError):
        RateLimiter(0)
    with pytest.raises(ValueError):
        RateLimiter(-1.0)


async def test_request_resilient_succeeds_end_to_end():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"ok": True})

    client = create_client(_settings(), transport=httpx.MockTransport(handler))
    response = await request_resilient(
        "resilient-ok",
        client,
        "GET",
        "https://api.example.com/x",
        settings=_settings(),
    )
    assert response.status_code == 200
    assert calls == 1


async def test_request_resilient_opens_breaker_and_short_circuits_transport():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("boom", request=request)

    client = create_client(_settings(), transport=httpx.MockTransport(handler))
    settings = _settings(retries=0, breaker_failure_threshold=2)

    for _ in range(2):
        with pytest.raises(httpx.ConnectError):
            await request_resilient(
                "resilient-fail",
                client,
                "GET",
                "https://api.example.com/x",
                settings=settings,
            )
    assert calls == 2

    with pytest.raises(CircuitOpenError):
        await request_resilient(
            "resilient-fail",
            client,
            "GET",
            "https://api.example.com/x",
            settings=settings,
        )
    # Breaker short-circuited before reaching the transport.
    assert calls == 2


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
async def test_exhausted_status_opens_breaker_and_failed_probe_reopens(
    status, monkeypatch
):
    from types import SimpleNamespace

    from maverick.platform import http

    clock = [100.0]
    monkeypatch.setattr(http, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    calls = 0
    returned_status = status

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(returned_status)

    settings = _settings(retries=0, breaker_failure_threshold=1)
    name = f"exhausted-{status}"
    async with create_client(
        settings, transport=httpx.MockTransport(handler)
    ) as client:

        async def call():
            return await request_resilient(
                name, client, "GET", "https://example.invalid", settings=settings
            )

        with pytest.raises(httpx.HTTPStatusError):
            await call()
        with pytest.raises(CircuitOpenError):
            await call()
        assert calls == 1
        clock[0] += 1
        with pytest.raises(httpx.HTTPStatusError):
            await call()
        assert get_breaker(name).state == "open"
        assert calls == 2
        clock[0] += 1
        returned_status = 200
        assert (await call()).status_code == 200
        assert get_breaker(name).state == "closed"
        assert calls == 3


async def test_nonretryable_status_keeps_response_for_provider_hint():
    settings = _settings(retries=0, breaker_failure_threshold=1)
    async with create_client(
        settings, transport=httpx.MockTransport(lambda request: httpx.Response(403))
    ) as client:
        response = await request_resilient(
            "forbidden-hint",
            client,
            "GET",
            "https://example.invalid",
            settings=settings,
        )
    assert response.status_code == 403
    assert get_breaker("forbidden-hint").state == "closed"


async def test_resilient_custom_retry_status_exhaustion_uses_same_policy():
    settings = _settings(retries=1, breaker_failure_threshold=1)
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(418)

    async with create_client(
        settings, transport=httpx.MockTransport(handler)
    ) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await request_resilient(
                "custom-status",
                client,
                "GET",
                "https://example.invalid",
                settings=settings,
                retry_statuses={418},
            )
    assert calls == 2
    assert get_breaker("custom-status").state == "open"
