"""HTTP resilience: retry, circuit breaker, and rate limiting.

The only module in maverick/ that wraps outbound HTTP calls with retry,
circuit-breaking, and rate-limiting behavior. Consumes `HttpSettings` from
`maverick.platform.config`.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import httpx

from maverick.platform.config import HttpSettings

T = TypeVar("T")

_DEFAULT_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class CircuitOpenError(Exception):
    """Raised when a call is rejected because the breaker is open."""

    def __init__(self, name: str, seconds_until_half_open: float) -> None:
        self.name = name
        self.seconds_until_half_open = seconds_until_half_open
        super().__init__(
            f"Circuit breaker '{name}' is open; retry in {seconds_until_half_open:.2f}s"
        )


class CircuitBreaker:
    """Per-service circuit breaker: CLOSED -> OPEN -> HALF_OPEN -> CLOSED.

    Counts consecutive failures. Opens once the failure threshold is
    reached. After the recovery window elapses, the next call is let
    through in the HALF_OPEN state: success closes the breaker, failure
    reopens it.
    """

    def __init__(self, name: str, settings: HttpSettings) -> None:
        self.name = name
        self._settings = settings
        self._state = "closed"
        self._failure_count = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> str:
        return self._state

    def reset(self) -> None:
        """Reset the breaker to CLOSED with no recorded failures."""
        self._state = "closed"
        self._failure_count = 0
        self._opened_at = None

    def _seconds_until_half_open(self) -> float:
        if self._opened_at is None:
            return 0.0
        elapsed = time.monotonic() - self._opened_at
        remaining = self._settings.breaker_recovery_seconds - elapsed
        return max(0.0, remaining)

    async def call(
        self, fn: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any
    ) -> T:
        if self._state == "open":
            if self._seconds_until_half_open() > 0:
                raise CircuitOpenError(self.name, self._seconds_until_half_open())
            self._state = "half_open"

        try:
            result = await fn(*args, **kwargs)
        except Exception:
            if self._state == "half_open":
                self._state = "open"
                self._opened_at = time.monotonic()
            else:
                self._failure_count += 1
                if self._failure_count >= self._settings.breaker_failure_threshold:
                    self._state = "open"
                    self._opened_at = time.monotonic()
            raise
        else:
            self._failure_count = 0
            self._state = "closed"
            self._opened_at = None
            return result


_breakers: dict[str, CircuitBreaker] = {}


def get_breaker(name: str, settings: HttpSettings | None = None) -> CircuitBreaker:
    """Return the process-global breaker for `name`, creating it on first use."""
    if name not in _breakers:
        _breakers[name] = CircuitBreaker(name, settings or HttpSettings())
    return _breakers[name]


def reset_breakers() -> None:
    """Clear the process-global breaker registry (for tests)."""
    _breakers.clear()


class RateLimiter:
    """Async token-bucket rate limiter."""

    def __init__(self, rate_per_second: float, burst: float | None = None) -> None:
        self._rate = rate_per_second
        self._capacity = burst if burst is not None else rate_per_second
        self._tokens = self._capacity
        self._updated_at = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self._updated_at
                self._updated_at = now
                self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)

                if self._tokens >= 1:
                    self._tokens -= 1
                    return

                deficit = 1 - self._tokens
                wait_seconds = deficit / self._rate
                await asyncio.sleep(wait_seconds)


async def request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    retries: int,
    backoff_base: float,
    retry_statuses: frozenset[int] | set[int] = _DEFAULT_RETRY_STATUSES,
    **kwargs: Any,
) -> httpx.Response:
    """Issue a request, retrying on named statuses and transport errors.

    Retries with exponential backoff (`backoff_base * 2**attempt`). When
    retries are exhausted, the last response is returned (not raised) if
    the failure was a retryable status; a transport error is re-raised.
    """
    response: httpx.Response | None = None
    for attempt in range(retries + 1):
        try:
            response = await client.request(method, url, **kwargs)
        except httpx.TransportError:
            if attempt >= retries:
                raise
            await asyncio.sleep(backoff_base * 2**attempt)
            continue

        if response.status_code not in retry_statuses:
            return response
        if attempt >= retries:
            return response
        await asyncio.sleep(backoff_base * 2**attempt)

    assert response is not None
    return response


def create_client(
    settings: HttpSettings | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.AsyncClient:
    """Build an `httpx.AsyncClient` configured from `HttpSettings`."""
    if settings is None:
        settings = HttpSettings()
    return httpx.AsyncClient(timeout=settings.timeout_seconds, transport=transport)
