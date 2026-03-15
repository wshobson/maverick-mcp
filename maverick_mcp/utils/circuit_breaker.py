"""
Circuit breaker facade — backward-compatible re-exports.

The actual implementation lives in ``circuit_breaker_adapter.py`` using
the ``circuitbreaker`` library.  This file preserves every public name
that used to be exported from the old 945-line custom implementation.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Callable
from enum import Enum
from typing import Any, ParamSpec, TypeVar, cast

from maverick_mcp.exceptions import (  # noqa: F401
    CircuitBreakerError,
    ExternalServiceError,
)
from maverick_mcp.utils.circuit_breaker_adapter import (
    SERVICE_CONFIGS,
    CircuitBreakerMetrics,  # noqa: F401
    CircuitState,  # noqa: F401
    MaverickCircuitBreaker,
    _breakers,  # noqa: F401
    get_all_circuit_breakers,  # noqa: F401
    get_circuit_breaker,
    get_circuit_breaker_status,  # noqa: F401
    get_or_create_breaker,
    initialize_circuit_breakers,
    register_circuit_breaker,  # noqa: F401
    reset_all_circuit_breakers,  # noqa: F401
)

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

# Backward-compatible alias
EnhancedCircuitBreaker = MaverickCircuitBreaker


class FailureDetectionStrategy(Enum):
    """Kept for backward compatibility; library uses consecutive only."""

    CONSECUTIVE_FAILURES = "consecutive"
    FAILURE_RATE = "failure_rate"
    TIMEOUT_RATE = "timeout_rate"
    COMBINED = "combined"


class CircuitBreakerConfig:
    """Backward-compatible config object."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        failure_rate_threshold: float = 0.5,
        timeout_threshold: float = 10.0,
        recovery_timeout: int = 60,
        success_threshold: int = 3,
        window_size: int = 60,
        detection_strategy: FailureDetectionStrategy = FailureDetectionStrategy.COMBINED,
        expected_exceptions: tuple[type[Exception], ...] = (Exception,),
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.failure_rate_threshold = failure_rate_threshold
        self.timeout_threshold = timeout_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.window_size = window_size
        self.detection_strategy = detection_strategy
        self.expected_exceptions = expected_exceptions


# Re-export the old CIRCUIT_BREAKER_CONFIGS built from SERVICE_CONFIGS
CIRCUIT_BREAKER_CONFIGS = {
    name: CircuitBreakerConfig(
        name=name,
        failure_threshold=cfg["failure_threshold"],
        recovery_timeout=cfg["recovery_timeout"],
    )
    for name, cfg in SERVICE_CONFIGS.items()
}


def _get_or_create_breaker(config: CircuitBreakerConfig) -> MaverickCircuitBreaker:
    """Backward-compatible helper."""
    return get_or_create_breaker(config.name)


# ---------------------------------------------------------------------------
# Decorator (backward-compatible)
# ---------------------------------------------------------------------------


def circuit_breaker(
    name: str | None = None,
    failure_threshold: int | None = None,
    failure_rate_threshold: float | None = None,
    timeout_threshold: float | None = None,
    recovery_timeout: int | None = None,
    expected_exceptions: tuple[type[Exception], ...] | None = None,
) -> Callable:
    """Decorator to apply circuit breaker to a function."""

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        cb_name = name or f"{func.__module__}.{getattr(func, '__name__', 'unknown')}"
        # If custom thresholds were provided, create a dedicated breaker
        # instead of reusing the shared one from the registry.
        if failure_threshold is not None or recovery_timeout is not None:
            from maverick_mcp.utils.circuit_breaker_adapter import (
                MaverickCircuitBreaker,
                register_circuit_breaker,
            )

            breaker = MaverickCircuitBreaker(
                name=cb_name,
                failure_threshold=failure_threshold or 5,
                recovery_timeout=recovery_timeout or 60,
            )
            register_circuit_breaker(cb_name, breaker)
        else:
            breaker = get_or_create_breaker(cb_name)

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await breaker.call_async(func, *args, **kwargs)

            return cast(Callable[..., T], async_wrapper)
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return breaker.call_sync(func, *args, **kwargs)

            return cast(Callable[..., T], sync_wrapper)

    return decorator


def with_circuit_breaker(service_name: str) -> Callable:
    """Decorator using a named circuit breaker."""

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            breaker = get_circuit_breaker(service_name)
            if not breaker:
                logger.warning("CB '%s' not found, executing unprotected", service_name)
                return func(*args, **kwargs)
            return breaker.call_sync(func, *args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def with_async_circuit_breaker(service_name: str) -> Callable:
    """Async decorator using a named circuit breaker."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            breaker = get_circuit_breaker(service_name)
            if not breaker:
                logger.warning("CB '%s' not found, executing unprotected", service_name)
                return await func(*args, **kwargs)
            return await breaker.call_async(func, *args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Manager (backward-compatible)
# ---------------------------------------------------------------------------


class CircuitBreakerManager:
    """Manager for all circuit breakers."""

    def __init__(self) -> None:
        self._breakers: dict[str, MaverickCircuitBreaker] = {}
        self._initialized = False

    def initialize(self) -> bool:
        if self._initialized:
            return True
        try:
            self._breakers = initialize_circuit_breakers()
            self._initialized = True
            return True
        except Exception as e:
            logger.error("Failed to initialize CB manager: %s", e)
            return False

    def get_breaker(self, service_name: str) -> MaverickCircuitBreaker | None:
        if not self._initialized:
            self.initialize()
        return self._breakers.get(service_name)

    def get_all_breakers(self) -> dict[str, MaverickCircuitBreaker]:
        if not self._initialized:
            self.initialize()
        return self._breakers.copy()

    def reset_breaker(self, service_name: str) -> bool:
        breaker = self.get_breaker(service_name)
        if breaker:
            breaker.reset()
            return True
        return False

    def reset_all_breakers(self) -> int:
        count = 0
        for name, breaker in self._breakers.items():
            try:
                breaker.reset()
                count += 1
            except Exception as e:
                logger.error("Failed to reset CB for %s: %s", name, e)
        return count

    def get_health_status(self) -> dict[str, dict[str, Any]]:
        if not self._initialized:
            self.initialize()
        status = {}
        for name, breaker in self._breakers.items():
            try:
                metrics = breaker.get_metrics()
                status[name] = {
                    "name": name,
                    "state": breaker.state.value,
                    "consecutive_failures": breaker.consecutive_failures,
                    "time_until_retry": breaker.time_until_retry(),
                    "metrics": {
                        "total_calls": metrics.get_total_calls(),
                        "success_rate": metrics.get_success_rate(),
                        "failure_rate": metrics.get_failure_rate(),
                        "avg_response_time": metrics.get_average_response_time(),
                        "last_failure_time": metrics.get_last_failure_time(),
                        "uptime_percentage": metrics.get_uptime_percentage(),
                    },
                }
            except Exception as e:
                status[name] = {"name": name, "state": "error", "error": str(e)}
        return status


# Global instance
_circuit_breaker_manager = CircuitBreakerManager()


def get_circuit_breaker_manager() -> CircuitBreakerManager:
    return _circuit_breaker_manager


def initialize_all_circuit_breakers() -> bool:
    return _circuit_breaker_manager.initialize()


def get_all_circuit_breaker_status() -> dict[str, dict[str, Any]]:
    return _circuit_breaker_manager.get_health_status()


# Service-specific shortcut decorators
def with_yfinance_circuit_breaker(func: F) -> F:  # noqa: UP047
    return cast(F, with_circuit_breaker("yfinance")(func))


def with_tiingo_circuit_breaker(func: F) -> F:  # noqa: UP047
    return cast(F, with_circuit_breaker("tiingo")(func))


def with_fred_circuit_breaker(func: F) -> F:  # noqa: UP047
    return cast(F, with_circuit_breaker("fred_api")(func))


def with_openrouter_circuit_breaker(func: F) -> F:  # noqa: UP047
    return cast(F, with_circuit_breaker("openrouter")(func))


def with_exa_circuit_breaker(func: F) -> F:  # noqa: UP047
    return cast(F, with_circuit_breaker("exa")(func))


def with_async_yfinance_circuit_breaker(func: F) -> F:  # noqa: UP047
    return cast(F, with_async_circuit_breaker("yfinance")(func))


def with_async_tiingo_circuit_breaker(func: F) -> F:  # noqa: UP047
    return cast(F, with_async_circuit_breaker("tiingo")(func))


def with_async_fred_circuit_breaker(func: F) -> F:  # noqa: UP047
    return cast(F, with_async_circuit_breaker("fred_api")(func))


def with_async_openrouter_circuit_breaker(func: F) -> F:  # noqa: UP047
    return cast(F, with_async_circuit_breaker("openrouter")(func))


def with_async_exa_circuit_breaker(func: F) -> F:  # noqa: UP047
    return cast(F, with_async_circuit_breaker("exa")(func))
