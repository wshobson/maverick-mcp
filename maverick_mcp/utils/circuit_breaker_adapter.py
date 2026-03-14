"""
Circuit breaker adapter using the `circuitbreaker` library.

Wraps the library's CircuitBreaker with the same status/metrics API
the codebase already depends on, so all existing imports keep working.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from enum import Enum
from typing import Any

import circuitbreaker as _cb

from maverick_mcp.exceptions import CircuitBreakerError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State enum (backward-compatible with the old CircuitState)
# ---------------------------------------------------------------------------


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# ---------------------------------------------------------------------------
# Lightweight metrics (replaces the old 100+ line class)
# ---------------------------------------------------------------------------


class CircuitBreakerMetrics:
    """Lightweight metrics using deque with window-based cleanup."""

    def __init__(self, window_size: int = 300):
        self.window_size = window_size
        self.calls: deque[tuple[float, bool, float]] = deque()
        self.state_changes: deque[tuple[float, str]] = deque()
        self._lock = threading.Lock()

    def record_call(self, success: bool, duration: float) -> None:
        with self._lock:
            now = time.time()
            self.calls.append((now, success, duration))
            self._cleanup(now)

    def record_state_change(self, new_state: str) -> None:
        with self._lock:
            self.state_changes.append((time.time(), new_state))

    # -- public accessors (keep same API) --

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            self._cleanup(time.time())
            if not self.calls:
                return {
                    "total_calls": 0,
                    "success_rate": 1.0,
                    "failure_rate": 0.0,
                    "avg_duration": 0.0,
                    "timeout_rate": 0.0,
                }
            total = len(self.calls)
            successes = sum(1 for _, s, _ in self.calls if s)
            durations = [d for _, _, d in self.calls]
            timeouts = sum(1 for _, s, d in self.calls if not s and d >= 10.0)
            return {
                "total_calls": total,
                "success_rate": successes / total,
                "failure_rate": (total - successes) / total,
                "avg_duration": sum(durations) / total,
                "timeout_rate": timeouts / total,
                "min_duration": min(durations),
                "max_duration": max(durations),
            }

    def get_total_calls(self) -> int:
        return self.get_stats()["total_calls"]

    def get_success_rate(self) -> float:
        return self.get_stats()["success_rate"]

    def get_failure_rate(self) -> float:
        return self.get_stats()["failure_rate"]

    def get_average_response_time(self) -> float:
        return self.get_stats()["avg_duration"]

    def get_last_failure_time(self) -> float | None:
        with self._lock:
            for ts, s, _ in reversed(self.calls):
                if not s:
                    return ts
        return None

    def get_uptime_percentage(self) -> float:
        with self._lock:
            if not self.state_changes:
                return 100.0
            now = time.time()
            window_start = now - self.window_size
            uptime = 0.0
            last_time = window_start
            last_state = "closed"
            for ts, state in self.state_changes:
                if ts < window_start:
                    last_state = state
                    continue
                if last_state == "closed":
                    uptime += ts - last_time
                last_time = ts
                last_state = state
            if last_state == "closed":
                uptime += now - last_time
            total = now - window_start
            return (uptime / total * 100) if total > 0 else 100.0

    def _cleanup(self, now: float) -> None:
        cutoff = now - self.window_size
        while self.calls and self.calls[0][0] < cutoff:
            self.calls.popleft()
        state_cutoff = now - self.window_size * 10
        while self.state_changes and self.state_changes[0][0] < state_cutoff:
            self.state_changes.popleft()


# ---------------------------------------------------------------------------
# Service configs (replaces 6 subclasses + duplicate CIRCUIT_BREAKER_CONFIGS)
# ---------------------------------------------------------------------------

SERVICE_CONFIGS: dict[str, dict[str, Any]] = {
    "yfinance": {"failure_threshold": 3, "recovery_timeout": 120},
    "tiingo": {"failure_threshold": 5, "recovery_timeout": 60},
    "fred_api": {"failure_threshold": 5, "recovery_timeout": 300},
    "openrouter": {"failure_threshold": 5, "recovery_timeout": 120},
    "exa": {"failure_threshold": 4, "recovery_timeout": 90},
    "news_api": {"failure_threshold": 3, "recovery_timeout": 300},
    "finviz": {"failure_threshold": 5, "recovery_timeout": 180},
    "external_api": {"failure_threshold": 3, "recovery_timeout": 60},
    "http_general": {"failure_threshold": 5, "recovery_timeout": 60},
    "finnhub": {"failure_threshold": 5, "recovery_timeout": 120},
}


# ---------------------------------------------------------------------------
# MaverickCircuitBreaker — thin wrapper around library CircuitBreaker
# ---------------------------------------------------------------------------


class MaverickCircuitBreaker:
    """
    Wraps ``circuitbreaker.CircuitBreaker`` and adds the status/metrics
    API that the rest of the codebase depends on.
    """

    def __init__(
        self,
        name_or_config: str | Any = "default",
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type[Exception] | tuple[type[Exception], ...] = Exception,
        *,
        name: str | None = None,
    ):
        # Support both MaverickCircuitBreaker("svc", 3, 60) and
        # MaverickCircuitBreaker(config) for backward compat.
        if name is not None:
            # Called as keyword: MaverickCircuitBreaker(name="svc")
            _name = name
        elif isinstance(name_or_config, str):
            _name = name_or_config
        else:
            # Assume it's a config-like object with .name, .failure_threshold, etc.
            cfg = name_or_config
            _name = cfg.name
            failure_threshold = getattr(cfg, "failure_threshold", failure_threshold)
            recovery_timeout = getattr(cfg, "recovery_timeout", recovery_timeout)
            expected_exception = getattr(cfg, "expected_exceptions", (Exception,))

        self.name = _name
        self._cb = _cb.CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
            name=_name,
        )
        self._metrics = CircuitBreakerMetrics()
        # Expose config-like attributes for backward compat
        self.config = _CompatConfig(
            name=_name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )

    # -- state properties --

    @property
    def state(self) -> CircuitState:
        s = self._cb.state
        if s == _cb.STATE_OPEN:
            return CircuitState.OPEN
        if s == _cb.STATE_HALF_OPEN:
            return CircuitState.HALF_OPEN
        return CircuitState.CLOSED

    @property
    def consecutive_failures(self) -> int:
        return self._cb.failure_count

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    @property
    def is_closed(self) -> bool:
        return self.state == CircuitState.CLOSED

    def get_metrics(self) -> CircuitBreakerMetrics:
        return self._metrics

    def time_until_retry(self) -> float | None:
        remaining = self._cb.open_remaining
        if remaining and remaining > 0:
            return remaining
        return None

    # -- call methods --

    def call_sync(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Call a sync function through the circuit breaker."""
        start = time.time()
        try:
            # Wrap the call through the library's __call__ path
            @self._cb
            def _wrapped():
                return func(*args, **kwargs)

            result = _wrapped()
            self._metrics.record_call(True, time.time() - start)
            return result
        except _cb.CircuitBreakerError:
            self._metrics.record_call(False, time.time() - start)
            raise CircuitBreakerError(
                service=self.name,
                failure_count=self._cb.failure_count,
                threshold=self._cb._failure_threshold,
                context={
                    "state": self.state.value,
                    "time_until_retry": round(self._cb.open_remaining or 0, 1),
                },
            )
        except Exception:
            self._metrics.record_call(False, time.time() - start)
            raise

    async def call_async(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Call an async function through the circuit breaker."""
        start = time.time()
        try:

            @self._cb
            async def _wrapped():
                return await func(*args, **kwargs)

            result = await _wrapped()
            self._metrics.record_call(True, time.time() - start)
            return result
        except _cb.CircuitBreakerError:
            self._metrics.record_call(False, time.time() - start)
            raise CircuitBreakerError(
                service=self.name,
                failure_count=self._cb.failure_count,
                threshold=self._cb._failure_threshold,
                context={
                    "state": self.state.value,
                    "time_until_retry": round(self._cb.open_remaining or 0, 1),
                },
            )
        except Exception:
            self._metrics.record_call(False, time.time() - start)
            raise

    def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        return self.call_sync(func, *args, **kwargs)

    def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        self._cb.reset()
        self._metrics.record_state_change("closed")
        logger.info("Circuit breaker '%s' manually reset", self.name)

    def get_status(self) -> dict[str, Any]:
        """Get detailed circuit breaker status (backward-compatible format)."""
        stats = self._metrics.get_stats()
        remaining = self._cb.open_remaining
        return {
            "name": self.name,
            "state": self.state.value,
            "consecutive_failures": self._cb.failure_count,
            "time_until_retry": round(remaining, 1)
            if remaining and remaining > 0
            else None,
            "metrics": stats,
            "config": {
                "failure_threshold": self._cb._failure_threshold,
                "failure_rate_threshold": 0.5,
                "timeout_threshold": 30.0,
                "recovery_timeout": self._cb._recovery_timeout,
                "detection_strategy": "consecutive",
            },
        }


class _CompatConfig:
    """Minimal config object for backward compatibility."""

    def __init__(self, name: str, failure_threshold: int, recovery_timeout: int):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_rate_threshold = 0.5
        self.timeout_threshold = 30.0
        self.success_threshold = 1
        self.window_size = 300
        self.expected_exceptions = (Exception,)


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

_breakers: dict[str, MaverickCircuitBreaker] = {}
_breakers_lock = threading.Lock()


def get_or_create_breaker(name: str) -> MaverickCircuitBreaker:
    """Get or lazily create a circuit breaker for the given service."""
    with _breakers_lock:
        if name not in _breakers:
            cfg = SERVICE_CONFIGS.get(
                name, {"failure_threshold": 5, "recovery_timeout": 60}
            )
            _breakers[name] = MaverickCircuitBreaker(
                name=name,
                failure_threshold=cfg["failure_threshold"],
                recovery_timeout=cfg["recovery_timeout"],
            )
        return _breakers[name]


def register_circuit_breaker(name: str, breaker: MaverickCircuitBreaker) -> None:
    with _breakers_lock:
        _breakers[name] = breaker
        logger.debug("Registered circuit breaker: %s", name)


def get_circuit_breaker(name: str) -> MaverickCircuitBreaker | None:
    return _breakers.get(name)


def get_all_circuit_breakers() -> dict[str, MaverickCircuitBreaker]:
    return _breakers.copy()


def reset_all_circuit_breakers() -> None:
    for b in _breakers.values():
        b.reset()


def get_circuit_breaker_status() -> dict[str, dict[str, Any]]:
    return {name: b.get_status() for name, b in _breakers.items()}


def initialize_circuit_breakers() -> dict[str, MaverickCircuitBreaker]:
    """Initialize all circuit breakers for known services."""
    breakers = {}
    for service_name in SERVICE_CONFIGS:
        try:
            breaker = get_or_create_breaker(service_name)
            breakers[service_name] = breaker
            logger.info("Initialized circuit breaker for %s", service_name)
        except Exception as e:
            logger.error("Failed to initialize CB for %s: %s", service_name, e)
    logger.info("Initialized %d circuit breakers", len(breakers))
    return breakers
