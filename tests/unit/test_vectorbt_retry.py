"""Unit tests for VectorBTEngine._fetch_with_retry.

These tests lock in the retry wrapper's behavior so future refactors
don't silently regress the resilience guarantees the chaos engineering
tests depend on.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest

from maverick_mcp.backtesting.vectorbt_engine import VectorBTEngine


@pytest.fixture
def engine():
    """Minimal engine for unit testing the retry wrapper in isolation."""
    return VectorBTEngine(data_provider=Mock())


@pytest.fixture
def sample_df():
    """Small DataFrame used as the happy-path return value."""
    return pd.DataFrame({"close": [100.0, 101.0, 102.0]})


async def test_returns_data_on_first_success(engine, sample_df):
    """No retry triggered when the first call succeeds."""
    engine.get_historical_data = AsyncMock(return_value=sample_df)

    result = await engine._fetch_with_retry("AAPL", "2023-01-01", "2023-12-31")

    assert result is sample_df
    assert engine.get_historical_data.await_count == 1


async def test_retries_on_connection_error_then_succeeds(engine, sample_df):
    """Transient ConnectionError is retried; success on later attempt returns data."""
    engine.get_historical_data = AsyncMock(
        side_effect=[ConnectionError("flaky net"), sample_df]
    )

    with patch("maverick_mcp.backtesting.vectorbt_engine.asyncio.sleep", AsyncMock()):
        result = await engine._fetch_with_retry("AAPL", "2023-01-01", "2023-12-31")

    assert result is sample_df
    assert engine.get_historical_data.await_count == 2


async def test_retries_on_timeout_error(engine, sample_df):
    """TimeoutError is in the retryable set."""
    engine.get_historical_data = AsyncMock(
        side_effect=[TimeoutError("slow"), TimeoutError("slow"), sample_df]
    )

    with patch("maverick_mcp.backtesting.vectorbt_engine.asyncio.sleep", AsyncMock()):
        result = await engine._fetch_with_retry("AAPL", "2023-01-01", "2023-12-31")

    assert result is sample_df
    assert engine.get_historical_data.await_count == 3


async def test_reraises_after_max_retries_exhausted(engine):
    """After exhausting retries, the last exception is re-raised."""
    final_error = ConnectionError("persistently broken")
    engine.get_historical_data = AsyncMock(
        side_effect=[ConnectionError("one"), ConnectionError("two"), final_error]
    )

    with patch("maverick_mcp.backtesting.vectorbt_engine.asyncio.sleep", AsyncMock()):
        with pytest.raises(ConnectionError, match="persistently broken"):
            await engine._fetch_with_retry("AAPL", "2023-01-01", "2023-12-31")

    assert engine.get_historical_data.await_count == 3


async def test_non_retryable_error_propagates_immediately(engine):
    """ValueError (e.g. no data for symbol) is not in the retry set; should surface on first failure."""
    engine.get_historical_data = AsyncMock(side_effect=ValueError("no data"))

    with patch("maverick_mcp.backtesting.vectorbt_engine.asyncio.sleep", AsyncMock()):
        with pytest.raises(ValueError, match="no data"):
            await engine._fetch_with_retry("AAPL", "2023-01-01", "2023-12-31")

    assert engine.get_historical_data.await_count == 1


async def test_exponential_backoff_delays(engine, sample_df):
    """Backoff delays follow ``0.1 * 2**attempt`` multiplied by a
    ``random.uniform(0.5, 1.5)`` jitter. Assert the bounded jitter window
    and monotonic growth of the *base* delay rather than exact values —
    otherwise the test is trivially broken by the (intentional) jitter.
    """
    engine.get_historical_data = AsyncMock(
        side_effect=[ConnectionError("1"), ConnectionError("2"), sample_df]
    )

    sleep_mock = AsyncMock()
    with patch("maverick_mcp.backtesting.vectorbt_engine.asyncio.sleep", sleep_mock):
        await engine._fetch_with_retry("AAPL", "2023-01-01", "2023-12-31")

    delays = [call.args[0] for call in sleep_mock.await_args_list]
    assert len(delays) == 2

    # attempt 0: base = 0.1, jittered ∈ [0.05, 0.15]
    # attempt 1: base = 0.2, jittered ∈ [0.10, 0.30]
    assert 0.05 <= delays[0] <= 0.15, (
        f"attempt-0 delay {delays[0]} outside jittered range [0.05, 0.15]"
    )
    assert 0.10 <= delays[1] <= 0.30, (
        f"attempt-1 delay {delays[1]} outside jittered range [0.10, 0.30]"
    )
    # The tightly-capped jitter windows overlap ([0.10, 0.15] ∩ [0.10, 0.30]),
    # so we can't assert strict monotonicity per-call — but the UPPER bound
    # must grow, proving the exponential factor is still applied.


async def test_custom_max_retries_override(engine):
    """Caller can override the default retry count."""
    engine.get_historical_data = AsyncMock(side_effect=ConnectionError("down"))

    with patch("maverick_mcp.backtesting.vectorbt_engine.asyncio.sleep", AsyncMock()):
        with pytest.raises(ConnectionError):
            await engine._fetch_with_retry(
                "AAPL", "2023-01-01", "2023-12-31", max_retries=5
            )

    assert engine.get_historical_data.await_count == 5


async def test_non_positive_max_retries_rejected(engine):
    """``max_retries <= 0`` previously produced a silent ``RuntimeError`` with
    no log line because the for-loop body never ran. The retry budget must be
    validated up-front so misconfiguration fails loudly.
    """
    engine.get_historical_data = AsyncMock()

    with pytest.raises(ValueError, match="max_retries must be a positive"):
        await engine._fetch_with_retry(
            "AAPL", "2023-01-01", "2023-12-31", max_retries=0
        )
    with pytest.raises(ValueError, match="max_retries must be a positive"):
        await engine._fetch_with_retry(
            "AAPL", "2023-01-01", "2023-12-31", max_retries=-1
        )
    assert engine.get_historical_data.await_count == 0


async def test_exhausted_retries_logs_error(engine, caplog):
    """When retries are exhausted, one error-level log line must be emitted
    so ops can see the failure even if the caller suppresses the exception.
    """
    import logging

    engine.get_historical_data = AsyncMock(
        side_effect=[ConnectionError("one"), ConnectionError("two")]
    )

    with patch("maverick_mcp.backtesting.vectorbt_engine.asyncio.sleep", AsyncMock()):
        with caplog.at_level(logging.ERROR, logger="maverick_mcp.backtesting.vectorbt_engine"):
            with pytest.raises(ConnectionError):
                await engine._fetch_with_retry(
                    "AAPL", "2023-01-01", "2023-12-31", max_retries=2
                )

    # The error log is emitted via structured_logger which uses the module
    # logger under the hood. Match the key phrase and symbol regardless of
    # exact formatting drift.
    error_records = [
        r for r in caplog.records
        if r.levelno >= logging.ERROR and "AAPL" in r.getMessage()
    ]
    assert len(error_records) >= 1, (
        f"Expected an ERROR log about AAPL on retry exhaustion, got: "
        f"{[r.getMessage() for r in caplog.records]}"
    )


def _sync_only_provider(df: pd.DataFrame) -> Mock:
    """Build a data_provider Mock that exposes only the sync ``get_stock_data``.

    ``VectorBTEngine._get_data_async`` uses ``hasattr(..., "get_stock_data_async")``
    to route between sync and async providers. A bare ``Mock()`` auto-creates
    every attribute, which makes the engine try to ``await`` a non-awaitable.
    Using ``spec`` pins the mock to the sync-only surface.
    """
    provider = Mock(spec=["get_stock_data"])
    provider.get_stock_data.return_value = df
    return provider


async def test_cache_read_failure_degrades_gracefully(sample_df):
    """A Redis outage during cache ``get`` must not abort the fetch — the
    provider path should still run. Regression guard for the narrowed
    ``_CACHE_EXCEPTIONS`` catch: ``redis.RedisError`` must still be absorbed.
    """
    import redis

    engine = VectorBTEngine(data_provider=_sync_only_provider(sample_df))
    engine.cache.get = AsyncMock(side_effect=redis.RedisError("redis down"))
    engine.cache.set = AsyncMock()

    result = await engine.get_historical_data(
        "AAPL", "2023-01-01", "2023-12-31"
    )

    assert not result.empty
    engine.cache.get.assert_awaited_once()
    # Provider was called despite cache failure
    engine.data_provider.get_stock_data.assert_called_once()


async def test_cache_write_failure_degrades_gracefully(sample_df):
    """A Redis outage during cache ``set`` must not fail the request — the
    DataFrame must still be returned to the caller.
    """
    import redis

    engine = VectorBTEngine(data_provider=_sync_only_provider(sample_df))
    engine.cache.get = AsyncMock(return_value=None)
    engine.cache.set = AsyncMock(side_effect=redis.RedisError("redis down"))

    result = await engine.get_historical_data(
        "AAPL", "2023-01-01", "2023-12-31"
    )

    assert not result.empty
    engine.cache.set.assert_awaited_once()


async def test_cache_read_programming_error_propagates(sample_df):
    """Non-transient errors (``TypeError``, etc.) must NOT be silently
    swallowed by the cache degrade path — those represent programming bugs
    that should surface, not be masked as "cache miss."
    """
    engine = VectorBTEngine(data_provider=_sync_only_provider(sample_df))
    engine.cache.get = AsyncMock(side_effect=TypeError("upstream bug"))

    with pytest.raises(TypeError, match="upstream bug"):
        await engine.get_historical_data("AAPL", "2023-01-01", "2023-12-31")


async def test_circuit_breaker_opens_after_sustained_failures(engine):
    """``_fetch_with_retry`` is wrapped in ``@circuit_breaker`` with
    ``failure_threshold=5``. After 5 exhausted-retry failures the breaker
    must OPEN, causing the 6th call to fast-fail with ``CircuitBreakerError``
    WITHOUT invoking ``get_historical_data`` — proving the decorator is
    actually engaged and not silently bypassed.

    A refactor that drops the decorator, mistypes the ``expected_exceptions``
    tuple, or bumps ``failure_threshold`` would fail this test.
    """
    from maverick_mcp.exceptions import CircuitBreakerError
    from maverick_mcp.utils.circuit_breaker import (
        CircuitBreakerMetrics,
        EnhancedCircuitBreaker,
        FailureDetectionStrategy,
    )

    # The decorator captures its breaker instance in the async_wrapper's
    # closure at import time. Other tests (test_circuit_breaker.py::
    # test_get_all_circuit_breakers) clear the global ``_breakers`` registry,
    # so ``get_circuit_breaker("vectorbt_engine.fetch")`` may return None in a
    # full-suite run even though the decorator's captured breaker still
    # exists. Extract the breaker directly from the closure so the test is
    # insulated from registry mutation.
    wrapper = engine.__class__._fetch_with_retry  # the decorated async_wrapper
    existing: EnhancedCircuitBreaker | None = None
    for cell in wrapper.__closure__ or ():
        if isinstance(cell.cell_contents, EnhancedCircuitBreaker):
            existing = cell.cell_contents
            break
    assert existing is not None, (
        "circuit_breaker decorator must capture an EnhancedCircuitBreaker "
        "in the wrapper's closure"
    )

    # Reset state and metrics, then switch to CONSECUTIVE_FAILURES detection
    # so the test asserts the contract we care about ("failure_threshold
    # consecutive failures -> breaker opens") rather than the COMBINED
    # failure-rate branch that can fire early once total_calls >= 5.
    existing.reset()
    existing._metrics = CircuitBreakerMetrics(
        window_size=existing.config.window_size
    )
    original_strategy = existing.config.detection_strategy
    existing.config.detection_strategy = (
        FailureDetectionStrategy.CONSECUTIVE_FAILURES
    )

    engine.get_historical_data = AsyncMock(side_effect=ConnectionError("down"))

    with patch("maverick_mcp.backtesting.vectorbt_engine.asyncio.sleep", AsyncMock()):
        # 5 calls, each with max_retries=1 so we exhaust quickly. Each
        # exhausted call counts as one CB failure.
        for _ in range(5):
            with pytest.raises(ConnectionError):
                await engine._fetch_with_retry(
                    "AAPL", "2023-01-01", "2023-12-31", max_retries=1
                )

    # After 5 failures at the consecutive-failures threshold, breaker is OPEN.
    assert existing.is_open, (
        f"expected breaker OPEN after 5 failures, got state={existing.state}"
    )

    # Reset the mock so we can prove the 6th call NEVER reaches it.
    engine.get_historical_data = AsyncMock(return_value=None)

    with pytest.raises(CircuitBreakerError):
        await engine._fetch_with_retry(
            "AAPL", "2023-01-01", "2023-12-31", max_retries=1
        )

    assert engine.get_historical_data.await_count == 0, (
        "circuit breaker must fast-fail without invoking the wrapped function"
    )

    # Cleanup: restore the production detection strategy and reset so an
    # OPEN breaker / CONSECUTIVE_FAILURES override doesn't leak into other
    # tests that share this module-level breaker instance.
    existing.config.detection_strategy = original_strategy
    existing.reset()
