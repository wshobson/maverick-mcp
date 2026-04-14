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
    """Backoff delays follow the pattern 0.1 * 2**attempt: 0.1s, then 0.2s."""
    engine.get_historical_data = AsyncMock(
        side_effect=[ConnectionError("1"), ConnectionError("2"), sample_df]
    )

    sleep_mock = AsyncMock()
    with patch("maverick_mcp.backtesting.vectorbt_engine.asyncio.sleep", sleep_mock):
        await engine._fetch_with_retry("AAPL", "2023-01-01", "2023-12-31")

    delays = [call.args[0] for call in sleep_mock.await_args_list]
    assert delays == pytest.approx([0.1, 0.2])


async def test_custom_max_retries_override(engine):
    """Caller can override the default retry count."""
    engine.get_historical_data = AsyncMock(side_effect=ConnectionError("down"))

    with patch("maverick_mcp.backtesting.vectorbt_engine.asyncio.sleep", AsyncMock()):
        with pytest.raises(ConnectionError):
            await engine._fetch_with_retry(
                "AAPL", "2023-01-01", "2023-12-31", max_retries=5
            )

    assert engine.get_historical_data.await_count == 5
