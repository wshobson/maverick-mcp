"""Tests for maverick.market_data.fetchers."""

import asyncio
import sys

import pandas as pd
import pytest

from maverick.market_data.fetchers import (
    MoverFetcher,
    YFinanceFetcher,
    _build_yfinance_tier,
    build_mover_fetcher,
)
from maverick.platform.config import HttpSettings
from maverick.platform.http import CircuitOpenError, get_breaker, reset_breakers

# ---------------------------------------------------------------------------
# YFinanceFetcher
# ---------------------------------------------------------------------------


def _tz_aware_frame() -> pd.DataFrame:
    index = pd.date_range("2026-07-13", periods=3, freq="D", tz="America/New_York")
    return pd.DataFrame(
        {
            "Open": [1.0, 2.0, 3.0],
            "High": [1.5, 2.5, 3.5],
            "Low": [0.5, 1.5, 2.5],
            "Close": [1.2, 2.2, 3.2],
            "Volume": [100, 200, 300],
        },
        index=index,
    )


async def test_history_strips_timezone_and_preserves_columns():
    frame = _tz_aware_frame()

    def fake_history(symbol, start, end, interval="1d"):
        assert symbol == "AAPL"
        return frame

    fetcher = YFinanceFetcher(history_fn=fake_history)
    result = await fetcher.history("AAPL", "2026-07-13", "2026-07-16")

    assert result.index.tz is None
    assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert list(result["Close"]) == [1.2, 2.2, 3.2]


async def test_history_passes_through_already_naive_frame():
    index = pd.date_range("2026-07-13", periods=2, freq="D")
    frame = pd.DataFrame({"Open": [1.0, 2.0]}, index=index)

    fetcher = YFinanceFetcher(history_fn=lambda *a, **k: frame)
    result = await fetcher.history("AAPL", "2026-07-13", "2026-07-16")

    assert result.index.tz is None
    assert list(result["Open"]) == [1.0, 2.0]


async def test_batch_history_strips_timezone_per_symbol():
    frames = {"AAPL": _tz_aware_frame(), "MSFT": _tz_aware_frame()}
    calls = []

    def fake_download(symbols, period="1d"):
        calls.append((tuple(symbols), period))
        return frames

    fetcher = YFinanceFetcher(download_fn=fake_download)
    result = await fetcher.batch_history(["AAPL", "MSFT"], period="5d")

    assert set(result) == {"AAPL", "MSFT"}
    assert result["AAPL"].index.tz is None
    assert result["MSFT"].index.tz is None
    assert calls == [(("AAPL", "MSFT"), "5d")]


async def test_info_returns_injected_dict():
    fetcher = YFinanceFetcher(
        info_fn=lambda symbol: {"symbol": symbol, "sector": "Tech"}
    )
    result = await fetcher.info("AAPL")

    assert result == {"symbol": "AAPL", "sector": "Tech"}


async def test_breaker_opens_after_repeated_fetcher_failures():
    calls = 0

    def failing_history(symbol, start, end, interval="1d"):
        nonlocal calls
        calls += 1
        raise RuntimeError("yfinance down")

    settings = HttpSettings(breaker_failure_threshold=2, breaker_recovery_seconds=60.0)
    fetcher = YFinanceFetcher(
        history_fn=failing_history,
        breaker_name="test-yfinance-breaker-opens",
        http_settings=settings,
    )

    for _ in range(2):
        with pytest.raises(RuntimeError):
            await fetcher.history("AAPL", "2026-07-13", "2026-07-16")

    calls_after_two_failures = calls
    assert calls_after_two_failures > 0

    with pytest.raises(CircuitOpenError):
        await fetcher.history("AAPL", "2026-07-13", "2026-07-16")

    # The breaker short-circuited before invoking the fetch function again.
    assert calls == calls_after_two_failures


# ---------------------------------------------------------------------------
# MoverFetcher
# ---------------------------------------------------------------------------


def _counting_sync(result):
    calls: list[tuple[str, int]] = []

    def fn(kind: str, limit: int):
        calls.append((kind, limit))
        return result

    fn.calls = calls  # type: ignore[attr-defined]
    return fn


def _raising_sync(exc: Exception):
    calls: list[tuple[str, int]] = []

    def fn(kind: str, limit: int):
        calls.append((kind, limit))
        raise exc

    fn.calls = calls  # type: ignore[attr-defined]
    return fn


async def test_mover_finviz_result_skips_yfinance_tier():
    finviz = _counting_sync([{"symbol": "AAPL"}])
    batch = _counting_sync([{"symbol": "SHOULD_NOT_APPEAR"}])

    fetcher = MoverFetcher(finviz_fn=finviz, batch_quote_fn=batch)

    result = await fetcher.gainers(5)

    assert result == [{"symbol": "AAPL"}]
    assert finviz.calls == [("gainers", 5)]
    assert batch.calls == []


async def test_mover_finviz_raises_falls_through_to_yfinance_tier():
    finviz = _raising_sync(RuntimeError("finviz down"))
    batch = _counting_sync([{"symbol": "MSFT"}])

    fetcher = MoverFetcher(finviz_fn=finviz, batch_quote_fn=batch)

    result = await fetcher.losers(3)

    assert result == [{"symbol": "MSFT"}]
    assert finviz.calls == [("losers", 3)]
    assert batch.calls == [("losers", 3)]


async def test_mover_finviz_empty_falls_through_to_yfinance_tier():
    finviz = _counting_sync([])
    batch = _counting_sync([{"symbol": "MSFT"}])

    fetcher = MoverFetcher(finviz_fn=finviz, batch_quote_fn=batch)

    result = await fetcher.gainers(4)

    assert result == [{"symbol": "MSFT"}]
    assert finviz.calls == [("gainers", 4)]
    assert batch.calls == [("gainers", 4)]


async def test_mover_all_tiers_fail_returns_empty_list():
    finviz = _raising_sync(RuntimeError("finviz down"))
    batch = _raising_sync(RuntimeError("yfinance down"))

    fetcher = MoverFetcher(finviz_fn=finviz, batch_quote_fn=batch)

    result = await fetcher.most_active(10)

    assert result == []
    assert finviz.calls == [("most_active", 10)]
    assert batch.calls == [("most_active", 10)]


async def test_mover_no_fns_injected_returns_empty_list():
    fetcher = MoverFetcher()

    assert await fetcher.gainers(5) == []
    assert await fetcher.losers(5) == []
    assert await fetcher.most_active(5) == []


# ---------------------------------------------------------------------------
# build_mover_fetcher
# ---------------------------------------------------------------------------


def test_build_mover_fetcher_binds_both_tiers():
    fetcher = build_mover_fetcher(YFinanceFetcher())

    assert isinstance(fetcher, MoverFetcher)
    assert fetcher._finviz_fn is not None
    assert fetcher._batch_quote_fn is not None


def test_build_mover_fetcher_never_imports_finvizfinance_at_construction():
    sys.modules.pop("finvizfinance", None)
    sys.modules.pop("finvizfinance.screener.overview", None)

    build_mover_fetcher(YFinanceFetcher())

    assert "finvizfinance" not in sys.modules


def test_yfinance_tier_calls_download_fn_directly_not_batch_history():
    """Tier 2's closure calls the raw sync download_fn -- never `yf.batch_history`.

    Regression coverage for the nested-event-loop deadlock: the old
    implementation ran `asyncio.run(yf.batch_history(...))` inside this
    closure. `batch_history` is monkeypatched to raise so any accidental
    reintroduction of that call is caught immediately (not just via the
    slower concurrency test below).
    """
    download_calls: list[tuple[tuple[str, ...], str]] = []

    def fake_download(symbols, period="1d"):
        download_calls.append((tuple(symbols), period))
        return {}

    async def _must_not_be_called(*args, **kwargs):
        raise AssertionError("tier 2 must not call yf.batch_history")

    yf = YFinanceFetcher(download_fn=fake_download)
    yf.batch_history = _must_not_be_called  # type: ignore[method-assign]

    tier2 = _build_yfinance_tier(yf._download_fn)
    result = tier2("gainers", 5)

    assert result == []
    assert len(download_calls) == 1
    assert download_calls[0][1] == "2d"


def test_build_mover_fetcher_tier2_binds_yf_download_fn_by_default():
    download_calls: list[str] = []

    def fake_download(symbols, period="1d"):
        download_calls.append(period)
        return {}

    yf = YFinanceFetcher(download_fn=fake_download)
    fetcher = build_mover_fetcher(yf)

    # Call tier 2's bound callable directly (not through the full tier
    # chain, which would otherwise hit the real finviz tier first).
    result = fetcher._batch_quote_fn("losers", 3)

    assert result == []
    assert download_calls == ["2d"]


def test_build_mover_fetcher_tier2_explicit_download_fn_overrides_yf_binding():
    yf_calls: list[str] = []

    def yf_default_download(symbols, period="1d"):
        yf_calls.append("yf")
        return {}

    override_calls: list[str] = []

    def override_download(symbols, period="1d"):
        override_calls.append("override")
        return {}

    yf = YFinanceFetcher(download_fn=yf_default_download)
    fetcher = build_mover_fetcher(yf, download_fn=override_download)

    fetcher._batch_quote_fn("gainers", 5)

    assert override_calls == ["override"]
    assert yf_calls == []


async def test_yfinance_tier_completes_while_yfinance_breaker_lock_is_held():
    """Regression test for the tier-2 nested-event-loop deadlock.

    Before the fix, `_build_yfinance_tier`'s closure ran `asyncio.run(yf.
    batch_history(...))` inside the worker thread `MoverFetcher.
    _from_batch_quote` schedules it onto (via `asyncio.to_thread`). That
    nested loop's attempt to acquire the shared "yfinance" breaker's
    `asyncio.Lock` -- while this test's main loop already holds it --
    would hang forever: an `asyncio.Lock`'s waiter `Future` binds to
    whichever loop first hits its contended `acquire()` path, and a
    `Future` resolved from a different thread's loop doesn't wake a
    selector blocked in that other thread. Post-fix, tier 2 never touches
    the breaker at all, so this completes immediately regardless of the
    held lock.
    """
    breaker_name = "test-yfinance-breaker-tier2-deadlock"
    reset_breakers()
    breaker = get_breaker(breaker_name)
    await breaker._lock.acquire()
    try:

        def fake_download(symbols, period="1d"):
            return {}

        yf = YFinanceFetcher(download_fn=fake_download, breaker_name=breaker_name)
        tier2 = _build_yfinance_tier(yf._download_fn)

        result = await asyncio.wait_for(
            asyncio.to_thread(tier2, "most_active", 5), timeout=5.0
        )

        assert result == []
    finally:
        breaker._lock.release()
