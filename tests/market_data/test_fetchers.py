"""Tests for maverick.market_data.fetchers."""

import httpx
import pandas as pd
import pytest
from pydantic import SecretStr

from maverick.market_data.config import MarketDataSettings
from maverick.market_data.fetchers import (
    MoverFetcher,
    YFinanceFetcher,
    fetch_capital_companion,
)
from maverick.platform.config import HttpSettings
from maverick.platform.http import CircuitOpenError

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


def _counting_async(result):
    calls: list[tuple[str, int]] = []

    async def fn(kind: str, limit: int):
        calls.append((kind, limit))
        return result

    fn.calls = calls  # type: ignore[attr-defined]
    return fn


async def test_mover_no_key_skips_external_tier_uses_finviz():
    external = _counting_async([{"symbol": "SHOULD_NOT_APPEAR"}])
    finviz = _counting_sync([{"symbol": "AAPL"}])
    batch = _counting_sync([{"symbol": "SHOULD_NOT_APPEAR_2"}])

    settings = MarketDataSettings(capital_companion_api_key=None)
    fetcher = MoverFetcher(
        external_client=external,
        finviz_fn=finviz,
        batch_quote_fn=batch,
        settings=settings,
    )

    result = await fetcher.gainers(5)

    assert result == [{"symbol": "AAPL"}]
    assert external.calls == []
    assert finviz.calls == [("gainers", 5)]
    assert batch.calls == []


async def test_mover_finviz_raises_falls_through_to_yfinance_tier():
    finviz = _raising_sync(RuntimeError("finviz down"))
    batch = _counting_sync([{"symbol": "MSFT"}])

    settings = MarketDataSettings(capital_companion_api_key=None)
    fetcher = MoverFetcher(finviz_fn=finviz, batch_quote_fn=batch, settings=settings)

    result = await fetcher.losers(3)

    assert result == [{"symbol": "MSFT"}]
    assert finviz.calls == [("losers", 3)]
    assert batch.calls == [("losers", 3)]


async def test_mover_all_tiers_fail_returns_empty_list():
    external = _counting_async([])
    finviz = _raising_sync(RuntimeError("finviz down"))
    batch = _raising_sync(RuntimeError("yfinance down"))

    settings = MarketDataSettings(capital_companion_api_key=SecretStr("key"))
    fetcher = MoverFetcher(
        external_client=external,
        finviz_fn=finviz,
        batch_quote_fn=batch,
        settings=settings,
    )

    result = await fetcher.most_active(10)

    assert result == []
    assert external.calls == [("most_active", 10)]
    assert finviz.calls == [("most_active", 10)]
    assert batch.calls == [("most_active", 10)]


async def test_mover_tier_order_respected_when_key_present():
    external = _counting_async([{"symbol": "EXTERNAL"}])
    finviz = _counting_sync([{"symbol": "SHOULD_NOT_APPEAR"}])
    batch = _counting_sync([{"symbol": "SHOULD_NOT_APPEAR_2"}])

    settings = MarketDataSettings(capital_companion_api_key=SecretStr("key"))
    fetcher = MoverFetcher(
        external_client=external,
        finviz_fn=finviz,
        batch_quote_fn=batch,
        settings=settings,
    )

    result = await fetcher.gainers(7)

    assert result == [{"symbol": "EXTERNAL"}]
    assert external.calls == [("gainers", 7)]
    assert finviz.calls == []
    assert batch.calls == []


async def test_mover_external_raises_falls_through_even_with_key_present():
    async def failing_external(kind: str, limit: int):
        raise RuntimeError("external down")

    finviz = _counting_sync([{"symbol": "FINVIZ"}])

    settings = MarketDataSettings(capital_companion_api_key=SecretStr("key"))
    fetcher = MoverFetcher(
        external_client=failing_external, finviz_fn=finviz, settings=settings
    )

    result = await fetcher.gainers(4)

    assert result == [{"symbol": "FINVIZ"}]
    assert finviz.calls == [("gainers", 4)]


async def test_mover_no_fns_injected_returns_empty_list():
    settings = MarketDataSettings(capital_companion_api_key=None)
    fetcher = MoverFetcher(settings=settings)

    assert await fetcher.gainers(5) == []
    assert await fetcher.losers(5) == []
    assert await fetcher.most_active(5) == []


# ---------------------------------------------------------------------------
# fetch_capital_companion
# ---------------------------------------------------------------------------


async def test_fetch_capital_companion_returns_list_on_200():
    seen_headers = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.update(request.headers)
        return httpx.Response(200, json=[{"symbol": "AAPL"}, {"symbol": "MSFT"}])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await fetch_capital_companion(
            client, "https://capitalcompanion.ai/gainers", "secret-key"
        )

    assert result == [{"symbol": "AAPL"}, {"symbol": "MSFT"}]
    assert seen_headers.get("x-api-key") == "secret-key"


async def test_fetch_capital_companion_retries_500_then_200():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(500)
        return httpx.Response(200, json=[{"symbol": "GME"}])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await fetch_capital_companion(
            client, "https://capitalcompanion.ai/losers", "secret-key"
        )

    assert result == [{"symbol": "GME"}]
    assert calls == 2


async def test_fetch_capital_companion_non_list_json_returns_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await fetch_capital_companion(
            client, "https://capitalcompanion.ai/most-active", "secret-key"
        )

    assert result == []
