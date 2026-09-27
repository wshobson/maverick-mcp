"""Tests for maverick.market_data.fetchers."""

import asyncio
import sys
from types import ModuleType
from typing import cast

import pandas as pd
import pytest

from maverick.market_data import fetchers
from maverick.market_data.fetchers import (
    MoverFetcher,
    YFinanceFetcher,
    _build_yfinance_tier,
    _finviz_tier,
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

    assert cast(pd.DatetimeIndex, result.index).tz is None
    assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert list(result["Close"]) == [1.2, 2.2, 3.2]


async def test_history_passes_through_already_naive_frame():
    index = pd.date_range("2026-07-13", periods=2, freq="D")
    frame = pd.DataFrame({"Open": [1.0, 2.0]}, index=index)

    fetcher = YFinanceFetcher(history_fn=lambda *a, **k: frame)
    result = await fetcher.history("AAPL", "2026-07-13", "2026-07-16")

    assert cast(pd.DatetimeIndex, result.index).tz is None
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
    assert cast(pd.DatetimeIndex, result["AAPL"].index).tz is None
    assert cast(pd.DatetimeIndex, result["MSFT"].index).tz is None
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
# YFinanceFetcher: class-share symbols (Yahoo spells BRK.B as BRK-B)
# ---------------------------------------------------------------------------

# What yfinance returns for a symbol Yahoo does not know (probed for BRK.B
# and TWTR): a stub of metadata keys with no price.
_PRICELESS_INFO = {"quoteType": "NONE", "language": "en-US", "maxAge": 86400}


def _history_by_symbol(frames: dict[str, pd.DataFrame]):
    calls: list[str] = []

    def fake_history(symbol, start, end, interval="1d"):
        calls.append(symbol)
        return frames.get(symbol, pd.DataFrame())

    return fake_history, calls


def _info_by_symbol(infos: dict[str, dict]):
    calls: list[str] = []

    def fake_info(symbol):
        calls.append(symbol)
        return infos.get(symbol, _PRICELESS_INFO)

    return fake_info, calls


async def test_history_retries_empty_class_share_symbol_with_dash():
    fake_history, calls = _history_by_symbol({"BRK-B": _tz_aware_frame()})
    fetcher = YFinanceFetcher(history_fn=fake_history)

    result = await fetcher.history("BRK.B", "2026-07-13", "2026-07-16")

    assert calls == ["BRK.B", "BRK-B"]
    assert list(result["Close"]) == [1.2, 2.2, 3.2]


async def test_history_keeps_dotted_symbol_whose_first_fetch_has_data():
    fake_history, calls = _history_by_symbol({"VOD.L": _tz_aware_frame()})
    fetcher = YFinanceFetcher(history_fn=fake_history)

    result = await fetcher.history("VOD.L", "2026-07-13", "2026-07-16")

    assert calls == ["VOD.L"]
    assert len(result) == 3


@pytest.mark.parametrize("symbol", ["7203.T", "SHOP.TO", "AAPL"])
async def test_history_never_rewrites_a_symbol_that_is_not_a_class_share(symbol):
    fake_history, calls = _history_by_symbol({})
    fetcher = YFinanceFetcher(history_fn=fake_history)

    result = await fetcher.history(symbol, "2026-07-13", "2026-07-16")

    assert calls == [symbol]
    assert result.empty


async def test_info_retries_priceless_class_share_symbol_with_dash():
    fake_info, calls = _info_by_symbol({"BRK-B": {"currentPrice": 505.48}})
    fetcher = YFinanceFetcher(info_fn=fake_info)

    result = await fetcher.info("BRK.B")

    assert calls == ["BRK.B", "BRK-B"]
    assert result == {"currentPrice": 505.48}


async def test_info_keeps_dotted_symbol_whose_first_fetch_has_a_price():
    fake_info, calls = _info_by_symbol({"VOD.L": {"currentPrice": 125.8}})
    fetcher = YFinanceFetcher(info_fn=fake_info)

    result = await fetcher.info("VOD.L")

    assert calls == ["VOD.L"]
    assert result == {"currentPrice": 125.8}


async def test_info_keeps_first_result_when_dash_spelling_has_no_price_either():
    first = {"quoteType": "EQUITY", "longName": "Priceless Class A"}
    fake_info, calls = _info_by_symbol({"ABC.A": first})
    fetcher = YFinanceFetcher(info_fn=fake_info)

    result = await fetcher.info("ABC.A")

    assert calls == ["ABC.A", "ABC-A"]
    assert result == first


# ---------------------------------------------------------------------------
# MoverFetcher
# ---------------------------------------------------------------------------


def _counting_sync(result):
    calls: list[tuple[str, int]] = []

    def fn(kind: str, limit: int):
        calls.append((kind, limit))
        return result

    fn.calls = calls  # ty: ignore[unresolved-attribute]  # function attribute
    return fn


def _raising_sync(exc: Exception):
    calls: list[tuple[str, int]] = []

    def fn(kind: str, limit: int):
        calls.append((kind, limit))
        raise exc

    fn.calls = calls  # ty: ignore[unresolved-attribute]  # function attribute
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


async def test_mover_all_tiers_empty_returns_empty_list():
    finviz = _counting_sync([])
    batch = _counting_sync([])

    fetcher = MoverFetcher(finviz_fn=finviz, batch_quote_fn=batch)

    result = await fetcher.most_active(3)

    assert result == []
    assert finviz.calls == [("most_active", 3)]
    assert batch.calls == [("most_active", 3)]


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
# _finviz_tier
# ---------------------------------------------------------------------------


def _finviz_frame(**overrides: list) -> pd.DataFrame:
    """Shaped like finvizfinance 1.3.0's `Overview().screener_view()`.

    Rows come back in ticker order, and the change column is headed
    "Change %" and holds the page text ("5.73%"), not a number. The fake
    screener ignores filters, so gainers and losers share one frame.
    """
    columns: dict[str, list] = {
        "Ticker": ["AAA", "BBB", "CCC", "DDD"],
        "Company": ["A Corp", "B Corp", "C Corp", "D Corp"],
        "Sector": ["Technology"] * 4,
        "Industry": ["Software"] * 4,
        "Country": ["USA"] * 4,
        "Market Cap": [1.0e9, 2.0e9, 3.0e9, 4.0e9],
        "P/E": [10.0, 20.0, 30.0, 40.0],
        "Price": [10.573, 11.25, 9.69, 9.2],
        "Change %": ["5.73%", "12.50%", "-3.10%", "-8.00%"],
        "Volume": [4.0e6, 1.0e6, 9.0e6, 2.0e6],
    }
    columns.update(overrides)
    return pd.DataFrame(columns)


@pytest.fixture
def serve_finviz(monkeypatch: pytest.MonkeyPatch):
    """Serve a frame from a stand-in `finvizfinance.screener.overview`."""

    def _serve(frame: pd.DataFrame | None) -> None:
        class FakeOverview:
            def set_filter(self, filters_dict: dict[str, str]) -> None:
                pass

            def screener_view(self) -> pd.DataFrame | None:
                return frame

        module = ModuleType("finvizfinance.screener.overview")
        monkeypatch.setattr(module, "Overview", FakeOverview, raising=False)
        monkeypatch.setitem(sys.modules, "finvizfinance.screener.overview", module)

    return _serve


def test_finviz_tier_ranks_gainers_by_change_percent(serve_finviz):
    serve_finviz(_finviz_frame())

    rows = _finviz_tier("gainers", 2)

    assert [row["symbol"] for row in rows] == ["BBB", "AAA"]
    assert [row["change_percent"] for row in rows] == pytest.approx([12.5, 5.73])


def test_finviz_tier_ranks_losers_most_negative_first(serve_finviz):
    serve_finviz(_finviz_frame())

    rows = _finviz_tier("losers", 2)

    assert [row["symbol"] for row in rows] == ["DDD", "CCC"]
    assert [row["change_percent"] for row in rows] == pytest.approx([-8.0, -3.1])


def test_finviz_tier_ranks_most_active_by_volume(serve_finviz):
    serve_finviz(_finviz_frame())

    rows = _finviz_tier("most_active", 3)

    assert [row["symbol"] for row in rows] == ["CCC", "AAA", "DDD"]


def test_finviz_tier_derives_change_from_price_and_percent(serve_finviz):
    serve_finviz(_finviz_frame())

    rows = {row["symbol"]: row for row in _finviz_tier("gainers", 4)}

    # 11.25 after +12.5% means a prior close of 10.00; 9.20 after -8% too.
    assert rows["BBB"]["change"] == pytest.approx(1.25)
    assert rows["DDD"]["change"] == pytest.approx(-0.8)
    assert rows["BBB"]["price"] == pytest.approx(11.25)
    assert rows["BBB"]["volume"] == pytest.approx(1.0e6)


def test_finviz_tier_reads_a_legacy_numeric_change_column(serve_finviz):
    # finvizfinance runs a column headed "Change" through `number_covert`,
    # which turns "5.73%" into the fraction 0.0573.
    frame = _finviz_frame(Change=[0.0573, 0.125, -0.031, -0.08]).drop(
        columns=["Change %"]
    )
    serve_finviz(frame)

    rows = _finviz_tier("gainers", 2)

    assert [row["symbol"] for row in rows] == ["BBB", "AAA"]
    assert [row["change_percent"] for row in rows] == pytest.approx([12.5, 5.73])


def test_finviz_tier_leaves_out_a_row_it_cannot_rank(serve_finviz):
    serve_finviz(_finviz_frame(**{"Change %": ["5.73%", "-", "-3.10%", "-8.00%"]}))

    rows = _finviz_tier("gainers", 4)

    assert [row["symbol"] for row in rows] == ["AAA", "CCC", "DDD"]


async def test_finviz_tier_with_nothing_rankable_falls_through_to_yfinance(
    serve_finviz,
):
    serve_finviz(_finviz_frame(**{"Change %": ["-", "-", "-", "-"]}))
    batch = _counting_sync([{"symbol": "MSFT"}])

    fetcher = MoverFetcher(finviz_fn=_finviz_tier, batch_quote_fn=batch)

    assert await fetcher.gainers(5) == [{"symbol": "MSFT"}]


async def test_finviz_tier_without_a_change_column_falls_through_to_yfinance(
    serve_finviz,
):
    serve_finviz(_finviz_frame().drop(columns=["Change %"]))
    batch = _counting_sync([{"symbol": "MSFT"}])

    fetcher = MoverFetcher(finviz_fn=_finviz_tier, batch_quote_fn=batch)

    assert await fetcher.gainers(5) == [{"symbol": "MSFT"}]
    assert batch.calls == [("gainers", 5)]


def test_finviz_tier_returns_empty_for_an_empty_screen(serve_finviz):
    serve_finviz(None)

    assert _finviz_tier("losers", 5) == []


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ("5.73%", 5.73),
        ("-3.1%", -3.1),
        (" 0.00% ", 0.0),
        (0.0573, 5.73),
        ("-", None),
        (float("nan"), None),
        (None, None),
    ],
)
def test_finviz_percent_reads_page_text_and_finvizfinance_fractions(
    cell: object, expected: float | None
) -> None:
    assert fetchers._finviz_percent(cell) == pytest.approx(expected)


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
    yf.batch_history = _must_not_be_called  # ty: ignore[invalid-assignment]  # instance method patch

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
    assert fetcher._batch_quote_fn is not None
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

    assert fetcher._batch_quote_fn is not None
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


@pytest.mark.parametrize(
    ("info", "expected"),
    [
        ({"currentPrice": float("nan"), "regularMarketPrice": 12.5}, 12.5),
        ({"currentPrice": float("inf")}, None),
        ({"currentPrice": -3.0}, None),
        ({"currentPrice": 0, "regularMarketPrice": 9.0}, 9.0),
    ],
)
def test_info_price_accepts_only_a_finite_positive_price(
    info: dict[str, float], expected: float | None
) -> None:
    assert fetchers.info_price(info) == expected
