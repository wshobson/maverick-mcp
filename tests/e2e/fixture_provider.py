"""Test-only deterministic provider launcher; this is NOT the unmodified CLI.

Only the synchronous external Yahoo/finviz bindings are replaced. Production
fetcher retries, calendars, caches, SQL storage, services, tools, assembly and
CLI lifecycle remain in use. No real market or paid research request is made.
"""

from __future__ import annotations

import json
import threading
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

_CONTROL = Path("fixture-control.json")
_CALLS = Path("provider-calls.jsonl")
_LOCK = threading.Lock()
_CALENDARS = {
    "L": "LSE",
    "T": "JPX",
    "TO": "TSX",
    "AX": "ASX",
    "HK": "HKEX",
    "DE": "XETR",
}


def control() -> dict[str, Any]:
    """Read scenario overrides from the isolated process directory."""
    return json.loads(_CONTROL.read_text()) if _CONTROL.exists() else {}


def record(operation: str, symbol: str, **details: Any) -> None:
    """Serialize synthetic provider calls under a thread lock."""
    with _LOCK, _CALLS.open("a") as stream:
        stream.write(
            json.dumps(
                {"operation": operation, "symbol": symbol, **details}, default=str
            )
            + "\n"
        )


def fixture_history(
    symbol: str, start: Any, end: Any, interval: str = "1d"
) -> pd.DataFrame:
    """Generate exchange-session OHLCV bars with explicit failure modes."""
    record("history", symbol, start=start, end=end, interval=interval)
    controls = control()
    if symbol == "BAD":
        raise ValueError("Synthetic provider failure for BAD")
    suffix = symbol.rpartition(".")[2] if "." in symbol else ""
    calendar = mcal.get_calendar(_CALENDARS.get(suffix, "NYSE"))
    index = calendar.schedule(
        start_date=start, end_date=pd.Timestamp(end).date() - timedelta(days=1)
    ).index
    index = pd.DatetimeIndex(index, name="Date").as_unit("ns")
    elapsed = (index - pd.Timestamp("2018-01-01")).days.to_numpy(dtype=float)
    seed = sum(map(ord, symbol)) % 31
    # Oscillating but positive, nonconstant returns over >3 years support ML,
    # covariance, ATR, and both bullish and bearish screen paths.
    trend = -0.025 if symbol == "BEAR" else 0.035
    close = (
        180
        + seed
        + trend * elapsed
        + 8 * np.sin(elapsed / (13 + seed % 7))
        + 3 * np.sin(elapsed / 4)
    )
    if symbol == "FLAT":
        close = np.full(len(index), 100.0)
    frame = pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 2,
            "Low": close - 2,
            "Close": close,
            "Volume": (
                2_000_000 + seed * 1000 + 400_000 * (1 + np.sin(elapsed / 9))
            ).astype(int),
        },
        index=index,
    )
    frame = frame.round(4)
    mode = controls.get("history_modes", {}).get(symbol, symbol.lower())
    if mode == "empty":
        return frame.iloc[:0]
    if mode == "short":
        return frame.tail(5)
    if mode == "partial" and len(frame) > 4:
        return frame.drop(frame.index[len(frame) // 2])
    if mode == "nonfinite" and len(frame):
        frame.iloc[-1, frame.columns.get_loc("Close")] = np.nan
    return frame


def fixture_info(symbol: str) -> dict[str, Any]:
    """Return deterministic company data and configurable quote prices."""
    record("info", symbol)
    if symbol == "BAD":
        raise ValueError("Synthetic provider failure for BAD")
    if symbol == "EMPTY":
        return {}
    prices = {"AAPL": 150.25, "MSFT": 300.5, "SPY": 500.0, "SUBCENT": 1.006}
    price = control().get("quotes", {}).get(symbol, prices.get(symbol, 100.0))
    return {
        "currentPrice": price,
        "previousClose": price - 1,
        "volume": 2_000_000,
        "longName": f"Synthetic {symbol}",
        "sector": "Technology" if symbol != "MSFT" else "Industrials",
        "industry": "Test fixtures",
        "website": "https://example.test",
        "longBusinessSummary": "Deterministic test data",
        "marketCap": 1_000_000_000,
        "enterpriseValue": 1_100_000_000,
        "sharesOutstanding": 10_000_000,
        "floatShares": 9_000_000,
        "trailingPE": 20,
        "forwardPE": 18,
        "pegRatio": 1.5,
        "priceToBook": 4,
        "priceToSalesTrailing12Months": 3,
        "totalRevenue": 100_000_000,
        "profitMargins": 0.2,
        "operatingMargins": 0.25,
        "returnOnEquity": 0.18,
        "returnOnAssets": 0.1,
        "averageVolume": 2_000_000,
        "averageVolume10days": 2_100_000,
        "beta": 1.1,
        "fiftyTwoWeekHigh": price + 20,
        "fiftyTwoWeekLow": price - 20,
    }


def fixture_download(symbols: list[str], period: str = "1d") -> dict[str, pd.DataFrame]:
    """Generate recent bars for batch market-overview requests."""
    record("download", ",".join(symbols), period=period)
    end = date.today() + timedelta(days=1)
    result = {}
    for symbol in symbols:
        frame = fixture_history(symbol, end - timedelta(days=14), end).tail(2)
        if symbol == "^VIX":
            frame["Close"] = [19.0, 21.0]
        result[symbol] = frame
    return result


def fixture_movers(kind: str, limit: int) -> list[dict[str, Any]]:
    """Return a bounded synthetic mover list for the requested direction."""
    record("movers", kind)
    return [
        {
            "symbol": "AAPL",
            "price": 150.25,
            "change": -2 if kind == "losers" else 2,
            "change_percent": -1.31 if kind == "losers" else 1.35,
            "volume": 2_000_000,
        }
    ][:limit]


def install() -> None:
    """Replace only external Yahoo and finviz fetcher bindings."""
    from maverick.market_data import fetchers

    for name, replacement in (
        ("_default_history_fn", fixture_history),
        ("_default_info_fn", fixture_info),
        ("_default_download_fn", fixture_download),
        ("_finviz_tier", fixture_movers),
    ):
        setattr(fetchers, name, replacement)


if __name__ == "__main__":
    install()
    from maverick.server.app import main

    main()
