"""Record pandas-ta golden values for ``maverick/technical/indicators.py``.

This script is the ONLY place in the repository allowed to import
``pandas_ta``. It builds two deterministic OHLCV fixtures (a 300-row seeded
random walk and a 60-row constant series), runs pandas-ta's reference
indicators -- sma, ema, rsi, atr, macd, bbands, stoch, adx -- against them
with the compiled TA-Lib backend disabled (``talib=False``), and writes the
inputs plus the last 200 output values to
``tests/technical/fixtures/indicator_goldens.json``.

Run once to (re)generate the fixture:

    uv run python scripts/record_indicator_fixtures.py

The pure-python implementations in ``maverick/technical/indicators.py`` are
hand-derived to match pandas-ta's non-TA-Lib formulas exactly (see that
module's docstrings). This script is never imported by package or test code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandas_ta as ta

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "technical"
    / "fixtures"
    / "indicator_goldens.json"
)
RANDOM_WALK_ROWS = 300
CONSTANT_ROWS = 60
TAIL = 200
SEED = 42


def _build_random_walk(rows: int = RANDOM_WALK_ROWS, seed: int = SEED) -> pd.DataFrame:
    """A plausible daily-bar OHLCV walk: seeded, reproducible, always high >= low > 0."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0003, scale=0.012, size=rows)
    close = 100.0 * np.cumprod(1.0 + returns)
    open_ = np.empty(rows)
    open_[0] = close[0] * (1 - returns[0] / 2)
    open_[1:] = close[:-1]
    spread = np.abs(rng.normal(loc=0.004, scale=0.002, size=rows)) * close
    high = np.maximum(open_, close) + spread
    low = np.clip(np.minimum(open_, close) - spread, 0.01, None)
    volume = rng.integers(1_000_000, 8_000_000, size=rows)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


def _build_constant(rows: int = CONSTANT_ROWS) -> pd.DataFrame:
    """A flat 60-row frame: unchanging close (drives the RSI 0/0 edge case)."""
    return pd.DataFrame(
        {
            "open": [100.0] * rows,
            "high": [101.0] * rows,
            "low": [99.0] * rows,
            "close": [100.0] * rows,
            "volume": [1_000_000] * rows,
        }
    )


def _series_to_json(series: pd.Series) -> list[float | None]:
    return [None if pd.isna(v) else float(v) for v in series.to_numpy()]


def _record_frame(frame: pd.DataFrame, tail: int) -> dict[str, Any]:
    close, high, low = frame["close"], frame["high"], frame["low"]

    sma_10 = ta.sma(close, length=10, talib=False)
    sma_50 = ta.sma(close, length=50, talib=False)
    ema_21 = ta.ema(close, length=21, talib=False)
    rsi_14 = ta.rsi(close, length=14, talib=False)
    atr_14 = ta.atr(high, low, close, length=14, talib=False)
    macd_df = ta.macd(close, fast=12, slow=26, signal=9, talib=False)
    macd_line, histogram, signal_line = (
        macd_df.iloc[:, 0],
        macd_df.iloc[:, 1],
        macd_df.iloc[:, 2],
    )
    bbands_df = ta.bbands(close, length=20, std=2.0, talib=False)
    bb_lower, bb_mid, bb_upper = (
        bbands_df.iloc[:, 0],
        bbands_df.iloc[:, 1],
        bbands_df.iloc[:, 2],
    )
    stoch_df = ta.stoch(high, low, close, k=14, d=3, smooth_k=3, talib=False)
    stoch_k, stoch_d = stoch_df.iloc[:, 0], stoch_df.iloc[:, 1]
    # NOTE: pandas-ta's adx() does not forward talib=False to its internal
    # ATR sub-call, so with the compiled TA-Lib backend installed (as it is
    # here), the ADX warmup uses TA-Lib's classic Wilder ATR seeding (first
    # value at row `length`, averaging TR[1:length+1]) rather than
    # pandas-ta's own presma ATR (first value at row `length - 1`). See
    # maverick/technical/indicators.py's adx() docstring for the matching
    # pure-pandas derivation.
    adx_df = ta.adx(high, low, close, length=14, talib=False)
    adx_14 = adx_df["ADX_14"]

    return {
        "input": {
            "open": _series_to_json(frame["open"]),
            "high": _series_to_json(frame["high"]),
            "low": _series_to_json(frame["low"]),
            "close": _series_to_json(frame["close"]),
            "volume": _series_to_json(frame["volume"]),
        },
        "tail": tail,
        "expected": {
            "sma_10": _series_to_json(sma_10.tail(tail)),
            "sma_50": _series_to_json(sma_50.tail(tail)),
            "ema_21": _series_to_json(ema_21.tail(tail)),
            "rsi_14": _series_to_json(rsi_14.tail(tail)),
            "atr_14": _series_to_json(atr_14.tail(tail)),
            "macd_12_26_9": {
                "macd": _series_to_json(macd_line.tail(tail)),
                "signal": _series_to_json(signal_line.tail(tail)),
                "histogram": _series_to_json(histogram.tail(tail)),
            },
            "bbands_20_2.0": {
                "mid": _series_to_json(bb_mid.tail(tail)),
                "upper": _series_to_json(bb_upper.tail(tail)),
                "lower": _series_to_json(bb_lower.tail(tail)),
            },
            "stoch_14_3_3": {
                "k": _series_to_json(stoch_k.tail(tail)),
                "d": _series_to_json(stoch_d.tail(tail)),
            },
            "adx_14": _series_to_json(adx_14.tail(tail)),
        },
    }


def main() -> None:
    random_walk = _build_random_walk()
    constant = _build_constant()

    goldens = {
        "random_walk": _record_frame(random_walk, TAIL),
        "constant": _record_frame(constant, CONSTANT_ROWS),
    }

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(goldens, indent=2) + "\n")
    print(f"Wrote {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
