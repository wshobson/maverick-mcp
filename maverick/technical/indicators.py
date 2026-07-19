"""Pure pandas/numpy technical indicators.

Every function here is derived, term for term, from pandas-ta's non-TA-Lib
formulas (``talib=False``) so that behavior stays reproducible without a
compiled dependency: see ``scripts/record_indicator_fixtures.py`` for how
the golden values this module is tested against were recorded, and
``tests/technical/test_indicators.py`` for the comparison. This module
imports nothing from pandas_ta -- only pandas and numpy.

Conventions shared by every function:

- Inputs are tz-naive ``pd.Series`` (``high``/``low``/``close``, one price
  per row); outputs are the same length and index as the input, with a
  NaN-headed warmup region matching pandas-ta's shape.
- When there isn't enough history for the requested period, the result is
  entirely NaN rather than a partially-computed series.
"""

import numpy as np
import pandas as pd


def sma(close: pd.Series, period: int) -> pd.Series:
    """Simple moving average: a plain rolling mean, NaN for the first
    ``period - 1`` rows."""
    return close.rolling(window=period, min_periods=period).mean()


def ema(close: pd.Series, period: int) -> pd.Series:
    """Exponential moving average, pandas ``ewm(span=period, adjust=False)``.

    Matches pandas-ta's default (TA-Lib-style) seeding: the first
    ``period - 1`` rows are NaN, the ``period``-th row is seeded with the
    simple average of the first ``period`` closes, and the EMA recursion
    (``y_t = alpha * x_t + (1 - alpha) * y_{t-1}``) runs from there.
    """
    if len(close) < period:
        return pd.Series(np.nan, index=close.index, dtype=float)
    seeded = close.astype(float).copy()
    seeded.iloc[: period - 1] = np.nan
    seeded.iloc[period - 1] = close.iloc[:period].mean()
    return seeded.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative strength index using Wilder smoothing
    (``ewm(alpha=1/period, adjust=False)``) on the gain/loss series.

    A persistently flat series (zero average gain and zero average loss)
    is defined as 50.0 -- the neutral reading -- rather than the NaN that a
    literal 0/0 division would produce.
    """
    if len(close) < period:
        return pd.Series(np.nan, index=close.index, dtype=float)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    total = avg_gain + avg_loss
    safe_total = total.replace(0, np.nan)
    result = 100 * avg_gain / safe_total
    return result.mask(total == 0, 50.0)


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """Moving average convergence/divergence.

    Returns a DataFrame with columns ``macd`` (fast EMA minus slow EMA),
    ``signal`` (EMA of the macd line), and ``histogram`` (macd minus
    signal). Built from this module's own :func:`ema`, so it shares its
    seeding and NaN-warmup behavior.
    """
    if slow < fast:
        fast, slow = slow, fast
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    first_valid = macd_line.first_valid_index()
    if first_valid is None:
        signal_line = pd.Series(np.nan, index=close.index, dtype=float)
    else:
        signal_line = ema(macd_line.loc[first_valid:], signal).reindex(close.index)
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": histogram}
    )


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Average true range: true range smoothed with Wilder's method.

    The first true range row uses ``high - low`` (no prior close to diff
    against); the first ``period`` true ranges are then averaged to seed
    the Wilder recursion, matching pandas-ta's default ATR shape.
    """
    if len(close) < period:
        return pd.Series(np.nan, index=close.index, dtype=float)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    seeded = true_range.astype(float).copy()
    seeded.iloc[: period - 1] = np.nan
    seeded.iloc[period - 1] = true_range.iloc[:period].mean()
    return seeded.ewm(alpha=1 / period, adjust=False).mean()
