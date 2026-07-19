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


def bollinger(close: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    """Bollinger bands: an SMA midline plus/minus a rolling standard
    deviation multiple.

    Returns a DataFrame with columns ``mid`` (the ``length``-period SMA),
    ``upper`` (``mid + std * rolling_std``), and ``lower``
    (``mid - std * rolling_std``). The rolling standard deviation uses
    ``ddof=1`` (pandas' default, sample stdev) -- this matches pandas-ta's
    ``bbands()`` default, which only switches to ``ddof=0`` when explicitly
    requested.
    """
    mid = close.rolling(window=length, min_periods=length).mean()
    rolling_std = close.rolling(window=length, min_periods=length).std(ddof=1)
    upper = mid + std * rolling_std
    lower = mid - std * rolling_std
    return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower})


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k: int = 14,
    d: int = 3,
    smooth_k: int = 3,
) -> pd.DataFrame:
    """Stochastic oscillator: %K (the close's position within the
    ``k``-period high/low range, smoothed by ``smooth_k``) and %D (a
    ``d``-period SMA of %K).

    Returns a DataFrame with columns ``k`` and ``d``. Matches pandas-ta's
    ``stoch()``: if the ``k``-period high/low range is ever exactly zero
    anywhere in the series, a machine-epsilon offset is added to the whole
    range series (not just the zero entries) before dividing, mirroring
    pandas-ta's ``non_zero_range`` helper.
    """
    lowest_low = low.rolling(window=k, min_periods=k).min()
    highest_high = high.rolling(window=k, min_periods=k).max()
    price_range = highest_high - lowest_low
    if (price_range == 0).any():
        price_range = price_range + np.finfo(float).eps
    raw_k = 100 * (close - lowest_low) / price_range
    k_line = raw_k.rolling(window=smooth_k, min_periods=smooth_k).mean()
    d_line = k_line.rolling(window=d, min_periods=d).mean()
    return pd.DataFrame({"k": k_line, "d": d_line})


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14
) -> pd.Series:
    """Average directional index: the Wilder-smoothed strength of the
    dominant directional movement.

    Built from Wilder-smoothed +DM/-DM (``rma``, i.e.
    ``ewm(alpha=1/length, adjust=False)`` with no extra seeding -- the first
    valid +DM/-DM row stands as its own initial value) divided by a
    Wilder-smoothed average true range, then a Wilder-smoothed DX
    (``100 * |+DM - -DM| / (+DM + -DM)``).

    The ATR term here intentionally does NOT match this module's own
    :func:`atr`: pandas-ta's ``adx()`` calls its internal ATR helper without
    forwarding ``talib=False``, so when the compiled TA-Lib backend is
    installed (as it is in this project's dev environment), that inner ATR
    silently uses TA-Lib's classic Wilder seeding instead of pandas-ta's own
    presma ATR. TA-Lib's convention needs one more bar of warmup: the first
    ATR value sits at row ``length`` (not ``length - 1``) and is the mean of
    the true-range values at rows ``1..length`` (row 0 has no prior close,
    so it is excluded from both the seed and the NaN-warmup count). This is
    the golden this function was recorded against -- see
    ``scripts/record_indicator_fixtures.py``.
    """
    if len(close) < length + 1:
        return pd.Series(np.nan, index=close.index, dtype=float)

    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    seeded_tr = true_range.astype(float).copy()
    seeded_tr.iloc[:length] = np.nan
    seeded_tr.iloc[length] = true_range.iloc[1 : length + 1].mean()
    atr_ = seeded_tr.ewm(alpha=1 / length, adjust=False).mean()

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move

    scalar = 100.0
    k = scalar / atr_
    plus_di = k * plus_dm.ewm(alpha=1 / length, adjust=False).mean()
    minus_di = k * minus_dm.ewm(alpha=1 / length, adjust=False).mean()
    dx = scalar * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / length, adjust=False).mean()
