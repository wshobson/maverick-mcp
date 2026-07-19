"""Rule-based signal dispatch. Third-layer sibling: imports config and types.

Ports `VectorBTEngine._generate_signals` and its twelve private
`_*_signals` helper methods from
`maverick_mcp/backtesting/vectorbt_engine.py` (lines ~325-538 and
~1195-1319) as one module-level `generate_signals(frame, strategy_type,
params) -> (entries, exits)` dispatcher plus twelve module-level helper
functions (the `self` parameter is simply dropped; every other line of
signal-generation math is copied verbatim). This is the signal-generation
side of the seam `engine.py`'s module docstring describes: `run_backtest`
takes already-computed `entries`/`exits`, and `optimize_parameters` takes a
`signal_fn: (frame, params) -> (entries, exits)` callable -- `generate_signals`
below has exactly that shape, so `partial(generate_signals, strategy_type=...)`
(or an equivalent closure) is a valid `signal_fn` for Task 7's service layer.

Every one of the 12 `STRATEGY_TEMPLATES` catalog entries (`templates.py`)
has a real, executable branch here -- including `online_learning`,
`regime_aware`, and `ensemble`, whose catalog `code` field is a descriptive
comment (see `templates.py`'s module docstring). Their *catalog* entries
are stubs; their *dispatch* branches are not. In the legacy engine,
`_online_learning_signals`/`_regime_aware_signals`/`_ensemble_signals` are
self-contained pandas/numpy computations (adaptive thresholds off rolling
mean/std, a trending-vs-ranging regime switch between trend-following and
mean-reversion rules, and majority-vote signal combination) -- none of them
instantiate or call into the sophisticated ML strategy classes ported in
Task 6 (`strategies/ml/online_learning.py`'s `OnlineLearningStrategy`,
`strategies/ml/regime_aware.py`'s `RegimeAwareStrategy`,
`strategies/ml/ensemble.py`'s `StrategyEnsemble`). Those ML classes remain
available as separate, more sophisticated `Strategy` subclasses for
whichever future tool wants them (e.g. `run_ml_strategy_backtest`); they
are simply not what `strategy_type="online_learning"` etc. dispatch to
today. This module ports what the legacy dispatch actually executes.

One intentional, behavior-preserving deviation: legacy `_mean_reversion_signals`
returns raw `numpy.ndarray` (from `np.where(...)`) rather than `pandas.Series`
for `entries`/`exits` -- harmless in the legacy engine (which immediately
calls `.astype(bool)`, which both types support, before handing the arrays to
`vbt.Portfolio.from_signals`), but incompatible with this module's and
`engine.py`'s typed `SignalFn = Callable[[DataFrame, dict], tuple[Series,
Series]]` contract. `_mean_reversion_signals` below wraps the identical
boolean arrays in `pd.Series(..., index=close.index)` before returning --
same values, now typed correctly.

Another: legacy `_regime_aware_signals`'s `trend_strength` computation calls
`close.rolling(window).apply(lambda x: (x[-1] - x[0]) / x[0] ...)` with
pandas' default `raw=False`, which passes each window as a `Series` and
triggers a `FutureWarning` on positional integer indexing
(`Series.__getitem__` with positions is deprecated). This port passes
`raw=True` instead, so `x` is a `numpy.ndarray` and `x[-1]`/`x[0]` are
unambiguous positional lookups -- identical numeric output, no warning.
"""

from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt


def generate_signals(
    frame: pd.DataFrame, strategy_type: str, params: dict[str, Any]
) -> tuple[pd.Series, pd.Series]:
    """Dispatch to the signal generator for `strategy_type`.

    Port of `VectorBTEngine._generate_signals`. `sma_cross` and
    `sma_crossover` are accepted as aliases for the same branch, matching
    the legacy `strategy_type in ["sma_cross", "sma_crossover"]` check.

    Raises:
        ValueError: `frame` has no `close` column, or `strategy_type` is
            not one of the 12 known types.
    """
    if "close" not in frame.columns:
        raise ValueError(
            f"Missing 'close' column in price data. Available columns: {list(frame.columns)}"
        )
    close = frame["close"]

    if strategy_type in ("sma_cross", "sma_crossover"):
        return _sma_crossover_signals(close, params)
    elif strategy_type == "rsi":
        return _rsi_signals(close, params)
    elif strategy_type == "macd":
        return _macd_signals(close, params)
    elif strategy_type == "bollinger":
        return _bollinger_bands_signals(close, params)
    elif strategy_type == "momentum":
        return _momentum_signals(close, params)
    elif strategy_type == "ema_cross":
        return _ema_crossover_signals(close, params)
    elif strategy_type == "mean_reversion":
        return _mean_reversion_signals(close, params)
    elif strategy_type == "breakout":
        return _breakout_signals(close, params)
    elif strategy_type == "volume_momentum":
        return _volume_momentum_signals(frame, params)
    elif strategy_type == "online_learning":
        return _online_learning_signals(frame, params)
    elif strategy_type == "regime_aware":
        return _regime_aware_signals(frame, params)
    elif strategy_type == "ensemble":
        return _ensemble_signals(frame, params)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")


def _sma_crossover_signals(
    close: pd.Series, params: dict[str, Any]
) -> tuple[pd.Series, pd.Series]:
    """Port of `VectorBTEngine._sma_crossover_signals`."""
    fast_period = params.get("fast_period", params.get("fast_window", 10))
    slow_period = params.get("slow_period", params.get("slow_window", 20))

    fast_ma: Any = vbt.MA.run(close, fast_period, short_name="fast")
    slow_ma: Any = vbt.MA.run(close, slow_period, short_name="slow")
    fast_sma = fast_ma.ma.squeeze()
    slow_sma = slow_ma.ma.squeeze()

    entries = (fast_sma > slow_sma) & (fast_sma.shift(1) <= slow_sma.shift(1))
    exits = (fast_sma < slow_sma) & (fast_sma.shift(1) >= slow_sma.shift(1))

    return entries, exits


def _rsi_signals(
    close: pd.Series, params: dict[str, Any]
) -> tuple[pd.Series, pd.Series]:
    """Port of `VectorBTEngine._rsi_signals`."""
    period = params.get("period", 14)
    oversold = params.get("oversold", 30)
    overbought = params.get("overbought", 70)

    rsi_ind: Any = vbt.RSI.run(close, period)
    rsi = rsi_ind.rsi.squeeze()

    entries = (rsi < oversold) & (rsi.shift(1) >= oversold)
    exits = (rsi > overbought) & (rsi.shift(1) <= overbought)

    return entries, exits


def _macd_signals(
    close: pd.Series, params: dict[str, Any]
) -> tuple[pd.Series, pd.Series]:
    """Port of `VectorBTEngine._macd_signals`."""
    fast_period = params.get("fast_period", 12)
    slow_period = params.get("slow_period", 26)
    signal_period = params.get("signal_period", 9)

    macd: Any = vbt.MACD.run(
        close,
        fast_window=fast_period,
        slow_window=slow_period,
        signal_window=signal_period,
    )

    macd_line = macd.macd.squeeze()
    signal_line = macd.signal.squeeze()

    entries = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
    exits = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))

    return entries, exits


def _bollinger_bands_signals(
    close: pd.Series, params: dict[str, Any]
) -> tuple[pd.Series, pd.Series]:
    """Port of `VectorBTEngine._bollinger_bands_signals`."""
    period = params.get("period", 20)
    std_dev = params.get("std_dev", 2)

    bb: Any = vbt.BBANDS.run(close, window=period, alpha=std_dev)
    upper = bb.upper.squeeze()
    lower = bb.lower.squeeze()

    entries = (close <= lower) & (close.shift(1) > lower.shift(1))
    exits = (close >= upper) & (close.shift(1) < upper.shift(1))

    return entries, exits


def _momentum_signals(
    close: pd.Series, params: dict[str, Any]
) -> tuple[pd.Series, pd.Series]:
    """Port of `VectorBTEngine._momentum_signals`."""
    lookback = params.get("lookback", 20)
    threshold = params.get("threshold", 0.05)

    returns = close.pct_change(lookback)

    entries = returns > threshold
    exits = returns < -threshold

    return entries, exits


def _ema_crossover_signals(
    close: pd.Series, params: dict[str, Any]
) -> tuple[pd.Series, pd.Series]:
    """Port of `VectorBTEngine._ema_crossover_signals`."""
    fast_period = params.get("fast_period", 12)
    slow_period = params.get("slow_period", 26)

    fast_ema_ind: Any = vbt.MA.run(close, fast_period, ewm=True)
    slow_ema_ind: Any = vbt.MA.run(close, slow_period, ewm=True)
    fast_ema = fast_ema_ind.ma.squeeze()
    slow_ema = slow_ema_ind.ma.squeeze()

    entries = (fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1))
    exits = (fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1))

    return entries, exits


def _mean_reversion_signals(
    close: pd.Series, params: dict[str, Any]
) -> tuple[pd.Series, pd.Series]:
    """Port of `VectorBTEngine._mean_reversion_signals`.

    See module docstring: wraps the legacy `np.where`-derived boolean
    arrays in `pd.Series` before returning (same values, correctly typed).
    """
    ma_period = params.get("ma_period", 20)
    entry_threshold = params.get("entry_threshold", 0.02)
    exit_threshold = params.get("exit_threshold", 0.01)

    ma_ind: Any = vbt.MA.run(close, ma_period)
    ma = ma_ind.ma.squeeze()

    with np.errstate(divide="ignore", invalid="ignore"):
        deviation = np.where(ma != 0, (close - ma) / ma, 0)

    entries = pd.Series(deviation < -entry_threshold, index=close.index)
    exits = pd.Series(deviation > exit_threshold, index=close.index)

    return entries, exits


def _breakout_signals(
    close: pd.Series, params: dict[str, Any]
) -> tuple[pd.Series, pd.Series]:
    """Port of `VectorBTEngine._breakout_signals`."""
    lookback = params.get("lookback", 20)
    exit_lookback = params.get("exit_lookback", 10)

    upper_channel = close.rolling(lookback).max()
    lower_channel = close.rolling(exit_lookback).min()

    entries = close > upper_channel.shift(1)
    exits = close < lower_channel.shift(1)

    return entries, exits


def _volume_momentum_signals(
    data: pd.DataFrame, params: dict[str, Any]
) -> tuple[pd.Series, pd.Series]:
    """Port of `VectorBTEngine._volume_momentum_signals`."""
    momentum_period = params.get("momentum_period", 20)
    volume_period = params.get("volume_period", 20)
    momentum_threshold = params.get("momentum_threshold", 0.05)
    volume_multiplier = params.get("volume_multiplier", 1.5)

    close = data["close"]
    volume = data.get("volume")

    if volume is None:
        returns = close.pct_change(momentum_period)
        entries = returns > momentum_threshold
        exits = returns < -momentum_threshold
        return entries, exits

    returns = close.pct_change(momentum_period)
    avg_volume = volume.rolling(volume_period).mean()
    volume_surge = volume > (avg_volume * volume_multiplier)

    entries = (returns > momentum_threshold) & volume_surge
    exits = (returns < -momentum_threshold) | (volume < avg_volume * 0.8)

    return entries, exits


def _online_learning_signals(
    data: pd.DataFrame, params: dict[str, Any]
) -> tuple[pd.Series, pd.Series]:
    """Port of `VectorBTEngine._online_learning_signals`.

    Simple implementation using momentum with adaptive thresholds -- not a
    delegation to `strategies/ml/online_learning.py`'s
    `OnlineLearningStrategy`. See module docstring.
    """
    lookback = params.get("lookback", 20)
    learning_rate = params.get("learning_rate", 0.01)

    close = data["close"]
    returns = close.pct_change(lookback)

    rolling_mean = returns.rolling(window=lookback).mean()
    rolling_std = returns.rolling(window=lookback).std()

    entry_threshold = rolling_mean + learning_rate * rolling_std
    exit_threshold = rolling_mean - learning_rate * rolling_std

    entries = returns > entry_threshold
    exits = returns < exit_threshold

    entries = entries.fillna(False)
    exits = exits.fillna(False)

    return entries, exits


def _regime_aware_signals(
    data: pd.DataFrame, params: dict[str, Any]
) -> tuple[pd.Series, pd.Series]:
    """Port of `VectorBTEngine._regime_aware_signals`.

    Detects market regime and applies appropriate strategy -- not a
    delegation to `strategies/ml/regime_aware.py`'s `RegimeAwareStrategy`
    (which uses a fitted HMM-style regime detector). See module docstring
    for the `raw=True` deviation in the `trend_strength` rolling-apply.
    """
    regime_window = params.get("regime_window", 50)
    threshold = params.get("threshold", 0.02)

    close = data["close"]

    returns = close.pct_change()
    volatility = returns.rolling(window=regime_window).std()
    trend_strength = close.rolling(window=regime_window).apply(
        lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0, raw=True
    )

    is_trending = trend_strength.abs() > threshold

    sma_short = close.rolling(window=regime_window // 2).mean()
    sma_long = close.rolling(window=regime_window).mean()
    trend_entries = (close > sma_long) & (sma_short > sma_long)
    trend_exits = (close < sma_long) & (sma_short < sma_long)

    bb_upper = sma_long + 2 * volatility
    bb_lower = sma_long - 2 * volatility
    reversion_entries = close < bb_lower
    reversion_exits = close > bb_upper

    entries = (is_trending & trend_entries) | (~is_trending & reversion_entries)
    exits = (is_trending & trend_exits) | (~is_trending & reversion_exits)

    entries = entries.fillna(False)
    exits = exits.fillna(False)

    return entries, exits


def _ensemble_signals(
    data: pd.DataFrame, params: dict[str, Any]
) -> tuple[pd.Series, pd.Series]:
    """Port of `VectorBTEngine._ensemble_signals`.

    Combines SMA/RSI/momentum signals by majority vote -- not a delegation
    to `strategies/ml/ensemble.py`'s `StrategyEnsemble`. See module
    docstring.
    """
    fast_period = params.get("fast_period", 10)
    slow_period = params.get("slow_period", 20)
    rsi_period = params.get("rsi_period", 14)

    close = data["close"]

    fast_sma = close.rolling(window=fast_period).mean()
    slow_sma = close.rolling(window=slow_period).mean()
    sma_entries = (fast_sma > slow_sma) & (fast_sma.shift(1) <= slow_sma.shift(1))
    sma_exits = (fast_sma < slow_sma) & (fast_sma.shift(1) >= slow_sma.shift(1))

    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_entries = (rsi < 30) & (rsi.shift(1) >= 30)
    rsi_exits = (rsi > 70) & (rsi.shift(1) <= 70)

    momentum = close.pct_change(20)
    mom_entries = momentum > 0.05
    mom_exits = momentum < -0.05

    entry_votes = (
        sma_entries.astype(int) + rsi_entries.astype(int) + mom_entries.astype(int)
    )
    exit_votes = sma_exits.astype(int) + rsi_exits.astype(int) + mom_exits.astype(int)

    entries = entry_votes >= 2
    exits = exit_votes >= 2

    entries = entries.fillna(False)
    exits = exits.fillna(False)

    return entries, exits
