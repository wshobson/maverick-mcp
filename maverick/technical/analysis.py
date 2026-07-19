"""Pure analysis rubrics. Third layer: imports config and types.

Every ``analyze_*`` function below takes a single "prepared" DataFrame --
one row per bar, most-recent bar last -- holding raw OHLCV columns and/or
precomputed indicator columns, plus `TechnicalSettings`. This module never
imports `maverick.technical.indicators`: the "Technical layers are
forward-only" import contract places `indicators` and `analysis` as
independent siblings, both below `service` (verified empirically -- adding
such an import here breaks `lint-imports`). The indicator math that
populates a prepared frame's columns always comes from `indicators.py`; it
is computed and merged onto the frame by the caller (the service tier in
`service.py`, or a test fixture), not by this module.

Expected columns, read only when the corresponding analysis needs them:

- OHLCV (as returned by `maverick.market_data`): ``Close``, ``High``,
  ``Low``, ``Volume``.
- RSI: ``rsi`` (`indicators.rsi`).
- MACD: ``macd``, ``macd_signal``, ``macd_hist`` (`indicators.macd`'s
  ``macd``/``signal``/``histogram`` columns).
- Stochastic: ``stoch_k``, ``stoch_d`` (`indicators.stochastic`'s
  ``k``/``d`` columns).
- Bollinger: ``bb_mid``, ``bb_upper``, ``bb_lower`` (`indicators.bollinger`'s
  ``mid``/``upper``/``lower`` columns).
- Trend: ``sma_short``, ``sma_long``, ``ema`` (`indicators.sma` at
  `settings.sma_short_period`/`settings.sma_long_period`, `indicators.ema`
  at `settings.ema_period`), plus ``rsi``, ``macd``, and ``adx``
  (`indicators.adx`).

Every threshold/labeling decision below is made against the *unrounded*
value, matching the legacy `maverick_mcp.core.technical_analysis` behavior
exactly; values are rounded to 2 decimal places only when stored on the
returned typed model.
"""

import pandas as pd

from maverick.technical.config import TechnicalSettings
from maverick.technical.types import (
    BollingerAnalysis,
    LevelsResult,
    MACDAnalysis,
    RSIAnalysis,
    StochasticAnalysis,
    TrendAnalysis,
    VolumeAnalysis,
)

# Trend-score bands shared by `_trend_direction` and `generate_outlook`'s
# trend contribution -- see `generate_outlook`'s docstring for the fixed
# behavior this replaces.
_TREND_BULLISH_SCORE = 5
_TREND_BEARISH_SCORE = 2


def _round2(value: float) -> float:
    return round(float(value), 2)


def _last(df: pd.DataFrame, column: str) -> float:
    """The last value of `column`, or NaN if the column/frame is absent."""
    if df.empty or column not in df.columns:
        return float("nan")
    return float(df[column].iloc[-1])


def _prev(df: pd.DataFrame, column: str) -> float:
    """The second-to-last value of `column`, or NaN with fewer than 2 rows."""
    if len(df) < 2 or column not in df.columns:
        return float("nan")
    return float(df[column].iloc[-2])


def analyze_rsi(df: pd.DataFrame, settings: TechnicalSettings) -> RSIAnalysis:
    """Label the current RSI reading: overbought above
    `settings.rsi_overbought`, oversold below `settings.rsi_oversold`,
    otherwise bullish above 50 and bearish at or below it."""
    current = _last(df, "rsi")
    if pd.isna(current):
        return RSIAnalysis(
            current=None,
            period=settings.rsi_period,
            signal="unavailable",
            description="RSI data not available (insufficient data points)",
        )

    if current > settings.rsi_overbought:
        signal = "overbought"
    elif current < settings.rsi_oversold:
        signal = "oversold"
    elif current > 50:
        signal = "bullish"
    else:
        signal = "bearish"

    rounded = _round2(current)
    return RSIAnalysis(
        current=rounded,
        period=settings.rsi_period,
        signal=signal,
        description=f"RSI is currently at {rounded}, indicating {signal} conditions.",
    )


def analyze_macd(df: pd.DataFrame, settings: TechnicalSettings) -> MACDAnalysis:
    """Label MACD-vs-signal-line position and histogram sign, plus a
    crossover check that needs >= 2 rows of history."""
    macd_val = _last(df, "macd")
    signal_val = _last(df, "macd_signal")
    hist_val = _last(df, "macd_hist")

    if pd.isna(macd_val) or pd.isna(signal_val) or pd.isna(hist_val):
        return MACDAnalysis(
            macd=None,
            signal_line=None,
            histogram=None,
            indicator_signal="unavailable",
            crossover="unavailable",
            description="MACD data not available (insufficient data points)",
        )

    if macd_val > signal_val and hist_val > 0:
        indicator_signal = "bullish"
    elif macd_val < signal_val and hist_val < 0:
        indicator_signal = "bearish"
    elif macd_val > signal_val and macd_val < 0:
        indicator_signal = "improving"
    elif macd_val < signal_val and macd_val > 0:
        indicator_signal = "weakening"
    else:
        indicator_signal = "neutral"

    crossover = "no recent crossover"
    if len(df) >= 2:
        prev_macd = _prev(df, "macd")
        prev_signal = _prev(df, "macd_signal")
        if pd.notna(prev_macd) and pd.notna(prev_signal):
            if prev_macd <= prev_signal and macd_val > signal_val:
                crossover = "bullish crossover detected"
            elif prev_macd >= prev_signal and macd_val < signal_val:
                crossover = "bearish crossover detected"

    return MACDAnalysis(
        macd=_round2(macd_val),
        signal_line=_round2(signal_val),
        histogram=_round2(hist_val),
        indicator_signal=indicator_signal,
        crossover=crossover,
        description=f"MACD is {indicator_signal} with {crossover}.",
    )


def analyze_stochastic(
    df: pd.DataFrame, settings: TechnicalSettings
) -> StochasticAnalysis:
    """Label %K/%D as overbought/oversold (`settings.stoch_overbought`/
    `settings.stoch_oversold`) or bullish/bearish by relative position,
    plus a crossover check that needs >= 2 rows of history."""
    k = _last(df, "stoch_k")
    d = _last(df, "stoch_d")

    if pd.isna(k) or pd.isna(d):
        return StochasticAnalysis(
            k=None,
            d=None,
            signal="unavailable",
            crossover="unavailable",
            description="Stochastic data not available (insufficient data points)",
        )

    if k > settings.stoch_overbought and d > settings.stoch_overbought:
        signal = "overbought"
    elif k < settings.stoch_oversold and d < settings.stoch_oversold:
        signal = "oversold"
    elif k > d:
        signal = "bullish"
    else:
        signal = "bearish"

    crossover = "no recent crossover"
    if len(df) >= 2:
        prev_k = _prev(df, "stoch_k")
        prev_d = _prev(df, "stoch_d")
        if pd.notna(prev_k) and pd.notna(prev_d):
            if prev_k <= prev_d and k > d:
                crossover = "bullish crossover detected"
            elif prev_k >= prev_d and k < d:
                crossover = "bearish crossover detected"

    return StochasticAnalysis(
        k=_round2(k),
        d=_round2(d),
        signal=signal,
        crossover=crossover,
        description=f"Stochastic Oscillator is {signal} with {crossover}.",
    )


def analyze_bollinger(
    df: pd.DataFrame, settings: TechnicalSettings
) -> BollingerAnalysis:
    """Label price position relative to the bands, plus a 5-bar
    squeeze/expansion read on band width ``(upper - lower) / middle``:
    contracting for 5 strictly-decreasing widths, expanding for 5
    strictly-increasing widths, stable otherwise."""
    current_price = _last(df, "Close")
    upper = _last(df, "bb_upper")
    lower = _last(df, "bb_lower")
    middle = _last(df, "bb_mid")

    if pd.isna(current_price) or pd.isna(upper) or pd.isna(lower) or pd.isna(middle):
        return BollingerAnalysis(
            upper=None,
            middle=None,
            lower=None,
            current_price=None,
            position="unavailable",
            volatility="unavailable",
            description="Bollinger Bands data not available (insufficient data points)",
        )

    if current_price > upper:
        position, signal = "above upper band", "overbought"
    elif current_price < lower:
        position, signal = "below lower band", "oversold"
    elif current_price > middle:
        position, signal = "above middle band", "bullish"
    else:
        position, signal = "below middle band", "bearish"

    volatility = "stable"
    if len(df) >= 5:
        widths: list[float] = []
        for offset in range(-5, 0):
            u = df["bb_upper"].iloc[offset]
            lo = df["bb_lower"].iloc[offset]
            m = df["bb_mid"].iloc[offset]
            if pd.notna(u) and pd.notna(lo) and pd.notna(m) and m != 0:
                widths.append((u - lo) / m)
        if len(widths) == 5:
            if all(widths[i] < widths[i - 1] for i in range(1, 5)):
                volatility = "contracting (potential breakout ahead)"
            elif all(widths[i] > widths[i - 1] for i in range(1, 5)):
                volatility = "expanding (increased volatility)"

    return BollingerAnalysis(
        upper=_round2(upper),
        middle=_round2(middle),
        lower=_round2(lower),
        current_price=_round2(current_price),
        position=position,
        volatility=volatility,
        description=(
            f"Price is {position}, indicating {signal} conditions. "
            f"Volatility is {volatility}."
        ),
    )


def analyze_volume(df: pd.DataFrame, settings: TechnicalSettings) -> VolumeAnalysis:
    """Label the current bar's volume against a 10-bar trailing average
    (or the whole frame's average when fewer than 10 bars exist)."""
    if df.empty or "Volume" not in df.columns:
        return VolumeAnalysis(
            current=None,
            average=None,
            ratio=None,
            description="unavailable",
            signal="unavailable",
        )

    current_volume = df["Volume"].iloc[-1]
    avg_volume = df["Volume"].mean() if len(df) < 10 else df["Volume"].iloc[-10:].mean()

    if pd.isna(current_volume) or pd.isna(avg_volume) or avg_volume == 0:
        return VolumeAnalysis(
            current=None,
            average=None,
            ratio=None,
            description="unavailable",
            signal="unavailable",
        )

    current_volume = float(current_volume)
    avg_volume = float(avg_volume)
    ratio = current_volume / avg_volume

    if ratio > settings.volume_high_ratio:
        description = "above average"
        rising = (
            len(df) >= 2
            and "Close" in df.columns
            and df["Close"].iloc[-1] > df["Close"].iloc[-2]
        )
        signal = (
            "bullish (high volume on up move)"
            if rising
            else "bearish (high volume on down move)"
        )
    elif ratio < settings.volume_low_ratio:
        description = "below average"
        signal = "weak conviction"
    else:
        description = "average"
        signal = "neutral"

    return VolumeAnalysis(
        current=current_volume,
        average=avg_volume,
        ratio=_round2(ratio),
        description=description,
        signal=signal,
    )


def _trend_direction(score: int) -> str:
    """Bucket a 0-7 trend score into the same three-way split
    `generate_outlook` uses for its trend contribution."""
    if score >= _TREND_BULLISH_SCORE:
        return "bullish"
    if score <= _TREND_BEARISH_SCORE:
        return "bearish"
    return "neutral"


def analyze_trend(df: pd.DataFrame, settings: TechnicalSettings) -> TrendAnalysis:
    """The legacy 0-7 trend-strength score: one point each for
    ``close > sma_short``, ``close > ema``, ``ema > sma_short``,
    ``sma_short > sma_long``, ``rsi > 50``, ``macd > 0``, and
    ``adx > settings.adx_trend_threshold`` -- each check contributes only
    when both operands are present (NaN-safe).

    `direction` buckets the score via `_trend_direction`: `score >= 5` is
    "bullish", `score <= 2` is "bearish", 3-4 is "neutral". This includes
    the resting score of 0 for missing/empty data, which reads as
    "bearish" -- a deliberate consequence of applying the threshold rule
    uniformly rather than carving out a "no data" exception.
    """
    close = _last(df, "Close")
    sma_short = _last(df, "sma_short")
    sma_long = _last(df, "sma_long")
    ema = _last(df, "ema")
    rsi = _last(df, "rsi")
    macd = _last(df, "macd")
    adx = _last(df, "adx")

    score = 0
    if pd.notna(close) and pd.notna(sma_short) and close > sma_short:
        score += 1
    if pd.notna(close) and pd.notna(ema) and close > ema:
        score += 1
    if pd.notna(ema) and pd.notna(sma_short) and ema > sma_short:
        score += 1
    if pd.notna(sma_short) and pd.notna(sma_long) and sma_short > sma_long:
        score += 1
    if pd.notna(rsi) and rsi > 50:
        score += 1
    if pd.notna(macd) and macd > 0:
        score += 1
    if pd.notna(adx) and adx > settings.adx_trend_threshold:
        score += 1

    return TrendAnalysis(
        score=score,
        direction=_trend_direction(score),
        adx=None if pd.isna(adx) else _round2(adx),
    )


def support_resistance(df: pd.DataFrame, settings: TechnicalSettings) -> LevelsResult:
    """The legacy simple algorithm, verbatim: a `settings.sr_lookback`-bar
    high/low window (or the whole frame when shorter) plus synthetic
    +-5%/+-10% levels around the current close. Real pivot detection is a
    future feature -- this is deliberately simple, not parity with
    professional S/R analysis."""
    if df.empty or not {"High", "Low", "Close"}.issubset(df.columns):
        return LevelsResult(support=[], resistance=[])

    lookback = settings.sr_lookback
    window = df.iloc[-lookback:] if len(df) >= lookback else df
    min_low = float(window["Low"].min())
    max_high = float(window["High"].max())
    close = float(df["Close"].iloc[-1])

    support = sorted(
        {round(min_low, 2), round(close * 0.95, 2), round(close * 0.90, 2)}
    )
    resistance = sorted(
        {round(max_high, 2), round(close * 1.05, 2), round(close * 1.10, 2)}
    )
    return LevelsResult(support=support, resistance=resistance)


def generate_outlook(
    trend: TrendAnalysis,
    rsi: RSIAnalysis,
    macd: MACDAnalysis,
    stoch: StochasticAnalysis,
) -> str:
    """Overall outlook from weighted bullish/bearish signal counts across
    trend/RSI/MACD/stochastic, banded strongly/moderately/neutral (legacy
    thresholds: >= 4 signals is "strongly", a plain majority is
    "moderately", otherwise "neutral").

    FIXED vs. legacy: `maverick_mcp.core.technical_analysis.generate_outlook`
    took `trend` as a string and compared it against the literals
    `"uptrend"`/`"downtrend"`, but every production caller
    (`api/routers/technical.py`, `api/routers/technical_enhanced.py`) passed
    `analyze_trend(df)`'s **integer** score straight through --
    `str(some_int) == "uptrend"` is always `False`, so the trend branch
    never contributed to a production outlook. Here `trend` is the typed
    `TrendAnalysis` and its score contributes directly, using the same two
    thresholds `_trend_direction` buckets on: `score >= 5` counts as 2
    bullish signals, `score <= 2` counts as 2 bearish signals (mirroring
    the legacy `+= 2` weighting), and 3-4 contributes nothing.
    """
    bullish = 0
    bearish = 0

    if trend.score >= _TREND_BULLISH_SCORE:
        bullish += 2
    elif trend.score <= _TREND_BEARISH_SCORE:
        bearish += 2

    if rsi.signal in ("bullish", "oversold"):
        bullish += 1
    elif rsi.signal in ("bearish", "overbought"):
        bearish += 1

    if (
        macd.indicator_signal == "bullish"
        or macd.crossover == "bullish crossover detected"
    ):
        bullish += 1
    elif (
        macd.indicator_signal == "bearish"
        or macd.crossover == "bearish crossover detected"
    ):
        bearish += 1

    if stoch.signal in ("bullish", "oversold"):
        bullish += 1
    elif stoch.signal in ("bearish", "overbought"):
        bearish += 1

    if bullish >= 4:
        return "strongly bullish"
    if bullish > bearish:
        return "moderately bullish"
    if bearish >= 4:
        return "strongly bearish"
    if bearish > bullish:
        return "moderately bearish"
    return "neutral"
