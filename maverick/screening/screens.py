"""Pure screen rubrics. Third layer: imports config and types."""

import pandas as pd

from maverick.screening.config import ScreeningSettings
from maverick.screening.types import ScreeningResult, ScreenName
from maverick.technical.indicators import atr, macd, rsi, sma

# Human-readable labels for every flag key each screen can produce, keyed by
# screen name. `_build_reason` is the single shared reason-string generator
# for all three screens (closing the legacy duplicate reason-generator
# debt); the label maps are the only per-screen data it needs.
_FLAG_LABELS: dict[ScreenName, dict[str, str]] = {
    "bullish": {
        "close_above_sma50": "close above SMA50",
        "close_above_sma150": "close above SMA150",
        "close_above_sma200": "close above SMA200",
        "ma_aligned": "moving averages aligned (SMA50 > SMA150 > SMA200)",
        "volume_surge": "volume surge",
        "rsi_not_overbought": "RSI not overbought",
    },
    "bearish": {
        "close_below_sma50": "close below SMA50",
        "close_below_sma200": "close below SMA200",
        "rsi_oversold": "RSI oversold",
        "rsi_weak": "RSI weak (below 40)",
        "macd_bearish": "MACD below signal",
        "volume_decline": "high-volume down day",
        "atr_contraction": "ATR contraction",
    },
    "supply_demand": {
        "close_above_sma150": "close above SMA150",
        "close_above_sma200": "close above SMA200",
        "sma150_above_sma200": "SMA150 above SMA200",
        "sma200_rising": "SMA200 trending up",
        "sma50_above_sma150": "SMA50 above SMA150",
        "sma50_above_sma200": "SMA50 above SMA200",
        "close_above_sma50": "close above SMA50",
        "volume_surge": "volume above 1.2x average",
        "near_52w_high": "within 25% of the 252-day high",
    },
}


def _build_reason(screen: ScreenName, flags: dict[str, bool]) -> str:
    """Render a human-readable reason string from a screen's fired flags.

    The single reason-generator shared by all three screens below, closing
    the legacy debt of three near-identical duplicate implementations.
    """
    labels = _FLAG_LABELS[screen]
    fired = [labels[name] for name in labels if flags.get(name)]
    title = screen.replace("_", " ")
    if not fired:
        return f"{title} screen: no criteria fired"
    return f"{title} screen: " + ", ".join(fired)


def _last_date(df: pd.DataFrame) -> str:
    """Render the frame's final index entry as an ISO date string."""
    timestamp: pd.Timestamp = df.index[-1]
    naive = timestamp.tz_localize(None) if timestamp.tzinfo is not None else timestamp
    return naive.date().isoformat()


def score_bullish(
    symbol: str, df: pd.DataFrame, settings: ScreeningSettings
) -> ScreeningResult | None:
    """Legacy Maverick momentum rubric.

    +25 each for close above SMA50/SMA150/SMA200, +25 for moving-average
    alignment (SMA50 > SMA150 > SMA200), +10 for volume above
    ``settings.volume_surge_multiplier`` x its 30-day average, +10 for
    RSI(14) below ``settings.rsi_overbought``. Qualifies at
    ``combined_score >= settings.bullish_min_score``.

    The legacy screen also awarded a separate "+15 Uptrend pattern" point
    from a pattern-detection heuristic; that heuristic isn't ported here,
    and its intent -- price trending up through a well-ordered stack of
    moving averages -- is folded into the MA-alignment criterion above.
    """
    if len(df) < settings.min_history_days:
        return None

    close_s = df["Close"]
    volume_s = df["Volume"]
    sma50 = sma(close_s, 50).iloc[-1]
    sma150 = sma(close_s, 150).iloc[-1]
    sma200 = sma(close_s, 200).iloc[-1]
    rsi14 = rsi(close_s, 14).iloc[-1]
    avg_volume_30d = volume_s.rolling(window=30, min_periods=30).mean().iloc[-1]
    close = float(close_s.iloc[-1])
    volume = float(volume_s.iloc[-1])

    flags = {
        "close_above_sma50": bool(close > sma50),
        "close_above_sma150": bool(close > sma150),
        "close_above_sma200": bool(close > sma200),
        "ma_aligned": bool(sma50 > sma150 > sma200),
        "volume_surge": bool(
            volume > settings.volume_surge_multiplier * avg_volume_30d
        ),
        "rsi_not_overbought": bool(rsi14 < settings.rsi_overbought),
    }
    combined_score = (
        25 * flags["close_above_sma50"]
        + 25 * flags["close_above_sma150"]
        + 25 * flags["close_above_sma200"]
        + 25 * flags["ma_aligned"]
        + 10 * flags["volume_surge"]
        + 10 * flags["rsi_not_overbought"]
    )
    if combined_score < settings.bullish_min_score:
        return None

    indicators: dict[str, int | float | None] = {
        "close": close,
        "sma50": float(sma50),
        "sma150": float(sma150),
        "sma200": float(sma200),
        "rsi14": float(rsi14),
        "volume": volume,
        "avg_volume_30d": float(avg_volume_30d),
    }
    return ScreeningResult(
        symbol=symbol,
        screen="bullish",
        date_analyzed=_last_date(df),
        close=close,
        combined_score=combined_score,
        momentum_score=None,
        indicators=indicators,
        flags=flags,
        reason=_build_reason("bullish", flags),
    )


def score_bearish(
    symbol: str, df: pd.DataFrame, settings: ScreeningSettings
) -> ScreeningResult | None:
    """Legacy bear-market rubric.

    +20 close below SMA50, +20 close below SMA200, +15 if RSI(14) is
    below ``settings.rsi_oversold`` else +10 if RSI(14) is below 40, +15
    if the MACD line is below its signal line, +20 for a high-volume down
    day (volume above ``settings.volume_decline_multiplier`` x its 30-day
    average, on a day where close is below the prior close), +10 for
    ATR(14) contracting below ``settings.atr_contraction_multiplier`` x
    its 20-day average. Qualifies at ``combined_score >=
    settings.bear_min_score``.
    """
    if len(df) < settings.min_history_days:
        return None

    close_s = df["Close"]
    volume_s = df["Volume"]
    sma50 = sma(close_s, 50).iloc[-1]
    sma200 = sma(close_s, 200).iloc[-1]
    rsi14 = rsi(close_s, 14).iloc[-1]
    macd_frame = macd(close_s)
    macd_line = macd_frame["macd"].iloc[-1]
    macd_signal = macd_frame["signal"].iloc[-1]
    atr14_s = atr(df["High"], df["Low"], close_s, 14)
    atr_now = atr14_s.iloc[-1]
    atr_avg_20d = atr14_s.rolling(window=20, min_periods=20).mean().iloc[-1]
    avg_volume_30d = volume_s.rolling(window=30, min_periods=30).mean().iloc[-1]
    close = float(close_s.iloc[-1])
    prior_close = float(close_s.iloc[-2])
    volume = float(volume_s.iloc[-1])
    down_day = close < prior_close

    rsi_oversold = bool(rsi14 < settings.rsi_oversold)
    flags = {
        "close_below_sma50": bool(close < sma50),
        "close_below_sma200": bool(close < sma200),
        "rsi_oversold": rsi_oversold,
        "rsi_weak": bool((not rsi_oversold) and rsi14 < 40),
        "macd_bearish": bool(macd_line < macd_signal),
        "volume_decline": bool(
            volume > settings.volume_decline_multiplier * avg_volume_30d and down_day
        ),
        "atr_contraction": bool(
            atr_now < settings.atr_contraction_multiplier * atr_avg_20d
        ),
    }
    combined_score = (
        20 * flags["close_below_sma50"]
        + 20 * flags["close_below_sma200"]
        + 15 * flags["rsi_oversold"]
        + 10 * flags["rsi_weak"]
        + 15 * flags["macd_bearish"]
        + 20 * flags["volume_decline"]
        + 10 * flags["atr_contraction"]
    )
    if combined_score < settings.bear_min_score:
        return None

    indicators: dict[str, int | float | None] = {
        "close": close,
        "sma50": float(sma50),
        "sma200": float(sma200),
        "rsi14": float(rsi14),
        "macd": float(macd_line),
        "macd_signal": float(macd_signal),
        "atr14": float(atr_now),
        "atr_avg_20d": float(atr_avg_20d),
        "volume": volume,
        "avg_volume_30d": float(avg_volume_30d),
        "prior_close": prior_close,
    }
    return ScreeningResult(
        symbol=symbol,
        screen="bearish",
        date_analyzed=_last_date(df),
        close=close,
        combined_score=combined_score,
        momentum_score=None,
        indicators=indicators,
        flags=flags,
        reason=_build_reason("bearish", flags),
    )


def score_supply_demand(
    symbol: str, df: pd.DataFrame, settings: ScreeningSettings
) -> ScreeningResult | None:
    """Legacy supply/demand breakout rubric.

    A boolean gate requiring ALL of: close above SMA150 and SMA200,
    SMA150 above SMA200, SMA200 rising over the last 22 bars (today's
    SMA200 above SMA200 from 22 bars ago), SMA50 above SMA150 and SMA200,
    and close above SMA50.

    When gated in, ``combined_score`` is the breakout_strength: a 50-point
    base, +25 if volume is above 1.2x its 30-day average, +25 if close is
    within 25% of the highest high over the trailing 252 bars (or all
    available history when shorter) -- so a qualifying row scores 50-100.

    ``momentum_score`` is a simple, documented proxy: the percent distance
    of close above SMA200, scaled by 5 and clipped to [0, 100]:
    ``min(100, max(0, (close / sma200 - 1) * 100 * 5))``.
    """
    if len(df) < settings.min_history_days:
        return None

    close_s = df["Close"]
    volume_s = df["Volume"]
    sma50_s = sma(close_s, 50)
    sma150_s = sma(close_s, 150)
    sma200_s = sma(close_s, 200)
    sma50 = sma50_s.iloc[-1]
    sma150 = sma150_s.iloc[-1]
    sma200 = sma200_s.iloc[-1]
    sma200_22_ago = sma200_s.iloc[-22]
    avg_volume_30d = volume_s.rolling(window=30, min_periods=30).mean().iloc[-1]
    high_252d = df["High"].rolling(window=252, min_periods=1).max().iloc[-1]
    close = float(close_s.iloc[-1])
    volume = float(volume_s.iloc[-1])

    flags = {
        "close_above_sma150": bool(close > sma150),
        "close_above_sma200": bool(close > sma200),
        "sma150_above_sma200": bool(sma150 > sma200),
        "sma200_rising": bool(sma200 > sma200_22_ago),
        "sma50_above_sma150": bool(sma50 > sma150),
        "sma50_above_sma200": bool(sma50 > sma200),
        "close_above_sma50": bool(close > sma50),
    }
    if not all(flags.values()):
        return None

    volume_surge = bool(volume > 1.2 * avg_volume_30d)
    near_52w_high = bool(close > high_252d * 0.75)
    flags["volume_surge"] = volume_surge
    flags["near_52w_high"] = near_52w_high

    breakout_strength = 50 + 25 * volume_surge + 25 * near_52w_high
    momentum_score = min(100.0, max(0.0, (close / sma200 - 1) * 100 * 5))

    indicators: dict[str, int | float | None] = {
        "close": close,
        "sma50": float(sma50),
        "sma150": float(sma150),
        "sma200": float(sma200),
        "sma200_22_bars_ago": float(sma200_22_ago),
        "volume": volume,
        "avg_volume_30d": float(avg_volume_30d),
        "high_252d": float(high_252d),
    }
    return ScreeningResult(
        symbol=symbol,
        screen="supply_demand",
        date_analyzed=_last_date(df),
        close=close,
        combined_score=breakout_strength,
        momentum_score=momentum_score,
        indicators=indicators,
        flags=flags,
        reason=_build_reason("supply_demand", flags),
    )
