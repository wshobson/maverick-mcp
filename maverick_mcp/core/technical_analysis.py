"""
Technical analysis functions for Maverick-MCP.

This module contains functions for performing technical analysis on financial data,
including calculating indicators, analyzing trends, and generating trading signals.

DISCLAIMER: All technical analysis functions in this module are for educational
purposes only. Technical indicators are mathematical calculations based on historical
data and do not predict future price movements. Past performance does not guarantee
future results. Always conduct thorough research and consult with qualified financial
professionals before making investment decisions.
"""

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import pandas_ta as ta

from maverick_mcp.config.technical_constants import TECHNICAL_CONFIG

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("maverick_mcp.technical_analysis")


def _get_column_case_insensitive(df: pd.DataFrame, column_name: str) -> str | None:
    """
    Get the actual column name from the dataframe in a case-insensitive way.

    Args:
        df: DataFrame to search
        column_name: Name of the column to find (case-insensitive)

    Returns:
        The actual column name if found, None otherwise
    """
    if column_name in df.columns:
        return column_name

    column_name_lower = column_name.lower()
    for col in df.columns:
        if col.lower() == column_name_lower:
            return col
    return None


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the dataframe

    Args:
        df: DataFrame with OHLCV price data

    Returns:
        DataFrame with added technical indicators
    """
    # Ensure column names are lowercase
    df = df.copy()
    df.columns = [col.lower() for col in df.columns]

    # Use pandas_ta for all indicators with configurable parameters
    # EMA
    df["ema_21"] = ta.ema(df["close"], length=TECHNICAL_CONFIG.EMA_PERIOD)
    # SMA
    df["sma_50"] = ta.sma(df["close"], length=TECHNICAL_CONFIG.SMA_SHORT_PERIOD)
    df["sma_200"] = ta.sma(df["close"], length=TECHNICAL_CONFIG.SMA_LONG_PERIOD)
    # RSI
    df["rsi"] = ta.rsi(df["close"], length=TECHNICAL_CONFIG.RSI_PERIOD)
    # MACD
    macd = ta.macd(
        df["close"],
        fast=TECHNICAL_CONFIG.MACD_FAST_PERIOD,
        slow=TECHNICAL_CONFIG.MACD_SLOW_PERIOD,
        signal=TECHNICAL_CONFIG.MACD_SIGNAL_PERIOD,
    )
    if macd is not None and not macd.empty:
        df["macd_12_26_9"] = macd["MACD_12_26_9"]
        df["macds_12_26_9"] = macd["MACDs_12_26_9"]
        df["macdh_12_26_9"] = macd["MACDh_12_26_9"]
    else:
        df["macd_12_26_9"] = np.nan
        df["macds_12_26_9"] = np.nan
        df["macdh_12_26_9"] = np.nan
    # Bollinger Bands
    bbands = ta.bbands(df["close"], length=20, std=2.0)
    if bbands is not None and not bbands.empty:
        resolved_columns = _resolve_bollinger_columns(bbands.columns)
        if resolved_columns:
            mid_col, upper_col, lower_col = resolved_columns
            df["sma_20"] = bbands[mid_col]
            df["bbu_20_2.0"] = bbands[upper_col]
            df["bbl_20_2.0"] = bbands[lower_col]
        else:
            logger.warning(
                "Bollinger Bands columns missing expected names: %s",
                list(bbands.columns),
            )
            df["sma_20"] = np.nan
            df["bbu_20_2.0"] = np.nan
            df["bbl_20_2.0"] = np.nan
    else:
        df["sma_20"] = np.nan
        df["bbu_20_2.0"] = np.nan
        df["bbl_20_2.0"] = np.nan
    df["stdev"] = df["close"].rolling(window=20).std()
    # ATR
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    # Stochastic Oscillator
    stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3, smooth_k=3)
    if stoch is not None and not stoch.empty:
        df["stochk_14_3_3"] = stoch["STOCHk_14_3_3"]
        df["stochd_14_3_3"] = stoch["STOCHd_14_3_3"]
    else:
        df["stochk_14_3_3"] = np.nan
        df["stochd_14_3_3"] = np.nan
    # ADX
    adx = ta.adx(df["high"], df["low"], df["close"], length=14)
    if adx is not None and not adx.empty:
        df["adx_14"] = adx["ADX_14"]
    else:
        df["adx_14"] = np.nan

    return df


def _resolve_bollinger_columns(columns: Sequence[str]) -> tuple[str, str, str] | None:
    """Resolve Bollinger Band column names across pandas-ta variants."""

    candidate_sets = [
        ("BBM_20_2.0", "BBU_20_2.0", "BBL_20_2.0"),
        ("BBM_20_2", "BBU_20_2", "BBL_20_2"),
    ]

    for candidate in candidate_sets:
        if set(candidate).issubset(columns):
            return candidate

    mid_candidates = [column for column in columns if column.startswith("BBM_")]
    upper_candidates = [column for column in columns if column.startswith("BBU_")]
    lower_candidates = [column for column in columns if column.startswith("BBL_")]

    if mid_candidates and upper_candidates and lower_candidates:
        return mid_candidates[0], upper_candidates[0], lower_candidates[0]

    return None


def identify_support_levels(df: pd.DataFrame) -> list[float]:
    """
    Identify support levels using recent lows

    Args:
        df: DataFrame with price data

    Returns:
        List of support price levels
    """
    # Use the lowest points in recent periods
    last_month = df.iloc[-30:] if len(df) >= 30 else df
    min_price = last_month["low"].min()

    # Additional support levels
    support_levels = [
        round(min_price, 2),
        round(df["close"].iloc[-1] * 0.95, 2),  # 5% below current price
        round(df["close"].iloc[-1] * 0.90, 2),  # 10% below current price
    ]

    return sorted(set(support_levels))


def identify_resistance_levels(df: pd.DataFrame) -> list[float]:
    """
    Identify resistance levels using recent highs

    Args:
        df: DataFrame with price data

    Returns:
        List of resistance price levels
    """
    # Use the highest points in recent periods
    last_month = df.iloc[-30:] if len(df) >= 30 else df
    max_price = last_month["high"].max()

    # Additional resistance levels
    resistance_levels = [
        round(max_price, 2),
        round(df["close"].iloc[-1] * 1.05, 2),  # 5% above current price
        round(df["close"].iloc[-1] * 1.10, 2),  # 10% above current price
    ]

    return sorted(set(resistance_levels))


def analyze_trend(df: pd.DataFrame) -> int:
    """
    Calculate the trend strength of a stock based on various technical indicators.

    Args:
        df: DataFrame with price and indicator data

    Returns:
        Integer trend strength score (0-7)
    """
    try:
        trend_strength = 0
        close_price = df["close"].iloc[-1]

        # Check SMA 50
        sma_50 = df["sma_50"].iloc[-1]
        if pd.notna(sma_50) and close_price > sma_50:
            trend_strength += 1

        # Check EMA 21
        ema_21 = df["ema_21"].iloc[-1]
        if pd.notna(ema_21) and close_price > ema_21:
            trend_strength += 1

        # Check EMA 21 vs SMA 50
        if pd.notna(ema_21) and pd.notna(sma_50) and ema_21 > sma_50:
            trend_strength += 1

        # Check SMA 50 vs SMA 200
        sma_200 = df["sma_200"].iloc[-1]
        if pd.notna(sma_50) and pd.notna(sma_200) and sma_50 > sma_200:
            trend_strength += 1

        # Check RSI
        rsi = df["rsi"].iloc[-1]
        if pd.notna(rsi) and rsi > 50:
            trend_strength += 1

        # Check MACD
        macd = df["macd_12_26_9"].iloc[-1]
        if pd.notna(macd) and macd > 0:
            trend_strength += 1

        # Check ADX
        adx = df["adx_14"].iloc[-1]
        if pd.notna(adx) and adx > 25:
            trend_strength += 1

        return trend_strength
    except Exception as e:
        logger.error(f"Error calculating trend strength: {e}")
        return 0


def analyze_rsi(df: pd.DataFrame) -> dict[str, Any]:
    """
    Analyze RSI indicator

    Args:
        df: DataFrame with price and indicator data

    Returns:
        Dictionary with RSI analysis
    """
    try:
        # Check if dataframe is valid and has RSI column
        if df.empty:
            return {
                "current": None,
                "signal": "unavailable",
                "description": "No data available for RSI calculation",
            }

        if "rsi" not in df.columns:
            return {
                "current": None,
                "signal": "unavailable",
                "description": "RSI indicator not calculated",
            }

        if len(df) == 0:
            return {
                "current": None,
                "signal": "unavailable",
                "description": "Insufficient data for RSI calculation",
            }

        rsi = df["rsi"].iloc[-1]

        # Check if RSI is NaN
        if pd.isna(rsi):
            return {
                "current": None,
                "signal": "unavailable",
                "description": "RSI data not available (insufficient data points)",
            }

        if rsi > 70:
            signal = "overbought"
        elif rsi < 30:
            signal = "oversold"
        elif rsi > 50:
            signal = "bullish"
        else:
            signal = "bearish"

        return {
            "current": round(rsi, 2),
            "signal": signal,
            "description": f"RSI is currently at {round(rsi, 2)}, indicating {signal} conditions.",
        }
    except Exception as e:
        logger.error(f"Error analyzing RSI: {e}")
        return {
            "current": None,
            "signal": "error",
            "description": f"Error calculating RSI: {str(e)}",
        }


def analyze_macd(df: pd.DataFrame) -> dict[str, Any]:
    """
    Analyze MACD indicator

    Args:
        df: DataFrame with price and indicator data

    Returns:
        Dictionary with MACD analysis
    """
    try:
        macd = df["macd_12_26_9"].iloc[-1]
        signal = df["macds_12_26_9"].iloc[-1]
        histogram = df["macdh_12_26_9"].iloc[-1]

        # Check if any values are NaN
        if pd.isna(macd) or pd.isna(signal) or pd.isna(histogram):
            return {
                "macd": None,
                "signal": None,
                "histogram": None,
                "indicator": "unavailable",
                "crossover": "unavailable",
                "description": "MACD data not available (insufficient data points)",
            }

        if macd > signal and histogram > 0:
            signal_type = "bullish"
        elif macd < signal and histogram < 0:
            signal_type = "bearish"
        elif macd > signal and macd < 0:
            signal_type = "improving"
        elif macd < signal and macd > 0:
            signal_type = "weakening"
        else:
            signal_type = "neutral"

        # Check for crossover (ensure we have enough data)
        crossover = "no recent crossover"
        if len(df) >= 2:
            prev_macd = df["macd_12_26_9"].iloc[-2]
            prev_signal = df["macds_12_26_9"].iloc[-2]
            if pd.notna(prev_macd) and pd.notna(prev_signal):
                if prev_macd <= prev_signal and macd > signal:
                    crossover = "bullish crossover detected"
                elif prev_macd >= prev_signal and macd < signal:
                    crossover = "bearish crossover detected"

        return {
            "macd": round(macd, 2),
            "signal": round(signal, 2),
            "histogram": round(histogram, 2),
            "indicator": signal_type,
            "crossover": crossover,
            "description": f"MACD is {signal_type} with {crossover}.",
        }
    except Exception as e:
        logger.error(f"Error analyzing MACD: {e}")
        return {
            "macd": None,
            "signal": None,
            "histogram": None,
            "indicator": "error",
            "crossover": "error",
            "description": "Error calculating MACD",
        }


def analyze_stochastic(df: pd.DataFrame) -> dict[str, Any]:
    """
    Analyze Stochastic Oscillator

    Args:
        df: DataFrame with price and indicator data

    Returns:
        Dictionary with stochastic oscillator analysis
    """
    try:
        k = df["stochk_14_3_3"].iloc[-1]
        d = df["stochd_14_3_3"].iloc[-1]

        # Check if values are NaN
        if pd.isna(k) or pd.isna(d):
            return {
                "k": None,
                "d": None,
                "signal": "unavailable",
                "crossover": "unavailable",
                "description": "Stochastic data not available (insufficient data points)",
            }

        if k > 80 and d > 80:
            signal = "overbought"
        elif k < 20 and d < 20:
            signal = "oversold"
        elif k > d:
            signal = "bullish"
        else:
            signal = "bearish"

        # Check for crossover (ensure we have enough data)
        crossover = "no recent crossover"
        if len(df) >= 2:
            prev_k = df["stochk_14_3_3"].iloc[-2]
            prev_d = df["stochd_14_3_3"].iloc[-2]
            if pd.notna(prev_k) and pd.notna(prev_d):
                if prev_k <= prev_d and k > d:
                    crossover = "bullish crossover detected"
                elif prev_k >= prev_d and k < d:
                    crossover = "bearish crossover detected"

        return {
            "k": round(k, 2),
            "d": round(d, 2),
            "signal": signal,
            "crossover": crossover,
            "description": f"Stochastic Oscillator is {signal} with {crossover}.",
        }
    except Exception as e:
        logger.error(f"Error analyzing Stochastic: {e}")
        return {
            "k": None,
            "d": None,
            "signal": "error",
            "crossover": "error",
            "description": "Error calculating Stochastic",
        }


def analyze_bollinger_bands(df: pd.DataFrame) -> dict[str, Any]:
    """
    Analyze Bollinger Bands

    Args:
        df: DataFrame with price and indicator data

    Returns:
        Dictionary with Bollinger Bands analysis
    """
    try:
        current_price = df["close"].iloc[-1]
        upper_band = df["bbu_20_2.0"].iloc[-1]
        lower_band = df["bbl_20_2.0"].iloc[-1]
        middle_band = df["sma_20"].iloc[-1]

        # Check if any values are NaN
        if pd.isna(upper_band) or pd.isna(lower_band) or pd.isna(middle_band):
            return {
                "upper_band": None,
                "middle_band": None,
                "lower_band": None,
                "position": "unavailable",
                "signal": "unavailable",
                "volatility": "unavailable",
                "description": "Bollinger Bands data not available (insufficient data points)",
            }

        if current_price > upper_band:
            position = "above upper band"
            signal = "overbought"
        elif current_price < lower_band:
            position = "below lower band"
            signal = "oversold"
        elif current_price > middle_band:
            position = "above middle band"
            signal = "bullish"
        else:
            position = "below middle band"
            signal = "bearish"

        # Check for BB squeeze (volatility contraction)
        volatility = "stable"
        if len(df) >= 5:
            try:
                bb_widths = []
                for i in range(-5, 0):
                    upper = df["bbu_20_2.0"].iloc[i]
                    lower = df["bbl_20_2.0"].iloc[i]
                    middle = df["sma_20"].iloc[i]
                    if (
                        pd.notna(upper)
                        and pd.notna(lower)
                        and pd.notna(middle)
                        and middle != 0
                    ):
                        bb_widths.append((upper - lower) / middle)

                if len(bb_widths) == 5:
                    if all(bb_widths[i] < bb_widths[i - 1] for i in range(1, 5)):
                        volatility = "contracting (potential breakout ahead)"
                    elif all(bb_widths[i] > bb_widths[i - 1] for i in range(1, 5)):
                        volatility = "expanding (increased volatility)"
            except Exception:
                # If volatility calculation fails, keep it as stable
                pass

        return {
            "upper_band": round(upper_band, 2),
            "middle_band": round(middle_band, 2),
            "lower_band": round(lower_band, 2),
            "position": position,
            "signal": signal,
            "volatility": volatility,
            "description": f"Price is {position}, indicating {signal} conditions. Volatility is {volatility}.",
        }
    except Exception as e:
        logger.error(f"Error analyzing Bollinger Bands: {e}")
        return {
            "upper_band": None,
            "middle_band": None,
            "lower_band": None,
            "position": "error",
            "signal": "error",
            "volatility": "error",
            "description": "Error calculating Bollinger Bands",
        }


def analyze_volume(df: pd.DataFrame) -> dict[str, Any]:
    """
    Analyze volume patterns

    Args:
        df: DataFrame with price and volume data

    Returns:
        Dictionary with volume analysis
    """
    try:
        current_volume = df["volume"].iloc[-1]

        # Check if we have enough data for average
        if len(df) < 10:
            avg_volume = df["volume"].mean()
        else:
            avg_volume = df["volume"].iloc[-10:].mean()

        # Check for invalid values
        if pd.isna(current_volume) or pd.isna(avg_volume) or avg_volume == 0:
            return {
                "current": None,
                "average": None,
                "ratio": None,
                "description": "unavailable",
                "signal": "unavailable",
            }

        volume_ratio = current_volume / avg_volume

        if volume_ratio > 1.5:
            volume_desc = "above average"
            if len(df) >= 2 and df["close"].iloc[-1] > df["close"].iloc[-2]:
                signal = "bullish (high volume on up move)"
            else:
                signal = "bearish (high volume on down move)"
        elif volume_ratio < 0.7:
            volume_desc = "below average"
            signal = "weak conviction"
        else:
            volume_desc = "average"
            signal = "neutral"

        return {
            "current": int(current_volume),
            "average": int(avg_volume),
            "ratio": round(volume_ratio, 2),
            "description": volume_desc,
            "signal": signal,
        }
    except Exception as e:
        logger.error(f"Error analyzing volume: {e}")
        return {
            "current": None,
            "average": None,
            "ratio": None,
            "description": "error",
            "signal": "error",
        }


def identify_chart_patterns(df: pd.DataFrame) -> list[str]:
    """
    Identify common chart patterns

    Args:
        df: DataFrame with price data

    Returns:
        List of identified chart patterns
    """
    patterns = []

    # Check for potential double bottom (W formation)
    if len(df) >= 40:
        recent_lows = df["low"].iloc[-40:].values
        potential_bottoms = []

        for i in range(1, len(recent_lows) - 1):
            if (
                recent_lows[i] < recent_lows[i - 1]
                and recent_lows[i] < recent_lows[i + 1]
            ):
                potential_bottoms.append(i)

        if (
            len(potential_bottoms) >= 2
            and potential_bottoms[-1] - potential_bottoms[-2] >= 5
        ):
            if (
                abs(
                    recent_lows[potential_bottoms[-1]]
                    - recent_lows[potential_bottoms[-2]]
                )
                / recent_lows[potential_bottoms[-2]]
                < 0.05
            ):
                patterns.append("Double Bottom (W)")

    # Check for potential double top (M formation)
    if len(df) >= 40:
        recent_highs = df["high"].iloc[-40:].values
        potential_tops = []

        for i in range(1, len(recent_highs) - 1):
            if (
                recent_highs[i] > recent_highs[i - 1]
                and recent_highs[i] > recent_highs[i + 1]
            ):
                potential_tops.append(i)

        if len(potential_tops) >= 2 and potential_tops[-1] - potential_tops[-2] >= 5:
            if (
                abs(recent_highs[potential_tops[-1]] - recent_highs[potential_tops[-2]])
                / recent_highs[potential_tops[-2]]
                < 0.05
            ):
                patterns.append("Double Top (M)")

    # Check for bullish flag/pennant
    if len(df) >= 20:
        recent_prices = df["close"].iloc[-20:].values
        if (
            recent_prices[0] < recent_prices[10]
            and all(
                recent_prices[i] >= recent_prices[i - 1] * 0.99 for i in range(1, 10)
            )
            and all(
                abs(recent_prices[i] - recent_prices[i - 1]) / recent_prices[i - 1]
                < 0.02
                for i in range(11, 20)
            )
        ):
            patterns.append("Bullish Flag/Pennant")

    # Check for bearish flag/pennant
    if len(df) >= 20:
        recent_prices = df["close"].iloc[-20:].values
        if (
            recent_prices[0] > recent_prices[10]
            and all(
                recent_prices[i] <= recent_prices[i - 1] * 1.01 for i in range(1, 10)
            )
            and all(
                abs(recent_prices[i] - recent_prices[i - 1]) / recent_prices[i - 1]
                < 0.02
                for i in range(11, 20)
            )
        ):
            patterns.append("Bearish Flag/Pennant")

    return patterns


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR) for the given dataframe.

    Args:
        df: DataFrame with high, low, and close price data
        period: Period for ATR calculation (default: 14)

    Returns:
        Series with ATR values
    """
    # Optimized to avoid copying the entire dataframe
    high_col = _get_column_case_insensitive(df, "high")
    low_col = _get_column_case_insensitive(df, "low")
    close_col = _get_column_case_insensitive(df, "close")

    if not (high_col and low_col and close_col):
        # Fallback to old method if columns are not found (unlikely if they exist)
        # This preserves previous behavior for missing columns which might raise error later or handle it
        logger.warning("Could not find High, Low, Close columns case-insensitively. Falling back to copy method.")
        df_copy = df.copy()
        df_copy.columns = [col.lower() for col in df_copy.columns]
        return ta.atr(df_copy["high"], df_copy["low"], df_copy["close"], length=period)

    # Use pandas_ta to calculate ATR
    atr = ta.atr(df[high_col], df[low_col], df[close_col], length=period)

    # Ensure we return a Series
    if isinstance(atr, pd.Series):
        return atr
    elif isinstance(atr, pd.DataFrame):
        # If it's a DataFrame, take the first column
        return pd.Series(atr.iloc[:, 0])
    elif atr is not None:
        # If it's a numpy array or other iterable
        return pd.Series(atr)
    else:
        # Return empty series if calculation failed
        return pd.Series(dtype=float)


def generate_outlook(
    df: pd.DataFrame,
    trend: str,
    rsi_analysis: dict[str, Any],
    macd_analysis: dict[str, Any],
    stoch_analysis: dict[str, Any],
) -> str:
    """
    Generate an overall outlook based on technical analysis

    Args:
        df: DataFrame with price and indicator data
        trend: Trend direction from analyze_trend
        rsi_analysis: RSI analysis from analyze_rsi
        macd_analysis: MACD analysis from analyze_macd
        stoch_analysis: Stochastic analysis from analyze_stochastic

    Returns:
        String with overall market outlook
    """
    bullish_signals = 0
    bearish_signals = 0

    # Count signals from different indicators
    if trend == "uptrend":
        bullish_signals += 2
    elif trend == "downtrend":
        bearish_signals += 2

    if rsi_analysis["signal"] == "bullish" or rsi_analysis["signal"] == "oversold":
        bullish_signals += 1
    elif rsi_analysis["signal"] == "bearish" or rsi_analysis["signal"] == "overbought":
        bearish_signals += 1

    if (
        macd_analysis["indicator"] == "bullish"
        or macd_analysis["crossover"] == "bullish crossover detected"
    ):
        bullish_signals += 1
    elif (
        macd_analysis["indicator"] == "bearish"
        or macd_analysis["crossover"] == "bearish crossover detected"
    ):
        bearish_signals += 1

    if stoch_analysis["signal"] == "bullish" or stoch_analysis["signal"] == "oversold":
        bullish_signals += 1
    elif (
        stoch_analysis["signal"] == "bearish"
        or stoch_analysis["signal"] == "overbought"
    ):
        bearish_signals += 1

    # Generate outlook based on signals
    if bullish_signals >= 4:
        return "strongly bullish"
    elif bullish_signals > bearish_signals:
        return "moderately bullish"
    elif bearish_signals >= 4:
        return "strongly bearish"
    elif bearish_signals > bullish_signals:
        return "moderately bearish"
    else:
        return "neutral"


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate RSI (Relative Strength Index) for the given dataframe.

    Args:
        df: DataFrame with price data
        period: Period for RSI calculation (default: 14)

    Returns:
        Series with RSI values
    """
    # Optimized to avoid copying the entire dataframe
    close_col = _get_column_case_insensitive(df, "close")

    # Ensure we have the required 'close' column
    if not close_col:
        # Check if we should fallback or raise error immediately.
        # Original code: copies, lowercases, then checks for "close".
        # So if we can't find it case-insensitively, we can raise ValueError.
        raise ValueError("DataFrame must contain a 'close' or 'Close' column")

    # Use pandas_ta to calculate RSI
    rsi = ta.rsi(df[close_col], length=period)

    # Ensure we return a Series
    if isinstance(rsi, pd.Series):
        return rsi
    elif rsi is not None:
        # If it's a numpy array or other iterable
        return pd.Series(rsi, index=df.index)
    else:
        # Return empty series if calculation failed
        return pd.Series(dtype=float, index=df.index)


def calculate_sma(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA) for the given dataframe.

    Args:
        df: DataFrame with price data
        period: Period for SMA calculation

    Returns:
        Series with SMA values
    """
    # Optimized to avoid copying the entire dataframe
    close_col = _get_column_case_insensitive(df, "close")

    # Ensure we have the required 'close' column
    if not close_col:
        raise ValueError("DataFrame must contain a 'close' or 'Close' column")

    # Use pandas_ta to calculate SMA
    sma = ta.sma(df[close_col], length=period)

    # Ensure we return a Series
    if isinstance(sma, pd.Series):
        return sma
    elif sma is not None:
        # If it's a numpy array or other iterable
        return pd.Series(sma, index=df.index)
    else:
        # Return empty series if calculation failed
        return pd.Series(dtype=float, index=df.index)
