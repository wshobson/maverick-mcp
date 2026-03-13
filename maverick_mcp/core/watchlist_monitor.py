"""
Watchlist monitoring module for Maverick-MCP.

Evaluates technical alert conditions against live stock data.
Stateless design — checks conditions on demand without persistent state.

DISCLAIMER: All alerts are for educational purposes only and do not
constitute investment advice. Always conduct thorough research and
consult with qualified financial professionals before making decisions.
"""

import logging
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from maverick_mcp.core.technical_analysis import (
    add_technical_indicators,
    analyze_bollinger_bands,
    analyze_macd,
    analyze_rsi,
    analyze_volume,
    identify_resistance_levels,
    identify_support_levels,
)
from maverick_mcp.data.models import UserPortfolio, get_db
from maverick_mcp.providers.stock_data import StockDataProvider

logger = logging.getLogger("maverick_mcp.watchlist_monitor")

# All supported alert condition types
ALL_CONDITIONS = [
    "rsi_overbought",
    "rsi_oversold",
    "macd_bullish_cross",
    "macd_bearish_cross",
    "price_above_resistance",
    "price_below_support",
    "volume_spike",
    "bollinger_squeeze",
    "trailing_stop",
]


def _fetch_analysis_data(ticker: str, days: int = 90) -> pd.DataFrame | None:
    """Fetch recent daily data and add technical indicators.

    Args:
        ticker: Stock ticker symbol.
        days: Number of trading days to fetch (default 90 for indicator warmup).

    Returns:
        DataFrame with OHLCV data and technical indicators, or None on failure.
    """
    try:
        provider = StockDataProvider()
        df = provider.get_stock_data(ticker, period=f"{days}d", interval="1d")
        if df is None or df.empty or len(df) < 20:
            return None
        df = add_technical_indicators(df)
        return df
    except Exception as e:
        logger.warning(f"Failed to fetch analysis data for {ticker}: {e}")
        return None


def _check_rsi_overbought(
    rsi_result: dict[str, Any], threshold: float
) -> dict[str, Any] | None:
    """Check if RSI is above overbought threshold."""
    current_rsi = rsi_result.get("current")
    if current_rsi is not None and current_rsi > threshold:
        return {
            "type": "rsi_overbought",
            "severity": "warning",
            "message": f"RSI at {current_rsi:.1f} — above overbought threshold of {threshold}",
            "value": round(current_rsi, 2),
            "threshold": threshold,
        }
    return None


def _check_rsi_oversold(
    rsi_result: dict[str, Any], threshold: float
) -> dict[str, Any] | None:
    """Check if RSI is below oversold threshold."""
    current_rsi = rsi_result.get("current")
    if current_rsi is not None and current_rsi < threshold:
        return {
            "type": "rsi_oversold",
            "severity": "warning",
            "message": f"RSI at {current_rsi:.1f} — below oversold threshold of {threshold}",
            "value": round(current_rsi, 2),
            "threshold": threshold,
        }
    return None


def _check_macd_bullish_cross(macd_result: dict[str, Any]) -> dict[str, Any] | None:
    """Check for bullish MACD crossover."""
    crossover = macd_result.get("crossover")
    if crossover == "bullish":
        return {
            "type": "macd_bullish_cross",
            "severity": "info",
            "message": "Bullish MACD crossover detected — MACD crossed above signal line",
            "value": macd_result.get("macd"),
            "threshold": macd_result.get("signal"),
        }
    return None


def _check_macd_bearish_cross(macd_result: dict[str, Any]) -> dict[str, Any] | None:
    """Check for bearish MACD crossover."""
    crossover = macd_result.get("crossover")
    if crossover == "bearish":
        return {
            "type": "macd_bearish_cross",
            "severity": "warning",
            "message": "Bearish MACD crossover detected — MACD crossed below signal line",
            "value": macd_result.get("macd"),
            "threshold": macd_result.get("signal"),
        }
    return None


def _check_price_above_resistance(
    current_price: float, df: pd.DataFrame
) -> dict[str, Any] | None:
    """Check if price has broken above nearest resistance level."""
    resistance_levels = identify_resistance_levels(df)
    if not resistance_levels:
        return None

    # Find resistance levels near or just below current price
    for level in sorted(resistance_levels):
        # Price is above resistance by up to 2%
        if current_price > level and (current_price - level) / level < 0.02:
            return {
                "type": "price_above_resistance",
                "severity": "info",
                "message": (
                    f"Price ${current_price:.2f} broke above resistance at ${level:.2f}"
                ),
                "value": round(current_price, 2),
                "threshold": round(level, 2),
            }
    return None


def _check_price_below_support(
    current_price: float, df: pd.DataFrame
) -> dict[str, Any] | None:
    """Check if price has broken below nearest support level."""
    support_levels = identify_support_levels(df)
    if not support_levels:
        return None

    # Find support levels near or just above current price
    for level in sorted(support_levels, reverse=True):
        # Price is below support by up to 2%
        if current_price < level and (level - current_price) / level < 0.02:
            return {
                "type": "price_below_support",
                "severity": "warning",
                "message": (
                    f"Price ${current_price:.2f} broke below support at ${level:.2f}"
                ),
                "value": round(current_price, 2),
                "threshold": round(level, 2),
            }
    return None


def _check_volume_spike(
    volume_result: dict[str, Any], multiplier: float
) -> dict[str, Any] | None:
    """Check if current volume exceeds average by multiplier."""
    ratio = volume_result.get("ratio")
    if ratio is not None and ratio > multiplier:
        return {
            "type": "volume_spike",
            "severity": "info",
            "message": (
                f"Volume spike — current volume is {ratio:.1f}x "
                f"the average (threshold: {multiplier}x)"
            ),
            "value": round(ratio, 2),
            "threshold": multiplier,
        }
    return None


def _check_bollinger_squeeze(bb_result: dict[str, Any]) -> dict[str, Any] | None:
    """Check for Bollinger Band squeeze (volatility contraction)."""
    volatility = bb_result.get("volatility", "")
    if "contracting" in str(volatility).lower():
        return {
            "type": "bollinger_squeeze",
            "severity": "info",
            "message": "Bollinger Band squeeze detected — volatility contracting, potential breakout ahead",
            "value": volatility,
            "threshold": "contracting",
        }
    return None


def _check_trailing_stop(
    current_price: float, df: pd.DataFrame, stop_pct: float
) -> dict[str, Any] | None:
    """Check if price has dropped by stop_pct from recent high."""
    if len(df) < 5:
        return None

    # Use 20-day high as the reference
    lookback = min(20, len(df))
    recent_high = df["close"].iloc[-lookback:].max()

    if recent_high > 0:
        decline_pct = ((recent_high - current_price) / recent_high) * 100
        if decline_pct >= stop_pct:
            return {
                "type": "trailing_stop",
                "severity": "critical",
                "message": (
                    f"Trailing stop triggered — price ${current_price:.2f} is "
                    f"{decline_pct:.1f}% below 20-day high of ${recent_high:.2f} "
                    f"(threshold: {stop_pct}%)"
                ),
                "value": round(decline_pct, 2),
                "threshold": stop_pct,
            }
    return None


def evaluate_alerts(
    df: pd.DataFrame,
    conditions: list[str] | None = None,
    rsi_overbought: float = 70.0,
    rsi_oversold: float = 30.0,
    volume_spike_multiplier: float = 2.0,
    trailing_stop_pct: float = 5.0,
) -> list[dict[str, Any]]:
    """Evaluate alert conditions against a DataFrame with technical indicators.

    This is the core evaluation function. It takes a DataFrame that already
    has technical indicators added and checks the specified conditions.

    Args:
        df: DataFrame with OHLCV data and technical indicators.
        conditions: List of condition names to check, or None for all.
        rsi_overbought: RSI overbought threshold (default 70).
        rsi_oversold: RSI oversold threshold (default 30).
        volume_spike_multiplier: Volume spike multiplier (default 2.0).
        trailing_stop_pct: Trailing stop percentage (default 5.0).

    Returns:
        List of triggered alert dicts, each with type, severity, message,
        value, and threshold.
    """
    active_conditions = set(conditions if conditions is not None else ALL_CONDITIONS)
    alerts: list[dict[str, Any]] = []

    current_price = float(df["close"].iloc[-1])

    # RSI checks
    if active_conditions & {"rsi_overbought", "rsi_oversold"}:
        try:
            rsi_result = analyze_rsi(df)
            if "rsi_overbought" in active_conditions:
                alert = _check_rsi_overbought(rsi_result, rsi_overbought)
                if alert:
                    alerts.append(alert)
            if "rsi_oversold" in active_conditions:
                alert = _check_rsi_oversold(rsi_result, rsi_oversold)
                if alert:
                    alerts.append(alert)
        except Exception as e:
            logger.debug(f"RSI check failed: {e}")

    # MACD checks
    if active_conditions & {"macd_bullish_cross", "macd_bearish_cross"}:
        try:
            macd_result = analyze_macd(df)
            if "macd_bullish_cross" in active_conditions:
                alert = _check_macd_bullish_cross(macd_result)
                if alert:
                    alerts.append(alert)
            if "macd_bearish_cross" in active_conditions:
                alert = _check_macd_bearish_cross(macd_result)
                if alert:
                    alerts.append(alert)
        except Exception as e:
            logger.debug(f"MACD check failed: {e}")

    # Support/resistance checks
    if "price_above_resistance" in active_conditions:
        try:
            alert = _check_price_above_resistance(current_price, df)
            if alert:
                alerts.append(alert)
        except Exception as e:
            logger.debug(f"Resistance check failed: {e}")

    if "price_below_support" in active_conditions:
        try:
            alert = _check_price_below_support(current_price, df)
            if alert:
                alerts.append(alert)
        except Exception as e:
            logger.debug(f"Support check failed: {e}")

    # Volume spike
    if "volume_spike" in active_conditions:
        try:
            volume_result = analyze_volume(df)
            alert = _check_volume_spike(volume_result, volume_spike_multiplier)
            if alert:
                alerts.append(alert)
        except Exception as e:
            logger.debug(f"Volume check failed: {e}")

    # Bollinger squeeze
    if "bollinger_squeeze" in active_conditions:
        try:
            bb_result = analyze_bollinger_bands(df)
            alert = _check_bollinger_squeeze(bb_result)
            if alert:
                alerts.append(alert)
        except Exception as e:
            logger.debug(f"Bollinger check failed: {e}")

    # Trailing stop
    if "trailing_stop" in active_conditions:
        try:
            alert = _check_trailing_stop(current_price, df, trailing_stop_pct)
            if alert:
                alerts.append(alert)
        except Exception as e:
            logger.debug(f"Trailing stop check failed: {e}")

    return alerts


def check_alerts_for_ticker(
    ticker: str,
    conditions: list[str] | None = None,
    rsi_overbought: float = 70.0,
    rsi_oversold: float = 30.0,
    volume_spike_multiplier: float = 2.0,
    trailing_stop_pct: float = 5.0,
) -> dict[str, Any]:
    """Fetch recent data for a ticker and evaluate all alert conditions.

    Args:
        ticker: Stock ticker symbol.
        conditions: List of condition names to check, or None for all.
        rsi_overbought: RSI overbought threshold (default 70).
        rsi_oversold: RSI oversold threshold (default 30).
        volume_spike_multiplier: Volume spike multiplier (default 2.0).
        trailing_stop_pct: Trailing stop percentage (default 5.0).

    Returns:
        Dict with ticker, current_price, alert_count, and alerts list.
    """
    ticker = ticker.upper()

    df = _fetch_analysis_data(ticker)
    if df is None:
        return {
            "ticker": ticker,
            "status": "error",
            "message": f"Could not fetch sufficient data for {ticker}",
            "alerts": [],
            "alert_count": 0,
        }

    current_price = float(df["close"].iloc[-1])

    alerts = evaluate_alerts(
        df,
        conditions=conditions,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
        volume_spike_multiplier=volume_spike_multiplier,
        trailing_stop_pct=trailing_stop_pct,
    )

    return {
        "ticker": ticker,
        "current_price": round(current_price, 2),
        "status": "ok",
        "alert_count": len(alerts),
        "alerts": alerts,
    }


def _get_portfolio_tickers(
    user_id: str = "default",
    portfolio_name: str = "My Portfolio",
) -> list[str]:
    """Get ticker symbols from portfolio positions.

    Args:
        user_id: User identifier.
        portfolio_name: Portfolio name.

    Returns:
        List of ticker symbols held in the portfolio.
    """
    try:
        db: Session = next(get_db())
        try:
            portfolio_db = (
                db.query(UserPortfolio)
                .filter_by(user_id=user_id, name=portfolio_name)
                .first()
            )
            if not portfolio_db:
                return []

            # Positions already loaded via selectin relationship
            return [p.ticker for p in portfolio_db.positions]
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"Failed to fetch portfolio positions: {e}")
        return []


def check_portfolio_alerts(
    user_id: str = "default",
    portfolio_name: str = "My Portfolio",
    conditions: list[str] | None = None,
    rsi_overbought: float = 70.0,
    rsi_oversold: float = 30.0,
    volume_spike_multiplier: float = 2.0,
    trailing_stop_pct: float = 5.0,
) -> dict[str, Any]:
    """Check alert conditions for all portfolio positions.

    Fetches portfolio positions from the database, evaluates alert conditions
    for each ticker, and returns aggregated results.

    Args:
        user_id: User identifier (default "default").
        portfolio_name: Portfolio name (default "My Portfolio").
        conditions: Conditions to check, or None for all.
        rsi_overbought: RSI overbought threshold.
        rsi_oversold: RSI oversold threshold.
        volume_spike_multiplier: Volume spike multiplier.
        trailing_stop_pct: Trailing stop percentage.

    Returns:
        Dict with portfolio-level summary and per-ticker alert results.
    """
    tickers = _get_portfolio_tickers(user_id, portfolio_name)

    if not tickers:
        return {
            "status": "empty",
            "message": "No portfolio positions found to monitor",
            "tickers_checked": 0,
            "total_alerts": 0,
            "results": [],
        }

    results = []
    total_alerts = 0

    for ticker in tickers:
        result = check_alerts_for_ticker(
            ticker,
            conditions=conditions,
            rsi_overbought=rsi_overbought,
            rsi_oversold=rsi_oversold,
            volume_spike_multiplier=volume_spike_multiplier,
            trailing_stop_pct=trailing_stop_pct,
        )
        results.append(result)
        total_alerts += result.get("alert_count", 0)

    # Sort by alert count descending so most urgent tickers appear first
    results.sort(key=lambda r: r.get("alert_count", 0), reverse=True)

    return {
        "status": "ok",
        "tickers_checked": len(tickers),
        "total_alerts": total_alerts,
        "results": results,
    }
