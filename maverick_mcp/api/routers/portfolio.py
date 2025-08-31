"""
Portfolio analysis router for Maverick-MCP.

This module contains all portfolio-related tools including
risk analysis, comparisons, and optimization functions.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import pandas_ta as ta
from fastmcp import FastMCP

from maverick_mcp.providers.stock_data import StockDataProvider
from maverick_mcp.utils.stock_helpers import get_stock_dataframe

logger = logging.getLogger(__name__)

# Create the portfolio router
portfolio_router: FastMCP = FastMCP("Portfolio_Analysis")

# Initialize data provider
stock_provider = StockDataProvider()


def risk_adjusted_analysis(
    ticker: str, risk_level: float | str | None = 50.0
) -> dict[str, Any]:
    """
    Perform risk-adjusted stock analysis with position sizing.

    DISCLAIMER: This analysis is for educational purposes only and does not
    constitute investment advice. All investments carry risk of loss. Always
    consult with qualified financial professionals before making investment decisions.

    This tool analyzes a stock with risk parameters tailored to different investment
    styles. It provides:
    - Position sizing recommendations based on ATR
    - Stop loss suggestions
    - Entry points with scaling
    - Risk/reward ratio calculations
    - Confidence score based on technicals

    The risk_level parameter (0-100) adjusts the analysis from conservative (low)
    to aggressive (high).

    Args:
        ticker: The ticker symbol to analyze
        risk_level: Risk tolerance from 0 (conservative) to 100 (aggressive)

    Returns:
        Dictionary containing risk-adjusted analysis results
    """
    try:
        # Convert risk_level to float if it's a string
        if isinstance(risk_level, str):
            try:
                risk_level = float(risk_level)
            except ValueError:
                risk_level = 50.0
        
        # Use explicit date range to avoid weekend/holiday issues
        from datetime import UTC, datetime, timedelta
        end_date = (datetime.now(UTC) - timedelta(days=7)).strftime("%Y-%m-%d")  # Last week to be safe
        start_date = (datetime.now(UTC) - timedelta(days=365)).strftime("%Y-%m-%d")  # 1 year ago
        df = stock_provider.get_stock_data(ticker, start_date=start_date, end_date=end_date)
        
        # Validate dataframe has required columns (check for both upper and lower case)
        required_cols = ["high", "low", "close"]
        actual_cols_lower = [col.lower() for col in df.columns]
        if df.empty or not all(col in actual_cols_lower for col in required_cols):
            return {
                "error": f"Insufficient data for {ticker}",
                "details": "Unable to retrieve required price data (High, Low, Close) for analysis",
                "ticker": ticker,
                "required_data": ["High", "Low", "Close", "Volume"],
                "available_columns": list(df.columns)
            }
        
        df["atr"] = ta.atr(df["High"], df["Low"], df["Close"], length=20)
        atr = df["atr"].iloc[-1]
        current_price = df["Close"].iloc[-1]
        risk_factor = (risk_level or 50.0) / 100  # Convert to 0-1 scale
        account_size = 100000
        analysis = {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "atr": round(atr, 2),
            "risk_level": risk_level,
            "position_sizing": {
                "suggested_position_size": round(account_size * 0.01 * risk_factor, 2),
                "max_shares": int((account_size * 0.01 * risk_factor) / current_price),
                "position_value": round(account_size * 0.01 * risk_factor, 2),
                "percent_of_portfolio": round(1 * risk_factor, 2),
            },
            "risk_management": {
                "stop_loss": round(current_price - (atr * (2 - risk_factor)), 2),
                "stop_loss_percent": round(
                    ((atr * (2 - risk_factor)) / current_price) * 100, 2
                ),
                "max_risk_amount": round(account_size * 0.01 * risk_factor, 2),
            },
            "entry_strategy": {
                "immediate_entry": round(current_price, 2),
                "scale_in_levels": [
                    round(current_price, 2),
                    round(current_price - (atr * 0.5), 2),
                    round(current_price - atr, 2),
                ],
            },
            "targets": {
                "price_target": round(current_price + (atr * 3 * risk_factor), 2),
                "profit_potential": round(atr * 3 * risk_factor, 2),
                "risk_reward_ratio": round(3 * risk_factor, 2),
            },
            "analysis": {
                "confidence_score": round(70 * risk_factor, 2),
                "strategy_type": "aggressive"
                if (risk_level or 50.0) > 70
                else "moderate"
                if (risk_level or 50.0) > 30
                else "conservative",
                "time_horizon": "short-term"
                if (risk_level or 50.0) > 70
                else "medium-term"
                if (risk_level or 50.0) > 30
                else "long-term",
            },
        }
        return analysis
    except Exception as e:
        logger.error(f"Error performing risk analysis for {ticker}: {e}")
        return {"error": str(e)}


def compare_tickers(tickers: list[str], days: int = 90) -> dict[str, Any]:
    """
    Compare multiple tickers using technical and fundamental metrics.

    This tool provides side-by-side comparison of stocks including:
    - Price performance
    - Technical indicators (RSI, trend strength)
    - Volume characteristics
    - Momentum strength ratings
    - Risk metrics

    Args:
        tickers: List of ticker symbols to compare (minimum 2)
        days: Number of days of historical data to analyze (default: 90)

    Returns:
        Dictionary containing comparison results
    """
    try:
        if len(tickers) < 2:
            raise ValueError("At least two tickers are required for comparison")

        from maverick_mcp.core.technical_analysis import analyze_rsi, analyze_trend

        results = {}
        for ticker in tickers:
            df = get_stock_dataframe(ticker, days)

            # Basic analysis for comparison
            current_price = df["close"].iloc[-1]
            rsi = analyze_rsi(df)
            trend = analyze_trend(df)

            # Calculate performance metrics
            start_price = df["close"].iloc[0]
            price_change_pct = ((current_price - start_price) / start_price) * 100

            # Calculate volatility (standard deviation of returns)
            returns = df["close"].pct_change().dropna()
            volatility = returns.std() * (252**0.5) * 100  # Annualized

            # Calculate volume metrics
            volume_change_pct = 0.0
            if len(df) >= 22 and df["volume"].iloc[-22] > 0:
                volume_change_pct = float(
                    (df["volume"].iloc[-1] / df["volume"].iloc[-22] - 1) * 100
                )

            avg_volume = df["volume"].mean()

            results[ticker] = {
                "current_price": float(current_price),
                "performance": {
                    "price_change_pct": round(price_change_pct, 2),
                    "period_high": float(df["high"].max()),
                    "period_low": float(df["low"].min()),
                    "volatility_annual": round(volatility, 2),
                },
                "technical": {
                    "rsi": rsi["current"] if rsi and "current" in rsi else None,
                    "rsi_signal": rsi["signal"]
                    if rsi and "signal" in rsi
                    else "unavailable",
                    "trend_strength": trend,
                    "trend_description": "Strong Uptrend"
                    if trend >= 6
                    else "Uptrend"
                    if trend >= 4
                    else "Neutral"
                    if trend >= 3
                    else "Downtrend",
                },
                "volume": {
                    "current_volume": int(df["volume"].iloc[-1]),
                    "avg_volume": int(avg_volume),
                    "volume_change_pct": volume_change_pct,
                    "volume_trend": "Increasing"
                    if volume_change_pct > 20
                    else "Decreasing"
                    if volume_change_pct < -20
                    else "Stable",
                },
            }

        # Add relative rankings
        tickers_list = list(results.keys())

        # Rank by performance
        def get_performance(ticker: str) -> float:
            ticker_result = results[ticker]
            assert isinstance(ticker_result, dict)
            perf_dict = ticker_result["performance"]
            assert isinstance(perf_dict, dict)
            return float(perf_dict["price_change_pct"])

        def get_trend(ticker: str) -> float:
            ticker_result = results[ticker]
            assert isinstance(ticker_result, dict)
            tech_dict = ticker_result["technical"]
            assert isinstance(tech_dict, dict)
            return float(tech_dict["trend_strength"])

        perf_sorted = sorted(tickers_list, key=get_performance, reverse=True)
        trend_sorted = sorted(tickers_list, key=get_trend, reverse=True)

        for i, ticker in enumerate(perf_sorted):
            results[ticker]["rankings"] = {
                "performance_rank": i + 1,
                "trend_rank": trend_sorted.index(ticker) + 1,
            }

        return {
            "comparison": results,
            "period_days": days,
            "as_of": datetime.now(UTC).isoformat(),
            "best_performer": perf_sorted[0],
            "strongest_trend": trend_sorted[0],
        }
    except Exception as e:
        logger.error(f"Error comparing tickers {tickers}: {str(e)}")
        return {"error": str(e), "status": "error"}


def portfolio_correlation_analysis(
    tickers: list[str], days: int = 252
) -> dict[str, Any]:
    """
    Analyze correlation between multiple securities.

    DISCLAIMER: This correlation analysis is for educational purposes only.
    Past correlations do not guarantee future relationships between securities.
    Always diversify appropriately and consult with financial professionals.

    This tool calculates the correlation matrix for a portfolio of stocks,
    helping to identify:
    - Highly correlated positions (diversification issues)
    - Negative correlations (natural hedges)
    - Overall portfolio correlation metrics

    Args:
        tickers: List of ticker symbols to analyze
        days: Number of days for correlation calculation (default: 252 for 1 year)

    Returns:
        Dictionary containing correlation analysis
    """
    try:
        if len(tickers) < 2:
            raise ValueError("At least two tickers required for correlation analysis")

        # Fetch data for all tickers
        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=days)

        price_data = {}
        for ticker in tickers:
            df = stock_provider.get_stock_data(
                ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
            )
            if not df.empty:
                price_data[ticker] = df["close"]

        # Create price DataFrame
        prices_df = pd.DataFrame(price_data)

        # Calculate returns
        returns_df = prices_df.pct_change().dropna()

        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()

        # Find highly correlated pairs
        high_correlation_pairs = []
        low_correlation_pairs = []

        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                corr_val = correlation_matrix.iloc[i, j]
                corr = float(corr_val.item() if hasattr(corr_val, "item") else corr_val)
                pair = (tickers[i], tickers[j])

                if corr > 0.7:
                    high_correlation_pairs.append(
                        {
                            "pair": pair,
                            "correlation": round(corr, 3),
                            "interpretation": "High positive correlation",
                        }
                    )
                elif corr < -0.3:
                    low_correlation_pairs.append(
                        {
                            "pair": pair,
                            "correlation": round(corr, 3),
                            "interpretation": "Negative correlation (potential hedge)",
                        }
                    )

        # Calculate average portfolio correlation
        mask = correlation_matrix.values != 1  # Exclude diagonal
        avg_correlation = correlation_matrix.values[mask].mean()

        return {
            "correlation_matrix": correlation_matrix.round(3).to_dict(),
            "average_portfolio_correlation": round(avg_correlation, 3),
            "high_correlation_pairs": high_correlation_pairs,
            "low_correlation_pairs": low_correlation_pairs,
            "diversification_score": round((1 - avg_correlation) * 100, 1),
            "recommendation": "Well diversified"
            if avg_correlation < 0.3
            else "Moderately diversified"
            if avg_correlation < 0.5
            else "Consider adding uncorrelated assets",
            "period_days": days,
            "data_points": len(returns_df),
        }

    except Exception as e:
        logger.error(f"Error in correlation analysis: {str(e)}")
        return {"error": str(e), "status": "error"}
