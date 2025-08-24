"""
Portfolio analysis async tasks.

This module contains Celery tasks for portfolio analysis operations
that can be time-consuming with large portfolios.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from maverick_mcp.providers.stock_data import StockDataProvider
from maverick_mcp.queue.celery_app import celery_app
from maverick_mcp.queue.tasks.base import BaseTask

logger = logging.getLogger(__name__)


class PortfolioTask(BaseTask):
    """Base class for portfolio analysis tasks."""

    def get_credit_cost(self, **kwargs) -> int:
        """Calculate credit cost based on portfolio size."""
        tickers = kwargs.get("tickers", [])
        ticker_count = len(tickers) if isinstance(tickers, list) else 1

        # Base cost of 10 credits, plus 2 credits per additional ticker
        base_cost = 10
        volume_cost = max(0, (ticker_count - 1) * 2)

        return min(base_cost + volume_cost, 50)  # Cap at 50 credits

    def get_estimated_duration(self, **kwargs) -> int | None:
        """Estimate duration based on portfolio size."""
        tickers = kwargs.get("tickers", [])
        ticker_count = len(tickers) if isinstance(tickers, list) else 1
        days = kwargs.get("days", 90)

        # Base time of 60 seconds, plus 10 seconds per ticker, plus time for data range
        base_time = 60
        ticker_time = ticker_count * 10
        data_time = max(5, days // 30)  # Additional time for larger date ranges

        return base_time + ticker_time + data_time


@celery_app.task(base=PortfolioTask, bind=True)
def portfolio_correlation_task(
    self, tickers: list[str], days: int = 252
) -> dict[str, Any]:
    """
    Async task for portfolio correlation analysis.

    Args:
        tickers: List of ticker symbols to analyze
        days: Number of days for correlation calculation

    Returns:
        Dictionary containing correlation analysis results
    """
    try:
        if len(tickers) < 2:
            return {
                "error": "At least two tickers required for correlation analysis",
                "status": "error",
                "job_type": "portfolio_correlation",
            }

        self.update_progress(
            10.0,
            stage_name="initialization",
            stage_description="Setting up correlation analysis",
            status_message=f"Initializing analysis for {len(tickers)} tickers",
            total_items=len(tickers),
        )

        provider = StockDataProvider()
        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=days)

        # Fetch data for all tickers
        price_data = {}
        failed_tickers = []

        for i, ticker in enumerate(tickers):
            try:
                self.update_progress(
                    10.0 + (40.0 * i / len(tickers)),
                    stage_name="data_fetch",
                    stage_description=f"Fetching data for {ticker}",
                    status_message=f"Retrieving {ticker} price data",
                    items_processed=i,
                    total_items=len(tickers),
                )

                df = provider.get_stock_data(
                    ticker,
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                )

                if not df.empty:
                    price_data[ticker] = df["close"]
                else:
                    failed_tickers.append(ticker)
                    logger.warning(f"No data found for ticker {ticker}")

            except Exception as e:
                failed_tickers.append(ticker)
                logger.error(f"Error fetching data for {ticker}: {str(e)}")

        if len(price_data) < 2:
            return {
                "error": f"Insufficient data available. Failed tickers: {failed_tickers}",
                "status": "error",
                "job_type": "portfolio_correlation",
            }

        self.update_progress(
            50.0,
            stage_name="processing",
            stage_description="Calculating correlations",
            status_message="Computing correlation matrix",
        )

        # Create price DataFrame and calculate returns
        prices_df = pd.DataFrame(price_data)
        returns_df = prices_df.pct_change().dropna()

        self.update_progress(
            70.0,
            stage_name="correlation_analysis",
            stage_description="Analyzing correlation patterns",
            status_message="Identifying correlation patterns",
        )

        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()

        # Find high and low correlation pairs
        high_correlation_pairs = []
        low_correlation_pairs = []
        working_tickers = list(price_data.keys())

        for i in range(len(working_tickers)):
            for j in range(i + 1, len(working_tickers)):
                corr_val = correlation_matrix.iloc[i, j]
                corr = float(corr_val.item() if hasattr(corr_val, "item") else corr_val)
                pair = (working_tickers[i], working_tickers[j])

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

        self.update_progress(
            90.0,
            stage_name="finalization",
            stage_description="Finalizing correlation results",
            status_message="Generating final analysis",
        )

        # Calculate portfolio metrics
        mask = correlation_matrix.values != 1  # Exclude diagonal
        avg_correlation = correlation_matrix.values[mask].mean()

        results = {
            "status": "success",
            "correlation_matrix": correlation_matrix.round(3).to_dict(),
            "average_portfolio_correlation": round(avg_correlation, 3),
            "high_correlation_pairs": high_correlation_pairs,
            "low_correlation_pairs": low_correlation_pairs,
            "diversification_score": round((1 - avg_correlation) * 100, 1),
            "recommendation": (
                "Well diversified"
                if avg_correlation < 0.3
                else "Moderately diversified"
                if avg_correlation < 0.5
                else "Consider adding uncorrelated assets"
            ),
            "period_days": days,
            "data_points": len(returns_df),
            "tickers_analyzed": working_tickers,
            "failed_tickers": failed_tickers,
            "job_type": "portfolio_correlation",
            "async_job": True,
        }

        logger.info(
            f"Portfolio correlation analysis completed for {len(working_tickers)} tickers"
        )
        return results

    except Exception as e:
        logger.error(f"Error in portfolio correlation task: {str(e)}")
        return {"error": str(e), "status": "error", "job_type": "portfolio_correlation"}


@celery_app.task(base=PortfolioTask, bind=True)
def multi_ticker_analysis_task(
    self, tickers: list[str], days: int = 90
) -> dict[str, Any]:
    """
    Async task for multi-ticker technical analysis.

    Args:
        tickers: List of ticker symbols to analyze
        days: Number of days of historical data

    Returns:
        Dictionary containing analysis results for all tickers
    """
    try:
        self.update_progress(
            5.0,
            stage_name="initialization",
            stage_description="Setting up multi-ticker analysis",
            status_message=f"Initializing analysis for {len(tickers)} tickers",
            total_items=len(tickers),
        )

        from maverick_mcp.core.technical_analysis import analyze_rsi, analyze_trend
        from maverick_mcp.utils.stock_helpers import get_stock_dataframe

        results = {}
        failed_tickers = []

        for i, ticker in enumerate(tickers):
            try:
                progress_percent = 5.0 + (80.0 * i / len(tickers))
                self.update_progress(
                    progress_percent,
                    stage_name="technical_analysis",
                    stage_description=f"Analyzing {ticker}",
                    status_message=f"Running technical analysis for {ticker}",
                    items_processed=i,
                    total_items=len(tickers),
                )

                # Get stock data
                df = get_stock_dataframe(ticker, days)

                if df.empty:
                    failed_tickers.append(ticker)
                    continue

                # Perform technical analysis
                current_price = df["close"].iloc[-1]
                rsi = analyze_rsi(df)
                trend = analyze_trend(df)

                # Calculate performance metrics
                start_price = df["close"].iloc[0]
                price_change_pct = ((current_price - start_price) / start_price) * 100

                # Calculate volatility
                returns = df["close"].pct_change().dropna()
                volatility = returns.std() * (252**0.5) * 100  # Annualized

                # Volume analysis
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
                        "trend_description": (
                            "Strong Uptrend"
                            if trend >= 6
                            else "Uptrend"
                            if trend >= 4
                            else "Neutral"
                            if trend >= 3
                            else "Downtrend"
                        ),
                    },
                    "volume": {
                        "current_volume": int(df["volume"].iloc[-1]),
                        "avg_volume": int(avg_volume),
                        "volume_change_pct": volume_change_pct,
                        "volume_trend": (
                            "Increasing"
                            if volume_change_pct > 20
                            else "Decreasing"
                            if volume_change_pct < -20
                            else "Stable"
                        ),
                    },
                }

            except Exception as e:
                logger.error(f"Error analyzing ticker {ticker}: {str(e)}")
                failed_tickers.append(ticker)

        self.update_progress(
            90.0,
            stage_name="ranking",
            stage_description="Calculating rankings",
            status_message="Ranking stocks by performance and trend",
        )

        # Add relative rankings
        successful_tickers = list(results.keys())

        if successful_tickers:
            # Rank by performance
            perf_sorted = sorted(
                successful_tickers,
                key=lambda t: results[t]["performance"]["price_change_pct"],
                reverse=True,
            )

            # Rank by trend strength
            trend_sorted = sorted(
                successful_tickers,
                key=lambda t: results[t]["technical"]["trend_strength"],
                reverse=True,
            )

            for i, ticker in enumerate(perf_sorted):
                results[ticker]["rankings"] = {
                    "performance_rank": i + 1,
                    "trend_rank": trend_sorted.index(ticker) + 1,
                }

        self.update_progress(
            95.0,
            stage_name="finalization",
            stage_description="Finalizing multi-ticker results",
            status_message="Completing analysis summary",
        )

        final_results = {
            "status": "success",
            "analysis": results,
            "period_days": days,
            "as_of": datetime.now(UTC).isoformat(),
            "tickers_analyzed": successful_tickers,
            "failed_tickers": failed_tickers,
            "best_performer": perf_sorted[0] if successful_tickers else None,
            "strongest_trend": trend_sorted[0] if successful_tickers else None,
            "job_type": "multi_ticker_analysis",
            "async_job": True,
        }

        logger.info(
            f"Multi-ticker analysis completed for {len(successful_tickers)} out of {len(tickers)} tickers"
        )
        return final_results

    except Exception as e:
        logger.error(f"Error in multi-ticker analysis task: {str(e)}")
        return {"error": str(e), "status": "error", "job_type": "multi_ticker_analysis"}


@celery_app.task(base=PortfolioTask, bind=True)
def risk_adjusted_analysis_task(
    self, tickers: list[str], risk_level: float = 50.0, account_size: float = 100000.0
) -> dict[str, Any]:
    """
    Async task for risk-adjusted portfolio analysis.

    Args:
        tickers: List of ticker symbols to analyze
        risk_level: Risk tolerance (0-100)
        account_size: Portfolio size for position sizing

    Returns:
        Dictionary containing risk-adjusted analysis for all tickers
    """
    try:
        self.update_progress(
            5.0,
            stage_name="initialization",
            stage_description="Setting up risk-adjusted analysis",
            status_message=f"Initializing risk analysis for {len(tickers)} tickers",
            total_items=len(tickers),
        )

        import pandas_ta as ta

        provider = StockDataProvider()
        risk_factor = risk_level / 100
        results = {}
        failed_tickers = []

        for i, ticker in enumerate(tickers):
            try:
                progress_percent = 5.0 + (85.0 * i / len(tickers))
                self.update_progress(
                    progress_percent,
                    stage_name="risk_analysis",
                    stage_description=f"Risk analysis for {ticker}",
                    status_message=f"Calculating risk metrics for {ticker}",
                    items_processed=i,
                    total_items=len(tickers),
                )

                # Get stock data
                df = provider.get_stock_data(ticker)

                if df.empty:
                    failed_tickers.append(ticker)
                    continue

                # Calculate ATR and current price
                df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=20)
                atr = df["atr"].iloc[-1]
                current_price = df["close"].iloc[-1]

                # Risk-adjusted analysis
                position_value = account_size * 0.01 * risk_factor
                max_shares = int(position_value / current_price)

                analysis = {
                    "ticker": ticker,
                    "current_price": round(current_price, 2),
                    "atr": round(atr, 2),
                    "position_sizing": {
                        "suggested_position_size": round(position_value, 2),
                        "max_shares": max_shares,
                        "position_value": round(position_value, 2),
                        "percent_of_portfolio": round(1 * risk_factor, 2),
                    },
                    "risk_management": {
                        "stop_loss": round(
                            current_price - (atr * (2 - risk_factor)), 2
                        ),
                        "stop_loss_percent": round(
                            ((atr * (2 - risk_factor)) / current_price) * 100, 2
                        ),
                        "max_risk_amount": round(position_value, 2),
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
                        "price_target": round(
                            current_price + (atr * 3 * risk_factor), 2
                        ),
                        "profit_potential": round(atr * 3 * risk_factor, 2),
                        "risk_reward_ratio": round(3 * risk_factor, 2),
                    },
                }

                results[ticker] = analysis

            except Exception as e:
                logger.error(f"Error in risk analysis for {ticker}: {str(e)}")
                failed_tickers.append(ticker)

        self.update_progress(
            95.0,
            stage_name="finalization",
            stage_description="Finalizing risk analysis",
            status_message="Completing portfolio risk assessment",
        )

        # Calculate portfolio-level metrics
        total_position_value = sum(
            result["position_sizing"]["position_value"] for result in results.values()
        )

        portfolio_summary = {
            "total_position_value": round(total_position_value, 2),
            "portfolio_utilization": round(
                (total_position_value / account_size) * 100, 2
            ),
            "number_of_positions": len(results),
            "average_position_size": round(total_position_value / len(results), 2)
            if results
            else 0,
            "risk_level": risk_level,
            "account_size": account_size,
        }

        final_results = {
            "status": "success",
            "risk_analysis": results,
            "portfolio_summary": portfolio_summary,
            "tickers_analyzed": list(results.keys()),
            "failed_tickers": failed_tickers,
            "job_type": "risk_adjusted_analysis",
            "async_job": True,
        }

        logger.info(
            f"Risk-adjusted analysis completed for {len(results)} out of {len(tickers)} tickers"
        )
        return final_results

    except Exception as e:
        logger.error(f"Error in risk-adjusted analysis task: {str(e)}")
        return {
            "error": str(e),
            "status": "error",
            "job_type": "risk_adjusted_analysis",
        }
