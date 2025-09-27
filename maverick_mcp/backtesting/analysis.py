"""Backtest result analysis utilities."""

import logging
from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt

logger = logging.getLogger(__name__)


def convert_to_native(value):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(value, np.int64 | np.int32 | np.int16 | np.int8):
        return int(value)
    elif isinstance(value, np.float64 | np.float32 | np.float16):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif hasattr(value, "item"):  # For numpy scalars
        return value.item()
    elif pd.isna(value):
        return None
    return value


class BacktestAnalyzer:
    """Analyzer for backtest results."""

    async def run_vectorbt_backtest(
        self,
        data: pd.DataFrame,
        entry_signals: pd.Series,
        exit_signals: pd.Series,
        initial_capital: float = 10000.0,
        fees: float = 0.001,
        slippage: float = 0.001,
    ) -> dict[str, Any]:
        """Run a backtest using VectorBT with given signals.

        Args:
            data: Price data with OHLCV columns
            entry_signals: Boolean series for entry signals
            exit_signals: Boolean series for exit signals
            initial_capital: Initial capital amount
            fees: Trading fees as percentage
            slippage: Slippage as percentage

        Returns:
            Backtest results dictionary
        """
        # Validate inputs to prevent empty array errors
        if data is None or len(data) == 0:
            logger.warning("Empty or invalid data provided to run_vectorbt_backtest")
            return self._create_empty_backtest_results(initial_capital)

        if entry_signals is None or exit_signals is None:
            logger.warning("Invalid signals provided to run_vectorbt_backtest")
            return self._create_empty_backtest_results(initial_capital)

        # Check for empty signals or all-False signals
        if (
            len(entry_signals) == 0
            or len(exit_signals) == 0
            or entry_signals.size == 0
            or exit_signals.size == 0
        ):
            logger.warning("Empty signal arrays provided to run_vectorbt_backtest")
            return self._create_empty_backtest_results(initial_capital)

        # Check if signals have any True values
        if not entry_signals.any() and not exit_signals.any():
            logger.info("No trading signals generated - returning buy-and-hold results")
            return self._create_buyhold_backtest_results(data, initial_capital)

        # Ensure we have close prices
        close = data["close"] if "close" in data.columns else data["Close"]

        try:
            # Run VectorBT portfolio simulation
            portfolio = vbt.Portfolio.from_signals(
                close=close,
                entries=entry_signals,
                exits=exit_signals,
                init_cash=initial_capital,
                fees=fees,
                slippage=slippage,
                freq="D",
            )
        except Exception as e:
            logger.error(f"VectorBT Portfolio.from_signals failed: {e}")
            return self._create_empty_backtest_results(initial_capital, error=str(e))

        # Extract metrics
        metrics = {
            "total_return": float(portfolio.total_return()),
            "annual_return": float(portfolio.annualized_return())
            if hasattr(portfolio, "annualized_return")
            else 0,
            "sharpe_ratio": float(portfolio.sharpe_ratio())
            if not np.isnan(portfolio.sharpe_ratio())
            else 0,
            "max_drawdown": float(portfolio.max_drawdown()),
            "win_rate": float(portfolio.trades.win_rate())
            if portfolio.trades.count() > 0
            else 0,
            "total_trades": int(portfolio.trades.count()),
            "profit_factor": float(portfolio.trades.profit_factor())
            if portfolio.trades.count() > 0
            else 0,
        }

        # Extract trades
        trades = []
        if portfolio.trades.count() > 0:
            try:
                # VectorBT trades are in a records array
                trade_records = portfolio.trades.records
                for i in range(len(trade_records)):
                    trade = trade_records[i]
                    trades.append(
                        {
                            "entry_time": convert_to_native(trade["entry_idx"])
                            if "entry_idx" in trade.dtype.names
                            else i,
                            "exit_time": convert_to_native(trade["exit_idx"])
                            if "exit_idx" in trade.dtype.names
                            else i + 1,
                            "pnl": convert_to_native(trade["pnl"])
                            if "pnl" in trade.dtype.names
                            else 0.0,
                            "return": convert_to_native(trade["return"])
                            if "return" in trade.dtype.names
                            else 0.0,
                        }
                    )
            except (AttributeError, TypeError, KeyError) as e:
                # Fallback for different trade formats
                logger.debug(f"Could not extract detailed trades: {e}")
                trades = [
                    {
                        "total_trades": int(portfolio.trades.count()),
                        "message": "Detailed trade data not available",
                    }
                ]

        # Convert equity curve to ensure all values are Python native types
        equity_curve_raw = portfolio.value().to_dict()
        equity_curve = {
            str(k): convert_to_native(v) for k, v in equity_curve_raw.items()
        }

        # Also get drawdown series with proper conversion
        drawdown_raw = (
            portfolio.drawdown().to_dict() if hasattr(portfolio, "drawdown") else {}
        )
        drawdown_series = {
            str(k): convert_to_native(v) for k, v in drawdown_raw.items()
        }

        return {
            "metrics": metrics,
            "trades": trades,
            "equity_curve": equity_curve,
            "drawdown_series": drawdown_series,
        }

    def analyze(self, results: dict[str, Any]) -> dict[str, Any]:
        """Analyze backtest results and provide insights.

        Args:
            results: Backtest results from VectorBTEngine

        Returns:
            Analysis with performance grade, risk assessment, and recommendations
        """
        metrics = results.get("metrics", {})
        trades = results.get("trades", [])

        analysis = {
            "performance_grade": self._grade_performance(metrics),
            "risk_assessment": self._assess_risk(metrics),
            "trade_quality": self._analyze_trades(trades, metrics),
            "strengths": self._identify_strengths(metrics),
            "weaknesses": self._identify_weaknesses(metrics),
            "recommendations": self._generate_recommendations(metrics),
            "summary": self._generate_summary(metrics),
        }

        return analysis

    def _grade_performance(self, metrics: dict[str, float]) -> str:
        """Grade overall performance (A-F)."""
        score = 0
        max_score = 100

        # Sharpe ratio (30 points)
        sharpe = metrics.get("sharpe_ratio", 0)
        if sharpe >= 2.0:
            score += 30
        elif sharpe >= 1.5:
            score += 25
        elif sharpe >= 1.0:
            score += 20
        elif sharpe >= 0.5:
            score += 10
        else:
            score += 5

        # Total return (25 points)
        total_return = metrics.get("total_return", 0)
        if total_return >= 0.50:  # 50%+
            score += 25
        elif total_return >= 0.30:
            score += 20
        elif total_return >= 0.15:
            score += 15
        elif total_return >= 0.05:
            score += 10
        elif total_return > 0:
            score += 5

        # Win rate (20 points)
        win_rate = metrics.get("win_rate", 0)
        if win_rate >= 0.60:
            score += 20
        elif win_rate >= 0.50:
            score += 15
        elif win_rate >= 0.40:
            score += 10
        else:
            score += 5

        # Max drawdown (15 points)
        max_dd = abs(metrics.get("max_drawdown", 0))
        if max_dd <= 0.10:  # Less than 10%
            score += 15
        elif max_dd <= 0.20:
            score += 12
        elif max_dd <= 0.30:
            score += 8
        elif max_dd <= 0.40:
            score += 4

        # Profit factor (10 points)
        profit_factor = metrics.get("profit_factor", 0)
        if profit_factor >= 2.0:
            score += 10
        elif profit_factor >= 1.5:
            score += 8
        elif profit_factor >= 1.2:
            score += 5
        elif profit_factor > 1.0:
            score += 3

        # Convert score to grade
        percentage = (score / max_score) * 100
        if percentage >= 90:
            return "A"
        elif percentage >= 80:
            return "B"
        elif percentage >= 70:
            return "C"
        elif percentage >= 60:
            return "D"
        else:
            return "F"

    def _assess_risk(self, metrics: dict[str, float]) -> dict[str, Any]:
        """Assess risk characteristics."""
        max_dd = abs(metrics.get("max_drawdown", 0))
        sortino = metrics.get("sortino_ratio", 0)
        sharpe = metrics.get("sharpe_ratio", 0)
        calmar = metrics.get("calmar_ratio", 0)
        recovery = metrics.get("recovery_factor", 0)

        risk_level = "Low"
        if max_dd > 0.40:
            risk_level = "Very High"
        elif max_dd > 0.30:
            risk_level = "High"
        elif max_dd > 0.20:
            risk_level = "Medium"
        elif max_dd > 0.10:
            risk_level = "Low-Medium"

        return {
            "risk_level": risk_level,
            "max_drawdown": max_dd,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "recovery_factor": recovery,
            "risk_adjusted_return": sortino if sortino > 0 else sharpe,
            "downside_protection": "Good"
            if sortino > 1.5
            else "Moderate"
            if sortino > 0.5
            else "Poor",
        }

    def _analyze_trades(
        self, trades: list[dict], metrics: dict[str, float]
    ) -> dict[str, Any]:
        """Analyze trade quality and patterns."""
        if not trades:
            return {
                "quality": "No trades",
                "total_trades": 0,
                "frequency": "None",
            }

        total_trades = metrics.get("total_trades", 0)
        win_rate = metrics.get("win_rate", 0)
        avg_duration = metrics.get("avg_duration", 0)

        # Determine trade frequency
        if total_trades < 10:
            frequency = "Very Low"
        elif total_trades < 50:
            frequency = "Low"
        elif total_trades < 100:
            frequency = "Moderate"
        elif total_trades < 200:
            frequency = "High"
        else:
            frequency = "Very High"

        # Determine trade quality
        if win_rate >= 0.60 and metrics.get("profit_factor", 0) >= 1.5:
            quality = "Excellent"
        elif win_rate >= 0.50 and metrics.get("profit_factor", 0) >= 1.2:
            quality = "Good"
        elif win_rate >= 0.40:
            quality = "Average"
        else:
            quality = "Poor"

        return {
            "quality": quality,
            "total_trades": total_trades,
            "frequency": frequency,
            "win_rate": win_rate,
            "avg_win": metrics.get("avg_win", 0),
            "avg_loss": metrics.get("avg_loss", 0),
            "best_trade": metrics.get("best_trade", 0),
            "worst_trade": metrics.get("worst_trade", 0),
            "avg_duration_days": avg_duration,
            "risk_reward_ratio": metrics.get("risk_reward_ratio", 0),
        }

    def _identify_strengths(self, metrics: dict[str, float]) -> list[str]:
        """Identify strategy strengths."""
        strengths = []

        if metrics.get("sharpe_ratio", 0) >= 1.5:
            strengths.append("Excellent risk-adjusted returns")
        if metrics.get("win_rate", 0) >= 0.60:
            strengths.append("High win rate")
        if abs(metrics.get("max_drawdown", 0)) <= 0.15:
            strengths.append("Low maximum drawdown")
        if metrics.get("profit_factor", 0) >= 1.5:
            strengths.append("Strong profit factor")
        if metrics.get("sortino_ratio", 0) >= 2.0:
            strengths.append("Excellent downside protection")
        if metrics.get("calmar_ratio", 0) >= 1.0:
            strengths.append("Good return vs drawdown ratio")
        if metrics.get("recovery_factor", 0) >= 3.0:
            strengths.append("Quick drawdown recovery")
        if metrics.get("total_return", 0) >= 0.30:
            strengths.append("High total returns")

        return strengths if strengths else ["Consistent performance"]

    def _identify_weaknesses(self, metrics: dict[str, float]) -> list[str]:
        """Identify strategy weaknesses."""
        weaknesses = []

        if metrics.get("sharpe_ratio", 0) < 0.5:
            weaknesses.append("Poor risk-adjusted returns")
        if metrics.get("win_rate", 0) < 0.40:
            weaknesses.append("Low win rate")
        if abs(metrics.get("max_drawdown", 0)) > 0.30:
            weaknesses.append("High maximum drawdown")
        if metrics.get("profit_factor", 0) < 1.0:
            weaknesses.append("Unprofitable trades overall")
        if metrics.get("total_trades", 0) < 10:
            weaknesses.append("Insufficient trade signals")
        if metrics.get("sortino_ratio", 0) < 0:
            weaknesses.append("Poor downside protection")
        if metrics.get("total_return", 0) < 0:
            weaknesses.append("Negative returns")

        return weaknesses if weaknesses else ["Room for optimization"]

    def _generate_recommendations(self, metrics: dict[str, float]) -> list[str]:
        """Generate improvement recommendations."""
        recommendations = []

        # Risk management recommendations
        if abs(metrics.get("max_drawdown", 0)) > 0.25:
            recommendations.append(
                "Implement tighter stop-loss rules to reduce drawdowns"
            )

        # Win rate improvements
        if metrics.get("win_rate", 0) < 0.45:
            recommendations.append("Refine entry signals to improve win rate")

        # Trade frequency
        if metrics.get("total_trades", 0) < 20:
            recommendations.append(
                "Consider more sensitive parameters for increased signals"
            )
        elif metrics.get("total_trades", 0) > 200:
            recommendations.append("Filter signals to reduce overtrading")

        # Risk-reward optimization
        if metrics.get("risk_reward_ratio", 0) < 1.5:
            recommendations.append("Adjust exit strategy for better risk-reward ratio")

        # Profit factor improvements
        if metrics.get("profit_factor", 0) < 1.2:
            recommendations.append(
                "Focus on cutting losses quicker and letting winners run"
            )

        # Sharpe ratio improvements
        if metrics.get("sharpe_ratio", 0) < 1.0:
            recommendations.append("Consider position sizing based on volatility")

        # Kelly criterion
        kelly = metrics.get("kelly_criterion", 0)
        if kelly > 0 and kelly < 0.25:
            recommendations.append(
                f"Consider position size of {kelly * 100:.1f}% based on Kelly Criterion"
            )

        return (
            recommendations
            if recommendations
            else ["Strategy performing well, consider live testing"]
        )

    def _generate_summary(self, metrics: dict[str, float]) -> str:
        """Generate a text summary of the backtest."""
        total_return = metrics.get("total_return", 0) * 100
        sharpe = metrics.get("sharpe_ratio", 0)
        max_dd = abs(metrics.get("max_drawdown", 0)) * 100
        win_rate = metrics.get("win_rate", 0) * 100
        total_trades = metrics.get("total_trades", 0)

        summary = f"The strategy generated a {total_return:.1f}% return with a Sharpe ratio of {sharpe:.2f}. "
        summary += f"Maximum drawdown was {max_dd:.1f}% with a {win_rate:.1f}% win rate across {total_trades} trades. "

        if sharpe >= 1.5 and max_dd <= 20:
            summary += (
                "Overall performance is excellent with strong risk-adjusted returns."
            )
        elif sharpe >= 1.0 and max_dd <= 30:
            summary += "Performance is good with acceptable risk levels."
        elif sharpe >= 0.5:
            summary += "Performance is moderate and could benefit from optimization."
        else:
            summary += "Performance needs significant improvement before live trading."

        return summary

    def compare_strategies(self, results_list: list[dict[str, Any]]) -> dict[str, Any]:
        """Compare multiple strategy results.

        Args:
            results_list: List of backtest results to compare

        Returns:
            Comparison analysis with rankings
        """
        if not results_list:
            return {"error": "No results to compare"}

        comparisons = []

        for result in results_list:
            metrics = result.get("metrics", {})
            comparisons.append(
                {
                    "strategy": result.get("strategy", "Unknown"),
                    "parameters": result.get("parameters", {}),
                    "total_return": metrics.get("total_return", 0),
                    "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                    "max_drawdown": abs(metrics.get("max_drawdown", 0)),
                    "win_rate": metrics.get("win_rate", 0),
                    "profit_factor": metrics.get("profit_factor", 0),
                    "total_trades": metrics.get("total_trades", 0),
                    "grade": self._grade_performance(metrics),
                }
            )

        # Sort by Sharpe ratio as default ranking
        comparisons.sort(key=lambda x: x["sharpe_ratio"], reverse=True)

        # Add rankings
        for i, comp in enumerate(comparisons, 1):
            comp["rank"] = i

        # Find best in each category
        best_return = max(comparisons, key=lambda x: x["total_return"])
        best_sharpe = max(comparisons, key=lambda x: x["sharpe_ratio"])
        best_drawdown = min(comparisons, key=lambda x: x["max_drawdown"])
        best_win_rate = max(comparisons, key=lambda x: x["win_rate"])

        return {
            "rankings": comparisons,
            "best_overall": comparisons[0] if comparisons else None,
            "best_return": best_return,
            "best_sharpe": best_sharpe,
            "best_drawdown": best_drawdown,
            "best_win_rate": best_win_rate,
            "summary": self._generate_comparison_summary(comparisons),
        }

    def _generate_comparison_summary(self, comparisons: list[dict]) -> str:
        """Generate summary of strategy comparison."""
        if not comparisons:
            return "No strategies to compare"

        best = comparisons[0]
        summary = f"The best performing strategy is {best['strategy']} "
        summary += f"with a Sharpe ratio of {best['sharpe_ratio']:.2f} "
        summary += f"and total return of {best['total_return'] * 100:.1f}%. "

        if len(comparisons) > 1:
            summary += (
                f"It outperformed {len(comparisons) - 1} other strategies tested."
            )

        return summary

    def _create_empty_backtest_results(
        self, initial_capital: float, error: str = None
    ) -> dict[str, Any]:
        """Create empty backtest results when no valid signals are available.

        Args:
            initial_capital: Initial capital amount
            error: Optional error message to include

        Returns:
            Empty backtest results dictionary
        """
        return {
            "metrics": {
                "total_return": 0.0,
                "annual_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
                "profit_factor": 0.0,
            },
            "trades": [],
            "equity_curve": {str(0): initial_capital},
            "drawdown_series": {str(0): 0.0},
            "error": error,
            "message": "No trading signals generated - empty backtest results returned",
        }

    def _create_buyhold_backtest_results(
        self, data: pd.DataFrame, initial_capital: float
    ) -> dict[str, Any]:
        """Create buy-and-hold backtest results when no trading signals are available.

        Args:
            data: Price data
            initial_capital: Initial capital amount

        Returns:
            Buy-and-hold backtest results dictionary
        """
        try:
            # Calculate buy and hold performance
            close = data["close"] if "close" in data.columns else data["Close"]
            if len(close) == 0:
                return self._create_empty_backtest_results(initial_capital)

            start_price = close.iloc[0]
            end_price = close.iloc[-1]
            total_return = (end_price - start_price) / start_price

            # Simple buy and hold equity curve
            normalized_prices = close / start_price * initial_capital
            equity_curve = {
                str(idx): convert_to_native(val)
                for idx, val in normalized_prices.to_dict().items()
            }

            # Calculate drawdown for buy and hold
            cummax = normalized_prices.expanding().max()
            drawdown = (normalized_prices - cummax) / cummax
            drawdown_series = {
                str(idx): convert_to_native(val)
                for idx, val in drawdown.to_dict().items()
            }

            return {
                "metrics": {
                    "total_return": float(total_return),
                    "annual_return": float(total_return * 252 / len(data))
                    if len(data) > 0
                    else 0.0,
                    "sharpe_ratio": 0.0,  # Cannot calculate without trading
                    "max_drawdown": float(drawdown.min()) if len(drawdown) > 0 else 0.0,
                    "win_rate": 0.0,  # No trades
                    "total_trades": 0,
                    "profit_factor": 0.0,  # No trades
                },
                "trades": [],
                "equity_curve": equity_curve,
                "drawdown_series": drawdown_series,
                "message": "No trading signals generated - returning buy-and-hold performance",
            }
        except Exception as e:
            logger.error(f"Error creating buy-and-hold results: {e}")
            return self._create_empty_backtest_results(initial_capital, error=str(e))
