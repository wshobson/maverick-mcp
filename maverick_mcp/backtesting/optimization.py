"""Strategy optimization utilities for VectorBT."""

from typing import Any

import numpy as np
import pandas as pd


class StrategyOptimizer:
    """Optimizer for trading strategy parameters."""

    def __init__(self, engine):
        """Initialize optimizer with VectorBT engine.

        Args:
            engine: VectorBTEngine instance
        """
        self.engine = engine

    def generate_param_grid(
        self, strategy_type: str, optimization_level: str = "medium"
    ) -> dict[str, list]:
        """Generate parameter grid based on strategy and optimization level.

        Args:
            strategy_type: Type of strategy
            optimization_level: Level of optimization (coarse, medium, fine)

        Returns:
            Parameter grid for optimization
        """
        if strategy_type == "sma_cross":
            return self._sma_param_grid(optimization_level)
        elif strategy_type == "rsi":
            return self._rsi_param_grid(optimization_level)
        elif strategy_type == "macd":
            return self._macd_param_grid(optimization_level)
        elif strategy_type == "bollinger":
            return self._bollinger_param_grid(optimization_level)
        elif strategy_type == "momentum":
            return self._momentum_param_grid(optimization_level)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

    def _sma_param_grid(self, level: str) -> dict[str, list]:
        """Generate SMA crossover parameter grid."""
        if level == "coarse":
            return {
                "fast_period": [5, 10, 20],
                "slow_period": [20, 50, 100],
            }
        elif level == "fine":
            return {
                "fast_period": list(range(5, 25, 2)),
                "slow_period": list(range(20, 101, 5)),
            }
        else:  # medium
            return {
                "fast_period": [5, 10, 15, 20],
                "slow_period": [20, 30, 50, 100],
            }

    def _rsi_param_grid(self, level: str) -> dict[str, list]:
        """Generate RSI parameter grid."""
        if level == "coarse":
            return {
                "period": [7, 14, 21],
                "oversold": [20, 30],
                "overbought": [70, 80],
            }
        elif level == "fine":
            return {
                "period": list(range(7, 22, 2)),
                "oversold": list(range(20, 41, 5)),
                "overbought": list(range(60, 81, 5)),
            }
        else:  # medium
            return {
                "period": [7, 14, 21],
                "oversold": [20, 25, 30, 35],
                "overbought": [65, 70, 75, 80],
            }

    def _macd_param_grid(self, level: str) -> dict[str, list]:
        """Generate MACD parameter grid."""
        if level == "coarse":
            return {
                "fast_period": [8, 12],
                "slow_period": [21, 26],
                "signal_period": [9],
            }
        elif level == "fine":
            return {
                "fast_period": list(range(8, 15)),
                "slow_period": list(range(20, 31)),
                "signal_period": list(range(7, 12)),
            }
        else:  # medium
            return {
                "fast_period": [8, 10, 12, 14],
                "slow_period": [21, 24, 26, 30],
                "signal_period": [7, 9, 11],
            }

    def _bollinger_param_grid(self, level: str) -> dict[str, list]:
        """Generate Bollinger Bands parameter grid."""
        if level == "coarse":
            return {
                "period": [10, 20],
                "std_dev": [1.5, 2.0, 2.5],
            }
        elif level == "fine":
            return {
                "period": list(range(10, 31, 2)),
                "std_dev": np.arange(1.0, 3.1, 0.25).tolist(),
            }
        else:  # medium
            return {
                "period": [10, 15, 20, 25],
                "std_dev": [1.5, 2.0, 2.5, 3.0],
            }

    def _momentum_param_grid(self, level: str) -> dict[str, list]:
        """Generate momentum parameter grid."""
        if level == "coarse":
            return {
                "lookback": [10, 20, 30],
                "threshold": [0.03, 0.05, 0.10],
            }
        elif level == "fine":
            return {
                "lookback": list(range(10, 41, 2)),
                "threshold": np.arange(0.02, 0.11, 0.01).tolist(),
            }
        else:  # medium
            return {
                "lookback": [10, 15, 20, 25, 30],
                "threshold": [0.02, 0.03, 0.05, 0.07, 0.10],
            }

    async def walk_forward_analysis(
        self,
        symbol: str,
        strategy_type: str,
        parameters: dict[str, Any],
        start_date: str,
        end_date: str,
        window_size: int = 252,  # Trading days in a year
        step_size: int = 63,  # Trading days in a quarter
        optimization_window: int = 504,  # 2 years for optimization
    ) -> dict[str, Any]:
        """Perform walk-forward analysis.

        Args:
            symbol: Stock symbol
            strategy_type: Strategy type
            parameters: Initial parameters
            start_date: Start date
            end_date: End date
            window_size: Test window size in days
            step_size: Step size for rolling window
            optimization_window: Optimization window size

        Returns:
            Walk-forward analysis results
        """
        results = []

        # Convert dates to pandas datetime
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        current = start + pd.Timedelta(days=optimization_window)

        while current <= end:
            # Optimization period
            opt_start = current - pd.Timedelta(days=optimization_window)
            opt_end = current

            # Test period
            test_start = current
            test_end = min(current + pd.Timedelta(days=window_size), end)

            # Optimize on training data
            param_grid = self.generate_param_grid(strategy_type, "coarse")
            optimization = await self.engine.optimize_parameters(
                symbol=symbol,
                strategy_type=strategy_type,
                param_grid=param_grid,
                start_date=opt_start.strftime("%Y-%m-%d"),
                end_date=opt_end.strftime("%Y-%m-%d"),
                top_n=1,
            )

            best_params = optimization["best_parameters"]

            # Test on out-of-sample data
            if test_start < test_end:
                test_result = await self.engine.run_backtest(
                    symbol=symbol,
                    strategy_type=strategy_type,
                    parameters=best_params,
                    start_date=test_start.strftime("%Y-%m-%d"),
                    end_date=test_end.strftime("%Y-%m-%d"),
                )

                results.append(
                    {
                        "period": f"{test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}",
                        "parameters": best_params,
                        "in_sample_sharpe": optimization["best_metric_value"],
                        "out_sample_return": test_result["metrics"]["total_return"],
                        "out_sample_sharpe": test_result["metrics"]["sharpe_ratio"],
                        "out_sample_drawdown": test_result["metrics"]["max_drawdown"],
                    }
                )

            # Move window forward
            current += pd.Timedelta(days=step_size)

        # Calculate aggregate metrics
        if results:
            avg_return = np.mean([r["out_sample_return"] for r in results])
            avg_sharpe = np.mean([r["out_sample_sharpe"] for r in results])
            avg_drawdown = np.mean([r["out_sample_drawdown"] for r in results])
            consistency = sum(1 for r in results if r["out_sample_return"] > 0) / len(
                results
            )
        else:
            avg_return = avg_sharpe = avg_drawdown = consistency = 0

        return {
            "symbol": symbol,
            "strategy": strategy_type,
            "periods_tested": len(results),
            "average_return": avg_return,
            "average_sharpe": avg_sharpe,
            "average_drawdown": avg_drawdown,
            "consistency": consistency,
            "walk_forward_results": results,
            "summary": self._generate_wf_summary(avg_return, avg_sharpe, consistency),
        }

    def _generate_wf_summary(
        self, avg_return: float, avg_sharpe: float, consistency: float
    ) -> str:
        """Generate walk-forward analysis summary."""
        summary = f"Walk-forward analysis shows {avg_return * 100:.1f}% average return "
        summary += f"with Sharpe ratio of {avg_sharpe:.2f}. "
        summary += f"Strategy was profitable in {consistency * 100:.0f}% of periods. "

        if avg_sharpe >= 1.0 and consistency >= 0.7:
            summary += "Results indicate robust performance across different market conditions."
        elif avg_sharpe >= 0.5 and consistency >= 0.5:
            summary += "Results show moderate robustness with room for improvement."
        else:
            summary += "Results suggest the strategy may not be robust to changing market conditions."

        return summary

    async def monte_carlo_simulation(
        self,
        backtest_results: dict[str, Any],
        num_simulations: int = 1000,
        confidence_levels: list[float] | None = None,
    ) -> dict[str, Any]:
        """Run Monte Carlo simulation on backtest results.

        Args:
            backtest_results: Results from run_backtest
            num_simulations: Number of simulations to run
            confidence_levels: Confidence levels for percentiles

        Returns:
            Monte Carlo simulation results
        """
        if confidence_levels is None:
            confidence_levels = [0.05, 0.25, 0.50, 0.75, 0.95]
        trades = backtest_results.get("trades", [])

        if not trades:
            return {"error": "No trades to simulate"}

        # Extract returns from trades
        trade_returns = [t["return"] for t in trades]

        # Run simulations
        simulated_returns = []
        simulated_drawdowns = []

        for _ in range(num_simulations):
            # Bootstrap sample with replacement
            sampled_returns = np.random.choice(
                trade_returns, size=len(trade_returns), replace=True
            )

            # Calculate cumulative return
            cumulative = np.cumprod(1 + np.array(sampled_returns))
            total_return = cumulative[-1] - 1

            # Calculate max drawdown
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = np.min(drawdown)

            simulated_returns.append(total_return)
            simulated_drawdowns.append(max_drawdown)

        # Calculate percentiles
        return_percentiles = np.percentile(
            simulated_returns, np.array(confidence_levels) * 100
        )
        drawdown_percentiles = np.percentile(
            simulated_drawdowns, np.array(confidence_levels) * 100
        )

        return {
            "num_simulations": num_simulations,
            "expected_return": np.mean(simulated_returns),
            "return_std": np.std(simulated_returns),
            "return_percentiles": dict(
                zip(
                    [f"p{int(cl * 100)}" for cl in confidence_levels],
                    return_percentiles.tolist(),
                    strict=False,
                )
            ),
            "expected_drawdown": np.mean(simulated_drawdowns),
            "drawdown_std": np.std(simulated_drawdowns),
            "drawdown_percentiles": dict(
                zip(
                    [f"p{int(cl * 100)}" for cl in confidence_levels],
                    drawdown_percentiles.tolist(),
                    strict=False,
                )
            ),
            "probability_profit": sum(1 for r in simulated_returns if r > 0)
            / num_simulations,
            "var_95": return_percentiles[0],  # Value at Risk at 95% confidence
            "summary": self._generate_mc_summary(
                np.mean(simulated_returns),
                return_percentiles[0],
                sum(1 for r in simulated_returns if r > 0) / num_simulations,
            ),
        }

    def _generate_mc_summary(
        self, expected_return: float, var_95: float, prob_profit: float
    ) -> str:
        """Generate Monte Carlo simulation summary."""
        summary = f"Monte Carlo simulation shows {expected_return * 100:.1f}% expected return "
        summary += f"with {prob_profit * 100:.1f}% probability of profit. "
        summary += f"95% Value at Risk is {abs(var_95) * 100:.1f}%. "

        if prob_profit >= 0.8 and expected_return > 0.10:
            summary += "Strategy shows strong probabilistic edge."
        elif prob_profit >= 0.6 and expected_return > 0:
            summary += "Strategy shows positive expectancy with moderate confidence."
        else:
            summary += "Strategy may not have sufficient edge for live trading."

        return summary
