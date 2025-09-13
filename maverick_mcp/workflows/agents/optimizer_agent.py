"""
Optimizer Agent for intelligent parameter optimization.

This agent performs regime-aware parameter optimization for selected strategies,
using adaptive grid sizes and optimization metrics based on market conditions.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

from maverick_mcp.backtesting import StrategyOptimizer, VectorBTEngine
from maverick_mcp.backtesting.strategies.templates import get_strategy_info
from maverick_mcp.workflows.state import BacktestingWorkflowState

logger = logging.getLogger(__name__)


class OptimizerAgent:
    """Intelligent parameter optimizer with regime-aware optimization."""

    def __init__(
        self,
        vectorbt_engine: VectorBTEngine | None = None,
        strategy_optimizer: StrategyOptimizer | None = None,
    ):
        """Initialize optimizer agent.

        Args:
            vectorbt_engine: VectorBT backtesting engine
            strategy_optimizer: Strategy optimization engine
        """
        self.engine = vectorbt_engine or VectorBTEngine()
        self.optimizer = strategy_optimizer or StrategyOptimizer(self.engine)

        # Optimization configurations for different regimes
        self.REGIME_OPTIMIZATION_CONFIG = {
            "trending": {
                "optimization_metric": "total_return",  # Focus on capturing trends
                "grid_size": "medium",
                "min_trades": 10,
                "max_drawdown_limit": 0.25,
            },
            "ranging": {
                "optimization_metric": "sharpe_ratio",  # Focus on risk-adjusted returns
                "grid_size": "fine",  # More precision needed for ranging markets
                "min_trades": 15,
                "max_drawdown_limit": 0.15,
            },
            "volatile": {
                "optimization_metric": "calmar_ratio",  # Risk-adjusted for volatility
                "grid_size": "coarse",  # Avoid overfitting in volatile conditions
                "min_trades": 8,
                "max_drawdown_limit": 0.35,
            },
            "volatile_trending": {
                "optimization_metric": "sortino_ratio",  # Focus on downside risk
                "grid_size": "medium",
                "min_trades": 10,
                "max_drawdown_limit": 0.30,
            },
            "low_volume": {
                "optimization_metric": "win_rate",  # Consistency important in low volume
                "grid_size": "medium",
                "min_trades": 12,
                "max_drawdown_limit": 0.20,
            },
            "unknown": {
                "optimization_metric": "sharpe_ratio",  # Balanced approach
                "grid_size": "medium",
                "min_trades": 10,
                "max_drawdown_limit": 0.20,
            },
        }

        # Strategy-specific optimization parameters
        self.STRATEGY_PARAM_RANGES = {
            "sma_cross": {
                "fast_period": {
                    "coarse": [5, 10, 15],
                    "medium": [5, 8, 10, 12, 15, 20],
                    "fine": list(range(5, 21)),
                },
                "slow_period": {
                    "coarse": [20, 30, 50],
                    "medium": [20, 25, 30, 40, 50],
                    "fine": list(range(20, 51, 5)),
                },
            },
            "rsi": {
                "period": {
                    "coarse": [10, 14, 21],
                    "medium": [10, 12, 14, 16, 21],
                    "fine": list(range(8, 25)),
                },
                "oversold": {
                    "coarse": [25, 30, 35],
                    "medium": [20, 25, 30, 35],
                    "fine": list(range(15, 36, 5)),
                },
                "overbought": {
                    "coarse": [65, 70, 75],
                    "medium": [65, 70, 75, 80],
                    "fine": list(range(65, 86, 5)),
                },
            },
            "macd": {
                "fast_period": {
                    "coarse": [8, 12, 16],
                    "medium": [8, 10, 12, 14, 16],
                    "fine": list(range(8, 17)),
                },
                "slow_period": {
                    "coarse": [21, 26, 32],
                    "medium": [21, 24, 26, 28, 32],
                    "fine": list(range(21, 35)),
                },
                "signal_period": {
                    "coarse": [6, 9, 12],
                    "medium": [6, 8, 9, 10, 12],
                    "fine": list(range(6, 15)),
                },
            },
            "bollinger": {
                "period": {
                    "coarse": [15, 20, 25],
                    "medium": [15, 18, 20, 22, 25],
                    "fine": list(range(12, 28)),
                },
                "std_dev": {
                    "coarse": [1.5, 2.0, 2.5],
                    "medium": [1.5, 1.8, 2.0, 2.2, 2.5],
                    "fine": [1.0, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0],
                },
            },
            "momentum": {
                "lookback": {
                    "coarse": [10, 20, 30],
                    "medium": [10, 15, 20, 25, 30],
                    "fine": list(range(5, 31, 5)),
                },
                "threshold": {
                    "coarse": [0.03, 0.05, 0.08],
                    "medium": [0.02, 0.03, 0.05, 0.07, 0.10],
                    "fine": [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15],
                },
            },
        }

        logger.info("OptimizerAgent initialized")

    async def optimize_parameters(
        self, state: BacktestingWorkflowState
    ) -> BacktestingWorkflowState:
        """Optimize parameters for selected strategies.

        Args:
            state: Current workflow state with selected strategies

        Returns:
            Updated state with optimization results
        """
        start_time = datetime.now()

        try:
            logger.info(
                f"Optimizing parameters for {len(state.selected_strategies)} strategies on {state.symbol}"
            )

            # Get optimization configuration based on regime
            optimization_config = self._get_optimization_config(
                state.market_regime, state.regime_confidence
            )

            # Generate parameter grids for each strategy
            parameter_grids = self._generate_parameter_grids(
                state.selected_strategies, optimization_config["grid_size"]
            )

            # Optimize each strategy
            optimization_results = {}
            best_parameters = {}
            total_iterations = 0

            # Use shorter timeframe for optimization to avoid overfitting
            opt_start_date = self._calculate_optimization_window(
                state.start_date, state.end_date
            )

            for strategy in state.selected_strategies:
                try:
                    logger.info(f"Optimizing {strategy} strategy...")

                    param_grid = parameter_grids.get(strategy, {})
                    if not param_grid:
                        logger.warning(
                            f"No parameter grid for {strategy}, using defaults"
                        )
                        continue

                    # Run optimization
                    result = await self.engine.optimize_parameters(
                        symbol=state.symbol,
                        strategy_type=strategy,
                        param_grid=param_grid,
                        start_date=opt_start_date,
                        end_date=state.end_date,
                        optimization_metric=optimization_config["optimization_metric"],
                        initial_capital=state.initial_capital,
                        top_n=min(
                            10, len(state.selected_strategies) * 2
                        ),  # Adaptive top_n
                    )

                    # Filter results by quality metrics
                    filtered_result = self._filter_optimization_results(
                        result, optimization_config
                    )

                    optimization_results[strategy] = filtered_result
                    best_parameters[strategy] = filtered_result.get(
                        "best_parameters", {}
                    )
                    total_iterations += filtered_result.get("valid_combinations", 0)

                    logger.info(
                        f"Optimized {strategy}: {filtered_result.get('best_metric_value', 0):.3f} {optimization_config['optimization_metric']}"
                    )

                except Exception as e:
                    logger.error(f"Failed to optimize {strategy}: {e}")
                    # Use default parameters as fallback
                    strategy_info = get_strategy_info(strategy)
                    best_parameters[strategy] = strategy_info.get("parameters", {})
                    state.fallback_strategies_used.append(
                        f"{strategy}_optimization_fallback"
                    )

            # Update state
            state.optimization_config = optimization_config
            state.parameter_grids = parameter_grids
            state.optimization_results = optimization_results
            state.best_parameters = best_parameters
            state.optimization_iterations = total_iterations

            # Record execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            state.optimization_time_ms = execution_time

            # Update workflow status
            state.workflow_status = "validating"
            state.current_step = "optimization_completed"
            state.steps_completed.append("parameter_optimization")

            logger.info(
                f"Parameter optimization completed for {state.symbol}: "
                f"{total_iterations} combinations tested in {execution_time:.0f}ms"
            )

            return state

        except Exception as e:
            error_info = {
                "step": "parameter_optimization",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "symbol": state.symbol,
            }
            state.errors_encountered.append(error_info)

            # Fallback to default parameters
            default_params = {}
            for strategy in state.selected_strategies:
                strategy_info = get_strategy_info(strategy)
                default_params[strategy] = strategy_info.get("parameters", {})

            state.best_parameters = default_params
            state.fallback_strategies_used.append("optimization_fallback")

            logger.error(f"Parameter optimization failed for {state.symbol}: {e}")
            return state

    def _get_optimization_config(
        self, regime: str, regime_confidence: float
    ) -> dict[str, Any]:
        """Get optimization configuration based on market regime."""
        base_config = self.REGIME_OPTIMIZATION_CONFIG.get(
            regime, self.REGIME_OPTIMIZATION_CONFIG["unknown"]
        ).copy()

        # Adjust grid size based on confidence
        if regime_confidence < 0.5:
            # Low confidence -> use coarser grid to avoid overfitting
            if base_config["grid_size"] == "fine":
                base_config["grid_size"] = "medium"
            elif base_config["grid_size"] == "medium":
                base_config["grid_size"] = "coarse"

        return base_config

    def _generate_parameter_grids(
        self, strategies: list[str], grid_size: str
    ) -> dict[str, dict[str, list]]:
        """Generate parameter grids for optimization."""
        parameter_grids = {}

        for strategy in strategies:
            if strategy in self.STRATEGY_PARAM_RANGES:
                param_ranges = self.STRATEGY_PARAM_RANGES[strategy]
                grid = {}

                for param_name, size_ranges in param_ranges.items():
                    if grid_size in size_ranges:
                        grid[param_name] = size_ranges[grid_size]
                    else:
                        # Fallback to medium if requested size not available
                        grid[param_name] = size_ranges.get(
                            "medium", size_ranges["coarse"]
                        )

                parameter_grids[strategy] = grid
            else:
                # For strategies not in our predefined ranges, use default minimal grid
                parameter_grids[strategy] = self._generate_default_grid(
                    strategy, grid_size
                )

        return parameter_grids

    def _generate_default_grid(self, strategy: str, grid_size: str) -> dict[str, list]:
        """Generate default parameter grid for unknown strategies."""
        # Get strategy info to understand default parameters
        strategy_info = get_strategy_info(strategy)
        default_params = strategy_info.get("parameters", {})

        grid = {}

        # Generate basic variations around default values
        for param_name, default_value in default_params.items():
            if isinstance(default_value, int | float):
                if grid_size == "coarse":
                    variations = [
                        default_value * 0.8,
                        default_value,
                        default_value * 1.2,
                    ]
                elif grid_size == "fine":
                    variations = [
                        default_value * 0.7,
                        default_value * 0.8,
                        default_value * 0.9,
                        default_value,
                        default_value * 1.1,
                        default_value * 1.2,
                        default_value * 1.3,
                    ]
                else:  # medium
                    variations = [
                        default_value * 0.8,
                        default_value * 0.9,
                        default_value,
                        default_value * 1.1,
                        default_value * 1.2,
                    ]

                # Convert back to appropriate type and filter valid values
                if isinstance(default_value, int):
                    grid[param_name] = [max(1, int(v)) for v in variations]
                else:
                    grid[param_name] = [max(0.001, v) for v in variations]
            else:
                # For non-numeric parameters, just use the default
                grid[param_name] = [default_value]

        return grid

    def _calculate_optimization_window(self, start_date: str, end_date: str) -> str:
        """Calculate optimization window to prevent overfitting."""
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        total_days = (end_dt - start_dt).days

        # Use 70% of data for optimization, leaving 30% for validation
        opt_days = int(total_days * 0.7)
        opt_start = end_dt - timedelta(days=opt_days)

        return opt_start.strftime("%Y-%m-%d")

    def _filter_optimization_results(
        self, result: dict[str, Any], optimization_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Filter optimization results based on quality criteria."""
        if "top_results" not in result or not result["top_results"]:
            return result

        # Quality filters
        min_trades = optimization_config.get("min_trades", 10)
        max_drawdown_limit = optimization_config.get("max_drawdown_limit", 0.25)

        # Filter results by quality criteria
        filtered_results = []
        for res in result["top_results"]:
            # Check minimum trades
            if res.get("total_trades", 0) < min_trades:
                continue

            # Check maximum drawdown
            if abs(res.get("max_drawdown", 0)) > max_drawdown_limit:
                continue

            filtered_results.append(res)

        # If no results pass filters, relax criteria
        if not filtered_results and result["top_results"]:
            logger.warning("No results passed quality filters, relaxing criteria")
            # Take top results but with warning
            filtered_results = result["top_results"][:3]  # Top 3 regardless of quality

        # Update result with filtered data
        filtered_result = result.copy()
        filtered_result["top_results"] = filtered_results

        if filtered_results:
            filtered_result["best_parameters"] = filtered_results[0]["parameters"]
            filtered_result["best_metric_value"] = filtered_results[0][
                optimization_config["optimization_metric"]
            ]
        else:
            # Complete fallback
            filtered_result["best_parameters"] = {}
            filtered_result["best_metric_value"] = 0.0

        return filtered_result

    def get_optimization_summary(
        self, state: BacktestingWorkflowState
    ) -> dict[str, Any]:
        """Get summary of optimization results."""
        if not state.optimization_results:
            return {"summary": "No optimization results available"}

        summary = {
            "total_strategies": len(state.selected_strategies),
            "optimized_strategies": len(state.optimization_results),
            "total_iterations": state.optimization_iterations,
            "execution_time_ms": state.optimization_time_ms,
            "optimization_config": state.optimization_config,
            "strategy_results": {},
        }

        for strategy, results in state.optimization_results.items():
            if results:
                summary["strategy_results"][strategy] = {
                    "best_metric": results.get("best_metric_value", 0),
                    "metric_type": state.optimization_config.get(
                        "optimization_metric", "unknown"
                    ),
                    "valid_combinations": results.get("valid_combinations", 0),
                    "best_parameters": state.best_parameters.get(strategy, {}),
                }

        return summary

    async def parallel_optimization(
        self, state: BacktestingWorkflowState, max_concurrent: int = 3
    ) -> BacktestingWorkflowState:
        """Run optimization for multiple strategies in parallel."""
        if len(state.selected_strategies) <= 1:
            return await self.optimize_parameters(state)

        start_time = datetime.now()
        logger.info(
            f"Running parallel optimization for {len(state.selected_strategies)} strategies"
        )

        # Create semaphore to limit concurrent optimizations
        semaphore = asyncio.Semaphore(max_concurrent)

        async def optimize_single_strategy(strategy: str) -> tuple[str, dict[str, Any]]:
            async with semaphore:
                try:
                    optimization_config = self._get_optimization_config(
                        state.market_regime, state.regime_confidence
                    )
                    parameter_grids = self._generate_parameter_grids(
                        [strategy], optimization_config["grid_size"]
                    )

                    opt_start_date = self._calculate_optimization_window(
                        state.start_date, state.end_date
                    )

                    result = await self.engine.optimize_parameters(
                        symbol=state.symbol,
                        strategy_type=strategy,
                        param_grid=parameter_grids.get(strategy, {}),
                        start_date=opt_start_date,
                        end_date=state.end_date,
                        optimization_metric=optimization_config["optimization_metric"],
                        initial_capital=state.initial_capital,
                        top_n=10,
                    )

                    filtered_result = self._filter_optimization_results(
                        result, optimization_config
                    )
                    return strategy, filtered_result

                except Exception as e:
                    logger.error(f"Failed to optimize {strategy}: {e}")
                    return strategy, {"error": str(e)}

        # Run optimizations in parallel
        tasks = [
            optimize_single_strategy(strategy) for strategy in state.selected_strategies
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        optimization_results = {}
        best_parameters = {}
        total_iterations = 0

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Parallel optimization failed: {result}")
                continue

            strategy, opt_result = result
            if "error" not in opt_result:
                optimization_results[strategy] = opt_result
                best_parameters[strategy] = opt_result.get("best_parameters", {})
                total_iterations += opt_result.get("valid_combinations", 0)

        # Update state
        optimization_config = self._get_optimization_config(
            state.market_regime, state.regime_confidence
        )
        state.optimization_config = optimization_config
        state.optimization_results = optimization_results
        state.best_parameters = best_parameters
        state.optimization_iterations = total_iterations

        # Record execution time
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        state.optimization_time_ms = execution_time

        logger.info(f"Parallel optimization completed in {execution_time:.0f}ms")
        return state
