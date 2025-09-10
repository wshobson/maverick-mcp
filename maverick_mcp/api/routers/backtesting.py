"""MCP router for VectorBT backtesting tools."""

from typing import Any

import numpy as np
from fastmcp import Context

from maverick_mcp.backtesting import (
    BacktestAnalyzer,
    StrategyOptimizer,
    VectorBTEngine,
)
from maverick_mcp.backtesting.strategies import STRATEGY_TEMPLATES, StrategyParser
from maverick_mcp.backtesting.strategies.templates import (
    get_strategy_info,
    list_available_strategies,
)
from maverick_mcp.backtesting.visualization import (
    generate_equity_curve,
    generate_optimization_heatmap,
    generate_performance_dashboard,
    generate_trade_scatter,
)


def setup_backtesting_tools(mcp):
    """Set up VectorBT backtesting tools for MCP.

    Args:
        mcp: FastMCP instance
    """

    @mcp.tool()
    async def run_backtest(
        ctx: Context,
        symbol: str,
        strategy: str = "sma_cross",
        start_date: str | None = None,
        end_date: str | None = None,
        initial_capital: float = 10000.0,
        fast_period: str | int | None = None,
        slow_period: str | int | None = None,
        period: str | int | None = None,
        oversold: str | float | None = None,
        overbought: str | float | None = None,
        signal_period: str | int | None = None,
        std_dev: str | float | None = None,
        lookback: str | int | None = None,
        threshold: str | float | None = None,
        z_score_threshold: str | float | None = None,
        breakout_factor: str | float | None = None,
    ) -> dict[str, Any]:
        """Run a VectorBT backtest with specified strategy and parameters.

        Args:
            symbol: Stock symbol to backtest
            strategy: Strategy type (sma_cross, rsi, macd, bollinger, momentum, etc.)
            start_date: Start date (YYYY-MM-DD), defaults to 1 year ago
            end_date: End date (YYYY-MM-DD), defaults to today
            initial_capital: Starting capital for backtest
            Strategy-specific parameters passed as individual arguments (e.g., fast_period=10, slow_period=20)

        Returns:
            Comprehensive backtest results including metrics, trades, and analysis

        Examples:
            run_backtest("AAPL", "sma_cross", fast_period=10, slow_period=20)
            run_backtest("TSLA", "rsi", period=14, oversold=30, overbought=70)
        """
        from datetime import datetime, timedelta

        # Default date range
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        # Convert string parameters to appropriate types
        def convert_param(value, param_type):
            """Convert string parameter to appropriate type."""
            if value is None:
                return None
            if isinstance(value, str):
                try:
                    if param_type is int:
                        return int(value)
                    elif param_type is float:
                        return float(value)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Invalid {param_type.__name__} value: {value}") from e
            return value

        # Build parameters dict from provided arguments with type conversion
        param_map = {
            "fast_period": convert_param(fast_period, int),
            "slow_period": convert_param(slow_period, int),
            "period": convert_param(period, int),
            "oversold": convert_param(oversold, float),
            "overbought": convert_param(overbought, float),
            "signal_period": convert_param(signal_period, int),
            "std_dev": convert_param(std_dev, float),
            "lookback": convert_param(lookback, int),
            "threshold": convert_param(threshold, float),
            "z_score_threshold": convert_param(z_score_threshold, float),
            "breakout_factor": convert_param(breakout_factor, float),
        }

        # Get default parameters for strategy
        if strategy in STRATEGY_TEMPLATES:
            parameters = dict(STRATEGY_TEMPLATES[strategy]["parameters"])
            # Override with provided non-None parameters
            for param_name, param_value in param_map.items():
                if param_value is not None:
                    parameters[param_name] = param_value
        else:
            # Use only provided parameters for unknown strategies
            parameters = {k: v for k, v in param_map.items() if v is not None}

        # Initialize engine
        engine = VectorBTEngine()

        # Run backtest
        results = await engine.run_backtest(
            symbol=symbol,
            strategy_type=strategy,
            parameters=parameters,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
        )

        # Analyze results
        analyzer = BacktestAnalyzer()
        analysis = analyzer.analyze(results)

        # Combine results and analysis
        results["analysis"] = analysis

        return results

    @mcp.tool()
    async def optimize_strategy(
        ctx: Context,
        symbol: str,
        strategy: str = "sma_cross",
        start_date: str | None = None,
        end_date: str | None = None,
        optimization_metric: str = "sharpe_ratio",
        optimization_level: str = "medium",
        top_n: int = 10,
    ) -> dict[str, Any]:
        """Optimize strategy parameters using VectorBT grid search.

        Args:
            symbol: Stock symbol to optimize
            strategy: Strategy type to optimize
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            optimization_metric: Metric to optimize (sharpe_ratio, total_return, win_rate, etc.)
            optimization_level: Level of optimization (coarse, medium, fine)
            top_n: Number of top results to return

        Returns:
            Optimization results with best parameters and performance metrics
        """
        from datetime import datetime, timedelta

        # Default date range
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365 * 2)).strftime("%Y-%m-%d")

        # Initialize engine and optimizer
        engine = VectorBTEngine()
        optimizer = StrategyOptimizer(engine)

        # Generate parameter grid
        param_grid = optimizer.generate_param_grid(strategy, optimization_level)

        # Run optimization
        results = await engine.optimize_parameters(
            symbol=symbol,
            strategy_type=strategy,
            param_grid=param_grid,
            start_date=start_date,
            end_date=end_date,
            optimization_metric=optimization_metric,
            top_n=top_n,
        )

        return results

    @mcp.tool()
    async def walk_forward_analysis(
        ctx: Context,
        symbol: str,
        strategy: str = "sma_cross",
        start_date: str | None = None,
        end_date: str | None = None,
        window_size: int = 252,
        step_size: int = 63,
    ) -> dict[str, Any]:
        """Perform walk-forward analysis to test strategy robustness.

        Args:
            symbol: Stock symbol to analyze
            strategy: Strategy type
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            window_size: Test window size in trading days (default: 1 year)
            step_size: Step size for rolling window (default: 1 quarter)

        Returns:
            Walk-forward analysis results with out-of-sample performance
        """
        from datetime import datetime, timedelta

        # Default date range (3 years for walk-forward)
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")

        # Initialize engine and optimizer
        engine = VectorBTEngine()
        optimizer = StrategyOptimizer(engine)

        # Get default parameters
        parameters = STRATEGY_TEMPLATES.get(strategy, {}).get("parameters", {})

        # Run walk-forward analysis
        results = await optimizer.walk_forward_analysis(
            symbol=symbol,
            strategy_type=strategy,
            parameters=parameters,
            start_date=start_date,
            end_date=end_date,
            window_size=window_size,
            step_size=step_size,
        )

        return results

    @mcp.tool()
    async def monte_carlo_simulation(
        ctx: Context,
        symbol: str,
        strategy: str = "sma_cross",
        start_date: str | None = None,
        end_date: str | None = None,
        num_simulations: int = 1000,
        fast_period: str | int | None = None,
        slow_period: str | int | None = None,
        period: str | int | None = None,
    ) -> dict[str, Any]:
        """Run Monte Carlo simulation on backtest results.

        Args:
            symbol: Stock symbol
            strategy: Strategy type
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            num_simulations: Number of Monte Carlo simulations
            Strategy-specific parameters as individual arguments

        Returns:
            Monte Carlo simulation results with confidence intervals
        """
        from datetime import datetime, timedelta

        # Default date range
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        # Convert string parameters to appropriate types
        def convert_param(value, param_type):
            """Convert string parameter to appropriate type."""
            if value is None:
                return None
            if isinstance(value, str):
                try:
                    if param_type is int:
                        return int(value)
                    elif param_type is float:
                        return float(value)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Invalid {param_type.__name__} value: {value}") from e
            return value

        # Build parameters dict from provided arguments with type conversion
        param_map = {
            "fast_period": convert_param(fast_period, int),
            "slow_period": convert_param(slow_period, int),
            "period": convert_param(period, int),
        }

        # Get parameters
        if strategy in STRATEGY_TEMPLATES:
            parameters = dict(STRATEGY_TEMPLATES[strategy]["parameters"])
            # Override with provided non-None parameters
            for param_name, param_value in param_map.items():
                if param_value is not None:
                    parameters[param_name] = param_value
        else:
            # Use only provided parameters for unknown strategies
            parameters = {k: v for k, v in param_map.items() if v is not None}

        # Run backtest first
        engine = VectorBTEngine()
        backtest_results = await engine.run_backtest(
            symbol=symbol,
            strategy_type=strategy,
            parameters=parameters,
            start_date=start_date,
            end_date=end_date,
        )

        # Run Monte Carlo simulation
        optimizer = StrategyOptimizer(engine)
        mc_results = await optimizer.monte_carlo_simulation(
            backtest_results=backtest_results,
            num_simulations=num_simulations,
        )

        return mc_results

    @mcp.tool()
    async def compare_strategies(
        ctx: Context,
        symbol: str,
        strategies: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """Compare multiple strategies on the same symbol.

        Args:
            symbol: Stock symbol
            strategies: List of strategy types to compare (defaults to all)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Comparison results with rankings and analysis
        """
        from datetime import datetime, timedelta

        # Default date range
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        # Default to comparing top strategies
        if not strategies:
            strategies = ["sma_cross", "rsi", "macd", "bollinger", "momentum"]

        # Run backtests for each strategy
        engine = VectorBTEngine()
        results_list = []

        for strategy in strategies:
            try:
                # Get default parameters
                parameters = STRATEGY_TEMPLATES.get(strategy, {}).get("parameters", {})

                # Run backtest
                results = await engine.run_backtest(
                    symbol=symbol,
                    strategy_type=strategy,
                    parameters=parameters,
                    start_date=start_date,
                    end_date=end_date,
                )
                results_list.append(results)
            except Exception:
                # Skip failed strategies
                continue

        # Compare results
        analyzer = BacktestAnalyzer()
        comparison = analyzer.compare_strategies(results_list)

        return comparison

    @mcp.tool()
    async def list_strategies(ctx: Context) -> dict[str, Any]:
        """List all available VectorBT strategies with descriptions.

        Returns:
            Dictionary of available strategies and their information
        """
        strategies = {}

        for strategy_type in list_available_strategies():
            strategies[strategy_type] = get_strategy_info(strategy_type)

        return {
            "available_strategies": strategies,
            "total_count": len(strategies),
            "categories": {
                "trend_following": ["sma_cross", "ema_cross", "macd", "breakout"],
                "mean_reversion": ["rsi", "bollinger", "mean_reversion"],
                "momentum": ["momentum", "volume_momentum"],
            },
        }

    @mcp.tool()
    async def parse_strategy(ctx: Context, description: str) -> dict[str, Any]:
        """Parse natural language strategy description into VectorBT parameters.

        Args:
            description: Natural language description of trading strategy

        Returns:
            Parsed strategy configuration with type and parameters

        Examples:
            "Buy when RSI is below 30 and sell when above 70"
            "Use 10-day and 20-day moving average crossover"
            "MACD strategy with standard parameters"
        """
        parser = StrategyParser()
        config = parser.parse_simple(description)

        # Validate the parsed strategy
        if parser.validate_strategy(config):
            return {
                "success": True,
                "strategy": config,
                "message": f"Successfully parsed as {config['strategy_type']} strategy",
            }
        else:
            return {
                "success": False,
                "strategy": config,
                "message": "Could not fully parse strategy, using defaults",
            }

    @mcp.tool()
    async def backtest_portfolio(
        ctx: Context,
        symbols: list[str],
        strategy: str = "sma_cross",
        start_date: str | None = None,
        end_date: str | None = None,
        initial_capital: float = 10000.0,
        position_size: float = 0.1,
        fast_period: str | int | None = None,
        slow_period: str | int | None = None,
        period: str | int | None = None,
    ) -> dict[str, Any]:
        """Backtest a strategy across multiple symbols (portfolio).

        Args:
            symbols: List of stock symbols
            strategy: Strategy type to apply
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Starting capital
            position_size: Position size per symbol (0.1 = 10%)
            Strategy-specific parameters as individual arguments

        Returns:
            Portfolio backtest results with aggregate metrics
        """
        from datetime import datetime, timedelta

        # Default date range
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        # Convert string parameters to appropriate types
        def convert_param(value, param_type):
            """Convert string parameter to appropriate type."""
            if value is None:
                return None
            if isinstance(value, str):
                try:
                    if param_type is int:
                        return int(value)
                    elif param_type is float:
                        return float(value)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Invalid {param_type.__name__} value: {value}") from e
            return value

        # Build parameters dict from provided arguments with type conversion
        param_map = {
            "fast_period": convert_param(fast_period, int),
            "slow_period": convert_param(slow_period, int),
            "period": convert_param(period, int),
        }

        # Get parameters
        if strategy in STRATEGY_TEMPLATES:
            parameters = dict(STRATEGY_TEMPLATES[strategy]["parameters"])
            # Override with provided non-None parameters
            for param_name, param_value in param_map.items():
                if param_value is not None:
                    parameters[param_name] = param_value
        else:
            # Use only provided parameters for unknown strategies
            parameters = {k: v for k, v in param_map.items() if v is not None}

        # Run backtests for each symbol
        engine = VectorBTEngine()
        portfolio_results = []
        capital_per_symbol = initial_capital * position_size

        for symbol in symbols:
            try:
                results = await engine.run_backtest(
                    symbol=symbol,
                    strategy_type=strategy,
                    parameters=parameters,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=capital_per_symbol,
                )
                portfolio_results.append(results)
            except Exception:
                # Skip failed symbols
                continue

        if not portfolio_results:
            return {"error": "No symbols could be backtested"}

        # Aggregate portfolio metrics
        total_return = sum(
            r["metrics"]["total_return"] for r in portfolio_results
        ) / len(portfolio_results)
        avg_sharpe = sum(r["metrics"]["sharpe_ratio"] for r in portfolio_results) / len(
            portfolio_results
        )
        max_drawdown = max(r["metrics"]["max_drawdown"] for r in portfolio_results)
        total_trades = sum(r["metrics"]["total_trades"] for r in portfolio_results)

        return {
            "portfolio_metrics": {
                "symbols_tested": len(portfolio_results),
                "total_return": total_return,
                "average_sharpe": avg_sharpe,
                "max_drawdown": max_drawdown,
                "total_trades": total_trades,
            },
            "individual_results": portfolio_results,
            "summary": f"Portfolio backtest of {len(portfolio_results)} symbols with {strategy} strategy",
        }

    @mcp.tool()
    async def generate_backtest_charts(
        ctx: Context,
        symbol: str,
        strategy: str = "sma_cross",
        start_date: str | None = None,
        end_date: str | None = None,
        theme: str = "light",
    ) -> dict[str, str]:
        """Generate comprehensive charts for a backtest.

        Args:
            symbol: Stock symbol
            strategy: Strategy type
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            theme: Chart theme (light or dark)

        Returns:
            Dictionary of base64-encoded chart images
        """
        from datetime import datetime, timedelta

        import pandas as pd

        # Default date range
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        # Run backtest
        engine = VectorBTEngine()

        # Get default parameters for the strategy
        from maverick_mcp.backtesting.strategies import STRATEGY_TEMPLATES

        parameters = STRATEGY_TEMPLATES.get(strategy, {}).get("parameters", {})

        results = await engine.run_backtest(
            symbol=symbol,
            strategy_type=strategy,
            parameters=parameters,
            start_date=start_date,
            end_date=end_date,
        )

        # Prepare data for charts
        equity_curve_data = results["equity_curve"]
        drawdown_data = results["drawdown_series"]

        # Convert to pandas Series for charting
        returns = pd.Series(equity_curve_data)
        drawdown = pd.Series(drawdown_data)
        trades = pd.DataFrame(results["trades"])

        # Generate charts
        charts = {
            "equity_curve": generate_equity_curve(
                returns, drawdown, f"{symbol} {strategy} Equity Curve", theme
            ),
            "trade_scatter": generate_trade_scatter(
                returns, trades, f"{symbol} {strategy} Trades", theme
            ),
            "performance_dashboard": generate_performance_dashboard(
                results["metrics"], f"{symbol} {strategy} Performance", theme
            ),
        }

        return charts

    @mcp.tool()
    async def generate_optimization_charts(
        ctx: Context,
        symbol: str,
        strategy: str = "sma_cross",
        start_date: str | None = None,
        end_date: str | None = None,
        theme: str = "light",
    ) -> dict[str, str]:
        """Generate chart for strategy parameter optimization.

        Args:
            symbol: Stock symbol
            strategy: Strategy type
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            theme: Chart theme (light or dark)

        Returns:
            Dictionary of base64-encoded chart images
        """
        from datetime import datetime, timedelta

        # Default date range
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        # Run optimization
        engine = VectorBTEngine()
        optimizer = StrategyOptimizer(engine)
        param_grid = optimizer.generate_param_grid(strategy, "medium")

        # Create optimization results dictionary for heatmap
        optimization_results = {}
        for param_set, results in param_grid.items():
            optimization_results[str(param_set)] = {
                "performance": results.get("total_return", 0)
            }

        # Generate optimization heatmap
        heatmap = generate_optimization_heatmap(
            optimization_results, f"{symbol} {strategy} Parameter Optimization", theme
        )

        return {"optimization_heatmap": heatmap}

    # ============ ML-ENHANCED STRATEGY TOOLS ============

    @mcp.tool()
    async def run_ml_strategy_backtest(
        ctx: Context,
        symbol: str,
        strategy_type: str = "ml_predictor",
        start_date: str | None = None,
        end_date: str | None = None,
        initial_capital: float = 10000.0,
        train_ratio: float = 0.8,
        model_type: str = "random_forest",
        n_estimators: int = 100,
        max_depth: int | None = None,
        learning_rate: float = 0.01,
        adaptation_method: str = "gradient",
    ) -> dict[str, Any]:
        """Run backtest using ML-enhanced strategies.

        Args:
            symbol: Stock symbol to backtest
            strategy_type: ML strategy type (ml_predictor, adaptive, ensemble, regime_aware)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Initial capital amount
            train_ratio: Ratio of data for training (0.0-1.0)
            Strategy-specific parameters passed as individual arguments

        Returns:
            Backtest results with ML-specific metrics
        """
        from datetime import datetime, timedelta

        from maverick_mcp.backtesting.strategies.ml import (
            AdaptiveStrategy,
            MLPredictor,
            RegimeAwareStrategy,
            StrategyEnsemble,
        )
        from maverick_mcp.backtesting.strategies.templates import (
            SimpleMovingAverageStrategy,
        )

        # Default date range
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=730)).strftime(
                "%Y-%m-%d"
            )  # 2 years for ML

        # Get historical data
        engine = VectorBTEngine()
        data = await engine.get_historical_data(symbol, start_date, end_date)

        if len(data) < 100:
            return {"error": "Insufficient data for ML strategy"}

        # Split data for training/testing
        split_idx = int(len(data) * train_ratio)
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]

        try:
            # Create ML strategy based on type
            if strategy_type == "ml_predictor":
                ml_strategy = MLPredictor(
                    model_type=model_type,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                )
                # Train the model
                training_metrics = ml_strategy.train(train_data)

            elif strategy_type == "adaptive":
                base_strategy = SimpleMovingAverageStrategy()
                ml_strategy = AdaptiveStrategy(
                    base_strategy,
                    learning_rate=learning_rate,
                    adaptation_method=adaptation_method,
                )
                training_metrics = {"adaptation_method": adaptation_method}

            elif strategy_type == "ensemble":
                # Create ensemble with basic strategies
                base_strategies = [
                    SimpleMovingAverageStrategy({"fast_period": 10, "slow_period": 20}),
                    SimpleMovingAverageStrategy({"fast_period": 5, "slow_period": 15}),
                ]
                ml_strategy = StrategyEnsemble(base_strategies)
                training_metrics = {"ensemble_size": len(base_strategies)}

            elif strategy_type == "regime_aware":
                base_strategies = {
                    0: SimpleMovingAverageStrategy(
                        {"fast_period": 5, "slow_period": 20}
                    ),  # Bear
                    1: SimpleMovingAverageStrategy(
                        {"fast_period": 10, "slow_period": 30}
                    ),  # Sideways
                    2: SimpleMovingAverageStrategy(
                        {"fast_period": 20, "slow_period": 50}
                    ),  # Bull
                }
                ml_strategy = RegimeAwareStrategy(base_strategies)
                # Fit regime detector
                ml_strategy.fit_regime_detector(train_data)
                training_metrics = {"n_regimes": len(base_strategies)}

            else:
                return {"error": f"Unsupported ML strategy type: {strategy_type}"}

            # Generate signals on test data
            entry_signals, exit_signals = ml_strategy.generate_signals(test_data)

            # Run backtest analysis on test period
            analyzer = BacktestAnalyzer()
            backtest_results = await analyzer.run_vectorbt_backtest(
                data=test_data,
                entry_signals=entry_signals,
                exit_signals=exit_signals,
                initial_capital=initial_capital,
            )

            # Add ML-specific metrics
            ml_metrics = {
                "strategy_type": strategy_type,
                "training_period": len(train_data),
                "testing_period": len(test_data),
                "train_test_split": train_ratio,
                "training_metrics": training_metrics,
            }

            # Add strategy-specific analysis
            if hasattr(ml_strategy, "get_feature_importance"):
                ml_metrics["feature_importance"] = ml_strategy.get_feature_importance()

            if hasattr(ml_strategy, "get_regime_analysis"):
                ml_metrics["regime_analysis"] = ml_strategy.get_regime_analysis()

            if hasattr(ml_strategy, "get_strategy_weights"):
                ml_metrics["strategy_weights"] = ml_strategy.get_strategy_weights()

            backtest_results["ml_metrics"] = ml_metrics

            return backtest_results

        except Exception as e:
            return {"error": f"ML backtest failed: {str(e)}"}

    @mcp.tool()
    async def train_ml_predictor(
        ctx: Context,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        model_type: str = "random_forest",
        target_periods: int = 5,
        return_threshold: float = 0.02,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
    ) -> dict[str, Any]:
        """Train an ML predictor model for trading signals.

        Args:
            symbol: Stock symbol to train on
            start_date: Start date for training data
            end_date: End date for training data
            model_type: ML model type (random_forest)
            target_periods: Forward periods for target variable
            return_threshold: Return threshold for signal classification
            n_estimators, max_depth, min_samples_split: Model-specific parameters

        Returns:
            Training results and model metrics
        """
        from datetime import datetime, timedelta

        from maverick_mcp.backtesting.strategies.ml import MLPredictor

        # Default date range (2 years for good ML training)
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

        try:
            # Get training data
            engine = VectorBTEngine()
            data = await engine.get_historical_data(symbol, start_date, end_date)

            if len(data) < 200:
                return {
                    "error": "Insufficient data for ML training (minimum 200 data points)"
                }

            # Create and train ML predictor
            ml_predictor = MLPredictor(
                model_type=model_type,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
            )
            training_metrics = ml_predictor.train(
                data=data,
                target_periods=target_periods,
                return_threshold=return_threshold,
            )

            # Create model parameters dictionary
            model_params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
            }
            # Add training details
            training_results = {
                "symbol": symbol,
                "model_type": model_type,
                "training_period": f"{start_date} to {end_date}",
                "data_points": len(data),
                "target_periods": target_periods,
                "return_threshold": return_threshold,
                "model_parameters": model_params,
                "training_metrics": training_metrics,
            }

            return training_results

        except Exception as e:
            return {"error": f"ML training failed: {str(e)}"}

    @mcp.tool()
    async def analyze_market_regimes(
        ctx: Context,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        method: str = "hmm",
        n_regimes: int = 3,
        lookback_period: int = 50,
    ) -> dict[str, Any]:
        """Analyze market regimes for a stock using ML methods.

        Args:
            symbol: Stock symbol to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            method: Detection method (hmm, kmeans, threshold)
            n_regimes: Number of regimes to detect
            lookback_period: Lookback period for regime detection

        Returns:
            Market regime analysis results
        """
        from datetime import datetime, timedelta

        from maverick_mcp.backtesting.strategies.ml.regime_aware import (
            MarketRegimeDetector,
        )

        # Default date range
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        try:
            # Get historical data
            engine = VectorBTEngine()
            data = await engine.get_historical_data(symbol, start_date, end_date)

            if len(data) < lookback_period + 50:
                return {
                    "error": f"Insufficient data for regime analysis (minimum {lookback_period + 50} data points)"
                }

            # Create regime detector and analyze
            regime_detector = MarketRegimeDetector(
                method=method, n_regimes=n_regimes, lookback_period=lookback_period
            )

            # Fit regime detector
            regime_detector.fit_regimes(data)

            # Analyze regimes over time
            regime_history = []
            regime_probabilities = []

            for i in range(lookback_period, len(data)):
                window_data = data.iloc[i - lookback_period : i + 1]
                current_regime = regime_detector.detect_current_regime(window_data)
                regime_probs = regime_detector.get_regime_probabilities(window_data)

                regime_history.append(
                    {
                        "date": data.index[i].strftime("%Y-%m-%d"),
                        "regime": int(current_regime),
                        "probabilities": regime_probs.tolist(),
                    }
                )
                regime_probabilities.append(regime_probs)

            # Calculate regime statistics
            regimes = [r["regime"] for r in regime_history]
            regime_counts = {i: regimes.count(i) for i in range(n_regimes)}
            regime_percentages = {
                k: (v / len(regimes)) * 100 for k, v in regime_counts.items()
            }

            # Calculate average regime durations
            regime_durations = {i: [] for i in range(n_regimes)}
            current_regime = regimes[0]
            duration = 1

            for regime in regimes[1:]:
                if regime == current_regime:
                    duration += 1
                else:
                    regime_durations[current_regime].append(duration)
                    current_regime = regime
                    duration = 1
            regime_durations[current_regime].append(duration)

            avg_durations = {
                k: np.mean(v) if v else 0 for k, v in regime_durations.items()
            }

            analysis_results = {
                "symbol": symbol,
                "analysis_period": f"{start_date} to {end_date}",
                "method": method,
                "n_regimes": n_regimes,
                "regime_names": {
                    0: "Bear/Declining",
                    1: "Sideways/Uncertain",
                    2: "Bull/Trending",
                },
                "current_regime": regimes[-1] if regimes else 1,
                "regime_counts": regime_counts,
                "regime_percentages": regime_percentages,
                "average_regime_durations": avg_durations,
                "recent_regime_history": regime_history[-20:],  # Last 20 periods
                "total_regime_switches": len(
                    [i for i in range(1, len(regimes)) if regimes[i] != regimes[i - 1]]
                ),
            }

            return analysis_results

        except Exception as e:
            return {"error": f"Regime analysis failed: {str(e)}"}

    @mcp.tool()
    async def create_strategy_ensemble(
        ctx: Context,
        symbols: list[str],
        base_strategies: list[str] | None = None,
        weighting_method: str = "performance",
        start_date: str | None = None,
        end_date: str | None = None,
        initial_capital: float = 10000.0,
    ) -> dict[str, Any]:
        """Create and backtest a strategy ensemble across multiple symbols.

        Args:
            symbols: List of stock symbols
            base_strategies: List of base strategy names to ensemble
            weighting_method: Weighting method (performance, equal, volatility)
            start_date: Start date for backtesting
            end_date: End date for backtesting
            initial_capital: Initial capital per symbol

        Returns:
            Ensemble backtest results with strategy weights
        """
        from datetime import datetime, timedelta

        from maverick_mcp.backtesting.strategies.ml import StrategyEnsemble
        from maverick_mcp.backtesting.strategies.templates import (
            MACDStrategy,
            RSIStrategy,
            SimpleMovingAverageStrategy,
        )

        # Default strategies if none provided
        if base_strategies is None:
            base_strategies = ["sma_cross", "rsi", "macd"]

        # Default date range
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        try:
            # Create base strategy instances
            strategy_instances = []
            for strategy_name in base_strategies:
                if strategy_name == "sma_cross":
                    strategy_instances.append(SimpleMovingAverageStrategy())
                elif strategy_name == "rsi":
                    strategy_instances.append(RSIStrategy())
                elif strategy_name == "macd":
                    strategy_instances.append(MACDStrategy())
                # Add more strategies as needed

            if not strategy_instances:
                return {"error": "No valid base strategies provided"}

            # Create ensemble strategy
            ensemble = StrategyEnsemble(
                strategies=strategy_instances, weighting_method=weighting_method
            )

            # Run ensemble backtest on multiple symbols
            ensemble_results = []
            total_return = 0
            total_trades = 0

            for symbol in symbols[:5]:  # Limit to 5 symbols for performance
                try:
                    # Get data and run backtest
                    engine = VectorBTEngine()
                    data = await engine.get_historical_data(
                        symbol, start_date, end_date
                    )

                    if len(data) < 100:
                        continue

                    # Generate ensemble signals
                    entry_signals, exit_signals = ensemble.generate_signals(data)

                    # Run backtest
                    analyzer = BacktestAnalyzer()
                    results = await analyzer.run_vectorbt_backtest(
                        data=data,
                        entry_signals=entry_signals,
                        exit_signals=exit_signals,
                        initial_capital=initial_capital,
                    )

                    # Add ensemble-specific metrics
                    results["ensemble_metrics"] = {
                        "strategy_weights": ensemble.get_strategy_weights(),
                        "strategy_performance": ensemble.get_strategy_performance(),
                    }

                    ensemble_results.append({"symbol": symbol, "results": results})

                    total_return += results["metrics"]["total_return"]
                    total_trades += results["metrics"]["total_trades"]

                except Exception:
                    continue

            if not ensemble_results:
                return {"error": "No symbols could be processed"}

            # Calculate aggregate metrics
            avg_return = total_return / len(ensemble_results)
            avg_trades = total_trades / len(ensemble_results)

            return {
                "ensemble_summary": {
                    "symbols_tested": len(ensemble_results),
                    "base_strategies": base_strategies,
                    "weighting_method": weighting_method,
                    "average_return": avg_return,
                    "total_trades": total_trades,
                    "average_trades_per_symbol": avg_trades,
                },
                "individual_results": ensemble_results,
                "final_strategy_weights": ensemble.get_strategy_weights(),
                "strategy_performance_analysis": ensemble.get_strategy_performance(),
            }

        except Exception as e:
            return {"error": f"Ensemble creation failed: {str(e)}"}
