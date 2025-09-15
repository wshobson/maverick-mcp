"""MCP Introspection Tools for Better Discovery and Understanding."""

from typing import Any

from fastmcp import FastMCP


def register_introspection_tools(mcp: FastMCP) -> None:
    """Register introspection tools for better discovery."""

    @mcp.tool(name="discover_capabilities")
    async def discover_capabilities() -> dict[str, Any]:
        """
        Discover all available capabilities of the MaverickMCP server.

        This tool provides comprehensive information about:
        - Available strategies (traditional and ML)
        - Tool categories and their functions
        - Parameter requirements for each strategy
        - Example usage patterns

        Use this as your first tool to understand what's available.
        """
        return {
            "server_info": {
                "name": "MaverickMCP",
                "version": "1.0.0",
                "description": "Advanced stock analysis and backtesting MCP server",
            },
            "capabilities": {
                "backtesting": {
                    "description": "Run and optimize trading strategies",
                    "strategies_available": 15,
                    "ml_strategies": ["online_learning", "regime_aware", "ensemble"],
                    "traditional_strategies": [
                        "sma_cross",
                        "rsi",
                        "macd",
                        "bollinger",
                        "momentum",
                        "ema_cross",
                        "mean_reversion",
                        "breakout",
                        "volume_momentum",
                    ],
                    "features": [
                        "parameter_optimization",
                        "strategy_comparison",
                        "walk_forward_analysis",
                    ],
                },
                "technical_analysis": {
                    "description": "Calculate technical indicators and patterns",
                    "indicators": [
                        "SMA",
                        "EMA",
                        "RSI",
                        "MACD",
                        "Bollinger Bands",
                        "Support/Resistance",
                    ],
                    "chart_analysis": True,
                    "pattern_recognition": True,
                },
                "screening": {
                    "description": "Pre-calculated S&P 500 screening results",
                    "strategies": [
                        "maverick_bullish",
                        "maverick_bearish",
                        "supply_demand_breakouts",
                    ],
                    "database": "520 S&P 500 stocks pre-seeded",
                },
                "research": {
                    "description": "AI-powered research with parallel execution",
                    "features": [
                        "comprehensive_research",
                        "company_analysis",
                        "sentiment_analysis",
                    ],
                    "performance": "7-256x speedup with parallel agents",
                    "ai_models": "400+ models via OpenRouter",
                },
            },
            "quick_start": {
                "first_command": "Run: discover_capabilities() to see this info",
                "simple_backtest": "run_backtest(symbol='AAPL', strategy_type='sma_cross')",
                "ml_strategy": "run_backtest(symbol='TSLA', strategy_type='online_learning')",
                "get_help": "Use prompts like 'backtest_strategy_guide' for detailed guides",
            },
        }

    @mcp.tool(name="list_all_strategies")
    async def list_all_strategies() -> list[dict[str, Any]]:
        """
        List all available backtesting strategies with their parameters.

        Returns detailed information about each strategy including:
        - Strategy name and description
        - Required and optional parameters
        - Default parameter values
        - Example usage
        """
        strategies = []  # Return as array

        # Traditional strategies
        strategies.extend([
                {
                    "name": "sma_cross",
                    "description": "Simple Moving Average Crossover",
                    "parameters": {
                        "fast_period": {
                            "type": "int",
                            "default": 10,
                            "description": "Fast MA period",
                        },
                        "slow_period": {
                            "type": "int",
                            "default": 20,
                            "description": "Slow MA period",
                        },
                    },
                    "example": "run_backtest(symbol='AAPL', strategy_type='sma_cross', fast_period=10, slow_period=20)",
                },
                {
                    "name": "rsi",
                    "description": "RSI Mean Reversion",
                    "parameters": {
                        "period": {
                            "type": "int",
                            "default": 14,
                            "description": "RSI calculation period",
                        },
                        "oversold": {
                            "type": "int",
                            "default": 30,
                            "description": "Oversold threshold",
                        },
                        "overbought": {
                            "type": "int",
                            "default": 70,
                            "description": "Overbought threshold",
                        },
                    },
                    "example": "run_backtest(symbol='MSFT', strategy_type='rsi', period=14)",
                },
                {
                    "name": "macd",
                    "description": "MACD Signal Line Crossover",
                    "parameters": {
                        "fast_period": {
                            "type": "int",
                            "default": 12,
                            "description": "Fast EMA period",
                        },
                        "slow_period": {
                            "type": "int",
                            "default": 26,
                            "description": "Slow EMA period",
                        },
                        "signal_period": {
                            "type": "int",
                            "default": 9,
                            "description": "Signal line period",
                        },
                    },
                    "example": "run_backtest(symbol='GOOGL', strategy_type='macd')",
                },
                {
                    "name": "bollinger",
                    "description": "Bollinger Bands Mean Reversion",
                    "parameters": {
                        "period": {
                            "type": "int",
                            "default": 20,
                            "description": "BB calculation period",
                        },
                        "std_dev": {
                            "type": "float",
                            "default": 2,
                            "description": "Standard deviations",
                        },
                    },
                    "example": "run_backtest(symbol='AMZN', strategy_type='bollinger')",
                },
                {
                    "name": "momentum",
                    "description": "Momentum Trading Strategy",
                    "parameters": {
                        "period": {
                            "type": "int",
                            "default": 10,
                            "description": "Momentum period",
                        },
                        "threshold": {
                            "type": "float",
                            "default": 0.02,
                            "description": "Entry threshold",
                        },
                    },
                    "example": "run_backtest(symbol='NVDA', strategy_type='momentum')",
                },
                {
                    "name": "ema_cross",
                    "description": "Exponential Moving Average Crossover",
                    "parameters": {
                        "fast_period": {
                            "type": "int",
                            "default": 12,
                            "description": "Fast EMA period",
                        },
                        "slow_period": {
                            "type": "int",
                            "default": 26,
                            "description": "Slow EMA period",
                        },
                    },
                    "example": "run_backtest(symbol='META', strategy_type='ema_cross')",
                },
                {
                    "name": "mean_reversion",
                    "description": "Statistical Mean Reversion",
                    "parameters": {
                        "lookback": {
                            "type": "int",
                            "default": 20,
                            "description": "Lookback period",
                        },
                        "entry_z": {
                            "type": "float",
                            "default": -2,
                            "description": "Entry z-score",
                        },
                        "exit_z": {
                            "type": "float",
                            "default": 0,
                            "description": "Exit z-score",
                        },
                    },
                    "example": "run_backtest(symbol='SPY', strategy_type='mean_reversion')",
                },
                {
                    "name": "breakout",
                    "description": "Channel Breakout Strategy",
                    "parameters": {
                        "lookback": {
                            "type": "int",
                            "default": 20,
                            "description": "Channel period",
                        },
                        "breakout_factor": {
                            "type": "float",
                            "default": 1.5,
                            "description": "Breakout multiplier",
                        },
                    },
                    "example": "run_backtest(symbol='QQQ', strategy_type='breakout')",
                },
                {
                    "name": "volume_momentum",
                    "description": "Volume-Weighted Momentum",
                    "parameters": {
                        "period": {
                            "type": "int",
                            "default": 10,
                            "description": "Momentum period",
                        },
                        "volume_factor": {
                            "type": "float",
                            "default": 1.5,
                            "description": "Volume multiplier",
                        },
                    },
                    "example": "run_backtest(symbol='TSLA', strategy_type='volume_momentum')",
                },
            ])

        # ML strategies
        strategies.extend([
                {
                    "name": "ml_predictor",
                    "description": "Machine Learning predictor using Random Forest",
                    "parameters": {
                        "model_type": {
                            "type": "str",
                            "default": "random_forest",
                            "description": "ML model type",
                        },
                        "n_estimators": {
                            "type": "int",
                            "default": 100,
                            "description": "Number of trees",
                        },
                        "max_depth": {
                            "type": "int",
                            "default": None,
                            "description": "Max tree depth",
                        },
                    },
                    "example": "run_ml_strategy_backtest(symbol='AAPL', strategy_type='ml_predictor', model_type='random_forest')",
                },
                {
                    "name": "online_learning",
                    "description": "Online learning adaptive strategy (alias for adaptive)",
                    "parameters": {
                        "learning_rate": {
                            "type": "float",
                            "default": 0.01,
                            "description": "Adaptation rate",
                        },
                        "adaptation_method": {
                            "type": "str",
                            "default": "gradient",
                            "description": "Method for adaptation",
                        },
                    },
                    "example": "run_ml_strategy_backtest(symbol='AAPL', strategy_type='online_learning')",
                },
                {
                    "name": "regime_aware",
                    "description": "Market regime detection and adaptation",
                    "parameters": {
                        "regime_window": {
                            "type": "int",
                            "default": 50,
                            "description": "Regime detection window",
                        },
                        "threshold": {
                            "type": "float",
                            "default": 0.02,
                            "description": "Regime change threshold",
                        },
                    },
                    "example": "run_backtest(symbol='SPY', strategy_type='regime_aware')",
                },
                {
                    "name": "ensemble",
                    "description": "Ensemble voting with multiple strategies",
                    "parameters": {
                        "fast_period": {
                            "type": "int",
                            "default": 10,
                            "description": "Fast MA for ensemble",
                        },
                        "slow_period": {
                            "type": "int",
                            "default": 20,
                            "description": "Slow MA for ensemble",
                        },
                        "rsi_period": {
                            "type": "int",
                            "default": 14,
                            "description": "RSI period for ensemble",
                        },
                    },
                    "example": "run_ml_strategy_backtest(symbol='MSFT', strategy_type='ensemble')",
                },
                {
                    "name": "adaptive",
                    "description": "Adaptive strategy that adjusts based on performance",
                    "parameters": {
                        "learning_rate": {
                            "type": "float",
                            "default": 0.01,
                            "description": "How quickly to adapt",
                        },
                        "adaptation_method": {
                            "type": "str",
                            "default": "gradient",
                            "description": "Method for adaptation",
                        },
                    },
                    "example": "run_ml_strategy_backtest(symbol='GOOGL', strategy_type='adaptive')",
                },
            ])

        return strategies  # Return array

    @mcp.tool(name="get_strategy_help")
    async def get_strategy_help(strategy_type: str) -> dict[str, Any]:
        """
        Get detailed help for a specific strategy.

        Args:
            strategy_type: Name of the strategy (e.g., 'sma_cross', 'online_learning')

        Returns:
            Detailed information about the strategy including theory, parameters, and best practices.
        """
        strategy_help = {
            "sma_cross": {
                "name": "Simple Moving Average Crossover",
                "theory": "Generates buy signals when fast SMA crosses above slow SMA, sell when opposite occurs",
                "best_for": "Trending markets with clear directional moves",
                "parameters": {
                    "fast_period": "Typically 10-20 days for short-term trends",
                    "slow_period": "Typically 20-50 days for medium-term trends",
                },
                "tips": [
                    "Works best in trending markets",
                    "Consider adding volume confirmation",
                    "Use wider periods for less noise",
                ],
            },
            "ml_predictor": {
                "name": "Machine Learning Predictor",
                "theory": "Uses Random Forest or other ML models to predict price movements",
                "best_for": "Complex markets with multiple factors",
                "parameters": {
                    "model_type": "Type of ML model (random_forest)",
                    "n_estimators": "Number of trees in forest (50-200)",
                    "max_depth": "Maximum tree depth (None or 5-20)",
                },
                "tips": [
                    "More estimators for better accuracy but slower",
                    "Limit depth to prevent overfitting",
                    "Requires sufficient historical data",
                ],
            },
            "online_learning": {
                "name": "Online Learning Strategy",
                "theory": "Continuously adapts strategy parameters based on recent performance",
                "best_for": "Dynamic markets with changing patterns",
                "parameters": {
                    "learning_rate": "How quickly to adapt (0.001-0.1)",
                    "adaptation_method": "Method for adaptation (gradient, bayesian)",
                },
                "tips": [
                    "Lower learning rates for stable adaptation",
                    "Works well in volatile markets",
                    "This is an alias for the adaptive strategy",
                ],
            },
            "adaptive": {
                "name": "Adaptive Strategy",
                "theory": "Dynamically adjusts strategy parameters based on performance",
                "best_for": "Markets with changing characteristics",
                "parameters": {
                    "learning_rate": "How quickly to adapt (0.001-0.1)",
                    "adaptation_method": "Method for adaptation (gradient, bayesian)",
                },
                "tips": [
                    "Start with lower learning rates",
                    "Monitor for overfitting",
                    "Works best with stable base strategy",
                ],
            },
            "ensemble": {
                "name": "Strategy Ensemble",
                "theory": "Combines multiple strategies with weighted voting",
                "best_for": "Risk reduction through diversification",
                "parameters": {
                    "base_strategies": "List of strategies to combine",
                    "weighting_method": "How to weight strategies (equal, performance, volatility)",
                },
                "tips": [
                    "Combine uncorrelated strategies",
                    "Performance weighting adapts to market",
                    "More strategies reduce single-point failure",
                ],
            },
            "regime_aware": {
                "name": "Market Regime Detection Strategy",
                "theory": "Identifies market regimes (trending vs ranging) and adapts strategy accordingly",
                "best_for": "Markets that alternate between trending and sideways movement",
                "parameters": {
                    "regime_window": "Period for regime detection (30-100 days)",
                    "threshold": "Sensitivity to regime changes (0.01-0.05)",
                },
                "tips": [
                    "Longer windows for major regime shifts",
                    "Lower thresholds for more sensitive detection",
                    "Combines well with other indicators",
                ],
            },
        }

        if strategy_type in strategy_help:
            return strategy_help[strategy_type]
        else:
            return {
                "error": f"Strategy '{strategy_type}' not found",
                "available_strategies": list(strategy_help.keys()),
                "tip": "Use list_all_strategies() to see all available strategies",
            }
