# Backtesting API Documentation

## Overview

The MaverickMCP backtesting system provides comprehensive strategy backtesting capabilities powered by VectorBT. It offers both traditional technical analysis strategies and advanced ML-enhanced approaches, with extensive optimization, validation, and analysis tools.

### Key Features

- **35+ Pre-built Strategies**: From simple moving averages to advanced ML ensembles
- **Strategy Optimization**: Grid search with coarse/medium/fine granularity
- **Walk-Forward Analysis**: Out-of-sample validation for strategy robustness
- **Monte Carlo Simulation**: Risk assessment with confidence intervals
- **Portfolio Backtesting**: Multi-symbol strategy application
- **Market Regime Analysis**: Intelligent strategy selection based on market conditions
- **ML-Enhanced Strategies**: Adaptive, ensemble, and regime-aware approaches
- **Comprehensive Visualization**: Charts, heatmaps, and performance dashboards

## Core Backtesting Tools

### run_backtest

Run a comprehensive backtest with specified strategy and parameters.

**Function**: `run_backtest`

**Parameters**:
- `symbol` (str, required): Stock symbol to backtest (e.g., "AAPL", "TSLA")
- `strategy` (str, default: "sma_cross"): Strategy type to use
- `start_date` (str, optional): Start date (YYYY-MM-DD), defaults to 1 year ago
- `end_date` (str, optional): End date (YYYY-MM-DD), defaults to today
- `initial_capital` (float, default: 10000.0): Starting capital for backtest

**Strategy-Specific Parameters**:
- `fast_period` (int, optional): Fast moving average period
- `slow_period` (int, optional): Slow moving average period
- `period` (int, optional): General period parameter (RSI, etc.)
- `oversold` (float, optional): RSI oversold threshold (default: 30)
- `overbought` (float, optional): RSI overbought threshold (default: 70)
- `signal_period` (int, optional): MACD signal line period
- `std_dev` (float, optional): Bollinger Bands standard deviation
- `lookback` (int, optional): Lookback period for momentum/breakout
- `threshold` (float, optional): Threshold for momentum strategies
- `z_score_threshold` (float, optional): Z-score threshold for mean reversion
- `breakout_factor` (float, optional): Breakout factor for channel strategies

**Returns**:
```json
{
  "symbol": "AAPL",
  "strategy": "sma_cross",
  "period": "2023-01-01 to 2024-01-01",
  "metrics": {
    "total_return": 0.15,
    "sharpe_ratio": 1.2,
    "max_drawdown": -0.08,
    "total_trades": 24,
    "win_rate": 0.58,
    "profit_factor": 1.45,
    "calmar_ratio": 1.85,
    "volatility": 0.18
  },
  "trades": [
    {
      "entry_date": "2023-01-15",
      "exit_date": "2023-02-10",
      "entry_price": 150.0,
      "exit_price": 158.5,
      "return": 0.057,
      "holding_period": 26
    }
  ],
  "equity_curve": [10000, 10150, 10200, ...],
  "drawdown_series": [0, -0.01, -0.02, ...],
  "analysis": {
    "risk_metrics": {...},
    "performance_analysis": {...}
  }
}
```

**Examples**:
```python
# Simple SMA crossover
run_backtest("AAPL", "sma_cross", fast_period=10, slow_period=20)

# RSI mean reversion
run_backtest("TSLA", "rsi", period=14, oversold=30, overbought=70)

# MACD strategy with custom parameters
run_backtest("MSFT", "macd", fast_period=12, slow_period=26, signal_period=9)

# Bollinger Bands strategy
run_backtest("GOOGL", "bollinger", period=20, std_dev=2.0)
```

### optimize_strategy

Optimize strategy parameters using grid search to find the best-performing configuration.

**Function**: `optimize_strategy`

**Parameters**:
- `symbol` (str, required): Stock symbol to optimize
- `strategy` (str, default: "sma_cross"): Strategy type to optimize
- `start_date` (str, optional): Start date (YYYY-MM-DD)
- `end_date` (str, optional): End date (YYYY-MM-DD)
- `optimization_metric` (str, default: "sharpe_ratio"): Metric to optimize ("sharpe_ratio", "total_return", "win_rate", "calmar_ratio")
- `optimization_level` (str, default: "medium"): Level of optimization ("coarse", "medium", "fine")
- `top_n` (int, default: 10): Number of top results to return

**Returns**:
```json
{
  "symbol": "AAPL",
  "strategy": "sma_cross",
  "optimization_metric": "sharpe_ratio",
  "optimization_level": "medium",
  "total_combinations": 64,
  "execution_time": 45.2,
  "best_parameters": {
    "fast_period": 8,
    "slow_period": 21,
    "sharpe_ratio": 1.85,
    "total_return": 0.28,
    "max_drawdown": -0.06
  },
  "top_results": [
    {
      "parameters": {"fast_period": 8, "slow_period": 21},
      "sharpe_ratio": 1.85,
      "total_return": 0.28,
      "max_drawdown": -0.06,
      "total_trades": 18
    }
  ],
  "parameter_sensitivity": {
    "fast_period": {"min": 5, "max": 20, "best": 8},
    "slow_period": {"min": 20, "max": 50, "best": 21}
  }
}
```

**Examples**:
```python
# Optimize SMA crossover for Sharpe ratio
optimize_strategy("AAPL", "sma_cross", optimization_metric="sharpe_ratio")

# Fine-tune RSI parameters for total return
optimize_strategy("TSLA", "rsi", optimization_metric="total_return", optimization_level="fine")

# Quick coarse optimization for multiple strategies
optimize_strategy("MSFT", "macd", optimization_level="coarse", top_n=5)
```

### walk_forward_analysis

Perform walk-forward analysis to test strategy robustness and out-of-sample performance.

**Function**: `walk_forward_analysis`

**Parameters**:
- `symbol` (str, required): Stock symbol to analyze
- `strategy` (str, default: "sma_cross"): Strategy type
- `start_date` (str, optional): Start date (YYYY-MM-DD)
- `end_date` (str, optional): End date (YYYY-MM-DD)
- `window_size` (int, default: 252): Test window size in trading days (default: 1 year)
- `step_size` (int, default: 63): Step size for rolling window (default: 1 quarter)

**Returns**:
```json
{
  "symbol": "AAPL",
  "strategy": "sma_cross",
  "total_windows": 8,
  "window_size": 252,
  "step_size": 63,
  "out_of_sample_performance": {
    "average_return": 0.12,
    "average_sharpe": 0.95,
    "consistency_score": 0.75,
    "best_window": {"period": "2023-Q2", "return": 0.28},
    "worst_window": {"period": "2023-Q4", "return": -0.05}
  },
  "window_results": [
    {
      "window_id": 1,
      "optimization_period": "2022-01-01 to 2022-12-31",
      "test_period": "2023-01-01 to 2023-03-31",
      "best_parameters": {"fast_period": 10, "slow_period": 25},
      "out_of_sample_return": 0.08,
      "out_of_sample_sharpe": 1.1
    }
  ],
  "stability_metrics": {
    "parameter_stability": 0.85,
    "performance_stability": 0.72,
    "overfitting_risk": "low"
  }
}
```

### monte_carlo_simulation

Run Monte Carlo simulation on backtest results to assess risk and confidence intervals.

**Function**: `monte_carlo_simulation`

**Parameters**:
- `symbol` (str, required): Stock symbol
- `strategy` (str, default: "sma_cross"): Strategy type
- `start_date` (str, optional): Start date (YYYY-MM-DD)
- `end_date` (str, optional): End date (YYYY-MM-DD)
- `num_simulations` (int, default: 1000): Number of Monte Carlo simulations
- Strategy-specific parameters (same as `run_backtest`)

**Returns**:
```json
{
  "symbol": "AAPL",
  "strategy": "sma_cross",
  "num_simulations": 1000,
  "confidence_intervals": {
    "95%": {"lower": 0.05, "upper": 0.32},
    "90%": {"lower": 0.08, "upper": 0.28},
    "68%": {"lower": 0.12, "upper": 0.22}
  },
  "risk_metrics": {
    "probability_of_loss": 0.15,
    "expected_return": 0.17,
    "value_at_risk_5%": -0.12,
    "expected_shortfall": -0.18,
    "maximum_drawdown_95%": -0.15
  },
  "simulation_statistics": {
    "mean_return": 0.168,
    "std_return": 0.089,
    "skewness": -0.23,
    "kurtosis": 2.85,
    "best_simulation": 0.45,
    "worst_simulation": -0.28
  }
}
```

### compare_strategies

Compare multiple strategies on the same symbol to identify the best performer.

**Function**: `compare_strategies`

**Parameters**:
- `symbol` (str, required): Stock symbol
- `strategies` (list[str], optional): List of strategy types to compare (defaults to top 5)
- `start_date` (str, optional): Start date (YYYY-MM-DD)
- `end_date` (str, optional): End date (YYYY-MM-DD)

**Returns**:
```json
{
  "symbol": "AAPL",
  "comparison_period": "2023-01-01 to 2024-01-01",
  "strategies_compared": ["sma_cross", "rsi", "macd", "bollinger", "momentum"],
  "rankings": {
    "by_sharpe_ratio": [
      {"strategy": "macd", "sharpe_ratio": 1.45},
      {"strategy": "sma_cross", "sharpe_ratio": 1.22},
      {"strategy": "momentum", "sharpe_ratio": 0.98}
    ],
    "by_total_return": [
      {"strategy": "momentum", "total_return": 0.32},
      {"strategy": "macd", "total_return": 0.28},
      {"strategy": "sma_cross", "total_return": 0.18}
    ]
  },
  "detailed_comparison": {
    "sma_cross": {
      "total_return": 0.18,
      "sharpe_ratio": 1.22,
      "max_drawdown": -0.08,
      "total_trades": 24,
      "win_rate": 0.58
    }
  },
  "best_overall": "macd",
  "recommendation": "MACD strategy provides best risk-adjusted returns"
}
```

### backtest_portfolio

Backtest a strategy across multiple symbols to create a diversified portfolio.

**Function**: `backtest_portfolio`

**Parameters**:
- `symbols` (list[str], required): List of stock symbols
- `strategy` (str, default: "sma_cross"): Strategy type to apply
- `start_date` (str, optional): Start date (YYYY-MM-DD)
- `end_date` (str, optional): End date (YYYY-MM-DD)
- `initial_capital` (float, default: 10000.0): Starting capital
- `position_size` (float, default: 0.1): Position size per symbol (0.1 = 10%)
- Strategy-specific parameters (same as `run_backtest`)

**Returns**:
```json
{
  "portfolio_metrics": {
    "symbols_tested": 5,
    "total_return": 0.22,
    "average_sharpe": 1.15,
    "max_drawdown": -0.12,
    "total_trades": 120,
    "diversification_benefit": 0.85
  },
  "individual_results": [
    {
      "symbol": "AAPL",
      "total_return": 0.18,
      "sharpe_ratio": 1.22,
      "max_drawdown": -0.08,
      "contribution_to_portfolio": 0.24
    }
  ],
  "correlation_matrix": {
    "AAPL": {"MSFT": 0.72, "GOOGL": 0.68},
    "MSFT": {"GOOGL": 0.75}
  },
  "summary": "Portfolio backtest of 5 symbols with sma_cross strategy"
}
```

## Strategy Management

### list_strategies

List all available backtesting strategies with descriptions and parameters.

**Function**: `list_strategies`

**Parameters**: None

**Returns**:
```json
{
  "available_strategies": {
    "sma_cross": {
      "type": "sma_cross",
      "name": "SMA Crossover",
      "description": "Buy when fast SMA crosses above slow SMA, sell when it crosses below",
      "default_parameters": {"fast_period": 10, "slow_period": 20},
      "optimization_ranges": {
        "fast_period": [5, 10, 15, 20],
        "slow_period": [20, 30, 50, 100]
      }
    }
  },
  "total_count": 9,
  "categories": {
    "trend_following": ["sma_cross", "ema_cross", "macd", "breakout"],
    "mean_reversion": ["rsi", "bollinger", "mean_reversion"],
    "momentum": ["momentum", "volume_momentum"]
  }
}
```

### parse_strategy

Parse natural language strategy description into VectorBT parameters.

**Function**: `parse_strategy`

**Parameters**:
- `description` (str, required): Natural language description of trading strategy

**Returns**:
```json
{
  "success": true,
  "strategy": {
    "strategy_type": "rsi",
    "parameters": {
      "period": 14,
      "oversold": 30,
      "overbought": 70
    }
  },
  "message": "Successfully parsed as rsi strategy"
}
```

**Examples**:
```python
# Parse natural language descriptions
parse_strategy("Buy when RSI is below 30 and sell when above 70")
parse_strategy("Use 10-day and 20-day moving average crossover")
parse_strategy("MACD strategy with standard parameters")
```

## Visualization Tools

### generate_backtest_charts

Generate comprehensive charts for a backtest including equity curve, trades, and performance dashboard.

**Function**: `generate_backtest_charts`

**Parameters**:
- `symbol` (str, required): Stock symbol
- `strategy` (str, default: "sma_cross"): Strategy type
- `start_date` (str, optional): Start date (YYYY-MM-DD)
- `end_date` (str, optional): End date (YYYY-MM-DD)
- `theme` (str, default: "light"): Chart theme ("light" or "dark")

**Returns**:
```json
{
  "equity_curve": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "trade_scatter": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "performance_dashboard": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
}
```

### generate_optimization_charts

Generate heatmap charts for strategy parameter optimization results.

**Function**: `generate_optimization_charts`

**Parameters**:
- `symbol` (str, required): Stock symbol
- `strategy` (str, default: "sma_cross"): Strategy type
- `start_date` (str, optional): Start date (YYYY-MM-DD)
- `end_date` (str, optional): End date (YYYY-MM-DD)
- `theme` (str, default: "light"): Chart theme ("light" or "dark")

**Returns**:
```json
{
  "optimization_heatmap": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
}
```

## ML-Enhanced Strategies

### run_ml_strategy_backtest

Run backtest using machine learning-enhanced strategies with adaptive capabilities.

**Function**: `run_ml_strategy_backtest`

**Parameters**:
- `symbol` (str, required): Stock symbol to backtest
- `strategy_type` (str, default: "ml_predictor"): ML strategy type ("ml_predictor", "adaptive", "ensemble", "regime_aware")
- `start_date` (str, optional): Start date (YYYY-MM-DD)
- `end_date` (str, optional): End date (YYYY-MM-DD)
- `initial_capital` (float, default: 10000.0): Initial capital amount
- `train_ratio` (float, default: 0.8): Ratio of data for training (0.0-1.0)
- `model_type` (str, default: "random_forest"): ML model type
- `n_estimators` (int, default: 100): Number of estimators for ensemble models
- `max_depth` (int, optional): Maximum tree depth
- `learning_rate` (float, default: 0.01): Learning rate for adaptive strategies
- `adaptation_method` (str, default: "gradient"): Adaptation method ("gradient", "momentum")

**Returns**:
```json
{
  "symbol": "AAPL",
  "strategy_type": "ml_predictor",
  "metrics": {
    "total_return": 0.24,
    "sharpe_ratio": 1.35,
    "max_drawdown": -0.09
  },
  "ml_metrics": {
    "training_period": 400,
    "testing_period": 100,
    "train_test_split": 0.8,
    "feature_importance": {
      "rsi": 0.25,
      "macd": 0.22,
      "volume_ratio": 0.18,
      "price_momentum": 0.16
    },
    "model_accuracy": 0.68,
    "prediction_confidence": 0.72
  }
}
```

### train_ml_predictor

Train a machine learning predictor model for generating trading signals.

**Function**: `train_ml_predictor`

**Parameters**:
- `symbol` (str, required): Stock symbol to train on
- `start_date` (str, optional): Start date for training data
- `end_date` (str, optional): End date for training data
- `model_type` (str, default: "random_forest"): ML model type
- `target_periods` (int, default: 5): Forward periods for target variable
- `return_threshold` (float, default: 0.02): Return threshold for signal classification
- `n_estimators` (int, default: 100): Number of estimators
- `max_depth` (int, optional): Maximum tree depth
- `min_samples_split` (int, default: 2): Minimum samples to split

**Returns**:
```json
{
  "symbol": "AAPL",
  "model_type": "random_forest",
  "training_period": "2022-01-01 to 2024-01-01",
  "data_points": 500,
  "target_periods": 5,
  "return_threshold": 0.02,
  "model_parameters": {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 2
  },
  "training_metrics": {
    "accuracy": 0.68,
    "precision": 0.72,
    "recall": 0.65,
    "f1_score": 0.68,
    "feature_importance": {
      "rsi_14": 0.25,
      "macd_signal": 0.22,
      "volume_sma_ratio": 0.18
    }
  }
}
```

### analyze_market_regimes

Analyze market regimes using machine learning to identify different market conditions.

**Function**: `analyze_market_regimes`

**Parameters**:
- `symbol` (str, required): Stock symbol to analyze
- `start_date` (str, optional): Start date for analysis
- `end_date` (str, optional): End date for analysis
- `method` (str, default: "hmm"): Detection method ("hmm", "kmeans", "threshold")
- `n_regimes` (int, default: 3): Number of regimes to detect
- `lookback_period` (int, default: 50): Lookback period for regime detection

**Returns**:
```json
{
  "symbol": "AAPL",
  "analysis_period": "2023-01-01 to 2024-01-01",
  "method": "hmm",
  "n_regimes": 3,
  "regime_names": {
    "0": "Bear/Declining",
    "1": "Sideways/Uncertain",
    "2": "Bull/Trending"
  },
  "current_regime": 2,
  "regime_counts": {"0": 45, "1": 89, "2": 118},
  "regime_percentages": {"0": 17.9, "1": 35.3, "2": 46.8},
  "average_regime_durations": {"0": 15.2, "1": 22.3, "2": 28.7},
  "recent_regime_history": [
    {
      "date": "2024-01-15",
      "regime": 2,
      "probabilities": [0.05, 0.15, 0.80]
    }
  ],
  "total_regime_switches": 18
}
```

### create_strategy_ensemble

Create and backtest a strategy ensemble that combines multiple base strategies.

**Function**: `create_strategy_ensemble`

**Parameters**:
- `symbols` (list[str], required): List of stock symbols
- `base_strategies` (list[str], optional): List of base strategy names (defaults to ["sma_cross", "rsi", "macd"])
- `weighting_method` (str, default: "performance"): Weighting method ("performance", "equal", "volatility")
- `start_date` (str, optional): Start date for backtesting
- `end_date` (str, optional): End date for backtesting
- `initial_capital` (float, default: 10000.0): Initial capital per symbol

**Returns**:
```json
{
  "ensemble_summary": {
    "symbols_tested": 5,
    "base_strategies": ["sma_cross", "rsi", "macd"],
    "weighting_method": "performance",
    "average_return": 0.19,
    "total_trades": 87,
    "average_trades_per_symbol": 17.4
  },
  "individual_results": [
    {
      "symbol": "AAPL",
      "results": {
        "total_return": 0.21,
        "sharpe_ratio": 1.18
      },
      "ensemble_metrics": {
        "strategy_weights": {"sma_cross": 0.4, "rsi": 0.3, "macd": 0.3},
        "strategy_performance": {"sma_cross": 0.15, "rsi": 0.12, "macd": 0.18}
      }
    }
  ],
  "final_strategy_weights": {"sma_cross": 0.42, "rsi": 0.28, "macd": 0.30}
}
```

## Intelligent Backtesting Workflow

### run_intelligent_backtest

Run comprehensive intelligent backtesting workflow with automatic market regime analysis and strategy optimization.

**Function**: `run_intelligent_backtest`

**Parameters**:
- `symbol` (str, required): Stock symbol to analyze (e.g., 'AAPL', 'TSLA')
- `start_date` (str, optional): Start date (YYYY-MM-DD), defaults to 1 year ago
- `end_date` (str, optional): End date (YYYY-MM-DD), defaults to today
- `initial_capital` (float, default: 10000.0): Starting capital for backtest
- `requested_strategy` (str, optional): User-preferred strategy (e.g., 'sma_cross', 'rsi', 'macd')

**Returns**:
```json
{
  "symbol": "AAPL",
  "analysis_period": "2023-01-01 to 2024-01-01",
  "execution_metadata": {
    "total_execution_time": 45.2,
    "steps_completed": 6,
    "confidence_score": 0.87
  },
  "market_regime_analysis": {
    "current_regime": "trending",
    "regime_confidence": 0.85,
    "market_characteristics": {
      "volatility_percentile": 35,
      "trend_strength": 0.72,
      "volume_profile": "above_average"
    }
  },
  "strategy_recommendations": [
    {
      "strategy": "macd",
      "fitness_score": 0.92,
      "recommended_parameters": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
      "expected_performance": {"sharpe_ratio": 1.45, "total_return": 0.28}
    },
    {
      "strategy": "sma_cross",
      "fitness_score": 0.88,
      "recommended_parameters": {"fast_period": 8, "slow_period": 21},
      "expected_performance": {"sharpe_ratio": 1.32, "total_return": 0.24}
    }
  ],
  "optimization_results": {
    "best_strategy": "macd",
    "optimized_parameters": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
    "optimization_method": "grid_search",
    "combinations_tested": 48
  },
  "validation_results": {
    "walk_forward_analysis": {
      "out_of_sample_sharpe": 1.28,
      "consistency_score": 0.82,
      "overfitting_risk": "low"
    },
    "monte_carlo_simulation": {
      "probability_of_loss": 0.12,
      "95_percent_confidence_interval": {"lower": 0.08, "upper": 0.35}
    }
  },
  "final_recommendation": {
    "recommended_strategy": "macd",
    "confidence_level": "high",
    "expected_annual_return": 0.28,
    "expected_sharpe_ratio": 1.45,
    "maximum_expected_drawdown": -0.09,
    "risk_assessment": "moderate",
    "implementation_notes": [
      "Strategy performs well in trending markets",
      "Consider position sizing based on volatility",
      "Monitor for regime changes"
    ]
  }
}
```

### quick_market_regime_analysis

Perform fast market regime analysis and basic strategy recommendations without full optimization.

**Function**: `quick_market_regime_analysis`

**Parameters**:
- `symbol` (str, required): Stock symbol to analyze
- `start_date` (str, optional): Start date (YYYY-MM-DD), defaults to 1 year ago
- `end_date` (str, optional): End date (YYYY-MM-DD), defaults to today

**Returns**:
```json
{
  "symbol": "AAPL",
  "analysis_type": "quick_analysis",
  "execution_time": 8.5,
  "market_regime": {
    "classification": "trending",
    "confidence": 0.78,
    "characteristics": {
      "trend_direction": "bullish",
      "volatility_level": "moderate",
      "volume_profile": "above_average"
    }
  },
  "strategy_recommendations": [
    {
      "strategy": "sma_cross",
      "fitness_score": 0.85,
      "reasoning": "Strong trend favors moving average strategies"
    },
    {
      "strategy": "macd",
      "fitness_score": 0.82,
      "reasoning": "MACD works well in trending environments"
    },
    {
      "strategy": "momentum",
      "fitness_score": 0.79,
      "reasoning": "Momentum strategies benefit from clear trends"
    }
  ],
  "market_conditions_summary": {
    "overall_assessment": "favorable_for_trend_following",
    "risk_level": "moderate",
    "recommended_position_sizing": "standard"
  }
}
```

### explain_market_regime

Get detailed explanation of market regime characteristics and suitable strategies.

**Function**: `explain_market_regime`

**Parameters**:
- `regime` (str, required): Market regime to explain ("trending", "ranging", "volatile", "volatile_trending", "low_volume")

**Returns**:
```json
{
  "regime": "trending",
  "explanation": {
    "description": "A market in a clear directional movement (up or down trend)",
    "characteristics": [
      "Strong directional price movement",
      "Higher highs and higher lows (uptrend) or lower highs and lower lows (downtrend)",
      "Good momentum indicators",
      "Volume supporting the trend direction"
    ],
    "best_strategies": ["sma_cross", "ema_cross", "macd", "breakout", "momentum"],
    "avoid_strategies": ["rsi", "mean_reversion", "bollinger"],
    "risk_factors": [
      "Trend reversals can be sudden",
      "False breakouts in weak trends",
      "Momentum strategies can give late signals"
    ]
  },
  "trading_tips": [
    "Focus on sma_cross, ema_cross, macd, breakout, momentum strategies",
    "Avoid rsi, mean_reversion, bollinger strategies",
    "Always use proper risk management",
    "Consider the broader market context"
  ]
}
```

## Available Strategies

### Traditional Technical Analysis Strategies

#### 1. SMA Crossover (`sma_cross`)
- **Description**: Buy when fast SMA crosses above slow SMA, sell when crosses below
- **Default Parameters**: `fast_period=10, slow_period=20`
- **Best For**: Trending markets
- **Optimization Ranges**: fast_period [5-20], slow_period [20-100]

#### 2. EMA Crossover (`ema_cross`)
- **Description**: Exponential moving average crossover with faster response than SMA
- **Default Parameters**: `fast_period=12, slow_period=26`
- **Best For**: Trending markets with more responsiveness
- **Optimization Ranges**: fast_period [8-20], slow_period [20-50]

#### 3. RSI Mean Reversion (`rsi`)
- **Description**: Buy oversold (RSI < 30), sell overbought (RSI > 70)
- **Default Parameters**: `period=14, oversold=30, overbought=70`
- **Best For**: Ranging/sideways markets
- **Optimization Ranges**: period [7-21], oversold [20-35], overbought [65-80]

#### 4. MACD Signal (`macd`)
- **Description**: Buy when MACD crosses above signal line, sell when crosses below
- **Default Parameters**: `fast_period=12, slow_period=26, signal_period=9`
- **Best For**: Trending markets with momentum confirmation
- **Optimization Ranges**: fast_period [8-14], slow_period [21-30], signal_period [7-11]

#### 5. Bollinger Bands (`bollinger`)
- **Description**: Buy at lower band (oversold), sell at upper band (overbought)
- **Default Parameters**: `period=20, std_dev=2.0`
- **Best For**: Mean-reverting/ranging markets
- **Optimization Ranges**: period [10-25], std_dev [1.5-3.0]

#### 6. Momentum (`momentum`)
- **Description**: Buy strong momentum, sell weak momentum based on returns threshold
- **Default Parameters**: `lookback=20, threshold=0.05`
- **Best For**: Trending markets with clear momentum
- **Optimization Ranges**: lookback [10-30], threshold [0.02-0.10]

#### 7. Mean Reversion (`mean_reversion`)
- **Description**: Buy when price is below moving average by threshold
- **Default Parameters**: `ma_period=20, entry_threshold=0.02, exit_threshold=0.01`
- **Best For**: Sideways/ranging markets
- **Optimization Ranges**: ma_period [15-50], entry_threshold [0.01-0.05]

#### 8. Channel Breakout (`breakout`)
- **Description**: Buy on breakout above rolling high, sell on breakdown below rolling low
- **Default Parameters**: `lookback=20, exit_lookback=10`
- **Best For**: Volatile trending markets
- **Optimization Ranges**: lookback [10-50], exit_lookback [5-20]

#### 9. Volume-Weighted Momentum (`volume_momentum`)
- **Description**: Momentum strategy filtered by volume surge
- **Default Parameters**: `momentum_period=20, volume_period=20, momentum_threshold=0.05, volume_multiplier=1.5`
- **Best For**: Markets with significant volume participation
- **Optimization Ranges**: momentum_period [10-30], volume_multiplier [1.2-2.0]

### ML-Enhanced Strategies

#### 1. ML Predictor (`ml_predictor`)
- Uses machine learning models (Random Forest, etc.) to predict future price movements
- Features: Technical indicators, price patterns, volume analysis
- Training/testing split with out-of-sample validation

#### 2. Adaptive Strategy (`adaptive`)
- Adapts base strategy parameters based on recent performance
- Uses gradient-based or momentum-based adaptation methods
- Continuously learns from market feedback

#### 3. Strategy Ensemble (`ensemble`)
- Combines multiple base strategies with dynamic weighting
- Weighting methods: performance-based, equal-weight, volatility-adjusted
- Provides diversification benefits

#### 4. Regime-Aware Strategy (`regime_aware`)
- Automatically switches between different strategies based on detected market regime
- Uses Hidden Markov Models or clustering for regime detection
- Optimizes strategy selection for current market conditions

## Performance Considerations

### Execution Times
- **Simple Backtest**: 2-5 seconds
- **Strategy Optimization**: 30-120 seconds (depending on level)
- **Walk-Forward Analysis**: 60-300 seconds
- **Monte Carlo Simulation**: 45-90 seconds
- **ML Strategy Training**: 60-180 seconds
- **Intelligent Backtest**: 120-300 seconds (full workflow)

### Memory Usage
- **Single Symbol**: 50-200 MB
- **Portfolio (5 symbols)**: 200-500 MB
- **ML Training**: 100-1000 MB (depending on data size)

### Optimization Levels
- **Coarse**: 16-36 parameter combinations, fastest
- **Medium**: 36-100 combinations, balanced speed/accuracy
- **Fine**: 100-500+ combinations, most thorough

## Error Handling

### Common Errors

#### Insufficient Data
```json
{
  "error": "Insufficient data for backtest (minimum 100 data points)",
  "symbol": "PENNY_STOCK",
  "message": "Please use a longer time period or different symbol"
}
```

#### Invalid Strategy
```json
{
  "error": "Unknown strategy type: invalid_strategy",
  "available_strategies": ["sma_cross", "rsi", "macd", ...],
  "message": "Please use one of the available strategy types"
}
```

#### Parameter Validation
```json
{
  "error": "Invalid parameter value",
  "parameter": "fast_period",
  "value": -5,
  "message": "fast_period must be positive integer"
}
```

#### ML Training Errors
```json
{
  "error": "ML training failed: Insufficient data for training (minimum 200 data points)",
  "symbol": "LOW_DATA_STOCK",
  "message": "ML strategies require more historical data"
}
```

### Troubleshooting

1. **Data Issues**: Ensure sufficient historical data (minimum 100 points, 200+ for ML)
2. **Parameter Validation**: Check parameter types and ranges
3. **Memory Issues**: Reduce number of symbols in portfolio backtests
4. **Timeout Issues**: Use coarse optimization for faster results
5. **Strategy Parsing**: Use exact strategy names from `list_strategies`

## Integration Examples

### Claude Desktop Usage

```
# Basic backtest
"Run a backtest for AAPL using RSI strategy with 14-day period"

# Strategy comparison
"Compare SMA crossover, RSI, and MACD strategies on Tesla stock"

# Intelligent analysis
"Run intelligent backtest on Microsoft stock and recommend the best strategy"

# Portfolio backtest
"Backtest momentum strategy on AAPL, MSFT, GOOGL, AMZN, and TSLA"

# Optimization
"Optimize MACD parameters for Netflix stock over the last 2 years"

# ML strategies
"Train an ML predictor on Amazon stock and test its performance"
```

### API Integration

```python
# Using MCP client
import mcp

client = mcp.Client("maverick-mcp")

# Run backtest
result = await client.call_tool("run_backtest", {
    "symbol": "AAPL",
    "strategy": "sma_cross",
    "fast_period": 10,
    "slow_period": 20,
    "initial_capital": 50000
})

# Optimize strategy
optimization = await client.call_tool("optimize_strategy", {
    "symbol": "TSLA",
    "strategy": "rsi",
    "optimization_level": "medium",
    "optimization_metric": "sharpe_ratio"
})

# Intelligent backtest
intelligent_result = await client.call_tool("run_intelligent_backtest", {
    "symbol": "MSFT",
    "start_date": "2022-01-01",
    "end_date": "2023-12-31"
})
```

### Workflow Integration

```python
# Complete backtesting workflow
symbols = ["AAPL", "MSFT", "GOOGL"]
strategies = ["sma_cross", "rsi", "macd"]

for symbol in symbols:
    # 1. Quick regime analysis
    regime = await client.call_tool("quick_market_regime_analysis", {
        "symbol": symbol
    })

    # 2. Strategy comparison
    comparison = await client.call_tool("compare_strategies", {
        "symbol": symbol,
        "strategies": strategies
    })

    # 3. Optimize best strategy
    best_strategy = comparison["best_overall"]
    optimization = await client.call_tool("optimize_strategy", {
        "symbol": symbol,
        "strategy": best_strategy
    })

    # 4. Validate with walk-forward
    validation = await client.call_tool("walk_forward_analysis", {
        "symbol": symbol,
        "strategy": best_strategy
    })
```

## Best Practices

### Strategy Selection
1. **Trending Markets**: Use sma_cross, ema_cross, macd, breakout, momentum
2. **Ranging Markets**: Use rsi, bollinger, mean_reversion
3. **Volatile Markets**: Use breakout, volatility_breakout with wider stops
4. **Unknown Conditions**: Use intelligent_backtest for automatic selection

### Parameter Optimization
1. **Start with Default**: Test default parameters first
2. **Use Medium Level**: Good balance of thoroughness and speed
3. **Validate Results**: Always use walk-forward analysis for final validation
4. **Avoid Overfitting**: Check for consistent out-of-sample performance

### Risk Management
1. **Position Sizing**: Never risk more than 1-2% per trade
2. **Diversification**: Test strategies across multiple symbols
3. **Regime Awareness**: Monitor market regime changes
4. **Drawdown Limits**: Set maximum acceptable drawdown levels

### Performance Optimization
1. **Parallel Processing**: Use portfolio backtests for batch analysis
2. **Caching**: Results are cached for faster repeated analysis
3. **Data Efficiency**: Use appropriate date ranges to balance data needs and speed
4. **ML Considerations**: Ensure sufficient training data for ML strategies

This comprehensive API documentation provides everything needed to effectively use the MaverickMCP backtesting system. Each tool is designed to work independently or as part of larger workflows, with extensive error handling and performance optimization built-in.