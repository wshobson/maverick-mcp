"""MCP Prompts for better tool discovery and usage guidance."""

from fastmcp import FastMCP


def register_mcp_prompts(mcp: FastMCP):
    """Register MCP prompts to help clients understand how to use the tools."""

    # Backtesting prompts
    @mcp.prompt()
    async def backtest_strategy_guide():
        """Guide for running backtesting strategies."""
        return """
# Backtesting Strategy Guide

## Available Strategies (15 total)

### Traditional Strategies (9):
- `sma_cross`: Simple Moving Average Crossover
- `rsi`: RSI Mean Reversion (oversold/overbought)
- `macd`: MACD Signal Line Crossover
- `bollinger`: Bollinger Bands (buy low, sell high)
- `momentum`: Momentum-based trading
- `ema_cross`: Exponential Moving Average Crossover
- `mean_reversion`: Mean Reversion Strategy
- `breakout`: Channel Breakout Strategy
- `volume_momentum`: Volume-Weighted Momentum

### ML Strategies (6):
- `online_learning`: Adaptive learning with dynamic thresholds
- `regime_aware`: Market regime detection (trending vs ranging)
- `ensemble`: Multiple strategy voting system

## Example Usage:

### Traditional Strategy:
"Run a backtest on AAPL using the sma_cross strategy from 2024-01-01 to 2024-12-31"

### ML Strategy:
"Test the online_learning strategy on TSLA for the past year with a learning rate of 0.01"

### Parameters:
- Most strategies have default parameters that work well
- You can customize: fast_period, slow_period, threshold, etc.
"""

    @mcp.prompt()
    async def ml_strategy_examples():
        """Examples of ML strategy usage."""
        return """
# ML Strategy Examples

## 1. Online Learning Strategy
"Run online_learning strategy on NVDA with parameters:
- lookback: 20 days
- learning_rate: 0.01
- start_date: 2024-01-01
- end_date: 2024-12-31"

## 2. Regime-Aware Strategy
"Test regime_aware strategy on SPY to detect market regimes:
- regime_window: 50 days
- threshold: 0.02
- Adapts between trending and ranging markets"

## 3. Ensemble Strategy
"Use ensemble strategy on AAPL combining multiple signals:
- Combines SMA, RSI, and Momentum
- Uses voting to generate signals
- More robust than single strategies"

## Important Notes:
- ML strategies work through the standard run_backtest tool
- Use strategy_type parameter: "online_learning", "regime_aware", or "ensemble"
- These are simplified ML strategies that don't require training
"""

    @mcp.prompt()
    async def optimization_guide():
        """Guide for parameter optimization."""
        return """
# Parameter Optimization Guide

## How to Optimize Strategy Parameters

### Basic Optimization:
"Optimize sma_cross parameters for MSFT over the past 6 months"

This will test combinations like:
- fast_period: [5, 10, 15, 20]
- slow_period: [20, 30, 50, 100]

### Custom Parameter Ranges:
"Optimize RSI strategy for TSLA with:
- period: [7, 14, 21]
- oversold: [20, 25, 30]
- overbought: [70, 75, 80]"

### Optimization Metrics:
- sharpe_ratio (default): Risk-adjusted returns
- total_return: Raw returns
- win_rate: Percentage of winning trades

## Results Include:
- Best parameter combination
- Performance metrics for top combinations
- Comparison across all tested parameters
"""

    @mcp.prompt()
    async def available_tools_summary():
        """Summary of all available MCP tools."""
        return """
# MaverickMCP Tools Summary

## 1. Backtesting Tools
- `run_backtest`: Run any strategy (traditional or ML)
- `optimize_parameters`: Find best parameters
- `compare_strategies`: Compare multiple strategies
- `get_strategy_info`: Get strategy details

## 2. Data Tools
- `get_stock_data`: Historical price data
- `get_stock_info`: Company information
- `get_multiple_stocks_data`: Batch data fetching

## 3. Technical Analysis
- `calculate_sma`, `calculate_ema`: Moving averages
- `calculate_rsi`: Relative Strength Index
- `calculate_macd`: MACD indicator
- `calculate_bollinger_bands`: Bollinger Bands
- `get_full_technical_analysis`: All indicators

## 4. Screening Tools
- `get_maverick_recommendations`: Bullish stocks
- `get_maverick_bear_recommendations`: Bearish setups
- `get_trending_breakout_recommendations`: Breakout candidates

## 5. Portfolio Tools
- `optimize_portfolio`: Portfolio optimization
- `analyze_portfolio_risk`: Risk assessment
- `calculate_correlation_matrix`: Asset correlations

## Usage Tips:
- Start with simple strategies before trying ML
- Use default parameters initially
- Optimize parameters after initial testing
- Compare multiple strategies on same data
"""

    @mcp.prompt()
    async def troubleshooting_guide():
        """Troubleshooting common issues."""
        return """
# Troubleshooting Guide

## Common Issues and Solutions

### 1. "Unknown strategy type"
**Solution**: Use one of these exact strategy names:
- Traditional: sma_cross, rsi, macd, bollinger, momentum, ema_cross, mean_reversion, breakout, volume_momentum
- ML: online_learning, regime_aware, ensemble

### 2. "No data available"
**Solution**:
- Check date range (use past dates, not future)
- Verify stock symbol (use standard tickers like AAPL, MSFT)
- Try shorter date ranges (1 year or less)

### 3. ML Strategy Issues
**Solution**: Use the standard run_backtest tool with:
```
strategy_type: "online_learning"  # or "regime_aware", "ensemble"
```
Don't use the run_ml_backtest tool for these strategies.

### 4. Parameter Errors
**Solution**: Start with no parameters (uses defaults):
"Run backtest on AAPL using sma_cross strategy"

Then customize if needed:
"Run backtest on AAPL using sma_cross with fast_period=10 and slow_period=30"

### 5. Connection Issues
**Solution**:
- Restart Claude Desktop
- Check server is running: The white circle should be blue
- Try a simple test: "Get AAPL stock data"
"""

    @mcp.prompt()
    async def quick_start():
        """Quick start guide for new users."""
        return """
# Quick Start Guide

## Test These Commands First:

### 1. Simple Backtest
"Run a backtest on AAPL using the sma_cross strategy for 2024"

### 2. Get Stock Data
"Get AAPL stock data for the last 3 months"

### 3. Technical Analysis
"Show me technical analysis for MSFT"

### 4. Stock Screening
"Show me bullish stock recommendations"

### 5. ML Strategy Test
"Test the online_learning strategy on TSLA for the past 6 months"

## Next Steps:
1. Try different strategies on your favorite stocks
2. Optimize parameters for better performance
3. Compare multiple strategies
4. Build a portfolio with top performers

## Pro Tips:
- Use 2024 dates for reliable data
- Start with liquid stocks (AAPL, MSFT, GOOGL)
- Default parameters usually work well
- ML strategies are experimental but fun to try
"""

    # Register a resources endpoint for better discovery
    @mcp.prompt()
    async def strategy_reference():
        """Complete strategy reference with all parameters."""
        strategies = {
            "sma_cross": {
                "description": "Buy when fast SMA crosses above slow SMA",
                "parameters": {
                    "fast_period": "Fast moving average period (default: 10)",
                    "slow_period": "Slow moving average period (default: 20)",
                },
                "example": "run_backtest(symbol='AAPL', strategy_type='sma_cross', fast_period=10, slow_period=20)",
            },
            "rsi": {
                "description": "Buy oversold (RSI < 30), sell overbought (RSI > 70)",
                "parameters": {
                    "period": "RSI calculation period (default: 14)",
                    "oversold": "Oversold threshold (default: 30)",
                    "overbought": "Overbought threshold (default: 70)",
                },
                "example": "run_backtest(symbol='MSFT', strategy_type='rsi', period=14, oversold=30)",
            },
            "online_learning": {
                "description": "ML strategy with adaptive thresholds",
                "parameters": {
                    "lookback": "Historical window (default: 20)",
                    "learning_rate": "Adaptation rate (default: 0.01)",
                },
                "example": "run_backtest(symbol='TSLA', strategy_type='online_learning', lookback=20)",
            },
            "regime_aware": {
                "description": "Detects and adapts to market regimes",
                "parameters": {
                    "regime_window": "Regime detection window (default: 50)",
                    "threshold": "Regime change threshold (default: 0.02)",
                },
                "example": "run_backtest(symbol='SPY', strategy_type='regime_aware', regime_window=50)",
            },
            "ensemble": {
                "description": "Combines multiple strategies with voting",
                "parameters": {
                    "fast_period": "Fast MA period (default: 10)",
                    "slow_period": "Slow MA period (default: 20)",
                    "rsi_period": "RSI period (default: 14)",
                },
                "example": "run_backtest(symbol='NVDA', strategy_type='ensemble')",
            },
        }

        import json

        return f"""
# Complete Strategy Reference

## All Available Strategies with Parameters

```json
{json.dumps(strategies, indent=2)}
```

## Usage Pattern:
All strategies use the same tool: `run_backtest`

Parameters:
- symbol: Stock ticker (required)
- strategy_type: Strategy name (required)
- start_date: YYYY-MM-DD format
- end_date: YYYY-MM-DD format
- initial_capital: Starting amount (default: 10000)
- Additional strategy-specific parameters

## Testing Order:
1. Start with sma_cross (simplest)
2. Try rsi or macd (intermediate)
3. Test online_learning (ML strategy)
4. Compare all with compare_strategies tool
"""

    # Register resources for better discovery
    @mcp.resource("strategies://list")
    def list_strategies_resource():
        """List of all available backtesting strategies with parameters."""
        return {
            "traditional_strategies": {
                "sma_cross": {
                    "name": "Simple Moving Average Crossover",
                    "parameters": ["fast_period", "slow_period"],
                    "default_values": {"fast_period": 10, "slow_period": 20}
                },
                "rsi": {
                    "name": "RSI Mean Reversion",
                    "parameters": ["period", "oversold", "overbought"],
                    "default_values": {"period": 14, "oversold": 30, "overbought": 70}
                },
                "macd": {
                    "name": "MACD Signal Line Crossover",
                    "parameters": ["fast_period", "slow_period", "signal_period"],
                    "default_values": {"fast_period": 12, "slow_period": 26, "signal_period": 9}
                },
                "bollinger": {
                    "name": "Bollinger Bands",
                    "parameters": ["period", "std_dev"],
                    "default_values": {"period": 20, "std_dev": 2}
                },
                "momentum": {
                    "name": "Momentum Trading",
                    "parameters": ["period", "threshold"],
                    "default_values": {"period": 10, "threshold": 0.02}
                },
                "ema_cross": {
                    "name": "EMA Crossover",
                    "parameters": ["fast_period", "slow_period"],
                    "default_values": {"fast_period": 12, "slow_period": 26}
                },
                "mean_reversion": {
                    "name": "Mean Reversion",
                    "parameters": ["lookback", "entry_z", "exit_z"],
                    "default_values": {"lookback": 20, "entry_z": -2, "exit_z": 0}
                },
                "breakout": {
                    "name": "Channel Breakout",
                    "parameters": ["lookback", "breakout_factor"],
                    "default_values": {"lookback": 20, "breakout_factor": 1.5}
                },
                "volume_momentum": {
                    "name": "Volume-Weighted Momentum",
                    "parameters": ["period", "volume_factor"],
                    "default_values": {"period": 10, "volume_factor": 1.5}
                }
            },
            "ml_strategies": {
                "online_learning": {
                    "name": "Online Learning Adaptive Strategy",
                    "parameters": ["lookback", "learning_rate"],
                    "default_values": {"lookback": 20, "learning_rate": 0.01}
                },
                "regime_aware": {
                    "name": "Market Regime Detection",
                    "parameters": ["regime_window", "threshold"],
                    "default_values": {"regime_window": 50, "threshold": 0.02}
                },
                "ensemble": {
                    "name": "Ensemble Voting Strategy",
                    "parameters": ["fast_period", "slow_period", "rsi_period"],
                    "default_values": {"fast_period": 10, "slow_period": 20, "rsi_period": 14}
                }
            },
            "total_strategies": 15
        }

    @mcp.resource("tools://categories")
    def tool_categories_resource():
        """Categorized list of all available MCP tools."""
        return {
            "backtesting": [
                "run_backtest",
                "optimize_parameters",
                "compare_strategies",
                "get_strategy_info"
            ],
            "data": [
                "get_stock_data",
                "get_stock_info",
                "get_multiple_stocks_data"
            ],
            "technical_analysis": [
                "calculate_sma",
                "calculate_ema",
                "calculate_rsi",
                "calculate_macd",
                "calculate_bollinger_bands",
                "get_full_technical_analysis"
            ],
            "screening": [
                "get_maverick_recommendations",
                "get_maverick_bear_recommendations",
                "get_trending_breakout_recommendations"
            ],
            "portfolio": [
                "optimize_portfolio",
                "analyze_portfolio_risk",
                "calculate_correlation_matrix"
            ],
            "research": [
                "research_comprehensive",
                "research_company",
                "analyze_market_sentiment",
                "coordinate_agents"
            ]
        }

    @mcp.resource("examples://backtesting")
    def backtesting_examples_resource():
        """Practical examples of using backtesting tools."""
        return {
            "simple_backtest": {
                "description": "Basic backtest with default parameters",
                "example": "run_backtest(symbol='AAPL', strategy_type='sma_cross')",
                "expected_output": "Performance metrics including total return, sharpe ratio, win rate"
            },
            "custom_parameters": {
                "description": "Backtest with custom strategy parameters",
                "example": "run_backtest(symbol='TSLA', strategy_type='rsi', period=21, oversold=25)",
                "expected_output": "Performance with adjusted RSI parameters"
            },
            "ml_strategy": {
                "description": "Running ML-based strategy",
                "example": "run_backtest(symbol='NVDA', strategy_type='online_learning', lookback=30)",
                "expected_output": "Adaptive strategy performance with online learning"
            },
            "optimization": {
                "description": "Optimize strategy parameters",
                "example": "optimize_parameters(symbol='MSFT', strategy_type='sma_cross')",
                "expected_output": "Best parameter combination and performance metrics"
            },
            "comparison": {
                "description": "Compare multiple strategies",
                "example": "compare_strategies(symbol='SPY', strategies=['sma_cross', 'rsi', 'online_learning'])",
                "expected_output": "Side-by-side comparison of strategy performance"
            }
        }

    return True
