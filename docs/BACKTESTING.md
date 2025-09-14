# MaverickMCP Backtesting Documentation

## Overview

MaverickMCP provides a comprehensive backtesting system powered by VectorBT with advanced parallel processing capabilities. The system supports 35+ pre-built strategies ranging from simple moving averages to advanced ML ensembles, with optimization, validation, and analysis tools.

## Quick Start

### Basic Backtesting

```python
# Simple SMA crossover backtest
run_backtest("AAPL", "sma_cross", fast_period=10, slow_period=20)

# RSI mean reversion strategy
run_backtest("TSLA", "rsi", period=14, oversold=30, overbought=70)

# MACD strategy
run_backtest("MSFT", "macd", fast_period=12, slow_period=26, signal_period=9)
```

### Parallel Execution (6-8x Performance Boost)

```python
from maverick_mcp.backtesting.strategy_executor import ExecutionContext, get_strategy_executor

# Create execution contexts for multiple strategies
contexts = [
    ExecutionContext(
        strategy_id="sma_AAPL",
        symbol="AAPL",
        strategy_type="sma_cross",
        parameters={"fast_period": 10, "slow_period": 20},
        start_date="2023-01-01",
        end_date="2024-01-01"
    )
]

# Execute in parallel
async with get_strategy_executor(max_concurrent_strategies=6) as executor:
    results = await executor.execute_strategies_parallel(contexts)
```

## Available Strategies

### Technical Analysis Strategies
- **sma_cross**: Simple Moving Average Crossover
- **ema_cross**: Exponential Moving Average Crossover
- **rsi**: Relative Strength Index Mean Reversion
- **macd**: MACD Crossover Strategy
- **bollinger**: Bollinger Bands Mean Reversion
- **momentum**: Price Momentum Strategy
- **breakout**: Price Channel Breakout
- **mean_reversion**: Statistical Mean Reversion
- **volume_weighted**: Volume-Weighted Moving Average
- **stochastic**: Stochastic Oscillator

### Advanced Strategies
- **adaptive_momentum**: ML-Enhanced Adaptive Momentum
- **ensemble**: Multi-Strategy Ensemble Approach
- **regime_aware**: Market Regime Detection & Switching
- **ml_enhanced**: Machine Learning Enhanced Trading
- **pairs_trading**: Statistical Arbitrage Pairs Trading

## Core API Functions

### run_backtest

Execute a comprehensive backtest with specified strategy and parameters.

```python
run_backtest(
    symbol="AAPL",
    strategy="sma_cross",
    start_date="2023-01-01",  # Optional, defaults to 1 year ago
    end_date="2024-01-01",     # Optional, defaults to today
    initial_capital=10000.0,
    fast_period=10,
    slow_period=20
)
```

**Returns:**
```json
{
  "symbol": "AAPL",
  "strategy": "sma_cross",
  "metrics": {
    "total_return": 0.15,
    "sharpe_ratio": 1.2,
    "max_drawdown": -0.08,
    "total_trades": 24,
    "win_rate": 0.58,
    "profit_factor": 1.45,
    "calmar_ratio": 1.85
  },
  "trades": [...],
  "equity_curve": [...],
  "analysis": {...}
}
```

### optimize_strategy

Find optimal parameters using grid search optimization.

```python
optimize_strategy(
    symbol="AAPL",
    strategy="sma_cross",
    optimization_params={
        "fast_period": [5, 10, 15, 20],
        "slow_period": [20, 30, 40, 50]
    },
    granularity="medium"  # "coarse", "medium", or "fine"
)
```

### validate_strategy

Validate strategy robustness using walk-forward analysis.

```python
validate_strategy(
    symbol="AAPL",
    strategy="sma_cross",
    parameters={"fast_period": 10, "slow_period": 20},
    n_splits=5,           # Number of walk-forward periods
    test_size=0.2,        # Out-of-sample test size
    validation_type="walk_forward"
)
```

### analyze_portfolio

Run portfolio-level backtesting across multiple symbols.

```python
analyze_portfolio(
    symbols=["AAPL", "MSFT", "GOOGL"],
    strategy="momentum",
    weights=[0.33, 0.33, 0.34],  # Optional, equal weight if not specified
    rebalance_frequency="monthly"
)
```

## Parallel Processing Configuration

### Performance Tuning

```python
# Development/Testing (conservative)
executor = StrategyExecutor(
    max_concurrent_strategies=4,
    max_concurrent_api_requests=8,
    connection_pool_size=50
)

# Production (aggressive)
executor = StrategyExecutor(
    max_concurrent_strategies=8,
    max_concurrent_api_requests=15,
    connection_pool_size=100
)

# High-volume backtesting
executor = StrategyExecutor(
    max_concurrent_strategies=12,
    max_concurrent_api_requests=20,
    connection_pool_size=200
)
```

### Environment Variables

```bash
# Database optimization
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40
DB_POOL_TIMEOUT=30

# Parallel execution limits
MAX_CONCURRENT_STRATEGIES=6
MAX_CONCURRENT_API_REQUESTS=10
CONNECTION_POOL_SIZE=100
```

## Database Optimization

### Indexes for Performance

The system automatically creates optimized indexes for fast data retrieval:

- **Composite index** for date range queries with symbol lookup
- **Covering index** for OHLCV queries (includes all price data)
- **Partial index** for recent data (PostgreSQL only)

### Batch Data Fetching

```python
from maverick_mcp.backtesting.strategy_executor import batch_fetch_stock_data

# Fetch data for multiple symbols efficiently
symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
data_dict = await batch_fetch_stock_data(
    symbols=symbols,
    start_date="2023-01-01",
    end_date="2024-01-01",
    max_concurrent=10
)
```

## Best Practices

### 1. Strategy Development
- Start with simple strategies before complex ones
- Always validate with out-of-sample data
- Use walk-forward analysis for robustness testing
- Consider transaction costs and slippage

### 2. Parameter Optimization
- Avoid overfitting with too many parameters
- Use coarse optimization first, then refine
- Validate optimal parameters on different time periods
- Consider parameter stability over time

### 3. Risk Management
- Always set appropriate position sizing
- Use stop-loss and risk limits
- Monitor maximum drawdown
- Diversify across strategies and assets

### 4. Performance Optimization
- Use parallel execution for multiple backtests
- Enable database caching for frequently accessed data
- Batch fetch data for multiple symbols
- Monitor memory usage with large datasets

## Troubleshooting

### Common Issues

**High memory usage**
- Reduce `max_concurrent_strategies`
- Use smaller date ranges for initial testing
- Enable database caching

**Slow performance**
- Ensure database indexes are created
- Increase connection pool size
- Use parallel execution
- Check API rate limits

**API rate limiting**
- Lower `max_concurrent_api_requests`
- Implement exponential backoff
- Use cached data when possible

**Data quality issues**
- Verify data source reliability
- Check for missing data periods
- Validate against multiple sources
- Handle corporate actions properly

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.getLogger("maverick_mcp.backtesting").setLevel(logging.DEBUG)
```

## Performance Metrics

### Key Metrics Explained

- **Total Return**: Overall strategy performance
- **Sharpe Ratio**: Risk-adjusted returns (>1.0 is good, >2.0 is excellent)
- **Max Drawdown**: Maximum peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss (>1.5 is good)
- **Calmar Ratio**: Annual return / Max drawdown (>1.0 is good)

### Benchmark Comparison

Compare strategy performance against buy-and-hold:

```python
results = run_backtest(...)
benchmark = results.get("benchmark_comparison")
print(f"Strategy vs Buy-Hold: {benchmark['excess_return']:.2%}")
```

## Advanced Features

### Monte Carlo Simulation

Assess strategy robustness with randomized scenarios:

```python
monte_carlo_results = run_monte_carlo_simulation(
    strategy_results=results,
    n_simulations=1000,
    confidence_level=0.95
)
```

### Market Regime Detection

Automatically adjust strategy based on market conditions:

```python
regime_results = analyze_market_regime(
    symbol="SPY",
    lookback_period=252,
    regime_indicators=["volatility", "trend", "momentum"]
)
```

### Multi-Strategy Ensemble

Combine multiple strategies for better risk-adjusted returns:

```python
ensemble_results = run_ensemble_backtest(
    symbol="AAPL",
    strategies=["sma_cross", "rsi", "momentum"],
    weights="equal",  # or "optimize" for dynamic weighting
    correlation_threshold=0.7
)
```

## Integration Examples

### With Claude Desktop

```python
# Use MCP tools for comprehensive analysis
"Run a backtest for AAPL using SMA crossover strategy with
optimization for the best parameters over the last 2 years"

# The system will:
# 1. Fetch historical data
# 2. Run parameter optimization
# 3. Execute backtest with optimal parameters
# 4. Provide detailed performance metrics
```

### Programmatic Usage

```python
from maverick_mcp.backtesting import BacktestingEngine

async def run_comprehensive_analysis():
    engine = BacktestingEngine()

    # Run backtest
    results = await engine.run_backtest(
        symbol="AAPL",
        strategy="momentum"
    )

    # Optimize parameters
    optimal = await engine.optimize_strategy(
        symbol="AAPL",
        strategy="momentum",
        granularity="fine"
    )

    # Validate robustness
    validation = await engine.validate_strategy(
        symbol="AAPL",
        strategy="momentum",
        parameters=optimal["best_params"]
    )

    return {
        "backtest": results,
        "optimization": optimal,
        "validation": validation
    }
```

## Testing

Run the test suite to verify functionality:

```bash
# Unit tests
pytest tests/test_backtesting.py

# Integration tests
pytest tests/test_strategy_executor.py

# Performance benchmarks
python scripts/benchmark_parallel_backtesting.py

# Comprehensive validation
python scripts/test_all_strategies.py
```

## Summary

MaverickMCP's backtesting system provides:

- **35+ pre-built strategies** with extensive customization
- **6-8x performance improvement** with parallel processing
- **Comprehensive optimization** and validation tools
- **Professional-grade metrics** and risk analysis
- **Production-ready architecture** with error handling and monitoring

The system is designed for both simple strategy testing and complex portfolio analysis, with a focus on performance, reliability, and ease of use.