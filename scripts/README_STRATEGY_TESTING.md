# Strategy Testing Documentation

## Overview

The `test_all_strategies.py` script provides comprehensive validation of ALL backtesting strategies in the MaverickMCP system. It tests both traditional technical analysis strategies and ML-enhanced strategies with real market data.

## Features

### Comprehensive Strategy Coverage
- **Traditional Strategies (9)**: SMA Cross, RSI, MACD, Bollinger Bands, Momentum, EMA Cross, Mean Reversion, Breakout, Volume Momentum
- **ML Strategies (6)**: Adaptive, Regime Aware, Ensemble, Online Learning, Hybrid Adaptive, Risk Adjusted Ensemble

### Multi-Dimensional Testing
- **5 Symbols**: AAPL, MSFT, GOOGL, TSLA, NVDA
- **4 Time Periods**: 1 Month, 3 Months, 6 Months, 1 Year
- **Total Tests**: 300 (15 strategies × 5 symbols × 4 periods)

### Validation Checks
- ✅ Strategy execution without errors
- ✅ Valid performance metrics (not NaN/infinite)
- ✅ Trade generation capability
- ✅ Reasonable return ranges
- ✅ Acceptable risk metrics
- ✅ Execution time monitoring

## Quick Start

```bash
# Run all strategy tests
make test-strategies

# Or run directly
uv run python scripts/test_all_strategies.py
```

## Expected Output

```
🚀 Starting Comprehensive Strategy Validation
============================================================
Testing 9 traditional + 6 ML strategies
Across 5 symbols and 4 time periods
Total tests: 300

📊 Testing Traditional Strategies...
  📈 Testing sma_cross...
    ✅ AAPL (1M): 0.45s - [6.7%]
    ✅ AAPL (3M): 0.52s - [13.3%]
    ✅ AAPL (6M): 0.61s - [20.0%]
    ✅ AAPL (1Y): 0.73s - [26.7%]
    ✅ MSFT (1M): 0.48s - [33.3%]
    ...

🤖 Testing ML Strategies...
  📈 Testing adaptive...
    ✅ AAPL (1M): 0.32s - [86.7%]
    ...

================================================================================
🎯 COMPREHENSIVE STRATEGY VALIDATION REPORT
================================================================================
📊 Test Summary:
   • Total Tests: 300
   • Successful: 285 (95.0%)
   • Failed: 15
   • Strategies: 9 traditional + 6 ML
   • Symbols: 5 (AAPL, MSFT, GOOGL, TSLA, NVDA)
   • Time Periods: 4 (1M, 3M, 6M, 1Y)

⚡ Performance:
   • Total Time: 156.3s
   • Avg Test Time: 0.52s
   • Max Test Time: 1.24s
   • Tests/Second: 1.9

📈 Strategy Breakdown:
   • Traditional: 180/180 (100.0%)
   • ML-Enhanced: 105/120 (87.5%)

🏆 Top Performing Strategies:
   1. sma_cross: 20/20 (100.0%) - 0.58s avg
   2. ema_cross: 20/20 (100.0%) - 0.61s avg
   3. rsi: 20/20 (100.0%) - 0.55s avg
   4. bollinger: 20/20 (100.0%) - 0.64s avg
   5. momentum: 20/20 (100.0%) - 0.52s avg

✅ Overall Status: EXCELLENT - All strategies working well
================================================================================
💾 Detailed report saved to: strategy_validation_report_20250913_143025.json
```

## Exit Codes

- **0**: Excellent (≥80% success rate)
- **1**: Good with warnings (≥50% success rate)
- **2**: Needs attention (<50% success rate)
- **130**: Interrupted by user

## Generated Reports

The script creates detailed JSON reports in the `scripts/` directory:
- `strategy_validation_report_YYYYMMDD_HHMMSS.json`

### Report Structure
```json
{
  "timestamp": "2024-09-13T14:30:25",
  "execution_time_seconds": 156.3,
  "summary": {
    "total_tests": 300,
    "successful_tests": 285,
    "success_rate": 0.95
  },
  "performance": {
    "avg_execution_time": 0.52,
    "tests_per_second": 1.9
  },
  "strategy_breakdown": {
    "traditional": {"success_rate": 1.0},
    "ml": {"success_rate": 0.875}
  },
  "detailed_results": {
    "by_strategy": {...},
    "by_symbol": {...},
    "failed_tests": [...],
    "warnings": [...]
  }
}
```

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure environment is activated
source .venv/bin/activate
# Or use uv
uv run python scripts/test_all_strategies.py
```

**API Rate Limits**
- The script includes 0.1s delays between tests
- If you hit rate limits, increase the delay in `_test_strategy_group()`

**Memory Issues**
- Each test is isolated to prevent memory leaks
- Large datasets may require more RAM for VectorBT operations

**Network Timeouts**
- Real market data fetching may timeout
- The script will fall back to synthetic data for ML strategies

### Debug Mode

Add debug logging by modifying the script:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Customization

### Test Different Symbols
```python
TEST_SYMBOLS = ["SPY", "QQQ", "IWM", "GLD", "TLT"]
```

### Test Different Periods
```python
TEST_PERIODS = {
    "2W": 14,
    "1M": 30,
    "3M": 90,
    "6M": 180,
    "1Y": 365,
    "2Y": 730
}
```

### Add Custom Strategies
1. Add strategy to appropriate list (`TRADITIONAL_STRATEGIES` or `ML_STRATEGIES`)
2. Ensure strategy is registered in the templates or ML modules
3. Add any custom parameter handling in the test methods

## Integration with CI/CD

The script is designed for automated testing:

```yaml
# GitHub Actions example
- name: Validate All Strategies
  run: make test-strategies
  timeout-minutes: 10
```

## Performance Benchmarks

Expected performance on modern hardware:
- **Total Runtime**: 2-5 minutes
- **Success Rate**: >90% for traditional, >80% for ML
- **Average Test Time**: 0.3-0.8 seconds per test
- **Memory Usage**: <1GB peak

## Development Notes

- Uses async/await for efficient data fetching
- Implements graceful error handling
- Provides progress indicators for long runs
- Generates synthetic data fallback for ML strategies
- Validates metrics for reasonableness (no NaN/infinite values)
- Saves detailed logs for debugging failures

This comprehensive testing ensures all backtesting strategies are working correctly and provides confidence in the system's reliability.