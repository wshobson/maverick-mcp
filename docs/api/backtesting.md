# Backtesting API Documentation

This is the canonical backtesting documentation for `maverick.backtesting`,
the phase 6 domain port (with `backtesting_parse_strategy` added in phase 7
on the new BYOK LLM seam). It describes the 12 `backtesting_*` MCP tools
registered from `maverick/backtesting/tools.py` and `tools_ml.py`.

## Overview

MaverickMCP provides VectorBT-powered backtesting with rule-based technical
strategies, ML-enhanced strategies, parameter optimization, walk-forward
analysis, Monte Carlo simulation, and multi-symbol portfolio backtesting.

The backtesting surface lives behind the optional `[backtesting]` dependency
extra (`vectorbt`, `numba`, `scikit-learn`, `scipy`, `pandas-ta`). On a base
install with the extra absent, the server still boots cleanly and registers
**zero** `backtesting_*` tools -- see [Installation](#installation) below.

### Key Features

- **20 strategies total**: 12 rule-based templates (`STRATEGY_TEMPLATES`)
  plus 8 ML strategy classes -- not "23" or "35+", which were stale claims
  from the legacy documentation.
- **Strategy Optimization**: Grid search with coarse/medium/fine granularity
  (5 of the 12 strategies support optimization: `sma_cross`, `rsi`, `macd`,
  `bollinger`, `momentum`).
- **Walk-Forward Analysis**: Out-of-sample validation for strategy
  robustness.
- **Monte Carlo Simulation**: Bootstrap-resampled return/drawdown
  distributions.
- **Portfolio Backtesting**: Multi-symbol strategy application.
- **Market Regime Analysis**: ML-based detection of bear/sideways/bull
  regimes.
- **ML-Enhanced Strategies**: Adaptive, ensemble, and regime-aware
  approaches.

### Not in this surface

- **Chart generation** (`generate_backtest_charts`,
  `generate_optimization_charts`) does not exist in this port, consistent
  with the rest of the modernized server's no-chart-images decision.
- **Persistence.** None of the 12 tools write to a database. A review of
  every legacy call site found zero calls into the old
  `BacktestPersistenceManager`, so this port carries no store -- all 12
  tools are `readOnlyHint=True`. The five `mcp_backtest_*` table
  definitions remain in git history for future reintroduction if a real
  consumer emerges.
- **Intelligent-backtesting workflow** (`run_intelligent_backtest`,
  `quick_market_regime_analysis`, `explain_market_regime`) was dead code
  (orphaned agent workflow with zero live callers) and was deleted, not
  ported.

## Installation

The core install has no backtesting tools. Install the extra to enable all
12:

```bash
uv sync --extra backtesting
```

or, from a published wheel:

```bash
pip install "maverick-mcp-server[backtesting]"
```

If the extra is absent, `maverick.backtesting.tools.register()` logs one
clear warning and registers zero tools -- the server boots normally with no
traceback either way. `import maverick.backtesting` itself always succeeds
on a base install: payload types, settings, and the rule-based strategy
catalog are importable without the extra; only the vectorbt/scikit-learn-
backed members (`BacktestingService`, the engine, ML strategy classes) raise
a clear `ImportError` naming the extra if accessed without it installed.

## Core Backtesting Tools

### backtesting_run_backtest

Run a single-strategy backtest and return metrics, trades, and analysis.

**Tool name**: `backtesting_run_backtest` (readOnlyHint: true)

**Parameters**:
- `symbol` (str, required): Stock symbol to backtest (e.g., "AAPL", "TSLA")
- `strategy` (str, default: "sma_cross"): One of the 12 `STRATEGY_TEMPLATES` keys
- `start_date` (str, optional): Start date (YYYY-MM-DD), defaults to 1 year ago
- `end_date` (str, optional): End date (YYYY-MM-DD), defaults to today
- `initial_capital` (float, default: 10000.0): Starting capital for the backtest

**Strategy-specific overrides** (only the ones relevant to the chosen strategy apply):
- `fast_period`, `slow_period` (int, optional): SMA/EMA/MACD crossover periods
- `period` (int, optional): RSI/Bollinger period
- `oversold`, `overbought` (float, optional): RSI thresholds
- `signal_period` (int, optional): MACD signal line period
- `std_dev` (float, optional): Bollinger Bands standard deviation
- `lookback` (int, optional): Momentum/breakout lookback
- `threshold` (float, optional): Momentum threshold
- `z_score_threshold` (float, optional): Mean-reversion z-score threshold
- `breakout_factor` (float, optional): Breakout factor

**Returns** (`RunBacktestResult`, `status: "success"` merged in):
```json
{
  "symbol": "AAPL",
  "strategy": "sma_cross",
  "parameters": {"fast_period": 10, "slow_period": 20},
  "metrics": {
    "total_return": 0.15,
    "annual_return": 0.14,
    "sharpe_ratio": 1.2,
    "sortino_ratio": 1.6,
    "calmar_ratio": 1.85,
    "max_drawdown": -0.08,
    "win_rate": 0.58,
    "profit_factor": 1.45,
    "expectancy": 0.02,
    "total_trades": 24,
    "winning_trades": 14,
    "losing_trades": 10,
    "avg_win": 0.04,
    "avg_loss": -0.02,
    "best_trade": 0.12,
    "worst_trade": -0.06,
    "avg_duration": 8.5,
    "kelly_criterion": 0.18,
    "recovery_factor": 1.9,
    "risk_reward_ratio": 2.0
  },
  "trades": [
    {
      "entry_date": "2023-01-15",
      "exit_date": "2023-02-10",
      "entry_price": 150.0,
      "exit_price": 158.5,
      "size": 66.0,
      "pnl": 561.0,
      "return": 0.057,
      "duration": "26 days"
    }
  ],
  "equity_curve": {"2023-01-01": 10000.0, "2023-01-02": 10012.5},
  "drawdown_series": {"2023-01-01": 0.0, "2023-01-02": -0.01},
  "start_date": "2023-01-01",
  "end_date": "2024-01-01",
  "initial_capital": 10000.0,
  "analysis": {
    "performance_grade": "B",
    "risk_assessment": {
      "risk_level": "moderate",
      "max_drawdown": -0.08,
      "sortino_ratio": 1.6,
      "calmar_ratio": 1.85,
      "recovery_factor": 1.9,
      "risk_adjusted_return": 0.17,
      "downside_protection": "good"
    },
    "trade_quality": {
      "quality": "good",
      "total_trades": 24,
      "frequency": "moderate"
    },
    "strengths": ["Consistent positive expectancy"],
    "weaknesses": [],
    "recommendations": ["Consider wider stops in choppy markets"],
    "summary": "Solid risk-adjusted returns with moderate drawdown."
  },
  "status": "success"
}
```

### backtesting_optimize_strategy

Grid-search a strategy's parameters. **Only 5 of the 12 templates are
supported** -- `sma_cross`, `rsi`, `macd`, `bollinger`, `momentum` -- a
faithfully-preserved legacy limitation (`generate_param_grid` raises
`ValueError` for every other strategy).

**Tool name**: `backtesting_optimize_strategy` (readOnlyHint: true)

**Parameters**:
- `symbol` (str, required)
- `strategy` (str, default: "sma_cross"): `sma_cross`, `rsi`, `macd`, `bollinger`, or `momentum`
- `start_date`, `end_date` (str, optional)
- `optimization_metric` (str, default: "sharpe_ratio")
- `optimization_level` (str, default: "medium"): `coarse`, `medium`, or `fine`
- `top_n` (int, default: 10)

**Returns** (`OptimizationResult`):
```json
{
  "symbol": "AAPL",
  "strategy": "sma_cross",
  "optimization_metric": "sharpe_ratio",
  "best_parameters": {"fast_period": 8, "slow_period": 21},
  "best_metric_value": 1.85,
  "top_results": [
    {
      "parameters": {"fast_period": 8, "slow_period": 21},
      "total_return": 0.28,
      "max_drawdown": -0.06,
      "total_trades": 18,
      "sharpe_ratio": 1.85
    }
  ],
  "total_combinations_tested": 64,
  "valid_combinations": 61,
  "status": "success"
}
```

### backtesting_walk_forward_analysis

Roll a strategy forward through repeated optimize/test windows to gauge
robustness.

**Tool name**: `backtesting_walk_forward_analysis` (readOnlyHint: true)

**Parameters**:
- `symbol` (str, required)
- `strategy` (str, default: "sma_cross")
- `start_date`, `end_date` (str, optional)
- `window_size` (int, default: 252): trading days per test window
- `step_size` (int, default: 63): rolling step between windows

**Returns** (`WalkForwardResult`):
```json
{
  "symbol": "AAPL",
  "strategy": "sma_cross",
  "periods_tested": 8,
  "average_return": 0.12,
  "average_sharpe": 0.95,
  "average_drawdown": -0.09,
  "consistency": 0.75,
  "walk_forward_results": [
    {
      "period": "2023-Q1",
      "parameters": {"fast_period": 10, "slow_period": 25},
      "in_sample_sharpe": 1.3,
      "out_sample_return": 0.08,
      "out_sample_sharpe": 1.1,
      "out_sample_drawdown": -0.05
    }
  ],
  "summary": "8 windows tested; average out-of-sample Sharpe 0.95.",
  "status": "success"
}
```

### backtesting_monte_carlo_simulation

Bootstrap-resample a backtest's trades to estimate a return/drawdown
distribution.

**Tool name**: `backtesting_monte_carlo_simulation` (readOnlyHint: true)

**Parameters**:
- `symbol` (str, required)
- `strategy` (str, default: "sma_cross")
- `start_date`, `end_date` (str, optional)
- `num_simulations` (int, default: 1000)
- `fast_period`, `slow_period`, `period` (int, optional): strategy overrides

**Returns** (`MonteCarloResult`; percentile keys are `p5`/`p25`/`p50`/`p75`/`p95`):
```json
{
  "num_simulations": 1000,
  "expected_return": 0.168,
  "return_std": 0.089,
  "return_percentiles": {"p5": 0.02, "p25": 0.10, "p50": 0.17, "p75": 0.23, "p95": 0.32},
  "expected_drawdown": -0.09,
  "drawdown_std": 0.03,
  "drawdown_percentiles": {"p5": -0.18, "p50": -0.08, "p95": -0.02},
  "probability_profit": 0.85,
  "var_95": -0.12,
  "summary": "85% probability of profit across 1000 simulations.",
  "status": "success"
}
```

### backtesting_compare_strategies

Backtest multiple strategies on the same symbol and rank them.

**Tool name**: `backtesting_compare_strategies` (readOnlyHint: true)

**Parameters**:
- `symbol` (str, required)
- `strategies` (list[str], optional): strategy keys to compare
- `start_date`, `end_date` (str, optional)

**Returns** (`StrategyComparisonResult`):
```json
{
  "rankings": [
    {
      "strategy": "macd",
      "parameters": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
      "total_return": 0.28,
      "sharpe_ratio": 1.45,
      "max_drawdown": -0.07,
      "win_rate": 0.6,
      "profit_factor": 1.6,
      "total_trades": 20,
      "grade": "A",
      "rank": 1
    }
  ],
  "best_overall": {"strategy": "macd", "rank": 1, "...": "..."},
  "best_return": {"strategy": "macd", "...": "..."},
  "best_sharpe": {"strategy": "macd", "...": "..."},
  "best_drawdown": {"strategy": "sma_cross", "...": "..."},
  "best_win_rate": {"strategy": "macd", "...": "..."},
  "summary": "macd ranked best overall across 3 strategies compared.",
  "status": "success"
}
```

### backtesting_backtest_portfolio

Backtest one strategy across multiple symbols and aggregate portfolio-level
metrics. Per-symbol backtests run with bounded concurrency (a semaphore of
6, matching the legacy effective parallelism).

**Tool name**: `backtesting_backtest_portfolio` (readOnlyHint: true)

**Parameters**:
- `symbols` (list[str], required)
- `strategy` (str, default: "sma_cross")
- `start_date`, `end_date` (str, optional)
- `initial_capital` (float, default: 10000.0)
- `position_size` (float, default: 0.1): fraction of capital per symbol
- `fast_period`, `slow_period`, `period` (int, optional): strategy overrides

**Returns** (`PortfolioBacktestResult`):
```json
{
  "portfolio_metrics": {
    "symbols_tested": 5,
    "total_return": 0.22,
    "average_sharpe": 1.15,
    "max_drawdown": -0.12,
    "total_trades": 120
  },
  "individual_results": [
    {"symbol": "AAPL", "strategy": "sma_cross", "metrics": {"...": "..."}}
  ],
  "summary": "Portfolio backtest of 5 symbols with sma_cross strategy.",
  "status": "success"
}
```

## Strategy Management

### backtesting_list_strategies

List every available rule-based strategy template with its default
parameters.

**Tool name**: `backtesting_list_strategies` (readOnlyHint: true)

**Parameters**: None

**Returns** (`StrategyCatalog`, all 12 `STRATEGY_TEMPLATES` entries):
```json
{
  "available_strategies": {
    "sma_cross": {
      "type": "sma_cross",
      "name": "SMA Crossover",
      "description": "Buy when fast SMA crosses above slow SMA, sell when it crosses below",
      "default_parameters": {"fast_period": 10, "slow_period": 20},
      "optimization_ranges": {"fast_period": [5, 10, 15, 20], "slow_period": [20, 30, 50, 100]}
    }
  },
  "total_count": 12,
  "categories": {"...": "..."},
  "status": "success"
}
```

### backtesting_parse_strategy

Parse a natural-language strategy description into a strategy type and
parameters. Unlike the other 11 tools, it needs no `BacktestingService` or
`vectorbt` at all (`strategies/parser.py`'s `StrategyParser` is pure Python
plus, optionally, the BYOK `platform.llm` seam) -- it registers only under
the same `[backtesting]`-extra guard as the rest of the domain for one
consistent registration surface, not because it needs vectorbt.

**Tool name**: `backtesting_parse_strategy` (readOnlyHint: true)

**Parameters**:
- `description` (str, required): natural-language strategy description,
  e.g. `"Buy when the 10-day SMA crosses above the 20-day SMA"`

**Behavior**: tries the configured BYOK LLM first
(`maverick.platform.llm.get_llm()`); degrades to zero-dependency
keyword/regex matching against `STRATEGY_TEMPLATES`
(`StrategyParser.parse_simple`) whenever no LLM is configured, its provider
package isn't installed, or the model's response isn't valid JSON -- this
degrade path is not a regression, it is the only path the live legacy tool
ever actually exercised (`parse_with_llm`'s `llm` argument was never wired
to any configuration in the legacy call graph).

**Returns**: unlike the other 11 tools, success responses use a `"success"`
boolean rather than `"status": "success"` (only the exception path returns
`"status": "error"`):
```json
{
  "success": true,
  "strategy": {
    "strategy_type": "sma_cross",
    "parameters": {"fast_period": 10, "slow_period": 20}
  },
  "method": "llm",
  "message": "Successfully parsed as sma_cross strategy"
}
```

`"method"` is `"llm"` for a successful model-backed parse or
`"simple_degraded"` whenever it fell back to keyword parsing. `"success"`
is `false` (with the same shape) when the parsed configuration doesn't
validate against the matched template's required parameters.

## Available Strategies

### Rule-based templates (`STRATEGY_TEMPLATES`, 12 total)

| Key | Name | Default parameters | Optimizable |
| --- | --- | --- | --- |
| `sma_cross` | SMA Crossover | `fast_period=10, slow_period=20` | yes |
| `rsi` | RSI Mean Reversion | `period=14, oversold=30, overbought=70` | yes |
| `macd` | MACD Signal | `fast_period=12, slow_period=26, signal_period=9` | yes |
| `bollinger` | Bollinger Bands | `period=20, std_dev=2.0` | yes |
| `momentum` | Momentum | `lookback=20, threshold=0.05` | yes |
| `ema_cross` | EMA Crossover | `fast_period=12, slow_period=26` | no |
| `mean_reversion` | Mean Reversion | `ma_period=20, entry_threshold=0.02, exit_threshold=0.01` | no |
| `breakout` | Channel Breakout | `lookback=20, exit_lookback=10` | no |
| `volume_momentum` | Volume-Weighted Momentum | `momentum_period=20, volume_period=20, momentum_threshold=0.05, volume_multiplier=1.5` | no |
| `online_learning` | Online Learning Strategy | `lookback=20, learning_rate=0.01, update_frequency=5` | no |
| `regime_aware` | Regime-Aware Strategy | `regime_window=50, threshold=0.02, trend_strategy=momentum, range_strategy=mean_reversion` | no |
| `ensemble` | Ensemble Strategy | `fast_period=10, slow_period=20, rsi_period=14, weight_method=equal` | no |

"Optimizable" means `backtesting_optimize_strategy` supports it; every
template works with `backtesting_run_backtest` and `backtesting_compare_strategies`.
The last three (`online_learning`, `regime_aware`, `ensemble`) carry a
descriptive `code` field in the catalog rather than an executable
vectorbt-expression string, but their runtime signal generation
(`strategies/signals.py`) is real, self-contained pandas/numpy logic -- not
a stub and not a delegation to the ML strategy classes below.

### ML strategy classes (8 total)

These back the ML-enhanced tools below, not `backtesting_run_backtest`
directly:

| Class | Module | Role |
| --- | --- | --- |
| `MLPredictor` | `strategies/ml/ml_predictor.py` | Random-forest price-movement classifier |
| `FeatureExtractor` | `strategies/ml/feature_engineering.py` | Technical-indicator feature pipeline |
| `AdaptiveStrategy` | `strategies/ml/adaptive.py` | Gradient/momentum parameter adaptation |
| `OnlineLearningStrategy` | `strategies/ml/online_learning.py` | Streaming SGD classifier |
| `HybridAdaptiveStrategy` | `strategies/ml/hybrid_adaptive.py` | Combines adaptive + online-learning signals |
| `RegimeAwareStrategy` | `strategies/ml/regime_aware.py` | Switches base strategy by detected regime |
| `MarketRegimeDetector` | `strategies/ml/regime_detector.py` | KMeans/GMM regime clustering |
| `StrategyEnsemble` | `strategies/ml/ensemble.py` | Weighted multi-strategy voting |

## ML-Enhanced Strategies

### backtesting_run_ml_strategy_backtest

Run a backtest using an ML-enhanced strategy.

**Tool name**: `backtesting_run_ml_strategy_backtest` (readOnlyHint: true)

**Parameters**:
- `symbol` (str, required)
- `strategy_type` (str, default: "ml_predictor"): `ml_predictor`, `adaptive`, `ensemble`, or `regime_aware`
- `start_date`, `end_date` (str, optional)
- `initial_capital` (float, default: 10000.0)
- `train_ratio` (float, default: 0.8)
- `model_type` (str, default: "random_forest")
- `n_estimators` (int, default: 100)
- `max_depth` (int, optional)
- `learning_rate` (float, default: 0.01)
- `adaptation_method` (str, default: "gradient")

**Returns** (`MLBacktestResult`):
```json
{
  "metrics": {
    "total_return": 0.24,
    "annual_return": 0.22,
    "sharpe_ratio": 1.35,
    "max_drawdown": -0.09,
    "win_rate": 0.62,
    "total_trades": 30,
    "profit_factor": 1.7
  },
  "trades": [{"entry_time": "2023-02-01", "exit_time": "2023-02-20", "pnl": 120.0, "return": 0.04}],
  "equity_curve": {"2023-01-01": 10000.0},
  "drawdown_series": {"2023-01-01": 0.0},
  "ml_metrics": {
    "training_period": 400,
    "testing_period": 100,
    "model_accuracy": 0.68,
    "feature_importance": {"rsi": 0.25, "macd": 0.22}
  },
  "status": "success"
}
```

### backtesting_train_ml_predictor

Train a random-forest ML predictor model for trading signals.

**Tool name**: `backtesting_train_ml_predictor` (readOnlyHint: true)

**Parameters**:
- `symbol` (str, required)
- `start_date`, `end_date` (str, optional)
- `model_type` (str, default: "random_forest")
- `target_periods` (int, default: 5)
- `return_threshold` (float, default: 0.02)
- `n_estimators` (int, default: 100)
- `max_depth` (int, optional)
- `min_samples_split` (int, default: 2)

**Returns** (`MLTrainingResult`):
```json
{
  "symbol": "AAPL",
  "model_type": "random_forest",
  "training_period": "2022-01-01 to 2024-01-01",
  "data_points": 500,
  "target_periods": 5,
  "return_threshold": 0.02,
  "model_parameters": {"n_estimators": 100, "max_depth": 10, "min_samples_split": 2},
  "training_metrics": {"accuracy": 0.68, "precision": 0.72, "recall": 0.65, "f1_score": 0.68},
  "status": "success"
}
```

### backtesting_analyze_market_regimes

Analyze market regimes (bear/sideways/bull) for a symbol using ML methods.

**Tool name**: `backtesting_analyze_market_regimes` (readOnlyHint: true)

**Parameters**:
- `symbol` (str, required)
- `start_date`, `end_date` (str, optional)
- `method` (str, default: "hmm"): `hmm`, `kmeans`, or `threshold`
- `n_regimes` (int, default: 3)
- `lookback_period` (int, default: 50)

**Returns** (`MarketRegimeAnalysis`):
```json
{
  "symbol": "AAPL",
  "analysis_period": "2023-01-01 to 2024-01-01",
  "method": "hmm",
  "n_regimes": 3,
  "regime_names": {"0": "Bear/Declining", "1": "Sideways/Uncertain", "2": "Bull/Trending"},
  "current_regime": 2,
  "regime_counts": {"0": 45, "1": 89, "2": 118},
  "regime_percentages": {"0": 17.9, "1": 35.3, "2": 46.8},
  "average_regime_durations": {"0": 15.2, "1": 22.3, "2": 28.7},
  "recent_regime_history": [{"date": "2024-01-15", "regime": 2, "probabilities": [0.05, 0.15, 0.80]}],
  "total_regime_switches": 18,
  "status": "success"
}
```

### backtesting_create_strategy_ensemble

Create and backtest a weighted ensemble of base strategies across multiple
symbols. Runs sequentially by design: `StrategyEnsemble` shares one mutable
instance across symbols (weights mutate per call), so concurrency would
make results order-dependent.

**Tool name**: `backtesting_create_strategy_ensemble` (readOnlyHint: true)

**Parameters**:
- `symbols` (list[str], required)
- `base_strategies` (list[str], optional; defaults to `["sma_cross", "rsi", "macd"]`)
- `weighting_method` (str, default: "performance"): `performance`, `equal`, or `volatility`
- `start_date`, `end_date` (str, optional)
- `initial_capital` (float, default: 10000.0)

**Returns** (`EnsembleBacktestResult`):
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
        "metrics": {"total_return": 0.21, "sharpe_ratio": 1.18},
        "ensemble_metrics": {"strategy_weights": {"sma_cross": 0.4, "rsi": 0.3, "macd": 0.3}}
      }
    }
  ],
  "final_strategy_weights": {"sma_cross": 0.42, "rsi": 0.28, "macd": 0.30},
  "strategy_performance_analysis": {"...": "..."},
  "status": "success"
}
```

## Error Handling

Every tool catches its own exceptions and returns a consistent error shape
instead of raising -- there is no separate error schema per tool:

```json
{"status": "error", "error": "No price history available for 'PENNY_STOCK' between 2023-01-01 and 2024-01-01"}
```

Common causes: an empty or too-short price history fetch, an unsupported
strategy name (`ValueError` from `get_strategy_template`), an unsupported
`optimize_strategy` target (only `sma_cross`/`rsi`/`macd`/`bollinger`/
`momentum` have parameter grids), or an analysis that exceeds
`BacktestingSettings.analysis_timeout_seconds` (120s by default, configurable
via `BACKTESTING_*` env vars -- see `maverick/backtesting/config.py`).

## Integration Examples

### Claude Desktop usage

```
# Basic backtest
"Run a backtest for AAPL using the RSI strategy with a 14-day period"

# Strategy comparison
"Compare SMA crossover, RSI, and MACD strategies on Tesla stock"

# Portfolio backtest
"Backtest the momentum strategy on AAPL, MSFT, GOOGL, AMZN, and TSLA"

# Optimization
"Optimize MACD parameters for Netflix stock over the last 2 years"

# ML strategies
"Train an ML predictor on Amazon stock and test its performance"
```

### MCP client usage

```python
import mcp

client = mcp.Client("maverick-mcp")

# Run a backtest
result = await client.call_tool("backtesting_run_backtest", {
    "symbol": "AAPL",
    "strategy": "sma_cross",
    "fast_period": 10,
    "slow_period": 20,
    "initial_capital": 50000,
})

# Optimize a strategy
optimization = await client.call_tool("backtesting_optimize_strategy", {
    "symbol": "TSLA",
    "strategy": "rsi",
    "optimization_level": "medium",
    "optimization_metric": "sharpe_ratio",
})

# List the strategy catalog
catalog = await client.call_tool("backtesting_list_strategies", {})
```

## Best Practices

### Strategy selection
1. Start with `backtesting_list_strategies` to see the current catalog and default parameters.
2. Use `backtesting_compare_strategies` before committing to one strategy for a symbol.
3. Only `sma_cross`, `rsi`, `macd`, `bollinger`, and `momentum` support `backtesting_optimize_strategy`.

### Parameter optimization
1. Test default parameters first with `backtesting_run_backtest`.
2. Use `optimization_level="medium"` for a balance of thoroughness and speed.
3. Validate optimized parameters with `backtesting_walk_forward_analysis` before trusting them.

### Risk management
1. Use `backtesting_monte_carlo_simulation` to understand the distribution of outcomes, not just the point estimate.
2. Use `backtesting_backtest_portfolio` to see diversification effects across symbols.
3. Watch `max_drawdown` and `sortino_ratio` in the `analysis.risk_assessment` block, not just `total_return`.
