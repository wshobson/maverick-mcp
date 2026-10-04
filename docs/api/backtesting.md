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
extra (`vectorbt` and `scikit-learn`, which bring in `numba` and `scipy`). On
a base install with the extra absent, the server still boots cleanly and
registers **zero** `backtesting_*` tools -- see [Installation](#installation)
below.

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

### Equity and drawdown series in responses

Every `equity_curve` and `drawdown_series` in a tool response holds at most
60 points. This includes the ones nested in `individual_results`. A longer
series is sampled at evenly spaced positions, and the first and last dates
are always kept. Keys stay date strings and values stay floats. The full
daily series is still used for every metric and for the analysis, so only
the returned series is shortened. A full daily series is about 100,000
characters for five years of data, which is more than MCP clients will show
a model.

## Installation

The core install has no backtesting tools. Install the extra to enable all
12:

```bash
uv sync --extra backtesting
```

or, from the release tag:

```bash
pip install "maverick-mcp-server[backtesting] @ git+https://github.com/wshobson/maverick-mcp@v1.1.0"
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
    "profit_factor_status": "finite",
    "expectancy": 152.4,
    "total_trades": 24,
    "winning_trades": 14,
    "losing_trades": 10,
    "avg_win": 412.0,
    "avg_loss": -206.0,
    "best_trade": 1180.0,
    "worst_trade": -590.0,
    "avg_duration": 8.5,
    "kelly_criterion": 0.18,
    "recovery_factor": 1.9,
    "risk_reward_ratio": 2.0
  },
  "trades": [
    {
      "entry_date": "2023-01-15 00:00:00",
      "exit_date": "2023-02-10 00:00:00",
      "entry_price": 150.0,
      "exit_price": 158.5,
      "size": 66.0,
      "pnl": 561.0,
      "return": 0.057,
      "duration": "26 days 00:00:00"
    }
  ],
  "equity_curve": {"2023-01-03 00:00:00": 10000.0, "2023-01-04 00:00:00": 10012.5},
  "drawdown_series": {"2023-01-03 00:00:00": 0.0, "2023-01-04 00:00:00": -0.01},
  "start_date": "2023-01-03",
  "end_date": "2023-12-29",
  "initial_capital": 10000.0,
  "memory_stats": null,
  "analysis": {
    "performance_grade": "C",
    "risk_assessment": {
      "risk_level": "Low",
      "max_drawdown": 0.08,
      "sortino_ratio": 1.6,
      "calmar_ratio": 1.85,
      "recovery_factor": 1.9,
      "risk_adjusted_return": 1.6,
      "downside_protection": "Good"
    },
    "trade_quality": {
      "quality": "Good",
      "total_trades": 24,
      "frequency": "Low",
      "win_rate": 0.58,
      "avg_win": 412.0,
      "avg_loss": -206.0,
      "best_trade": 1180.0,
      "worst_trade": -590.0,
      "avg_duration_days": 8.5,
      "risk_reward_ratio": 2.0
    },
    "strengths": ["Low maximum drawdown", "Good return vs drawdown ratio"],
    "weaknesses": ["Room for optimization"],
    "recommendations": ["Consider position size of 18.0% based on Kelly Criterion"],
    "summary": "The strategy generated a 15.0% return with a Sharpe ratio of 1.20. Maximum drawdown was 8.0% with a 58.0% win rate across 24 trades. Performance is good with acceptable risk levels."
  },
  "status": "success"
}
```

`avg_win`, `avg_loss`, `best_trade`, `worst_trade`, and `expectancy` are
per-trade P&L in account currency, not returns. Each trade's `duration` is
elapsed calendar time between entry and exit timestamps. Open trades end at
the last observed bar. `avg_duration` and `avg_duration_days` are calendar days,
including fractional days. With no
trades, `trade_quality` reports `"quality": "No trades"`,
`"frequency": "None"`, and `null` for the per-trade fields. `equity_curve`
and `drawdown_series` hold at most 60 points (see
[Equity and drawdown series in responses](#equity-and-drawdown-series-in-responses)).

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
  "best_metric_status": null,
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
  "memory_stats": null,
  "status": "success"
}
```

### backtesting_walk_forward_analysis

Roll a strategy forward through repeated optimize/test windows to gauge
robustness. Each step re-optimizes on the preceding 504 calendar days with
the `coarse` grid, then tests the best parameters on the next window, so
`strategy` must be one of the five optimizable templates.

**Tool name**: `backtesting_walk_forward_analysis` (readOnlyHint: true)

**Parameters**:
- `symbol` (str, required)
- `strategy` (str, default: "sma_cross"): `sma_cross`, `rsi`, `macd`, `bollinger`, or `momentum`
- `start_date`, `end_date` (str, optional): defaults to the last 3 years (1095 days)
- `window_size` (int, default: 252): calendar days per test window
- `step_size` (int, default: 63): calendar days between windows

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
      "period": "2023-01-02 to 2023-09-11",
      "parameters": {"fast_period": 10, "slow_period": 20},
      "in_sample_sharpe": 1.3,
      "out_sample_return": 0.08,
      "out_sample_sharpe": 1.1,
      "out_sample_drawdown": -0.05
    }
  ],
  "summary": "Walk-forward analysis shows 12.0% average return with Sharpe ratio of 0.95. Strategy was profitable in 75% of periods. Results show moderate robustness with room for improvement.",
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

**Returns** (`MonteCarloResult`; percentile keys are `p5`/`p25`/`p50`/`p75`/`p95`,
and `var_95` is the `p5` return):
```json
{
  "num_simulations": 1000,
  "expected_return": 0.168,
  "return_std": 0.089,
  "return_percentiles": {"p5": 0.02, "p25": 0.10, "p50": 0.17, "p75": 0.23, "p95": 0.32},
  "expected_drawdown": -0.09,
  "drawdown_std": 0.03,
  "drawdown_percentiles": {"p5": -0.18, "p25": -0.11, "p50": -0.08, "p75": -0.05, "p95": -0.02},
  "probability_profit": 0.85,
  "var_95": 0.02,
  "summary": "Monte Carlo simulation shows 16.8% expected return with 85.0% probability of profit. 95% Value at Risk is 2.0%. Strategy shows strong probabilistic edge.",
  "status": "success"
}
```

A backtest with no trades has nothing to resample and returns an error.

### backtesting_compare_strategies

Backtest multiple strategies on the same symbol and rank them.

**Tool name**: `backtesting_compare_strategies` (readOnlyHint: true)

**Parameters**:
- `symbol` (str, required)
- `strategies` (list[str], optional): strategy keys to compare; defaults to
  `sma_cross`, `rsi`, `macd`, `bollinger`, `momentum`. A strategy whose
  backtest fails is left out of the rankings and listed in `failed` with
  its error message, and the summary says how many failed. If every
  strategy fails, the call returns the error `No strategies could be
  backtested: <strategy>: <error>; ...`.
- `start_date`, `end_date` (str, optional)

**Returns** (`StrategyComparisonResult`; `rankings` is sorted by Sharpe ratio,
each row's `max_drawdown` is the absolute value, and `failed` is `[]` when
every strategy ran):
```json
{
  "rankings": [
    {
      "strategy": "macd",
      "parameters": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
      "total_return": 0.28,
      "sharpe_ratio": 1.45,
      "max_drawdown": 0.07,
      "win_rate": 0.6,
      "profit_factor": 1.6,
    "profit_factor_status": "finite",
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
  "summary": "The best performing strategy is macd with a Sharpe ratio of 1.45 and total return of 28.0%. It outperformed 2 other strategies tested. 1 of 4 strategies failed (see failed).",
  "failed": [
    {"strategy": "not_a_strategy", "error": "Unknown strategy type: not_a_strategy"}
  ],
  "status": "success"
}
```

### backtesting_backtest_portfolio

Backtest one strategy across multiple symbols and aggregate portfolio-level
metrics. Per-symbol backtests run with bounded concurrency (a semaphore of
6, matching the legacy effective parallelism). Each symbol is backtested
independently -- there is no combined equity curve and no cross-symbol
correlation. `total_return`/`average_sharpe` are per-symbol averages;
`max_drawdown` is the worst (most negative) constituent drawdown, not a
joint-portfolio calculation. A symbol whose backtest fails is left out of
`individual_results` and the aggregate metrics and is listed in `failed`
with its error message, and the summary says how many failed. `failed` is
`[]` when every symbol ran. If every symbol fails, the call returns the
error `No symbols could be backtested: <symbol>: <error>; ...`. Each entry
in `individual_results` has the `backtesting_run_backtest` shape without
`analysis`, so its `equity_curve` and `drawdown_series` hold at most 60
points.

**Tool name**: `backtesting_backtest_portfolio` (readOnlyHint: true)

**Parameters**:
- `symbols` (list[str], required)
- `strategy` (str, default: "sma_cross")
- `start_date`, `end_date` (str, optional)
- `initial_capital` (float, default: 10000.0)
- `position_size` (float, default: 0.1): fraction of capital per symbol (each
  symbol's backtest starts with `initial_capital * position_size`)
- `fast_period`, `slow_period`, `period` (int, optional): strategy overrides

**Returns** (`PortfolioBacktestResult`):
```json
{
  "portfolio_metrics": {
    "symbols_tested": 4,
    "total_return": 0.22,
    "average_sharpe": 1.15,
    "max_drawdown": -0.12,
    "total_trades": 96
  },
  "individual_results": [
    {"symbol": "AAPL", "strategy": "sma_cross", "metrics": {"...": "..."}}
  ],
  "summary": "Portfolio backtest of 4 symbols with sma_cross strategy; 1 of 5 symbols failed (see failed)",
  "failed": [
    {"symbol": "XYZ", "error": "No price history available for 'XYZ' between 2023-01-01 and 2024-01-01"}
  ],
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
descriptive `code` field in `STRATEGY_TEMPLATES` rather than an executable
vectorbt-expression string, but their runtime signal generation
(`strategies/signals.py`) is real, self-contained pandas/numpy logic -- not
a stub and not a delegation to the ML strategy classes below.

### ML strategy classes (8 total)

Six of these back the ML-enhanced tools below, not `backtesting_run_backtest`
directly. `OnlineLearningStrategy` and `HybridAdaptiveStrategy` are ported
but no tool reaches them:

| Class | Module | Role |
| --- | --- | --- |
| `MLPredictor` | `strategies/ml/ml_predictor.py` | Random-forest price-movement classifier |
| `FeatureExtractor` | `strategies/ml/feature_engineering.py` | Technical-indicator feature pipeline |
| `AdaptiveStrategy` | `strategies/ml/adaptive.py` | Gradient or random-search parameter adaptation |
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
- `strategy_type` (str, default: "ml_predictor"): `ml_predictor`, `adaptive`, `ensemble`, or `regime_aware` (`online_learning` is accepted as an alias for `adaptive`)
- `start_date`, `end_date` (str, optional)
- `initial_capital` (float, default: 10000.0)
- `train_ratio` (float, default: 0.8): the first `train_ratio` of the bars
  train the model (`ml_predictor`) or fit the regime detector
  (`regime_aware`); signals and metrics come from the remaining bars only
- `model_type` (str, default: "random_forest")
- `n_estimators` (int, default: 100)
- `max_depth` (int, optional)
- `learning_rate` (float, default: 0.01)
- `adaptation_method` (str, default: "gradient"): `gradient` or `random_search`

**Returns** (`MLBacktestResult`; `trades` rows have the same shape as
`backtesting_run_backtest`'s):
```json
{
  "metrics": {
    "total_return": 0.24,
    "annual_return": 0.22,
    "sharpe_ratio": 1.35,
    "max_drawdown": -0.09,
    "win_rate": 0.62,
    "total_trades": 30,
    "profit_factor": 1.7,
    "profit_factor_status": "finite"
  },
  "trades": [{"entry_date": "2023-02-01 00:00:00", "exit_date": "2023-02-20 00:00:00", "entry_price": 150.0, "exit_price": 156.0, "size": 20.0, "pnl": 120.0, "return": 0.04, "duration": "19 days 00:00:00"}],
  "equity_curve": {"2023-01-03 00:00:00": 10000.0},
  "drawdown_series": {"2023-01-03 00:00:00": 0.0},
  "ml_metrics": {
    "strategy_type": "ml_predictor",
    "training_period": 400,
    "testing_period": 100,
    "train_test_split": 0.8,
    "training_metrics": {"train_accuracy": 0.68, "n_samples": 400, "n_features": 75, "...": "..."},
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
  "training_metrics": {
    "train_accuracy": 0.68,
    "n_samples": 500,
    "n_features": 75,
    "target_distribution": {"0": 171, "1": 174, "2": 155},
    "feature_importance": {"rsi": 0.25, "macd": 0.22}
  },
  "status": "success"
}
```

`train_accuracy` is in-sample accuracy on the training data; there is no
held-out evaluation.

### backtesting_analyze_market_regimes

Analyze market regimes (bear/sideways/bull) for a symbol using ML methods.

`method="hmm"` fits an `sklearn.mixture.GaussianMixture`, not a genuine
Hidden Markov Model -- the name is kept for backward compatibility.
`method="kmeans"` is real k-means clustering; `method="threshold"` is a
zero-model rule-based classifier on trend slope and volatility. The
returned `method` field reports what was **actually** used, which can
differ from the request: fitting silently falls back to `"threshold"` when
there isn't enough history to fit a genuine statistical model or the fit
itself fails (in practice, this tool's own default 365-day lookback is
usually *not* enough for `n_regimes=3`; use a longer `start_date`/`end_date`
range to fit a real model). Each `recent_regime_history` entry's
`probabilities` are the fitted model's real posterior when one exists,
otherwise an honest one-hot vector at the assigned regime -- never a
fabricated uniform distribution.

**Tool name**: `backtesting_analyze_market_regimes` (readOnlyHint: true)

**Parameters**:
- `symbol` (str, required)
- `start_date`, `end_date` (str, optional)
- `method` (str, default: "hmm"): `hmm` (Gaussian-mixture clustering; see above), `kmeans`, or `threshold`
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

Each entry in `base_strategies` must be a valid name from the strategy
catalog (`backtesting_list_strategies`) -- for example `"sma_cross"`,
`"rsi"`, `"macd"`, `"bollinger"`, `"momentum"`. Each runs its own real
signal logic (so `"rsi"` is genuine RSI mean-reversion, not a relabeled
SMA-crossover variant) and keeps its own template name in the result, so
`final_strategy_weights`/`strategy_performance_analysis` stay separately
addressable per requested strategy rather than collapsing onto one shared
key. An unknown name raises a clear error rather than being silently
dropped.

Only the first five `symbols` are backtested. A symbol with fewer than 100
bars in the range, or whose backtest fails, is skipped, and the call errors
only when no symbol succeeds.

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
        "ensemble_metrics": {"strategy_weights": {"SMA Crossover": 0.4, "RSI Mean Reversion": 0.3, "MACD Signal": 0.3}}
      }
    }
  ],
  "final_strategy_weights": {"SMA Crossover": 0.42, "RSI Mean Reversion": 0.28, "MACD Signal": 0.30},
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
strategy name (`ValueError` from `strategies.signals.generate_signals`), an
unsupported `optimize_strategy` or `walk_forward_analysis` target (only
`sma_cross`/`rsi`/`macd`/`bollinger`/`momentum` have parameter grids), or an
analysis that exceeds `BacktestingSettings.analysis_timeout_seconds` (120s;
not env-configurable -- only initial capital, fees, and slippage read
`BACKTESTING_*` env vars, see `maverick/backtesting/config.py`).

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
import asyncio

from fastmcp import Client


async def main():
    # `make dev` serves Streamable HTTP here (no trailing slash).
    async with Client("http://localhost:8003/mcp") as client:
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


asyncio.run(main())
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
2. Use `backtesting_backtest_portfolio` to see how one strategy holds up across symbols (each symbol runs on its own; it does not model diversification).
3. Watch `max_drawdown` and `sortino_ratio` in the `analysis.risk_assessment` block, not just `total_return`.

## Ensemble weight timing

Ensemble weights use lagged strategy returns strictly before each rebalance
boundary, over the configured lookback. Each set of weights applies from that
boundary to the bar before the next boundary. Weights stay equal when history
is insufficient, and each run starts with fresh state.

The return proxy uses the previous bar's entry or exit signal and the current
bar's price return. It is not a simulation of held-position profit and loss.
Later prices do not revise earlier ensemble weights or signals when the
component strategies themselves use only information available at each bar.

## Feature warm-up

Machine learning features use earlier observations to fill missing values, then
use zero when no earlier value is available. RSI features wait for their full
warm-up period before exposing a value or threshold flag. Stochastic features
handle a zero price range at that row, so a later flat period cannot change
earlier features. The separate technical-indicator compatibility formulas stay
unchanged. Prediction uses the scaler fitted on training data.


## Metric conventions

Annual return, Sharpe, Sortino, and Calmar use 252 trading sessions per year in
both ordinary backtests and grid optimization. This is a session convention,
not an exchange-calendar count. The engine does not change global vectorbt settings.

Profit factor divides positive trade P&L by the absolute value of negative trade
P&L. The existing vectorbt all-trades set includes open trades marked to market
at the last observed bar. Full, simplified, and comparison results include
`profit_factor_status` alongside the nullable `profit_factor` value.

| Status | Meaning | Numeric value |
| --- | --- | --- |
| `finite` | Losses exist. | Finite ratio, including zero for loss-only trades. |
| `no_losses` | Positive profit exists without losses. | `null` |
| `no_trades` | There are no trade records. | `null` |
| `no_realized_pnl` | All recorded trade P&L is zero. | `null` |

The last status is the API name for breakeven-only evidence; it does not restrict
the trade set to closed trades. Undefined ratios are never serialized as infinity
or substituted with zero. Profit-factor optimization ranks `no_losses` first,
then finite ratios from highest to lowest, then no-trade and breakeven-only
results tied last. A no-loss winner has `best_metric_value: null` and
`best_metric_status: no_losses`; other optimization metrics have a null status.
