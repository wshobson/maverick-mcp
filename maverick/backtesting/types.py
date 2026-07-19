"""Backtesting payload types. Bottom layer: imports nothing from this domain.

Field sets are derived from the 11 surviving backtesting tools in
`maverick_mcp/api/routers/backtesting.py` and the underlying result dicts in
`maverick_mcp/backtesting/vectorbt_engine.py` (`_extract_metrics`,
`_extract_trades`, `run_backtest`, `optimize_parameters`) and
`maverick_mcp/backtesting/analysis.py` (`BacktestAnalyzer.analyze`,
`BacktestAnalyzer.run_vectorbt_backtest`, `BacktestAnalyzer.compare_strategies`)
plus `maverick_mcp/backtesting/optimization.py`
(`StrategyOptimizer.walk_forward_analysis`, `StrategyOptimizer.monte_carlo_simulation`).

Two distinct "metrics" and "trade" shapes exist in the legacy code depending on
which execution path produced them:

- `BacktestMetrics` / `TradeRecord`: the full-fidelity shapes produced by
  `VectorBTEngine._extract_metrics` / `_extract_trades` (used by `run_backtest`,
  `compare_strategies`, `backtest_portfolio`).
- `SimpleBacktestMetrics`: the smaller shape produced by
  `BacktestAnalyzer.run_vectorbt_backtest` (used by `run_ml_strategy_backtest`
  and `create_strategy_ensemble`). Its trade rows have no fixed shape (they
  vary with a fallback branch), so they are modeled as `dict[str, Any]`.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# -- Core backtest metrics and trades ---------------------------------------


class BacktestMetrics(BaseModel):
    """Full metrics shape from `VectorBTEngine._extract_metrics`.

    Every field is produced via `safe_float_metric`/`int()` casts in the
    legacy code, so none of these are ever `None` -- invalid values are
    coerced to `0.0` upstream.
    """

    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    expectancy: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    avg_duration: float
    kelly_criterion: float
    recovery_factor: float
    risk_reward_ratio: float


class TradeRecord(BaseModel):
    """Trade row shape from `VectorBTEngine._extract_trades`."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    return_: float = Field(alias="return")
    duration: str


class SimpleBacktestMetrics(BaseModel):
    """Smaller metrics shape from `BacktestAnalyzer.run_vectorbt_backtest`."""

    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profit_factor: float


# -- run_backtest -------------------------------------------------------


class BacktestResult(BaseModel):
    """Raw result shape from `VectorBTEngine.run_backtest`.

    This is also the shape of each item in `backtest_portfolio`'s
    `individual_results` and each item fed into `compare_strategies`.
    """

    symbol: str
    strategy: str
    parameters: dict[str, Any]
    metrics: BacktestMetrics
    trades: list[TradeRecord]
    equity_curve: dict[str, float]
    drawdown_series: dict[str, float]
    start_date: str
    end_date: str
    initial_capital: float
    memory_stats: dict[str, Any] | None = None


class RiskAssessment(BaseModel):
    risk_level: str
    max_drawdown: float
    sortino_ratio: float
    calmar_ratio: float
    recovery_factor: float
    risk_adjusted_return: float
    downside_protection: str


class TradeQuality(BaseModel):
    """From `BacktestAnalyzer._analyze_trades`.

    When there are no trades, the legacy code returns only `quality`,
    `total_trades`, and `frequency` -- the remaining fields are absent from
    the dict entirely rather than present as `None`, so they default to
    `None` here.
    """

    quality: str
    total_trades: int
    frequency: str
    win_rate: float | None = None
    avg_win: float | None = None
    avg_loss: float | None = None
    best_trade: float | None = None
    worst_trade: float | None = None
    avg_duration_days: float | None = None
    risk_reward_ratio: float | None = None


class BacktestAnalysis(BaseModel):
    """From `BacktestAnalyzer.analyze`."""

    performance_grade: str
    risk_assessment: RiskAssessment
    trade_quality: TradeQuality
    strengths: list[str]
    weaknesses: list[str]
    recommendations: list[str]
    summary: str


class RunBacktestResult(BacktestResult):
    """Return shape of the `run_backtest` tool: the raw engine result with
    `analysis` merged in by the router."""

    analysis: BacktestAnalysis


# -- optimize_strategy ----------------------------------------------------


class OptimizationResultRow(BaseModel):
    """One row of `optimize_parameters`' `top_results`.

    The legacy dict also carries a key named after whichever
    `optimization_metric` was requested (e.g. `"sharpe_ratio"`), which is not
    a fixed field name -- it is preserved via `extra="allow"` rather than
    invented as a fixed field.
    """

    model_config = ConfigDict(extra="allow")

    parameters: dict[str, Any]
    total_return: float
    max_drawdown: float
    total_trades: int


class OptimizationResult(BaseModel):
    """Return shape of the `optimize_strategy` tool
    (`VectorBTEngine.optimize_parameters`)."""

    symbol: str
    strategy: str
    optimization_metric: str
    best_parameters: dict[str, Any]
    best_metric_value: float
    top_results: list[OptimizationResultRow]
    total_combinations_tested: int
    valid_combinations: int
    memory_stats: dict[str, Any] | None = None


# -- walk_forward_analysis ------------------------------------------------


class WalkForwardPeriodResult(BaseModel):
    period: str
    parameters: dict[str, Any]
    in_sample_sharpe: float
    out_sample_return: float
    out_sample_sharpe: float
    out_sample_drawdown: float


class WalkForwardResult(BaseModel):
    """Return shape of the `walk_forward_analysis` tool
    (`StrategyOptimizer.walk_forward_analysis`)."""

    symbol: str
    strategy: str
    periods_tested: int
    average_return: float
    average_sharpe: float
    average_drawdown: float
    consistency: float
    walk_forward_results: list[WalkForwardPeriodResult]
    summary: str


# -- monte_carlo_simulation ------------------------------------------------


class MonteCarloResult(BaseModel):
    """Return shape of the `monte_carlo_simulation` tool
    (`StrategyOptimizer.monte_carlo_simulation`). Percentile band keys are
    `p{N}` for each requested confidence level (default: p5, p25, p50, p75,
    p95)."""

    num_simulations: int
    expected_return: float
    return_std: float
    return_percentiles: dict[str, float]
    expected_drawdown: float
    drawdown_std: float
    drawdown_percentiles: dict[str, float]
    probability_profit: float
    var_95: float
    summary: str


# -- compare_strategies -----------------------------------------------------


class StrategyComparisonRow(BaseModel):
    """From `BacktestAnalyzer.compare_strategies`."""

    strategy: str
    parameters: dict[str, Any]
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    grade: str
    rank: int


class StrategyComparisonResult(BaseModel):
    """Return shape of the `compare_strategies` tool."""

    rankings: list[StrategyComparisonRow]
    best_overall: StrategyComparisonRow | None
    best_return: StrategyComparisonRow
    best_sharpe: StrategyComparisonRow
    best_drawdown: StrategyComparisonRow
    best_win_rate: StrategyComparisonRow
    summary: str


# -- list_strategies ------------------------------------------------------


class StrategyCatalogEntry(BaseModel):
    """From `get_strategy_info`."""

    type: str
    name: str
    description: str
    default_parameters: dict[str, Any]
    optimization_ranges: dict[str, Any]


class StrategyCatalog(BaseModel):
    """Return shape of the `list_strategies` tool."""

    available_strategies: dict[str, StrategyCatalogEntry]
    total_count: int
    categories: dict[str, list[str]]


# -- backtest_portfolio -----------------------------------------------------


class PortfolioBacktestMetrics(BaseModel):
    symbols_tested: int
    total_return: float
    average_sharpe: float
    max_drawdown: float
    total_trades: int


class PortfolioBacktestResult(BaseModel):
    """Return shape of the `backtest_portfolio` tool."""

    portfolio_metrics: PortfolioBacktestMetrics
    individual_results: list[BacktestResult]
    summary: str


# -- ML-enhanced strategy tools ---------------------------------------------


class MLBacktestResult(BaseModel):
    """Return shape of the `run_ml_strategy_backtest` tool.

    Built from `BacktestAnalyzer.run_vectorbt_backtest`'s output plus
    `ml_metrics`. `trades` rows have no fixed shape in the legacy code (they
    vary between `{entry_time, exit_time, pnl, return}` and a
    `{total_trades, message}` fallback), so they are modeled as
    `dict[str, Any]`. `ml_metrics` content varies by ML strategy type
    (feature_importance / regime_analysis / strategy_weights are only added
    when the strategy object exposes them), so it is also `dict[str, Any]`.
    """

    metrics: SimpleBacktestMetrics
    trades: list[dict[str, Any]]
    equity_curve: dict[str, float]
    drawdown_series: dict[str, float]
    ml_metrics: dict[str, Any]


class MLTrainingResult(BaseModel):
    """Return shape of the `train_ml_predictor` tool."""

    symbol: str
    model_type: str
    training_period: str
    data_points: int
    target_periods: int
    return_threshold: float
    model_parameters: dict[str, Any]
    training_metrics: dict[str, Any]


class RegimeHistoryEntry(BaseModel):
    date: str
    regime: int
    probabilities: list[float]


class MarketRegimeAnalysis(BaseModel):
    """Return shape of the `analyze_market_regimes` tool."""

    symbol: str
    analysis_period: str
    method: str
    n_regimes: int
    regime_names: dict[int, str]
    current_regime: int
    regime_counts: dict[int, int]
    regime_percentages: dict[int, float]
    average_regime_durations: dict[int, float]
    recent_regime_history: list[RegimeHistoryEntry]
    total_regime_switches: int


class EnsembleMemberResult(BaseModel):
    """Per-symbol result nested in `create_strategy_ensemble`'s
    `individual_results`, built from the same
    `BacktestAnalyzer.run_vectorbt_backtest` shape as `MLBacktestResult` but
    with `ensemble_metrics` instead of `ml_metrics`."""

    metrics: SimpleBacktestMetrics
    trades: list[dict[str, Any]]
    equity_curve: dict[str, float]
    drawdown_series: dict[str, float]
    ensemble_metrics: dict[str, Any]


class EnsembleIndividualResult(BaseModel):
    symbol: str
    results: EnsembleMemberResult


class EnsembleSummary(BaseModel):
    symbols_tested: int
    base_strategies: list[str]
    weighting_method: str
    average_return: float
    total_trades: int
    average_trades_per_symbol: float


class EnsembleBacktestResult(BaseModel):
    """Return shape of the `create_strategy_ensemble` tool."""

    ensemble_summary: EnsembleSummary
    individual_results: list[EnsembleIndividualResult]
    final_strategy_weights: dict[str, Any]
    strategy_performance_analysis: dict[str, Any]
