"""Store-local pydantic input and read-back ("record") types.

## Types boundary

`save_backtest_result` (in `store.results`) accepts
`maverick.backtesting.types.BacktestResult` (imported there as
`BacktestResultPayload` to avoid colliding with this module's own SQL-row-
shaped `BacktestResultRecord`) plus its `list[TradeRecord]` -- both Task 2
types -- since that payload is a natural fit for the top-level result and
per-trade fields the legacy schema can represent unambiguously.

For `save_optimization_results`, `save_walk_forward_test`, and
`save_backtest_portfolio`, the corresponding Task 2 types
(`OptimizationResultRow`, `WalkForwardResult`/`WalkForwardPeriodResult`,
`PortfolioBacktestResult`) are *tool response* shapes, not persistence
shapes, and don't carry the fields the legacy tables require (e.g.
`OptimizationResultRow` has no `objective_value`/`rank`/statistical-
significance/`p_value` fields; `BacktestPortfolio`'s table has no Task 2
counterpart at all). Rather than force-fit those types or fall back to raw
dicts, this module defines narrow, purpose-built pydantic input models
(`OptimizationResultInput`, `WalkForwardTestInput`, `BacktestPortfolioInput`)
scoped to exactly what each table needs, with required fields matching the
legacy columns' `nullable=False` constraints -- so a missing required field
is a pydantic `ValidationError` at construction time, not a silently
written `NULL`.

## Fields left NULL rather than guessed

`BacktestMetrics` (Task 2's metrics payload, derived from the *current*
`VectorBTEngine._extract_metrics`) does not carry every field the legacy
`mcp_backtest_results` columns expect (`annualized_return`,
`max_drawdown_duration`, `volatility`, `downside_volatility`,
`average_win`, `average_loss`, `largest_win`, `largest_loss`,
`final_portfolio_value`, `peak_portfolio_value`, `beta`, `alpha`). Some of
these look like plausible renames of Task 2 fields (e.g. `annual_return` ->
`annualized_return`, `best_trade`/`worst_trade` -> `largest_win`/
`largest_loss`), but `engine.py` (the module that would confirm whether
those really are the same computation) is still docstring-only pending
Task 5 -- guessing a mapping now risks writing silently wrong data, which
is worse than a NULL in a nullable column. `BacktestResultRecord` (below)
therefore has no fields for these; `store.results.save_backtest_result`
leaves the corresponding columns unpopulated, and a future task with the
real engine in hand can revisit and fill them in with a verified mapping.

## Trade field gaps

`TradeRecord` (Task 2's per-trade payload, from `VectorBTEngine._extract_trades`)
has no `direction` field, but `mcp_backtest_trades.direction` is
`nullable=False`. The legacy `_save_trades` already defaults a missing
`direction` to `"long"` (`trade.get("direction", "long")`); `store.results`
applies the same literal default rather than inventing new behavior.
Likewise `TradeRecord` has no `mae`/`mfe`/`duration_days`/`duration_hours`/
`exit_reason`/`fees_paid`/`slippage_cost`/`entry_time`/`exit_time` -- all
nullable columns, left `NULL`, and so absent from `BacktestTradeRecord`
below. `TradeRecord.return_` (the `"return"` field) is written into the
legacy `pnl_percent` column as the closest match; its exact
fraction-vs-percentage semantics aren't independently verified for the same
reason noted above (engine.py not yet ported).
"""

import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict

# -- Input types (see "Types boundary" above) --------------------------------


class OptimizationResultInput(BaseModel):
    """One row to save via `store.optimization.save_optimization_results`.

    Distinct from `types.OptimizationResultRow` (the `optimize_strategy`
    tool's *response* row shape) -- that type only carries `parameters`,
    `total_return`, `max_drawdown`, `total_trades`, plus one dynamic extra
    key, and has no `objective_value`/`sharpe_ratio`/`win_rate`/
    `profit_factor`/`rank`/statistical-significance/`p_value` fields, all of
    which `mcp_optimization_results` has columns for.
    """

    parameters: dict[str, Any]
    objective_value: float | None = None
    total_return: float | None = None
    sharpe_ratio: float | None = None
    max_drawdown: float | None = None
    win_rate: float | None = None
    profit_factor: float | None = None
    total_trades: int | None = None
    rank: int | None = None
    is_statistically_significant: bool = False
    p_value: float | None = None


class WalkForwardTestInput(BaseModel):
    """The row to save via `store.walk_forward.save_walk_forward_test`.

    Distinct from `types.WalkForwardResult`/`WalkForwardPeriodResult` (the
    `walk_forward_analysis` tool's response shape, keyed by a `period: str`
    label) -- the legacy table instead stores explicit training/test date
    ranges and window/step sizes in months, which the tool response doesn't
    carry.
    """

    window_size_months: int
    step_size_months: int
    training_start: date
    training_end: date
    test_period_start: date
    test_period_end: date
    optimal_parameters: dict[str, Any] | None = None
    training_performance: float | None = None
    out_of_sample_return: float | None = None
    out_of_sample_sharpe: float | None = None
    out_of_sample_drawdown: float | None = None
    out_of_sample_trades: int | None = None
    performance_ratio: float | None = None
    degradation_factor: float | None = None
    is_profitable: bool | None = None
    is_statistically_significant: bool = False


class BacktestPortfolioInput(BaseModel):
    """The row to save via `store.portfolios.save_backtest_portfolio`.

    No Task 2 type models this shape -- `types.PortfolioBacktestResult` (the
    `backtest_portfolio` tool's response) is aggregate metrics plus a list
    of per-symbol `BacktestResult`s, not the composition/config fields
    (`symbols`, `weights`, `rebalance_frequency`, position sizing, risk
    limits) `mcp_backtest_portfolios` stores.
    """

    portfolio_name: str
    start_date: date
    end_date: date
    symbols: list[str]
    description: str | None = None
    weights: dict[str, float] | None = None
    rebalance_frequency: str | None = None
    initial_capital: float = 100000.0
    max_positions: int | None = None
    position_sizing_method: str | None = None
    portfolio_stop_loss: float | None = None
    max_sector_allocation: float | None = None
    correlation_threshold: float | None = None
    total_return: float | None = None
    annualized_return: float | None = None
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    max_drawdown: float | None = None
    volatility: float | None = None
    diversification_ratio: float | None = None
    concentration_index: float | None = None
    turnover_rate: float | None = None
    component_backtest_ids: list[str] | None = None
    portfolio_equity_curve: dict[str, float] | None = None
    portfolio_weights_history: dict[str, Any] | None = None
    status: str = "completed"
    notes: str | None = None


# -- Record types (read-back shapes) -----------------------------------------


class BacktestResultRecord(BaseModel):
    """A row read back from `mcp_backtest_results`."""

    model_config = ConfigDict(frozen=True)

    backtest_id: uuid.UUID
    symbol: str
    strategy_type: str
    backtest_date: datetime
    start_date: date
    end_date: date
    initial_capital: Decimal | None
    parameters: dict[str, Any] | None
    total_return: Decimal | None
    annualized_return: Decimal | None
    sharpe_ratio: Decimal | None
    sortino_ratio: Decimal | None
    calmar_ratio: Decimal | None
    max_drawdown: Decimal | None
    total_trades: int | None
    winning_trades: int | None
    losing_trades: int | None
    win_rate: Decimal | None
    profit_factor: Decimal | None
    equity_curve: dict[str, float] | None
    drawdown_series: dict[str, float] | None
    execution_time_seconds: Decimal | None
    data_points: int | None
    status: str | None
    notes: str | None


class BacktestTradeRecord(BaseModel):
    """A row read back from `mcp_backtest_trades`."""

    model_config = ConfigDict(frozen=True)

    trade_id: uuid.UUID
    backtest_id: uuid.UUID
    trade_number: int
    entry_date: date
    entry_price: Decimal
    exit_date: date | None
    exit_price: Decimal | None
    position_size: Decimal | None
    direction: str
    pnl: Decimal | None
    pnl_percent: Decimal | None


class OptimizationResultRecord(BaseModel):
    """A row read back from `mcp_optimization_results`."""

    model_config = ConfigDict(frozen=True)

    optimization_id: uuid.UUID
    backtest_id: uuid.UUID
    parameter_set: int
    parameters: dict[str, Any]
    objective_function: str | None
    objective_value: Decimal | None
    total_return: Decimal | None
    sharpe_ratio: Decimal | None
    max_drawdown: Decimal | None
    win_rate: Decimal | None
    profit_factor: Decimal | None
    total_trades: int | None
    rank: int | None
    is_statistically_significant: bool
    p_value: Decimal | None


class WalkForwardTestRecord(BaseModel):
    """A row read back from `mcp_walk_forward_tests`."""

    model_config = ConfigDict(frozen=True)

    walk_forward_id: uuid.UUID
    parent_backtest_id: uuid.UUID
    window_size_months: int
    step_size_months: int
    training_start: date
    training_end: date
    test_period_start: date
    test_period_end: date
    optimal_parameters: dict[str, Any] | None
    training_performance: Decimal | None
    out_of_sample_return: Decimal | None
    out_of_sample_sharpe: Decimal | None
    out_of_sample_drawdown: Decimal | None
    out_of_sample_trades: int | None
    performance_ratio: Decimal | None
    degradation_factor: Decimal | None
    is_profitable: bool | None
    is_statistically_significant: bool


class BacktestPortfolioRecord(BaseModel):
    """A row read back from `mcp_backtest_portfolios`."""

    model_config = ConfigDict(frozen=True)

    portfolio_backtest_id: uuid.UUID
    portfolio_name: str
    description: str | None
    backtest_date: datetime
    start_date: date
    end_date: date
    symbols: list[str]
    weights: dict[str, float] | None
    rebalance_frequency: str | None
    initial_capital: Decimal | None
    total_return: Decimal | None
    sharpe_ratio: Decimal | None
    max_drawdown: Decimal | None
    status: str | None
    notes: str | None
