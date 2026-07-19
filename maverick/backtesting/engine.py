"""Pure vectorbt backtest engine. Third layer: imports config and types.

Ports the alive core of `maverick_mcp/backtesting/vectorbt_engine.py`'s
`VectorBTEngine`: signal-driven portfolio simulation (`run_backtest`) and
grid-search parameter optimization (`optimize_parameters`), including its
`total_combos > 100` chunked-processing path (now driven by
`BacktestingSettings.optimization_chunk_threshold` /
`optimization_chunk_size_min` / `optimization_chunk_size_max` instead of
hardcoded literals). Everything fetch/cache/persistence/memory-profiling-
shaped -- `EnhancedStockDataProvider`, `CacheManager`, `CacheWarmer`,
`structured_logger`, `memory_profiler` -- is deliberately dropped: this
module's contract is pure compute, DataFrames and config values in, typed
results out.

Not ported: `run_memory_efficient_backtest` / `_run_chunked_backtest` /
`_combine_chunked_results`. That path chunks a *date range* and re-fetches
each chunk from the provider to bound peak memory during fetch -- it has no
well-defined meaning once fetching is out of scope, and reinterpreting it as
"slice the in-memory frame into date windows and run independent
sub-backtests" would silently change results (positions reset at each
chunk boundary) rather than port existing behavior. `self.chunker`
(`DataChunker`), which that path would have used, is dead code in the
legacy engine -- never referenced outside `__init__`. The one chunked
execution path that is both alive and pure is `optimize_parameters`'s
parameter-grid chunking, which is what's ported below.

Signal-generation seam
-----------------------
The legacy `_generate_signals` dispatch (13 inline `strategy_type`
branches, none of which delegate to `strategies/templates.py` -- it does
not exist yet in the legacy tree) is NOT ported here. `engine.py` sits in
the same import-linter layer as the future `strategies/templates.py`
(siblings under the "Backtesting layers are forward-only" contract in
`pyproject.toml`), so it must not depend on strategy template code Task 5
has not written yet. Instead:

- `run_backtest` takes already-computed `entries`/`exits` boolean Series --
  the caller (service layer, or a test) generates signals however it
  likes; this module only runs the vectorbt simulation and extracts
  metrics.
- `optimize_parameters` takes a `signal_fn` callable
  (`(frame, params) -> (entries, exits)`), invoked once per grid point.
  This is the signal-generation seam the parameter grid needs, expressed
  as a caller-supplied hook rather than a string-keyed dispatch table, so
  this module stays decoupled from `strategies/templates.py`.

Characterization tests hardcode a minimal sma-crossover `signal_fn`
locally (see `tests/backtesting/test_engine.py`) rather than importing
anything from `strategies/`, which remains a docstring-only stub reserved
for Task 5.
"""

import gc
import itertools
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pandas as pd
import vectorbt as vbt

from maverick.backtesting.config import BacktestingSettings, get_backtesting_settings
from maverick.backtesting.types import (
    BacktestMetrics,
    BacktestResult,
    OptimizationResult,
    OptimizationResultRow,
    TradeRecord,
)

SignalFn = Callable[[pd.DataFrame, dict[str, Any]], tuple[pd.Series, pd.Series]]

_MIN_ROWS = 2

# `vbt.Portfolio`'s metrics/trades accessors (`.sharpe_ratio`, `.trades`, ...)
# are attached via vectorbt's own caching decorators, which erase their
# signatures from static analysis -- every `portfolio` below is typed `Any`
# rather than `vbt.Portfolio` so real attribute access isn't flagged.
_METRIC_FUNCS: dict[str, Callable[[Any], Any]] = {
    "total_return": lambda p: p.total_return(),
    "sharpe_ratio": lambda p: p.sharpe_ratio(),
    "sortino_ratio": lambda p: p.sortino_ratio(),
    "calmar_ratio": lambda p: p.calmar_ratio(),
    # Higher is better for every other metric; negate drawdown (a negative
    # number) so "reverse=True" sorting still picks the smallest drawdown.
    "max_drawdown": lambda p: -p.max_drawdown(),
    "win_rate": lambda p: p.trades.win_rate() or 0,
    "profit_factor": lambda p: p.trades.profit_factor() or 0,
}


def _validate_frame(frame: pd.DataFrame) -> pd.Series:
    """Shared precondition check for `run_backtest`/`optimize_parameters`.

    Returns the `close` price Series once the frame passes validation.
    """
    if frame is None or frame.empty:
        raise ValueError("frame must be a non-empty DataFrame")
    if len(frame) < _MIN_ROWS:
        raise ValueError(
            f"frame must have at least {_MIN_ROWS} rows to backtest, got {len(frame)}"
        )
    if "close" in frame.columns:
        return frame["close"]
    if "Close" in frame.columns:
        return frame["Close"]
    raise ValueError(
        f"frame is missing a 'close' column; available columns: {list(frame.columns)}"
    )


def _format_date(value: Any) -> str:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError(f"invalid date value in frame index: {value!r}")
    return cast(pd.Timestamp, ts).strftime("%Y-%m-%d")


def _safe_float(value_fn: Callable[[], Any], default: float = 0.0) -> float:
    """Port of `VectorBTEngine._extract_metrics.safe_float_metric`: any
    exception, `None`, `NaN`, or `inf` collapses to `default` rather than
    propagating."""
    try:
        value = value_fn()
        if value is None or np.isnan(value) or np.isinf(value):
            return default
        return float(value)
    except (ZeroDivisionError, ValueError, TypeError):
        return default


def _kelly_criterion(portfolio: Any) -> float:
    """Port of `VectorBTEngine._calculate_kelly`."""
    if portfolio.trades.count() == 0:
        return 0.0
    try:
        win_rate = portfolio.trades.win_rate()
        if win_rate is None or np.isnan(win_rate):
            return 0.0

        winning = portfolio.trades.winning
        losing = portfolio.trades.losing
        avg_win = abs(winning.returns.mean() or 0) if winning.count() > 0 else 0
        avg_loss = abs(losing.returns.mean() or 0) if losing.count() > 0 else 0

        if avg_loss == 0 or avg_win == 0 or np.isnan(avg_win) or np.isnan(avg_loss):
            return 0.0

        with np.errstate(divide="ignore", invalid="ignore"):
            kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win

        if np.isnan(kelly) or np.isinf(kelly):
            return 0.0
        return float(min(max(kelly, -1.0), 0.25))  # capped -100%..25%
    except (ZeroDivisionError, ValueError, TypeError):
        return 0.0


def _recovery_factor(portfolio: Any) -> float:
    """Port of `VectorBTEngine._calculate_recovery_factor`."""
    try:
        max_dd = portfolio.max_drawdown()
        total_return = portfolio.total_return()
        if (
            max_dd is None
            or np.isnan(max_dd)
            or max_dd == 0
            or total_return is None
            or np.isnan(total_return)
        ):
            return 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            recovery = total_return / abs(max_dd)
        if np.isnan(recovery) or np.isinf(recovery):
            return 0.0
        return float(recovery)
    except (ZeroDivisionError, ValueError, TypeError):
        return 0.0


def _risk_reward_ratio(portfolio: Any) -> float:
    """Port of `VectorBTEngine._calculate_risk_reward`."""
    if portfolio.trades.count() == 0:
        return 0.0
    try:
        winning = portfolio.trades.winning
        losing = portfolio.trades.losing
        avg_win = abs(winning.pnl.mean() or 0) if winning.count() > 0 else 0
        avg_loss = abs(losing.pnl.mean() or 0) if losing.count() > 0 else 0
        if (
            avg_loss == 0
            or avg_win == 0
            or np.isnan(avg_win)
            or np.isnan(avg_loss)
            or np.isinf(avg_win)
            or np.isinf(avg_loss)
        ):
            return 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = avg_win / avg_loss
        if np.isnan(ratio) or np.isinf(ratio):
            return 0.0
        return float(ratio)
    except (ZeroDivisionError, ValueError, TypeError):
        return 0.0


def _extract_metrics(portfolio: Any) -> BacktestMetrics:
    """Port of `VectorBTEngine._extract_metrics`."""
    winning = portfolio.trades.winning
    losing = portfolio.trades.losing
    return BacktestMetrics(
        total_return=_safe_float(portfolio.total_return),
        annual_return=_safe_float(portfolio.annualized_return),
        sharpe_ratio=_safe_float(portfolio.sharpe_ratio),
        sortino_ratio=_safe_float(portfolio.sortino_ratio),
        calmar_ratio=_safe_float(portfolio.calmar_ratio),
        max_drawdown=_safe_float(portfolio.max_drawdown),
        win_rate=_safe_float(portfolio.trades.win_rate),
        profit_factor=_safe_float(portfolio.trades.profit_factor),
        expectancy=_safe_float(portfolio.trades.expectancy),
        total_trades=int(portfolio.trades.count()),
        winning_trades=int(winning.count()),
        losing_trades=int(losing.count()),
        avg_win=_safe_float(
            lambda: winning.pnl.mean() if winning.count() > 0 else None
        ),
        avg_loss=_safe_float(lambda: losing.pnl.mean() if losing.count() > 0 else None),
        best_trade=_safe_float(
            lambda: portfolio.trades.pnl.max() if portfolio.trades.count() > 0 else None
        ),
        worst_trade=_safe_float(
            lambda: portfolio.trades.pnl.min() if portfolio.trades.count() > 0 else None
        ),
        avg_duration=_safe_float(lambda: portfolio.trades.duration.mean()),
        kelly_criterion=_kelly_criterion(portfolio),
        recovery_factor=_recovery_factor(portfolio),
        risk_reward_ratio=_risk_reward_ratio(portfolio),
    )


def _extract_trades(portfolio: Any) -> list[TradeRecord]:
    """Port of `VectorBTEngine._extract_trades`.

    Built via `TradeRecord.model_validate(dict)` rather than keyword
    arguments: `TradeRecord.return_` is `Field(alias="return")`, and
    `return` is a reserved word, so the alias can only be supplied as a
    dict key (`{"return": ...}`), never as a literal `return=...` keyword.
    """
    if portfolio.trades.count() == 0:
        return []
    records = portfolio.trades.records_readable
    return [
        TradeRecord.model_validate(
            {
                "entry_date": str(row.get("Entry Timestamp", "")),
                "exit_date": str(row.get("Exit Timestamp", "")),
                "entry_price": float(row.get("Avg Entry Price", 0)),
                "exit_price": float(row.get("Avg Exit Price", 0)),
                "size": float(row.get("Size", 0)),
                "pnl": float(row.get("PnL", 0)),
                "return": float(row.get("Return", 0)),
                "duration": str(row.get("Duration", "")),
            }
        )
        for _, row in records.iterrows()
    ]


def run_backtest(
    frame: pd.DataFrame,
    entries: pd.Series,
    exits: pd.Series,
    *,
    symbol: str = "",
    strategy: str = "",
    parameters: dict[str, Any] | None = None,
    settings: BacktestingSettings | None = None,
) -> BacktestResult:
    """Run a single vectorized backtest over `frame` given `entries`/`exits`
    signals, and return the full-fidelity `BacktestResult`.

    Port of `VectorBTEngine.run_backtest` minus data fetching, caching,
    structured logging/memory-profiling instrumentation, and chart code.
    `symbol`/`strategy`/`parameters` are pure metadata carried onto the
    result; `start_date`/`end_date` are derived from `frame`'s index rather
    than accepted as separate arguments, since the frame is the only source
    of truth about what period was actually backtested.
    """
    settings = settings or get_backtesting_settings()
    close = _validate_frame(frame)
    if len(entries) != len(frame) or len(exits) != len(frame):
        raise ValueError("entries/exits length must match frame length")

    portfolio: Any = vbt.Portfolio.from_signals(
        close=close.astype(np.float32),
        entries=entries.astype(bool),
        exits=exits.astype(bool),
        init_cash=settings.initial_capital,
        fees=settings.fees,
        slippage=settings.slippage,
        freq="D",
        cash_sharing=False,
        call_seq="auto",
        group_by=False,
        broadcast_kwargs={"wrapper_kwargs": {"freq": "D"}},
    )

    equity_curve = {str(k): float(v) for k, v in portfolio.value().to_dict().items()}
    drawdown_series = {
        str(k): float(v) for k, v in portfolio.drawdown().to_dict().items()
    }

    return BacktestResult(
        symbol=symbol,
        strategy=strategy,
        parameters=parameters or {},
        metrics=_extract_metrics(portfolio),
        trades=_extract_trades(portfolio),
        equity_curve=equity_curve,
        drawdown_series=drawdown_series,
        start_date=_format_date(frame.index[0]),
        end_date=_format_date(frame.index[-1]),
        initial_capital=settings.initial_capital,
        memory_stats=None,
    )


def _get_metric_value(portfolio: Any, metric_name: str) -> float:
    """Port of `VectorBTEngine._get_metric_value`."""
    if metric_name not in _METRIC_FUNCS:
        raise ValueError(f"Unknown metric: {metric_name}")
    return _safe_float(lambda: _METRIC_FUNCS[metric_name](portfolio))


def optimize_parameters(
    frame: pd.DataFrame,
    signal_fn: SignalFn,
    param_grid: dict[str, list[Any]],
    *,
    symbol: str = "",
    strategy: str = "",
    optimization_metric: str = "sharpe_ratio",
    top_n: int = 10,
    settings: BacktestingSettings | None = None,
) -> OptimizationResult:
    """Grid-search `param_grid` by calling `signal_fn(frame, params)` for
    every combination and ranking by `optimization_metric`.

    Port of `VectorBTEngine.optimize_parameters`, unifying its two
    near-duplicate loops (plain vs. `_optimize_parameters_chunked`) into
    one: below `settings.optimization_chunk_threshold` combinations, the
    "chunk size" is simply every combination, which is exactly the
    unchunked behavior. Above the threshold, chunk size is
    `min(optimization_chunk_size_max, max(optimization_chunk_size_min,
    total_combos // 10))`, matching the legacy adaptive formula
    (`min(50, max(10, total_combos // 10))`) with the literals now sourced
    from `settings`.

    Legacy quirk preserved as-is: unlike `run_backtest`, the per-combination
    portfolio simulation here never applies slippage (the legacy grid loop
    omits the `slippage=` kwarg entirely) -- only `fees` (now
    `settings.fees` instead of a hardcoded `0.001`) is applied, trading
    fidelity for grid-search speed.
    """
    settings = settings or get_backtesting_settings()
    close = _validate_frame(frame)
    if optimization_metric not in _METRIC_FUNCS:
        raise ValueError(f"Unknown metric: {optimization_metric}")

    param_keys = list(param_grid.keys())
    param_combos = [
        dict(zip(param_keys, vals, strict=False))
        for vals in itertools.product(*param_grid.values())
    ]
    total_combos = len(param_combos)
    if total_combos == 0:
        raise ValueError("param_grid produced no parameter combinations")

    close_f32 = close.astype(np.float32)
    chunk_size = (
        min(
            settings.optimization_chunk_size_max,
            max(settings.optimization_chunk_size_min, total_combos // 10),
        )
        if total_combos > settings.optimization_chunk_threshold
        else total_combos
    )

    results: list[dict[str, Any]] = []
    for start in range(0, total_combos, chunk_size):
        for params in param_combos[start : start + chunk_size]:
            try:
                entries, exits = signal_fn(frame, params)
                portfolio: Any = vbt.Portfolio.from_signals(
                    close=close_f32,
                    entries=entries.astype(bool),
                    exits=exits.astype(bool),
                    init_cash=settings.initial_capital,
                    fees=settings.fees,
                    freq="D",
                    cash_sharing=False,
                    call_seq="auto",
                    group_by=False,
                )
            except Exception:  # noqa: BLE001 - skip invalid combos, faithful to legacy
                continue
            results.append(
                {
                    "parameters": params,
                    optimization_metric: _get_metric_value(
                        portfolio, optimization_metric
                    ),
                    "total_return": _safe_float(portfolio.total_return),
                    "max_drawdown": _safe_float(portfolio.max_drawdown),
                    "total_trades": int(portfolio.trades.count()),
                }
            )
        gc.collect()

    results.sort(key=lambda row: row[optimization_metric], reverse=True)
    top_rows = results[:top_n]
    top_results = [OptimizationResultRow(**row) for row in top_rows]

    return OptimizationResult(
        symbol=symbol,
        strategy=strategy,
        optimization_metric=optimization_metric,
        best_parameters=top_rows[0]["parameters"] if top_rows else {},
        best_metric_value=top_rows[0][optimization_metric] if top_rows else 0.0,
        top_results=top_results,
        total_combinations_tested=total_combos,
        valid_combinations=len(results),
        memory_stats=None,
    )
