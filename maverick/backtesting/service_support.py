"""Shared pure helpers for `service.py`/`service_ml.py`. Fourth layer: imports engine,
analysis, optimization, strategies, config, and types -- same tier as `service.py`, just without
a `MarketDataService` dependency (nothing here fetches).

Split out purely to break an import cycle: `service.py` defines `BacktestingService`, which
inherits `_ExtendedBacktestingMixin` from `service_ml.py` (to keep both files under the repo's
500-line-per-file cap); both need these functions, so they live in a third file neither of those
two needs to import from the other for. `SimpleMovingAverageStrategy` (also used by both) lives
here for the same reason -- see `service_ml.py`'s module docstring for what it is and why it's
net-new.
"""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import date, timedelta
from typing import Any

import pandas as pd

from maverick.backtesting import engine
from maverick.backtesting.strategies import signals as signal_dispatch
from maverick.backtesting.strategies import templates
from maverick.backtesting.strategies.base import Strategy
from maverick.backtesting.types import (
    BacktestMetrics,
    BacktestResult,
    SimpleBacktestMetrics,
)


class SimpleMovingAverageStrategy(Strategy):
    """Tools-tier SMA-crossover `Strategy` shim for the ML orchestration methods in
    `service_ml.py`. See that module's docstring for why this is net-new."""

    def __init__(self, parameters: dict[str, Any] | None = None) -> None:
        super().__init__(parameters or {"fast_period": 10, "slow_period": 20})

    @property
    def name(self) -> str:
        return "SMA Crossover"

    @property
    def description(self) -> str:
        return "Simple moving average crossover strategy"

    def generate_signals(self, data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        return signal_dispatch.generate_signals(data, "sma_cross", self.parameters)


# Matches legacy `strategy_executor.py`'s `max_concurrent_strategies: int = 6` /
# `batch_processing.py`'s `run_parallel_backtests(max_workers: int = 6)` -- the discoverable
# "legacy effective parallelism" for `compare_strategies`/`backtest_portfolio`'s bounded
# concurrency (see `service.py`'s module docstring).
BATCH_CONCURRENCY = 6

# `StrategyOptimizer.walk_forward_analysis`'s hardcoded optimization-window constant (not exposed
# as a tool parameter in legacy either).
WALK_FORWARD_OPTIMIZATION_WINDOW_DAYS = 504


def resolve_dates(
    start_date: str | None, end_date: str | None, *, default_days: int
) -> tuple[date, date]:
    """Resolve `[start, end]` using each tool's own legacy default-lookback in `default_days`."""
    end = date.fromisoformat(end_date) if end_date else date.today()
    start = (
        date.fromisoformat(start_date)
        if start_date
        else end - timedelta(days=default_days)
    )
    return start, end


def merge_parameters(strategy: str, overrides: dict[str, Any]) -> dict[str, Any]:
    """`dict(STRATEGY_TEMPLATES[strategy]["parameters"])` (or `{}` for an unknown strategy)
    overridden by every non-`None` entry in `overrides` -- reproduces every legacy tool's
    "if strategy in STRATEGY_TEMPLATES: ... else: only provided params" two-branch parameter
    resolution."""
    if strategy in templates.STRATEGY_TEMPLATES:
        parameters = dict(templates.STRATEGY_TEMPLATES[strategy]["parameters"])
    else:
        parameters = {}
    for key, value in overrides.items():
        if value is not None:
            parameters[key] = value
    return parameters


def to_simple_metrics(metrics: BacktestMetrics) -> SimpleBacktestMetrics:
    """Project `SimpleBacktestMetrics`'s 7 fields straight off the full `BacktestMetrics` --
    a strict subset, see `types.py`'s module docstring."""
    return SimpleBacktestMetrics(
        total_return=metrics.total_return,
        annual_return=metrics.annual_return,
        sharpe_ratio=metrics.sharpe_ratio,
        max_drawdown=metrics.max_drawdown,
        win_rate=metrics.win_rate,
        total_trades=metrics.total_trades,
        profit_factor=metrics.profit_factor,
    )


def signal_fn_for(strategy: str) -> "engine.SignalFn":
    """Closure matching `engine.SignalFn`'s `(frame, params) -> (entries, exits)` shape, bound to
    `strategy`. Not `functools.partial(generate_signals, strategy_type=strategy)`: that collides
    positionally with `generate_signals(frame, strategy_type, params)`'s signature and raises
    "multiple values for argument 'strategy_type'"."""

    def _fn(frame, params: dict[str, Any]):
        return signal_dispatch.generate_signals(frame, strategy, params)

    return _fn


async def gather_bounded(
    items: list[str], run_one: Callable[[str], Awaitable[BacktestResult]]
) -> list[BacktestResult]:
    """Run `run_one(item)` for every item under a shared `BATCH_CONCURRENCY`-bounded semaphore,
    skipping any item whose call raises (mirrors legacy's per-item `except Exception: continue`).
    Shared by `compare_strategies`/`backtest_portfolio` -- see `service.py`'s module docstring's
    "Concurrency" note; `create_strategy_ensemble` deliberately does not use this (stays
    sequential)."""
    sem = asyncio.Semaphore(BATCH_CONCURRENCY)

    async def _guarded(item: str) -> BacktestResult | None:
        try:
            async with sem:
                return await run_one(item)
        except Exception:
            return None

    raw = await asyncio.gather(*(_guarded(item) for item in items))
    return [r for r in raw if r is not None]


def generate_wf_summary(
    avg_return: float, avg_sharpe: float, consistency: float
) -> str:
    """Port of `StrategyOptimizer._generate_wf_summary`."""
    summary = f"Walk-forward analysis shows {avg_return * 100:.1f}% average return "
    summary += f"with Sharpe ratio of {avg_sharpe:.2f}. "
    summary += f"Strategy was profitable in {consistency * 100:.0f}% of periods. "

    if avg_sharpe >= 1.0 and consistency >= 0.7:
        summary += (
            "Results indicate robust performance across different market conditions."
        )
    elif avg_sharpe >= 0.5 and consistency >= 0.5:
        summary += "Results show moderate robustness with room for improvement."
    else:
        summary += "Results suggest the strategy may not be robust to changing market conditions."
    return summary
