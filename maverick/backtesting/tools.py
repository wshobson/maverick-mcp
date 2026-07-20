"""MCP tool registrations for backtesting. Top layer: imports service and below.

**The phase's availability contract.** `service.py` (transitively, via `engine`/`analysis`/
`strategies.signals`/`strategies.ml.*`) imports `vectorbt`/`sklearn` at module top level. This
module -- and `tools_support.py`/`tools_ml.py`, which it composes -- must import cleanly on a
base install with zero backtesting extras, so none of the three ever imports
`maverick.backtesting.service` (or anything that imports it) at module level; `BacktestingService`
is referenced only under `TYPE_CHECKING` (`from __future__ import annotations` keeps the
`configure(service: BacktestingService)` hint a lazy string). `register()` probes for the extra
via `tools_support.backtesting_extra_available()` (re-exported here as `_backtesting_extra_available`
so tests can `monkeypatch.setattr(tools, "_backtesting_extra_available", ...)` to simulate the
extra's absence without needing to actually uninstall vectorbt) and registers zero tools with one
clear warning log if it's missing -- the base install must boot with no traceback.

**readOnlyHint.** All 11 tools are marked `readOnlyHint=True`: grepping the legacy
`maverick_mcp/api/routers/backtesting.py` for every surviving tool finds zero persistence call
sites (see `service.py`'s module docstring), so none of these tools mutate server-side state --
they only fetch market data and compute.

Tool functions 8-11 (the ML-strategy tools) live in `tools_ml.py`, split out to keep this file
under the repo's 500-line-per-file cap; `_READ_ONLY_TOOLS` below still registers all 11.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fastmcp import FastMCP

from maverick.backtesting.tools_ml import (
    backtesting_analyze_market_regimes,
    backtesting_create_strategy_ensemble,
    backtesting_run_ml_strategy_backtest,
    backtesting_train_ml_predictor,
)
from maverick.backtesting.tools_support import (
    READ_ONLY_ANNOTATIONS,
    configure,
)
from maverick.backtesting.tools_support import (
    backtesting_extra_available as _backtesting_extra_available,
)
from maverick.backtesting.tools_support import (
    require_service as _require_service,
)

if TYPE_CHECKING:
    from maverick.backtesting.service import BacktestingService  # noqa: F401

__all__ = ["configure", "register"]

logger = logging.getLogger(__name__)


async def backtesting_run_backtest(
    symbol: str,
    strategy: str = "sma_cross",
    start_date: str | None = None,
    end_date: str | None = None,
    initial_capital: float = 10000.0,
    fast_period: int | None = None,
    slow_period: int | None = None,
    period: int | None = None,
    oversold: float | None = None,
    overbought: float | None = None,
    signal_period: int | None = None,
    std_dev: float | None = None,
    lookback: int | None = None,
    threshold: float | None = None,
    z_score_threshold: float | None = None,
    breakout_factor: float | None = None,
) -> dict[str, Any]:
    """Run a single-strategy backtest and return metrics, trades, and analysis."""
    try:
        service = _require_service()
        result = await service.run_backtest(
            symbol,
            strategy=strategy,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            fast_period=fast_period,
            slow_period=slow_period,
            period=period,
            oversold=oversold,
            overbought=overbought,
            signal_period=signal_period,
            std_dev=std_dev,
            lookback=lookback,
            threshold=threshold,
            z_score_threshold=z_score_threshold,
            breakout_factor=breakout_factor,
        )
        payload = result.model_dump(mode="json")
        payload["status"] = "success"
        return payload
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


async def backtesting_optimize_strategy(
    symbol: str,
    strategy: str = "sma_cross",
    start_date: str | None = None,
    end_date: str | None = None,
    optimization_metric: str = "sharpe_ratio",
    optimization_level: str = "medium",
    top_n: int = 10,
) -> dict[str, Any]:
    """Grid-search a strategy's parameters (sma_cross, rsi, macd, bollinger, or momentum only)."""
    try:
        service = _require_service()
        result = await service.optimize_strategy(
            symbol,
            strategy=strategy,
            start_date=start_date,
            end_date=end_date,
            optimization_metric=optimization_metric,
            optimization_level=optimization_level,
            top_n=top_n,
        )
        payload = result.model_dump(mode="json")
        payload["status"] = "success"
        return payload
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


async def backtesting_walk_forward_analysis(
    symbol: str,
    strategy: str = "sma_cross",
    start_date: str | None = None,
    end_date: str | None = None,
    window_size: int = 252,
    step_size: int = 63,
) -> dict[str, Any]:
    """Roll a strategy forward through repeated optimize/test windows to gauge robustness."""
    try:
        service = _require_service()
        result = await service.walk_forward_analysis(
            symbol,
            strategy=strategy,
            start_date=start_date,
            end_date=end_date,
            window_size=window_size,
            step_size=step_size,
        )
        payload = result.model_dump(mode="json")
        payload["status"] = "success"
        return payload
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


async def backtesting_monte_carlo_simulation(
    symbol: str,
    strategy: str = "sma_cross",
    start_date: str | None = None,
    end_date: str | None = None,
    num_simulations: int = 1000,
    fast_period: int | None = None,
    slow_period: int | None = None,
    period: int | None = None,
) -> dict[str, Any]:
    """Bootstrap-resample a backtest's trades to estimate a return/drawdown distribution."""
    try:
        service = _require_service()
        result = await service.monte_carlo_simulation(
            symbol,
            strategy=strategy,
            start_date=start_date,
            end_date=end_date,
            num_simulations=num_simulations,
            fast_period=fast_period,
            slow_period=slow_period,
            period=period,
        )
        payload = result.model_dump(mode="json")
        payload["status"] = "success"
        return payload
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


async def backtesting_compare_strategies(
    symbol: str,
    strategies: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Backtest multiple strategies on the same symbol and rank them."""
    try:
        service = _require_service()
        result = await service.compare_strategies(
            symbol, strategies=strategies, start_date=start_date, end_date=end_date
        )
        payload = result.model_dump(mode="json")
        payload["status"] = "success"
        return payload
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


async def backtesting_list_strategies() -> dict[str, Any]:
    """List every available rule-based strategy template with its default parameters."""
    try:
        service = _require_service()
        result = await service.list_strategies()
        payload = result.model_dump(mode="json")
        payload["status"] = "success"
        return payload
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


async def backtesting_backtest_portfolio(
    symbols: list[str],
    strategy: str = "sma_cross",
    start_date: str | None = None,
    end_date: str | None = None,
    initial_capital: float = 10000.0,
    position_size: float = 0.1,
    fast_period: int | None = None,
    slow_period: int | None = None,
    period: int | None = None,
) -> dict[str, Any]:
    """Backtest one strategy across multiple symbols and aggregate portfolio-level metrics."""
    try:
        service = _require_service()
        result = await service.backtest_portfolio(
            symbols,
            strategy=strategy,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            position_size=position_size,
            fast_period=fast_period,
            slow_period=slow_period,
            period=period,
        )
        payload = result.model_dump(mode="json")
        payload["status"] = "success"
        return payload
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


_READ_ONLY_TOOLS = (
    backtesting_run_backtest,
    backtesting_optimize_strategy,
    backtesting_walk_forward_analysis,
    backtesting_monte_carlo_simulation,
    backtesting_compare_strategies,
    backtesting_list_strategies,
    backtesting_backtest_portfolio,
    backtesting_run_ml_strategy_backtest,
    backtesting_train_ml_predictor,
    backtesting_analyze_market_regimes,
    backtesting_create_strategy_ensemble,
)  # all 11: nothing persists (see module docstring), so nothing is non-read-only.


def register(mcp: FastMCP) -> None:
    """Register all 11 `backtesting_*` tools, or zero of them with one clear warning log if the
    `[backtesting]` extra isn't installed -- the phase's availability contract. Never raises
    either way."""
    if not _backtesting_extra_available():
        logger.warning(
            "backtesting.tools: the '[backtesting]' extra is not installed (vectorbt missing); "
            "registering zero backtesting tools. Install with `uv sync --extra backtesting` to "
            "enable them."
        )
        return
    for fn in _READ_ONLY_TOOLS:
        mcp.tool(name=fn.__name__, annotations=READ_ONLY_ANNOTATIONS)(fn)
