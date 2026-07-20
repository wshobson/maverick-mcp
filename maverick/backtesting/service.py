"""Backtesting business logic. Fourth layer: imports store, engine, optimization, analysis,
strategies, config, and types.

`BacktestingService(market_data, settings=None)` backs all 11 `backtesting_*` MCP tools with
async methods. It owns price-history fetch via the injected `MarketDataService`; everything below
`service`/`tools` in the layer contract (`engine`, `analysis`, `optimization`,
`strategies.signals`/`strategies.templates`/`strategies.ml.*`) is pure compute, never fetching.
Methods 3 (`walk_forward_analysis`) and 8-11 (the ML-strategy tools) live on
`_ExtendedBacktestingMixin` in `service_ml.py`, and shared pure helpers live in
`service_support.py` -- both split out to keep every file under the repo's 500-line-per-file cap
while `BacktestingService` (this class) remains the one place all 11 methods are callable from.

Design notes (see the exec plan / task report for full rationale):

- **No persistence.** `maverick.backtesting.store` is not imported here: grepping the legacy
  `maverick_mcp/api/routers/backtesting.py` for all 11 surviving tools finds zero call sites into
  `BacktestPersistenceManager`/`save_*`. Corroborated by this task's frozen constructor signature
  `BacktestingService(market_data, settings=None)` -- no `engine`/session-factory parameter,
  unlike the persisting `PortfolioService`/`ScreeningService`, which both take an `Engine`.
- **Column casing.** `MarketDataService.get_price_history` returns yfinance-cased columns
  (`Open`/`High`/`Low`/`Close`/`Volume`), but `strategies.signals.generate_signals` requires
  lowercase `"close"`/`"volume"` with no fallback. `_fetch_frame` lowercases on fetch, mirroring
  legacy `VectorBTEngine.get_historical_data`'s own `data.columns = [c.lower() for c in ...]`.
  It also raises a clear `ValueError` for an empty fetch before any engine/strategy call.
- **Date-range semantics.** Each method preserves its own legacy tool's exact default-lookback
  (365/730/1095 days) and fetches exactly `[start, end]`, no warmup padding -- legacy never
  padded either.
- **Concurrency.** `compare_strategies`/`backtest_portfolio` bound their per-item backtests with
  `service_support.gather_bounded` (a semaphore of 6, matching legacy `strategy_executor.py`'s
  `max_concurrent_strategies: int = 6` / `batch_processing.py`'s `max_workers: int = 6` -- the
  discoverable "legacy effective parallelism"), merging the legacy `strategy_executor.py`/
  `batch_processing.py` duplication. `create_strategy_ensemble` stays sequential on purpose: it
  shares one mutable `StrategyEnsemble` instance across symbols (weights mutate per call), so
  concurrency there would make results order-dependent -- a correctness hazard, not style.
- **Parameter-override simplification (disclosed).** Legacy accepts override params
  (`fast_period` etc.) as `str | int | None` with a hand-rolled string-coercion helper --
  MCP-client-compat cruft no other ported domain uses. This port uses plain `int | None`/
  `float | None`. Same for `compare_strategies`'s `strategies`: plain `list[str] | None`, not
  legacy's JSON-string fallback.
- **readOnlyHint.** Because nothing persists, `tools.py` marks all 11 tools `readOnlyHint=True`.
"""

import asyncio
from datetime import date
from typing import Any

import pandas as pd

from maverick.backtesting import analysis, engine, optimization
from maverick.backtesting.config import BacktestingSettings, get_backtesting_settings
from maverick.backtesting.service_ml import _ExtendedBacktestingMixin
from maverick.backtesting.service_support import (
    gather_bounded,
    merge_parameters,
    resolve_dates,
    signal_fn_for,
)
from maverick.backtesting.strategies import signals as signal_dispatch
from maverick.backtesting.strategies import templates
from maverick.backtesting.types import (
    BacktestResult,
    MonteCarloResult,
    OptimizationResult,
    PortfolioBacktestMetrics,
    PortfolioBacktestResult,
    RunBacktestResult,
    StrategyCatalog,
    StrategyComparisonResult,
)
from maverick.market_data.service import MarketDataService


class BacktestingService(_ExtendedBacktestingMixin):
    """Domain service: fetches price history via the injected `MarketDataService`, generates
    signals via `strategies.signals`, and runs the pure `engine`/`analysis`/`optimization`
    functions. See module docstring for the `service_ml.py`/`service_support.py` split.
    """

    def __init__(
        self,
        market_data: MarketDataService,
        settings: BacktestingSettings | None = None,
    ) -> None:
        self._market_data = market_data
        self._settings = settings or get_backtesting_settings()

    @property
    def settings(self) -> BacktestingSettings:
        return self._settings

    async def _run(self, coro: Any) -> Any:
        """Apply `settings.analysis_timeout_seconds` to `coro`, translating a timeout into the
        same clear `ValueError` shape every other service failure uses. Mirrors
        `TechnicalService._run` exactly."""
        try:
            return await asyncio.wait_for(
                coro, timeout=self._settings.analysis_timeout_seconds
            )
        except TimeoutError as exc:
            raise ValueError(
                "Backtesting analysis timed out after "
                f"{self._settings.analysis_timeout_seconds}s"
            ) from exc

    async def _fetch_frame(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Fetch and lowercase-normalize a symbol's OHLCV frame (see module docstring's "Column
        casing" note). Raises a clear `ValueError` for an empty fetch before any engine/strategy
        call ever sees the frame."""
        frame = await self._market_data.get_price_history(symbol, start, end)
        if frame is None or frame.empty:
            raise ValueError(
                f"No price history available for {symbol!r} between {start} and {end}"
            )
        return frame.rename(columns=str.lower)

    async def _run_single_backtest(
        self,
        symbol: str,
        strategy: str,
        start: date,
        end: date,
        *,
        initial_capital: float,
        parameters: dict[str, Any] | None = None,
    ) -> BacktestResult:
        """Shared single-symbol/single-strategy fetch -> signal -> backtest path used by
        `run_backtest`, `monte_carlo_simulation`, `compare_strategies`, and `backtest_portfolio`."""
        frame = await self._fetch_frame(symbol, start, end)
        params = (
            parameters if parameters is not None else merge_parameters(strategy, {})
        )
        entries, exits = signal_dispatch.generate_signals(frame, strategy, params)
        settings = (
            self._settings
            if initial_capital == self._settings.initial_capital
            else self._settings.model_copy(update={"initial_capital": initial_capital})
        )
        return engine.run_backtest(
            frame,
            entries,
            exits,
            symbol=symbol,
            strategy=strategy,
            parameters=params,
            settings=settings,
        )

    # -- 1. run_backtest ----------------------------------------------------

    async def run_backtest(
        self,
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
    ) -> RunBacktestResult:
        async def _impl() -> RunBacktestResult:
            start, end = resolve_dates(start_date, end_date, default_days=365)
            overrides = {
                "fast_period": fast_period,
                "slow_period": slow_period,
                "period": period,
                "oversold": oversold,
                "overbought": overbought,
                "signal_period": signal_period,
                "std_dev": std_dev,
                "lookback": lookback,
                "threshold": threshold,
                "z_score_threshold": z_score_threshold,
                "breakout_factor": breakout_factor,
            }
            parameters = merge_parameters(strategy, overrides)
            result = await self._run_single_backtest(
                symbol,
                strategy,
                start,
                end,
                initial_capital=initial_capital,
                parameters=parameters,
            )
            return RunBacktestResult(
                **result.model_dump(), analysis=analysis.analyze(result)
            )

        return await self._run(_impl())

    # -- 2. optimize_strategy ------------------------------------------------

    async def optimize_strategy(
        self,
        symbol: str,
        strategy: str = "sma_cross",
        start_date: str | None = None,
        end_date: str | None = None,
        optimization_metric: str = "sharpe_ratio",
        optimization_level: str = "medium",
        top_n: int = 10,
    ) -> OptimizationResult:
        async def _impl() -> OptimizationResult:
            start, end = resolve_dates(start_date, end_date, default_days=730)
            frame = await self._fetch_frame(symbol, start, end)
            grid = optimization.generate_param_grid(strategy, optimization_level)
            return engine.optimize_parameters(
                frame,
                signal_fn_for(strategy),
                grid,
                symbol=symbol,
                strategy=strategy,
                optimization_metric=optimization_metric,
                top_n=top_n,
                settings=self._settings,
            )

        return await self._run(_impl())

    # -- 4. monte_carlo_simulation --------------------------------------------

    async def monte_carlo_simulation(
        self,
        symbol: str,
        strategy: str = "sma_cross",
        start_date: str | None = None,
        end_date: str | None = None,
        num_simulations: int = 1000,
        fast_period: int | None = None,
        slow_period: int | None = None,
        period: int | None = None,
    ) -> MonteCarloResult:
        async def _impl() -> MonteCarloResult:
            start, end = resolve_dates(start_date, end_date, default_days=365)
            overrides = {
                "fast_period": fast_period,
                "slow_period": slow_period,
                "period": period,
            }
            parameters = merge_parameters(strategy, overrides)
            result = await self._run_single_backtest(
                symbol,
                strategy,
                start,
                end,
                initial_capital=self._settings.initial_capital,
                parameters=parameters,
            )
            return analysis.monte_carlo_simulation(
                result.trades, num_simulations=num_simulations
            )

        return await self._run(_impl())

    # -- 5. compare_strategies -------------------------------------------------

    async def compare_strategies(
        self,
        symbol: str,
        strategies: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> StrategyComparisonResult:
        async def _impl() -> StrategyComparisonResult:
            strategy_list = strategies or [
                "sma_cross",
                "rsi",
                "macd",
                "bollinger",
                "momentum",
            ]
            start, end = resolve_dates(start_date, end_date, default_days=365)

            async def _one(strategy_name: str) -> BacktestResult:
                parameters = merge_parameters(strategy_name, {})
                return await self._run_single_backtest(
                    symbol,
                    strategy_name,
                    start,
                    end,
                    initial_capital=self._settings.initial_capital,
                    parameters=parameters,
                )

            results = await gather_bounded(strategy_list, _one)
            return analysis.compare_strategies(results)

        return await self._run(_impl())

    # -- 6. list_strategies ------------------------------------------------

    async def list_strategies(self) -> StrategyCatalog:
        async def _impl() -> StrategyCatalog:
            strategy_infos = {
                t: templates.get_strategy_info(t)
                for t in templates.list_available_strategies()
            }
            return StrategyCatalog(
                available_strategies=strategy_infos,
                total_count=len(strategy_infos),
                categories={
                    "trend_following": ["sma_cross", "ema_cross", "macd", "breakout"],
                    "mean_reversion": ["rsi", "bollinger", "mean_reversion"],
                    "momentum": ["momentum", "volume_momentum"],
                },
            )

        return await self._run(_impl())

    # -- 7. backtest_portfolio -------------------------------------------------

    async def backtest_portfolio(
        self,
        symbols: list[str],
        strategy: str = "sma_cross",
        start_date: str | None = None,
        end_date: str | None = None,
        initial_capital: float = 10000.0,
        position_size: float = 0.1,
        fast_period: int | None = None,
        slow_period: int | None = None,
        period: int | None = None,
    ) -> PortfolioBacktestResult:
        async def _impl() -> PortfolioBacktestResult:
            start, end = resolve_dates(start_date, end_date, default_days=365)
            overrides = {
                "fast_period": fast_period,
                "slow_period": slow_period,
                "period": period,
            }
            parameters = merge_parameters(strategy, overrides)
            capital_per_symbol = initial_capital * position_size

            async def _one(sym: str) -> BacktestResult:
                return await self._run_single_backtest(
                    sym,
                    strategy,
                    start,
                    end,
                    initial_capital=capital_per_symbol,
                    parameters=parameters,
                )

            results = await gather_bounded(symbols, _one)
            if not results:
                raise ValueError("No symbols could be backtested")

            total_return = sum(r.metrics.total_return for r in results) / len(results)
            average_sharpe = sum(r.metrics.sharpe_ratio for r in results) / len(results)
            max_drawdown = max(r.metrics.max_drawdown for r in results)
            total_trades = sum(r.metrics.total_trades for r in results)

            return PortfolioBacktestResult(
                portfolio_metrics=PortfolioBacktestMetrics(
                    symbols_tested=len(results),
                    total_return=total_return,
                    average_sharpe=average_sharpe,
                    max_drawdown=max_drawdown,
                    total_trades=total_trades,
                ),
                individual_results=results,
                summary=(
                    f"Portfolio backtest of {len(results)} symbols with {strategy} strategy"
                ),
            )

        return await self._run(_impl())
