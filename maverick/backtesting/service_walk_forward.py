"""`walk_forward_analysis`, split out of `service_ml.py` to keep that file under the
repo's 500-line-per-file cap. `_WalkForwardMixin` is the base of `_ExtendedBacktestingMixin`
(`service_ml.py`), which `BacktestingService` (`service.py`) inherits, so the method is still
callable on `BacktestingService` exactly as before. Same layer as `service_ml.py`; its methods
call `self._run`/`self._fetch_frame`/`self._settings`, defined on `BacktestingService`.
"""

from datetime import date, timedelta
from typing import TYPE_CHECKING, Any

import pandas as pd

from maverick.backtesting import engine, optimization
from maverick.backtesting.config import BacktestingSettings
from maverick.backtesting.service_support import (
    WALK_FORWARD_OPTIMIZATION_WINDOW_DAYS,
    generate_wf_summary,
    resolve_dates,
    signal_fn_for,
)
from maverick.backtesting.strategies import signals as signal_dispatch
from maverick.backtesting.types import WalkForwardPeriodResult, WalkForwardResult


class _WalkForwardMixin:
    """Base of `_ExtendedBacktestingMixin`; see module docstring."""

    # Declared for the type checker only: provided by `BacktestingService` at runtime.
    _settings: BacktestingSettings
    if TYPE_CHECKING:

        async def _run(self, coro: Any) -> Any: ...
        async def _fetch_frame(
            self, symbol: str, start: date, end: date
        ) -> pd.DataFrame: ...

    # -- 3. walk_forward_analysis ---------------------------------------------

    async def walk_forward_analysis(
        self,
        symbol: str,
        strategy: str = "sma_cross",
        start_date: str | None = None,
        end_date: str | None = None,
        window_size: int = 252,
        step_size: int = 63,
    ) -> WalkForwardResult:
        async def _impl() -> WalkForwardResult:
            start, end = resolve_dates(start_date, end_date, default_days=1095)
            optimization_window = WALK_FORWARD_OPTIMIZATION_WINDOW_DAYS

            results: list[WalkForwardPeriodResult] = []
            current = start + timedelta(days=optimization_window)
            while current <= end:
                opt_start = current - timedelta(days=optimization_window)
                opt_end = current
                test_start = current
                test_end = min(current + timedelta(days=window_size), end)

                opt_frame = await self._fetch_frame(symbol, opt_start, opt_end)
                grid = optimization.generate_param_grid(strategy, "coarse")
                opt_result = engine.optimize_parameters(
                    opt_frame,
                    signal_fn_for(strategy),
                    grid,
                    symbol=symbol,
                    strategy=strategy,
                    top_n=1,
                    settings=self._settings,
                )
                if opt_result.best_metric_value is None:
                    raise ValueError(
                        f"No valid optimization candidates for {symbol} "
                        f"from {opt_start} to {opt_end}"
                    )
                best_params = opt_result.best_parameters

                if test_start < test_end:
                    test_frame = await self._fetch_frame(symbol, test_start, test_end)
                    entries, exits = signal_dispatch.generate_signals(
                        test_frame, strategy, best_params
                    )
                    test_result = engine.run_backtest(
                        test_frame,
                        entries,
                        exits,
                        symbol=symbol,
                        strategy=strategy,
                        parameters=best_params,
                        settings=self._settings,
                    )
                    results.append(
                        WalkForwardPeriodResult(
                            period=f"{test_start:%Y-%m-%d} to {test_end:%Y-%m-%d}",
                            parameters=best_params,
                            in_sample_sharpe=opt_result.best_metric_value,
                            out_sample_return=test_result.metrics.total_return,
                            out_sample_sharpe=test_result.metrics.sharpe_ratio,
                            out_sample_drawdown=test_result.metrics.max_drawdown,
                        )
                    )

                current += timedelta(days=step_size)

            if results:
                n = len(results)
                avg_return = sum(r.out_sample_return for r in results) / n
                avg_sharpe = sum(r.out_sample_sharpe for r in results) / n
                avg_drawdown = sum(r.out_sample_drawdown for r in results) / n
                consistency = sum(1 for r in results if r.out_sample_return > 0) / n
            else:
                avg_return = avg_sharpe = avg_drawdown = consistency = 0.0

            return WalkForwardResult(
                symbol=symbol,
                strategy=strategy,
                periods_tested=len(results),
                average_return=avg_return,
                average_sharpe=avg_sharpe,
                average_drawdown=avg_drawdown,
                consistency=consistency,
                walk_forward_results=results,
                summary=generate_wf_summary(avg_return, avg_sharpe, consistency),
            )

        return await self._run(_impl())
