"""`walk_forward_analysis` and the 4 ML-strategy-tool methods, split out of `service.py` to keep
both files under the repo's 500-line-per-file cap. `_ExtendedBacktestingMixin` is mixed into
`BacktestingService` (`service.py`); its methods call `self._fetch_frame`/`self._settings`,
defined on `BacktestingService` -- standard mixin usage, resolved via `self` at runtime.

`SimpleMovingAverageStrategy` (below) is a net-new, tools-tier class: the legacy ML tools build
base strategies from a `SimpleMovingAverageStrategy` that
`maverick/backtesting/strategies/templates.py` deliberately did not port (see that file's module
docstring: a tools-tier concern this service layer may add). It reuses
`strategies.signals.generate_signals(data, "sma_cross", self.parameters)` rather than
reimplementing SMA-crossover math a third time -- numerically equivalent to legacy's standalone
`rolling().mean()` version, disclosed as an implementation detail, not a behavior change.
"""

from datetime import date, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from maverick.backtesting import engine, optimization
from maverick.backtesting.config import BacktestingSettings
from maverick.backtesting.service_support import (
    WALK_FORWARD_OPTIMIZATION_WINDOW_DAYS,
    SimpleMovingAverageStrategy,
    generate_wf_summary,
    resolve_dates,
    signal_fn_for,
    to_simple_metrics,
)
from maverick.backtesting.strategies import signals as signal_dispatch
from maverick.backtesting.strategies.base import Strategy
from maverick.backtesting.strategies.ml.adaptive import AdaptiveStrategy
from maverick.backtesting.strategies.ml.ensemble import StrategyEnsemble
from maverick.backtesting.strategies.ml.ml_predictor import MLPredictor
from maverick.backtesting.strategies.ml.regime_aware import RegimeAwareStrategy
from maverick.backtesting.strategies.ml.regime_detector import MarketRegimeDetector
from maverick.backtesting.types import (
    EnsembleBacktestResult,
    EnsembleIndividualResult,
    EnsembleMemberResult,
    EnsembleSummary,
    MarketRegimeAnalysis,
    MLBacktestResult,
    MLTrainingResult,
    RegimeHistoryEntry,
    WalkForwardPeriodResult,
    WalkForwardResult,
)


class _ExtendedBacktestingMixin:
    """Mixed into `BacktestingService`; see module docstring."""

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

    # -- 8. run_ml_strategy_backtest -------------------------------------------

    async def run_ml_strategy_backtest(
        self,
        symbol: str,
        strategy_type: str = "ml_predictor",
        start_date: str | None = None,
        end_date: str | None = None,
        initial_capital: float = 10000.0,
        train_ratio: float = 0.8,
        model_type: str = "random_forest",
        n_estimators: int = 100,
        max_depth: int | None = None,
        learning_rate: float = 0.01,
        adaptation_method: str = "gradient",
    ) -> MLBacktestResult:
        async def _impl() -> MLBacktestResult:
            start, end = resolve_dates(start_date, end_date, default_days=730)
            frame = await self._fetch_frame(symbol, start, end)
            if len(frame) < 200:
                raise ValueError(
                    f"Insufficient data for ML strategy: {len(frame)} < 200 required"
                )
            split_idx = int(len(frame) * train_ratio)
            train_data = frame.iloc[:split_idx]
            test_data = frame.iloc[split_idx:]
            if len(train_data) < 100:
                raise ValueError(
                    f"Insufficient training data: {len(train_data)} < 100 required"
                )
            if len(test_data) < 50:
                raise ValueError(
                    f"Insufficient test data: {len(test_data)} < 50 required"
                )

            ml_strategy: Any
            training_metrics: dict[str, Any]
            if strategy_type == "ml_predictor":
                ml_strategy = MLPredictor(
                    model_type=model_type,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                )
                training_metrics = ml_strategy.train(train_data)
            elif strategy_type in ("adaptive", "online_learning"):
                base = SimpleMovingAverageStrategy()
                ml_strategy = AdaptiveStrategy(
                    base,
                    learning_rate=learning_rate,
                    adaptation_method=adaptation_method,
                )
                training_metrics = {
                    "adaptation_method": adaptation_method,
                    "strategy_alias": strategy_type,
                }
            elif strategy_type == "ensemble":
                bases: list[Strategy] = [
                    SimpleMovingAverageStrategy({"fast_period": 10, "slow_period": 20}),
                    SimpleMovingAverageStrategy({"fast_period": 5, "slow_period": 15}),
                ]
                ml_strategy = StrategyEnsemble(bases)
                training_metrics = {"ensemble_size": len(bases)}
            elif strategy_type == "regime_aware":
                bases_by_regime: dict[int, Strategy] = {
                    0: SimpleMovingAverageStrategy(
                        {"fast_period": 5, "slow_period": 20}
                    ),
                    1: SimpleMovingAverageStrategy(
                        {"fast_period": 10, "slow_period": 30}
                    ),
                    2: SimpleMovingAverageStrategy(
                        {"fast_period": 20, "slow_period": 50}
                    ),
                }
                ml_strategy = RegimeAwareStrategy(bases_by_regime)
                ml_strategy.fit_regime_detector(train_data)
                training_metrics = {"n_regimes": len(bases_by_regime)}
            else:
                raise ValueError(f"Unsupported ML strategy type: {strategy_type}")

            entries, exits = ml_strategy.generate_signals(test_data)
            full_result = engine.run_backtest(
                test_data,
                entries,
                exits,
                symbol=symbol,
                strategy=strategy_type,
                settings=self._settings.model_copy(
                    update={"initial_capital": initial_capital}
                ),
            )

            ml_metrics: dict[str, Any] = {
                "strategy_type": strategy_type,
                "training_period": len(train_data),
                "testing_period": len(test_data),
                "train_test_split": train_ratio,
                "training_metrics": training_metrics,
            }
            # `getattr(..., None)` (not `hasattr`) so the method reference itself, not its
            # result, is what gets null-checked -- a checker can't narrow `ml_strategy`'s type
            # through `hasattr`, and semantics must match legacy: call unconditionally once
            # present, even if the call happens to return `None`.
            get_fi = getattr(ml_strategy, "get_feature_importance", None)
            if get_fi is not None:
                ml_metrics["feature_importance"] = get_fi()
            get_ra = getattr(ml_strategy, "get_regime_analysis", None)
            if get_ra is not None:
                ml_metrics["regime_analysis"] = get_ra()
            get_sw = getattr(ml_strategy, "get_strategy_weights", None)
            if get_sw is not None:
                ml_metrics["strategy_weights"] = get_sw()

            return MLBacktestResult(
                metrics=to_simple_metrics(full_result.metrics),
                trades=[t.model_dump(by_alias=True) for t in full_result.trades],
                equity_curve=full_result.equity_curve,
                drawdown_series=full_result.drawdown_series,
                ml_metrics=ml_metrics,
            )

        return await self._run(_impl())

    # -- 9. train_ml_predictor --------------------------------------------------

    async def train_ml_predictor(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        model_type: str = "random_forest",
        target_periods: int = 5,
        return_threshold: float = 0.02,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
    ) -> MLTrainingResult:
        async def _impl() -> MLTrainingResult:
            start, end = resolve_dates(start_date, end_date, default_days=730)
            frame = await self._fetch_frame(symbol, start, end)
            if len(frame) < 200:
                raise ValueError(
                    "Insufficient data for ML training (minimum 200 data points)"
                )

            predictor = MLPredictor(
                model_type=model_type,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
            )
            training_metrics = predictor.train(
                frame, target_periods=target_periods, return_threshold=return_threshold
            )

            return MLTrainingResult(
                symbol=symbol,
                model_type=model_type,
                training_period=f"{start} to {end}",
                data_points=len(frame),
                target_periods=target_periods,
                return_threshold=return_threshold,
                model_parameters={
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                },
                training_metrics=training_metrics,
            )

        return await self._run(_impl())

    # -- 10. analyze_market_regimes ----------------------------------------------

    async def analyze_market_regimes(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        method: str = "hmm",
        n_regimes: int = 3,
        lookback_period: int = 50,
    ) -> MarketRegimeAnalysis:
        async def _impl() -> MarketRegimeAnalysis:
            start, end = resolve_dates(start_date, end_date, default_days=365)
            frame = await self._fetch_frame(symbol, start, end)
            if len(frame) < lookback_period + 50:
                raise ValueError(
                    "Insufficient data for regime analysis (minimum "
                    f"{lookback_period + 50} data points)"
                )

            detector = MarketRegimeDetector(
                method=method, n_regimes=n_regimes, lookback_period=lookback_period
            )
            detector.fit_regimes(frame)

            regime_history: list[RegimeHistoryEntry] = []
            for i in range(lookback_period, len(frame)):
                window = frame.iloc[i - lookback_period : i + 1]
                regime = detector.detect_current_regime(window)
                probs = detector.get_regime_probabilities(window)
                regime_history.append(
                    RegimeHistoryEntry(
                        date=frame.index[i].strftime("%Y-%m-%d"),
                        regime=int(regime),
                        probabilities=probs.tolist(),
                    )
                )

            regimes = [r.regime for r in regime_history]
            regime_counts = {i: regimes.count(i) for i in range(n_regimes)}
            regime_percentages = {
                k: (v / len(regimes)) * 100 for k, v in regime_counts.items()
            }

            regime_durations: dict[int, list[int]] = {i: [] for i in range(n_regimes)}
            current_regime = regimes[0]
            duration = 1
            for regime in regimes[1:]:
                if regime == current_regime:
                    duration += 1
                else:
                    regime_durations[current_regime].append(duration)
                    current_regime = regime
                    duration = 1
            regime_durations[current_regime].append(duration)
            avg_durations = {
                k: (float(np.mean(v)) if v else 0.0)
                for k, v in regime_durations.items()
            }

            return MarketRegimeAnalysis(
                symbol=symbol,
                analysis_period=f"{start} to {end}",
                method=method,
                n_regimes=n_regimes,
                # Hardcoded regardless of n_regimes -- a legacy quirk, preserved verbatim.
                regime_names={
                    0: "Bear/Declining",
                    1: "Sideways/Uncertain",
                    2: "Bull/Trending",
                },
                current_regime=regimes[-1] if regimes else 1,
                regime_counts=regime_counts,
                regime_percentages=regime_percentages,
                average_regime_durations=avg_durations,
                recent_regime_history=regime_history[-20:],
                total_regime_switches=sum(
                    1 for i in range(1, len(regimes)) if regimes[i] != regimes[i - 1]
                ),
            )

        return await self._run(_impl())

    # -- 11. create_strategy_ensemble --------------------------------------------

    async def create_strategy_ensemble(
        self,
        symbols: list[str],
        base_strategies: list[str] | None = None,
        weighting_method: str = "performance",
        start_date: str | None = None,
        end_date: str | None = None,
        initial_capital: float = 10000.0,
    ) -> EnsembleBacktestResult:
        async def _impl() -> EnsembleBacktestResult:
            strategy_list = base_strategies or ["sma_cross", "rsi", "macd"]
            start, end = resolve_dates(start_date, end_date, default_days=365)

            instances: list[Strategy] = []
            for name in strategy_list:
                if name == "sma_cross":
                    instances.append(SimpleMovingAverageStrategy())
                elif name == "rsi":
                    instances.append(
                        SimpleMovingAverageStrategy(
                            {"fast_period": 14, "slow_period": 28}
                        )
                    )
                elif name == "macd":
                    instances.append(
                        SimpleMovingAverageStrategy(
                            {"fast_period": 12, "slow_period": 26}
                        )
                    )
            if not instances:
                raise ValueError("No valid base strategies provided")

            ensemble = StrategyEnsemble(instances, weighting_method=weighting_method)

            # Deliberately sequential: `ensemble` mutates shared state per call (see docstring).
            results: list[EnsembleIndividualResult] = []
            total_return = 0.0
            total_trades = 0
            for symbol in symbols[:5]:
                try:
                    frame = await self._fetch_frame(symbol, start, end)
                    if len(frame) < 100:
                        continue
                    entries, exits = ensemble.generate_signals(frame)
                    full_result = engine.run_backtest(
                        frame,
                        entries,
                        exits,
                        symbol=symbol,
                        strategy="ensemble",
                        settings=self._settings.model_copy(
                            update={"initial_capital": initial_capital}
                        ),
                    )
                    member = EnsembleMemberResult(
                        metrics=to_simple_metrics(full_result.metrics),
                        trades=[
                            t.model_dump(by_alias=True) for t in full_result.trades
                        ],
                        equity_curve=full_result.equity_curve,
                        drawdown_series=full_result.drawdown_series,
                        ensemble_metrics={
                            "strategy_weights": ensemble.get_strategy_weights(),
                            "strategy_performance": ensemble.get_strategy_performance(),
                        },
                    )
                    results.append(
                        EnsembleIndividualResult(symbol=symbol, results=member)
                    )
                    total_return += full_result.metrics.total_return
                    total_trades += full_result.metrics.total_trades
                except Exception:
                    continue

            if not results:
                raise ValueError("No symbols could be processed")
            avg_return = total_return / len(results)
            avg_trades = total_trades / len(results)
            return EnsembleBacktestResult(
                ensemble_summary=EnsembleSummary(
                    symbols_tested=len(results),
                    base_strategies=strategy_list,
                    weighting_method=weighting_method,
                    average_return=avg_return,
                    total_trades=total_trades,
                    average_trades_per_symbol=avg_trades,
                ),
                individual_results=results,
                final_strategy_weights=ensemble.get_strategy_weights(),
                strategy_performance_analysis=ensemble.get_strategy_performance(),
            )

        return await self._run(_impl())
