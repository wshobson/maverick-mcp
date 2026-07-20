"""Tests for `maverick.backtesting.service`/`service_ml`/`service_support`.

`service.py` transitively imports `vectorbt` (via `engine`) and `sklearn` (via `service_ml`'s ML
strategy imports), so this whole module is guarded.
"""

import asyncio
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("vectorbt")
pytest.importorskip("sklearn")

from maverick.backtesting.config import BacktestingSettings
from maverick.backtesting.service import BacktestingService
from maverick.backtesting.types import (
    EnsembleBacktestResult,
    MarketRegimeAnalysis,
    MLBacktestResult,
    MLTrainingResult,
    MonteCarloResult,
    OptimizationResult,
    PortfolioBacktestResult,
    RunBacktestResult,
    StrategyCatalog,
    StrategyComparisonResult,
    WalkForwardResult,
)


class StubMarketData:
    """Async fake matching `MarketDataService.get_price_history`'s surface. Ignores the
    requested `start`/`end` window and returns a fixed frame per symbol (mirrors
    `tests/technical/test_service.py`'s `StubMarketData` -- date-range slicing correctness is
    already covered at the `engine`/`analysis` level in `test_engine.py`/`test_analysis.py`;
    this stub exists to prove the service's wiring, not re-derive engine math)."""

    def __init__(
        self,
        frame: pd.DataFrame | None = None,
        *,
        frames: dict[str, pd.DataFrame] | None = None,
        delay: float = 0.0,
        raise_for: dict[str, Exception] | None = None,
    ) -> None:
        self._frame = frame
        self._frames = frames or {}
        self._delay = delay
        self._raise_for = raise_for or {}
        self.calls: list[tuple[str, date | None, date | None]] = []

    async def get_price_history(
        self, symbol: str, start: date | None, end: date | None
    ) -> pd.DataFrame:
        self.calls.append((symbol, start, end))
        if self._delay:
            await asyncio.sleep(self._delay)
        if symbol in self._raise_for:
            raise self._raise_for[symbol]
        return self._frames.get(symbol, self._frame)


def _make_ohlcv(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Deterministic, Title-cased (`Open`/`High`/`Low`/`Close`/`Volume`) OHLCV frame -- Title
    casing is deliberate: it exercises `_fetch_frame`'s lowercasing step for real, matching what
    `MarketDataService.get_price_history` actually returns."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2022-01-03", periods=n)
    returns = rng.normal(0.0004, 0.015, n)
    close = 100 * np.cumprod(1 + returns)
    open_ = close * rng.uniform(0.99, 1.01, n)
    high = np.maximum(close, open_) * rng.uniform(1.0, 1.02, n)
    low = np.minimum(close, open_) * rng.uniform(0.98, 1.0, n)
    volume = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def _short_frame(n: int = 50) -> pd.DataFrame:
    return _make_ohlcv(n)


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])


@pytest.fixture(scope="module")
def ohlcv() -> pd.DataFrame:
    return _make_ohlcv()


def _service(
    market_data, settings: BacktestingSettings | None = None
) -> BacktestingService:
    return BacktestingService(market_data, settings=settings)


# ---------------------------------------------------------------------------
# 1. run_backtest
# ---------------------------------------------------------------------------


async def test_run_backtest_returns_typed_result_with_analysis(ohlcv):
    service = _service(StubMarketData(ohlcv))

    result = await service.run_backtest("AAPL", strategy="sma_cross", fast_period=10)

    assert isinstance(result, RunBacktestResult)
    assert result.symbol == "AAPL"
    assert result.strategy == "sma_cross"
    assert result.parameters["fast_period"] == 10
    assert result.analysis.performance_grade in {"A", "B", "C", "D", "F"}


async def test_run_backtest_raises_on_empty_fetch():
    service = _service(StubMarketData(_empty_frame()))

    with pytest.raises(ValueError, match="No price history"):
        await service.run_backtest("AAPL")


# ---------------------------------------------------------------------------
# 2. optimize_strategy
# ---------------------------------------------------------------------------


async def test_optimize_strategy_returns_typed_result(ohlcv):
    service = _service(StubMarketData(ohlcv))

    result = await service.optimize_strategy(
        "AAPL", strategy="sma_cross", optimization_level="coarse", top_n=3
    )

    assert isinstance(result, OptimizationResult)
    assert result.symbol == "AAPL"
    assert result.total_combinations_tested == 9  # 3 fast x 3 slow, coarse grid
    assert len(result.top_results) <= 3


async def test_optimize_strategy_rejects_unsupported_strategy(ohlcv):
    service = _service(StubMarketData(ohlcv))

    with pytest.raises(ValueError, match="Unknown strategy type"):
        await service.optimize_strategy("AAPL", strategy="ema_cross")


# ---------------------------------------------------------------------------
# 3. walk_forward_analysis
# ---------------------------------------------------------------------------


async def test_walk_forward_analysis_returns_typed_result_with_periods(ohlcv):
    service = _service(StubMarketData(ohlcv))
    end = date.today()
    # optimization_window (504d) + a bit of room -> at least one out-of-sample test period.
    start = end - timedelta(days=504 + 90)

    result = await service.walk_forward_analysis(
        "AAPL",
        strategy="sma_cross",
        start_date=start.isoformat(),
        end_date=end.isoformat(),
    )

    assert isinstance(result, WalkForwardResult)
    assert result.periods_tested >= 1
    assert len(result.walk_forward_results) == result.periods_tested


async def test_walk_forward_analysis_zero_periods_is_all_zero(ohlcv):
    service = _service(StubMarketData(ohlcv))
    end = date.today()
    start = end - timedelta(days=100)  # shorter than the 504d optimization window

    result = await service.walk_forward_analysis(
        "AAPL", start_date=start.isoformat(), end_date=end.isoformat()
    )

    assert result.periods_tested == 0
    assert result.average_return == 0.0
    assert result.consistency == 0.0


# ---------------------------------------------------------------------------
# 4. monte_carlo_simulation
# ---------------------------------------------------------------------------


async def test_monte_carlo_simulation_returns_typed_result(ohlcv):
    service = _service(StubMarketData(ohlcv))

    result = await service.monte_carlo_simulation("AAPL", num_simulations=200)

    assert isinstance(result, MonteCarloResult)
    assert result.num_simulations == 200
    assert set(result.return_percentiles.keys()) == {"p5", "p25", "p50", "p75", "p95"}


# ---------------------------------------------------------------------------
# 5. compare_strategies
# ---------------------------------------------------------------------------


async def test_compare_strategies_skips_failures_and_ranks_survivors(ohlcv):
    market_data = StubMarketData(frames={"AAPL": ohlcv}, raise_for={})
    service = _service(market_data)

    result = await service.compare_strategies(
        "AAPL", strategies=["sma_cross", "not_a_real_strategy", "rsi"]
    )

    assert isinstance(result, StrategyComparisonResult)
    strategy_names = {row.strategy for row in result.rankings}
    assert strategy_names == {"sma_cross", "rsi"}
    assert result.best_overall is not None


# ---------------------------------------------------------------------------
# 6. list_strategies
# ---------------------------------------------------------------------------


async def test_list_strategies_returns_all_12_templates(ohlcv):
    service = _service(StubMarketData(ohlcv))

    result = await service.list_strategies()

    assert isinstance(result, StrategyCatalog)
    assert result.total_count == 12
    assert "sma_cross" in result.available_strategies


# ---------------------------------------------------------------------------
# 7. backtest_portfolio
# ---------------------------------------------------------------------------


async def test_backtest_portfolio_aggregates_across_symbols(ohlcv):
    service = _service(StubMarketData(ohlcv))

    result = await service.backtest_portfolio(["AAPL", "MSFT", "GOOG"])

    assert isinstance(result, PortfolioBacktestResult)
    assert result.portfolio_metrics.symbols_tested == 3
    assert len(result.individual_results) == 3


async def test_backtest_portfolio_raises_when_every_symbol_fails():
    market_data = StubMarketData(
        raise_for={"AAPL": ValueError("boom"), "MSFT": ValueError("boom")}
    )
    service = _service(market_data)

    with pytest.raises(ValueError, match="No symbols could be backtested"):
        await service.backtest_portfolio(["AAPL", "MSFT"])


# ---------------------------------------------------------------------------
# 8. run_ml_strategy_backtest
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "strategy_type", ["ml_predictor", "adaptive", "ensemble", "regime_aware"]
)
async def test_run_ml_strategy_backtest_each_branch(ohlcv, strategy_type):
    service = _service(StubMarketData(ohlcv))

    result = await service.run_ml_strategy_backtest(
        "AAPL", strategy_type=strategy_type, n_estimators=10
    )

    assert isinstance(result, MLBacktestResult)
    assert result.ml_metrics["strategy_type"] == strategy_type


async def test_run_ml_strategy_backtest_rejects_unsupported_type(ohlcv):
    service = _service(StubMarketData(ohlcv))

    with pytest.raises(ValueError, match="Unsupported ML strategy type"):
        await service.run_ml_strategy_backtest("AAPL", strategy_type="not_a_type")


async def test_run_ml_strategy_backtest_rejects_insufficient_total_data():
    service = _service(StubMarketData(_short_frame(50)))

    with pytest.raises(ValueError, match="Insufficient data for ML strategy"):
        await service.run_ml_strategy_backtest("AAPL")


async def test_run_ml_strategy_backtest_rejects_insufficient_test_data():
    # 200 rows total but train_ratio=0.95 leaves only 10 test rows (< 50 required).
    service = _service(StubMarketData(_short_frame(200)))

    with pytest.raises(ValueError, match="Insufficient test data"):
        await service.run_ml_strategy_backtest("AAPL", train_ratio=0.95)


# ---------------------------------------------------------------------------
# 9. train_ml_predictor
# ---------------------------------------------------------------------------


async def test_train_ml_predictor_returns_typed_result(ohlcv):
    service = _service(StubMarketData(ohlcv))

    result = await service.train_ml_predictor("AAPL", n_estimators=10)

    assert isinstance(result, MLTrainingResult)
    assert result.symbol == "AAPL"
    assert result.data_points == len(ohlcv)


async def test_train_ml_predictor_rejects_insufficient_data():
    service = _service(StubMarketData(_short_frame(50)))

    with pytest.raises(ValueError, match="Insufficient data for ML training"):
        await service.train_ml_predictor("AAPL")


# ---------------------------------------------------------------------------
# 10. analyze_market_regimes
# ---------------------------------------------------------------------------


async def test_analyze_market_regimes_returns_typed_result(ohlcv):
    service = _service(StubMarketData(ohlcv))

    result = await service.analyze_market_regimes("AAPL", method="kmeans")

    assert isinstance(result, MarketRegimeAnalysis)
    assert result.symbol == "AAPL"
    assert result.n_regimes == 3
    assert len(result.recent_regime_history) <= 20


async def test_analyze_market_regimes_rejects_insufficient_data():
    service = _service(StubMarketData(_short_frame(60)))

    with pytest.raises(ValueError, match="Insufficient data for regime analysis"):
        await service.analyze_market_regimes("AAPL", lookback_period=50)


# ---------------------------------------------------------------------------
# 11. create_strategy_ensemble
# ---------------------------------------------------------------------------


async def test_create_strategy_ensemble_returns_typed_result(ohlcv):
    market_data = StubMarketData(ohlcv)
    service = _service(market_data)

    result = await service.create_strategy_ensemble(["AAPL", "MSFT"])

    assert isinstance(result, EnsembleBacktestResult)
    assert result.ensemble_summary.symbols_tested == 2


async def test_create_strategy_ensemble_calls_symbols_in_order_sequentially(ohlcv):
    """Proves the loop is sequential (no bounded-concurrency semaphore) -- a shared mutable
    `StrategyEnsemble` instance makes concurrent calls order-dependent (see service_ml.py)."""
    market_data = StubMarketData(ohlcv)
    service = _service(market_data)

    await service.create_strategy_ensemble(["MSFT", "AAPL", "GOOG"])

    assert [c[0] for c in market_data.calls] == ["MSFT", "AAPL", "GOOG"]


async def test_create_strategy_ensemble_raises_when_no_valid_base_strategies(ohlcv):
    service = _service(StubMarketData(ohlcv))

    with pytest.raises(ValueError, match="No valid base strategies"):
        await service.create_strategy_ensemble(
            ["AAPL"], base_strategies=["not_a_strategy"]
        )


# ---------------------------------------------------------------------------
# timeout
# ---------------------------------------------------------------------------


async def test_slow_fetch_raises_value_error_not_hang(ohlcv):
    market_data = StubMarketData(ohlcv, delay=0.2)
    service = _service(
        market_data, settings=BacktestingSettings(analysis_timeout_seconds=0.01)
    )

    with pytest.raises(ValueError, match="timed out"):
        await service.run_backtest("AAPL")
