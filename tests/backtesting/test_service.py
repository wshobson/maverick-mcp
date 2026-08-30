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
    BacktestMetrics,
    BacktestResult,
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


def _canned_backtest_result(symbol: str, max_drawdown: float) -> BacktestResult:
    """Minimal `BacktestResult` with a controlled `metrics.max_drawdown`, for
    aggregation tests that don't need a real vectorbt-driven backtest to
    produce a specific drawdown."""
    return BacktestResult(
        symbol=symbol,
        strategy="sma_cross",
        parameters={},
        metrics=BacktestMetrics(
            total_return=0.0,
            annual_return=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown=max_drawdown,
            win_rate=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            avg_win=0.0,
            avg_loss=0.0,
            best_trade=0.0,
            worst_trade=0.0,
            avg_duration=0.0,
            kelly_criterion=0.0,
            recovery_factor=0.0,
            risk_reward_ratio=0.0,
        ),
        trades=[],
        equity_curve={},
        drawdown_series={},
        start_date="2022-01-01",
        end_date="2022-12-31",
        initial_capital=1000.0,
    )


def _stub_per_symbol_drawdowns(
    service: BacktestingService, drawdowns: dict[str, float]
) -> None:
    """Monkeypatch `_run_single_backtest` to return a canned result per symbol,
    isolating `backtest_portfolio`'s aggregation from vectorbt/signal-generation
    entirely."""

    async def _fake(symbol, strategy, start, end, *, initial_capital, parameters=None):
        return _canned_backtest_result(symbol, drawdowns[symbol])

    service._run_single_backtest = _fake  # type: ignore[method-assign]


async def test_backtest_portfolio_max_drawdown_selects_worst_not_mildest():
    """Regression test for the `max()`-on-signed-values bug: a portfolio
    containing a mild -12.4% drawdown and a severe -21.3% drawdown must report
    the severe one, not the mild one."""
    service = _service(StubMarketData(pd.DataFrame()))
    _stub_per_symbol_drawdowns(service, {"AAPL": -0.124, "MSFT": -0.213})

    result = await service.backtest_portfolio(["AAPL", "MSFT"])

    assert result.portfolio_metrics.max_drawdown == pytest.approx(-0.213)


async def test_backtest_portfolio_max_drawdown_is_order_independent():
    service_a = _service(StubMarketData(pd.DataFrame()))
    _stub_per_symbol_drawdowns(
        service_a, {"AAPL": -0.124, "MSFT": -0.213, "JPM": -0.05}
    )
    service_b = _service(StubMarketData(pd.DataFrame()))
    _stub_per_symbol_drawdowns(
        service_b, {"AAPL": -0.124, "MSFT": -0.213, "JPM": -0.05}
    )

    result_forward = await service_a.backtest_portfolio(["AAPL", "MSFT", "JPM"])
    result_reversed = await service_b.backtest_portfolio(["JPM", "MSFT", "AAPL"])

    assert result_forward.portfolio_metrics.max_drawdown == pytest.approx(-0.213)
    assert result_reversed.portfolio_metrics.max_drawdown == pytest.approx(-0.213)


async def test_backtest_portfolio_max_drawdown_zero_when_no_symbol_drew_down():
    service = _service(StubMarketData(pd.DataFrame()))
    _stub_per_symbol_drawdowns(service, {"AAPL": 0.0, "MSFT": 0.0})

    result = await service.backtest_portfolio(["AAPL", "MSFT"])

    assert result.portfolio_metrics.max_drawdown == pytest.approx(0.0)


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


async def test_analyze_market_regimes_reports_actual_fallback_method():
    """Regression test: with enough data to pass the tool's own minimum-data
    guard but not enough to clear `MarketRegimeDetector`'s internal
    fit-sample threshold, the detector silently falls back to the rule-based
    "threshold" method. The reported `method` field must reflect that actual
    fallback, not the originally-requested "hmm"/"kmeans" -- previously it
    always echoed the request, misrepresenting what was actually used."""
    service = _service(StubMarketData(_make_ohlcv(150)))

    result = await service.analyze_market_regimes(
        "AAPL", method="hmm", lookback_period=50
    )

    assert result.method == "threshold"
    # Every probability entry must be the exact one-hot vector at its own
    # `regime` -- never the previous fabricated uniform [1/3, 1/3, 1/3], and
    # not merely "sums to 1 with a max of 1" (which a non-one-hot vector
    # could also satisfy).
    for entry in result.recent_regime_history:
        expected = [0.0, 0.0, 0.0]
        expected[entry.regime] = 1.0
        assert entry.probabilities == expected


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


async def test_create_strategy_ensemble_raises_clearly_for_unknown_base_strategy(ohlcv):
    """`TemplateStrategy` now validates each requested name eagerly against
    `STRATEGY_TEMPLATES` and raises immediately naming the bad value, rather than silently
    dropping it and only failing later with a generic "no valid base strategies" error."""
    service = _service(StubMarketData(ohlcv))

    with pytest.raises(
        ValueError, match="Unknown ensemble base strategy: 'not_a_strategy'"
    ):
        await service.create_strategy_ensemble(
            ["AAPL"], base_strategies=["not_a_strategy"]
        )


async def test_create_strategy_ensemble_rsi_and_macd_use_real_signal_logic(ohlcv):
    """Regression test for the rsi/macd mislabeling defect: requesting "rsi"/"macd" base
    strategies must run actual RSI/MACD signal generation (via `TemplateStrategy`), not
    SMA-crossover under a different name, and must keep three distinct, separately-addressable
    identities in the result -- not collapse into one shared "SMA Crossover" key."""
    service = _service(StubMarketData(ohlcv))

    result = await service.create_strategy_ensemble(
        ["AAPL"], base_strategies=["sma_cross", "rsi", "macd"]
    )

    weight_names = set(result.final_strategy_weights)
    assert weight_names == {"SMA Crossover", "RSI Mean Reversion", "MACD Signal"}
    performance_names = set(result.strategy_performance_analysis)
    assert performance_names == {"SMA Crossover", "RSI Mean Reversion", "MACD Signal"}


async def test_create_strategy_ensemble_rsi_signals_differ_from_sma_signals(ohlcv):
    """`TemplateStrategy("rsi")` must dispatch to real RSI logic, not SMA -- confirmed by
    comparing its generated signals directly against `TemplateStrategy("sma_cross")`'s on the
    same frame; identical signal generation would mean the mislabeling regressed."""
    from maverick.backtesting.service_support import TemplateStrategy

    frame = ohlcv.rename(columns=str.lower)
    sma_entries, sma_exits = TemplateStrategy("sma_cross").generate_signals(frame)
    rsi_entries, rsi_exits = TemplateStrategy("rsi").generate_signals(frame)
    macd_entries, macd_exits = TemplateStrategy("macd").generate_signals(frame)

    assert not sma_entries.equals(rsi_entries) or not sma_exits.equals(rsi_exits)
    assert not sma_entries.equals(macd_entries) or not sma_exits.equals(macd_exits)
    assert not rsi_entries.equals(macd_entries) or not rsi_exits.equals(macd_exits)


async def test_template_strategy_names_match_templates():
    from maverick.backtesting.service_support import TemplateStrategy

    assert TemplateStrategy("sma_cross").name == "SMA Crossover"
    assert TemplateStrategy("rsi").name == "RSI Mean Reversion"
    assert TemplateStrategy("macd").name == "MACD Signal"


async def test_template_strategy_rejects_unknown_strategy_type():
    from maverick.backtesting.service_support import TemplateStrategy

    with pytest.raises(ValueError, match="Unknown ensemble base strategy: 'bogus'"):
        TemplateStrategy("bogus")


async def test_template_strategy_to_dict_exposes_real_template_defaults():
    """Regression test: `Strategy.to_dict()` calls `get_default_parameters()`, which the base
    class defaults to `{}`. Without overriding it, `TemplateStrategy.to_dict()` would report
    an empty `default_parameters` even though the selected template's real defaults are known
    via `self.strategy_type`."""
    from maverick.backtesting.service_support import TemplateStrategy

    rsi_defaults = TemplateStrategy("rsi").to_dict()["default_parameters"]
    assert rsi_defaults == {"period": 14, "oversold": 30, "overbought": 70}

    # Overriding parameters at construction must not change the template's own defaults.
    overridden = TemplateStrategy(
        "rsi", {"period": 21, "oversold": 25, "overbought": 75}
    )
    assert overridden.to_dict()["default_parameters"] == rsi_defaults
    assert overridden.to_dict()["parameters"] == {
        "period": 21,
        "oversold": 25,
        "overbought": 75,
    }


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
