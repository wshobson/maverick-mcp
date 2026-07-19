"""Characterization tests for `maverick.backtesting.engine`.

`pytest.importorskip("vectorbt")` guards this whole module: vectorbt is
only installed under the `backtesting` extra, so these tests are skipped
(not failed) when it's absent. The OHLCV fixture is a deterministic,
seeded synthetic series (no network) following the same shape as
`tests/test_ml_strategies.py`'s `sample_market_data` fixture, but built
from a seeded `np.random.default_rng` instead of the legacy global
`np.random.seed`, so it never depends on -- or mutates -- interpreter-wide
random state.

`sma_cross` signal generation is hardcoded here (`_sma_cross_signals`)
rather than imported from `maverick.backtesting.strategies.templates`,
which remains a docstring-only stub reserved for Task 5 -- see
`engine.py`'s module docstring for why the engine itself never imports
strategy template code.

Numeric assertions below are *pinned*: they were recorded by actually
running this code against the fixture below (not derived independently),
per the task's characterization-testing brief. A change to this module's
math should change these numbers and fail the test; that is the point.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("vectorbt")

import vectorbt as vbt

from maverick.backtesting.config import BacktestingSettings
from maverick.backtesting.engine import optimize_parameters, run_backtest
from maverick.backtesting.types import BacktestResult, OptimizationResult


def _sma_cross_signals(
    frame: pd.DataFrame, params: dict
) -> tuple[pd.Series, pd.Series]:
    """Minimal, self-contained port of the sma-crossover branch of legacy
    `VectorBTEngine._sma_crossover_signals` -- just enough to characterize
    `run_backtest`/`optimize_parameters` without depending on
    `strategies/templates.py` (Task 5)."""
    close = frame["close"]
    fast = vbt.MA.run(
        close, params.get("fast_period", 10), short_name="fast"
    ).ma.squeeze()
    slow = vbt.MA.run(
        close, params.get("slow_period", 20), short_name="slow"
    ).ma.squeeze()
    entries = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    exits = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    return entries.fillna(False), exits.fillna(False)


@pytest.fixture
def ohlcv_frame() -> pd.DataFrame:
    """300 rows of deterministic synthetic daily OHLCV data, seed 42."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=300, freq="D")
    returns = rng.normal(0.0005, 0.02, len(dates))
    prices = 100 * np.cumprod(1 + returns)
    volumes = rng.integers(1_000_000, 5_000_000, len(dates))
    frame = pd.DataFrame(
        {
            "open": prices * rng.uniform(0.98, 1.02, len(dates)),
            "high": prices * rng.uniform(1.00, 1.05, len(dates)),
            "low": prices * rng.uniform(0.95, 1.00, len(dates)),
            "close": prices,
            "volume": volumes,
        },
        index=dates,
    )
    frame["high"] = np.maximum(
        frame["high"], np.maximum(frame["open"], frame["close"])
    )
    frame["low"] = np.minimum(frame["low"], np.minimum(frame["open"], frame["close"]))
    return frame


# -- run_backtest -------------------------------------------------------


def test_sma_cross_end_to_end_pins_metrics(ohlcv_frame):
    entries, exits = _sma_cross_signals(
        ohlcv_frame, {"fast_period": 10, "slow_period": 20}
    )

    result = run_backtest(
        ohlcv_frame,
        entries,
        exits,
        symbol="TEST",
        strategy="sma_cross",
        parameters={"fast_period": 10, "slow_period": 20},
        settings=BacktestingSettings(),
    )

    assert isinstance(result, BacktestResult)
    assert result.symbol == "TEST"
    assert result.strategy == "sma_cross"
    assert result.parameters == {"fast_period": 10, "slow_period": 20}
    assert result.initial_capital == 10000.0
    assert result.start_date == "2023-01-01"
    assert result.end_date == "2023-10-27"

    metrics = result.metrics
    assert metrics.total_trades == 8
    assert metrics.winning_trades == 2
    assert metrics.losing_trades == 6
    assert metrics.total_return == pytest.approx(-0.1939859295537608, rel=1e-6)
    assert metrics.sharpe_ratio == pytest.approx(-1.1935470172889275, rel=1e-6)
    assert metrics.sortino_ratio == pytest.approx(-1.6432226726027797, rel=1e-6)
    assert metrics.max_drawdown == pytest.approx(-0.25587123351061325, rel=1e-6)
    assert metrics.win_rate == pytest.approx(0.25, rel=1e-6)
    assert metrics.profit_factor == pytest.approx(0.16752545611905578, rel=1e-6)
    assert metrics.kelly_criterion == pytest.approx(-1.0, rel=1e-6)
    assert metrics.avg_duration == pytest.approx(11.5, rel=1e-6)


def test_sma_cross_first_trade_pinned(ohlcv_frame):
    entries, exits = _sma_cross_signals(
        ohlcv_frame, {"fast_period": 10, "slow_period": 20}
    )
    result = run_backtest(ohlcv_frame, entries, exits, settings=BacktestingSettings())

    first = result.trades[0]
    assert first.entry_date == "2023-03-13 00:00:00"
    assert first.exit_date == "2023-03-17 00:00:00"
    assert first.entry_price == pytest.approx(106.17708541870115, rel=1e-6)
    assert first.exit_price == pytest.approx(107.01805792236328, rel=1e-6)
    assert first.pnl == pytest.approx(59.06643675872606, rel=1e-6)
    assert first.return_ == pytest.approx(0.005912550319548479, rel=1e-6)
    # No "Duration" column in vectorbt 1.0's `records_readable` -- the
    # legacy `.get("Duration", "")` fallback is exercised, not a bug in
    # this port. See `engine.py::_extract_trades`.
    assert first.duration == ""


def test_run_backtest_rejects_empty_frame():
    empty = pd.DataFrame(columns=["close"])
    with pytest.raises(ValueError, match="non-empty"):
        run_backtest(empty, pd.Series(dtype=bool), pd.Series(dtype=bool))


def test_run_backtest_rejects_too_short_frame():
    dates = pd.date_range("2023-01-01", periods=1, freq="D")
    frame = pd.DataFrame({"close": [100.0]}, index=dates)
    entries = pd.Series([False], index=dates)
    exits = pd.Series([False], index=dates)
    with pytest.raises(ValueError, match="at least"):
        run_backtest(frame, entries, exits)


def test_run_backtest_rejects_missing_close_column(ohlcv_frame):
    frame = ohlcv_frame.drop(columns=["close"])
    entries = pd.Series(False, index=frame.index)
    exits = pd.Series(False, index=frame.index)
    with pytest.raises(ValueError, match="close"):
        run_backtest(frame, entries, exits)


def test_run_backtest_rejects_mismatched_signal_length(ohlcv_frame):
    entries = pd.Series(False, index=ohlcv_frame.index[:-1])
    exits = pd.Series(False, index=ohlcv_frame.index)
    with pytest.raises(ValueError, match="length"):
        run_backtest(ohlcv_frame, entries, exits)


# -- optimize_parameters --------------------------------------------------


def test_optimize_parameters_pins_best_result(ohlcv_frame):
    result = optimize_parameters(
        ohlcv_frame,
        _sma_cross_signals,
        {"fast_period": [5, 10, 15], "slow_period": [20, 30]},
        symbol="TEST",
        strategy="sma_cross",
        optimization_metric="sharpe_ratio",
        top_n=3,
        settings=BacktestingSettings(),
    )

    assert isinstance(result, OptimizationResult)
    assert result.total_combinations_tested == 6
    assert result.valid_combinations == 6
    assert result.best_parameters == {"fast_period": 15, "slow_period": 20}
    assert result.best_metric_value == pytest.approx(-0.6870859573334221, rel=1e-6)
    assert len(result.top_results) == 3

    best_row = result.top_results[0]
    assert best_row.parameters == {"fast_period": 15, "slow_period": 20}
    assert best_row.total_return == pytest.approx(-0.14557239794854704, rel=1e-6)
    assert best_row.max_drawdown == pytest.approx(-0.2707973404531474, rel=1e-6)
    assert best_row.total_trades == 12
    # Dynamic key named after `optimization_metric`, preserved via
    # `OptimizationResultRow`'s `extra="allow"`.
    assert best_row.sharpe_ratio == pytest.approx(-0.6870859573334221, rel=1e-6)


def test_optimize_parameters_no_slippage_differs_from_run_backtest(ohlcv_frame):
    """Legacy quirk, faithfully preserved: `optimize_parameters` never
    applies slippage, so its per-combo total_return differs from
    `run_backtest`'s for the identical parameters/frame."""
    entries, exits = _sma_cross_signals(
        ohlcv_frame, {"fast_period": 10, "slow_period": 20}
    )
    backtest = run_backtest(ohlcv_frame, entries, exits, settings=BacktestingSettings())

    result = optimize_parameters(
        ohlcv_frame,
        _sma_cross_signals,
        {"fast_period": [10], "slow_period": [20]},
        optimization_metric="total_return",
        settings=BacktestingSettings(),
    )

    optimize_return = result.top_results[0].total_return
    assert optimize_return != pytest.approx(backtest.metrics.total_return)
    assert optimize_return == pytest.approx(-0.18098597781057052, rel=1e-6)
    assert result.top_results[0].total_trades == backtest.metrics.total_trades


def test_optimize_parameters_chunking_matches_unchunked(ohlcv_frame):
    """Forcing the `total_combos > threshold` chunked path (via a low
    `optimization_chunk_threshold`) must not change which combination
    wins -- chunking only changes processing order/grouping."""
    grid = {"fast_period": [5, 10, 15], "slow_period": [20, 30]}

    unchunked = optimize_parameters(
        ohlcv_frame, _sma_cross_signals, grid, settings=BacktestingSettings()
    )
    chunked = optimize_parameters(
        ohlcv_frame,
        _sma_cross_signals,
        grid,
        settings=BacktestingSettings(
            optimization_chunk_threshold=2,
            optimization_chunk_size_min=1,
            optimization_chunk_size_max=2,
        ),
    )

    assert chunked.best_parameters == unchunked.best_parameters
    assert chunked.best_metric_value == pytest.approx(unchunked.best_metric_value)
    assert chunked.valid_combinations == unchunked.valid_combinations


def test_optimize_parameters_rejects_unknown_metric(ohlcv_frame):
    with pytest.raises(ValueError, match="Unknown metric"):
        optimize_parameters(
            ohlcv_frame,
            _sma_cross_signals,
            {"fast_period": [10]},
            optimization_metric="not_a_real_metric",
        )


def test_optimize_parameters_rejects_empty_grid(ohlcv_frame):
    with pytest.raises(ValueError, match="no parameter combinations"):
        optimize_parameters(ohlcv_frame, _sma_cross_signals, {"fast_period": []})


def test_optimize_parameters_rejects_empty_frame():
    empty = pd.DataFrame(columns=["close"])
    with pytest.raises(ValueError, match="non-empty"):
        optimize_parameters(empty, _sma_cross_signals, {"fast_period": [10]})


def test_optimize_parameters_rejects_too_short_frame():
    dates = pd.date_range("2023-01-01", periods=1, freq="D")
    frame = pd.DataFrame({"close": [100.0]}, index=dates)
    with pytest.raises(ValueError, match="at least"):
        optimize_parameters(frame, _sma_cross_signals, {"fast_period": [10]})
