"""Characterization tests for `maverick.backtesting.analysis`.

`pytest.importorskip("vectorbt")` guards this module because `analyze`/
`monte_carlo_simulation` are exercised here against a real
`maverick.backtesting.engine.run_backtest` result (so vectorbt has to be
importable to build the fixture), even though neither function imports
vectorbt itself. `compare_strategies` is exercised against hand-built
`BacktestResult`s (no vectorbt needed) using the same `_metrics`/`_result`
helper pattern as `tests/backtesting/test_store.py`.

Numeric assertions are pinned: recorded by actually running this code, per
the task's characterization-testing brief.
"""

import pandas as pd
import pytest

pytest.importorskip("vectorbt")

from maverick.backtesting.analysis import (
    analyze,
    compare_strategies,
    monte_carlo_simulation,
)
from maverick.backtesting.config import BacktestingSettings
from maverick.backtesting.engine import run_backtest
from maverick.backtesting.types import (
    BacktestAnalysis,
    BacktestMetrics,
    BacktestResult,
    MonteCarloResult,
    StrategyComparisonResult,
)

# `tiny_backtest_result` below is the first fixture in the alphabetically-first
# backtesting test module to call the real `engine.run_backtest`, so on a cold
# CI runner (no warm numba on-disk JIT cache) it eats the one-time cost of
# numba compiling vectorbt's portfolio/indicator kernels from scratch --
# observed at just over CI's default 60s `--timeout`. Every later call in the
# same process reuses the compiled functions and is fast (single-digit
# seconds), so only this module needs the wider budget.
pytestmark = pytest.mark.timeout(300)


def _metrics(**overrides) -> BacktestMetrics:
    fields = {
        "total_return": 0.2534,
        "annual_return": 0.1892,
        "sharpe_ratio": 1.2345,
        "sortino_ratio": 1.5678,
        "calmar_ratio": 0.9876,
        "max_drawdown": -0.1234,
        "win_rate": 0.5556,
        "profit_factor": 1.8765,
        "expectancy": 12.34,
        "total_trades": 20,
        "winning_trades": 11,
        "losing_trades": 9,
        "avg_win": 150.25,
        "avg_loss": -80.5,
        "best_trade": 320.75,
        "worst_trade": -150.0,
        "avg_duration": 5.5,
        "kelly_criterion": 0.15,
        "recovery_factor": 2.1,
        "risk_reward_ratio": 1.9,
    }
    fields.update(overrides)
    return BacktestMetrics(**fields)


def _result(**overrides) -> BacktestResult:
    fields = {
        "symbol": "TEST",
        "strategy": "A",
        "parameters": {},
        "metrics": _metrics(),
        "trades": [],
        "equity_curve": {},
        "drawdown_series": {},
        "start_date": "2023-01-01",
        "end_date": "2023-06-01",
        "initial_capital": 10000.0,
    }
    fields.update(overrides)
    return BacktestResult(**fields)


@pytest.fixture
def tiny_backtest_result() -> BacktestResult:
    """A tiny, hand-crafted 6-row frame with two full round-trip trades
    (one winner, one loser), run through the real engine so `analyze`/
    `monte_carlo_simulation` are exercised against genuine engine output
    rather than a hand-built `BacktestResult`."""
    dates = pd.date_range("2023-01-01", periods=6, freq="D")
    frame = pd.DataFrame(
        {"close": [100.0, 105.0, 102.0, 108.0, 104.0, 110.0]}, index=dates
    )
    entries = pd.Series([True, False, False, True, False, False], index=dates)
    exits = pd.Series([False, True, False, False, True, False], index=dates)
    return run_backtest(
        frame,
        entries,
        exits,
        symbol="TINY",
        strategy="manual",
        settings=BacktestingSettings(),
    )


# -- analyze ----------------------------------------------------------------


def test_analyze_pins_grade_and_risk_assessment(tiny_backtest_result):
    analysis = analyze(tiny_backtest_result)

    assert isinstance(analysis, BacktestAnalysis)
    assert analysis.performance_grade == "F"
    assert analysis.risk_assessment.risk_level == "Low"
    assert analysis.risk_assessment.downside_protection == "Moderate"
    assert analysis.risk_assessment.max_drawdown == pytest.approx(
        0.0408811967253524, rel=1e-6
    )
    assert analysis.trade_quality.quality == "Average"
    assert analysis.trade_quality.total_trades == 2
    assert analysis.trade_quality.frequency == "Very Low"
    assert analysis.weaknesses == ["Insufficient trade signals"]
    assert "Consider more sensitive parameters for increased signals" in (
        analysis.recommendations
    )
    assert analysis.summary == (
        "The strategy generated a 0.3% return with a Sharpe ratio of 0.57. "
        "Maximum drawdown was 4.1% with a 50.0% win rate across 2 trades. "
        "Performance is moderate and could benefit from optimization."
    )


def test_analyze_trade_quality_reports_no_trades_for_empty_trade_list():
    result = _result(metrics=_metrics(total_trades=0), trades=[])
    analysis = analyze(result)
    assert analysis.trade_quality.quality == "No trades"
    assert analysis.trade_quality.total_trades == 0
    assert analysis.trade_quality.frequency == "None"
    assert analysis.trade_quality.win_rate is None


# -- compare_strategies -------------------------------------------------------


def test_compare_strategies_ranks_by_sharpe_and_picks_bests():
    result_a = _result(
        strategy="A",
        metrics=_metrics(
            total_return=0.30,
            sharpe_ratio=1.8,
            max_drawdown=-0.12,
            win_rate=0.55,
            profit_factor=1.6,
            total_trades=40,
        ),
    )
    result_b = _result(
        strategy="B",
        metrics=_metrics(
            total_return=0.10,
            sharpe_ratio=2.1,
            max_drawdown=-0.05,
            win_rate=0.65,
            profit_factor=2.2,
            total_trades=15,
        ),
    )

    comparison = compare_strategies([result_a, result_b])

    assert isinstance(comparison, StrategyComparisonResult)
    assert [row.strategy for row in comparison.rankings] == ["B", "A"]
    assert [row.rank for row in comparison.rankings] == [1, 2]
    assert comparison.best_overall.strategy == "B"
    assert comparison.best_return.strategy == "A"
    assert comparison.best_sharpe.strategy == "B"
    assert comparison.best_drawdown.strategy == "B"
    assert comparison.best_win_rate.strategy == "B"
    assert comparison.summary == (
        "The best performing strategy is B with a Sharpe ratio of 2.10 "
        "and total return of 10.0%. It outperformed 1 other strategies tested."
    )


def test_compare_strategies_rejects_empty_list():
    with pytest.raises(ValueError, match="at least one result"):
        compare_strategies([])


# -- monte_carlo_simulation --------------------------------------------------


def test_monte_carlo_simulation_pins_percentile_bands(tiny_backtest_result):
    mc = monte_carlo_simulation(
        tiny_backtest_result.trades, num_simulations=500, seed=7
    )

    assert isinstance(mc, MonteCarloResult)
    assert mc.num_simulations == 500
    assert set(mc.return_percentiles) == {"p5", "p25", "p50", "p75", "p95"}
    assert mc.expected_return == pytest.approx(0.002154510716596274, rel=1e-6)
    assert mc.return_std == pytest.approx(0.06139191887645371, rel=1e-6)
    assert mc.return_percentiles["p5"] == pytest.approx(-0.08016953938269455, rel=1e-6)
    assert mc.return_percentiles["p50"] == pytest.approx(0.00305566890836495, rel=1e-6)
    assert mc.return_percentiles["p95"] == pytest.approx(0.09381099888124167, rel=1e-6)
    assert mc.drawdown_percentiles["p5"] == pytest.approx(
        -0.04092207792207757, rel=1e-6
    )
    assert mc.probability_profit == pytest.approx(0.734, rel=1e-6)
    assert mc.var_95 == pytest.approx(-0.08016953938269455, rel=1e-6)
    assert mc.summary == (
        "Monte Carlo simulation shows 0.2% expected return with 73.4% "
        "probability of profit. 95% Value at Risk is 8.0%. Strategy shows "
        "positive expectancy with moderate confidence."
    )


def test_monte_carlo_simulation_is_deterministic_for_a_fixed_seed(
    tiny_backtest_result,
):
    first = monte_carlo_simulation(
        tiny_backtest_result.trades, num_simulations=200, seed=99
    )
    second = monte_carlo_simulation(
        tiny_backtest_result.trades, num_simulations=200, seed=99
    )
    assert first.model_dump() == second.model_dump()


def test_monte_carlo_simulation_rejects_empty_trades():
    with pytest.raises(ValueError, match="at least one trade"):
        monte_carlo_simulation([])
