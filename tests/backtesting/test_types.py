"""Tests for maverick.backtesting.types."""

import pytest
from pydantic import ValidationError

from maverick.backtesting.types import (
    BacktestAnalysis,
    BacktestMetrics,
    BacktestResult,
    EnsembleBacktestResult,
    EnsembleIndividualResult,
    EnsembleMemberResult,
    EnsembleSummary,
    MarketRegimeAnalysis,
    MLBacktestResult,
    MLTrainingResult,
    MonteCarloResult,
    OptimizationResult,
    OptimizationResultRow,
    PortfolioBacktestMetrics,
    PortfolioBacktestResult,
    RegimeHistoryEntry,
    RiskAssessment,
    RunBacktestResult,
    SimpleBacktestMetrics,
    StrategyCatalog,
    StrategyCatalogEntry,
    StrategyComparisonResult,
    StrategyComparisonRow,
    TradeQuality,
    TradeRecord,
    WalkForwardPeriodResult,
    WalkForwardResult,
)

# -- BacktestMetrics / TradeRecord / SimpleBacktestMetrics ------------------


def _make_metrics(**overrides) -> BacktestMetrics:
    fields = {
        "total_return": 0.35,
        "annual_return": 0.28,
        "sharpe_ratio": 1.42,
        "sortino_ratio": 1.85,
        "calmar_ratio": 1.1,
        "max_drawdown": -0.18,
        "win_rate": 0.55,
        "profit_factor": 1.6,
        "expectancy": 12.5,
        "total_trades": 42,
        "winning_trades": 23,
        "losing_trades": 19,
        "avg_win": 150.0,
        "avg_loss": -90.0,
        "best_trade": 800.0,
        "worst_trade": -300.0,
        "avg_duration": 5.5,
        "kelly_criterion": 0.12,
        "recovery_factor": 2.1,
        "risk_reward_ratio": 1.7,
    }
    fields.update(overrides)
    return BacktestMetrics(**fields)


def test_backtest_metrics_round_trips_and_has_exact_fields():
    metrics = _make_metrics()
    data = metrics.model_dump()
    assert set(data) == {
        "total_return",
        "annual_return",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "max_drawdown",
        "win_rate",
        "profit_factor",
        "expectancy",
        "total_trades",
        "winning_trades",
        "losing_trades",
        "avg_win",
        "avg_loss",
        "best_trade",
        "worst_trade",
        "avg_duration",
        "kelly_criterion",
        "recovery_factor",
        "risk_reward_ratio",
    }
    assert BacktestMetrics(**data) == metrics


def test_backtest_metrics_never_optional_no_none_allowed():
    with pytest.raises(ValidationError):
        _make_metrics(sharpe_ratio=None)


def _make_trade(**overrides) -> TradeRecord:
    fields = {
        "entry_date": "2024-01-05",
        "exit_date": "2024-01-12",
        "entry_price": 100.0,
        "exit_price": 110.0,
        "size": 10.0,
        "pnl": 100.0,
        "return": 0.10,
        "duration": "7 days",
    }
    fields.update(overrides)
    return TradeRecord(**fields)


def test_trade_record_round_trips_with_return_alias():
    trade = _make_trade()
    data = trade.model_dump()
    assert data["return"] == 0.10
    assert "return_" not in data
    assert TradeRecord(**data) == trade
    assert trade.return_ == 0.10


def _make_simple_metrics(**overrides) -> SimpleBacktestMetrics:
    fields = {
        "total_return": 0.12,
        "annual_return": 0.09,
        "sharpe_ratio": 0.8,
        "max_drawdown": -0.1,
        "win_rate": 0.5,
        "total_trades": 8,
        "profit_factor": 1.2,
    }
    fields.update(overrides)
    return SimpleBacktestMetrics(**fields)


def test_simple_backtest_metrics_round_trips_and_has_exact_fields():
    metrics = _make_simple_metrics()
    data = metrics.model_dump()
    assert set(data) == {
        "total_return",
        "annual_return",
        "sharpe_ratio",
        "max_drawdown",
        "win_rate",
        "total_trades",
        "profit_factor",
    }
    assert SimpleBacktestMetrics(**data) == metrics


# -- BacktestResult / RunBacktestResult ------------------------------------


def _make_backtest_result(**overrides) -> BacktestResult:
    fields = {
        "symbol": "AAPL",
        "strategy": "sma_cross",
        "parameters": {"fast_period": 10, "slow_period": 20},
        "metrics": _make_metrics(),
        "trades": [_make_trade()],
        "equity_curve": {"1704412800000000000": 10000.0, "1704499200000000000": 10100.0},
        "drawdown_series": {"1704412800000000000": 0.0, "1704499200000000000": -0.01},
        "start_date": "2023-01-01",
        "end_date": "2024-01-01",
        "initial_capital": 10000.0,
    }
    fields.update(overrides)
    return BacktestResult(**fields)


def test_backtest_result_round_trips_through_model_dump():
    result = _make_backtest_result()
    data = result.model_dump()
    assert data["symbol"] == "AAPL"
    assert data["equity_curve"] == {
        "1704412800000000000": 10000.0,
        "1704499200000000000": 10100.0,
    }
    assert BacktestResult(**data) == result


def test_backtest_result_memory_stats_defaults_to_none():
    result = _make_backtest_result()
    assert result.memory_stats is None


def test_backtest_result_memory_stats_accepts_dict():
    result = _make_backtest_result(memory_stats={"current_memory_mb": 120.5})
    assert result.memory_stats == {"current_memory_mb": 120.5}


def _make_risk_assessment(**overrides) -> RiskAssessment:
    fields = {
        "risk_level": "Low-Medium",
        "max_drawdown": 0.18,
        "sortino_ratio": 1.85,
        "calmar_ratio": 1.1,
        "recovery_factor": 2.1,
        "risk_adjusted_return": 1.85,
        "downside_protection": "Good",
    }
    fields.update(overrides)
    return RiskAssessment(**fields)


def _make_trade_quality(**overrides) -> TradeQuality:
    fields = {
        "quality": "Good",
        "total_trades": 42,
        "frequency": "Moderate",
        "win_rate": 0.55,
        "avg_win": 150.0,
        "avg_loss": -90.0,
        "best_trade": 800.0,
        "worst_trade": -300.0,
        "avg_duration_days": 5.5,
        "risk_reward_ratio": 1.7,
    }
    fields.update(overrides)
    return TradeQuality(**fields)


def test_trade_quality_no_trades_case_omits_fields_to_none():
    quality = TradeQuality(quality="No trades", total_trades=0, frequency="None")
    assert quality.win_rate is None
    assert quality.avg_win is None
    assert quality.risk_reward_ratio is None


def test_trade_quality_round_trips_through_model_dump():
    quality = _make_trade_quality()
    data = quality.model_dump()
    assert TradeQuality(**data) == quality


def _make_analysis(**overrides) -> BacktestAnalysis:
    fields = {
        "performance_grade": "B",
        "risk_assessment": _make_risk_assessment(),
        "trade_quality": _make_trade_quality(),
        "strengths": ["Strong profit factor"],
        "weaknesses": ["Room for optimization"],
        "recommendations": ["Consider position sizing based on volatility"],
        "summary": "The strategy generated a 35.0% return with a Sharpe ratio of 1.42.",
    }
    fields.update(overrides)
    return BacktestAnalysis(**fields)


def test_backtest_analysis_round_trips_through_model_dump():
    analysis = _make_analysis()
    data = analysis.model_dump()
    restored = BacktestAnalysis.model_validate(data)
    assert restored == analysis
    assert restored.risk_assessment == analysis.risk_assessment
    assert restored.trade_quality == analysis.trade_quality


def test_run_backtest_result_composes_backtest_result_and_analysis():
    result = RunBacktestResult(
        **_make_backtest_result().model_dump(),
        analysis=_make_analysis(),
    )
    data = result.model_dump()
    assert data["symbol"] == "AAPL"
    assert data["analysis"]["performance_grade"] == "B"
    restored = RunBacktestResult.model_validate(data)
    assert restored == result


# -- OptimizationResult -----------------------------------------------------


def test_optimization_result_row_preserves_dynamic_metric_key():
    row = OptimizationResultRow(
        parameters={"fast_period": 10, "slow_period": 20},
        total_return=0.25,
        max_drawdown=-0.12,
        total_trades=15,
        sharpe_ratio=1.3,
    )
    data = row.model_dump()
    assert data["sharpe_ratio"] == 1.3
    restored = OptimizationResultRow(**data)
    assert restored == row
    assert restored.model_dump()["sharpe_ratio"] == 1.3


def test_optimization_result_round_trips_through_model_dump():
    row = OptimizationResultRow(
        parameters={"fast_period": 10, "slow_period": 20},
        total_return=0.25,
        max_drawdown=-0.12,
        total_trades=15,
        sharpe_ratio=1.3,
    )
    result = OptimizationResult(
        symbol="AAPL",
        strategy="sma_cross",
        optimization_metric="sharpe_ratio",
        best_parameters={"fast_period": 10, "slow_period": 20},
        best_metric_value=1.3,
        top_results=[row],
        total_combinations_tested=100,
        valid_combinations=97,
    )
    data = result.model_dump()
    assert OptimizationResult(**data) == result
    assert result.memory_stats is None


# -- WalkForwardResult -------------------------------------------------------


def test_walk_forward_result_round_trips_through_model_dump():
    period = WalkForwardPeriodResult(
        period="2023-01-01 to 2023-04-01",
        parameters={"fast_period": 10, "slow_period": 20},
        in_sample_sharpe=1.2,
        out_sample_return=0.08,
        out_sample_sharpe=0.95,
        out_sample_drawdown=-0.1,
    )
    result = WalkForwardResult(
        symbol="AAPL",
        strategy="sma_cross",
        periods_tested=4,
        average_return=0.07,
        average_sharpe=1.0,
        average_drawdown=-0.11,
        consistency=0.75,
        walk_forward_results=[period],
        summary="Walk-forward analysis shows 7.0% average return",
    )
    data = result.model_dump()
    assert WalkForwardResult(**data) == result


# -- MonteCarloResult --------------------------------------------------------


def test_monte_carlo_result_round_trips_through_model_dump():
    result = MonteCarloResult(
        num_simulations=1000,
        expected_return=0.15,
        return_std=0.08,
        return_percentiles={"p5": -0.05, "p25": 0.08, "p50": 0.14, "p75": 0.22, "p95": 0.35},
        expected_drawdown=-0.12,
        drawdown_std=0.05,
        drawdown_percentiles={"p5": -0.25, "p25": -0.16, "p50": -0.11, "p75": -0.07, "p95": -0.02},
        probability_profit=0.82,
        var_95=-0.05,
        summary="Monte Carlo simulation shows 15.0% expected return",
    )
    data = result.model_dump()
    assert data["return_percentiles"]["p50"] == 0.14
    assert MonteCarloResult(**data) == result


# -- StrategyComparisonResult -------------------------------------------------


def _make_comparison_row(**overrides) -> StrategyComparisonRow:
    fields = {
        "strategy": "sma_cross",
        "parameters": {"fast_period": 10, "slow_period": 20},
        "total_return": 0.25,
        "sharpe_ratio": 1.3,
        "max_drawdown": 0.12,
        "win_rate": 0.55,
        "profit_factor": 1.6,
        "total_trades": 20,
        "grade": "B",
        "rank": 1,
    }
    fields.update(overrides)
    return StrategyComparisonRow(**fields)


def test_strategy_comparison_result_round_trips_and_allows_none_best_overall():
    row = _make_comparison_row()
    result = StrategyComparisonResult(
        rankings=[row],
        best_overall=None,
        best_return=row,
        best_sharpe=row,
        best_drawdown=row,
        best_win_rate=row,
        summary="The best performing strategy is sma_cross",
    )
    data = result.model_dump()
    assert data["best_overall"] is None
    assert StrategyComparisonResult(**data) == result


# -- StrategyCatalog ----------------------------------------------------------


def test_strategy_catalog_round_trips_through_model_dump():
    entry = StrategyCatalogEntry(
        type="sma_cross",
        name="SMA Crossover",
        description="Buy when fast SMA crosses above slow SMA",
        default_parameters={"fast_period": 10, "slow_period": 20},
        optimization_ranges={"fast_period": [5, 10, 15, 20]},
    )
    catalog = StrategyCatalog(
        available_strategies={"sma_cross": entry},
        total_count=1,
        categories={"trend_following": ["sma_cross"]},
    )
    data = catalog.model_dump()
    assert StrategyCatalog(**data) == catalog


# -- PortfolioBacktestResult --------------------------------------------------


def test_portfolio_backtest_result_round_trips_through_model_dump():
    metrics = PortfolioBacktestMetrics(
        symbols_tested=2,
        total_return=0.2,
        average_sharpe=1.1,
        max_drawdown=0.15,
        total_trades=30,
    )
    result = PortfolioBacktestResult(
        portfolio_metrics=metrics,
        individual_results=[_make_backtest_result()],
        summary="Portfolio backtest of 2 symbols with sma_cross strategy",
    )
    data = result.model_dump()
    assert PortfolioBacktestResult(**data) == result


# -- ML / regime / ensemble results -------------------------------------------


def test_ml_backtest_result_round_trips_through_model_dump():
    result = MLBacktestResult(
        metrics=_make_simple_metrics(),
        trades=[{"entry_time": 0, "exit_time": 5, "pnl": 12.5, "return": 0.05}],
        equity_curve={"0": 10000.0, "1": 10100.0},
        drawdown_series={"0": 0.0, "1": -0.01},
        ml_metrics={
            "strategy_type": "ml_predictor",
            "training_period": 400,
            "testing_period": 100,
            "train_test_split": 0.8,
            "training_metrics": {"train_accuracy": 0.62},
        },
    )
    data = result.model_dump()
    assert MLBacktestResult(**data) == result


def test_ml_backtest_result_trades_tolerate_fallback_shape():
    result = MLBacktestResult(
        metrics=_make_simple_metrics(),
        trades=[{"total_trades": 3, "message": "Detailed trade data not available"}],
        equity_curve={},
        drawdown_series={},
        ml_metrics={},
    )
    assert result.trades == [
        {"total_trades": 3, "message": "Detailed trade data not available"}
    ]


def test_ml_training_result_round_trips_through_model_dump():
    result = MLTrainingResult(
        symbol="AAPL",
        model_type="random_forest",
        training_period="2022-01-01 to 2024-01-01",
        data_points=500,
        target_periods=5,
        return_threshold=0.02,
        model_parameters={"n_estimators": 100, "max_depth": None, "min_samples_split": 2},
        training_metrics={"train_accuracy": 0.61, "n_samples": 500, "n_features": 12},
    )
    data = result.model_dump()
    assert MLTrainingResult(**data) == result
    assert data["model_parameters"]["max_depth"] is None


def test_market_regime_analysis_round_trips_with_int_keyed_dicts():
    entry = RegimeHistoryEntry(date="2024-01-05", regime=2, probabilities=[0.1, 0.2, 0.7])
    result = MarketRegimeAnalysis(
        symbol="AAPL",
        analysis_period="2023-01-01 to 2024-01-01",
        method="hmm",
        n_regimes=3,
        regime_names={0: "Bear/Declining", 1: "Sideways/Uncertain", 2: "Bull/Trending"},
        current_regime=2,
        regime_counts={0: 40, 1: 60, 2: 100},
        regime_percentages={0: 20.0, 1: 30.0, 2: 50.0},
        average_regime_durations={0: 5.5, 1: 8.2, 2: 12.1},
        recent_regime_history=[entry],
        total_regime_switches=12,
    )
    data = result.model_dump()
    assert MarketRegimeAnalysis(**data) == result
    assert result.regime_names[2] == "Bull/Trending"


def test_ensemble_backtest_result_round_trips_through_model_dump():
    member = EnsembleMemberResult(
        metrics=_make_simple_metrics(),
        trades=[{"entry_time": 0, "exit_time": 5, "pnl": 12.5, "return": 0.05}],
        equity_curve={"0": 10000.0},
        drawdown_series={"0": 0.0},
        ensemble_metrics={
            "strategy_weights": {"sma_cross": 0.5, "rsi": 0.5},
            "strategy_performance": {"sma_cross": 0.1, "rsi": 0.05},
        },
    )
    individual = EnsembleIndividualResult(symbol="AAPL", results=member)
    summary = EnsembleSummary(
        symbols_tested=1,
        base_strategies=["sma_cross", "rsi", "macd"],
        weighting_method="performance",
        average_return=0.1,
        total_trades=10,
        average_trades_per_symbol=10.0,
    )
    result = EnsembleBacktestResult(
        ensemble_summary=summary,
        individual_results=[individual],
        final_strategy_weights={"sma_cross": 0.5, "rsi": 0.5},
        strategy_performance_analysis={"sma_cross": 0.1, "rsi": 0.05},
    )
    data = result.model_dump()
    assert EnsembleBacktestResult(**data) == result
    assert data["individual_results"][0]["symbol"] == "AAPL"
