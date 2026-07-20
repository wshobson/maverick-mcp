"""Smoke tests for maverick.backtesting's public API surface (`__init__.py`).

Two paths are proven, per the phase's availability contract:

- **Extra installed** (this dev environment has vectorbt/scikit-learn via
  `--extra backtesting`): every name in `__all__` -- both the always-safe
  base layer and the lazily-resolved extra-only layer -- is importable from
  the package root, mirroring how `maverick.technical`'s close-out (phase 5)
  exported its public surface via `__init__.py`.
- **Extra absent, simulated**: `monkeypatch.setattr(backtesting, "backtesting_extra_available", ...)`
  the same way `tests/backtesting/test_tools_availability.py` monkeypatches
  `tools._backtesting_extra_available` -- proving `import maverick.backtesting`
  and access to the base-layer names never depend on vectorbt/scikit-learn
  actually being installed, and that touching an extra-only name fails with
  one clear `ImportError` naming the extra rather than an opaque upstream
  trace or a bare `AttributeError`.
"""

import pytest

import maverick.backtesting as backtesting


def test_import_types_from_package():
    from pydantic import BaseModel

    from maverick.backtesting import (
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

    for model in (
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
    ):
        assert issubclass(model, BaseModel)


def test_import_config_from_package():
    from maverick.backtesting import (
        BacktestingSettings,
        get_backtesting_settings,
        reset_backtesting_settings,
    )

    assert callable(get_backtesting_settings)
    assert callable(reset_backtesting_settings)
    assert issubclass(BacktestingSettings, object)
    assert isinstance(get_backtesting_settings(), BacktestingSettings)


def test_import_strategy_catalog_from_package():
    from maverick.backtesting import (
        STRATEGY_TEMPLATES,
        Strategy,
        get_strategy_info,
        get_strategy_template,
        list_available_strategies,
    )

    assert len(STRATEGY_TEMPLATES) == 12
    assert isinstance(Strategy, type)
    assert callable(get_strategy_info)
    assert callable(get_strategy_template)
    assert callable(list_available_strategies)


def test_import_pure_analysis_functions_from_package():
    from maverick.backtesting import (
        analyze,
        compare_strategies,
        generate_param_grid,
        monte_carlo_simulation,
    )

    for fn in (
        analyze,
        compare_strategies,
        generate_param_grid,
        monte_carlo_simulation,
    ):
        assert callable(fn)


def test_import_tool_wiring_from_package():
    from maverick.backtesting import backtesting_extra_available, configure, register

    assert callable(backtesting_extra_available)
    assert callable(configure)
    assert callable(register)


def test_import_extra_only_members_from_package():
    """Proves the lazy layer resolves when the extra IS installed (true in this
    dev environment -- CI installs `--extra backtesting`)."""
    from maverick.backtesting import (
        AdaptiveStrategy,
        BacktestingService,
        FeatureExtractor,
        MLPredictor,
        RegimeAwareStrategy,
        StrategyEnsemble,
        generate_signals,
        optimize_parameters,
        run_backtest,
    )

    assert callable(BacktestingService)
    assert callable(run_backtest)
    assert callable(optimize_parameters)
    assert callable(generate_signals)
    for cls in (
        AdaptiveStrategy,
        FeatureExtractor,
        MLPredictor,
        RegimeAwareStrategy,
        StrategyEnsemble,
    ):
        assert isinstance(cls, type)


def test_all_matches_expected_export_set():
    assert set(backtesting.__all__) == {
        # types
        "BacktestAnalysis",
        "BacktestMetrics",
        "BacktestResult",
        "EnsembleBacktestResult",
        "EnsembleIndividualResult",
        "EnsembleMemberResult",
        "EnsembleSummary",
        "MarketRegimeAnalysis",
        "MLBacktestResult",
        "MLTrainingResult",
        "MonteCarloResult",
        "OptimizationResult",
        "OptimizationResultRow",
        "PortfolioBacktestMetrics",
        "PortfolioBacktestResult",
        "RegimeHistoryEntry",
        "RiskAssessment",
        "RunBacktestResult",
        "SimpleBacktestMetrics",
        "StrategyCatalog",
        "StrategyCatalogEntry",
        "StrategyComparisonResult",
        "StrategyComparisonRow",
        "TradeQuality",
        "TradeRecord",
        "WalkForwardPeriodResult",
        "WalkForwardResult",
        # config
        "BacktestingSettings",
        "get_backtesting_settings",
        "reset_backtesting_settings",
        # strategies (always importable)
        "Strategy",
        "STRATEGY_TEMPLATES",
        "get_strategy_info",
        "get_strategy_template",
        "list_available_strategies",
        # pure analysis/optimization functions
        "analyze",
        "compare_strategies",
        "generate_param_grid",
        "monte_carlo_simulation",
        # tool wiring
        "backtesting_extra_available",
        "configure",
        "register",
        # extra-only (lazy)
        "AdaptiveStrategy",
        "BacktestingService",
        "FeatureExtractor",
        "MLPredictor",
        "RegimeAwareStrategy",
        "StrategyEnsemble",
        "generate_signals",
        "optimize_parameters",
        "run_backtest",
    }
    assert len(backtesting.__all__) == len(set(backtesting.__all__)), (
        "__all__ has duplicate entries"
    )


def test_every_exported_name_resolves_on_the_package():
    """With the extra installed, every __all__ name -- including the lazy
    ones -- must resolve via __getattr__."""
    for name in backtesting.__all__:
        assert hasattr(backtesting, name), (
            f"{name!r} listed in __all__ but not resolvable on the package"
        )


def test_getattr_rejects_unknown_name():
    with pytest.raises(AttributeError):
        _ = backtesting.this_name_does_not_exist


# -- Simulated-absent path: extra is NOT installed --------------------------


def test_import_succeeds_and_base_layer_accessible_with_extra_simulated_absent(
    monkeypatch,
):
    """`import maverick.backtesting` never touches vectorbt/scikit-learn (the
    import already happened at module-collection time above -- this test
    proves the base layer stays fully usable even when the availability
    guard reports the extra missing)."""
    monkeypatch.setattr(backtesting, "backtesting_extra_available", lambda: False)

    assert backtesting.STRATEGY_TEMPLATES is not None
    assert len(backtesting.STRATEGY_TEMPLATES) == 12
    settings = backtesting.get_backtesting_settings()
    assert settings.initial_capital == 10000.0
    assert callable(backtesting.configure)
    assert callable(backtesting.register)
    assert issubclass(backtesting.BacktestResult, object)


def test_extra_only_attribute_raises_clear_import_error_when_simulated_absent(
    monkeypatch,
):
    monkeypatch.setattr(backtesting, "backtesting_extra_available", lambda: False)

    with pytest.raises(ImportError, match="\\[backtesting\\]"):
        _ = backtesting.BacktestingService

    with pytest.raises(ImportError, match="\\[backtesting\\]"):
        _ = backtesting.run_backtest

    with pytest.raises(ImportError, match="\\[backtesting\\]"):
        _ = backtesting.AdaptiveStrategy
