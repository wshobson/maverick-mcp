"""Public API of the backtesting domain, gated behind the optional
`[backtesting]` extra (vectorbt, numba, scikit-learn, scipy, pandas-ta).

**The base-install contract.** `import maverick.backtesting` must always
succeed, with no extra installed, and never print a traceback -- mirroring
`tools.py`'s own availability guard (`tools_support.backtesting_extra_available`,
re-exported here as `backtesting_extra_available`). Two tiers of names live
on this package:

- **Always available** (imported eagerly below): `types.py`'s payload
  models, `config.py`'s settings accessor, `strategies.base.Strategy`, the
  12-entry rule-based `strategies.templates` catalog, the pure
  `analysis.py`/`optimization.py` functions, and `tools.configure`/
  `tools.register` -- none of these modules import vectorbt or scikit-learn,
  directly or transitively.
- **Extra-only** (resolved lazily via module `__getattr__`, PEP 562):
  `BacktestingService`, `engine.run_backtest`/`optimize_parameters`,
  `strategies.signals.generate_signals`, and 5 of the 8 ML strategy classes
  in `strategies.ml` (`AdaptiveStrategy`, `FeatureExtractor`, `MLPredictor`,
  `RegimeAwareStrategy`, `StrategyEnsemble` -- the five `strategies/ml/__init__.py`
  itself re-exports; the remaining three, `HybridAdaptiveStrategy`,
  `MarketRegimeDetector`, `OnlineLearningStrategy`, are reachable only via
  their own submodules). Accessing one of these attributes without the
  extra installed raises a clear `ImportError` naming the extra, rather
  than either succeeding silently or surfacing vectorbt's own confusing
  import trace.

Honest strategy count for docs and callers: 12 rule-based templates
(`STRATEGY_TEMPLATES`) + 8 ML strategy classes -- not "23" or "35+" (see
`docs/api/backtesting.md`).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from maverick.backtesting.analysis import (
    analyze,
    compare_strategies,
    monte_carlo_simulation,
)
from maverick.backtesting.config import (
    BacktestingSettings,
    get_backtesting_settings,
    reset_backtesting_settings,
)
from maverick.backtesting.optimization import generate_param_grid
from maverick.backtesting.strategies.base import Strategy
from maverick.backtesting.strategies.templates import (
    STRATEGY_TEMPLATES,
    get_strategy_info,
    get_strategy_template,
    list_available_strategies,
)
from maverick.backtesting.tools import configure, register
from maverick.backtesting.tools_support import backtesting_extra_available
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

if TYPE_CHECKING:
    # Extra-only members -- resolved lazily by __getattr__ below at runtime so
    # importing this package never touches vectorbt/scikit-learn.
    from maverick.backtesting.engine import optimize_parameters, run_backtest
    from maverick.backtesting.service import BacktestingService
    from maverick.backtesting.strategies.ml import (
        AdaptiveStrategy,
        FeatureExtractor,
        MLPredictor,
        RegimeAwareStrategy,
        StrategyEnsemble,
    )
    from maverick.backtesting.strategies.signals import generate_signals

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "BacktestingService": ("maverick.backtesting.service", "BacktestingService"),
    "run_backtest": ("maverick.backtesting.engine", "run_backtest"),
    "optimize_parameters": ("maverick.backtesting.engine", "optimize_parameters"),
    "generate_signals": (
        "maverick.backtesting.strategies.signals",
        "generate_signals",
    ),
    "AdaptiveStrategy": ("maverick.backtesting.strategies.ml", "AdaptiveStrategy"),
    "FeatureExtractor": ("maverick.backtesting.strategies.ml", "FeatureExtractor"),
    "MLPredictor": ("maverick.backtesting.strategies.ml", "MLPredictor"),
    "RegimeAwareStrategy": (
        "maverick.backtesting.strategies.ml",
        "RegimeAwareStrategy",
    ),
    "StrategyEnsemble": ("maverick.backtesting.strategies.ml", "StrategyEnsemble"),
}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy resolution for the extra-only names in `_LAZY_EXPORTS`.

    Checks the same availability guard `tools.register` uses before ever
    importing vectorbt/scikit-learn, so a missing extra raises one clear
    `ImportError` here instead of either an opaque upstream import trace or
    (worse) a bare `AttributeError` that looks like a typo.
    """
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if not backtesting_extra_available():
        raise ImportError(
            f"maverick.backtesting.{name} requires the '[backtesting]' extra "
            "(vectorbt, scikit-learn, ...). Install with "
            "`uv sync --extra backtesting`."
        )
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
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
    # strategies (base + rule-based templates: always importable)
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
]
