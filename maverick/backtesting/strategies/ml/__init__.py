"""ML strategy implementations: feature engineering, adaptive, regime-aware, ensemble. Third-layer sibling: imports config and types.

Ported from `maverick_mcp/backtesting/strategies/ml/__init__.py`, which
re-exported five names backed by four submodules
(`feature_engineering`/`adaptive`/`regime_aware`/`ensemble`). This package
still has exactly those five names at the top level, but several of the
legacy submodules were each too large to survive the port as one file
(over this repo's 500-line-per-module cap) and were split further by
responsibility -- see the Task 6 report for the full rationale per split:

- `feature_engineering.py` (`FeatureExtractor`) / `ml_predictor.py`
  (`MLPredictor`, split out of the legacy `feature_engineering.py`).
- `adaptive.py` (`AdaptiveStrategy`) / `online_learning.py`
  (`OnlineLearningStrategy`) / `hybrid_adaptive.py`
  (`HybridAdaptiveStrategy`), all split out of the legacy `adaptive.py`.
- `regime_features.py` (module-level `extract_regime_features`, pulled out
  of `MarketRegimeDetector` because it never referenced `self`) /
  `regime_detector.py` (`MarketRegimeDetector`) / `regime_aware.py`
  (`RegimeAwareStrategy`), all split out of the legacy `regime_aware.py`.
- `ensemble.py` (`StrategyEnsemble`) needed no split.

Two legacy classes are not ported at all -- `RiskAdjustedEnsemble` and
`AdaptiveRegimeStrategy` were dead code with zero callers anywhere outside
their own unused `__all__` export; see the Task 6 report.
"""

from .adaptive import AdaptiveStrategy
from .ensemble import StrategyEnsemble
from .feature_engineering import FeatureExtractor
from .ml_predictor import MLPredictor
from .regime_aware import RegimeAwareStrategy

__all__ = [
    "AdaptiveStrategy",
    "FeatureExtractor",
    "MLPredictor",
    "RegimeAwareStrategy",
    "StrategyEnsemble",
]
