"""ML strategies bridge module for easier imports."""

from .ml.adaptive import AdaptiveStrategy as OnlineLearningStrategy
from .ml.regime_aware import RegimeAwareStrategy
from .ml.ensemble import StrategyEnsemble as EnsembleStrategy

__all__ = [
    "OnlineLearningStrategy",
    "RegimeAwareStrategy",
    "EnsembleStrategy"
]