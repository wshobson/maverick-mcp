"""Strategy modules for VectorBT backtesting."""

from .base import Strategy

# ML-enhanced strategies
from .ml import (
    AdaptiveStrategy,
    FeatureExtractor,
    MLPredictor,
    RegimeAwareStrategy,
    StrategyEnsemble,
)
from .ml.adaptive import HybridAdaptiveStrategy, OnlineLearningStrategy
from .ml.ensemble import RiskAdjustedEnsemble
from .ml.regime_aware import AdaptiveRegimeStrategy, MarketRegimeDetector
from .parser import StrategyParser
from .templates import STRATEGY_TEMPLATES

__all__ = [
    "Strategy",
    "StrategyParser",
    "STRATEGY_TEMPLATES",
    # ML strategies
    "AdaptiveStrategy",
    "FeatureExtractor",
    "MLPredictor",
    "RegimeAwareStrategy",
    "StrategyEnsemble",
    # Advanced ML strategies
    "OnlineLearningStrategy",
    "HybridAdaptiveStrategy",
    "RiskAdjustedEnsemble",
    "MarketRegimeDetector",
    "AdaptiveRegimeStrategy",
]
