"""Hybrid strategy combining parameter adaptation with online learning.

Split out of `adaptive.py` (see the Task 6 report). Composes
`AdaptiveStrategy` (subclassed) and `OnlineLearningStrategy` (owned
instance) -- both live in their own modules now, so this one imports both.
No behavior changed by moving this class here.
"""

from typing import Any

from pandas import DataFrame, Series

from maverick.backtesting.strategies.base import Strategy

from .adaptive import AdaptiveStrategy
from .online_learning import OnlineLearningStrategy


class HybridAdaptiveStrategy(AdaptiveStrategy):
    """Hybrid strategy combining parameter adaptation with online learning."""

    def __init__(
        self, base_strategy: Strategy, online_learning_weight: float = 0.3, **kwargs
    ):
        """Initialize hybrid adaptive strategy.

        Args:
            base_strategy: Base strategy to adapt
            online_learning_weight: Weight for online learning component
            **kwargs: Additional parameters for AdaptiveStrategy
        """
        super().__init__(base_strategy, **kwargs)
        self.online_learning_weight = online_learning_weight
        self.online_strategy = OnlineLearningStrategy()

    @property
    def name(self) -> str:
        """Get strategy name."""
        return f"HybridAdaptive({self.base_strategy.name})"

    @property
    def description(self) -> str:
        """Get strategy description."""
        return "Hybrid adaptive strategy combining parameter adaptation with online learning"

    def generate_signals(self, data: DataFrame) -> tuple[Series, Series]:
        """Generate hybrid adaptive signals.

        Args:
            data: Price data with OHLCV columns

        Returns:
            Tuple of (entry_signals, exit_signals) as boolean Series
        """
        # Get signals from adaptive base strategy
        adaptive_entry, adaptive_exit = super().generate_signals(data)

        # Get signals from online learning component
        online_entry, online_exit = self.online_strategy.generate_signals(data)

        # Combine signals with weighting
        base_weight = 1.0 - self.online_learning_weight

        # Weighted combination for entry signals
        combined_entry = (
            adaptive_entry.astype(float) * base_weight
            + online_entry.astype(float) * self.online_learning_weight
        ) > 0.5

        # Weighted combination for exit signals
        combined_exit = (
            adaptive_exit.astype(float) * base_weight
            + online_exit.astype(float) * self.online_learning_weight
        ) > 0.5

        return combined_entry, combined_exit

    def get_hybrid_info(self) -> dict[str, Any]:
        """Get information about hybrid strategy components.

        Returns:
            Dictionary with hybrid strategy information
        """
        return {
            "adaptation_history": self.get_adaptation_history(),
            "online_learning_info": self.online_strategy.get_model_info(),
            "online_learning_weight": self.online_learning_weight,
            "base_weight": 1.0 - self.online_learning_weight,
        }
