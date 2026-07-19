"""Adaptive trading strategies: base parameter-adaptation strategy.

Ported from `maverick_mcp/backtesting/strategies/ml/adaptive.py`, which also
held `OnlineLearningStrategy` (now `online_learning.py`) and
`HybridAdaptiveStrategy` (now `hybrid_adaptive.py`) -- split out to stay
under this repo's 500-line-per-module cap; see the Task 6 report. No dead
code identified in any of the three: all three classes are exercised by
legacy `tests/test_ml_strategies.py` and re-exported from the package's
public `__all__`.

One new seam, disclosed per the Task 6 determinism rule: `AdaptiveStrategy`
gained a trailing `random_state: int | None = None` constructor parameter.
Legacy `adapt_parameters_random_search` calls the *global* `np.random.normal`
with no way to inject a seed -- the only unseeded randomness source in the
adaptive.py split (the `SGDClassifier` in `OnlineLearningStrategy` already
hardcodes `random_state=42`, so it is deterministic by default with no seam
needed). When `random_state` is `None` (the default), behavior is
byte-for-byte identical to legacy: the method still draws from
`np.random.normal`. When a seed is supplied, perturbations are drawn from a
local `np.random.default_rng(random_state)` instead, so characterization
tests can pin `adapt_parameters_random_search` without touching global numpy
state.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from maverick.backtesting.strategies.base import Strategy

logger = logging.getLogger(__name__)


class AdaptiveStrategy(Strategy):
    """Base class for adaptive strategies that adjust parameters based on performance."""

    def __init__(
        self,
        base_strategy: Strategy,
        adaptation_method: str = "gradient",
        learning_rate: float = 0.01,
        lookback_period: int = 50,
        adaptation_frequency: int = 10,
        parameters: dict[str, Any] | None = None,
        random_state: int | None = None,
    ):
        """Initialize adaptive strategy.

        Args:
            base_strategy: Base strategy to adapt
            adaptation_method: Method for parameter adaptation
            learning_rate: Learning rate for parameter updates
            lookback_period: Period for performance evaluation
            adaptation_frequency: How often to adapt parameters
            parameters: Additional parameters
            random_state: Optional seed for `adapt_parameters_random_search`'s
                perturbations. `None` (default) preserves legacy behavior --
                draws from the global `np.random` state.
        """
        super().__init__(parameters)
        self.base_strategy = base_strategy
        self.adaptation_method = adaptation_method
        self.learning_rate = learning_rate
        self.lookback_period = lookback_period
        self.adaptation_frequency = adaptation_frequency
        self.random_state = random_state
        self._rng = (
            np.random.default_rng(random_state) if random_state is not None else None
        )

        # Performance tracking
        self.performance_history = []
        self.parameter_history = []
        self.last_adaptation = 0

        # Store original parameters for reference
        self.original_parameters = base_strategy.parameters.copy()

    @property
    def name(self) -> str:
        """Get strategy name."""
        return f"Adaptive({self.base_strategy.name})"

    @property
    def description(self) -> str:
        """Get strategy description."""
        return f"Adaptive version of {self.base_strategy.name} using {self.adaptation_method} method"

    def calculate_performance_metric(self, returns: Series) -> float:
        """Calculate performance metric for parameter adaptation.

        Args:
            returns: Strategy returns

        Returns:
            Performance score
        """
        if len(returns) == 0:
            return 0.0

        # Use Sharpe ratio as default performance metric
        if returns.std() == 0:
            return 0.0

        sharpe = returns.mean() / returns.std() * np.sqrt(252)
        return sharpe

    def adapt_parameters_gradient(self, performance_gradient: float) -> None:
        """Adapt parameters using gradient-based method.

        Args:
            performance_gradient: Gradient of performance with respect to parameters
        """
        adaptable_params = self.get_adaptable_parameters()

        for param_name, param_info in adaptable_params.items():
            if param_name in self.base_strategy.parameters:
                current_value = self.base_strategy.parameters[param_name]

                # Calculate parameter update
                param_gradient = performance_gradient * self.learning_rate

                # Apply bounds and constraints
                min_val = param_info.get("min", current_value * 0.5)
                max_val = param_info.get("max", current_value * 2.0)
                step_size = param_info.get("step", abs(current_value) * 0.01)

                # Update parameter
                new_value = current_value + param_gradient * step_size
                new_value = max(min_val, min(max_val, new_value))

                self.base_strategy.parameters[param_name] = new_value

                logger.debug(
                    f"Adapted {param_name}: {current_value:.4f} -> {new_value:.4f}"
                )

    def adapt_parameters_random_search(self) -> None:
        """Adapt parameters using random search with performance feedback."""
        adaptable_params = self.get_adaptable_parameters()
        normal = self._rng.normal if self._rng is not None else np.random.normal

        # Try random perturbations and keep improvements
        for param_name, param_info in adaptable_params.items():
            if param_name in self.base_strategy.parameters:
                current_value = self.base_strategy.parameters[param_name]

                # Generate random perturbation
                min_val = param_info.get("min", current_value * 0.5)
                max_val = param_info.get("max", current_value * 2.0)

                # Small random step
                perturbation = normal(0, abs(current_value) * 0.1)
                new_value = current_value + perturbation
                new_value = max(min_val, min(max_val, new_value))

                # Store new value for trial
                self.base_strategy.parameters[param_name] = new_value

                # Note: Performance evaluation would happen in next cycle
                # For now, keep the change and let performance tracking decide

    def get_adaptable_parameters(self) -> dict[str, dict]:
        """Get parameters that can be adapted.

        Returns:
            Dictionary of adaptable parameters with their constraints
        """
        # Default adaptable parameters - can be overridden by subclasses
        return {
            "lookback_period": {"min": 5, "max": 200, "step": 1},
            "threshold": {"min": 0.001, "max": 0.1, "step": 0.001},
            "window": {"min": 5, "max": 100, "step": 1},
            "period": {"min": 5, "max": 200, "step": 1},
        }

    def adapt_parameters(self, recent_performance: float) -> None:
        """Adapt strategy parameters based on recent performance.

        Args:
            recent_performance: Recent performance metric
        """
        try:
            if self.adaptation_method == "gradient":
                # Approximate gradient based on performance change
                if len(self.performance_history) > 1:
                    performance_gradient = (
                        recent_performance - self.performance_history[-2]
                    )
                    self.adapt_parameters_gradient(performance_gradient)

            elif self.adaptation_method == "random_search":
                # Use random search with performance feedback
                self.adapt_parameters_random_search()

            else:
                logger.warning(f"Unknown adaptation method: {self.adaptation_method}")

            # Store adapted parameters
            self.parameter_history.append(self.base_strategy.parameters.copy())

        except Exception as e:
            logger.error(f"Error adapting parameters: {e}")

    def generate_signals(self, data: DataFrame) -> tuple[Series, Series]:
        """Generate adaptive trading signals.

        Args:
            data: Price data with OHLCV columns

        Returns:
            Tuple of (entry_signals, exit_signals) as boolean Series
        """
        try:
            # Generate signals from base strategy
            entry_signals, exit_signals = self.base_strategy.generate_signals(data)

            # Calculate strategy performance for adaptation
            positions = entry_signals.astype(int) - exit_signals.astype(int)
            returns = positions.shift(1) * data["close"].pct_change()

            # Track performance over time for adaptation
            for idx in range(
                self.adaptation_frequency, len(data), self.adaptation_frequency
            ):
                if idx > self.last_adaptation + self.adaptation_frequency:
                    # Calculate recent performance
                    recent_returns = returns.iloc[
                        max(0, idx - self.lookback_period) : idx
                    ]
                    if len(recent_returns) > 0:
                        recent_performance = self.calculate_performance_metric(
                            recent_returns
                        )
                        self.performance_history.append(recent_performance)

                        # Adapt parameters based on performance
                        self.adapt_parameters(recent_performance)

                        # Re-generate signals with adapted parameters
                        entry_signals, exit_signals = (
                            self.base_strategy.generate_signals(data)
                        )

                    self.last_adaptation = idx

            return entry_signals, exit_signals

        except Exception as e:
            logger.error(f"Error generating adaptive signals: {e}")
            return pd.Series(False, index=data.index), pd.Series(
                False, index=data.index
            )

    def get_adaptation_history(self) -> dict[str, Any]:
        """Get history of parameter adaptations.

        Returns:
            Dictionary with adaptation history
        """
        return {
            "performance_history": self.performance_history,
            "parameter_history": self.parameter_history,
            "current_parameters": self.base_strategy.parameters,
            "original_parameters": self.original_parameters,
        }

    def reset_to_original(self) -> None:
        """Reset strategy parameters to original values."""
        self.base_strategy.parameters = self.original_parameters.copy()
        self.performance_history = []
        self.parameter_history = []
        self.last_adaptation = 0
