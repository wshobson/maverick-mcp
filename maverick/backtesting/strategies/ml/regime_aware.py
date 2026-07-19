"""Regime-switching trading strategy: routes signals to a per-regime strategy.

Split out of `regime_aware.py`'s original form (see the Task 6 report): the
ported legacy module put `MarketRegimeDetector` and `RegimeAwareStrategy` in
one 899-line file, over this repo's 500-line-per-module cap.
`MarketRegimeDetector` now lives in `regime_detector.py`, imported here.

Dead code removed (logged in the Task 6 report): the legacy
`AdaptiveRegimeStrategy` subclass (previously defined right after this
class) had zero callers anywhere in `maverick_mcp` outside its own package's
`__all__` re-export -- no router, service, or test referenced it (`grep -rn
"AdaptiveRegimeStrategy" maverick_mcp tests` matches only the class
definition and the unused `strategies/__init__.py` export). This class
itself is live: `maverick_mcp/api/routers/backtesting.py` calls
`fit_regime_detector`, `get_regime_analysis`, and `get_regime_probabilities`
directly.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from maverick.backtesting.strategies.base import Strategy

from .regime_detector import MarketRegimeDetector

logger = logging.getLogger(__name__)


class RegimeAwareStrategy(Strategy):
    """Strategy that switches between different strategies based on market regime."""

    def __init__(
        self,
        regime_strategies: dict[int, Strategy],
        regime_detector: MarketRegimeDetector | None = None,
        regime_names: dict[int, str] | None = None,
        switch_threshold: float = 0.7,
        min_regime_duration: int = 5,
        parameters: dict[str, Any] | None = None,
    ):
        """Initialize regime-aware strategy.

        Args:
            regime_strategies: Dictionary mapping regime labels to strategies
            regime_detector: Market regime detector instance
            regime_names: Names for each regime
            switch_threshold: Probability threshold for regime switching
            min_regime_duration: Minimum duration before switching regimes
            parameters: Additional parameters
        """
        super().__init__(parameters)
        self.regime_strategies = regime_strategies
        self.regime_detector = regime_detector or MarketRegimeDetector()
        self.regime_names = regime_names or {0: "Bear", 1: "Sideways", 2: "Bull"}
        self.switch_threshold = switch_threshold
        self.min_regime_duration = min_regime_duration

        # Regime tracking
        self.current_regime = 1  # Start with sideways
        self.regime_history = []
        self.regime_duration = 0
        self.regime_switches = 0

    @property
    def name(self) -> str:
        """Get strategy name."""
        strategy_names = [s.name for s in self.regime_strategies.values()]
        return f"RegimeAware({','.join(strategy_names)})"

    @property
    def description(self) -> str:
        """Get strategy description."""
        return f"Regime-aware strategy switching between {len(self.regime_strategies)} strategies based on market conditions"

    def fit_regime_detector(self, data: DataFrame) -> None:
        """Fit regime detector to historical data.

        Args:
            data: Historical price data
        """
        self.regime_detector.fit_regimes(data)

    def update_current_regime(self, data: DataFrame, current_idx: int) -> bool:
        """Update current market regime.

        Args:
            data: Price data
            current_idx: Current index in data

        Returns:
            True if regime changed, False otherwise
        """
        # Get regime probabilities
        window_data = data.iloc[
            max(0, current_idx - self.regime_detector.lookback_period) : current_idx + 1
        ]
        regime_probs = self.regime_detector.get_regime_probabilities(window_data)

        # Find most likely regime (cast off numpy's intp so `current_regime`
        # stays a plain `int` -- it's used as a `dict[int, Strategy]` key)
        most_likely_regime = int(np.argmax(regime_probs))
        max_prob = regime_probs[most_likely_regime]

        # Check if we should switch regimes
        regime_changed = False

        if (
            most_likely_regime != self.current_regime
            and max_prob >= self.switch_threshold
            and self.regime_duration >= self.min_regime_duration
        ):
            old_regime = self.current_regime
            self.current_regime = most_likely_regime
            self.regime_duration = 0
            self.regime_switches += 1
            regime_changed = True

            old_name = self.regime_names.get(old_regime, str(old_regime))
            new_name = self.regime_names.get(
                self.current_regime, str(self.current_regime)
            )
            logger.info(
                f"Regime switch: {old_name} -> {new_name} (prob: {max_prob:.3f})"
            )
        else:
            self.regime_duration += 1

        # Track regime history
        self.regime_history.append(
            {
                "index": current_idx,
                "regime": self.current_regime,
                "probabilities": regime_probs.tolist(),
                "duration": self.regime_duration,
                "switched": regime_changed,
            }
        )

        return regime_changed

    def get_active_strategy(self) -> Strategy:
        """Get currently active strategy based on regime.

        Returns:
            Active strategy for current regime
        """
        if self.current_regime in self.regime_strategies:
            return self.regime_strategies[self.current_regime]
        else:
            # Fallback to first available strategy
            return next(iter(self.regime_strategies.values()))

    def generate_signals(self, data: DataFrame) -> tuple[Series, Series]:
        """Generate regime-aware trading signals.

        Args:
            data: Price data with OHLCV columns

        Returns:
            Tuple of (entry_signals, exit_signals) as boolean Series
        """
        try:
            # Validate input data
            if data is None or len(data) == 0:
                logger.warning("Empty or invalid data provided to generate_signals")
                # Create empty Series with a dummy index to avoid empty array issues
                dummy_index = pd.DatetimeIndex([pd.Timestamp.now()])
                return pd.Series(False, index=dummy_index), pd.Series(
                    False, index=dummy_index
                )

            # Ensure minimum data requirements
            min_required_data = max(50, self.regime_detector.lookback_period)
            if len(data) < min_required_data:
                logger.warning(
                    f"Insufficient data for regime-aware strategy: {len(data)} < {min_required_data}"
                )
                # Return all False signals but with valid data index
                return pd.Series(False, index=data.index), pd.Series(
                    False, index=data.index
                )

            # Fit regime detector if not already done
            if not self.regime_detector.is_fitted:
                try:
                    self.fit_regime_detector(data)
                except Exception as e:
                    logger.error(
                        f"Failed to fit regime detector: {e}, falling back to single strategy"
                    )
                    # Fallback to using first available strategy without regime switching
                    fallback_strategy = next(iter(self.regime_strategies.values()))
                    return fallback_strategy.generate_signals(data)

            entry_signals = pd.Series(False, index=data.index)
            exit_signals = pd.Series(False, index=data.index)

            # Generate signals with regime awareness
            current_strategy = None

            for idx in range(len(data)):
                # Update regime
                regime_changed = self.update_current_regime(data, idx)

                # Get active strategy
                active_strategy = self.get_active_strategy()

                # If regime changed, regenerate signals from new strategy
                if regime_changed or current_strategy != active_strategy:
                    current_strategy = active_strategy

                    # Generate signals for remaining data
                    remaining_data = data.iloc[idx:]
                    if len(remaining_data) > 0:
                        strategy_entry, strategy_exit = (
                            current_strategy.generate_signals(remaining_data)
                        )

                        # Update signals for remaining period
                        end_idx = min(idx + len(strategy_entry), len(data))
                        entry_signals.iloc[idx:end_idx] = strategy_entry.iloc[
                            : end_idx - idx
                        ]
                        exit_signals.iloc[idx:end_idx] = strategy_exit.iloc[
                            : end_idx - idx
                        ]

            logger.info(
                f"Generated regime-aware signals with {self.regime_switches} regime switches"
            )

            return entry_signals, exit_signals

        except Exception as e:
            logger.error(f"Error generating regime-aware signals: {e}")
            # Ensure we always return valid series even on error
            if data is not None and len(data) > 0:
                return pd.Series(False, index=data.index), pd.Series(
                    False, index=data.index
                )
            else:
                # Create dummy index to avoid empty array issues
                dummy_index = pd.DatetimeIndex([pd.Timestamp.now()])
                return pd.Series(False, index=dummy_index), pd.Series(
                    False, index=dummy_index
                )

    def get_regime_analysis(self) -> dict[str, Any]:
        """Get analysis of regime detection and switching.

        Returns:
            Dictionary with regime analysis
        """
        if not self.regime_history:
            return {}

        regime_counts = {}
        regime_durations = {}

        for record in self.regime_history:
            regime = record["regime"]
            regime_name = self.regime_names.get(regime, f"Regime_{regime}")

            if regime_name not in regime_counts:
                regime_counts[regime_name] = 0
                regime_durations[regime_name] = []

            regime_counts[regime_name] += 1

            # Track regime durations
            if record["switched"] and len(self.regime_history) > 1:
                # Find duration of previous regime
                prev_regime_start = 0
                for i in range(len(self.regime_history) - 2, -1, -1):
                    if (
                        self.regime_history[i]["regime"]
                        != self.regime_history[-1]["regime"]
                    ):
                        prev_regime_start = i + 1
                        break

                duration = len(self.regime_history) - prev_regime_start - 1
                prev_regime = self.regime_history[prev_regime_start]["regime"]
                prev_regime_name = self.regime_names.get(
                    prev_regime, f"Regime_{prev_regime}"
                )

                if prev_regime_name in regime_durations:
                    regime_durations[prev_regime_name].append(duration)

        # Calculate average durations
        avg_durations = {}
        for regime_name, durations in regime_durations.items():
            if durations:
                avg_durations[regime_name] = np.mean(durations)
            else:
                avg_durations[regime_name] = 0

        return {
            "current_regime": (
                self.regime_names[self.current_regime]
                if self.current_regime in self.regime_names
                else self.current_regime
            ),
            "total_switches": self.regime_switches,
            "regime_counts": regime_counts,
            "average_regime_durations": avg_durations,
            "regime_history": self.regime_history[-50:],  # Last 50 records
            "active_strategy": self.get_active_strategy().name,
        }

    def validate_parameters(self) -> bool:
        """Validate regime-aware strategy parameters.

        Returns:
            True if parameters are valid
        """
        if not self.regime_strategies:
            return False

        if self.switch_threshold < 0 or self.switch_threshold > 1:
            return False

        if self.min_regime_duration < 0:
            return False

        # Validate individual strategies
        for strategy in self.regime_strategies.values():
            if not strategy.validate_parameters():
                return False

        return True

    def get_default_parameters(self) -> dict[str, Any]:
        """Get default parameters for regime-aware strategy.

        Returns:
            Dictionary of default parameters
        """
        return {
            "switch_threshold": 0.7,
            "min_regime_duration": 5,
            "regime_detection_method": "hmm",
            "n_regimes": 3,
            "lookback_period": 50,
        }
