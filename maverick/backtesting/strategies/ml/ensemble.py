"""Strategy ensemble methods for combining multiple trading strategies.

Ported from `maverick_mcp/backtesting/strategies/ml/ensemble.py`. Dead code
removed (see Task 6 report): `RiskAdjustedEnsemble` had zero callers anywhere
outside its own package's unused `__all__` re-export (`grep -rn
"RiskAdjustedEnsemble" maverick_mcp tests` matches only the definition and
that export). `StrategyEnsemble` is live -- the backtesting router constructs
and calls it directly. No randomness in this module; no seeding seam needed.
The weighted voting behind `combine_signals` lives in `ensemble_voting.py`
(split out to stay under the 500-line cap; no behavior change).
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from maverick.backtesting.strategies.base import Strategy

from .ensemble_voting import combine_weighted_signals

logger = logging.getLogger(__name__)


class StrategyEnsemble(Strategy):
    """Ensemble strategy that combines multiple strategies with dynamic weighting."""

    def __init__(
        self,
        strategies: list[Strategy],
        weighting_method: str = "performance",
        lookback_period: int = 50,
        rebalance_frequency: int = 20,
        parameters: dict[str, Any] | None = None,
    ):
        """Initialize strategy ensemble.

        Args:
            strategies: List of base strategies to combine
            weighting_method: Method for calculating weights ('performance', 'equal', 'volatility')
            lookback_period: Period for calculating performance metrics
            rebalance_frequency: How often to update weights
            parameters: Additional parameters
        """
        super().__init__(parameters)
        self.strategies = strategies
        self.weighting_method = weighting_method
        self.lookback_period = lookback_period
        self.rebalance_frequency = rebalance_frequency

        # Initialize strategy weights
        self.weights = np.ones(len(strategies)) / len(strategies)
        self.strategy_returns: dict[int, list[float]] = {}
        self.strategy_signals = {}
        self.last_rebalance = 0
        self._weight_history = pd.DataFrame(columns=range(len(strategies)))

    @property
    def name(self) -> str:
        """Get strategy name."""
        strategy_names = [s.name for s in self.strategies]
        return f"Ensemble({','.join(strategy_names)})"

    @property
    def description(self) -> str:
        """Get strategy description."""
        return f"Dynamic ensemble combining {len(self.strategies)} strategies using {self.weighting_method} weighting"

    def calculate_performance_weights(self, data: DataFrame) -> np.ndarray:
        """Calculate performance-based weights for strategies.

        Args:
            data: Price data for performance calculation

        Returns:
            Array of strategy weights
        """
        if any(
            len(self.strategy_returns.get(i, [])) < self.lookback_period
            for i in range(len(self.strategies))
        ):
            return np.ones(len(self.strategies)) / len(self.strategies)

        # Calculate Sharpe ratios for each strategy
        sharpe_ratios = []
        for i, _strategy in enumerate(self.strategies):
            returns = pd.Series(self.strategy_returns[i][-self.lookback_period :])
            std = returns.std(ddof=1 if len(returns) > 1 else 0)
            sharpe = returns.mean() / (std + 1e-8) * np.sqrt(252)
            sharpe_ratios.append(max(0, sharpe))  # Ensure non-negative

        # Convert to weights (softmax-like normalization)
        sharpe_array = np.array(sharpe_ratios)
        if sharpe_array.size == 0 or np.sum(sharpe_array) == 0:
            weights = np.ones(len(self.strategies)) / len(self.strategies)
        else:
            # Exponential weighting to emphasize better performers
            exp_sharpe = np.exp((sharpe_array - sharpe_array.max()) * 2)
            weights = exp_sharpe / exp_sharpe.sum()

        return weights

    def calculate_volatility_weights(self, data: DataFrame) -> np.ndarray:
        """Calculate inverse volatility weights for strategies.

        Args:
            data: Price data for volatility calculation

        Returns:
            Array of strategy weights
        """
        if any(
            len(self.strategy_returns.get(i, [])) < self.lookback_period
            for i in range(len(self.strategies))
        ):
            return np.ones(len(self.strategies)) / len(self.strategies)

        # Calculate volatilities for each strategy
        volatilities = []
        for i, _strategy in enumerate(self.strategies):
            returns = pd.Series(self.strategy_returns[i][-self.lookback_period :])
            vol = returns.std(ddof=1 if len(returns) > 1 else 0) * np.sqrt(252)
            volatilities.append(max(0.01, vol))  # Minimum volatility

        # Inverse volatility weighting
        vol_array = np.array(volatilities)
        inv_vol = 1.0 / vol_array
        weights = inv_vol / inv_vol.sum()

        return weights

    def update_weights(self, data: DataFrame, current_index: int) -> None:
        """Update strategy weights based on recent performance.

        Args:
            data: Price data
            current_index: Current position in data
        """
        # Check if it's time to rebalance
        if current_index - self.last_rebalance < self.rebalance_frequency:
            return

        try:
            if self.weighting_method == "performance":
                self.weights = self.calculate_performance_weights(data)
            elif self.weighting_method == "volatility":
                self.weights = self.calculate_volatility_weights(data)
            elif self.weighting_method == "equal":
                self.weights = np.ones(len(self.strategies)) / len(self.strategies)
            else:
                logger.warning(f"Unknown weighting method: {self.weighting_method}")

            self.last_rebalance = current_index

            logger.debug(
                f"Updated ensemble weights: {dict(zip([s.name for s in self.strategies], self.weights, strict=False))}"
            )

        except Exception as e:
            logger.error(f"Error updating weights: {e}")

    def generate_individual_signals(
        self, data: DataFrame
    ) -> dict[int, tuple[Series, Series]]:
        """Generate signals from all individual strategies with enhanced error handling.

        Args:
            data: Price data

        Returns:
            Dictionary mapping strategy index to (entry_signals, exit_signals)
        """
        signals = {}
        failed_strategies = []

        for i, strategy in enumerate(self.strategies):
            try:
                entry_signals, exit_signals = strategy.generate_signals(data)

                # Validate signals
                if not isinstance(entry_signals, pd.Series) or not isinstance(
                    exit_signals, pd.Series
                ):
                    raise ValueError(
                        f"Strategy {strategy.name} returned invalid signal types"
                    )

                if len(entry_signals) != len(data) or len(exit_signals) != len(data):
                    raise ValueError(
                        f"Strategy {strategy.name} returned signals with wrong length"
                    )

                if not entry_signals.dtype == bool or not exit_signals.dtype == bool:
                    entry_signals = entry_signals.astype(bool)
                    exit_signals = exit_signals.astype(bool)

                signals[i] = (entry_signals, exit_signals)

                logger.debug(
                    f"Strategy {strategy.name}: {entry_signals.sum()} entries, {exit_signals.sum()} exits"
                )

            except Exception as e:
                logger.error(
                    f"Error generating signals for strategy {strategy.name}: {e}"
                )
                failed_strategies.append(i)

                try:
                    signals[i] = (
                        pd.Series(False, index=data.index),
                        pd.Series(False, index=data.index),
                    )
                except Exception:
                    # If even creating empty signals fails, skip this strategy
                    logger.error(f"Cannot create fallback signals for strategy {i}")
                    continue

        # Log summary of strategy performance
        if failed_strategies:
            failed_names = [self.strategies[i].name for i in failed_strategies]
            logger.warning(f"Failed strategies: {failed_names}")

        successful_strategies = len(signals) - len(failed_strategies)
        logger.info(
            f"Successfully generated signals from {successful_strategies}/{len(self.strategies)} strategies"
        )

        return signals

    def combine_signals(
        self, individual_signals: dict[int, tuple[Series, Series]]
    ) -> tuple[Series, Series]:
        """Combine individual strategy signals using enhanced weighted voting.

        Delegates to `ensemble_voting.combine_weighted_signals`.
        """
        return combine_weighted_signals(
            individual_signals, self.weights, self.parameters
        )

    def generate_signals(self, data: DataFrame) -> tuple[Series, Series]:
        """Generate ensemble trading signals.

        Args:
            data: Price data with OHLCV columns

        Returns:
            Tuple of (entry_signals, exit_signals) as boolean Series
        """
        # Each run starts from equal weights, including reused and empty runs.
        self.weights = np.ones(len(self.strategies)) / len(self.strategies)
        self.strategy_returns = {}
        self.strategy_signals = {}
        self.last_rebalance = 0
        self._weight_history = pd.DataFrame(
            index=data.index, columns=range(len(self.strategies)), dtype=float
        )

        if data.empty:
            return pd.Series(False, index=data.index), pd.Series(
                False, index=data.index
            )

        try:
            # Generate signals from all individual strategies
            individual_signals = self.generate_individual_signals(data)

            if not individual_signals:
                return pd.Series(False, index=data.index), pd.Series(
                    False, index=data.index
                )

            price_returns = data["close"].pct_change()
            returns = {
                i: (entry.astype(int) - exit_.astype(int)).shift(1) * price_returns
                for i, (entry, exit_) in individual_signals.items()
            }
            for start in range(0, len(data), self.rebalance_frequency):
                if start:
                    # Return at t uses the signal at t-1. A boundary at b can
                    # use only returns in [b-lookback, b), never return b.
                    self.strategy_returns = {}
                    for i, strategy_returns in returns.items():
                        window = strategy_returns.iloc[
                            max(0, start - self.lookback_period) : start
                        ]
                        self.strategy_returns[i] = window[np.isfinite(window)].tolist()
                    self.update_weights(data.iloc[:start], start)
                stop = min(start + self.rebalance_frequency, len(data))
                self._weight_history.iloc[start:stop] = self.weights

            entry_signals, exit_signals = combine_weighted_signals(
                individual_signals, self._weight_history.to_numpy(), self.parameters
            )
            # Keep recent run-local returns for the public performance summary;
            # these are populated only after every historical weight is fixed.
            self.strategy_returns = {}
            for i, strategy_returns in returns.items():
                window = strategy_returns.iloc[-self.lookback_period * 2 :]
                self.strategy_returns[i] = window[np.isfinite(window)].tolist()

            logger.info(
                f"Generated ensemble signals: {entry_signals.sum()} entries, {exit_signals.sum()} exits"
            )

            return entry_signals, exit_signals

        except Exception as e:
            logger.error(f"Error generating ensemble signals: {e}")
            return pd.Series(False, index=data.index), pd.Series(
                False, index=data.index
            )

    def get_strategy_weights(self) -> dict[str, float]:
        """Get current strategy weights.

        Returns:
            Dictionary mapping strategy names to weights
        """
        return dict(zip([s.name for s in self.strategies], self.weights, strict=False))

    def get_strategy_performance(self) -> dict[str, dict[str, float]]:
        """Get performance metrics for individual strategies.

        Returns:
            Dictionary mapping strategy names to performance metrics
        """
        performance = {}

        for i, strategy in enumerate(self.strategies):
            if i in self.strategy_returns and len(self.strategy_returns[i]) > 0:
                returns = pd.Series(self.strategy_returns[i])

                performance[strategy.name] = {
                    "total_return": returns.sum(),
                    "annual_return": returns.mean() * 252,
                    "volatility": returns.std() * np.sqrt(252),
                    "sharpe_ratio": returns.mean()
                    / (returns.std() + 1e-8)
                    * np.sqrt(252),
                    "max_drawdown": (
                        returns.cumsum() - returns.cumsum().expanding().max()
                    ).min(),
                    "win_rate": (returns > 0).mean(),
                    "current_weight": self.weights[i],
                }
            else:
                performance[strategy.name] = {
                    "total_return": 0.0,
                    "annual_return": 0.0,
                    "volatility": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "win_rate": 0.0,
                    "current_weight": self.weights[i] if i < len(self.weights) else 0.0,
                }

        return performance

    def validate_parameters(self) -> bool:
        """Validate ensemble parameters.

        Returns:
            True if parameters are valid
        """
        if not self.strategies:
            return False

        if self.weighting_method not in ["performance", "equal", "volatility"]:
            return False

        if self.lookback_period <= 0 or self.rebalance_frequency <= 0:
            return False

        # Validate individual strategies
        for strategy in self.strategies:
            if not strategy.validate_parameters():
                return False

        return True

    def get_default_parameters(self) -> dict[str, Any]:
        """Get default ensemble parameters.

        Returns:
            Dictionary of default parameters
        """
        return {
            "weighting_method": "performance",
            "lookback_period": 50,
            "rebalance_frequency": 20,
            "entry_threshold": 0.5,
            "exit_threshold": 0.5,
            "voting_method": "weighted",  # weighted, majority, supermajority, consensus
            "min_signal_strength": 0.1,  # Minimum signal strength to avoid weak signals
            "conflict_resolution": "stronger",  # How to resolve entry/exit conflicts
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert ensemble to dictionary representation.

        Returns:
            Dictionary with ensemble details
        """
        base_dict = super().to_dict()
        base_dict.update(
            {
                "strategies": [s.to_dict() for s in self.strategies],
                "current_weights": self.get_strategy_weights(),
                "weighting_method": self.weighting_method,
                "lookback_period": self.lookback_period,
                "rebalance_frequency": self.rebalance_frequency,
            }
        )

        return base_dict
