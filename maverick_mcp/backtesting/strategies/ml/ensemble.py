"""Strategy ensemble methods for combining multiple trading strategies."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from maverick_mcp.backtesting.strategies.base import Strategy

logger = logging.getLogger(__name__)


class StrategyEnsemble(Strategy):
    """Ensemble strategy that combines multiple strategies with dynamic weighting."""

    def __init__(
        self,
        strategies: list[Strategy],
        weighting_method: str = "performance",
        lookback_period: int = 50,
        rebalance_frequency: int = 20,
        parameters: dict[str, Any] = None,
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
        self.strategy_returns = {}
        self.strategy_signals = {}
        self.last_rebalance = 0

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
        if len(self.strategy_returns) < 2:
            return self.weights

        # Calculate Sharpe ratios for each strategy
        sharpe_ratios = []
        for i, _strategy in enumerate(self.strategies):
            if (
                i in self.strategy_returns
                and len(self.strategy_returns[i]) >= self.lookback_period
            ):
                returns = pd.Series(self.strategy_returns[i][-self.lookback_period :])
                sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
                sharpe_ratios.append(max(0, sharpe))  # Ensure non-negative
            else:
                sharpe_ratios.append(0.1)  # Small positive weight for new strategies

        # Convert to weights (softmax-like normalization)
        sharpe_array = np.array(sharpe_ratios)
        # Fix: Properly check for empty array and zero sum conditions
        if sharpe_array.size == 0 or np.sum(sharpe_array) == 0:
            weights = np.ones(len(self.strategies)) / len(self.strategies)
        else:
            # Exponential weighting to emphasize better performers
            exp_sharpe = np.exp(sharpe_array * 2)
            weights = exp_sharpe / exp_sharpe.sum()

        return weights

    def calculate_volatility_weights(self, data: DataFrame) -> np.ndarray:
        """Calculate inverse volatility weights for strategies.

        Args:
            data: Price data for volatility calculation

        Returns:
            Array of strategy weights
        """
        if len(self.strategy_returns) < 2:
            return self.weights

        # Calculate volatilities for each strategy
        volatilities = []
        for i, _strategy in enumerate(self.strategies):
            if (
                i in self.strategy_returns
                and len(self.strategy_returns[i]) >= self.lookback_period
            ):
                returns = pd.Series(self.strategy_returns[i][-self.lookback_period :])
                vol = returns.std() * np.sqrt(252)
                volatilities.append(max(0.01, vol))  # Minimum volatility
            else:
                volatilities.append(0.2)  # Default volatility assumption

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
                # Generate signals with timeout protection
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
                    # Convert to boolean if necessary
                    entry_signals = entry_signals.astype(bool)
                    exit_signals = exit_signals.astype(bool)

                signals[i] = (entry_signals, exit_signals)

                # Calculate strategy returns for weight updates (with error handling)
                try:
                    positions = entry_signals.astype(int) - exit_signals.astype(int)
                    price_returns = data["close"].pct_change()
                    returns = positions.shift(1) * price_returns

                    # Remove invalid returns
                    valid_returns = returns.dropna()
                    valid_returns = valid_returns[np.isfinite(valid_returns)]

                    if i not in self.strategy_returns:
                        self.strategy_returns[i] = []

                    if len(valid_returns) > 0:
                        self.strategy_returns[i].extend(valid_returns.tolist())

                        # Keep only recent returns for performance calculation
                        if len(self.strategy_returns[i]) > self.lookback_period * 2:
                            self.strategy_returns[i] = self.strategy_returns[i][
                                -self.lookback_period * 2 :
                            ]

                except Exception as return_error:
                    logger.debug(
                        f"Error calculating returns for strategy {strategy.name}: {return_error}"
                    )

                logger.debug(
                    f"Strategy {strategy.name}: {entry_signals.sum()} entries, {exit_signals.sum()} exits"
                )

            except Exception as e:
                logger.error(
                    f"Error generating signals for strategy {strategy.name}: {e}"
                )
                failed_strategies.append(i)

                # Create safe fallback signals
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

        Args:
            individual_signals: Dictionary of individual strategy signals

        Returns:
            Tuple of combined (entry_signals, exit_signals)
        """
        if not individual_signals:
            # Return empty series with minimal index when no individual signals available
            empty_index = pd.Index([])
            return pd.Series(False, index=empty_index), pd.Series(
                False, index=empty_index
            )

        # Get data index from first strategy
        first_signals = next(iter(individual_signals.values()))
        data_index = first_signals[0].index

        # Initialize voting arrays
        entry_votes = np.zeros(len(data_index))
        exit_votes = np.zeros(len(data_index))
        total_weights = 0

        # Collect votes with weights and confidence scores
        valid_strategies = 0

        for i, (entry_signals, exit_signals) in individual_signals.items():
            weight = self.weights[i] if i < len(self.weights) else 0

            if weight > 0:
                # Add weighted votes
                entry_votes += weight * entry_signals.astype(float)
                exit_votes += weight * exit_signals.astype(float)
                total_weights += weight
                valid_strategies += 1

        if total_weights == 0 or valid_strategies == 0:
            logger.warning("No valid strategies with positive weights")
            return pd.Series(False, index=data_index), pd.Series(
                False, index=data_index
            )

        # Normalize votes by total weights
        entry_votes = entry_votes / total_weights
        exit_votes = exit_votes / total_weights

        # Enhanced voting mechanisms
        voting_method = self.parameters.get("voting_method", "weighted")

        if voting_method == "majority":
            # Simple majority vote (more than half of strategies agree)
            entry_threshold = 0.5
            exit_threshold = 0.5
        elif voting_method == "supermajority":
            # Require 2/3 agreement
            entry_threshold = 0.67
            exit_threshold = 0.67
        elif voting_method == "consensus":
            # Require near-unanimous agreement
            entry_threshold = 0.8
            exit_threshold = 0.8
        else:  # weighted (default)
            entry_threshold = self.parameters.get("entry_threshold", 0.5)
            exit_threshold = self.parameters.get("exit_threshold", 0.5)

        # Anti-conflict mechanism: don't signal entry and exit simultaneously
        combined_entry = entry_votes > entry_threshold
        combined_exit = exit_votes > exit_threshold

        # Resolve conflicts (simultaneous entry and exit signals)
        conflicts = combined_entry & combined_exit
        # Fix: Check array size and ensure it's not empty before evaluating boolean truth
        if conflicts.size > 0 and np.any(conflicts):
            logger.debug(f"Resolving {conflicts.sum()} signal conflicts")
            # In case of conflict, use the stronger signal
            entry_strength = entry_votes[conflicts]
            exit_strength = exit_votes[conflicts]

            # Keep only the stronger signal
            stronger_entry = entry_strength > exit_strength
            combined_entry[conflicts] = stronger_entry
            combined_exit[conflicts] = ~stronger_entry

        # Quality filter: require minimum signal strength
        min_signal_strength = self.parameters.get("min_signal_strength", 0.1)
        weak_entry_signals = (combined_entry) & (entry_votes < min_signal_strength)
        weak_exit_signals = (combined_exit) & (exit_votes < min_signal_strength)

        # Fix: Ensure arrays are not empty before boolean indexing
        if weak_entry_signals.size > 0:
            combined_entry[weak_entry_signals] = False
        if weak_exit_signals.size > 0:
            combined_exit[weak_exit_signals] = False

        # Convert to pandas Series
        combined_entry = pd.Series(combined_entry, index=data_index)
        combined_exit = pd.Series(combined_exit, index=data_index)

        return combined_entry, combined_exit

    def generate_signals(self, data: DataFrame) -> tuple[Series, Series]:
        """Generate ensemble trading signals.

        Args:
            data: Price data with OHLCV columns

        Returns:
            Tuple of (entry_signals, exit_signals) as boolean Series
        """
        try:
            # Generate signals from all individual strategies
            individual_signals = self.generate_individual_signals(data)

            if not individual_signals:
                return pd.Series(False, index=data.index), pd.Series(
                    False, index=data.index
                )

            # Update weights periodically
            for idx in range(
                self.rebalance_frequency, len(data), self.rebalance_frequency
            ):
                self.update_weights(data.iloc[:idx], idx)

            # Combine signals
            entry_signals, exit_signals = self.combine_signals(individual_signals)

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


class RiskAdjustedEnsemble(StrategyEnsemble):
    """Risk-adjusted ensemble with position sizing and risk management."""

    def __init__(
        self,
        strategies: list[Strategy],
        max_position_size: float = 0.1,
        max_correlation: float = 0.7,
        risk_target: float = 0.15,
        **kwargs,
    ):
        """Initialize risk-adjusted ensemble.

        Args:
            strategies: List of base strategies
            max_position_size: Maximum position size per strategy
            max_correlation: Maximum correlation between strategies
            risk_target: Target portfolio volatility
            **kwargs: Additional parameters for base ensemble
        """
        super().__init__(strategies, **kwargs)
        self.max_position_size = max_position_size
        self.max_correlation = max_correlation
        self.risk_target = risk_target

    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix between strategy returns.

        Returns:
            Correlation matrix as DataFrame
        """
        if len(self.strategy_returns) < 2:
            return pd.DataFrame()

        # Create returns DataFrame
        min_length = min(
            len(returns)
            for returns in self.strategy_returns.values()
            if len(returns) > 0
        )
        if min_length == 0:
            return pd.DataFrame()

        returns_data = {}
        for i, strategy in enumerate(self.strategies):
            if (
                i in self.strategy_returns
                and len(self.strategy_returns[i]) >= min_length
            ):
                returns_data[strategy.name] = self.strategy_returns[i][-min_length:]

        if not returns_data:
            return pd.DataFrame()

        returns_df = pd.DataFrame(returns_data)
        return returns_df.corr()

    def adjust_weights_for_correlation(self, weights: np.ndarray) -> np.ndarray:
        """Adjust weights to account for strategy correlation.

        Args:
            weights: Original weights

        Returns:
            Correlation-adjusted weights
        """
        corr_matrix = self.calculate_correlation_matrix()

        if corr_matrix.empty:
            return weights

        try:
            # Penalize highly correlated strategies
            adjusted_weights = weights.copy()

            for i, strategy_i in enumerate(self.strategies):
                for j, strategy_j in enumerate(self.strategies):
                    if (
                        i != j
                        and strategy_i.name in corr_matrix.index
                        and strategy_j.name in corr_matrix.columns
                    ):
                        correlation = abs(
                            corr_matrix.loc[strategy_i.name, strategy_j.name]
                        )

                        if correlation > self.max_correlation:
                            # Reduce weight of both strategies
                            penalty = (correlation - self.max_correlation) * 0.5
                            adjusted_weights[i] *= 1 - penalty
                            adjusted_weights[j] *= 1 - penalty

            # Renormalize weights
            # Fix: Check array size and sum properly before normalization
            if adjusted_weights.size > 0 and np.sum(adjusted_weights) > 0:
                adjusted_weights /= adjusted_weights.sum()
            else:
                adjusted_weights = np.ones(len(self.strategies)) / len(self.strategies)

            return adjusted_weights

        except Exception as e:
            logger.error(f"Error adjusting weights for correlation: {e}")
            return weights

    def calculate_risk_adjusted_weights(self, data: DataFrame) -> np.ndarray:
        """Calculate risk-adjusted weights based on target volatility.

        Args:
            data: Price data

        Returns:
            Risk-adjusted weights
        """
        # Start with performance-based weights
        base_weights = self.calculate_performance_weights(data)

        # Adjust for correlation
        corr_adjusted_weights = self.adjust_weights_for_correlation(base_weights)

        # Apply position size limits
        position_adjusted_weights = np.minimum(
            corr_adjusted_weights, self.max_position_size
        )

        # Renormalize
        # Fix: Check array size and sum properly before normalization
        if position_adjusted_weights.size > 0 and np.sum(position_adjusted_weights) > 0:
            position_adjusted_weights /= position_adjusted_weights.sum()
        else:
            position_adjusted_weights = np.ones(len(self.strategies)) / len(
                self.strategies
            )

        return position_adjusted_weights

    def update_weights(self, data: DataFrame, current_index: int) -> None:
        """Update risk-adjusted weights.

        Args:
            data: Price data
            current_index: Current position in data
        """
        if current_index - self.last_rebalance < self.rebalance_frequency:
            return

        try:
            self.weights = self.calculate_risk_adjusted_weights(data)
            self.last_rebalance = current_index

            logger.debug(
                f"Updated risk-adjusted weights: {dict(zip([s.name for s in self.strategies], self.weights, strict=False))}"
            )

        except Exception as e:
            logger.error(f"Error updating risk-adjusted weights: {e}")

    @property
    def name(self) -> str:
        """Get strategy name."""
        return f"RiskAdjusted{super().name}"

    @property
    def description(self) -> str:
        """Get strategy description."""
        return "Risk-adjusted ensemble with correlation control and position sizing"
