"""Adaptive trading strategies with online learning and parameter adjustment."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from maverick_mcp.backtesting.strategies.base import Strategy

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
    ):
        """Initialize adaptive strategy.

        Args:
            base_strategy: Base strategy to adapt
            adaptation_method: Method for parameter adaptation
            learning_rate: Learning rate for parameter updates
            lookback_period: Period for performance evaluation
            adaptation_frequency: How often to adapt parameters
            parameters: Additional parameters
        """
        super().__init__(parameters)
        self.base_strategy = base_strategy
        self.adaptation_method = adaptation_method
        self.learning_rate = learning_rate
        self.lookback_period = lookback_period
        self.adaptation_frequency = adaptation_frequency

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

        # Alternative metrics could be:
        # - Calmar ratio: returns.mean() / abs(max_drawdown)
        # - Sortino ratio: returns.mean() / downside_deviation
        # - Information ratio: excess_returns.mean() / tracking_error

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

        # Try random perturbations and keep improvements
        for param_name, param_info in adaptable_params.items():
            if param_name in self.base_strategy.parameters:
                current_value = self.base_strategy.parameters[param_name]

                # Generate random perturbation
                min_val = param_info.get("min", current_value * 0.5)
                max_val = param_info.get("max", current_value * 2.0)

                # Small random step
                perturbation = np.random.normal(0, abs(current_value) * 0.1)
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


class OnlineLearningStrategy(Strategy):
    """Strategy with online learning capabilities using streaming ML algorithms."""

    def __init__(
        self,
        model_type: str = "sgd",
        update_frequency: int = 10,
        feature_window: int = 20,
        confidence_threshold: float = 0.6,
        parameters: dict[str, Any] | None = None,
    ):
        """Initialize online learning strategy.

        Args:
            model_type: Type of online learning model
            update_frequency: How often to update the model
            feature_window: Window for feature calculation
            confidence_threshold: Minimum confidence for signals
            parameters: Additional parameters
        """
        super().__init__(parameters)
        self.model_type = model_type
        self.update_frequency = update_frequency
        self.feature_window = feature_window
        self.confidence_threshold = confidence_threshold

        # Initialize online learning components
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_buffer = []
        self.last_update = 0

        self._initialize_model()

    def _initialize_model(self):
        """Initialize online learning model."""
        if self.model_type == "sgd":
            self.model = SGDClassifier(
                loss="log_loss", learning_rate="adaptive", eta0=0.01, random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    @property
    def name(self) -> str:
        """Get strategy name."""
        return f"OnlineLearning({self.model_type.upper()})"

    @property
    def description(self) -> str:
        """Get strategy description."""
        return (
            f"Online learning strategy using {self.model_type} with streaming updates"
        )

    def extract_features(self, data: DataFrame, end_idx: int) -> np.ndarray:
        """Extract features for online learning.

        Args:
            data: Price data
            end_idx: End index for feature calculation

        Returns:
            Feature array
        """
        start_idx = max(0, end_idx - self.feature_window)
        window_data = data.iloc[start_idx : end_idx + 1]

        if len(window_data) < 2:
            return np.array([])

        features = []

        # Price features
        returns = window_data["close"].pct_change()
        features.extend(
            [
                returns.mean(),
                returns.std(),
                returns.skew() if len(returns) > 2 else 0,
                returns.kurt() if len(returns) > 2 else 0,
            ]
        )

        # Technical indicators
        if len(window_data) >= 5:
            sma_5 = window_data["close"].rolling(5).mean().iloc[-1]
            features.append(window_data["close"].iloc[-1] / sma_5 if sma_5 > 0 else 1.0)
        else:
            features.append(1.0)

        if len(window_data) >= 10:
            sma_10 = window_data["close"].rolling(10).mean().iloc[-1]
            features.append(
                window_data["close"].iloc[-1] / sma_10 if sma_10 > 0 else 1.0
            )
        else:
            features.append(1.0)

        # Volume features (if available)
        if "volume" in window_data.columns:
            volume_ma = window_data["volume"].rolling(5).mean().iloc[-1]
            features.append(
                window_data["volume"].iloc[-1] / volume_ma if volume_ma > 0 else 1.0
            )
        else:
            features.append(1.0)

        return np.array(features)

    def create_target(self, data: DataFrame, idx: int, forward_periods: int = 3) -> int:
        """Create target variable for online learning.

        Args:
            data: Price data
            idx: Current index
            forward_periods: Periods to look forward

        Returns:
            Target class (0: sell, 1: hold, 2: buy)
        """
        if idx + forward_periods >= len(data):
            return 1  # Hold as default

        current_price = data["close"].iloc[idx]
        future_price = data["close"].iloc[idx + forward_periods]

        return_threshold = 0.02  # 2% threshold
        forward_return = (future_price - current_price) / current_price

        if forward_return > return_threshold:
            return 2  # Buy
        elif forward_return < -return_threshold:
            return 0  # Sell
        else:
            return 1  # Hold

    def update_model(self, data: DataFrame, current_idx: int) -> None:
        """Update online learning model with new data.

        Args:
            data: Price data
            current_idx: Current index
        """
        if current_idx - self.last_update < self.update_frequency:
            return

        try:
            # Collect recent training examples
            recent_examples = []
            recent_targets = []

            # Look back for training examples
            lookback_start = max(0, current_idx - self.update_frequency * 2)

            for idx in range(lookback_start, current_idx):
                features = self.extract_features(data, idx)
                if len(features) > 0:
                    target = self.create_target(data, idx)
                    recent_examples.append(features)
                    recent_targets.append(target)

            if not recent_examples:
                return

            X = np.array(recent_examples)
            y = np.array(recent_targets)

            # Scale features
            if not self.is_trained:
                # Fit scaler on initial data
                X_scaled = self.scaler.fit_transform(X)
                self.model.fit(X_scaled, y)
                self.is_trained = True
            else:
                # Update scaler and model incrementally
                X_scaled = self.scaler.transform(X)
                self.model.partial_fit(X_scaled, y)

            self.last_update = current_idx

            logger.debug(
                f"Updated online model at index {current_idx} with {len(X)} samples"
            )

        except Exception as e:
            logger.error(f"Error updating online model: {e}")

    def generate_signals(self, data: DataFrame) -> tuple[Series, Series]:
        """Generate signals using online learning.

        Args:
            data: Price data with OHLCV columns

        Returns:
            Tuple of (entry_signals, exit_signals) as boolean Series
        """
        entry_signals = pd.Series(False, index=data.index)
        exit_signals = pd.Series(False, index=data.index)

        try:
            # Need minimum data for features
            start_idx = max(self.feature_window, 10)

            for idx in range(start_idx, len(data)):
                # Update model periodically
                self.update_model(data, idx)

                if not self.is_trained:
                    continue

                # Extract features for current point
                features = self.extract_features(data, idx)
                if len(features) == 0:
                    continue

                # Make prediction
                X = self.scaler.transform([features])
                prediction = self.model.predict(X)[0]
                confidence = (
                    max(self.model.predict_proba(X)[0])
                    if hasattr(self.model, "predict_proba")
                    else 0.5
                )

                # Generate signals based on prediction and confidence
                if confidence >= self.confidence_threshold:
                    if prediction == 2:  # Buy signal
                        entry_signals.iloc[idx] = True
                    elif prediction == 0:  # Sell signal
                        exit_signals.iloc[idx] = True

        except Exception as e:
            logger.error(f"Error generating online learning signals: {e}")

        return entry_signals, exit_signals

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the online learning model.

        Returns:
            Dictionary with model information
        """
        info = {
            "model_type": self.model_type,
            "is_trained": self.is_trained,
            "feature_window": self.feature_window,
            "update_frequency": self.update_frequency,
            "confidence_threshold": self.confidence_threshold,
        }

        if hasattr(self.model, "coef_") and self.model.coef_ is not None:
            info["model_coefficients"] = self.model.coef_.tolist()

        return info


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
