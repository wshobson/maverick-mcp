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
        min_training_samples: int = 100,
        initial_training_period: int = 200,
        parameters: dict[str, Any] | None = None,
    ):
        """Initialize online learning strategy.

        Args:
            model_type: Type of online learning model
            update_frequency: How often to update the model
            feature_window: Window for feature calculation
            confidence_threshold: Minimum confidence for signals
            min_training_samples: Minimum samples before enabling predictions
            initial_training_period: Period for initial batch training
            parameters: Additional parameters
        """
        super().__init__(parameters)
        self.model_type = model_type
        self.update_frequency = update_frequency
        self.feature_window = feature_window
        self.confidence_threshold = confidence_threshold
        self.min_training_samples = min_training_samples
        self.initial_training_period = initial_training_period

        # Initialize online learning components
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.is_initial_trained = False
        self.training_buffer = []
        self.last_update = 0
        self.training_samples_count = 0

        # Feature consistency tracking
        self.expected_feature_count = None
        self.feature_stats_buffer = []

        self._initialize_model()

    def _initialize_model(self):
        """Initialize online learning model with proper configuration."""
        if self.model_type == "sgd":
            self.model = SGDClassifier(
                loss="log_loss",
                learning_rate="adaptive",
                eta0=0.01,
                random_state=42,
                max_iter=1000,
                tol=1e-4,
                warm_start=True,  # Enable incremental learning
                alpha=0.01,  # Regularization
                fit_intercept=True
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Initialize scaler as None - will be created during first fit
        self.scaler = None

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
        """Extract features for online learning with enhanced stability.

        Args:
            data: Price data
            end_idx: End index for feature calculation

        Returns:
            Feature array with consistent dimensionality
        """
        try:
            start_idx = max(0, end_idx - self.feature_window)
            window_data = data.iloc[start_idx : end_idx + 1]

            # Need minimum data for meaningful features
            if len(window_data) < max(5, self.feature_window // 4):
                return np.array([])

            features = []

            # Price features with error handling
            returns = window_data["close"].pct_change().dropna()
            if len(returns) == 0:
                return np.array([])

            # Basic return statistics (robust to small samples)
            mean_return = returns.mean() if len(returns) > 0 else 0.0
            std_return = returns.std() if len(returns) > 1 else 0.01  # Small default
            skew_return = returns.skew() if len(returns) > 3 else 0.0
            kurt_return = returns.kurtosis() if len(returns) > 3 else 0.0

            # Replace NaN/inf values
            features.extend([
                mean_return if np.isfinite(mean_return) else 0.0,
                std_return if np.isfinite(std_return) else 0.01,
                skew_return if np.isfinite(skew_return) else 0.0,
                kurt_return if np.isfinite(kurt_return) else 0.0,
            ])

            # Technical indicators with fallbacks
            current_price = window_data["close"].iloc[-1]

            # Short-term moving average ratio
            if len(window_data) >= 5:
                sma_5 = window_data["close"].rolling(5).mean().iloc[-1]
                features.append(current_price / sma_5 if sma_5 > 0 else 1.0)
            else:
                features.append(1.0)

            # Medium-term moving average ratio
            if len(window_data) >= 10:
                sma_10 = window_data["close"].rolling(10).mean().iloc[-1]
                features.append(current_price / sma_10 if sma_10 > 0 else 1.0)
            else:
                features.append(1.0)

            # Long-term moving average ratio (if enough data)
            if len(window_data) >= 20:
                sma_20 = window_data["close"].rolling(20).mean().iloc[-1]
                features.append(current_price / sma_20 if sma_20 > 0 else 1.0)
            else:
                features.append(1.0)

            # Volatility feature
            if len(returns) > 10:
                vol_ratio = std_return / returns.rolling(10).std().mean()
                features.append(vol_ratio if np.isfinite(vol_ratio) else 1.0)
            else:
                features.append(1.0)

            # Volume features (if available)
            if "volume" in window_data.columns and len(window_data) >= 5:
                current_volume = window_data["volume"].iloc[-1]
                volume_ma = window_data["volume"].rolling(5).mean().iloc[-1]
                volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1.0
                features.append(volume_ratio if np.isfinite(volume_ratio) else 1.0)

                # Volume trend
                if len(window_data) >= 10:
                    volume_ma_long = window_data["volume"].rolling(10).mean().iloc[-1]
                    volume_trend = volume_ma / volume_ma_long if volume_ma_long > 0 else 1.0
                    features.append(volume_trend if np.isfinite(volume_trend) else 1.0)
                else:
                    features.append(1.0)
            else:
                features.extend([1.0, 1.0])

            feature_array = np.array(features)

            # Validate feature consistency
            if self.expected_feature_count is None:
                self.expected_feature_count = len(feature_array)
            elif len(feature_array) != self.expected_feature_count:
                logger.warning(f"Feature count mismatch: expected {self.expected_feature_count}, got {len(feature_array)}")
                return np.array([])

            # Check for any remaining NaN or inf values
            if not np.all(np.isfinite(feature_array)):
                logger.warning("Non-finite features detected, replacing with defaults")
                feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1.0, neginf=-1.0)

            return feature_array

        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return np.array([])

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

    def _initial_training(self, data: DataFrame, current_idx: int) -> bool:
        """Perform initial batch training on historical data.

        Args:
            data: Price data
            current_idx: Current index

        Returns:
            True if initial training successful
        """
        try:
            if current_idx < self.initial_training_period:
                return False

            # Collect initial training data
            training_examples = []
            training_targets = []

            # Use a substantial portion of historical data for initial training
            start_idx = max(self.feature_window, current_idx - self.initial_training_period)

            for idx in range(start_idx, current_idx - 10):  # Leave some data for validation
                features = self.extract_features(data, idx)
                if len(features) > 0:
                    target = self.create_target(data, idx)
                    training_examples.append(features)
                    training_targets.append(target)

            if len(training_examples) < self.min_training_samples:
                logger.debug(f"Insufficient training samples: {len(training_examples)} < {self.min_training_samples}")
                return False

            X = np.array(training_examples)
            y = np.array(training_targets)

            # Check for class balance
            unique_classes, class_counts = np.unique(y, return_counts=True)
            if len(unique_classes) < 2:
                logger.warning(f"Insufficient class diversity for training: {unique_classes}")
                return False

            # Initialize scaler with training data
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            # Train initial model
            self.model.fit(X_scaled, y)
            self.is_initial_trained = True
            self.is_trained = True
            self.training_samples_count = len(X)

            logger.info(f"Initial training completed with {len(X)} samples, classes: {dict(zip(unique_classes, class_counts, strict=False))}")
            return True

        except Exception as e:
            logger.error(f"Error in initial training: {e}")
            return False

    def update_model(self, data: DataFrame, current_idx: int) -> None:
        """Update online learning model with new data.

        Args:
            data: Price data
            current_idx: Current index
        """
        # Perform initial training if not done yet
        if not self.is_initial_trained:
            if self._initial_training(data, current_idx):
                self.last_update = current_idx
            return

        # Check if update is needed
        if current_idx - self.last_update < self.update_frequency:
            return

        try:
            # Collect recent training examples for incremental update
            recent_examples = []
            recent_targets = []

            # Look back for recent training examples (smaller window for incremental updates)
            lookback_start = max(0, current_idx - self.update_frequency)

            for idx in range(lookback_start, current_idx):
                features = self.extract_features(data, idx)
                if len(features) > 0:
                    target = self.create_target(data, idx)
                    recent_examples.append(features)
                    recent_targets.append(target)

            if len(recent_examples) < 2:  # Need minimum samples for incremental update
                return

            X = np.array(recent_examples)
            y = np.array(recent_targets)

            # Scale features using existing scaler
            X_scaled = self.scaler.transform(X)

            # Incremental update using partial_fit
            # Ensure we have all classes that the model has seen before
            existing_classes = self.model.classes_
            self.model.partial_fit(X_scaled, y, classes=existing_classes)

            self.training_samples_count += len(X)
            self.last_update = current_idx

            logger.debug(
                f"Updated online model at index {current_idx} with {len(X)} samples (total: {self.training_samples_count})"
            )

        except Exception as e:
            logger.error(f"Error updating online model: {e}")
            # Reset training flag to attempt re-initialization
            if "partial_fit" in str(e).lower():
                logger.warning("Partial fit failed, will attempt re-initialization")
                self.is_trained = False
                self.is_initial_trained = False

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
            start_idx = max(self.feature_window, self.initial_training_period + 10)

            if len(data) < start_idx:
                logger.warning(f"Insufficient data for online learning: {len(data)} < {start_idx}")
                return entry_signals, exit_signals

            for idx in range(start_idx, len(data)):
                # Update model periodically
                self.update_model(data, idx)

                if not self.is_trained or self.scaler is None:
                    continue

                # Extract features for current point
                features = self.extract_features(data, idx)
                if len(features) == 0:
                    continue

                try:
                    # Make prediction with error handling
                    X = self.scaler.transform([features])
                    prediction = self.model.predict(X)[0]

                    # Get confidence score
                    if hasattr(self.model, "predict_proba"):
                        probabilities = self.model.predict_proba(X)[0]
                        confidence = max(probabilities)
                    else:
                        # For models without predict_proba, use decision function
                        if hasattr(self.model, "decision_function"):
                            decision_values = self.model.decision_function(X)[0]
                            # Convert to pseudo-probability (sigmoid-like)
                            confidence = 1.0 / (1.0 + np.exp(-abs(decision_values)))
                        else:
                            confidence = 0.6  # Default confidence

                    # Generate signals based on prediction and confidence
                    if confidence >= self.confidence_threshold:
                        if prediction == 2:  # Buy signal
                            entry_signals.iloc[idx] = True
                        elif prediction == 0:  # Sell signal
                            exit_signals.iloc[idx] = True

                except Exception as pred_error:
                    logger.debug(f"Prediction error at index {idx}: {pred_error}")
                    continue

            # Log summary statistics
            total_entry_signals = entry_signals.sum()
            total_exit_signals = exit_signals.sum()
            logger.info(f"Generated {total_entry_signals} entry and {total_exit_signals} exit signals using online learning")

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
            "is_initial_trained": self.is_initial_trained,
            "feature_window": self.feature_window,
            "update_frequency": self.update_frequency,
            "confidence_threshold": self.confidence_threshold,
            "min_training_samples": self.min_training_samples,
            "initial_training_period": self.initial_training_period,
            "training_samples_count": self.training_samples_count,
            "expected_feature_count": self.expected_feature_count,
        }

        if hasattr(self.model, "coef_") and self.model.coef_ is not None:
            info["model_coefficients"] = self.model.coef_.tolist()

        if hasattr(self.model, "classes_") and self.model.classes_ is not None:
            info["model_classes"] = self.model.classes_.tolist()

        if self.scaler is not None:
            info["feature_scaling"] = {
                "mean": self.scaler.mean_.tolist() if hasattr(self.scaler, "mean_") else None,
                "scale": self.scaler.scale_.tolist() if hasattr(self.scaler, "scale_") else None,
            }

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
