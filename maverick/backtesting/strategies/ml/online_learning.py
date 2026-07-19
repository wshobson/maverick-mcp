"""Online learning trading strategy using a streaming SGD classifier.

Split out of `adaptive.py` (see the Task 6 report): the ported legacy
module put `AdaptiveStrategy`/`OnlineLearningStrategy`/
`HybridAdaptiveStrategy` in one 789-line file, over this repo's
500-line-per-module cap. No behavior changed by moving this class here.
`SGDClassifier` already hardcodes `random_state=42` -- deterministic by
default with no seam needed.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from maverick.backtesting.strategies.base import Strategy

logger = logging.getLogger(__name__)


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
        self.model: SGDClassifier | None = None
        self.scaler: StandardScaler | None = None
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
                fit_intercept=True,
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
            features.extend(
                [
                    mean_return if np.isfinite(mean_return) else 0.0,
                    std_return if np.isfinite(std_return) else 0.01,
                    skew_return if np.isfinite(skew_return) else 0.0,
                    kurt_return if np.isfinite(kurt_return) else 0.0,
                ]
            )

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
                    volume_trend = (
                        volume_ma / volume_ma_long if volume_ma_long > 0 else 1.0
                    )
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
                logger.warning(
                    f"Feature count mismatch: expected {self.expected_feature_count}, got {len(feature_array)}"
                )
                return np.array([])

            # Check for any remaining NaN or inf values
            if not np.all(np.isfinite(feature_array)):
                logger.warning("Non-finite features detected, replacing with defaults")
                feature_array = np.nan_to_num(
                    feature_array, nan=0.0, posinf=1.0, neginf=-1.0
                )

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
            start_idx = max(
                self.feature_window, current_idx - self.initial_training_period
            )

            for idx in range(
                start_idx, current_idx - 10
            ):  # Leave some data for validation
                features = self.extract_features(data, idx)
                if len(features) > 0:
                    target = self.create_target(data, idx)
                    training_examples.append(features)
                    training_targets.append(target)

            if len(training_examples) < self.min_training_samples:
                logger.debug(
                    f"Insufficient training samples: {len(training_examples)} < {self.min_training_samples}"
                )
                return False

            X = np.array(training_examples)
            y = np.array(training_targets)

            # Check for class balance
            unique_classes, class_counts = np.unique(y, return_counts=True)
            if len(unique_classes) < 2:
                logger.warning(
                    f"Insufficient class diversity for training: {unique_classes}"
                )
                return False

            # Initialize scaler with training data
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            # Train initial model
            assert self.model is not None  # set by _initialize_model in __init__
            self.model.fit(X_scaled, y)
            self.is_initial_trained = True
            self.is_trained = True
            self.training_samples_count = len(X)

            logger.info(
                f"Initial training completed with {len(X)} samples, classes: {dict(zip(unique_classes, class_counts, strict=False))}"
            )
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

            # Scale features using existing scaler (set during initial training,
            # which must have completed to reach this branch)
            assert self.scaler is not None
            assert self.model is not None
            X_scaled = self.scaler.transform(X)

            # Incremental update using partial_fit
            # Ensure we have all classes that the model has seen before
            # (sklearn only sets `classes_` after a fit, so it's absent from
            # the stub -- `getattr` sidesteps that without changing behavior)
            existing_classes = getattr(self.model, "classes_")  # noqa: B009
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
                logger.warning(
                    f"Insufficient data for online learning: {len(data)} < {start_idx}"
                )
                return entry_signals, exit_signals

            for idx in range(start_idx, len(data)):
                # Update model periodically
                self.update_model(data, idx)

                if not self.is_trained or self.scaler is None or self.model is None:
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
            logger.info(
                f"Generated {total_entry_signals} entry and {total_exit_signals} exit signals using online learning"
            )

        except Exception as e:
            logger.error(f"Error generating online learning signals: {e}")

        return entry_signals, exit_signals

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the online learning model.

        Returns:
            Dictionary with model information
        """
        info: dict[str, Any] = {
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

        if (
            self.model is not None
            and hasattr(self.model, "coef_")
            and self.model.coef_ is not None
        ):
            info["model_coefficients"] = self.model.coef_.tolist()

        # classes_/mean_/scale_ are fit-time-only, absent from stubs; getattr avoids that.
        if (
            self.model is not None
            and hasattr(self.model, "classes_")
            and getattr(self.model, "classes_") is not None  # noqa: B009
        ):
            info["model_classes"] = getattr(self.model, "classes_").tolist()  # noqa: B009

        if self.scaler is not None:
            info["feature_scaling"] = {
                "mean": getattr(self.scaler, "mean_").tolist()  # noqa: B009
                if hasattr(self.scaler, "mean_")
                else None,
                "scale": getattr(self.scaler, "scale_").tolist()  # noqa: B009
                if hasattr(self.scaler, "scale_")
                else None,
            }

        return info
