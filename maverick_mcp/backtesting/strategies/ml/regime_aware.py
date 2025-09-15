"""Market regime-aware trading strategies with automatic strategy switching."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from maverick_mcp.backtesting.strategies.base import Strategy

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """Detect market regimes using various statistical methods."""

    def __init__(
        self, method: str = "hmm", n_regimes: int = 3, lookback_period: int = 50
    ):
        """Initialize regime detector.

        Args:
            method: Detection method ('hmm', 'kmeans', 'threshold')
            n_regimes: Number of market regimes to detect
            lookback_period: Period for regime detection
        """
        self.method = method
        self.n_regimes = n_regimes
        self.lookback_period = lookback_period
        self.scaler = StandardScaler()

        # Initialize detection model
        self.model = None
        self.is_fitted = False
        self._initialize_model()

    def _initialize_model(self):
        """Initialize regime detection model with better configurations."""
        if self.method == "hmm":
            # Use GaussianMixture with more stable configuration
            self.model = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type="diag",  # Use diagonal covariance for stability
                random_state=42,
                max_iter=200,
                tol=1e-6,
                reg_covar=1e-6,  # Regularization for numerical stability
                init_params='kmeans',  # Better initialization
                warm_start=False
            )
        elif self.method == "kmeans":
            self.model = KMeans(
                n_clusters=self.n_regimes,
                random_state=42,
                n_init=10,
                max_iter=500,
                tol=1e-6,
                algorithm='lloyd'  # More stable algorithm
            )
        elif self.method == "threshold":
            # Threshold-based regime detection
            self.model = None
        else:
            raise ValueError(f"Unsupported regime detection method: {self.method}")

    def extract_regime_features(self, data: DataFrame) -> np.ndarray:
        """Extract robust features for regime detection.

        Args:
            data: Price data

        Returns:
            Feature array with consistent dimensionality and stability
        """
        try:
            # Validate input data
            if data is None or data.empty or len(data) < 10:
                logger.debug("Insufficient data for regime feature extraction")
                return np.array([])

            if "close" not in data.columns:
                logger.warning("Close price data not available for regime features")
                return np.array([])

            features = []
            returns = data["close"].pct_change().dropna()

            if len(returns) == 0:
                logger.debug("No valid returns data for regime features")
                return np.array([])

            # Rolling statistics with robust error handling
            for window in [5, 10, 20]:
                if len(returns) >= window:
                    window_returns = returns.rolling(window)

                    mean_return = window_returns.mean().iloc[-1]
                    std_return = window_returns.std().iloc[-1]

                    # Robust skewness and kurtosis
                    if window >= 5:
                        skew_return = window_returns.skew().iloc[-1]
                        kurt_return = window_returns.kurt().iloc[-1]
                    else:
                        skew_return = 0.0
                        kurt_return = 0.0

                    # Replace NaN/inf values with sensible defaults
                    features.extend([
                        mean_return if np.isfinite(mean_return) else 0.0,
                        std_return if np.isfinite(std_return) else 0.01,
                        skew_return if np.isfinite(skew_return) else 0.0,
                        kurt_return if np.isfinite(kurt_return) else 0.0,
                    ])
                else:
                    # Default values for insufficient data
                    features.extend([0.0, 0.01, 0.0, 0.0])

            # Enhanced technical indicators for regime detection
            current_price = data["close"].iloc[-1]

            # Multiple timeframe trend strength
            if len(data) >= 20:
                # Short-term trend (20-day)
                sma_20 = data["close"].rolling(20).mean()
                sma_20_value = float(sma_20.iloc[-1]) if not pd.isna(sma_20.iloc[-1]) else 0.0
                if sma_20_value != 0.0:
                    trend_strength_20 = (current_price - sma_20_value) / sma_20_value
                else:
                    trend_strength_20 = 0.0
                features.append(trend_strength_20 if np.isfinite(trend_strength_20) else 0.0)

                # Price momentum (rate of change)
                prev_price = float(data["close"].iloc[-20]) if not pd.isna(data["close"].iloc[-20]) else current_price
                if prev_price != 0.0:
                    momentum_20 = (current_price - prev_price) / prev_price
                else:
                    momentum_20 = 0.0
                features.append(momentum_20 if np.isfinite(momentum_20) else 0.0)
            else:
                features.extend([0.0, 0.0])

            # Multi-timeframe volatility regime detection
            if len(returns) >= 20:
                vol_short = returns.rolling(20).std().iloc[-1] * np.sqrt(252)  # Annualized
                vol_medium = returns.rolling(60).std().iloc[-1] * np.sqrt(252) if len(returns) >= 60 else vol_short

                # Volatility regime indicator
                vol_regime = vol_short / vol_medium if vol_medium > 0 else 1.0
                features.append(vol_regime if np.isfinite(vol_regime) else 1.0)

                # Absolute volatility level (normalized)
                vol_level = min(vol_short / 0.3, 3.0)  # Cap at 3x of 30% volatility
                features.append(vol_level if np.isfinite(vol_level) else 1.0)
            else:
                features.extend([1.0, 1.0])

            # Market structure and volume features (if available)
            if "volume" in data.columns and len(data) >= 10:
                current_volume = data["volume"].iloc[-1]

                # Volume trend
                if len(data) >= 20:
                    volume_ma_short = data["volume"].rolling(10).mean().iloc[-1]
                    volume_ma_long = data["volume"].rolling(20).mean().iloc[-1]

                    volume_trend = (volume_ma_short / volume_ma_long
                                    if volume_ma_long > 0 else 1.0)
                    features.append(volume_trend if np.isfinite(volume_trend) else 1.0)

                    # Volume surge indicator
                    volume_surge = (current_volume / volume_ma_long
                                   if volume_ma_long > 0 else 1.0)
                    features.append(min(volume_surge, 10.0) if np.isfinite(volume_surge) else 1.0)
                else:
                    features.extend([1.0, 1.0])
            else:
                features.extend([1.0, 1.0])

            # Price dispersion (high-low range analysis)
            if "high" in data.columns and "low" in data.columns and len(data) >= 10:
                hl_range = (data["high"] - data["low"]) / data["close"]
                avg_range = hl_range.rolling(20).mean().iloc[-1] if len(data) >= 20 else hl_range.mean()
                current_range = hl_range.iloc[-1]

                range_regime = current_range / avg_range if avg_range > 0 else 1.0
                features.append(range_regime if np.isfinite(range_regime) else 1.0)
            else:
                features.append(1.0)

            feature_array = np.array(features)

            # Final validation and cleaning
            if len(feature_array) == 0:
                return np.array([])

            # Replace any remaining NaN/inf values
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1.0, neginf=-1.0)

            return feature_array

        except Exception as e:
            logger.error(f"Error extracting regime features: {e}")
            return np.array([])

    def detect_regime_threshold(self, data: DataFrame) -> int:
        """Detect regime using threshold-based method.

        Args:
            data: Price data

        Returns:
            Regime label (0: bear/declining, 1: sideways, 2: bull/trending)
        """
        if len(data) < 20:
            return 1  # Default to sideways

        # Calculate trend and volatility measures
        returns = data["close"].pct_change()

        # Trend measure (20-day slope)
        x = np.arange(20)
        y = data["close"].iloc[-20:].values
        trend_slope = np.polyfit(x, y, 1)[0] / y[-1]  # Normalized slope

        # Volatility measure
        vol_20 = returns.rolling(20).std().iloc[-1] * np.sqrt(252)

        # Define regime thresholds
        trend_threshold = 0.001  # 0.1% daily trend threshold
        vol_threshold = 0.25  # 25% annual volatility threshold

        # Classify regime
        if trend_slope > trend_threshold and vol_20 < vol_threshold:
            return 2  # Bull/trending market
        elif trend_slope < -trend_threshold and vol_20 > vol_threshold:
            return 0  # Bear/declining market
        else:
            return 1  # Sideways/uncertain market

    def fit_regimes(self, data: DataFrame) -> None:
        """Fit regime detection model to historical data with enhanced robustness.

        Args:
            data: Historical price data
        """
        if self.method == "threshold":
            self.is_fitted = True
            return

        try:
            # Need sufficient data for stable regime detection
            min_required_samples = max(50, self.n_regimes * 20)
            if len(data) < min_required_samples + self.lookback_period:
                logger.warning(f"Insufficient data for regime fitting: {len(data)} < {min_required_samples + self.lookback_period}")
                self.is_fitted = True
                return

            # Extract features for regime detection with temporal consistency
            feature_list = []
            feature_consistency_count = None

            # Use overlapping windows for more stable regime detection
            step_size = max(1, self.lookback_period // 10)

            for i in range(self.lookback_period, len(data), step_size):
                window_data = data.iloc[max(0, i - self.lookback_period): i + 1]
                features = self.extract_regime_features(window_data)

                if len(features) > 0 and np.all(np.isfinite(features)):
                    # Check feature consistency
                    if feature_consistency_count is None:
                        feature_consistency_count = len(features)
                    elif len(features) != feature_consistency_count:
                        logger.warning(f"Feature dimension mismatch: expected {feature_consistency_count}, got {len(features)}")
                        continue

                    feature_list.append(features)

            if len(feature_list) < min_required_samples:
                logger.warning(
                    f"Insufficient valid samples for regime fitting: {len(feature_list)} < {min_required_samples}"
                )
                self.is_fitted = True
                return

            # Ensure we have valid feature_list before creating array
            if len(feature_list) == 0:
                logger.warning("Empty feature list after filtering, cannot create feature matrix")
                self.is_fitted = True
                return

            X = np.array(feature_list)

            # Additional data quality checks
            if X.size == 0:
                logger.warning("Empty feature matrix, cannot fit regime detector")
                self.is_fitted = True
                return
            elif np.any(np.isnan(X)) or np.any(np.isinf(X)):
                logger.warning("Found NaN or inf values in feature matrix, cleaning...")
                X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

            # Check for zero variance features
            feature_std = np.std(X, axis=0)
            zero_variance_features = np.where(feature_std < 1e-8)[0]
            if len(zero_variance_features) > 0:
                logger.debug(f"Found {len(zero_variance_features)} zero-variance features")
                # Add small noise to zero-variance features
                for idx in zero_variance_features:
                    X[:, idx] += np.random.normal(0, 1e-6, X.shape[0])

            # Scale features with robust scaler
            X_scaled = self.scaler.fit_transform(X)

            # Fit model with better error handling
            try:
                if self.method == "hmm":
                    # For GaussianMixture, ensure numerical stability
                    self.model.fit(X_scaled)

                    # Validate fitted model
                    if not hasattr(self.model, 'weights_') or len(self.model.weights_) != self.n_regimes:
                        raise ValueError("Model fitting failed - invalid weights")

                    # Check convergence
                    if not self.model.converged_:
                        logger.warning("GaussianMixture did not converge, but will proceed")

                elif self.method == "kmeans":
                    self.model.fit(X_scaled)

                    # Validate fitted model
                    if not hasattr(self.model, 'cluster_centers_') or len(self.model.cluster_centers_) != self.n_regimes:
                        raise ValueError("KMeans fitting failed - invalid cluster centers")

                self.is_fitted = True

                # Log fitting success with model diagnostics
                if self.method == "hmm":
                    avg_log_likelihood = self.model.score(X_scaled) / len(X_scaled)
                    logger.info(f"Fitted {self.method} regime detector with {len(X)} samples, avg log-likelihood: {avg_log_likelihood:.4f}")
                else:
                    inertia = self.model.inertia_ if hasattr(self.model, 'inertia_') else 'N/A'
                    logger.info(f"Fitted {self.method} regime detector with {len(X)} samples, inertia: {inertia}")

            except Exception as model_error:
                logger.error(f"Model fitting failed: {model_error}")
                logger.info("Falling back to threshold method")
                self.method = "threshold"  # Fallback to threshold method
                self.is_fitted = True

        except Exception as e:
            logger.error(f"Error fitting regime detector: {e}")
            self.is_fitted = True  # Allow fallback to threshold method

    def detect_current_regime(self, data: DataFrame) -> int:
        """Detect current market regime with enhanced error handling.

        Args:
            data: Recent price data

        Returns:
            Regime label (0: bear, 1: sideways, 2: bull)
        """
        if not self.is_fitted:
            logger.debug("Regime detector not fitted, using threshold method")
            return self.detect_regime_threshold(data)

        try:
            if self.method == "threshold":
                return self.detect_regime_threshold(data)

            # Extract features for current regime
            features = self.extract_regime_features(data)

            if len(features) == 0:
                logger.debug("No features extracted, falling back to threshold method")
                return self.detect_regime_threshold(data)

            # Check for non-finite features only if features array is not empty
            if features.size > 0 and np.any(~np.isfinite(features)):
                logger.debug("Non-finite features detected, cleaning and proceeding")
                features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

            # Validate feature consistency with training
            expected_features = self.scaler.n_features_in_ if hasattr(self.scaler, 'n_features_in_') else None
            if expected_features is not None and len(features) != expected_features:
                logger.warning(f"Feature count mismatch in prediction: expected {expected_features}, got {len(features)}")
                return self.detect_regime_threshold(data)

            # Scale features and predict regime
            try:
                X = self.scaler.transform([features])
                regime = self.model.predict(X)[0]

                # Validate regime prediction
                if regime < 0 or regime >= self.n_regimes:
                    logger.warning(f"Invalid regime prediction: {regime}, using threshold method")
                    return self.detect_regime_threshold(data)

                return int(regime)

            except Exception as pred_error:
                logger.debug(f"Prediction error: {pred_error}, falling back to threshold method")
                return self.detect_regime_threshold(data)

        except Exception as e:
            logger.error(f"Error detecting current regime: {e}")
            return self.detect_regime_threshold(data)  # Always fallback to threshold

    def get_regime_probabilities(self, data: DataFrame) -> np.ndarray:
        """Get probabilities for each regime.

        Args:
            data: Recent price data

        Returns:
            Array of regime probabilities
        """
        if not self.is_fitted or self.method == "threshold":
            # For threshold method, return deterministic probabilities
            regime = self.detect_current_regime(data)
            probs = np.zeros(self.n_regimes)
            probs[regime] = 1.0
            return probs

        try:
            features = self.extract_regime_features(data)

            if len(features) == 0:
                return np.ones(self.n_regimes) / self.n_regimes
            elif features.size > 0 and np.any(np.isnan(features)):
                return np.ones(self.n_regimes) / self.n_regimes

            X = self.scaler.transform([features])

            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(X)[0]
            else:
                # For methods without probabilities, return one-hot encoding
                regime = self.model.predict(X)[0]
                probs = np.zeros(self.n_regimes)
                probs[regime] = 1.0
                return probs

        except Exception as e:
            logger.error(f"Error getting regime probabilities: {e}")
            return np.ones(self.n_regimes) / self.n_regimes


class RegimeAwareStrategy(Strategy):
    """Strategy that switches between different strategies based on market regime."""

    def __init__(
        self,
        regime_strategies: dict[int, Strategy],
        regime_detector: MarketRegimeDetector = None,
        regime_names: dict[int, str] = None,
        switch_threshold: float = 0.7,
        min_regime_duration: int = 5,
        parameters: dict[str, Any] = None,
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

        # Find most likely regime
        most_likely_regime = np.argmax(regime_probs)
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

            logger.info(
                f"Regime switch: {self.regime_names.get(old_regime, old_regime)} -> "
                f"{self.regime_names.get(self.current_regime, self.current_regime)} "
                f"(prob: {max_prob:.3f})"
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
                return pd.Series(False, index=dummy_index), pd.Series(False, index=dummy_index)

            # Ensure minimum data requirements
            min_required_data = max(50, self.regime_detector.lookback_period)
            if len(data) < min_required_data:
                logger.warning(f"Insufficient data for regime-aware strategy: {len(data)} < {min_required_data}")
                # Return all False signals but with valid data index
                return pd.Series(False, index=data.index), pd.Series(False, index=data.index)

            # Fit regime detector if not already done
            if not self.regime_detector.is_fitted:
                try:
                    self.fit_regime_detector(data)
                except Exception as e:
                    logger.error(f"Failed to fit regime detector: {e}, falling back to single strategy")
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
                return pd.Series(False, index=data.index), pd.Series(False, index=data.index)
            else:
                # Create dummy index to avoid empty array issues
                dummy_index = pd.DatetimeIndex([pd.Timestamp.now()])
                return pd.Series(False, index=dummy_index), pd.Series(False, index=dummy_index)

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
            "current_regime": self.regime_names.get(
                self.current_regime, self.current_regime
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


class AdaptiveRegimeStrategy(RegimeAwareStrategy):
    """Advanced regime-aware strategy with adaptive regime detection."""

    def __init__(
        self,
        regime_strategies: dict[int, Strategy],
        adaptation_frequency: int = 100,
        regime_confidence_threshold: float = 0.6,
        **kwargs,
    ):
        """Initialize adaptive regime strategy.

        Args:
            regime_strategies: Dictionary mapping regime labels to strategies
            adaptation_frequency: How often to re-fit regime detector
            regime_confidence_threshold: Minimum confidence for regime detection
            **kwargs: Additional parameters for RegimeAwareStrategy
        """
        super().__init__(regime_strategies, **kwargs)
        self.adaptation_frequency = adaptation_frequency
        self.regime_confidence_threshold = regime_confidence_threshold
        self.last_adaptation = 0

    @property
    def name(self) -> str:
        """Get strategy name."""
        return f"Adaptive{super().name}"

    def adapt_regime_detector(self, data: DataFrame, current_idx: int) -> None:
        """Re-fit regime detector with recent data.

        Args:
            data: Price data
            current_idx: Current index
        """
        if current_idx - self.last_adaptation < self.adaptation_frequency:
            return

        try:
            # Use recent data for adaptation
            adaptation_data = data.iloc[max(0, current_idx - 500) : current_idx]

            if len(adaptation_data) >= self.regime_detector.lookback_period:
                logger.info(f"Adapting regime detector at index {current_idx}")
                self.regime_detector.fit_regimes(adaptation_data)
                self.last_adaptation = current_idx

        except Exception as e:
            logger.error(f"Error adapting regime detector: {e}")

    def generate_signals(self, data: DataFrame) -> tuple[Series, Series]:
        """Generate adaptive regime-aware signals.

        Args:
            data: Price data with OHLCV columns

        Returns:
            Tuple of (entry_signals, exit_signals) as boolean Series
        """
        # Periodically adapt regime detector
        for idx in range(
            self.adaptation_frequency, len(data), self.adaptation_frequency
        ):
            self.adapt_regime_detector(data, idx)

        # Generate signals using parent method
        return super().generate_signals(data)
