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
        """Initialize regime detection model."""
        if self.method == "hmm":
            # Use GaussianMixture as approximation to HMM
            self.model = GaussianMixture(
                n_components=self.n_regimes, covariance_type="full", random_state=42
            )
        elif self.method == "kmeans":
            self.model = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
        elif self.method == "threshold":
            # Threshold-based regime detection
            self.model = None
        else:
            raise ValueError(f"Unsupported regime detection method: {self.method}")

    def extract_regime_features(self, data: DataFrame) -> np.ndarray:
        """Extract features for regime detection.

        Args:
            data: Price data

        Returns:
            Feature matrix for regime detection
        """
        features = []

        # Returns and volatility
        returns = data["close"].pct_change()

        # Rolling statistics
        for window in [5, 10, 20]:
            if len(returns) >= window:
                features.extend(
                    [
                        returns.rolling(window).mean().iloc[-1],
                        returns.rolling(window).std().iloc[-1],
                        returns.rolling(window).skew().iloc[-1] if window >= 3 else 0,
                        returns.rolling(window).kurt().iloc[-1] if window >= 4 else 0,
                    ]
                )
            else:
                features.extend([0, 0, 0, 0])

        # Technical indicators for regime detection
        if len(data) >= 20:
            # Trend indicators
            sma_20 = data["close"].rolling(20).mean()
            trend_strength = (data["close"] - sma_20) / sma_20
            features.append(
                trend_strength.iloc[-1] if not pd.isna(trend_strength.iloc[-1]) else 0
            )

            # Volatility regime
            vol_20 = returns.rolling(20).std()
            vol_regime = (
                vol_20.iloc[-1] / vol_20.rolling(60).mean().iloc[-1]
                if len(vol_20) >= 60
                else 1
            )
            features.append(vol_regime if not pd.isna(vol_regime) else 1)
        else:
            features.extend([0, 1])

        # Market structure features
        if "volume" in data.columns and len(data) >= 10:
            volume_trend = (
                data["volume"].rolling(10).mean().iloc[-1]
                / data["volume"].rolling(30).mean().iloc[-1]
                if len(data) >= 30
                else 1
            )
            features.append(volume_trend if not pd.isna(volume_trend) else 1)
        else:
            features.append(1)

        return np.array(features)

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
        """Fit regime detection model to historical data.

        Args:
            data: Historical price data
        """
        if self.method == "threshold":
            self.is_fitted = True
            return

        try:
            # Extract features for regime detection
            feature_list = []

            for i in range(self.lookback_period, len(data)):
                window_data = data.iloc[i - self.lookback_period : i + 1]
                features = self.extract_regime_features(window_data)

                if len(features) > 0 and not np.any(np.isnan(features)):
                    feature_list.append(features)

            if len(feature_list) < self.n_regimes:
                logger.warning(
                    f"Insufficient data for regime fitting: {len(feature_list)} samples"
                )
                self.is_fitted = True
                return

            X = np.array(feature_list)

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Fit model
            self.model.fit(X_scaled)
            self.is_fitted = True

            logger.info(f"Fitted {self.method} regime detector with {len(X)} samples")

        except Exception as e:
            logger.error(f"Error fitting regime detector: {e}")
            self.is_fitted = True  # Allow fallback to threshold method

    def detect_current_regime(self, data: DataFrame) -> int:
        """Detect current market regime.

        Args:
            data: Recent price data

        Returns:
            Regime label
        """
        if not self.is_fitted:
            logger.warning("Regime detector not fitted, using threshold method")
            return self.detect_regime_threshold(data)

        try:
            if self.method == "threshold":
                return self.detect_regime_threshold(data)

            # Extract features for current regime
            features = self.extract_regime_features(data)

            if len(features) == 0 or np.any(np.isnan(features)):
                return 1  # Default to sideways regime

            # Scale features and predict regime
            X = self.scaler.transform([features])
            regime = self.model.predict(X)[0]

            return regime

        except Exception as e:
            logger.error(f"Error detecting current regime: {e}")
            return 1  # Default to sideways regime

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

            if len(features) == 0 or np.any(np.isnan(features)):
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
            # Fit regime detector if not already done
            if not self.regime_detector.is_fitted:
                self.fit_regime_detector(data)

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
            return pd.Series(False, index=data.index), pd.Series(
                False, index=data.index
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
