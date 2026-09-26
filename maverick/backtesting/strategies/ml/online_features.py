"""Feature and target construction for `OnlineLearningStrategy`, split out of
`online_learning.py` to keep that module under this repo's 500-line-per-module
cap. `_OnlineFeaturesMixin` holds `extract_features` and `create_target`
verbatim; `OnlineLearningStrategy` inherits both, so they stay callable on it
exactly as before. No behavior change.
"""

import logging

import numpy as np
from pandas import DataFrame

logger = logging.getLogger(__name__)


class _OnlineFeaturesMixin:
    """Mixed into `OnlineLearningStrategy`; see module docstring."""

    # Declared for the type checker only: set by `OnlineLearningStrategy.__init__`.
    feature_window: int
    expected_feature_count: int | None

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
