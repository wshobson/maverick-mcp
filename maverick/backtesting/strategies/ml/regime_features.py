"""Pure regime-feature extraction, split out of `regime_aware.py`'s
`MarketRegimeDetector` (see the Task 6 report): the legacy method never
referenced `self`, so it is a standalone function here, imported by
`regime_detector.py`. Moving it out was necessary to keep
`MarketRegimeDetector` under this repo's 500-line-per-module cap -- the
method alone was 166 of that class's 510 lines. No behavior changed.
"""

import logging

import numpy as np
import pandas as pd
from pandas import DataFrame

logger = logging.getLogger(__name__)


def extract_regime_features(data: DataFrame) -> np.ndarray:
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
                features.extend(
                    [
                        mean_return if np.isfinite(mean_return) else 0.0,
                        std_return if np.isfinite(std_return) else 0.01,
                        skew_return if np.isfinite(skew_return) else 0.0,
                        kurt_return if np.isfinite(kurt_return) else 0.0,
                    ]
                )
            else:
                # Default values for insufficient data
                features.extend([0.0, 0.01, 0.0, 0.0])

        # Enhanced technical indicators for regime detection
        current_price = data["close"].iloc[-1]

        # Multiple timeframe trend strength
        if len(data) >= 20:
            # Short-term trend (20-day)
            sma_20 = data["close"].rolling(20).mean()
            sma_20_value = (
                float(sma_20.iloc[-1]) if not pd.isna(sma_20.iloc[-1]) else 0.0
            )
            if sma_20_value != 0.0:
                trend_strength_20 = (current_price - sma_20_value) / sma_20_value
            else:
                trend_strength_20 = 0.0
            features.append(
                trend_strength_20 if np.isfinite(trend_strength_20) else 0.0
            )

            # Price momentum (rate of change)
            prev_price = (
                float(data["close"].iloc[-20])
                if not pd.isna(data["close"].iloc[-20])
                else current_price
            )
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
            vol_medium = (
                returns.rolling(60).std().iloc[-1] * np.sqrt(252)
                if len(returns) >= 60
                else vol_short
            )

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

                volume_trend = (
                    volume_ma_short / volume_ma_long if volume_ma_long > 0 else 1.0
                )
                features.append(volume_trend if np.isfinite(volume_trend) else 1.0)

                # Volume surge indicator
                volume_surge = (
                    current_volume / volume_ma_long if volume_ma_long > 0 else 1.0
                )
                features.append(
                    min(volume_surge, 10.0) if np.isfinite(volume_surge) else 1.0
                )
            else:
                features.extend([1.0, 1.0])
        else:
            features.extend([1.0, 1.0])

        # Price dispersion (high-low range analysis)
        if "high" in data.columns and "low" in data.columns and len(data) >= 10:
            hl_range = (data["high"] - data["low"]) / data["close"]
            avg_range = (
                hl_range.rolling(20).mean().iloc[-1]
                if len(data) >= 20
                else hl_range.mean()
            )
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
