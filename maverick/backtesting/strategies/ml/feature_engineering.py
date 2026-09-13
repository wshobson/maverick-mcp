"""Feature engineering for ML trading strategies.

Ported from `maverick_mcp/backtesting/strategies/ml/feature_engineering.py`,
which also held `MLPredictor` (now `ml_predictor.py` -- split out to stay
under this repo's 500-line-per-module cap; see the Task 6 report).

`safe_divide` was a nested closure redefined identically inside four
methods; it is now the single module-level `_safe_divide` below.

2026-09-13: the technical features come from `maverick.technical.indicators`
instead of `pandas_ta` (see `docs/design-docs/2026-09-13-pandas-ta-removal.md`).
The pandas-ta `None`/empty fallbacks and the manual Bollinger helper are
gone; warmup rows are NaN like every other rolling feature here.
"""

import logging

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler

from maverick.technical import indicators

logger = logging.getLogger(__name__)


def _safe_divide(numerator, denominator, default=0.0):
    """Safely divide two values, handling None, NaN, and zero cases."""
    if numerator is None or denominator is None:
        return default
    num = np.asarray(numerator)
    den = np.asarray(denominator)
    return np.divide(
        num, den, out=np.full_like(num, default, dtype=float), where=(den != 0)
    )


class FeatureExtractor:
    """Extract technical and statistical features for ML models."""

    def __init__(self, lookback_periods: list[int] | None = None):
        """Initialize feature extractor.

        Args:
            lookback_periods: Lookback periods for rolling features
        """
        self.lookback_periods = lookback_periods or [5, 10, 20, 50]
        self.scaler = StandardScaler()

    def extract_price_features(self, data: DataFrame) -> DataFrame:
        """Extract price-based features.

        Args:
            data: OHLCV price data

        Returns:
            DataFrame with price features
        """
        features = pd.DataFrame(index=data.index)

        # Normalize column names to handle both cases
        high = data.get("high", data.get("High"))
        low = data.get("low", data.get("Low"))
        close = data.get("close", data.get("Close"))
        open_ = data.get("open", data.get("Open"))

        # Price ratios and spreads with safe division
        features["high_low_ratio"] = _safe_divide(high, low, 1.0)
        features["close_open_ratio"] = _safe_divide(close, open_, 1.0)
        features["hl_spread"] = (
            _safe_divide(high - low, close, 0.0)
            if high is not None and low is not None and close is not None
            else 0.0
        )
        features["co_spread"] = (
            _safe_divide(close - open_, open_, 0.0)
            if close is not None and open_ is not None
            else 0.0
        )

        # Returns with safe calculation
        if close is not None:
            features["returns"] = close.pct_change().fillna(0)
            # Safe log returns calculation
            price_ratio = _safe_divide(close, close.shift(1), 1.0)
            features["log_returns"] = np.log(
                np.maximum(price_ratio, 1e-8)
            )  # Prevent log(0)
        else:
            features["returns"] = 0
            features["log_returns"] = 0

        # Volume features with safe calculations
        volume = data.get("volume", data.get("Volume"))
        if volume is not None and close is not None:
            volume_ma = volume.rolling(20).mean()
            features["volume_ma_ratio"] = _safe_divide(volume, volume_ma, 1.0)
            features["price_volume"] = close * volume
            features["volume_returns"] = volume.pct_change().fillna(0)
        else:
            features["volume_ma_ratio"] = 1.0
            features["price_volume"] = 0.0
            features["volume_returns"] = 0.0

        return features

    def extract_technical_features(self, data: DataFrame) -> DataFrame:
        """Extract technical indicator features.

        Every indicator comes from `maverick.technical.indicators`. Warmup
        rows are NaN, the same as the other rolling features in this module.

        Args:
            data: OHLCV price data

        Returns:
            DataFrame with technical features
        """
        features = pd.DataFrame(index=data.index)

        # Normalize column names
        close = data.get("close", data.get("Close"))
        high = data.get("high", data.get("High"))
        low = data.get("low", data.get("Low"))

        # Moving averages with safe calculations
        for period in self.lookback_periods:
            if close is not None:
                sma = indicators.sma(close, period)
                ema = indicators.ema(close, period)
                features[f"sma_{period}_ratio"] = _safe_divide(close, sma, 1.0)
                features[f"ema_{period}_ratio"] = _safe_divide(close, ema, 1.0)
                features[f"sma_ema_diff_{period}"] = _safe_divide(sma - ema, close, 0.0)
            else:
                features[f"sma_{period}_ratio"] = 1.0
                features[f"ema_{period}_ratio"] = 1.0
                features[f"sma_ema_diff_{period}"] = 0.0

        # RSI
        rsi = indicators.rsi(close, 14)
        features["rsi"] = rsi
        features["rsi_oversold"] = (rsi < 30).astype(int)
        features["rsi_overbought"] = (rsi > 70).astype(int)

        # MACD
        macd = indicators.macd(close)
        features["macd"] = macd["macd"]
        features["macd_signal"] = macd["signal"]
        features["macd_histogram"] = macd["histogram"]
        features["macd_bullish"] = (features["macd"] > features["macd_signal"]).astype(
            int
        )

        # Bollinger Bands
        bb = indicators.bollinger(close, length=20, std=2.0)
        features["bb_upper"] = bb["upper"]
        features["bb_middle"] = bb["mid"]
        features["bb_lower"] = bb["lower"]
        bb_width = features["bb_upper"] - features["bb_lower"]
        features["bb_position"] = _safe_divide(
            close - features["bb_lower"], bb_width, 0.5
        )
        features["bb_squeeze"] = _safe_divide(bb_width, features["bb_middle"], 0.1)

        # Stochastic
        if high is not None and low is not None and close is not None:
            stoch = indicators.stochastic(high, low, close)
            features["stoch_k"] = stoch["k"]
            features["stoch_d"] = stoch["d"]
        else:
            features["stoch_k"] = 50
            features["stoch_d"] = 50

        # ATR (Average True Range) with safe calculation
        if high is not None and low is not None and close is not None:
            features["atr"] = indicators.atr(high, low, close)
            features["atr_ratio"] = _safe_divide(
                features["atr"], close, 0.02
            )  # Default 2% ATR ratio
        else:
            features["atr"] = 0
            features["atr_ratio"] = 0.02

        return features

    def extract_statistical_features(self, data: DataFrame) -> DataFrame:
        """Extract statistical features.

        Args:
            data: OHLCV price data

        Returns:
            DataFrame with statistical features
        """
        features = pd.DataFrame(index=data.index)

        # Rolling statistics
        for period in self.lookback_periods:
            returns = data["close"].pct_change()

            # Volatility with safe calculations
            vol_short = returns.rolling(period).std()
            vol_long = returns.rolling(period * 2).std()
            features[f"volatility_{period}"] = vol_short
            features[f"volatility_ratio_{period}"] = _safe_divide(
                vol_short, vol_long, 1.0
            )

            # Skewness and Kurtosis
            features[f"skewness_{period}"] = returns.rolling(period).skew()
            features[f"kurtosis_{period}"] = returns.rolling(period).kurt()

            # Min/Max ratios with safe division
            if "high" in data.columns and "low" in data.columns:
                rolling_high = data["high"].rolling(period).max()
                rolling_low = data["low"].rolling(period).min()
                features[f"high_ratio_{period}"] = _safe_divide(
                    data["close"], rolling_high, 1.0
                )
                features[f"low_ratio_{period}"] = _safe_divide(
                    data["close"], rolling_low, 1.0
                )
            else:
                features[f"high_ratio_{period}"] = 1.0
                features[f"low_ratio_{period}"] = 1.0

            # Momentum features with safe division
            features[f"momentum_{period}"] = _safe_divide(
                data["close"], data["close"].shift(period), 1.0
            )
            features[f"roc_{period}"] = data["close"].pct_change(periods=period)

        return features

    def extract_microstructure_features(self, data: DataFrame) -> DataFrame:
        """Extract market microstructure features.

        Args:
            data: OHLCV price data

        Returns:
            DataFrame with microstructure features
        """
        features = pd.DataFrame(index=data.index)

        # Bid-ask spread proxy (high-low spread) with safe calculation
        if "high" in data.columns and "low" in data.columns:
            mid_price = (data["high"] + data["low"]) / 2
            features["spread_proxy"] = _safe_divide(
                data["high"] - data["low"], mid_price, 0.02
            )
        else:
            features["spread_proxy"] = 0.02

        # Price impact measures with safe calculations
        if "volume" in data.columns:
            returns_abs = abs(data["close"].pct_change())
            features["amihud_illiquidity"] = _safe_divide(
                returns_abs, data["volume"], 0.0
            )

            if "high" in data.columns and "low" in data.columns:
                features["volume_weighted_price"] = (
                    data["high"] + data["low"] + data["close"]
                ) / 3
            else:
                features["volume_weighted_price"] = data["close"]
        else:
            features["amihud_illiquidity"] = 0.0
            features["volume_weighted_price"] = data.get("close", 0.0)

        # Intraday patterns with safe calculations
        if "open" in data.columns and "close" in data.columns:
            prev_close = data["close"].shift(1)
            features["open_gap"] = _safe_divide(
                data["open"] - prev_close, prev_close, 0.0
            )
        else:
            features["open_gap"] = 0.0

        if "high" in data.columns and "low" in data.columns and "close" in data.columns:
            features["close_to_high"] = _safe_divide(
                data["high"] - data["close"], data["close"], 0.0
            )
            features["close_to_low"] = _safe_divide(
                data["close"] - data["low"], data["close"], 0.0
            )
        else:
            features["close_to_high"] = 0.0
            features["close_to_low"] = 0.0

        return features

    def create_target_variable(
        self, data: DataFrame, forward_periods: int = 5, threshold: float = 0.02
    ) -> Series:
        """Create target variable for classification.

        Args:
            data: Price data
            forward_periods: Number of periods to look forward
            threshold: Return threshold for classification

        Returns:
            Target variable (0: sell, 1: hold, 2: buy)
        """
        close = data.get("close", data.get("Close"))
        forward_returns = close.pct_change(periods=forward_periods).shift(
            -forward_periods
        )

        target = pd.Series(1, index=data.index)  # Default to hold
        target[forward_returns > threshold] = 2  # Buy
        target[forward_returns < -threshold] = 0  # Sell

        return target

    def extract_all_features(self, data: DataFrame) -> DataFrame:
        """Extract all features for ML model.

        Args:
            data: OHLCV price data

        Returns:
            DataFrame with all features
        """
        try:
            # Validate input data
            if data is None or data.empty:
                logger.warning("Empty or None data provided to extract_all_features")
                return pd.DataFrame()

            # Extract all feature types with individual error handling
            feature_dfs = []

            try:
                price_features = self.extract_price_features(data)
                if not price_features.empty:
                    feature_dfs.append(price_features)
            except Exception as e:
                logger.warning(f"Failed to extract price features: {e}")

            try:
                technical_features = self.extract_technical_features(data)
                if not technical_features.empty:
                    feature_dfs.append(technical_features)
            except Exception as e:
                logger.warning(f"Failed to extract technical features: {e}")

            try:
                statistical_features = self.extract_statistical_features(data)
                if not statistical_features.empty:
                    feature_dfs.append(statistical_features)
            except Exception as e:
                logger.warning(f"Failed to extract statistical features: {e}")

            try:
                microstructure_features = self.extract_microstructure_features(data)
                if not microstructure_features.empty:
                    feature_dfs.append(microstructure_features)
            except Exception as e:
                logger.warning(f"Failed to extract microstructure features: {e}")

            # Combine all successfully extracted features
            if feature_dfs:
                all_features = pd.concat(feature_dfs, axis=1)
            else:
                # Fallback: create minimal feature set
                logger.warning(
                    "No features extracted successfully, creating minimal fallback features"
                )
                all_features = pd.DataFrame(
                    {
                        "returns": data.get("close", pd.Series(0, index=data.index))
                        .pct_change()
                        .fillna(0),
                        "close": data.get("close", pd.Series(0, index=data.index)),
                    },
                    index=data.index,
                )

            # Handle missing values with robust method
            if not all_features.empty:
                # Forward fill, then backward fill, then zero fill
                all_features = all_features.ffill().bfill().fillna(0)

                # Replace any infinite values
                all_features = all_features.replace([np.inf, -np.inf], 0)

                logger.info(
                    f"Extracted {len(all_features.columns)} features for {len(all_features)} data points"
                )
            else:
                logger.warning("No features could be extracted")

            return all_features

        except Exception as e:
            logger.error(f"Critical error extracting features: {e}")
            # Return minimal fallback instead of raising
            return pd.DataFrame(
                {
                    "returns": pd.Series(
                        0, index=data.index if data is not None else [0]
                    ),
                    "close": pd.Series(
                        0, index=data.index if data is not None else [0]
                    ),
                }
            )
