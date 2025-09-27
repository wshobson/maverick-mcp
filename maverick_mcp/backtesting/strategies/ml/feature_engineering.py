"""Feature engineering for ML trading strategies."""

import logging
from typing import Any

import numpy as np
import pandas as pd
import pandas_ta as ta
from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract technical and statistical features for ML models."""

    def __init__(self, lookback_periods: list[int] = None):
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

        # Safe division helper function
        def safe_divide(numerator, denominator, default=0.0):
            """Safely divide two values, handling None, NaN, and zero cases."""
            if numerator is None or denominator is None:
                return default
            # Convert to numpy arrays to handle pandas Series
            num = np.asarray(numerator)
            den = np.asarray(denominator)
            # Use numpy divide with where condition for safety
            return np.divide(
                num, den, out=np.full_like(num, default, dtype=float), where=(den != 0)
            )

        # Price ratios and spreads with safe division
        features["high_low_ratio"] = safe_divide(high, low, 1.0)
        features["close_open_ratio"] = safe_divide(close, open_, 1.0)
        features["hl_spread"] = (
            safe_divide(high - low, close, 0.0)
            if high is not None and low is not None and close is not None
            else 0.0
        )
        features["co_spread"] = (
            safe_divide(close - open_, open_, 0.0)
            if close is not None and open_ is not None
            else 0.0
        )

        # Returns with safe calculation
        if close is not None:
            features["returns"] = close.pct_change().fillna(0)
            # Safe log returns calculation
            price_ratio = safe_divide(close, close.shift(1), 1.0)
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
            features["volume_ma_ratio"] = safe_divide(volume, volume_ma, 1.0)
            features["price_volume"] = close * volume
            features["volume_returns"] = volume.pct_change().fillna(0)
        else:
            features["volume_ma_ratio"] = 1.0
            features["price_volume"] = 0.0
            features["volume_returns"] = 0.0

        return features

    def extract_technical_features(self, data: DataFrame) -> DataFrame:
        """Extract technical indicator features.

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

        # Safe division helper (reused from price features)
        def safe_divide(numerator, denominator, default=0.0):
            """Safely divide two values, handling None, NaN, and zero cases."""
            if numerator is None or denominator is None:
                return default
            # Convert to numpy arrays to handle pandas Series
            num = np.asarray(numerator)
            den = np.asarray(denominator)
            # Use numpy divide with where condition for safety
            return np.divide(
                num, den, out=np.full_like(num, default, dtype=float), where=(den != 0)
            )

        # Moving averages with safe calculations
        for period in self.lookback_periods:
            if close is not None:
                sma = ta.sma(close, length=period)
                ema = ta.ema(close, length=period)

                features[f"sma_{period}_ratio"] = safe_divide(close, sma, 1.0)
                features[f"ema_{period}_ratio"] = safe_divide(close, ema, 1.0)
                features[f"sma_ema_diff_{period}"] = (
                    safe_divide(sma - ema, close, 0.0)
                    if sma is not None and ema is not None
                    else 0.0
                )
            else:
                features[f"sma_{period}_ratio"] = 1.0
                features[f"ema_{period}_ratio"] = 1.0
                features[f"sma_ema_diff_{period}"] = 0.0

        # RSI
        rsi = ta.rsi(close, length=14)
        features["rsi"] = rsi
        features["rsi_oversold"] = (rsi < 30).astype(int)
        features["rsi_overbought"] = (rsi > 70).astype(int)

        # MACD
        macd = ta.macd(close)
        if macd is not None and not macd.empty:
            macd_cols = macd.columns
            macd_col = [
                col
                for col in macd_cols
                if "MACD" in col and "h" not in col and "s" not in col.lower()
            ]
            signal_col = [
                col for col in macd_cols if "signal" in col.lower() or "MACDs" in col
            ]
            hist_col = [
                col for col in macd_cols if "hist" in col.lower() or "MACDh" in col
            ]

            if macd_col:
                features["macd"] = macd[macd_col[0]]
            else:
                features["macd"] = 0

            if signal_col:
                features["macd_signal"] = macd[signal_col[0]]
            else:
                features["macd_signal"] = 0

            if hist_col:
                features["macd_histogram"] = macd[hist_col[0]]
            else:
                features["macd_histogram"] = 0

            features["macd_bullish"] = (
                features["macd"] > features["macd_signal"]
            ).astype(int)
        else:
            features["macd"] = 0
            features["macd_signal"] = 0
            features["macd_histogram"] = 0
            features["macd_bullish"] = 0

        # Bollinger Bands
        bb = ta.bbands(close, length=20)
        if bb is not None and not bb.empty:
            # Handle different pandas_ta versions that may have different column names
            bb_cols = bb.columns
            upper_col = [
                col for col in bb_cols if "BBU" in col or "upper" in col.lower()
            ]
            middle_col = [
                col for col in bb_cols if "BBM" in col or "middle" in col.lower()
            ]
            lower_col = [
                col for col in bb_cols if "BBL" in col or "lower" in col.lower()
            ]

            if upper_col and middle_col and lower_col:
                features["bb_upper"] = bb[upper_col[0]]
                features["bb_middle"] = bb[middle_col[0]]
                features["bb_lower"] = bb[lower_col[0]]

                # Safe BB position calculation
                bb_width = features["bb_upper"] - features["bb_lower"]
                features["bb_position"] = safe_divide(
                    close - features["bb_lower"], bb_width, 0.5
                )
                features["bb_squeeze"] = safe_divide(
                    bb_width, features["bb_middle"], 0.1
                )
            else:
                # Fallback to manual calculation with safe operations
                if close is not None:
                    sma_20 = close.rolling(20).mean()
                    std_20 = close.rolling(20).std()
                    features["bb_upper"] = sma_20 + (std_20 * 2)
                    features["bb_middle"] = sma_20
                    features["bb_lower"] = sma_20 - (std_20 * 2)

                    # Safe BB calculations
                    bb_width = features["bb_upper"] - features["bb_lower"]
                    features["bb_position"] = safe_divide(
                        close - features["bb_lower"], bb_width, 0.5
                    )
                    features["bb_squeeze"] = safe_divide(
                        bb_width, features["bb_middle"], 0.1
                    )
                else:
                    features["bb_upper"] = 0
                    features["bb_middle"] = 0
                    features["bb_lower"] = 0
                    features["bb_position"] = 0.5
                    features["bb_squeeze"] = 0.1
        else:
            # Manual calculation fallback with safe operations
            if close is not None:
                sma_20 = close.rolling(20).mean()
                std_20 = close.rolling(20).std()
                features["bb_upper"] = sma_20 + (std_20 * 2)
                features["bb_middle"] = sma_20
                features["bb_lower"] = sma_20 - (std_20 * 2)

                # Safe BB calculations
                bb_width = features["bb_upper"] - features["bb_lower"]
                features["bb_position"] = safe_divide(
                    close - features["bb_lower"], bb_width, 0.5
                )
                features["bb_squeeze"] = safe_divide(
                    bb_width, features["bb_middle"], 0.1
                )
            else:
                features["bb_upper"] = 0
                features["bb_middle"] = 0
                features["bb_lower"] = 0
                features["bb_position"] = 0.5
                features["bb_squeeze"] = 0.1

        # Stochastic
        stoch = ta.stoch(high, low, close)
        if stoch is not None and not stoch.empty:
            stoch_cols = stoch.columns
            k_col = [col for col in stoch_cols if "k" in col.lower()]
            d_col = [col for col in stoch_cols if "d" in col.lower()]

            if k_col:
                features["stoch_k"] = stoch[k_col[0]]
            else:
                features["stoch_k"] = 50

            if d_col:
                features["stoch_d"] = stoch[d_col[0]]
            else:
                features["stoch_d"] = 50
        else:
            features["stoch_k"] = 50
            features["stoch_d"] = 50

        # ATR (Average True Range) with safe calculation
        if high is not None and low is not None and close is not None:
            features["atr"] = ta.atr(high, low, close)
            features["atr_ratio"] = safe_divide(
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

        # Safe division helper function
        def safe_divide(numerator, denominator, default=0.0):
            """Safely divide two values, handling None, NaN, and zero cases."""
            if numerator is None or denominator is None:
                return default
            # Convert to numpy arrays to handle pandas Series
            num = np.asarray(numerator)
            den = np.asarray(denominator)
            # Use numpy divide with where condition for safety
            return np.divide(
                num, den, out=np.full_like(num, default, dtype=float), where=(den != 0)
            )

        # Rolling statistics
        for period in self.lookback_periods:
            returns = data["close"].pct_change()

            # Volatility with safe calculations
            vol_short = returns.rolling(period).std()
            vol_long = returns.rolling(period * 2).std()
            features[f"volatility_{period}"] = vol_short
            features[f"volatility_ratio_{period}"] = safe_divide(
                vol_short, vol_long, 1.0
            )

            # Skewness and Kurtosis
            features[f"skewness_{period}"] = returns.rolling(period).skew()
            features[f"kurtosis_{period}"] = returns.rolling(period).kurt()

            # Min/Max ratios with safe division
            if "high" in data.columns and "low" in data.columns:
                rolling_high = data["high"].rolling(period).max()
                rolling_low = data["low"].rolling(period).min()
                features[f"high_ratio_{period}"] = safe_divide(
                    data["close"], rolling_high, 1.0
                )
                features[f"low_ratio_{period}"] = safe_divide(
                    data["close"], rolling_low, 1.0
                )
            else:
                features[f"high_ratio_{period}"] = 1.0
                features[f"low_ratio_{period}"] = 1.0

            # Momentum features with safe division
            features[f"momentum_{period}"] = safe_divide(
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

        # Safe division helper function
        def safe_divide(numerator, denominator, default=0.0):
            """Safely divide two values, handling None, NaN, and zero cases."""
            if numerator is None or denominator is None:
                return default
            # Convert to numpy arrays to handle pandas Series
            num = np.asarray(numerator)
            den = np.asarray(denominator)
            # Use numpy divide with where condition for safety
            return np.divide(
                num, den, out=np.full_like(num, default, dtype=float), where=(den != 0)
            )

        # Bid-ask spread proxy (high-low spread) with safe calculation
        if "high" in data.columns and "low" in data.columns:
            mid_price = (data["high"] + data["low"]) / 2
            features["spread_proxy"] = safe_divide(
                data["high"] - data["low"], mid_price, 0.02
            )
        else:
            features["spread_proxy"] = 0.02

        # Price impact measures with safe calculations
        if "volume" in data.columns:
            returns_abs = abs(data["close"].pct_change())
            features["amihud_illiquidity"] = safe_divide(
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
            features["open_gap"] = safe_divide(
                data["open"] - prev_close, prev_close, 0.0
            )
        else:
            features["open_gap"] = 0.0

        if "high" in data.columns and "low" in data.columns and "close" in data.columns:
            features["close_to_high"] = safe_divide(
                data["high"] - data["close"], data["close"], 0.0
            )
            features["close_to_low"] = safe_divide(
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
                # Create empty DataFrame with same index as fallback
                price_features = pd.DataFrame(index=data.index)

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


class MLPredictor:
    """Machine learning predictor for trading signals."""

    def __init__(self, model_type: str = "random_forest", **model_params):
        """Initialize ML predictor.

        Args:
            model_type: Type of ML model to use
            **model_params: Model parameters
        """
        self.model_type = model_type
        self.model_params = model_params
        self.model = None
        self.scaler = StandardScaler()
        self.feature_extractor = FeatureExtractor()
        self.is_trained = False

    def _create_model(self):
        """Create ML model based on type."""
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=self.model_params.get("n_estimators", 100),
                max_depth=self.model_params.get("max_depth", 10),
                random_state=self.model_params.get("random_state", 42),
                **{
                    k: v
                    for k, v in self.model_params.items()
                    if k not in ["n_estimators", "max_depth", "random_state"]
                },
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def prepare_data(
        self, data: DataFrame, target_periods: int = 5, return_threshold: float = 0.02
    ) -> tuple[DataFrame, Series]:
        """Prepare features and target for training.

        Args:
            data: OHLCV price data
            target_periods: Periods to look forward for target
            return_threshold: Return threshold for classification

        Returns:
            Tuple of (features, target)
        """
        # Extract features
        features = self.feature_extractor.extract_all_features(data)

        # Create target variable
        target = self.feature_extractor.create_target_variable(
            data, target_periods, return_threshold
        )

        # Align features and target (remove NaN values)
        valid_idx = features.dropna().index.intersection(target.dropna().index)
        features = features.loc[valid_idx]
        target = target.loc[valid_idx]

        return features, target

    def train(
        self, data: DataFrame, target_periods: int = 5, return_threshold: float = 0.02
    ) -> dict[str, Any]:
        """Train the ML model.

        Args:
            data: OHLCV price data
            target_periods: Periods to look forward for target
            return_threshold: Return threshold for classification

        Returns:
            Training metrics
        """
        try:
            # Prepare data
            features, target = self.prepare_data(data, target_periods, return_threshold)

            if len(features) == 0:
                raise ValueError("No valid training data available")

            # Create and train model
            self._create_model()

            # Scale features
            features_scaled = self.scaler.fit_transform(features)

            # Train model
            self.model.fit(features_scaled, target)
            self.is_trained = True

            # Calculate training metrics
            train_score = self.model.score(features_scaled, target)

            # Convert numpy int64 to Python int for JSON serialization
            target_dist = target.value_counts().to_dict()
            target_dist = {int(k): int(v) for k, v in target_dist.items()}

            metrics = {
                "train_accuracy": float(
                    train_score
                ),  # Convert numpy float to Python float
                "n_samples": int(len(features)),
                "n_features": int(len(features.columns)),
                "target_distribution": target_dist,
            }

            # Feature importance (if available)
            if hasattr(self.model, "feature_importances_"):
                # Convert numpy floats to Python floats
                feature_importance = {
                    str(col): float(imp)
                    for col, imp in zip(
                        features.columns, self.model.feature_importances_, strict=False
                    )
                }
                metrics["feature_importance"] = feature_importance

            logger.info(f"Model trained successfully: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise

    def generate_signals(self, data: DataFrame) -> tuple[Series, Series]:
        """Generate trading signals using the trained model.

        Alias for predict() to match the expected interface.

        Args:
            data: OHLCV price data

        Returns:
            Tuple of (entry_signals, exit_signals)
        """
        return self.predict(data)

    def predict(self, data: DataFrame) -> tuple[Series, Series]:
        """Generate trading signals using the trained model.

        Args:
            data: OHLCV price data

        Returns:
            Tuple of (entry_signals, exit_signals)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        try:
            # Extract features
            features = self.feature_extractor.extract_all_features(data)

            # Handle missing values
            features = features.ffill().fillna(0)

            # Scale features
            features_scaled = self.scaler.transform(features)

            # Make predictions
            predictions = self.model.predict(features_scaled)
            prediction_proba = self.model.predict_proba(features_scaled)

            # Convert to signals
            predictions_series = pd.Series(predictions, index=features.index)

            # Entry signals (buy predictions with high confidence)
            entry_signals = (predictions_series == 2) & (
                pd.Series(prediction_proba[:, 2], index=features.index) > 0.6
            )

            # Exit signals (sell predictions or low confidence holds)
            exit_signals = (predictions_series == 0) | (
                (predictions_series == 1)
                & (pd.Series(prediction_proba[:, 1], index=features.index) < 0.4)
            )

            return entry_signals, exit_signals

        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance from trained model.

        Returns:
            Dictionary of feature importance scores
        """
        if not self.is_trained or not hasattr(self.model, "feature_importances_"):
            return {}

        feature_names = self.feature_extractor.extract_all_features(
            pd.DataFrame()  # Empty DataFrame to get column names
        ).columns

        return dict(zip(feature_names, self.model.feature_importances_, strict=False))

    def update_model(
        self, data: DataFrame, target_periods: int = 5, return_threshold: float = 0.02
    ) -> dict[str, Any]:
        """Update model with new data (online learning simulation).

        Args:
            data: New OHLCV price data
            target_periods: Periods to look forward for target
            return_threshold: Return threshold for classification

        Returns:
            Update metrics
        """
        try:
            # For now, retrain the model with all data
            # In production, this could use partial_fit for online learning
            return self.train(data, target_periods, return_threshold)

        except Exception as e:
            logger.error(f"Error updating model: {e}")
            raise
