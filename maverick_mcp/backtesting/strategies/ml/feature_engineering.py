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

        # Price ratios and spreads
        features["high_low_ratio"] = data["high"] / data["low"]
        features["close_open_ratio"] = data["close"] / data["open"]
        features["hl_spread"] = (data["high"] - data["low"]) / data["close"]
        features["co_spread"] = (data["close"] - data["open"]) / data["open"]

        # Returns
        features["returns"] = data["close"].pct_change()
        features["log_returns"] = np.log(data["close"] / data["close"].shift(1))

        # Volume features
        if "volume" in data.columns:
            features["volume_ma_ratio"] = (
                data["volume"] / data["volume"].rolling(20).mean()
            )
            features["price_volume"] = data["close"] * data["volume"]
            features["volume_returns"] = data["volume"].pct_change()

        return features

    def extract_technical_features(self, data: DataFrame) -> DataFrame:
        """Extract technical indicator features.

        Args:
            data: OHLCV price data

        Returns:
            DataFrame with technical features
        """
        features = pd.DataFrame(index=data.index)

        # Moving averages
        for period in self.lookback_periods:
            sma = ta.sma(data["close"], length=period)
            ema = ta.ema(data["close"], length=period)

            features[f"sma_{period}_ratio"] = data["close"] / sma
            features[f"ema_{period}_ratio"] = data["close"] / ema
            features[f"sma_ema_diff_{period}"] = (sma - ema) / data["close"]

        # RSI
        rsi = ta.rsi(data["close"], length=14)
        features["rsi"] = rsi
        features["rsi_oversold"] = (rsi < 30).astype(int)
        features["rsi_overbought"] = (rsi > 70).astype(int)

        # MACD
        macd = ta.macd(data["close"])
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
        bb = ta.bbands(data["close"], length=20)
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
                features["bb_position"] = (data["close"] - features["bb_lower"]) / (
                    features["bb_upper"] - features["bb_lower"]
                )
                features["bb_squeeze"] = (
                    features["bb_upper"] - features["bb_lower"]
                ) / features["bb_middle"]
            else:
                # Fallback to manual calculation
                sma_20 = data["close"].rolling(20).mean()
                std_20 = data["close"].rolling(20).std()
                features["bb_upper"] = sma_20 + (std_20 * 2)
                features["bb_middle"] = sma_20
                features["bb_lower"] = sma_20 - (std_20 * 2)
                features["bb_position"] = (data["close"] - features["bb_lower"]) / (
                    features["bb_upper"] - features["bb_lower"]
                )
                features["bb_squeeze"] = (
                    features["bb_upper"] - features["bb_lower"]
                ) / features["bb_middle"]
        else:
            # Manual calculation fallback
            sma_20 = data["close"].rolling(20).mean()
            std_20 = data["close"].rolling(20).std()
            features["bb_upper"] = sma_20 + (std_20 * 2)
            features["bb_middle"] = sma_20
            features["bb_lower"] = sma_20 - (std_20 * 2)
            features["bb_position"] = (data["close"] - features["bb_lower"]) / (
                features["bb_upper"] - features["bb_lower"]
            )
            features["bb_squeeze"] = (
                features["bb_upper"] - features["bb_lower"]
            ) / features["bb_middle"]

        # Stochastic
        stoch = ta.stoch(data["high"], data["low"], data["close"])
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

        # ATR (Average True Range)
        features["atr"] = ta.atr(data["high"], data["low"], data["close"])
        features["atr_ratio"] = features["atr"] / data["close"]

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

            # Volatility
            features[f"volatility_{period}"] = returns.rolling(period).std()
            features[f"volatility_ratio_{period}"] = (
                features[f"volatility_{period}"] / returns.rolling(period * 2).std()
            )

            # Skewness and Kurtosis
            features[f"skewness_{period}"] = returns.rolling(period).skew()
            features[f"kurtosis_{period}"] = returns.rolling(period).kurt()

            # Min/Max ratios
            rolling_high = data["high"].rolling(period).max()
            rolling_low = data["low"].rolling(period).min()
            features[f"high_ratio_{period}"] = data["close"] / rolling_high
            features[f"low_ratio_{period}"] = data["close"] / rolling_low

            # Momentum features
            features[f"momentum_{period}"] = data["close"] / data["close"].shift(period)
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

        # Bid-ask spread proxy (high-low spread)
        features["spread_proxy"] = (data["high"] - data["low"]) / (
            (data["high"] + data["low"]) / 2
        )

        # Price impact measures
        if "volume" in data.columns:
            features["amihud_illiquidity"] = (
                abs(data["close"].pct_change()) / data["volume"]
            )
            features["volume_weighted_price"] = (
                data["high"] + data["low"] + data["close"]
            ) / 3

        # Intraday patterns
        features["open_gap"] = (data["open"] - data["close"].shift(1)) / data[
            "close"
        ].shift(1)
        features["close_to_high"] = (data["high"] - data["close"]) / data["close"]
        features["close_to_low"] = (data["close"] - data["low"]) / data["close"]

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
        forward_returns = (
            data["close"].pct_change(periods=forward_periods).shift(-forward_periods)
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
            # Extract all feature types
            price_features = self.extract_price_features(data)
            technical_features = self.extract_technical_features(data)
            statistical_features = self.extract_statistical_features(data)
            microstructure_features = self.extract_microstructure_features(data)

            # Combine all features
            all_features = pd.concat(
                [
                    price_features,
                    technical_features,
                    statistical_features,
                    microstructure_features,
                ],
                axis=1,
            )

            # Handle missing values
            all_features = all_features.ffill().fillna(0)

            logger.info(
                f"Extracted {len(all_features.columns)} features for {len(all_features)} data points"
            )

            return all_features

        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise


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

            metrics = {
                "train_accuracy": train_score,
                "n_samples": len(features),
                "n_features": len(features.columns),
                "target_distribution": target.value_counts().to_dict(),
            }

            # Feature importance (if available)
            if hasattr(self.model, "feature_importances_"):
                feature_importance = dict(
                    zip(features.columns, self.model.feature_importances_, strict=False)
                )
                metrics["feature_importance"] = feature_importance

            logger.info(f"Model trained successfully: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise

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
