"""Random-forest ML predictor for trading signals.

Split out of `feature_engineering.py` (see the Task 6 report): the ported
legacy module put `FeatureExtractor` and `MLPredictor` in one 772-line file,
over this repo's 500-line-per-module cap (`tests/structure/test_harness_rules
.py::test_files_stay_under_the_size_cap`, which explicitly recommends
splitting by responsibility). `MLPredictor` composes a `FeatureExtractor`
internally; no behavior changed by moving it here.

`RandomForestClassifier` in `_create_model` already receives an explicit
`random_state` (defaulted to 42 via `model_params.get("random_state", 42)`)
-- an existing legacy seam -- so no new seeding parameter is needed for
determinism.
"""

import logging
from typing import Any

import pandas as pd
from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from .feature_engineering import FeatureExtractor

logger = logging.getLogger(__name__)


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
        self.model: RandomForestClassifier | None = None
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
            assert self.model is not None  # set by _create_model, never left None

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

            metrics: dict[str, Any] = {
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
        assert self.model is not None  # is_trained implies _create_model ran

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
        if (
            not self.is_trained
            or self.model is None
            or not hasattr(self.model, "feature_importances_")
        ):
            return {}

        feature_names = self.feature_extractor.extract_all_features(
            pd.DataFrame()  # Empty DataFrame to get column names
        ).columns

        return dict(zip(feature_names, self.model.feature_importances_, strict=False))
