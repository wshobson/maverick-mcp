"""Characterization tests for `maverick.backtesting.strategies.ml.feature_engineering`.

Ported from `maverick_mcp/backtesting/strategies/ml/feature_engineering.py`
(see `.superpowers/sdd/p6-task-6-report.md` for the two dedup trims and the
one dead-code removal). `RandomForestClassifier`'s `random_state` is an
existing legacy seam (`model_params.get("random_state", 42)`); no new
seeding parameter was added here.

Uses the shared `ohlcv` fixture from `tests/backtesting/conftest.py`.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("pandas_ta")

from maverick.backtesting.strategies.ml.feature_engineering import (  # noqa: E402
    FeatureExtractor,
)
from maverick.backtesting.strategies.ml.ml_predictor import MLPredictor  # noqa: E402


class TestFeatureExtractor:
    def test_extract_price_features_shape(self, ohlcv):
        features = FeatureExtractor().extract_price_features(ohlcv)
        assert len(features) == len(ohlcv)
        assert set(features.columns) == {
            "high_low_ratio",
            "close_open_ratio",
            "hl_spread",
            "co_spread",
            "returns",
            "log_returns",
            "volume_ma_ratio",
            "price_volume",
            "volume_returns",
        }

    def test_extract_technical_features_shape(self, ohlcv):
        features = FeatureExtractor().extract_technical_features(ohlcv)
        assert len(features) == len(ohlcv)
        # 3 cols per lookback period (default 4 periods=12) + rsi(3) + macd(4)
        # + bbands(5) + stoch(2) + atr(2) == 28
        assert features.shape[1] == 28

    def test_extract_statistical_features_shape(self, ohlcv):
        features = FeatureExtractor().extract_statistical_features(ohlcv)
        assert len(features) == len(ohlcv)
        # 8 cols per lookback period (default 4 periods) == 32
        assert features.shape[1] == 32

    def test_extract_microstructure_features_shape(self, ohlcv):
        features = FeatureExtractor().extract_microstructure_features(ohlcv)
        assert len(features) == len(ohlcv)
        assert features.shape[1] == 6

    def test_extract_all_features_nan_policy(self, ohlcv):
        """`extract_all_features` ffill/bfill/zero-fills and clips +/-inf to 0."""
        features = FeatureExtractor().extract_all_features(ohlcv)
        assert len(features) == len(ohlcv)
        assert features.shape[1] == 9 + 28 + 32 + 6  # == 75
        assert not features.isna().any().any()
        assert not np.isinf(features.to_numpy()).any()

    def test_extract_all_features_empty_input(self):
        assert FeatureExtractor().extract_all_features(pd.DataFrame()).empty

    def test_create_target_variable_exact_counts(self, ohlcv):
        """Target labeling is pure pandas comparison -- pin the exact counts
        for the fixed fixture and default `forward_periods=5`,
        `threshold=0.02`.
        """
        target = FeatureExtractor().create_target_variable(ohlcv)
        assert target.value_counts().to_dict() == {1: 200, 0: 122, 2: 78}


class TestMLPredictor:
    def test_train_is_deterministic_given_random_state(self, ohlcv):
        """Two independently constructed predictors with the same
        `random_state` must train to bit-identical metrics.
        """
        metrics_a = MLPredictor(random_state=42, n_estimators=50, max_depth=5).train(
            ohlcv
        )
        metrics_b = MLPredictor(random_state=42, n_estimators=50, max_depth=5).train(
            ohlcv
        )
        assert metrics_a["train_accuracy"] == metrics_b["train_accuracy"]
        assert metrics_a["n_samples"] == metrics_b["n_samples"] == 400
        assert metrics_a["n_features"] == metrics_b["n_features"] == 75
        assert metrics_a["target_distribution"] == {1: 200, 0: 122, 2: 78}

    def test_predict_shape_and_dtype(self, ohlcv):
        predictor = MLPredictor(random_state=42, n_estimators=50, max_depth=5)
        predictor.train(ohlcv)
        entry, exit_ = predictor.predict(ohlcv)
        assert len(entry) == len(exit_) == len(ohlcv)
        assert entry.dtype == bool
        assert exit_.dtype == bool

    def test_predict_before_train_raises(self):
        with pytest.raises(ValueError, match="must be trained"):
            MLPredictor().predict(pd.DataFrame({"close": [1.0, 2.0]}))

    def test_get_feature_importance_is_always_empty(self, ohlcv):
        """Characterizes a legacy quirk, not a fix: `get_feature_importance`
        calls `extract_all_features(pd.DataFrame())` (an *empty* frame) just
        to read off column names, but `extract_all_features` early-returns
        an empty frame for empty input (`if data is None or data.empty:
        return pd.DataFrame()`). `zip(feature_names, ..., strict=False)`
        then always yields nothing, so this method always returns `{}` --
        identical to the legacy module, preserved as-is per the port's
        no-behavior-change rule.
        """
        predictor = MLPredictor(random_state=42, n_estimators=50, max_depth=5)
        predictor.train(ohlcv)
        assert predictor.get_feature_importance() == {}
