"""Characterization tests for `maverick.backtesting.strategies.ml.feature_engineering`.

Ported from `maverick_mcp/backtesting/strategies/ml/feature_engineering.py`
(see `.superpowers/sdd/p6-task-6-report.md` for the two dedup trims and the
one dead-code removal). `RandomForestClassifier`'s `random_state` is an
existing legacy seam (`model_params.get("random_state", 42)`); no new
seeding parameter was added here.

Uses the shared `ohlcv` fixture from `tests/backtesting/conftest.py`.
"""

import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

import maverick.backtesting.strategies.ml.feature_engineering as feature_engineering
from maverick.backtesting.strategies.ml.feature_engineering import (  # noqa: E402
    FeatureExtractor,
)
from maverick.backtesting.strategies.ml.ml_predictor import MLPredictor  # noqa: E402


def test_module_does_not_use_pandas_ta():
    assert not hasattr(feature_engineering, "ta")
    assert "pandas_ta" not in sys.modules


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
        """`extract_all_features` forward-fills, zero-fills warmup, and clips +/-inf to 0."""
        features = FeatureExtractor().extract_all_features(ohlcv)
        assert len(features) == len(ohlcv)
        assert features.shape[1] == 9 + 28 + 32 + 6  # == 75
        assert not features.isna().any().any()
        assert not np.isinf(features.to_numpy()).any()

    def test_extract_all_features_empty_input(self):
        assert FeatureExtractor().extract_all_features(pd.DataFrame()).empty

    def test_feature_prefix_is_independent_of_future_bars(self, ohlcv):
        data = ohlcv.copy()
        data.loc[data.index[12], "close"] = np.nan
        data.loc[data.index[12], "high"] = np.nan
        data.loc[data.index[12], "low"] = np.nan
        data.loc[data.index[20], "volume"] = 0

        extractor = FeatureExtractor()
        short = extractor.extract_all_features(data.iloc[:30])
        extended = extractor.extract_all_features(data.iloc[:100])

        pd.testing.assert_frame_equal(short, extended.iloc[:30])
        assert short["sma_50_ratio"].iloc[0] == 0
        assert extended["sma_50_ratio"].iloc[0] == 0
        assert np.isfinite(short.to_numpy()).all()
        pd.testing.assert_frame_equal(
            short, extractor.extract_all_features(data.iloc[:30])
        )

    def test_short_prefixes_match_extended_rsi_warmup(self):
        dates = pd.bdate_range("2024-01-01", periods=100)
        close = pd.Series(np.arange(100.0, 200.0), index=dates)
        data = pd.DataFrame(
            {
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": 1000.0,
            },
            index=dates,
        )
        extractor = FeatureExtractor()
        extended = extractor.extract_all_features(data)

        for length in (1, 10, 13, 14, 30):
            prefix = extractor.extract_all_features(data.iloc[:length])
            pd.testing.assert_frame_equal(prefix, extended.iloc[:length])

        assert (extended["rsi"].iloc[:13] == 0).all()
        assert (extended["rsi_overbought"].iloc[:13] == 0).all()
        assert (extended["rsi"].iloc[13:] == 100).all()
        assert (extended["rsi_overbought"].iloc[13:] == 1).all()

    def test_future_flat_prices_do_not_change_stochastic_prefix(self):
        dates = pd.bdate_range("2024-01-01", periods=60)
        prefix_close = np.linspace(1e-12, 3e-12, 30)
        close_values = np.concatenate([prefix_close, np.full(30, 3e-12)])
        high_values = np.concatenate([prefix_close * 1.01, np.full(30, 3e-12)])
        low_values = np.concatenate([prefix_close * 0.99, np.full(30, 3e-12)])
        data = pd.DataFrame(
            {
                "open": close_values,
                "high": high_values,
                "low": low_values,
                "close": close_values,
                "volume": 1000.0,
            },
            index=dates,
        )
        extractor = FeatureExtractor()
        prefix = extractor.extract_all_features(data.iloc[:30])
        extended = extractor.extract_all_features(data)

        pd.testing.assert_frame_equal(prefix, extended.iloc[:30])

    def test_short_frame_keeps_the_full_feature_width(self, ohlcv):
        extractor = FeatureExtractor()
        full = extractor.extract_all_features(ohlcv)
        short = extractor.extract_all_features(ohlcv.iloc[:30])
        assert list(short.columns) == list(full.columns)
        # A 50-bar lookback on 30 rows is all-NaN and fills to 0.
        assert (short["sma_50_ratio"] == 0).all()

    def test_technical_features_come_from_the_indicator_core(self, ohlcv):
        from maverick.technical import indicators

        features = FeatureExtractor().extract_technical_features(ohlcv)
        close, high, low = ohlcv["close"], ohlcv["high"], ohlcv["low"]

        expected_rsi = indicators.rsi(close, 14)
        expected_rsi.iloc[:13] = np.nan
        pd.testing.assert_series_equal(features["rsi"], expected_rsi, check_names=False)
        macd = indicators.macd(close)
        pd.testing.assert_series_equal(
            features["macd_histogram"], macd["histogram"], check_names=False
        )
        bb = indicators.bollinger(close, length=20, std=2.0)
        pd.testing.assert_series_equal(
            features["bb_middle"], bb["mid"], check_names=False
        )
        stoch = indicators.stochastic(high, low, close)
        pd.testing.assert_series_equal(
            features["stoch_k"], stoch["k"], check_names=False
        )
        pd.testing.assert_series_equal(
            features["atr"], indicators.atr(high, low, close), check_names=False
        )
        np.testing.assert_allclose(
            np.asarray(features["sma_20_ratio"], dtype=float),
            (close / indicators.sma(close, 20)).to_numpy(),
            equal_nan=True,
        )
        # Warmup rows are NaN, not placeholders.
        assert np.isnan(features["macd_histogram"].iloc[0])
        assert np.isnan(features["bb_middle"].iloc[0])
        assert np.isnan(features["stoch_k"].iloc[0])

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

    def test_prediction_does_not_fit_scaler_on_held_out_rows(self, ohlcv):
        predictor = MLPredictor(random_state=42, n_estimators=10, max_depth=5)
        predictor.train(ohlcv.iloc[:300])
        mean_before = np.asarray(predictor.scaler.mean_).copy()
        scale_before = np.asarray(predictor.scaler.scale_).copy()

        predictor.predict(ohlcv.iloc[300:])

        np.testing.assert_array_equal(predictor.scaler.mean_, mean_before)
        np.testing.assert_array_equal(predictor.scaler.scale_, scale_before)
        assert predictor.scaler.n_samples_seen_ == 300

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
