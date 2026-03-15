"""Tests for ML strategy integration with VectorBTEngine.

Verifies that VectorBTEngine dispatches to real ML strategy classes
and falls back to simple implementations on failure.
"""

import numpy as np
import pandas as pd
import pytest

from maverick_mcp.backtesting.strategies.ml.adaptive import OnlineLearningStrategy

# --- Fixtures ---


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=500, freq="B")
    base_price = 100

    # Generate realistic price data with trend and noise
    returns = np.random.normal(0.0005, 0.015, len(dates))
    prices = base_price * np.exp(np.cumsum(returns))

    data = pd.DataFrame(
        {
            "open": prices * (1 + np.random.normal(0, 0.002, len(dates))),
            "high": prices * (1 + abs(np.random.normal(0.005, 0.003, len(dates)))),
            "low": prices * (1 - abs(np.random.normal(0.005, 0.003, len(dates)))),
            "close": prices,
            "volume": np.random.randint(100000, 10000000, len(dates)).astype(float),
        },
        index=dates,
    )
    return data


@pytest.fixture
def short_ohlcv_data():
    """Generate short OHLCV data that triggers fallback paths."""
    dates = pd.date_range("2023-01-01", periods=30, freq="B")
    prices = np.linspace(100, 105, len(dates))
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.full(len(dates), 1000000.0),
        },
        index=dates,
    )


# --- OnlineLearningStrategy Tests ---


class TestOnlineLearningStrategyIntegration:
    def test_initialization(self):
        strategy = OnlineLearningStrategy(
            model_type="sgd",
            update_frequency=10,
            feature_window=20,
        )
        assert strategy.model_type == "sgd"
        assert strategy.feature_window == 20
        assert strategy.is_trained is False

    def test_generate_signals_returns_boolean_series(self, sample_ohlcv_data):
        strategy = OnlineLearningStrategy(
            initial_training_period=100,
            feature_window=20,
        )
        entries, exits = strategy.generate_signals(sample_ohlcv_data)

        assert isinstance(entries, pd.Series)
        assert isinstance(exits, pd.Series)
        assert len(entries) == len(sample_ohlcv_data)
        assert len(exits) == len(sample_ohlcv_data)
        assert entries.dtype == bool
        assert exits.dtype == bool

    def test_insufficient_data_returns_empty_signals(self, short_ohlcv_data):
        strategy = OnlineLearningStrategy(
            initial_training_period=200,
            feature_window=20,
        )
        entries, exits = strategy.generate_signals(short_ohlcv_data)

        assert entries.sum() == 0
        assert exits.sum() == 0

    def test_feature_extraction(self, sample_ohlcv_data):
        strategy = OnlineLearningStrategy(feature_window=20)
        features = strategy.extract_features(sample_ohlcv_data, 50)

        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        assert np.all(np.isfinite(features))

    def test_model_info(self):
        strategy = OnlineLearningStrategy()
        info = strategy.get_model_info()
        assert "model_type" in info
        assert "is_trained" in info
        assert info["model_type"] == "sgd"


class TestOnlineLearningEnhancedFeatures:
    def test_enhanced_feature_extraction_fallback(self, sample_ohlcv_data):
        """Test that enhanced feature extraction falls back to inline on failure."""
        strategy = OnlineLearningStrategy(feature_window=20)
        features = strategy.extract_features_enhanced(sample_ohlcv_data, 50)

        assert isinstance(features, np.ndarray)
        assert len(features) > 0

    def test_save_model_untrained(self):
        """Can't save an untrained model."""
        strategy = OnlineLearningStrategy()
        result = strategy.save_trained_model()
        assert result is False

    def test_load_model_nonexistent(self):
        """Loading a nonexistent model returns False."""
        strategy = OnlineLearningStrategy()
        result = strategy.load_trained_model("nonexistent_model_xyz")
        assert result is False


# --- VectorBTEngine ML Dispatch Tests ---


class TestVectorBTEngineMLDispatch:
    """Test that VectorBTEngine routes to ML strategies correctly."""

    def test_online_learning_dispatch(self, sample_ohlcv_data):
        """VectorBTEngine._online_learning_signals should use OnlineLearningStrategy."""
        from maverick_mcp.backtesting.vectorbt_engine import VectorBTEngine

        engine = VectorBTEngine()
        params = {"lookback": 20, "learning_rate": 0.01}

        # Should not raise
        entries, exits = engine._online_learning_signals(sample_ohlcv_data, params)
        assert isinstance(entries, pd.Series)
        assert isinstance(exits, pd.Series)
        assert len(entries) == len(sample_ohlcv_data)

    def test_regime_aware_dispatch(self, sample_ohlcv_data):
        """VectorBTEngine._regime_aware_signals should attempt RegimeAwareStrategy."""
        from maverick_mcp.backtesting.vectorbt_engine import VectorBTEngine

        engine = VectorBTEngine()
        params = {"regime_window": 50, "threshold": 0.02}

        entries, exits = engine._regime_aware_signals(sample_ohlcv_data, params)
        assert isinstance(entries, pd.Series)
        assert isinstance(exits, pd.Series)
        assert len(entries) == len(sample_ohlcv_data)

    def test_ensemble_dispatch(self, sample_ohlcv_data):
        """VectorBTEngine._ensemble_signals should attempt StrategyEnsemble."""
        from maverick_mcp.backtesting.vectorbt_engine import VectorBTEngine

        engine = VectorBTEngine()
        params = {"fast_period": 10, "slow_period": 20, "rsi_period": 14}

        entries, exits = engine._ensemble_signals(sample_ohlcv_data, params)
        assert isinstance(entries, pd.Series)
        assert isinstance(exits, pd.Series)
        assert len(entries) == len(sample_ohlcv_data)

    def test_online_learning_simple_fallback(self, sample_ohlcv_data):
        """Test the simple fallback directly."""
        from maverick_mcp.backtesting.vectorbt_engine import VectorBTEngine

        engine = VectorBTEngine()
        params = {"lookback": 20, "learning_rate": 0.01}

        entries, exits = engine._online_learning_signals_simple(
            sample_ohlcv_data, params
        )
        assert isinstance(entries, pd.Series)
        assert len(entries) == len(sample_ohlcv_data)

    def test_regime_aware_simple_fallback(self, sample_ohlcv_data):
        """Test the simple regime-aware fallback directly."""
        from maverick_mcp.backtesting.vectorbt_engine import VectorBTEngine

        engine = VectorBTEngine()
        params = {"regime_window": 50, "threshold": 0.02}

        entries, exits = engine._regime_aware_signals_simple(sample_ohlcv_data, params)
        assert isinstance(entries, pd.Series)
        assert len(entries) == len(sample_ohlcv_data)

    def test_ensemble_simple_fallback(self, sample_ohlcv_data):
        """Test the simple ensemble fallback directly."""
        from maverick_mcp.backtesting.vectorbt_engine import VectorBTEngine

        engine = VectorBTEngine()
        params = {"fast_period": 10, "slow_period": 20, "rsi_period": 14}

        entries, exits = engine._ensemble_signals_simple(sample_ohlcv_data, params)
        assert isinstance(entries, pd.Series)
        assert len(entries) == len(sample_ohlcv_data)


class TestVectorBTEngineHelperMethods:
    """Test helper methods for creating sub-strategies."""

    def test_create_regime_sub_strategies(self):
        from maverick_mcp.backtesting.vectorbt_engine import VectorBTEngine

        engine = VectorBTEngine()
        strategies = engine._create_regime_sub_strategies({})

        assert 0 in strategies  # Bear
        assert 1 in strategies  # Sideways
        assert 2 in strategies  # Bull

        for _regime_id, strategy in strategies.items():
            assert hasattr(strategy, "generate_signals")
            assert hasattr(strategy, "name")

    def test_create_ensemble_base_strategies(self):
        from maverick_mcp.backtesting.vectorbt_engine import VectorBTEngine

        engine = VectorBTEngine()
        params = {"fast_period": 10, "slow_period": 20, "rsi_period": 14}
        strategies = engine._create_ensemble_base_strategies(params)

        assert len(strategies) == 3
        for s in strategies:
            assert hasattr(s, "generate_signals")
            assert hasattr(s, "name")

    def test_regime_sub_strategies_generate_signals(self, sample_ohlcv_data):
        from maverick_mcp.backtesting.vectorbt_engine import VectorBTEngine

        engine = VectorBTEngine()
        strategies = engine._create_regime_sub_strategies({})

        for _regime_id, strategy in strategies.items():
            entries, exits = strategy.generate_signals(sample_ohlcv_data)
            assert isinstance(entries, pd.Series)
            assert isinstance(exits, pd.Series)
            assert len(entries) == len(sample_ohlcv_data)

    def test_ensemble_base_strategies_generate_signals(self, sample_ohlcv_data):
        from maverick_mcp.backtesting.vectorbt_engine import VectorBTEngine

        engine = VectorBTEngine()
        params = {"fast_period": 10, "slow_period": 20, "rsi_period": 14}
        strategies = engine._create_ensemble_base_strategies(params)

        for strategy in strategies:
            entries, exits = strategy.generate_signals(sample_ohlcv_data)
            assert isinstance(entries, pd.Series)
            assert isinstance(exits, pd.Series)


# --- Strategy Template Tests ---


class TestStrategyTemplates:
    """Verify templates have real code, not just comments."""

    def test_templates_have_imports(self):
        from maverick_mcp.backtesting.strategies.templates import STRATEGY_TEMPLATES

        for strategy_name in ("online_learning", "regime_aware", "ensemble"):
            template = STRATEGY_TEMPLATES[strategy_name]
            code = template["code"]
            assert "from maverick_mcp" in code, (
                f"{strategy_name} template should import from maverick_mcp"
            )
            assert "generate_signals" in code, (
                f"{strategy_name} template should call generate_signals"
            )
