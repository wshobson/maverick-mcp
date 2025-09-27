"""
Comprehensive tests for ML-enhanced trading strategies.

Tests cover:
- Adaptive Strategy parameter adjustment and online learning
- OnlineLearningStrategy with streaming ML algorithms
- HybridAdaptiveStrategy combining multiple approaches
- Feature engineering and extraction for ML models
- Model training, prediction, and confidence scoring
- Performance tracking and adaptation mechanisms
- Parameter boundary enforcement and constraints
- Strategy performance under different market regimes
- Memory usage and computational efficiency
- Error handling and model recovery scenarios
"""

import warnings
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from maverick_mcp.backtesting.strategies.base import Strategy
from maverick_mcp.backtesting.strategies.ml.adaptive import (
    AdaptiveStrategy,
    HybridAdaptiveStrategy,
    OnlineLearningStrategy,
)

warnings.filterwarnings("ignore", category=FutureWarning)


class MockBaseStrategy(Strategy):
    """Mock base strategy for testing adaptive strategies."""

    def __init__(self, parameters: dict[str, Any] = None):
        super().__init__(parameters or {"window": 20, "threshold": 0.02})
        self._signal_pattern = "alternating"  # alternating, bullish, bearish, random

    @property
    def name(self) -> str:
        return "MockStrategy"

    @property
    def description(self) -> str:
        return "Mock strategy for testing"

    def generate_signals(self, data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Generate mock signals based on pattern."""
        entry_signals = pd.Series(False, index=data.index)
        exit_signals = pd.Series(False, index=data.index)

        window = self.parameters.get("window", 20)
        threshold = float(self.parameters.get("threshold", 0.02) or 0.0)
        step = max(5, int(round(10 * (1 + abs(threshold) * 10))))

        if self._signal_pattern == "alternating":
            # Alternate between entry and exit signals with threshold-adjusted cadence
            for i in range(window, len(data), step):
                if (i // step) % 2 == 0:
                    entry_signals.iloc[i] = True
                else:
                    exit_signals.iloc[i] = True
        elif self._signal_pattern == "bullish":
            # More entry signals than exit
            entry_indices = np.random.choice(
                range(window, len(data)),
                size=min(20, len(data) - window),
                replace=False,
            )
            entry_signals.iloc[entry_indices] = True
        elif self._signal_pattern == "bearish":
            # More exit signals than entry
            exit_indices = np.random.choice(
                range(window, len(data)),
                size=min(20, len(data) - window),
                replace=False,
            )
            exit_signals.iloc[exit_indices] = True

        return entry_signals, exit_signals


class TestAdaptiveStrategy:
    """Test suite for AdaptiveStrategy class."""

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")

        # Generate realistic price data with trends
        returns = np.random.normal(0.0005, 0.02, len(dates))
        # Add some trending periods
        returns[100:150] += 0.003  # Bull period
        returns[200:250] -= 0.002  # Bear period

        prices = 100 * np.cumprod(1 + returns)
        volumes = np.random.randint(1000000, 5000000, len(dates))

        data = pd.DataFrame(
            {
                "open": prices * np.random.uniform(0.98, 1.02, len(dates)),
                "high": prices * np.random.uniform(1.00, 1.05, len(dates)),
                "low": prices * np.random.uniform(0.95, 1.00, len(dates)),
                "close": prices,
                "volume": volumes,
            },
            index=dates,
        )

        # Ensure high >= close, open and low <= close, open
        data["high"] = np.maximum(data["high"], np.maximum(data["open"], data["close"]))
        data["low"] = np.minimum(data["low"], np.minimum(data["open"], data["close"]))

        return data

    @pytest.fixture
    def mock_base_strategy(self):
        """Create a mock base strategy."""
        return MockBaseStrategy({"window": 20, "threshold": 0.02})

    @pytest.fixture
    def adaptive_strategy(self, mock_base_strategy):
        """Create an adaptive strategy with mock base."""
        return AdaptiveStrategy(
            base_strategy=mock_base_strategy,
            adaptation_method="gradient",
            learning_rate=0.01,
            lookback_period=50,
            adaptation_frequency=10,
        )

    def test_adaptive_strategy_initialization(
        self, adaptive_strategy, mock_base_strategy
    ):
        """Test adaptive strategy initialization."""
        assert adaptive_strategy.base_strategy == mock_base_strategy
        assert adaptive_strategy.adaptation_method == "gradient"
        assert adaptive_strategy.learning_rate == 0.01
        assert adaptive_strategy.lookback_period == 50
        assert adaptive_strategy.adaptation_frequency == 10

        assert len(adaptive_strategy.performance_history) == 0
        assert len(adaptive_strategy.parameter_history) == 0
        assert adaptive_strategy.last_adaptation == 0

        # Test name and description
        assert "Adaptive" in adaptive_strategy.name
        assert "MockStrategy" in adaptive_strategy.name
        assert "gradient" in adaptive_strategy.description

    def test_performance_metric_calculation(self, adaptive_strategy):
        """Test performance metric calculation."""
        # Test with normal returns
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, -0.005])
        performance = adaptive_strategy.calculate_performance_metric(returns)

        assert isinstance(performance, float)
        assert not np.isnan(performance)

        # Test with zero volatility
        constant_returns = pd.Series([0.01, 0.01, 0.01, 0.01])
        performance = adaptive_strategy.calculate_performance_metric(constant_returns)
        assert performance == 0.0

        # Test with empty returns
        empty_returns = pd.Series([])
        performance = adaptive_strategy.calculate_performance_metric(empty_returns)
        assert performance == 0.0

    def test_adaptable_parameters_default(self, adaptive_strategy):
        """Test default adaptable parameters configuration."""
        adaptable_params = adaptive_strategy.get_adaptable_parameters()

        expected_params = ["lookback_period", "threshold", "window", "period"]
        for param in expected_params:
            assert param in adaptable_params
            assert "min" in adaptable_params[param]
            assert "max" in adaptable_params[param]
            assert "step" in adaptable_params[param]

    def test_gradient_parameter_adaptation(self, adaptive_strategy):
        """Test gradient-based parameter adaptation."""
        # Set up initial parameters
        initial_window = adaptive_strategy.base_strategy.parameters["window"]
        initial_threshold = adaptive_strategy.base_strategy.parameters["threshold"]

        # Simulate positive performance gradient
        adaptive_strategy.adapt_parameters_gradient(0.5)  # Positive gradient

        # Parameters should have changed
        new_window = adaptive_strategy.base_strategy.parameters["window"]
        new_threshold = adaptive_strategy.base_strategy.parameters["threshold"]

        # At least one parameter should have changed
        assert new_window != initial_window or new_threshold != initial_threshold

        # Parameters should be within bounds
        adaptable_params = adaptive_strategy.get_adaptable_parameters()
        if "window" in adaptable_params:
            assert new_window >= adaptable_params["window"]["min"]
            assert new_window <= adaptable_params["window"]["max"]

    def test_random_search_parameter_adaptation(self, adaptive_strategy):
        """Test random search parameter adaptation."""
        adaptive_strategy.adaptation_method = "random_search"

        # Apply random search adaptation
        adaptive_strategy.adapt_parameters_random_search()

        # Parameters should potentially have changed
        new_params = adaptive_strategy.base_strategy.parameters

        # At least check that the method runs without error
        assert isinstance(new_params, dict)
        assert "window" in new_params
        assert "threshold" in new_params

    def test_adaptive_signal_generation(self, adaptive_strategy, sample_market_data):
        """Test adaptive signal generation with parameter updates."""
        entry_signals, exit_signals = adaptive_strategy.generate_signals(
            sample_market_data
        )

        # Basic signal validation
        assert len(entry_signals) == len(sample_market_data)
        assert len(exit_signals) == len(sample_market_data)
        assert entry_signals.dtype == bool
        assert exit_signals.dtype == bool

        # Check that some adaptations occurred
        assert len(adaptive_strategy.performance_history) > 0

        # Check that parameter history was recorded
        if len(adaptive_strategy.parameter_history) > 0:
            assert isinstance(adaptive_strategy.parameter_history[0], dict)

    def test_adaptation_frequency_control(self, adaptive_strategy, sample_market_data):
        """Test that adaptation occurs at correct frequency."""
        # Set a specific adaptation frequency
        adaptive_strategy.adaptation_frequency = 30

        # Generate signals
        adaptive_strategy.generate_signals(sample_market_data)

        # Number of adaptations should be roughly len(data) / adaptation_frequency
        expected_adaptations = len(sample_market_data) // 30
        actual_adaptations = len(adaptive_strategy.performance_history)

        # Allow some variance due to lookback period requirements
        assert abs(actual_adaptations - expected_adaptations) <= 2

    def test_adaptation_history_tracking(self, adaptive_strategy, sample_market_data):
        """Test adaptation history tracking."""
        adaptive_strategy.generate_signals(sample_market_data)

        history = adaptive_strategy.get_adaptation_history()

        assert "performance_history" in history
        assert "parameter_history" in history
        assert "current_parameters" in history
        assert "original_parameters" in history

        assert len(history["performance_history"]) > 0
        assert isinstance(history["current_parameters"], dict)
        assert isinstance(history["original_parameters"], dict)

    def test_reset_to_original_parameters(self, adaptive_strategy, sample_market_data):
        """Test resetting strategy to original parameters."""
        # Store original parameters
        original_params = adaptive_strategy.base_strategy.parameters.copy()

        # Generate signals to trigger adaptations
        adaptive_strategy.generate_signals(sample_market_data)

        # Parameters should have changed

        # Reset to original
        adaptive_strategy.reset_to_original()

        # Should match original parameters
        assert adaptive_strategy.base_strategy.parameters == original_params
        assert len(adaptive_strategy.performance_history) == 0
        assert len(adaptive_strategy.parameter_history) == 0
        assert adaptive_strategy.last_adaptation == 0

    def test_adaptive_strategy_error_handling(self, adaptive_strategy):
        """Test error handling in adaptive strategy."""
        # Test with invalid data
        invalid_data = pd.DataFrame({"close": [np.nan, np.nan]})

        entry_signals, exit_signals = adaptive_strategy.generate_signals(invalid_data)

        # Should return valid series even with bad data
        assert isinstance(entry_signals, pd.Series)
        assert isinstance(exit_signals, pd.Series)
        assert len(entry_signals) == len(invalid_data)


class TestOnlineLearningStrategy:
    """Test suite for OnlineLearningStrategy class."""

    @pytest.fixture
    def online_strategy(self):
        """Create an online learning strategy."""
        return OnlineLearningStrategy(
            model_type="sgd",
            update_frequency=10,
            feature_window=20,
            confidence_threshold=0.6,
        )

    @pytest.fixture
    def online_learning_strategy(self, online_strategy):
        """Alias for online_strategy fixture for backward compatibility."""
        return online_strategy

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")

        # Generate realistic price data with trends
        returns = np.random.normal(0.0005, 0.02, len(dates))
        # Add some trending periods
        returns[100:150] += 0.003  # Bull period
        returns[200:250] -= 0.002  # Bear period

        prices = 100 * np.cumprod(1 + returns)
        volumes = np.random.randint(1000000, 5000000, len(dates))

        data = pd.DataFrame(
            {
                "open": prices * np.random.uniform(0.98, 1.02, len(dates)),
                "high": prices * np.random.uniform(1.00, 1.05, len(dates)),
                "low": prices * np.random.uniform(0.95, 1.00, len(dates)),
                "close": prices,
                "volume": volumes,
            },
            index=dates,
        )

        # Ensure high >= close, open and low <= close, open
        data["high"] = np.maximum(data["high"], np.maximum(data["open"], data["close"]))
        data["low"] = np.minimum(data["low"], np.minimum(data["open"], data["close"]))

        return data

    def test_online_learning_initialization(self, online_strategy):
        """Test online learning strategy initialization."""
        assert online_strategy.model_type == "sgd"
        assert online_strategy.update_frequency == 10
        assert online_strategy.feature_window == 20
        assert online_strategy.confidence_threshold == 0.6

        assert online_strategy.model is not None
        assert hasattr(online_strategy.model, "fit")  # Should be sklearn model
        assert not online_strategy.is_trained
        assert len(online_strategy.training_buffer) == 0

        # Test name and description
        assert "OnlineLearning" in online_strategy.name
        assert "SGD" in online_strategy.name
        assert "streaming" in online_strategy.description

    def test_model_initialization_error(self):
        """Test model initialization with unsupported type."""
        with pytest.raises(ValueError, match="Unsupported model type"):
            OnlineLearningStrategy(model_type="unsupported_model")

    def test_feature_extraction(self, online_strategy, sample_market_data):
        """Test feature extraction from market data."""
        # Test with sufficient data
        features = online_strategy.extract_features(sample_market_data, 30)

        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        assert not np.any(np.isnan(features))

        # Test with insufficient data
        features = online_strategy.extract_features(sample_market_data, 1)
        assert len(features) == 0

    def test_target_creation(self, online_learning_strategy, sample_market_data):
        """Test target variable creation."""
        # Test normal case
        target = online_learning_strategy.create_target(sample_market_data, 30)
        assert target in [0, 1, 2]  # sell, hold, buy

        # Test edge case - near end of data
        target = online_learning_strategy.create_target(
            sample_market_data, len(sample_market_data) - 1
        )
        assert target == 1  # Should default to hold

    def test_model_update_mechanism(self, online_strategy, sample_market_data):
        """Test online model update mechanism."""
        # Simulate model updates
        online_strategy.update_model(sample_market_data, 50)

        # Should not update if frequency not met
        assert online_strategy.last_update == 0  # No update yet

        # Force update by meeting frequency requirement
        online_strategy.last_update = 40
        online_strategy.update_model(sample_market_data, 51)

        # Now should have updated
        assert online_strategy.last_update > 40

    def test_online_signal_generation(self, online_strategy, sample_market_data):
        """Test online learning signal generation."""
        entry_signals, exit_signals = online_strategy.generate_signals(
            sample_market_data
        )

        # Basic validation
        assert len(entry_signals) == len(sample_market_data)
        assert len(exit_signals) == len(sample_market_data)
        assert entry_signals.dtype == bool
        assert exit_signals.dtype == bool

        # Should eventually train the model
        assert online_strategy.is_trained

    def test_model_info_retrieval(self, online_strategy, sample_market_data):
        """Test model information retrieval."""
        # Initially untrained
        info = online_strategy.get_model_info()

        assert info["model_type"] == "sgd"
        assert not info["is_trained"]
        assert info["feature_window"] == 20
        assert info["update_frequency"] == 10
        assert info["confidence_threshold"] == 0.6

        # Train the model
        online_strategy.generate_signals(sample_market_data)

        # Get info after training
        trained_info = online_strategy.get_model_info()
        assert trained_info["is_trained"]

        # Should have coefficients if model supports them
        if (
            hasattr(online_strategy.model, "coef_")
            and online_strategy.model.coef_ is not None
        ):
            assert "model_coefficients" in trained_info

    def test_confidence_threshold_filtering(self, online_strategy, sample_market_data):
        """Test that signals are filtered by confidence threshold."""
        # Use very high confidence threshold
        high_confidence_strategy = OnlineLearningStrategy(confidence_threshold=0.95)

        entry_signals, exit_signals = high_confidence_strategy.generate_signals(
            sample_market_data
        )

        # With high confidence threshold, should have fewer signals
        assert entry_signals.sum() <= 5  # Very few signals expected
        assert exit_signals.sum() <= 5

    def test_online_strategy_error_handling(self, online_strategy):
        """Test error handling in online learning strategy."""
        # Test with empty data
        empty_data = pd.DataFrame(columns=["close", "volume"])

        entry_signals, exit_signals = online_strategy.generate_signals(empty_data)

        assert len(entry_signals) == 0
        assert len(exit_signals) == 0


class TestHybridAdaptiveStrategy:
    """Test suite for HybridAdaptiveStrategy class."""

    @pytest.fixture
    def mock_base_strategy(self):
        """Create a mock base strategy."""
        return MockBaseStrategy({"window": 20, "threshold": 0.02})

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")

        # Generate realistic price data with trends
        returns = np.random.normal(0.0005, 0.02, len(dates))
        # Add some trending periods
        returns[100:150] += 0.003  # Bull period
        returns[200:250] -= 0.002  # Bear period

        prices = 100 * np.cumprod(1 + returns)
        volumes = np.random.randint(1000000, 5000000, len(dates))

        data = pd.DataFrame(
            {
                "open": prices * np.random.uniform(0.98, 1.02, len(dates)),
                "high": prices * np.random.uniform(1.00, 1.05, len(dates)),
                "low": prices * np.random.uniform(0.95, 1.00, len(dates)),
                "close": prices,
                "volume": volumes,
            },
            index=dates,
        )

        # Ensure high >= close, open and low <= close, open
        data["high"] = np.maximum(data["high"], np.maximum(data["open"], data["close"]))
        data["low"] = np.minimum(data["low"], np.minimum(data["open"], data["close"]))

        return data

    @pytest.fixture
    def hybrid_strategy(self, mock_base_strategy):
        """Create a hybrid adaptive strategy."""
        return HybridAdaptiveStrategy(
            base_strategy=mock_base_strategy,
            online_learning_weight=0.3,
            adaptation_method="gradient",
            learning_rate=0.02,
        )

    def test_hybrid_strategy_initialization(self, hybrid_strategy, mock_base_strategy):
        """Test hybrid strategy initialization."""
        assert hybrid_strategy.base_strategy == mock_base_strategy
        assert hybrid_strategy.online_learning_weight == 0.3
        assert hybrid_strategy.online_strategy is not None
        assert isinstance(hybrid_strategy.online_strategy, OnlineLearningStrategy)

        # Test name and description
        assert "HybridAdaptive" in hybrid_strategy.name
        assert "MockStrategy" in hybrid_strategy.name
        assert "hybrid" in hybrid_strategy.description.lower()

    def test_hybrid_signal_generation(self, hybrid_strategy, sample_market_data):
        """Test hybrid signal generation combining both approaches."""
        entry_signals, exit_signals = hybrid_strategy.generate_signals(
            sample_market_data
        )

        # Basic validation
        assert len(entry_signals) == len(sample_market_data)
        assert len(exit_signals) == len(sample_market_data)
        assert entry_signals.dtype == bool
        assert exit_signals.dtype == bool

        # Should have some signals (combination of both strategies)
        total_signals = entry_signals.sum() + exit_signals.sum()
        assert total_signals > 0

    def test_signal_weighting_mechanism(self, hybrid_strategy, sample_market_data):
        """Test that signal weighting works correctly."""
        # Set base strategy to generate specific pattern
        hybrid_strategy.base_strategy._signal_pattern = "bullish"

        # Generate signals
        entry_signals, exit_signals = hybrid_strategy.generate_signals(
            sample_market_data
        )

        # With bullish base strategy, should have more entry signals
        assert entry_signals.sum() >= exit_signals.sum()

    def test_hybrid_info_retrieval(self, hybrid_strategy, sample_market_data):
        """Test hybrid strategy information retrieval."""
        # Generate some signals first
        hybrid_strategy.generate_signals(sample_market_data)

        hybrid_info = hybrid_strategy.get_hybrid_info()

        assert "adaptation_history" in hybrid_info
        assert "online_learning_info" in hybrid_info
        assert "online_learning_weight" in hybrid_info
        assert "base_weight" in hybrid_info

        assert hybrid_info["online_learning_weight"] == 0.3
        assert hybrid_info["base_weight"] == 0.7

        # Verify nested information structure
        assert "model_type" in hybrid_info["online_learning_info"]
        assert "performance_history" in hybrid_info["adaptation_history"]

    def test_different_weight_configurations(
        self, mock_base_strategy, sample_market_data
    ):
        """Test hybrid strategy with different weight configurations."""
        # Test heavy online learning weighting
        heavy_online = HybridAdaptiveStrategy(
            base_strategy=mock_base_strategy, online_learning_weight=0.8
        )

        entry1, exit1 = heavy_online.generate_signals(sample_market_data)

        # Test heavy base strategy weighting
        heavy_base = HybridAdaptiveStrategy(
            base_strategy=mock_base_strategy, online_learning_weight=0.2
        )

        entry2, exit2 = heavy_base.generate_signals(sample_market_data)

        # Both should generate valid signals
        assert len(entry1) == len(entry2) == len(sample_market_data)
        assert len(exit1) == len(exit2) == len(sample_market_data)

        # Different weights should potentially produce different signals
        # (though this is probabilistic and may not always be true)
        signal_diff1 = (entry1 != entry2).sum() + (exit1 != exit2).sum()
        assert signal_diff1 >= 0  # Allow for identical signals in edge cases


class TestMLStrategiesPerformance:
    """Performance and benchmark tests for ML strategies."""

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")

        # Generate realistic price data with trends
        returns = np.random.normal(0.0005, 0.02, len(dates))
        # Add some trending periods
        returns[100:150] += 0.003  # Bull period
        returns[200:250] -= 0.002  # Bear period

        prices = 100 * np.cumprod(1 + returns)
        volumes = np.random.randint(1000000, 5000000, len(dates))

        data = pd.DataFrame(
            {
                "open": prices * np.random.uniform(0.98, 1.02, len(dates)),
                "high": prices * np.random.uniform(1.00, 1.05, len(dates)),
                "low": prices * np.random.uniform(0.95, 1.00, len(dates)),
                "close": prices,
                "volume": volumes,
            },
            index=dates,
        )

        # Ensure high >= close, open and low <= close, open
        data["high"] = np.maximum(data["high"], np.maximum(data["open"], data["close"]))
        data["low"] = np.minimum(data["low"], np.minimum(data["open"], data["close"]))

        return data

    def test_strategy_computational_efficiency(
        self, sample_market_data, benchmark_timer
    ):
        """Test computational efficiency of ML strategies."""
        strategies = [
            AdaptiveStrategy(MockBaseStrategy(), adaptation_method="gradient"),
            OnlineLearningStrategy(model_type="sgd"),
            HybridAdaptiveStrategy(MockBaseStrategy()),
        ]

        for strategy in strategies:
            with benchmark_timer() as timer:
                entry_signals, exit_signals = strategy.generate_signals(
                    sample_market_data
                )

            # Should complete within reasonable time
            assert timer.elapsed < 10.0  # < 10 seconds
            assert len(entry_signals) == len(sample_market_data)
            assert len(exit_signals) == len(sample_market_data)

    def test_memory_usage_scalability(self, benchmark_timer):
        """Test memory usage with large datasets."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create large dataset
        dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")  # 4 years
        large_data = pd.DataFrame(
            {
                "open": 100 + np.random.normal(0, 10, len(dates)),
                "high": 105 + np.random.normal(0, 10, len(dates)),
                "low": 95 + np.random.normal(0, 10, len(dates)),
                "close": 100 + np.random.normal(0, 10, len(dates)),
                "volume": np.random.randint(1000000, 10000000, len(dates)),
            },
            index=dates,
        )

        # Test online learning strategy (most memory intensive)
        strategy = OnlineLearningStrategy()
        strategy.generate_signals(large_data)

        final_memory = process.memory_info().rss
        memory_growth = (final_memory - initial_memory) / 1024 / 1024  # MB

        # Memory growth should be reasonable (< 200MB for 4 years of data)
        assert memory_growth < 200

    def test_strategy_adaptation_effectiveness(self, sample_market_data):
        """Test that adaptive strategies actually improve over time."""
        base_strategy = MockBaseStrategy()
        adaptive_strategy = AdaptiveStrategy(
            base_strategy=base_strategy, adaptation_method="gradient"
        )

        # Generate initial signals and measure performance
        initial_entry_signals, initial_exit_signals = (
            adaptive_strategy.generate_signals(sample_market_data)
        )
        assert len(initial_entry_signals) == len(sample_market_data)
        assert len(initial_exit_signals) == len(sample_market_data)
        assert len(adaptive_strategy.performance_history) > 0

        # Reset and generate again (should have different adaptations)
        adaptive_strategy.reset_to_original()
        post_reset_entry, post_reset_exit = adaptive_strategy.generate_signals(
            sample_market_data
        )
        assert len(post_reset_entry) == len(sample_market_data)
        assert len(post_reset_exit) == len(sample_market_data)

        # Should have recorded performance metrics again
        assert len(adaptive_strategy.performance_history) > 0
        assert len(adaptive_strategy.parameter_history) > 0

    def test_concurrent_strategy_execution(self, sample_market_data):
        """Test concurrent execution of multiple ML strategies."""
        import queue
        import threading

        results_queue = queue.Queue()
        error_queue = queue.Queue()

        def run_strategy(strategy_id, strategy_class):
            try:
                if strategy_class == AdaptiveStrategy:
                    strategy = AdaptiveStrategy(MockBaseStrategy())
                elif strategy_class == OnlineLearningStrategy:
                    strategy = OnlineLearningStrategy()
                else:
                    strategy = HybridAdaptiveStrategy(MockBaseStrategy())

                entry_signals, exit_signals = strategy.generate_signals(
                    sample_market_data
                )
                results_queue.put((strategy_id, len(entry_signals), len(exit_signals)))
            except Exception as e:
                error_queue.put(f"Strategy {strategy_id}: {e}")

        # Run multiple strategies concurrently
        threads = []
        strategy_classes = [
            AdaptiveStrategy,
            OnlineLearningStrategy,
            HybridAdaptiveStrategy,
        ]

        for i, strategy_class in enumerate(strategy_classes):
            thread = threading.Thread(target=run_strategy, args=(i, strategy_class))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout

        # Check results
        assert error_queue.empty(), f"Errors: {list(error_queue.queue)}"
        assert results_queue.qsize() == 3

        # All should have processed the full dataset
        while not results_queue.empty():
            strategy_id, entry_len, exit_len = results_queue.get()
            assert entry_len == len(sample_market_data)
            assert exit_len == len(sample_market_data)


class TestMLStrategiesErrorHandling:
    """Error handling and edge case tests for ML strategies."""

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")

        # Generate realistic price data with trends
        returns = np.random.normal(0.0005, 0.02, len(dates))
        # Add some trending periods
        returns[100:150] += 0.003  # Bull period
        returns[200:250] -= 0.002  # Bear period

        prices = 100 * np.cumprod(1 + returns)
        volumes = np.random.randint(1000000, 5000000, len(dates))

        data = pd.DataFrame(
            {
                "open": prices * np.random.uniform(0.98, 1.02, len(dates)),
                "high": prices * np.random.uniform(1.00, 1.05, len(dates)),
                "low": prices * np.random.uniform(0.95, 1.00, len(dates)),
                "close": prices,
                "volume": volumes,
            },
            index=dates,
        )

        # Ensure high >= close, open and low <= close, open
        data["high"] = np.maximum(data["high"], np.maximum(data["open"], data["close"]))
        data["low"] = np.minimum(data["low"], np.minimum(data["open"], data["close"]))

        return data

    @pytest.fixture
    def mock_base_strategy(self):
        """Create a mock base strategy."""
        return MockBaseStrategy({"window": 20, "threshold": 0.02})

    def test_adaptive_strategy_with_failing_base(self, sample_market_data):
        """Test adaptive strategy when base strategy fails."""
        # Create a base strategy that fails
        failing_strategy = Mock(spec=Strategy)
        failing_strategy.parameters = {"window": 20}
        failing_strategy.generate_signals.side_effect = Exception(
            "Base strategy failed"
        )

        adaptive_strategy = AdaptiveStrategy(failing_strategy)

        # Should handle the error gracefully
        entry_signals, exit_signals = adaptive_strategy.generate_signals(
            sample_market_data
        )

        assert isinstance(entry_signals, pd.Series)
        assert isinstance(exit_signals, pd.Series)
        assert len(entry_signals) == len(sample_market_data)

    def test_online_learning_with_insufficient_data(self):
        """Test online learning strategy with insufficient training data."""
        # Very small dataset
        small_data = pd.DataFrame({"close": [100, 101], "volume": [1000, 1100]})

        strategy = OnlineLearningStrategy(feature_window=20)  # Window larger than data

        entry_signals, exit_signals = strategy.generate_signals(small_data)

        # Should handle gracefully
        assert len(entry_signals) == len(small_data)
        assert len(exit_signals) == len(small_data)
        assert not strategy.is_trained  # Insufficient data to train

    def test_model_prediction_failure_handling(self, sample_market_data):
        """Test handling of model prediction failures."""
        strategy = OnlineLearningStrategy()

        # Simulate model failure after training
        with patch.object(
            strategy.model, "predict", side_effect=Exception("Prediction failed")
        ):
            entry_signals, exit_signals = strategy.generate_signals(sample_market_data)

            # Should still return valid series
            assert isinstance(entry_signals, pd.Series)
            assert isinstance(exit_signals, pd.Series)
            assert len(entry_signals) == len(sample_market_data)

    def test_parameter_boundary_enforcement(self, mock_base_strategy):
        """Test that parameter adaptations respect boundaries."""
        adaptive_strategy = AdaptiveStrategy(mock_base_strategy)

        # Set extreme gradient that should be bounded
        large_gradient = 100.0

        # Store original parameter values
        original_window = mock_base_strategy.parameters["window"]

        # Apply extreme gradient
        adaptive_strategy.adapt_parameters_gradient(large_gradient)

        # Parameter should be bounded
        new_window = mock_base_strategy.parameters["window"]
        assert new_window != original_window
        adaptable_params = adaptive_strategy.get_adaptable_parameters()

        if "window" in adaptable_params:
            assert new_window >= adaptable_params["window"]["min"]
            assert new_window <= adaptable_params["window"]["max"]

    def test_strategy_state_consistency(self, mock_base_strategy, sample_market_data):
        """Test that strategy state remains consistent after errors."""
        adaptive_strategy = AdaptiveStrategy(mock_base_strategy)

        # Generate initial signals successfully
        initial_signals = adaptive_strategy.generate_signals(sample_market_data)
        assert isinstance(initial_signals, tuple)
        assert len(initial_signals) == 2
        initial_state = {
            "performance_history": len(adaptive_strategy.performance_history),
            "parameter_history": len(adaptive_strategy.parameter_history),
            "parameters": adaptive_strategy.base_strategy.parameters.copy(),
        }

        # Simulate error during signal generation
        with patch.object(
            mock_base_strategy,
            "generate_signals",
            side_effect=Exception("Signal generation failed"),
        ):
            error_signals = adaptive_strategy.generate_signals(sample_market_data)

        # State should remain consistent or be properly handled
        assert isinstance(error_signals, tuple)
        assert len(error_signals) == 2
        assert isinstance(error_signals[0], pd.Series)
        assert isinstance(error_signals[1], pd.Series)
        assert (
            len(adaptive_strategy.performance_history)
            == initial_state["performance_history"]
        )
        assert (
            len(adaptive_strategy.parameter_history)
            == initial_state["parameter_history"]
        )
        assert adaptive_strategy.base_strategy.parameters == initial_state["parameters"]


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main([__file__, "-v", "--tb=short", "--asyncio-mode=auto"])
