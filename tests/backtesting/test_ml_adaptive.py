"""Characterization tests for `maverick.backtesting.strategies.ml.adaptive`.

Ported from `maverick_mcp/backtesting/strategies/ml/adaptive.py` (see
`.superpowers/sdd/p6-task-6-report.md` for the new
`AdaptiveStrategy.random_state` seam -- the only genuinely unseeded
randomness in the module; `SGDClassifier` already hardcodes
`random_state=42`).

Uses the shared `ohlcv` fixture and `MockStrategy` from
`tests/backtesting/conftest.py`.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from maverick.backtesting.strategies.ml.adaptive import AdaptiveStrategy  # noqa: E402
from maverick.backtesting.strategies.ml.hybrid_adaptive import (  # noqa: E402
    HybridAdaptiveStrategy,
)
from maverick.backtesting.strategies.ml.online_learning import (  # noqa: E402
    OnlineLearningStrategy,
)

from .conftest import MockStrategy


class TestAdaptiveStrategy:
    def test_gradient_adaptation_is_hand_computable(self):
        """No randomness in the gradient path -- exact arithmetic:
        `threshold`: 0.02 + (0.5*0.01)*0.001 == 0.020005
        `window`:    20   + (0.5*0.01)*1     == 20.005
        (`step` comes from `get_adaptable_parameters`'s explicit "step" key
        for both params, not the `abs(current_value) * 0.01` fallback.)
        """
        strategy = AdaptiveStrategy(
            MockStrategy(), adaptation_method="gradient", learning_rate=0.01
        )
        strategy.adapt_parameters_gradient(0.5)
        assert strategy.base_strategy.parameters["threshold"] == pytest.approx(0.020005)
        assert strategy.base_strategy.parameters["window"] == pytest.approx(20.005)

    def test_random_search_default_uses_global_numpy_state(self, monkeypatch):
        """`random_state=None` (the default) must preserve legacy behavior:
        draw from the *global* `np.random.normal`, not a local generator.
        """
        calls = []
        original = np.random.normal

        def spy(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        monkeypatch.setattr(np.random, "normal", spy)
        strategy = AdaptiveStrategy(MockStrategy(), adaptation_method="random_search")
        assert strategy._rng is None
        strategy.adapt_parameters_random_search()
        assert len(calls) == 2  # one draw per adaptable param present on the mock

    def test_random_search_with_seed_is_reproducible_and_isolated(self):
        """Two independently constructed strategies with the same
        `random_state` must produce identical perturbed parameters, and
        must not consume the global `np.random` state to do it.
        """
        global_state_before = np.random.get_state()[1].copy()

        strategy_a = AdaptiveStrategy(
            MockStrategy(), adaptation_method="random_search", random_state=42
        )
        strategy_a.adapt_parameters_random_search()

        assert (np.random.get_state()[1] == global_state_before).all()

        strategy_b = AdaptiveStrategy(
            MockStrategy(), adaptation_method="random_search", random_state=42
        )
        strategy_b.adapt_parameters_random_search()

        assert (
            strategy_a.base_strategy.parameters == strategy_b.base_strategy.parameters
        )
        # Matches a fresh `default_rng(42)`'s first two draws directly, in
        # `get_adaptable_parameters`'s dict order (lookback_period,
        # threshold, window, period) -- the mock only has "window" and
        # "threshold", so "threshold" draws first, "window" second.
        rng = np.random.default_rng(42)
        expected_threshold = 0.02 + rng.normal(0, abs(0.02) * 0.1)
        expected_window = 20 + rng.normal(0, abs(20) * 0.1)
        assert strategy_a.base_strategy.parameters["threshold"] == pytest.approx(
            expected_threshold
        )
        assert strategy_a.base_strategy.parameters["window"] == pytest.approx(
            expected_window
        )

    def test_generate_signals_deterministic_series(self, ohlcv):
        strategy_a = AdaptiveStrategy(
            MockStrategy(),
            adaptation_method="gradient",
            adaptation_frequency=20,
            lookback_period=30,
        )
        strategy_b = AdaptiveStrategy(
            MockStrategy(),
            adaptation_method="gradient",
            adaptation_frequency=20,
            lookback_period=30,
        )
        entry_a, exit_a = strategy_a.generate_signals(ohlcv)
        entry_b, exit_b = strategy_b.generate_signals(ohlcv)
        assert entry_a.equals(entry_b)
        assert exit_a.equals(exit_b)
        assert len(strategy_a.performance_history) > 0

    def test_reset_to_original(self):
        base = MockStrategy()
        strategy = AdaptiveStrategy(base, adaptation_method="gradient")
        strategy.adapt_parameters_gradient(0.5)
        assert strategy.base_strategy.parameters != strategy.original_parameters
        strategy.reset_to_original()
        assert strategy.base_strategy.parameters == strategy.original_parameters
        assert strategy.performance_history == []


class TestOnlineLearningStrategy:
    """`SGDClassifier` already hardcodes `random_state=42` -- deterministic
    by default with no seam needed.
    """

    def test_generate_signals_is_deterministic(self, ohlcv):
        kwargs = {
            "update_frequency": 10,
            "feature_window": 10,
            "min_training_samples": 20,
            "initial_training_period": 60,
        }
        entry_a, exit_a = OnlineLearningStrategy(**kwargs).generate_signals(ohlcv)
        entry_b, exit_b = OnlineLearningStrategy(**kwargs).generate_signals(ohlcv)
        assert entry_a.equals(entry_b)
        assert exit_a.equals(exit_b)
        assert len(entry_a) == len(ohlcv)

    def test_insufficient_data_returns_all_false(self):
        strategy = OnlineLearningStrategy(initial_training_period=200)
        data = pd.DataFrame({"close": np.linspace(100, 110, 50)})
        entry, exit_ = strategy.generate_signals(data)
        assert not entry.any() and not exit_.any()


class TestHybridAdaptiveStrategy:
    def test_generate_signals_is_deterministic(self, ohlcv):
        kwargs = {"online_learning_weight": 0.3, "adaptation_method": "gradient"}
        strategy_a = HybridAdaptiveStrategy(MockStrategy(), **kwargs)
        strategy_b = HybridAdaptiveStrategy(MockStrategy(), **kwargs)
        entry_a, exit_a = strategy_a.generate_signals(ohlcv)
        entry_b, exit_b = strategy_b.generate_signals(ohlcv)
        assert entry_a.equals(entry_b)
        assert exit_a.equals(exit_b)

    def test_hybrid_info(self, ohlcv):
        strategy = HybridAdaptiveStrategy(MockStrategy(), online_learning_weight=0.3)
        strategy.generate_signals(ohlcv)
        info = strategy.get_hybrid_info()
        assert info["online_learning_weight"] == 0.3
        assert info["base_weight"] == pytest.approx(0.7)
