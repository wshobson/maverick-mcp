"""Characterization tests for `maverick.backtesting.strategies.ml.regime_aware`.

Ported from `maverick_mcp/backtesting/strategies/ml/regime_aware.py` (see
`.superpowers/sdd/p6-task-6-report.md` for the `AdaptiveRegimeStrategy`
dead-code removal and the new `MarketRegimeDetector.random_state` seam).

Uses the shared `ohlcv` fixture and `SilentStrategy` from
`tests/backtesting/conftest.py`.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from maverick.backtesting.strategies.ml.regime_aware import (  # noqa: E402
    RegimeAwareStrategy,
)
from maverick.backtesting.strategies.ml.regime_detector import (  # noqa: E402
    MarketRegimeDetector,
)

from .conftest import SilentStrategy


class TestMarketRegimeDetector:
    @pytest.mark.parametrize("method", ["kmeans", "hmm"])
    def test_fit_and_detect_is_stable_with_explicit_seed(self, ohlcv, method):
        """Two independently constructed, independently fitted detectors
        with the *same explicit* `random_state` must label the same data
        identically -- both the estimator and the zero-variance-feature
        noise injection (see `fit_regimes`) are seeded from it.

        This deliberately does not test two *default*-constructed
        detectors against each other: `random_state=None` preserves legacy
        behavior, which means the noise-injection branch (when it fires)
        still draws from the unseeded global `np.random` state, so two
        default instances are not guaranteed to match. That is why the new
        seam exists.
        """
        det_a = MarketRegimeDetector(
            method=method, n_regimes=3, lookback_period=50, random_state=99
        )
        det_a.fit_regimes(ohlcv)
        det_b = MarketRegimeDetector(
            method=method, n_regimes=3, lookback_period=50, random_state=99
        )
        det_b.fit_regimes(ohlcv)

        assert det_a.is_fitted and det_b.is_fitted
        regime_a = det_a.detect_current_regime(ohlcv)
        regime_b = det_b.detect_current_regime(ohlcv)
        assert regime_a == regime_b
        # Repeated calls on the same fitted detector are also stable.
        assert det_a.detect_current_regime(ohlcv) == regime_a

        probs = det_a.get_regime_probabilities(ohlcv)
        assert probs.shape == (3,)
        assert probs.sum() == pytest.approx(1.0)

    def test_new_random_state_seam_overrides_hardcoded_seed(self, ohlcv):
        """The new trailing `random_state` param feeds both the estimator
        and the zero-variance noise injection; a different seed is free to
        produce a different (but still internally stable) label.
        """
        det = MarketRegimeDetector(
            method="kmeans", n_regimes=3, lookback_period=50, random_state=123
        )
        det.fit_regimes(ohlcv)
        assert det.is_fitted
        assert det.model.random_state == 123
        # Determinism still holds for the overridden seed.
        assert det.detect_current_regime(ohlcv) == det.detect_current_regime(ohlcv)

    def test_detect_regime_threshold_uptrend(self):
        data = pd.DataFrame({"close": np.linspace(100, 130, 25)})
        assert (
            MarketRegimeDetector(method="threshold").detect_regime_threshold(data) == 2
        )

    def test_detect_regime_threshold_flat(self):
        data = pd.DataFrame({"close": np.full(25, 100.0)})
        assert (
            MarketRegimeDetector(method="threshold").detect_regime_threshold(data) == 1
        )

    def test_detect_regime_threshold_smooth_downtrend_is_sideways_not_bear(self):
        """A perfectly smooth linear decline has near-zero daily volatility,
        and the bear branch requires `vol_20 > 0.25` *and* a negative trend
        -- so this case falls through to the sideways default. This is
        legacy behavior, not a bug in the port.
        """
        data = pd.DataFrame({"close": np.linspace(130, 100, 25)})
        assert (
            MarketRegimeDetector(method="threshold").detect_regime_threshold(data) == 1
        )

    def test_detect_regime_threshold_noisy_downtrend_is_bear(self):
        rng = np.random.default_rng(3)
        close = np.linspace(130, 95, 25) + rng.normal(0, 3.0, 25)
        data = pd.DataFrame({"close": close})
        assert (
            MarketRegimeDetector(method="threshold").detect_regime_threshold(data) == 0
        )

    def test_insufficient_data_falls_back_to_threshold(self):
        """Regression test: with too little data to fit a genuine statistical model,
        `fit_regimes` must honestly switch `self.method` to `"threshold"` (matching the
        existing explicit-request/model.fit()-exception convention), not merely claim
        `is_fitted = True` while leaving `self.method` at `"kmeans"`/`"hmm"` with
        `self.scaler`/`self.model` never actually fit -- the previous behavior
        let `get_regime_probabilities` reach an unfitted `StandardScaler`,
        raise `NotFittedError`, and silently fabricate a uniform distribution
        (see `regime_detector.py`'s `fit_regimes`)."""
        det = MarketRegimeDetector(method="kmeans", n_regimes=3, lookback_period=50)
        det.fit_regimes(pd.DataFrame({"close": np.linspace(100, 110, 60)}))
        assert det.is_fitted
        assert det.method == "threshold"
        # Too few valid windows to fit -- estimator stays unfitted.
        assert det.model is not None
        assert not hasattr(det.model, "cluster_centers_")

        # Both regime detection and probability reporting must now consistently
        # use the rule-based threshold path, and probabilities must be an
        # honest one-hot -- never the old fabricated uniform distribution.
        data = pd.DataFrame({"close": np.linspace(100, 110, 60)})
        regime = det.detect_current_regime(data)
        probs = det.get_regime_probabilities(data)
        assert probs.shape == (3,)
        assert probs.sum() == pytest.approx(1.0)
        assert probs[regime] == pytest.approx(1.0)
        assert not np.allclose(probs, 1 / 3)


class TestRegimeProbabilities:
    """Regression tests for the uniform-probability defect:
    `get_regime_probabilities` must return the fitted model's real posterior
    when genuinely available, and an honest one-hot (never a fabricated
    uniform distribution) otherwise."""

    @staticmethod
    def _well_separated_series(
        n_per_segment: int = 200, seed: int = 11
    ) -> pd.DataFrame:
        """Repeated bull/bear/sideways legs -- enough data (>= the 60-sample
        fit minimum for `n_regimes=3`) and clearly distinct regimes for the
        fitted model to express real, confident, non-uniform posteriors
        rather than a degenerate/insufficient-data fit."""
        rng = np.random.default_rng(seed)
        segments = []
        for _ in range(3):
            bull = np.cumprod(1 + rng.normal(0.004, 0.008, n_per_segment))
            bear = np.cumprod(1 + rng.normal(-0.004, 0.02, n_per_segment))
            sideways = np.cumprod(1 + rng.normal(0.0, 0.005, n_per_segment))
            segments.extend([bull, bear, sideways])
        close = 100 * np.concatenate(segments)
        return pd.DataFrame(
            {
                "close": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "volume": rng.integers(1_000_000, 5_000_000, len(close)),
            }
        )

    def test_well_separated_regimes_produce_non_uniform_probabilities(self):
        data = self._well_separated_series()
        det = MarketRegimeDetector(
            method="hmm", n_regimes=3, lookback_period=50, random_state=0
        )
        det.fit_regimes(data)
        assert det.is_fitted and det.method == "hmm"

        window = data.iloc[-51:]
        regime = det.detect_current_regime(window)
        probs = det.get_regime_probabilities(window)

        assert probs.shape == (3,)
        assert probs.sum() == pytest.approx(1.0)
        assert not np.allclose(probs, 1 / 3)
        # The assigned regime must be the model's own highest-probability component.
        assert int(np.argmax(probs)) == regime

    def test_hmm_method_is_gaussian_mixture_not_hidden_markov_model(self):
        """Locks in the naming decision: `"hmm"` is documented as -- and
        actually instantiates -- `sklearn.mixture.GaussianMixture`, kept for
        backward compatibility rather than a real Hidden Markov Model."""
        det = MarketRegimeDetector(method="hmm", n_regimes=3, lookback_period=50)
        assert type(det.model).__name__ == "GaussianMixture"

    def test_probability_lookup_failure_falls_back_to_one_hot_not_uniform(self):
        """If computing real probabilities raises for any reason after a
        genuine fit, the fallback must be an honest one-hot tied to the
        actual classification decision -- never the previous `np.ones(n) / n`
        fabrication."""
        data = self._well_separated_series()
        det = MarketRegimeDetector(
            method="hmm", n_regimes=3, lookback_period=50, random_state=0
        )
        det.fit_regimes(data)
        assert det.is_fitted

        def _boom(_features):
            raise RuntimeError("simulated scaler failure")

        det.scaler.transform = _boom
        window = data.iloc[-51:]
        regime = det.detect_current_regime(window)
        probs = det.get_regime_probabilities(window)

        assert probs.sum() == pytest.approx(1.0)
        assert probs[regime] == pytest.approx(1.0)
        assert not np.allclose(probs, 1 / 3)

    def test_threshold_fallback_does_not_index_error_with_fewer_than_three_regimes(
        self,
    ):
        """Regression test (found via CodeRabbit review): `detect_regime_threshold` always
        returns one of exactly three hardcoded labels (0/1/2) regardless of `n_regimes`, so a
        detector configured with `n_regimes=2` could previously raise `IndexError` inside
        `_one_hot_via_threshold_fallback` whenever the threshold classifier returned label 2.
        The label must be clamped into `[0, n_regimes)` instead."""
        det = MarketRegimeDetector(method="threshold", n_regimes=2)
        # A clear uptrend makes `detect_regime_threshold` return 2 (bull/trending), which is
        # out of range for a 2-regime detector.
        data = pd.DataFrame({"close": np.linspace(100, 130, 25)})

        probs = det.get_regime_probabilities(data)

        assert probs.shape == (2,)
        assert probs.sum() == pytest.approx(1.0)
        assert probs[-1] == pytest.approx(1.0)


class TestRegimeAwareStrategy:
    def test_generate_signals_shape_and_regime_analysis(self, ohlcv):
        strategy = RegimeAwareStrategy(
            regime_strategies={
                0: SilentStrategy("Bear"),
                1: SilentStrategy("Side"),
                2: SilentStrategy("Bull"),
            },
            regime_detector=MarketRegimeDetector(
                method="kmeans", n_regimes=3, lookback_period=50
            ),
        )
        entry, exit_ = strategy.generate_signals(ohlcv)
        assert len(entry) == len(exit_) == len(ohlcv)
        assert not entry.any() and not exit_.any()  # every component is silent

        analysis = strategy.get_regime_analysis()
        assert analysis["total_switches"] == strategy.regime_switches
        assert sum(analysis["regime_counts"].values()) == len(ohlcv)
