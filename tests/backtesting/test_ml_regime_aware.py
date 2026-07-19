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
        det = MarketRegimeDetector(method="kmeans", n_regimes=3, lookback_period=50)
        det.fit_regimes(pd.DataFrame({"close": np.linspace(100, 110, 60)}))
        assert det.is_fitted
        # Too few valid windows to fit -- estimator stays unfitted.
        assert det.model is not None
        assert not hasattr(det.model, "cluster_centers_")


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
