"""Characterization tests for `maverick.backtesting.strategies.ml.ensemble`.

Ported from `maverick_mcp/backtesting/strategies/ml/ensemble.py` (see
`.superpowers/sdd/p6-task-6-report.md` for the `RiskAdjustedEnsemble`
dead-code removal). No randomness anywhere in this module -- all weighting
math is deterministic, so every assertion here is hand-computed rather than
just a determinism check.

Uses the shared `MockStrategy`/`SilentStrategy` from
`tests/backtesting/conftest.py`. No `sklearn`/`pandas_ta` dependency, but
importorskip("sklearn") is kept for consistency with the sibling
`test_ml_*` suites (this module is part of the same `ml/` package split).
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from maverick.backtesting.strategies.ml.ensemble import StrategyEnsemble  # noqa: E402

from .conftest import MockStrategy, SilentStrategy


class TestStrategyEnsemble:
    def test_calculate_performance_weights_hand_computed(self):
        ensemble = StrategyEnsemble(
            [SilentStrategy("S1"), SilentStrategy("S2")], lookback_period=10
        )
        r0 = [0.01, 0.02, -0.005, 0.015, -0.01, 0.008, 0.012, -0.004, 0.006, 0.011]
        r1 = [-0.005, 0.02, -0.015, 0.03, -0.02, 0.01, -0.008, 0.025, -0.012, 0.018]
        ensemble.strategy_returns = {0: r0, 1: r1}

        weights = ensemble.calculate_performance_weights(pd.DataFrame())

        sharpe0 = max(
            0, pd.Series(r0).mean() / (pd.Series(r0).std() + 1e-8) * np.sqrt(252)
        )
        sharpe1 = max(
            0, pd.Series(r1).mean() / (pd.Series(r1).std() + 1e-8) * np.sqrt(252)
        )
        exp_sharpe = np.exp(np.array([sharpe0, sharpe1]) * 2)
        expected = exp_sharpe / exp_sharpe.sum()

        np.testing.assert_allclose(weights, expected)

    def test_calculate_volatility_weights_hand_computed(self):
        ensemble = StrategyEnsemble(
            [SilentStrategy("S1"), SilentStrategy("S2")], lookback_period=10
        )
        r0 = [0.01, 0.02, -0.005, 0.015, -0.01, 0.008, 0.012, -0.004, 0.006, 0.011]
        r1 = [-0.005, 0.02, -0.015, 0.03, -0.02, 0.01, -0.008, 0.025, -0.012, 0.018]
        ensemble.strategy_returns = {0: r0, 1: r1}

        weights = ensemble.calculate_volatility_weights(pd.DataFrame())

        vol0 = max(0.01, pd.Series(r0).std() * np.sqrt(252))
        vol1 = max(0.01, pd.Series(r1).std() * np.sqrt(252))
        inv_vol = 1.0 / np.array([vol0, vol1])
        expected = inv_vol / inv_vol.sum()

        np.testing.assert_allclose(weights, expected)

    def test_weights_default_to_equal(self):
        ensemble = StrategyEnsemble([SilentStrategy("S1"), SilentStrategy("S2")])
        np.testing.assert_allclose(ensemble.weights, [0.5, 0.5])

    def test_combine_signals_hand_computed(self):
        """5-row weighted vote, hand-traced against `combine_signals`:

        entry_votes = 0.6*[1,0,1,0,0] + 0.4*[0,0,1,1,0] = [0.6, 0, 1.0, 0.4, 0]
        exit_votes  = 0.6*[0,0,0,0,0] + 0.4*[0,1,0,0,1] = [0,   0.4, 0,  0,   0.4]
        default entry/exit threshold is 0.5 -> combined_entry = [T,F,T,F,F]
        combined_exit stays all-False (max vote 0.4 < 0.5).
        No conflicts, no weak-signal filtering triggers (0.6/1.0 both > the
        default 0.1 `min_signal_strength`).
        """
        idx = pd.date_range("2023-01-01", periods=5)
        sig0 = (
            pd.Series([True, False, True, False, False], index=idx),
            pd.Series([False] * 5, index=idx),
        )
        sig1 = (
            pd.Series([False, False, True, True, False], index=idx),
            pd.Series([False, True, False, False, True], index=idx),
        )
        ensemble = StrategyEnsemble([SilentStrategy("S1"), SilentStrategy("S2")])
        ensemble.weights = np.array([0.6, 0.4])

        entry, exit_ = ensemble.combine_signals({0: sig0, 1: sig1})

        assert entry.tolist() == [True, False, True, False, False]
        assert exit_.tolist() == [False, False, False, False, False]

    def test_combine_signals_empty_input(self):
        entry, exit_ = StrategyEnsemble([SilentStrategy("S1")]).combine_signals({})
        assert len(entry) == 0 and len(exit_) == 0

    def test_update_weights_respects_rebalance_frequency(self):
        ensemble = StrategyEnsemble(
            [SilentStrategy("S1"), SilentStrategy("S2")],
            weighting_method="equal",
            rebalance_frequency=20,
        )
        ensemble.weights = np.array([0.9, 0.1])
        ensemble.update_weights(pd.DataFrame(), current_index=5)
        # Too soon to rebalance -- weights untouched.
        np.testing.assert_allclose(ensemble.weights, [0.9, 0.1])

        ensemble.update_weights(pd.DataFrame(), current_index=25)
        # Rebalanced with "equal" -> back to uniform.
        np.testing.assert_allclose(ensemble.weights, [0.5, 0.5])
        assert ensemble.last_rebalance == 25

    def test_generate_signals_shape(self, ohlcv):
        ensemble = StrategyEnsemble(
            [MockStrategy(step=15), MockStrategy(step=21)], rebalance_frequency=20
        )
        entry, exit_ = ensemble.generate_signals(ohlcv)
        assert len(entry) == len(exit_) == len(ohlcv)
        assert entry.dtype == bool and exit_.dtype == bool

    def test_validate_parameters(self):
        assert StrategyEnsemble([SilentStrategy("S1")]).validate_parameters()
        assert not StrategyEnsemble([]).validate_parameters()
        assert not StrategyEnsemble(
            [SilentStrategy("S1")], weighting_method="bogus"
        ).validate_parameters()
