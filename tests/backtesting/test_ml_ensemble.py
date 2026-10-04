"""Characterization tests for `maverick.backtesting.strategies.ml.ensemble`.

Ported from `maverick_mcp/backtesting/strategies/ml/ensemble.py` (see
`.superpowers/sdd/p6-task-6-report.md` for the `RiskAdjustedEnsemble`
dead-code removal). No randomness anywhere in this module -- all weighting
math is deterministic, so every assertion here is hand-computed rather than
just a determinism check.

Uses the shared `MockStrategy`/`SilentStrategy` from
`tests/backtesting/conftest.py`. No `sklearn` dependency, but
importorskip("sklearn") is kept for consistency with the sibling `test_ml_*`
suites (this module is part of the same `ml/` package split).
"""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
vbt = pytest.importorskip("vectorbt")

from maverick.backtesting.service_support import TemplateStrategy  # noqa: E402
from maverick.backtesting.strategies.ml.ensemble import StrategyEnsemble  # noqa: E402

from .conftest import MockStrategy, SilentStrategy, _make_ohlcv  # noqa: E402


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


@pytest.fixture
def causal_prices():
    return _make_ohlcv(n=300)


def _template_ensemble(weighting_method):
    return StrategyEnsemble(
        [TemplateStrategy(kind) for kind in ("sma_cross", "rsi", "macd")],
        weighting_method=weighting_method,
        parameters={"entry_threshold": 0.3, "exit_threshold": 0.3},
    )


@pytest.mark.parametrize("weighting_method", ["performance", "volatility", "equal"])
@pytest.mark.parametrize("reuse", [False, True])
def test_future_suffix_cannot_change_prefix_signals(
    causal_prices, weighting_method, reuse
):
    changed = causal_prices.copy()
    suffix = np.arange(100)
    changed.iloc[200:, changed.columns.get_loc("close")] = causal_prices["close"].iloc[
        199
    ] * (1 + 0.005 * suffix) + 12 * np.sin(suffix / 2)
    ensemble = _template_ensemble(weighting_method)
    before = ensemble.generate_signals(causal_prices)
    if not reuse:
        ensemble = _template_ensemble(weighting_method)
    after = ensemble.generate_signals(changed)
    for original, mutated in zip(before, after, strict=True):
        pd.testing.assert_series_equal(original.iloc[:200], mutated.iloc[:200])


@pytest.mark.parametrize("weighting_method", ["performance", "volatility", "equal"])
@pytest.mark.parametrize("prefix_length", [197, 200])
def test_appending_data_cannot_change_prefix_signals(
    causal_prices, weighting_method, prefix_length
):
    ensemble = _template_ensemble(weighting_method)
    prefix = ensemble.generate_signals(causal_prices.iloc[:prefix_length])
    appended = ensemble.generate_signals(causal_prices)
    for original, extended in zip(prefix, appended, strict=True):
        pd.testing.assert_series_equal(original, extended.iloc[:prefix_length])


@pytest.mark.parametrize("weighting_method", ["performance", "volatility", "equal"])
def test_reuse_matches_fresh_instance(causal_prices, weighting_method):
    reused = _template_ensemble(weighting_method)
    reused.generate_signals(_make_ohlcv(n=400, seed=11))
    actual = reused.generate_signals(causal_prices)
    fresh = _template_ensemble(weighting_method)
    expected = fresh.generate_signals(causal_prices)
    for result, reference in zip(actual, expected, strict=True):
        pd.testing.assert_series_equal(result, reference)
    np.testing.assert_allclose(reused.weights, fresh.weights)
    assert reused.last_rebalance == fresh.last_rebalance
    assert reused.strategy_returns == fresh.strategy_returns


@pytest.mark.parametrize("weighting_method", ["performance", "volatility"])
def test_weights_use_only_returns_before_segment_boundary(
    causal_prices, weighting_method
):
    ensemble = _template_ensemble(weighting_method)
    ensemble.lookback_period = 10
    ensemble.rebalance_frequency = 20
    individual = {
        i: strategy.generate_signals(causal_prices)
        for i, strategy in enumerate(ensemble.strategies)
    }
    actual = ensemble.generate_signals(causal_prices)
    price_returns = causal_prices["close"].pct_change()
    reference = _template_ensemble(weighting_method)
    reference.lookback_period = 10
    expected_entry = pd.Series(False, index=causal_prices.index)
    expected_exit = pd.Series(False, index=causal_prices.index)
    for start in range(0, len(causal_prices), 20):
        if start:
            reference.strategy_returns = {
                i: ((entry.astype(int) - exit_.astype(int)).shift(1) * price_returns)
                .iloc[start - 10 : start]
                .tolist()
                for i, (entry, exit_) in individual.items()
            }
            if weighting_method == "performance":
                reference.weights = reference.calculate_performance_weights(
                    pd.DataFrame()
                )
            else:
                reference.weights = reference.calculate_volatility_weights(
                    pd.DataFrame()
                )
        stop = min(start + 20, len(causal_prices))
        segment = {
            i: (entry.iloc[start:stop], exit_.iloc[start:stop])
            for i, (entry, exit_) in individual.items()
        }
        entries, exits = reference.combine_signals(segment)
        expected_entry.iloc[start:stop] = entries.to_numpy()
        expected_exit.iloc[start:stop] = exits.to_numpy()
        np.testing.assert_allclose(
            ensemble._weight_history.iloc[start:stop],
            np.tile(reference.weights, (stop - start, 1)),
        )
    pd.testing.assert_series_equal(actual[0], expected_entry)
    pd.testing.assert_series_equal(actual[1], expected_exit)
    assert np.isfinite(ensemble._weight_history.to_numpy()).all()
    np.testing.assert_allclose(ensemble._weight_history.sum(axis=1), 1)


def test_performance_weights_stay_finite_for_constant_positive_returns():
    ensemble = StrategyEnsemble(
        [SilentStrategy("S1"), SilentStrategy("S2")], lookback_period=10
    )
    ensemble.strategy_returns = {0: [0.01] * 10, 1: [0.02] * 10}
    weights = ensemble.calculate_performance_weights(pd.DataFrame())
    assert np.isfinite(weights).all()
    np.testing.assert_allclose(weights.sum(), 1)
    assert weights[1] > weights[0]


@pytest.mark.parametrize("weighting_method", ["performance", "volatility", "equal"])
def test_warmup_and_empty_reuse_reset_state(causal_prices, weighting_method):
    ensemble = _template_ensemble(weighting_method)
    ensemble.generate_signals(causal_prices)
    short = causal_prices.iloc[:45]
    ensemble.generate_signals(short)
    np.testing.assert_allclose(ensemble._weight_history, 1 / 3)
    assert ensemble.last_rebalance == 40
    empty = ensemble.generate_signals(causal_prices.iloc[:0])
    assert all(signal.empty for signal in empty)
    assert ensemble.strategy_returns == {}
    assert ensemble.last_rebalance == 0
    assert ensemble._weight_history.empty


@pytest.mark.parametrize("weighting_method", ["performance", "volatility", "equal"])
def test_ensemble_leaves_global_vectorbt_settings_unchanged(
    causal_prices, weighting_method
):
    before = deepcopy(vbt.settings)
    _template_ensemble(weighting_method).generate_signals(causal_prices)
    assert vbt.settings == before
