"""Tests for `maverick.backtesting.optimization`.

No `pytest.importorskip` needed: this module imports nothing but `numpy`, so it runs
identically on a base install and under the `[backtesting]` extra.
"""

import numpy as np
import pytest

from maverick.backtesting.optimization import generate_param_grid


def test_sma_cross_grids():
    assert generate_param_grid("sma_cross", "coarse") == {
        "fast_period": [5, 10, 20],
        "slow_period": [20, 50, 100],
    }
    assert generate_param_grid("sma_cross", "medium") == {
        "fast_period": [5, 10, 15, 20],
        "slow_period": [20, 30, 50, 100],
    }
    assert generate_param_grid("sma_cross") == generate_param_grid(
        "sma_cross", "medium"
    )
    assert generate_param_grid("sma_cross", "fine") == {
        "fast_period": list(range(5, 25, 2)),
        "slow_period": list(range(20, 101, 5)),
    }


def test_rsi_grids():
    assert generate_param_grid("rsi", "coarse") == {
        "period": [7, 14, 21],
        "oversold": [20, 30],
        "overbought": [70, 80],
    }
    assert generate_param_grid("rsi", "medium") == {
        "period": [7, 14, 21],
        "oversold": [20, 25, 30, 35],
        "overbought": [65, 70, 75, 80],
    }
    assert generate_param_grid("rsi", "fine") == {
        "period": list(range(7, 22, 2)),
        "oversold": list(range(20, 41, 5)),
        "overbought": list(range(60, 81, 5)),
    }


def test_macd_grids():
    assert generate_param_grid("macd", "coarse") == {
        "fast_period": [8, 12],
        "slow_period": [21, 26],
        "signal_period": [9],
    }
    assert generate_param_grid("macd", "medium") == {
        "fast_period": [8, 10, 12, 14],
        "slow_period": [21, 24, 26, 30],
        "signal_period": [7, 9, 11],
    }
    assert generate_param_grid("macd", "fine") == {
        "fast_period": list(range(8, 15)),
        "slow_period": list(range(20, 31)),
        "signal_period": list(range(7, 12)),
    }


def test_bollinger_grids():
    assert generate_param_grid("bollinger", "coarse") == {
        "period": [10, 20],
        "std_dev": [1.5, 2.0, 2.5],
    }
    assert generate_param_grid("bollinger", "medium") == {
        "period": [10, 15, 20, 25],
        "std_dev": [1.5, 2.0, 2.5, 3.0],
    }
    fine = generate_param_grid("bollinger", "fine")
    assert fine["period"] == list(range(10, 31, 2))
    assert fine["std_dev"] == pytest.approx(np.arange(1.0, 3.1, 0.25).tolist())


def test_momentum_grids():
    assert generate_param_grid("momentum", "coarse") == {
        "lookback": [10, 20, 30],
        "threshold": [0.03, 0.05, 0.10],
    }
    assert generate_param_grid("momentum", "medium") == {
        "lookback": [10, 15, 20, 25, 30],
        "threshold": [0.02, 0.03, 0.05, 0.07, 0.10],
    }
    fine = generate_param_grid("momentum", "fine")
    assert fine["lookback"] == list(range(10, 41, 2))
    assert fine["threshold"] == pytest.approx(np.arange(0.02, 0.11, 0.01).tolist())


def test_unsupported_but_valid_template_strategy_raises():
    """ema_cross is a real STRATEGY_TEMPLATES entry, but legacy's optimizer never supported
    it -- faithfully preserved as a ValueError, not silently accepted."""
    with pytest.raises(ValueError, match="Unknown strategy type"):
        generate_param_grid("ema_cross", "medium")


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown strategy type"):
        generate_param_grid("not_a_strategy", "medium")
