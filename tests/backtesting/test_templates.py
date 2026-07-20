"""Characterization tests for `maverick.backtesting.strategies.templates`
and `maverick.backtesting.strategies.signals`.

`pytest.importorskip("vectorbt")` guards this whole module for the same
reason as `tests/backtesting/test_engine.py`: vectorbt is only installed
under the `backtesting` extra. The OHLCV fixture below is copied verbatim
from that file's `ohlcv_frame` fixture (same seed, same shape) so both
suites exercise the identical frame -- `test_sma_cross_end_to_end_pins_metrics`
there and `test_golden_run[sma_cross]` here pin the same
`total_return`/`total_trades` values for the sma_cross strategy, which is
a cross-check that `signals.py`'s `_sma_crossover_signals` and
`test_engine.py`'s local `_sma_cross_signals` compute identically.

Golden numeric assertions below were recorded by actually running
`generate_signals` + `engine.run_backtest` end to end against the fixture
(not derived independently), per this task's characterization-testing
brief -- a change to `signals.py`'s math should change these numbers and
fail the test.

Catalog count note: `STRATEGY_TEMPLATES` has 12 entries, not the 13
described in the task brief -- see `templates.py`'s module docstring for
the recount.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("vectorbt")

from maverick.backtesting.config import BacktestingSettings
from maverick.backtesting.engine import run_backtest
from maverick.backtesting.strategies.signals import generate_signals
from maverick.backtesting.strategies.templates import (
    STRATEGY_TEMPLATES,
    get_strategy_info,
    get_strategy_template,
    list_available_strategies,
)
from maverick.backtesting.types import StrategyCatalogEntry


@pytest.fixture
def ohlcv_frame() -> pd.DataFrame:
    """300 rows of deterministic synthetic daily OHLCV data, seed 42.

    Identical recipe to `tests/backtesting/test_engine.py`'s `ohlcv_frame`
    fixture -- kept as a verbatim copy (not a shared import) so this file
    stays self-contained, matching that file's own stated rationale.
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=300, freq="D")
    returns = rng.normal(0.0005, 0.02, len(dates))
    prices = 100 * np.cumprod(1 + returns)
    volumes = rng.integers(1_000_000, 5_000_000, len(dates))
    frame = pd.DataFrame(
        {
            "open": prices * rng.uniform(0.98, 1.02, len(dates)),
            "high": prices * rng.uniform(1.00, 1.05, len(dates)),
            "low": prices * rng.uniform(0.95, 1.00, len(dates)),
            "close": prices,
            "volume": volumes,
        },
        index=dates,
    )
    frame["high"] = np.maximum(frame["high"], np.maximum(frame["open"], frame["close"]))
    frame["low"] = np.minimum(frame["low"], np.minimum(frame["open"], frame["close"]))
    return frame


# -- Catalog: STRATEGY_TEMPLATES / accessors -------------------------------

EXPECTED_STRATEGY_TYPES = [
    "sma_cross",
    "rsi",
    "macd",
    "bollinger",
    "momentum",
    "ema_cross",
    "mean_reversion",
    "breakout",
    "volume_momentum",
    "online_learning",
    "regime_aware",
    "ensemble",
]


def test_catalog_has_exactly_12_pinned_strategy_types():
    assert list_available_strategies() == EXPECTED_STRATEGY_TYPES
    assert len(STRATEGY_TEMPLATES) == 12


@pytest.mark.parametrize("strategy_type", EXPECTED_STRATEGY_TYPES)
def test_get_strategy_template_returns_parameters_and_ranges(strategy_type):
    template = get_strategy_template(strategy_type)
    assert "name" in template
    assert "description" in template
    assert "parameters" in template
    assert "optimization_ranges" in template
    assert "code" in template
    # Legacy quirk, faithfully preserved: `regime_aware`'s `trend_strategy`/
    # `range_strategy` and `ensemble`'s `weight_method` are string-valued
    # selector parameters with no declared optimization range -- every
    # other parameter has one.
    assert set(template["optimization_ranges"]) <= set(template["parameters"])


def test_get_strategy_template_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown strategy type: bogus"):
        get_strategy_template("bogus")


@pytest.mark.parametrize("strategy_type", EXPECTED_STRATEGY_TYPES)
def test_get_strategy_info_returns_typed_catalog_entry(strategy_type):
    info = get_strategy_info(strategy_type)
    assert isinstance(info, StrategyCatalogEntry)
    assert info.type == strategy_type
    assert info.name == STRATEGY_TEMPLATES[strategy_type]["name"]
    assert info.description == STRATEGY_TEMPLATES[strategy_type]["description"]
    assert info.default_parameters == STRATEGY_TEMPLATES[strategy_type]["parameters"]
    assert (
        info.optimization_ranges
        == STRATEGY_TEMPLATES[strategy_type]["optimization_ranges"]
    )


def test_get_strategy_info_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown strategy type"):
        get_strategy_info("bogus")


def test_stub_templates_have_comment_only_code_field():
    """The 3 catalog entries the task brief calls "stubs": their `code`
    field is a descriptive comment, not an executable vectorbt expression
    -- unlike the other 9, which reference `entries`/`exits` directly."""
    for strategy_type in ("online_learning", "regime_aware", "ensemble"):
        code = STRATEGY_TEMPLATES[strategy_type]["code"]
        assert "entries" not in code
        assert "exits" not in code

    for strategy_type in set(EXPECTED_STRATEGY_TYPES) - {
        "online_learning",
        "regime_aware",
        "ensemble",
    }:
        code = STRATEGY_TEMPLATES[strategy_type]["code"]
        assert "entries" in code
        assert "exits" in code


# -- generate_signals: validation ------------------------------------------


def test_generate_signals_rejects_missing_close_column(ohlcv_frame):
    frame = ohlcv_frame.drop(columns=["close"])
    with pytest.raises(ValueError, match="close"):
        generate_signals(frame, "sma_cross", {})


def test_generate_signals_rejects_unknown_strategy_type(ohlcv_frame):
    with pytest.raises(ValueError, match="Unknown strategy type: bogus"):
        generate_signals(ohlcv_frame, "bogus", {})


def test_generate_signals_sma_crossover_alias_matches_sma_cross(ohlcv_frame):
    params = {"fast_period": 10, "slow_period": 20}
    entries_a, exits_a = generate_signals(ohlcv_frame, "sma_cross", params)
    entries_b, exits_b = generate_signals(ohlcv_frame, "sma_crossover", params)
    pd.testing.assert_series_equal(entries_a, entries_b)
    pd.testing.assert_series_equal(exits_a, exits_b)


def test_generate_signals_returns_series_for_every_strategy_type(ohlcv_frame):
    """Every dispatch branch -- including the 3 "stub" catalog entries --
    returns real, frame-aligned `pd.Series`, not `NotImplementedError` or a
    delegation into `strategies/ml/`. See `signals.py`'s module docstring."""
    for strategy_type in EXPECTED_STRATEGY_TYPES:
        params = STRATEGY_TEMPLATES[strategy_type]["parameters"]
        entries, exits = generate_signals(ohlcv_frame, strategy_type, params)
        assert isinstance(entries, pd.Series)
        assert isinstance(exits, pd.Series)
        assert len(entries) == len(ohlcv_frame)
        assert len(exits) == len(ohlcv_frame)
        assert entries.dtype == bool
        assert exits.dtype == bool


# -- Golden runs: generate_signals + engine.run_backtest, all 12 types ----

# Recorded by running `generate_signals(ohlcv_frame, strategy_type,
# STRATEGY_TEMPLATES[strategy_type]["parameters"])` through
# `engine.run_backtest` against the fixture above. Pinned, not derived.
GOLDEN_RUNS = {
    "sma_cross": (8, 9, -0.1939859295537608, 8),
    "rsi": (6, 9, 0.0786439798414809, 3),
    "macd": (11, 12, -0.24972229558602765, 11),
    "bollinger": (8, 6, 0.17822428539173615, 5),
    "momentum": (72, 74, -0.09374865627834533, 5),
    "ema_cross": (4, 5, -0.1695965272899284, 4),
    "mean_reversion": (86, 108, 0.026922494684228696, 9),
    "breakout": (29, 48, -0.17410249459351454, 6),
    "volume_momentum": (8, 146, -0.04928367992395524, 7),
    "online_learning": (125, 135, -0.20602085098792777, 19),
    "regime_aware": (85, 128, -0.2762088516928612, 12),
    "ensemble": (0, 1, 0.0, 0),
}


@pytest.mark.parametrize("strategy_type", EXPECTED_STRATEGY_TYPES)
def test_golden_run(ohlcv_frame, strategy_type):
    exp_entries, exp_exits, exp_total_return, exp_total_trades = GOLDEN_RUNS[
        strategy_type
    ]
    params = STRATEGY_TEMPLATES[strategy_type]["parameters"]

    entries, exits = generate_signals(ohlcv_frame, strategy_type, params)
    assert int(entries.sum()) == exp_entries
    assert int(exits.sum()) == exp_exits

    result = run_backtest(
        ohlcv_frame,
        entries,
        exits,
        symbol="TEST",
        strategy=strategy_type,
        parameters=params,
        settings=BacktestingSettings(),
    )
    assert result.metrics.total_return == pytest.approx(exp_total_return, rel=1e-6)
    assert result.metrics.total_trades == exp_total_trades


def test_sma_cross_golden_run_matches_engine_characterization(ohlcv_frame):
    """Cross-check against `test_engine.py::test_sma_cross_end_to_end_pins_metrics`,
    which pins the identical `total_return` for its own hand-rolled
    `_sma_cross_signals` -- confirms `signals.py`'s `_sma_crossover_signals`
    is the same computation."""
    entries, exits = generate_signals(
        ohlcv_frame, "sma_cross", {"fast_period": 10, "slow_period": 20}
    )
    result = run_backtest(ohlcv_frame, entries, exits, settings=BacktestingSettings())
    assert result.metrics.total_return == pytest.approx(-0.1939859295537608, rel=1e-6)
