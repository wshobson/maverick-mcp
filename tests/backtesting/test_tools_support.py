"""Tests for `maverick.backtesting.tools_support`. No `importorskip`: like `tools.py`, this
module never imports vectorbt/sklearn."""

import pandas as pd

from maverick.backtesting.tools_support import MAX_SERIES_POINTS, downsample_series


def _daily_series(n: int) -> dict[str, float]:
    dates = pd.bdate_range("2020-01-02", periods=n)
    return {str(d): 10000.0 + i for i, d in enumerate(dates)}


def test_downsample_series_cuts_a_long_series_and_keeps_both_endpoints():
    series = _daily_series(1304)
    keys = list(series)

    result = downsample_series(series)

    assert MAX_SERIES_POINTS == 60
    assert len(result) == MAX_SERIES_POINTS
    assert next(iter(result)) == keys[0]
    assert list(result)[-1] == keys[-1]
    assert all(result[k] == series[k] for k in result)


def test_downsample_series_spaces_points_evenly_in_date_order():
    series = _daily_series(1304)
    position = {k: i for i, k in enumerate(series)}

    kept = [position[k] for k in downsample_series(series)]

    gaps = [b - a for a, b in zip(kept, kept[1:], strict=False)]
    assert kept == sorted(kept)
    assert max(gaps) - min(gaps) <= 1


def test_downsample_series_leaves_a_short_series_unchanged():
    series = _daily_series(MAX_SERIES_POINTS)

    assert downsample_series(series) == series
    assert downsample_series({}) == {}


def test_downsample_series_just_over_the_cap():
    series = _daily_series(MAX_SERIES_POINTS + 1)
    keys = list(series)

    result = downsample_series(series)

    assert len(result) == MAX_SERIES_POINTS
    assert list(result)[0] == keys[0]
    assert list(result)[-1] == keys[-1]
