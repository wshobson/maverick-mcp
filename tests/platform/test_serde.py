"""Tests for maverick.platform.serde."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from maverick.platform.serde import deserialize, ensure_timezone_naive, serialize


def _ohlcv() -> pd.DataFrame:
    idx = pd.date_range("2026-01-02", periods=5, freq="D", name="date")
    return pd.DataFrame(
        {
            "Open": np.linspace(10, 14, 5),
            "Close": np.linspace(10.5, 14.5, 5),
            "Volume": np.arange(5, dtype=np.int64) * 1000,
        },
        index=idx,
    )


def test_dataframe_round_trip_preserves_everything():
    df = _ohlcv()
    result = deserialize(serialize(df))
    pd.testing.assert_frame_equal(result, df)


def test_dataframe_payload_is_compressed():
    df = pd.concat([_ohlcv()] * 200)
    payload = serialize(df)
    assert len(payload) < df.memory_usage(deep=True).sum()


def test_dict_of_dataframes_round_trip():
    data = {"AAPL": _ohlcv(), "MSFT": _ohlcv() * 2}
    result = deserialize(serialize(data))
    assert set(result) == {"AAPL", "MSFT"}
    pd.testing.assert_frame_equal(result["AAPL"], data["AAPL"])


def test_plain_structures_round_trip():
    value = {"a": [1, 2, 3], "b": "text", "c": {"nested": True}, "d": 1.5}
    assert deserialize(serialize(value)) == value


def test_json_fallback_types():
    stamp = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
    value = {"when": stamp, "tags": {"x", "y"}, "series": pd.Series([1, 2, 3])}
    result = deserialize(serialize(value))
    assert "2026-07-18" in str(result["when"])
    assert sorted(result["tags"]) == ["x", "y"]
    assert list(result["series"]) == [1, 2, 3]


def test_timezone_aware_index_normalized():
    df = _ohlcv()
    df.index = df.index.tz_localize("US/Eastern")
    naive = ensure_timezone_naive(df)
    assert naive.index.tz is None
    round_tripped = deserialize(serialize(df))
    assert round_tripped.index.tz is None
