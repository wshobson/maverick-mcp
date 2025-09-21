"""Tests for secure cache serialization helpers."""

from __future__ import annotations

import pandas as pd
import pandas.testing as pdt
import pytest

from maverick_mcp.data import cache as cache_module


@pytest.fixture(autouse=True)
def _memory_cache_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure Redis is not used and memory cache starts clean."""

    monkeypatch.setattr(cache_module, "get_redis_client", lambda: None)
    cache_module._memory_cache.clear()


def test_dataframe_round_trip() -> None:
    """DataFrames should round-trip through the cache without pickle usage."""

    key = "test:dataframe"
    df = pd.DataFrame(
        {"open": [1.0, 2.0], "close": [1.5, 2.5]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    assert cache_module.save_to_cache(key, df, ttl=60)
    cached = cache_module.get_from_cache(key)
    assert isinstance(cached, pd.DataFrame)
    pdt.assert_frame_equal(cached, df)


def test_dict_with_dataframe_round_trip() -> None:
    """Dictionaries containing DataFrames should round-trip safely."""

    key = "test:dict"
    frame = pd.DataFrame(
        {"volume": [100, 200]},
        index=pd.to_datetime(["2024-01-03", "2024-01-04"]),
    )
    payload = {
        "meta": {"status": "ok"},
        "frame": frame,
        "values": [1, 2, 3],
    }

    assert cache_module.save_to_cache(key, payload, ttl=60)
    cached = cache_module.get_from_cache(key)
    assert isinstance(cached, dict)
    assert cached["meta"] == payload["meta"]
    assert cached["values"] == payload["values"]
    pdt.assert_frame_equal(cached["frame"], frame)


def test_unsupported_type_not_cached() -> None:
    """Unsupported data types should not be cached silently."""

    class _Unsupported:
        pass

    key = "test:unsupported"
    assert not cache_module.save_to_cache(key, _Unsupported(), ttl=60)
    assert key not in cache_module._memory_cache
