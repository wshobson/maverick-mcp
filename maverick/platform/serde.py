"""Serialization cascade for cache payloads.

Preserves the legacy cascade's behavior (msgpack+zlib for DataFrames, plain
msgpack for msgpack-safe values, JSON fallback for everything else) without
its stats counters. Standalone: no dependency on config or telemetry.
"""

import io
import json
import zlib
from datetime import date, datetime
from typing import Any, cast

import msgpack
import pandas as pd

_DATAFRAME_TYPE = "dataframe"
_DATAFRAME_JSON_TYPE = "dataframe_json"
_DATAFRAME_DICT_TYPE = "dataframe_dict"


def normalize_timezone(index: pd.Index) -> pd.DatetimeIndex:
    """Return a timezone-naive :class:`~pandas.DatetimeIndex` in UTC."""
    dt_index = index if isinstance(index, pd.DatetimeIndex) else pd.DatetimeIndex(index)
    if dt_index.tz is not None:
        dt_index = dt_index.tz_convert("UTC").tz_localize(None)
    return dt_index


def ensure_timezone_naive(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a DataFrame has a timezone-naive datetime index.

    Args:
        df: DataFrame with potentially timezone-aware index

    Returns:
        DataFrame with timezone-naive index
    """
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = normalize_timezone(df.index)
    return df


def _dataframe_to_msgpack_dict(df: pd.DataFrame) -> dict[str, Any]:
    df = ensure_timezone_naive(df)
    is_datetime_index = isinstance(df.index, pd.DatetimeIndex)
    columns = list(df.columns)
    return {
        "_type": _DATAFRAME_TYPE,
        "columns": columns,
        # .tolist() converts numpy scalars (int64, float64, ...) to native
        # Python types msgpack can pack directly.
        "column_data": {col: df[col].tolist() for col in columns},
        "dtypes": {col: str(df[col].dtype) for col in columns},
        "index_type": "datetime" if is_datetime_index else "other",
        "index_name": df.index.name,
        "index_dtype": str(df.index.dtype),
        "index_data": [str(idx) for idx in df.index],
    }


def _msgpack_dict_to_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
    columns = payload["columns"]
    column_data = payload["column_data"]
    df = pd.DataFrame({col: column_data[col] for col in columns}, columns=columns)

    for col, dtype in payload["dtypes"].items():
        df[col] = df[col].astype(dtype)

    if payload.get("index_type") == "datetime":
        df.index = pd.to_datetime(payload["index_data"])
        df.index = normalize_timezone(df.index)
        inferred_freq = pd.infer_freq(df.index)
        if inferred_freq is not None:
            df.index.freq = inferred_freq
    else:
        index = pd.Index(payload["index_data"])
        index_dtype = payload.get("index_dtype")
        if index_dtype:
            try:
                index = index.astype(index_dtype)
            except (ValueError, TypeError):
                pass  # stored dtype no longer applies; keep the string index
        df.index = index
    df.index.name = payload.get("index_name")

    return df


def _dataframe_to_json_envelope(df: pd.DataFrame) -> dict[str, Any]:
    """JSON-safe DataFrame representation, used when msgpack packing fails.

    A non-index datetime64 column serializes to ``pd.Timestamp`` values via
    ``.tolist()``, which msgpack cannot pack. Falling back to
    ``DataFrame.to_json`` sidesteps that entirely.
    """
    df = ensure_timezone_naive(df)
    is_datetime_index = isinstance(df.index, pd.DatetimeIndex)
    return {
        "_type": _DATAFRAME_JSON_TYPE,
        "json": df.to_json(orient="split", date_format="iso"),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "index_type": "datetime" if is_datetime_index else "other",
        "index_name": df.index.name,
    }


def _json_envelope_to_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
    df = pd.read_json(io.StringIO(payload["json"]), orient="split")

    for col, dtype in payload.get("dtypes", {}).items():
        # to_json's ISO strings aren't auto-detected as dates for arbitrary
        # column names, so datetime64 columns need an explicit conversion
        # rather than a plain astype.
        if dtype.startswith("datetime64"):
            df[col] = pd.to_datetime(df[col])
        else:
            df[col] = df[col].astype(dtype)

    if payload.get("index_type") == "datetime":
        df.index = pd.to_datetime(df.index)
        df.index = normalize_timezone(df.index)
        inferred_freq = pd.infer_freq(df.index)
        if inferred_freq is not None:
            df.index.freq = inferred_freq
    df.index.name = payload.get("index_name")

    return df


def _reconstruct_dataframe_value(value: Any) -> Any:
    """Reconstruct a single dict-of-DataFrames entry, passing non-DataFrame
    values through unchanged."""
    if isinstance(value, dict):
        if value.get("_type") == _DATAFRAME_TYPE:
            return _msgpack_dict_to_dataframe(value)
        if value.get("_type") == _DATAFRAME_JSON_TYPE:
            return _json_envelope_to_dataframe(value)
    return value


def _json_default(value: Any) -> Any:
    """JSON serializer for unsupported types."""
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, set):
        return list(value)
    raise TypeError(f"Unsupported type {type(value)!r} for serialization")


def serialize(value: Any) -> bytes:
    """Serialize a value to bytes.

    DataFrames and dicts of DataFrames become msgpack+zlib with index,
    column, and dtype round-tripping and timezone normalization to naive.
    Other msgpack-safe values use plain msgpack. Everything else falls back
    to JSON with a default handler for datetime, date, Timestamp, Series,
    and set.
    """
    if isinstance(value, pd.DataFrame):
        try:
            packed = cast(bytes, msgpack.packb(_dataframe_to_msgpack_dict(value)))
        except Exception:
            # A non-index datetime64 column (or similar) isn't msgpack-safe;
            # degrade gracefully to the JSON envelope instead of raising.
            packed = cast(bytes, msgpack.packb(_dataframe_to_json_envelope(value)))
        return zlib.compress(packed, level=1)

    if isinstance(value, dict) and any(
        isinstance(v, pd.DataFrame) for v in value.values()
    ):
        try:
            payload = {
                "_type": _DATAFRAME_DICT_TYPE,
                "data": {
                    key: (
                        _dataframe_to_msgpack_dict(v)
                        if isinstance(v, pd.DataFrame)
                        else v
                    )
                    for key, v in value.items()
                },
            }
            packed = cast(bytes, msgpack.packb(payload))
        except Exception:
            payload = {
                "_type": _DATAFRAME_DICT_TYPE,
                "data": {
                    key: (
                        _dataframe_to_json_envelope(v)
                        if isinstance(v, pd.DataFrame)
                        else v
                    )
                    for key, v in value.items()
                },
            }
            packed = cast(bytes, msgpack.packb(payload))
        return zlib.compress(packed, level=1)

    if isinstance(value, dict | list | str | int | float | bool | type(None)):
        try:
            return cast(bytes, msgpack.packb(value))
        except Exception:
            pass

    return json.dumps(value, default=_json_default).encode("utf-8")


def _looks_like_zlib(payload: bytes) -> bool:
    """Sniff whether ``payload`` starts with a valid zlib header.

    The CMF byte is always 0x78 for the default window size, but the FLG
    byte (and therefore the second magic byte) varies with the compression
    level used, so check the two-byte header checksum instead of a fixed
    ``\\x78\\x9c`` prefix.
    """
    if len(payload) < 2 or payload[0] != 0x78:
        return False
    return (payload[0] * 256 + payload[1]) % 31 == 0


def deserialize(payload: bytes) -> Any:
    """Deserialize bytes produced by :func:`serialize`.

    Sniffs the zlib magic bytes first, then tries msgpack, then JSON. Raises
    ``ValueError`` if none of the three formats parse (e.g. a corrupted or
    foreign payload that merely happens to start with a zlib-shaped header).
    """
    if _looks_like_zlib(payload):
        try:
            payload = zlib.decompress(payload)
        except zlib.error:
            pass  # header looked like zlib but the body isn't; fall through

    try:
        result = msgpack.unpackb(payload, raw=False)
    except Exception:
        try:
            return json.loads(payload.decode("utf-8"))
        except Exception as exc:
            raise ValueError(
                "Unable to deserialize payload: not valid zlib+msgpack, "
                "msgpack, or JSON"
            ) from exc

    if isinstance(result, dict):
        if result.get("_type") == _DATAFRAME_TYPE:
            return _msgpack_dict_to_dataframe(result)
        if result.get("_type") == _DATAFRAME_JSON_TYPE:
            return _json_envelope_to_dataframe(result)
        if result.get("_type") == _DATAFRAME_DICT_TYPE:
            return {
                key: _reconstruct_dataframe_value(v)
                for key, v in result["data"].items()
            }

    return result
