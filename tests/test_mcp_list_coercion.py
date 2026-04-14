"""Contract tests for MCP list-parameter coercion.

Locks in the behavior of `StrList` / `OptionalStrList` in
`maverick_mcp.utils.mcp_types`. Prevents regressions if Pydantic strictening
or an accidental removal of the `BeforeValidator` breaks the tolerance that
11 MCP tools rely on for loose-client compatibility.
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from maverick_mcp.utils.mcp_types import OptionalStrList, StrList

optional_adapter = TypeAdapter(OptionalStrList)
required_adapter = TypeAdapter(StrList)


@pytest.mark.parametrize(
    "raw, expected",
    [
        pytest.param(["AAPL", "GOOG"], ["AAPL", "GOOG"], id="native-list"),
        pytest.param('["ANET","MRVL"]', ["ANET", "MRVL"], id="stringified-list"),
        pytest.param('  ["A", "B"]  ', ["A", "B"], id="stringified-list-whitespace"),
        pytest.param("AAPL", ["AAPL"], id="bare-scalar-string"),
        pytest.param(None, None, id="none-passthrough"),
    ],
)
def test_optional_str_list_coerces(raw, expected):
    assert optional_adapter.validate_python(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        pytest.param(["A"], ["A"], id="native-list"),
        pytest.param('["A","B"]', ["A", "B"], id="stringified-list"),
        pytest.param("SPY", ["SPY"], id="bare-scalar-string"),
    ],
)
def test_required_str_list_coerces(raw, expected):
    assert required_adapter.validate_python(raw) == expected


def test_required_str_list_rejects_none():
    with pytest.raises(ValidationError):
        required_adapter.validate_python(None)


def test_malformed_json_falls_through_to_validator():
    """Unparseable JSON-looking input should raise a ValidationError, not silently succeed."""
    with pytest.raises(ValidationError):
        optional_adapter.validate_python('["unterminated')


def test_non_string_non_list_rejected():
    """Ints, dicts, etc. must still fail — the coercer only handles strings."""
    with pytest.raises(ValidationError):
        optional_adapter.validate_python(42)
    with pytest.raises(ValidationError):
        optional_adapter.validate_python({"a": 1})
