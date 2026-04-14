"""Shared Pydantic type aliases for MCP tool parameters.

Some MCP clients (notably Claude Desktop via `mcp-remote`) serialize list
arguments as JSON-encoded strings instead of JSON arrays. Pydantic v2 strict
validation rejects this with `list_type` errors. These aliases apply a
`BeforeValidator` that accepts both native lists and JSON-string lists, so tool
signatures stay expressive without each site needing manual string handling.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, TypeVar

from pydantic import BeforeValidator

T = TypeVar("T")


def _coerce_json_list(value: Any) -> Any:
    """Accept native lists, JSON-string lists, or bare strings as a single-item list.

    - `["A", "B"]` → unchanged
    - `'["A", "B"]'` → `["A", "B"]` (stringified-by-client case)
    - `"AAPL"` → `["AAPL"]` (single scalar convenience)
    - `None` → unchanged (preserves optional semantics)
    """
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return value
        return parsed
    return [value]


StrList = Annotated[list[str], BeforeValidator(_coerce_json_list)]
"""Required list of strings. Accepts JSON-string input from loose MCP clients."""

OptionalStrList = Annotated[list[str] | None, BeforeValidator(_coerce_json_list)]
"""Optional list of strings. Accepts JSON-string input from loose MCP clients."""
