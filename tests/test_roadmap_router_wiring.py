"""Integration tests for Roadmap v1 MCP router wiring.

Service-layer unit tests cover the business logic inside each router
(signals, screening_pipeline, journal, watchlist, risk_dashboard). These
tests cover the ``register_*_tools(mcp)`` wiring itself:

1. Each router registers at least one tool.
2. Every registered tool has a non-empty, substantive description.
3. Any tool parameter that accepts a list of strings is declared with
   ``StrList`` / ``OptionalStrList`` (not a bare ``list[str]``), so MCP
   clients that JSON-stringify array arguments (Claude Desktop via
   ``mcp-remote``) still validate successfully.

A broken decorator (missing description, bare ``list[str]``) previously
only showed up in production or via the static
``scripts/check_mcp_list_types.py`` check. This runtime test makes the
contract fail CI instead.
"""

from __future__ import annotations

import asyncio
import typing
from collections.abc import Callable

import pytest
from fastmcp import FastMCP

from maverick_mcp.api.routers.journal import register_journal_tools
from maverick_mcp.api.routers.risk_dashboard import register_risk_dashboard_tools
from maverick_mcp.api.routers.screening_pipeline import (
    register_screening_pipeline_tools,
)
from maverick_mcp.api.routers.signals import register_signal_tools
from maverick_mcp.api.routers.watchlist import register_watchlist_tools
from maverick_mcp.utils.mcp_types import OptionalStrList, StrList

_ROADMAP_ROUTERS: list[tuple[str, Callable[[FastMCP], None]]] = [
    ("signals", register_signal_tools),
    ("screening_pipeline", register_screening_pipeline_tools),
    ("journal", register_journal_tools),
    ("watchlist", register_watchlist_tools),
    ("risk_dashboard", register_risk_dashboard_tools),
]

# Valid list-of-string aliases. Parameters declared with either are accepted.
_ACCEPTED_STR_LIST_ALIASES = {StrList, OptionalStrList}


def _list_registered_tools(register_fn: Callable[[FastMCP], None]):
    """Register tools onto a fresh FastMCP instance and return them."""
    mcp = FastMCP(f"wiring_test_{register_fn.__name__}")
    register_fn(mcp)
    return list(asyncio.run(mcp.list_tools()))


def _is_str_list_schema(schema: dict) -> bool:
    """Return True when a JSON-schema fragment describes a list-of-strings.

    Handles the shape ``OptionalStrList`` emits (``anyOf`` with array+null)
    as well as the bare ``StrList`` form.
    """
    if schema.get("type") == "array":
        items = schema.get("items") or {}
        return items.get("type") == "string"
    for variant in schema.get("anyOf", ()):
        if variant.get("type") == "array":
            items = variant.get("items") or {}
            if items.get("type") == "string":
                return True
    return False


@pytest.mark.parametrize(
    ("router_name", "register_fn"),
    _ROADMAP_ROUTERS,
    ids=[r[0] for r in _ROADMAP_ROUTERS],
)
def test_router_registers_at_least_one_tool(
    router_name: str, register_fn: Callable[[FastMCP], None]
) -> None:
    """A silent no-op ``register_*_tools`` would otherwise drop an entire
    domain from the MCP surface. Guard that every router emits >= 1 tool."""
    tools = _list_registered_tools(register_fn)
    assert tools, f"{router_name}: register_*_tools produced zero tools"


@pytest.mark.parametrize(
    ("router_name", "register_fn"),
    _ROADMAP_ROUTERS,
    ids=[r[0] for r in _ROADMAP_ROUTERS],
)
def test_tools_have_substantive_descriptions(
    router_name: str, register_fn: Callable[[FastMCP], None]
) -> None:
    """MCP clients (and the LLM) rely on the tool description to decide when
    to call a tool. An empty or one-word description is effectively broken.
    """
    tools = _list_registered_tools(register_fn)
    bad = [
        t.name for t in tools if not t.description or len(t.description.strip()) < 20
    ]
    assert not bad, (
        f"{router_name}: tools with missing or too-short descriptions: {bad}"
    )


@pytest.mark.parametrize(
    ("router_name", "register_fn"),
    _ROADMAP_ROUTERS,
    ids=[r[0] for r in _ROADMAP_ROUTERS],
)
def test_list_string_params_use_strlist_alias(
    router_name: str, register_fn: Callable[[FastMCP], None]
) -> None:
    """Any tool parameter that serializes as a list-of-strings must be
    declared with ``StrList`` or ``OptionalStrList``.

    Background: Claude Desktop via ``mcp-remote`` JSON-stringifies array
    arguments (e.g. sends ``'["AAPL"]'`` instead of ``["AAPL"]``). A bare
    ``list[str]`` annotation rejects the stringified form with a Pydantic
    ``list_type`` error. ``StrList`` / ``OptionalStrList`` apply a
    ``BeforeValidator`` that accepts both.

    We detect violations at the type-hint level — introspecting the
    function's ``Annotated`` metadata — because it's the authoritative
    source (the JSON schema alone doesn't tell us whether a
    ``BeforeValidator`` is attached).
    """
    tools = _list_registered_tools(register_fn)
    violations: list[str] = []

    for tool in tools:
        # FunctionTool subclasses expose .fn; Tool base class doesn't declare
        # it in type stubs. All router tools are FunctionTool instances.
        fn = getattr(tool, "fn", None)
        if fn is None:
            continue
        hints = typing.get_type_hints(fn, include_extras=True)
        props: dict = tool.parameters.get("properties", {})
        for param_name, schema in props.items():
            if not _is_str_list_schema(schema):
                continue

            annotation = hints.get(param_name)
            if annotation is None:
                violations.append(
                    f"{tool.name}.{param_name}: array-typed param has no "
                    f"type hint to inspect"
                )
                continue

            if annotation in _ACCEPTED_STR_LIST_ALIASES:
                continue

            # Unwrap Optional[StrList]-style shapes: check each member of a
            # ``X | None`` / ``Union`` for a match.
            origin = typing.get_origin(annotation)
            if origin is typing.Union:
                if any(
                    member in _ACCEPTED_STR_LIST_ALIASES
                    for member in typing.get_args(annotation)
                ):
                    continue

            violations.append(
                f"{tool.name}.{param_name}: list-of-strings param must use "
                f"StrList / OptionalStrList (got {annotation!r})"
            )

    assert not violations, (
        f"{router_name}: MCP serialization-compatibility violations:\n  - "
        + "\n  - ".join(violations)
    )
