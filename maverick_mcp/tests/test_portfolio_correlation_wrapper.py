"""Regression checks for the portfolio correlation MCP wrapper."""

import ast
import asyncio
import inspect
import sys
import types
from pathlib import Path
from typing import Any


class _FakeMCP:
    def tool(self):
        def decorator(func):
            return func

        return decorator


def _portfolio_correlation_wrapper_node() -> ast.AsyncFunctionDef:
    source = Path("maverick_mcp/api/server.py").read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AsyncFunctionDef)
            and node.name == "portfolio_correlation_analysis"
        ):
            return node
    raise AssertionError("portfolio_correlation_analysis wrapper not found")


def _compile_portfolio_correlation_wrapper():
    node = _portfolio_correlation_wrapper_node()
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)

    namespace = {"Any": Any, "mcp": _FakeMCP()}
    exec(compile(module, "<portfolio_correlation_wrapper>", "exec"), namespace)
    return namespace["portfolio_correlation_analysis"]


def test_portfolio_correlation_analysis_exposes_portfolio_name() -> None:
    wrapper = _compile_portfolio_correlation_wrapper()

    parameter = inspect.signature(wrapper).parameters["portfolio_name"]

    assert parameter.default == "My Portfolio"


def test_portfolio_correlation_analysis_forwards_portfolio_name() -> None:
    wrapper = _compile_portfolio_correlation_wrapper()
    calls = []

    fake_module = types.ModuleType("maverick_mcp.api.routers.portfolio")

    def fake_portfolio_correlation_analysis(**kwargs):
        calls.append(kwargs)
        return {"portfolio": kwargs["portfolio_name"]}

    fake_module.portfolio_correlation_analysis = fake_portfolio_correlation_analysis

    module_name = "maverick_mcp.api.routers.portfolio"
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = fake_module
    try:
        result = asyncio.run(wrapper(days=30, portfolio_name="Core Portfolio"))
    finally:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module

    assert result == {"portfolio": "Core Portfolio"}
    assert calls == [
        {
            "tickers": None,
            "days": 30,
            "user_id": "default",
            "portfolio_name": "Core Portfolio",
        }
    ]
