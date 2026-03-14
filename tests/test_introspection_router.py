"""Tests for the introspection router — discover_capabilities, list_all_strategies, get_strategy_help."""

from unittest.mock import MagicMock

import pytest

from maverick_mcp.api.routers.introspection import register_introspection_tools


def _register_and_get_tools():
    """Register introspection tools on a mock MCP and return the inner functions."""
    tools = {}
    mcp = MagicMock()

    def tool_decorator(name=None, **kwargs):
        def decorator(func):
            tools[name or func.__name__] = func
            return func

        return decorator

    mcp.tool = MagicMock(side_effect=tool_decorator)
    register_introspection_tools(mcp)
    return tools


@pytest.fixture(scope="module")
def intro_tools():
    return _register_and_get_tools()


class TestDiscoverCapabilities:
    @pytest.mark.asyncio
    async def test_structure(self, intro_tools):
        result = await intro_tools["discover_capabilities"]()
        assert "server_info" in result
        assert result["server_info"]["name"] == "MaverickMCP"
        assert "capabilities" in result
        caps = result["capabilities"]
        assert "backtesting" in caps
        assert "technical_analysis" in caps
        assert "screening" in caps
        assert "research" in caps
        assert "quick_start" in result


class TestListAllStrategies:
    @pytest.mark.asyncio
    async def test_returns_list(self, intro_tools):
        result = await intro_tools["list_all_strategies"]()
        assert isinstance(result, list)
        # Should have 9 traditional + 5 ML = 14 strategies
        assert len(result) >= 14

    @pytest.mark.asyncio
    async def test_each_strategy_has_fields(self, intro_tools):
        result = await intro_tools["list_all_strategies"]()
        for strategy in result:
            assert "name" in strategy
            assert "description" in strategy
            assert "parameters" in strategy
            assert "example" in strategy


class TestGetStrategyHelp:
    @pytest.mark.asyncio
    async def test_known_strategy(self, intro_tools):
        result = await intro_tools["get_strategy_help"](strategy_type="sma_cross")
        assert result["name"] == "Simple Moving Average Crossover"
        assert "theory" in result
        assert "parameters" in result
        assert "tips" in result

    @pytest.mark.asyncio
    async def test_ml_strategy(self, intro_tools):
        result = await intro_tools["get_strategy_help"](strategy_type="ml_predictor")
        assert result["name"] == "Machine Learning Predictor"

    @pytest.mark.asyncio
    async def test_unknown_strategy(self, intro_tools):
        result = await intro_tools["get_strategy_help"](strategy_type="nonexistent")
        assert "error" in result
        assert "available_strategies" in result
