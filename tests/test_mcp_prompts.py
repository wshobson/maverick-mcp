"""Tests for MCP prompts registration."""

from unittest.mock import MagicMock

import pytest

from maverick_mcp.api.routers.mcp_prompts import register_mcp_prompts


def _register_and_get():
    """Register MCP prompts and return captured prompts and resources."""
    prompts = {}
    resources = {}
    mcp = MagicMock()

    def prompt_decorator(**kwargs):
        def decorator(func):
            prompts[func.__name__] = func
            return func

        return decorator

    def resource_decorator(uri):
        def decorator(func):
            resources[uri] = func
            return func

        return decorator

    # @mcp.prompt() returns decorator
    mcp.prompt = MagicMock(side_effect=lambda **kw: prompt_decorator(**kw))
    mcp.prompt.return_value = lambda func: prompts.update({func.__name__: func}) or func

    mcp.resource = MagicMock(side_effect=resource_decorator)

    result = register_mcp_prompts(mcp)
    return prompts, resources, result


@pytest.fixture(scope="module")
def registered():
    return _register_and_get()


class TestPromptRegistration:
    def test_returns_true(self, registered):
        _, _, result = registered
        assert result is True

    def test_prompts_registered(self, registered):
        prompts, _, _ = registered
        expected = {
            "backtest_strategy_guide",
            "ml_strategy_examples",
            "optimization_guide",
            "available_tools_summary",
            "troubleshooting_guide",
            "quick_start",
            "strategy_reference",
        }
        assert expected.issubset(set(prompts.keys()))

    def test_resources_registered(self, registered):
        _, resources, _ = registered
        assert "strategies://list" in resources
        assert "tools://categories" in resources
        assert "examples://backtesting" in resources


class TestPromptContent:
    @pytest.mark.asyncio
    async def test_backtest_guide_content(self, registered):
        prompts, _, _ = registered
        content = await prompts["backtest_strategy_guide"]()
        assert "sma_cross" in content
        assert "rsi" in content

    @pytest.mark.asyncio
    async def test_strategy_reference_content(self, registered):
        prompts, _, _ = registered
        content = await prompts["strategy_reference"]()
        assert "sma_cross" in content
        assert "online_learning" in content


class TestResourceContent:
    def test_strategies_list(self, registered):
        _, resources, _ = registered
        result = resources["strategies://list"]()
        assert "traditional_strategies" in result
        assert "ml_strategies" in result
        assert result["total_strategies"] == 15

    def test_tool_categories(self, registered):
        _, resources, _ = registered
        result = resources["tools://categories"]()
        assert "backtesting" in result
        assert "data" in result
        assert "technical_analysis" in result
