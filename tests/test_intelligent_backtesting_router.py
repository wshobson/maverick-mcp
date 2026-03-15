"""Tests for the intelligent backtesting router."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from maverick_mcp.api.routers.intelligent_backtesting import (
    setup_intelligent_backtesting_tools,
)


def _register_and_get_tools():
    """Register intelligent backtesting tools on a mock MCP and return inner functions."""
    tools = {}
    mcp = MagicMock()

    def tool_decorator(**kwargs):
        def decorator(func):
            tools[func.__name__] = func
            return func

        return decorator

    mcp.tool = MagicMock(side_effect=lambda **kw: tool_decorator(**kw))
    mcp.tool.return_value = lambda func: tools.update({func.__name__: func}) or func

    setup_intelligent_backtesting_tools(mcp)
    return tools


@pytest.fixture(scope="module")
def bt_tools():
    return _register_and_get_tools()


class TestRunIntelligentBacktest:
    @pytest.mark.asyncio
    async def test_success(self, bt_tools):
        fn = bt_tools["run_intelligent_backtest"]
        mock_ctx = MagicMock()
        mock_results = {"symbol": "AAPL", "total_return": 0.15, "sharpe": 1.2}
        with patch(
            "maverick_mcp.api.routers.intelligent_backtesting.BacktestingWorkflow"
        ) as MockWf:
            MockWf.return_value.run_intelligent_backtest = AsyncMock(
                return_value=mock_results
            )
            result = await fn(ctx=mock_ctx, symbol="AAPL")
            assert result["symbol"] == "AAPL"
            assert result["total_return"] == 0.15

    @pytest.mark.asyncio
    async def test_exception(self, bt_tools):
        fn = bt_tools["run_intelligent_backtest"]
        mock_ctx = MagicMock()
        with patch(
            "maverick_mcp.api.routers.intelligent_backtesting.BacktestingWorkflow"
        ) as MockWf:
            MockWf.return_value.run_intelligent_backtest = AsyncMock(
                side_effect=ValueError("bad symbol")
            )
            result = await fn(ctx=mock_ctx, symbol="INVALID")
            assert "error" in result
            assert "bad symbol" in result["error"]


class TestQuickMarketRegimeAnalysis:
    @pytest.mark.asyncio
    async def test_success(self, bt_tools):
        fn = bt_tools["quick_market_regime_analysis"]
        mock_ctx = MagicMock()
        mock_results = {"symbol": "SPY", "regime": "trending"}
        with patch(
            "maverick_mcp.api.routers.intelligent_backtesting.BacktestingWorkflow"
        ) as MockWf:
            MockWf.return_value.run_quick_analysis = AsyncMock(
                return_value=mock_results
            )
            result = await fn(ctx=mock_ctx, symbol="SPY")
            assert result["regime"] == "trending"

    @pytest.mark.asyncio
    async def test_exception(self, bt_tools):
        fn = bt_tools["quick_market_regime_analysis"]
        mock_ctx = MagicMock()
        with patch(
            "maverick_mcp.api.routers.intelligent_backtesting.BacktestingWorkflow"
        ) as MockWf:
            MockWf.return_value.run_quick_analysis = AsyncMock(
                side_effect=RuntimeError("timeout")
            )
            result = await fn(ctx=mock_ctx, symbol="SPY")
            assert "error" in result


class TestExplainMarketRegime:
    @pytest.mark.asyncio
    async def test_known_regime(self, bt_tools):
        fn = bt_tools["explain_market_regime"]
        mock_ctx = MagicMock()
        result = await fn(ctx=mock_ctx, regime="trending")
        assert result["regime"] == "trending"
        assert "explanation" in result
        assert "best_strategies" in result["explanation"]

    @pytest.mark.asyncio
    async def test_unknown_regime(self, bt_tools):
        fn = bt_tools["explain_market_regime"]
        mock_ctx = MagicMock()
        result = await fn(ctx=mock_ctx, regime="unknown_regime")
        assert "error" in result
        assert "available_regimes" in result
