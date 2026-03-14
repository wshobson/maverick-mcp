"""Tests for the tool registry — verifies all register_* functions register tools correctly."""

from unittest.mock import MagicMock, patch

from maverick_mcp.api.routers.tool_registry import (
    register_all_router_tools,
    register_data_tools,
    register_finnhub_tools,
    register_options_tools,
    register_performance_tools,
    register_portfolio_tools,
    register_screening_tools,
    register_streaming_tools,
    register_technical_tools,
)


def _make_mock_mcp():
    """Create a mock FastMCP that tracks tool registrations."""
    mcp = MagicMock()
    registered_tools = {}

    def tool_decorator(name=None, **kwargs):
        def decorator(func):
            registered_tools[name] = func
            return func

        return decorator

    mcp.tool = MagicMock(side_effect=tool_decorator)
    mcp._registered = registered_tools
    return mcp


class TestRegisterTechnicalTools:
    def test_registers_five_tools(self):
        mcp = _make_mock_mcp()
        register_technical_tools(mcp)
        names = [c.kwargs["name"] for c in mcp.tool.call_args_list]
        assert "technical_get_rsi_analysis" in names
        assert "technical_get_macd_analysis" in names
        assert "technical_get_support_resistance" in names
        assert "technical_get_full_technical_analysis" in names
        assert "technical_get_stock_chart_analysis" in names
        assert len(names) == 5


class TestRegisterScreeningTools:
    def test_registers_five_tools(self):
        mcp = _make_mock_mcp()
        register_screening_tools(mcp)
        names = [c.kwargs["name"] for c in mcp.tool.call_args_list]
        assert len(names) == 5
        assert all(n.startswith("screening_") for n in names)


class TestRegisterPortfolioTools:
    def test_registers_seven_tools(self):
        mcp = _make_mock_mcp()
        register_portfolio_tools(mcp)
        names = [c.kwargs["name"] for c in mcp.tool.call_args_list]
        assert len(names) == 7
        assert "portfolio_add_position" in names
        assert "portfolio_get_my_portfolio" in names
        assert "portfolio_risk_adjusted_analysis" in names


class TestRegisterDataTools:
    def test_registers_data_tools(self):
        mcp = _make_mock_mcp()
        register_data_tools(mcp)
        names = [c.kwargs["name"] for c in mcp.tool.call_args_list]
        assert len(names) >= 9
        assert "data_fetch_stock_data" in names
        assert "data_get_stock_info" in names
        assert "data_get_news_sentiment" in names
        assert "data_clear_cache" in names


class TestRegisterPerformanceTools:
    def test_registers_seven_tools(self):
        mcp = _make_mock_mcp()
        register_performance_tools(mcp)
        names = [c.kwargs["name"] for c in mcp.tool.call_args_list]
        assert len(names) == 7
        assert all(n.startswith("performance_") for n in names)


class TestRegisterOptionsTools:
    def test_registers_seven_tools(self):
        mcp = _make_mock_mcp()
        register_options_tools(mcp)
        names = [c.kwargs["name"] for c in mcp.tool.call_args_list]
        assert len(names) == 7
        assert "options_get_chain" in names
        assert "options_calculate_greeks" in names


class TestRegisterFinnhubTools:
    def test_registers_eight_tools(self):
        mcp = _make_mock_mcp()
        register_finnhub_tools(mcp)
        names = [c.kwargs["name"] for c in mcp.tool.call_args_list]
        assert len(names) == 8
        assert "finnhub_company_news" in names
        assert "finnhub_market_news" in names


class TestRegisterStreamingTools:
    def test_registers_seven_tools(self):
        mcp = _make_mock_mcp()
        register_streaming_tools(mcp)
        names = [c.kwargs["name"] for c in mcp.tool.call_args_list]
        assert len(names) == 7
        assert "streaming_start_price_stream" in names


class TestRegisterAllRouterTools:
    def test_registers_all_without_error(self):
        """Full registration should not raise even if some modules fail."""
        mcp = _make_mock_mcp()
        # This may log warnings for missing optional modules but should not raise
        register_all_router_tools(mcp)
        # Should have registered at least the core tools
        assert mcp.tool.call_count >= 30

    def test_graceful_on_import_error(self):
        """Agent/research/backtesting failures should not prevent other registrations."""
        mcp = _make_mock_mcp()
        with patch(
            "maverick_mcp.api.routers.tool_registry.register_agent_tools",
            side_effect=ImportError("no agents"),
        ):
            # Should still complete without raising
            register_all_router_tools(mcp)
            assert mcp.tool.call_count >= 20
