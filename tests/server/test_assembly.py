"""Tests for maverick.server.assembly.build_server().

Uses an in-memory `fastmcp.Client` against the real `build_server()` --
`tests/server/conftest.py` points `DATABASE_URL` at an isolated tmp SQLite
file so the real services (each of which creates its own schema against the
one shared engine) work without touching a real deployment's database. The
dev environment has both the `[backtesting]` and `[research]` extras
installed, so the "extras present" assertions run for real here; the
"zero-extra degradation" test simulates their absence by monkeypatching
each domain's own probe function directly (mirrors
`tests/backtesting/test_tools_availability.py`/
`tests/research/test_tools_availability.py`), rather than uninstalling
anything.
"""

import pytest
from fastmcp import Client

from maverick.backtesting import backtesting_extra_available
from maverick.backtesting import tools as backtesting_tools
from maverick.research import research_extra_available
from maverick.research import tools as research_tools
from maverick.server import assembly, prompts
from maverick.server.assembly import build_server

CORE_TOOL_NAMES = frozenset(
    {
        # market_data (7)
        "market_data_get_price_history",
        "market_data_get_price_history_batch",
        "market_data_get_quote",
        "market_data_get_stock_fundamentals",
        "market_data_get_market_overview",
        "market_data_get_chart_links",
        "market_data_clear_market_cache",
        # screening (6)
        "screening_get_bullish",
        "screening_get_bearish",
        "screening_get_supply_demand",
        "screening_get_all",
        "screening_get_by_criteria",
        "screening_run_screens",
        # portfolio (20)
        "portfolio_add_position",
        "portfolio_get_my_portfolio",
        "portfolio_remove_position",
        "portfolio_clear_portfolio",
        "portfolio_risk_adjusted_analysis",
        "portfolio_compare_tickers",
        "portfolio_correlation_analysis",
        "portfolio_get_risk_dashboard",
        "portfolio_check_position_risk",
        "portfolio_get_regime_adjusted_sizing",
        "portfolio_get_risk_alerts",
        "portfolio_watchlist_create",
        "portfolio_watchlist_add",
        "portfolio_watchlist_remove",
        "portfolio_watchlist_brief",
        "portfolio_journal_add_trade",
        "portfolio_journal_close_trade",
        "portfolio_journal_list_trades",
        "portfolio_journal_review",
        "portfolio_get_strategy_performance",
        # technical (4)
        "technical_get_rsi_analysis",
        "technical_get_macd_analysis",
        "technical_get_support_resistance",
        "technical_get_full_technical_analysis",
    }
)

BACKTESTING_TOOL_NAMES = frozenset(
    {
        "backtesting_run_backtest",
        "backtesting_optimize_strategy",
        "backtesting_walk_forward_analysis",
        "backtesting_monte_carlo_simulation",
        "backtesting_compare_strategies",
        "backtesting_list_strategies",
        "backtesting_backtest_portfolio",
        "backtesting_run_ml_strategy_backtest",
        "backtesting_train_ml_predictor",
        "backtesting_analyze_market_regimes",
        "backtesting_create_strategy_ensemble",
        "backtesting_parse_strategy",
    }
)

RESEARCH_TOOL_NAMES = frozenset(
    {
        "research_run_comprehensive",
        "research_analyze_company",
        "research_analyze_sentiment",
    }
)

assert len(CORE_TOOL_NAMES) == 37
assert len(BACKTESTING_TOOL_NAMES) == 12
assert len(RESEARCH_TOOL_NAMES) == 3


async def _tool_names(mcp) -> set[str]:
    async with Client(mcp) as client:
        tools = await client.list_tools()
    return {tool.name for tool in tools}


async def _prompt_names(mcp) -> set[str]:
    async with Client(mcp) as client:
        result = await client.list_prompts()
    return {p.name for p in result}


class TestCoreToolSurface:
    """The full core tool list registers by exact name, regardless of extras."""

    async def test_core_tools_are_exactly_present(self):
        mcp = build_server()
        names = await _tool_names(mcp)
        assert CORE_TOOL_NAMES <= names

    async def test_backtesting_tools_present_when_extra_available(self):
        if not backtesting_extra_available():
            pytest.skip("[backtesting] extra not installed")
        mcp = build_server()
        names = await _tool_names(mcp)
        assert BACKTESTING_TOOL_NAMES <= names

    async def test_research_tools_present_when_extra_available(self):
        if not research_extra_available():
            pytest.skip("[research] extra not installed")
        mcp = build_server()
        names = await _tool_names(mcp)
        assert RESEARCH_TOOL_NAMES <= names

    async def test_total_tool_count_matches_installed_extras(self):
        mcp = build_server()
        names = await _tool_names(mcp)
        expected = set(CORE_TOOL_NAMES)
        if backtesting_extra_available():
            expected |= BACKTESTING_TOOL_NAMES
        if research_extra_available():
            expected |= RESEARCH_TOOL_NAMES
        assert names == expected

    async def test_prompts_registered(self):
        mcp = build_server()
        names = await _prompt_names(mcp)
        expected = {"analyze_stock", "review_portfolio"}
        if backtesting_extra_available():
            expected.add("run_backtest_workflow")
        assert names == expected


class TestZeroExtraDegradation:
    """A base install (neither extra) boots with only the 37 core tools and
    2 prompts, and never raises -- simulated by monkeypatching every copy of
    each domain's own availability probe that assembly/tools/prompts call
    (assembly.py's own imported name gates service construction; each
    `tools.py`'s private alias gates `register()`; `prompts.py`'s private
    alias gates the third prompt)."""

    @pytest.fixture(autouse=True)
    def _no_extras(self, monkeypatch):
        monkeypatch.setattr(assembly, "backtesting_extra_available", lambda: False)
        monkeypatch.setattr(assembly, "research_extra_available", lambda: False)
        monkeypatch.setattr(
            backtesting_tools, "_backtesting_extra_available", lambda: False
        )
        monkeypatch.setattr(research_tools, "_research_extra_available", lambda: False)
        monkeypatch.setattr(prompts, "_backtesting_extra_available", lambda: False)

    async def test_build_server_does_not_raise(self):
        build_server()

    async def test_only_core_tools_registered(self):
        mcp = build_server()
        names = await _tool_names(mcp)
        assert names == CORE_TOOL_NAMES

    async def test_only_two_prompts_registered(self):
        mcp = build_server()
        names = await _prompt_names(mcp)
        assert names == {"analyze_stock", "review_portfolio"}


class TestSmokeRoundTrip:
    """One cheap, network-free read tool actually round-trips through the
    real assembled server -- `market_data_get_chart_links` builds static
    URLs from its `ticker` argument with no service/DB/network call, so it
    needs no stubbing."""

    async def test_get_chart_links_round_trip(self):
        mcp = build_server()
        async with Client(mcp) as client:
            result = await client.call_tool(
                "market_data_get_chart_links", {"ticker": "AAPL"}
            )
        payload = result.data
        assert payload["status"] == "success"
        assert payload["ticker"] == "AAPL"
        assert "trading_view" in payload["charts"]
