"""Tests for maverick.server.prompts.

Mirrors `tests/backtesting/test_tools_availability.py`'s pattern: monkeypatch
the module's own private probe alias rather than uninstalling anything (both
extras ARE installed in this dev environment).
"""

from fastmcp import FastMCP

from maverick.server import prompts


async def test_register_attaches_all_three_when_extra_available(monkeypatch):
    monkeypatch.setattr(prompts, "_backtesting_extra_available", lambda: True)
    mcp = FastMCP("test")

    prompts.register(mcp)

    registered = await mcp.list_prompts()
    names = {p.name for p in registered}
    assert names == {"analyze_stock", "review_portfolio", "run_backtest_workflow"}


async def test_register_skips_backtest_workflow_when_extra_unavailable(monkeypatch):
    monkeypatch.setattr(prompts, "_backtesting_extra_available", lambda: False)
    mcp = FastMCP("test")

    prompts.register(mcp)

    registered = await mcp.list_prompts()
    names = {p.name for p in registered}
    assert names == {"analyze_stock", "review_portfolio"}


async def test_analyze_stock_references_real_tool_names():
    text = await prompts.analyze_stock("aapl")
    assert "AAPL" in text
    assert "market_data_get_quote" in text
    assert "technical_get_full_technical_analysis" in text
    assert "screening_get_by_criteria" in text


async def test_review_portfolio_references_real_tool_names():
    text = await prompts.review_portfolio("My Portfolio")
    assert "portfolio_get_my_portfolio" in text
    assert "portfolio_get_risk_dashboard" in text
    assert "portfolio_get_risk_alerts" in text


async def test_run_backtest_workflow_references_real_tool_names():
    text = await prompts.run_backtest_workflow("tsla", "rsi")
    assert "TSLA" in text
    assert "backtesting_run_backtest" in text
    assert "backtesting_optimize_strategy" in text
