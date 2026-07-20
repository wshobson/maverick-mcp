"""Proves the phase's availability contract: `register()` registers zero backtesting tools (and
logs one clear warning) when the `[backtesting]` extra is absent, without raising -- verified by
monkeypatching the guard probe rather than `importorskip`, since vectorbt IS installed in this
dev environment. This file imports `maverick.backtesting.tools` at plain module level with no
guard of any kind: the mere fact that it collects and runs is itself part of the proof that
`tools.py`'s import graph never touches vectorbt/sklearn, so this file behaves identically on a
base install with zero backtesting extras.
"""

import logging

from fastmcp import FastMCP

from maverick.backtesting import tools


async def test_register_skips_when_extra_unavailable(monkeypatch, caplog):
    monkeypatch.setattr(tools, "_backtesting_extra_available", lambda: False)
    mcp = FastMCP("test")

    with caplog.at_level(logging.WARNING):
        tools.register(mcp)

    registered = await mcp.list_tools()
    assert registered == []
    assert any("backtesting" in record.message.lower() for record in caplog.records)


async def test_register_attaches_tools_when_extra_available(monkeypatch):
    monkeypatch.setattr(tools, "_backtesting_extra_available", lambda: True)
    mcp = FastMCP("test")

    tools.register(mcp)

    registered = await mcp.list_tools()
    assert len(registered) == 11
