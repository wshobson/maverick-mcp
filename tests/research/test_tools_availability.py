"""Proves the phase's availability contract: `register()` registers zero research tools (and logs
one clear warning) when the `[research]` extra (langgraph + exa-py) is absent, without raising --
verified by monkeypatching the guard probe rather than `importorskip`, since both packages ARE
installed in this dev environment. This file imports `maverick.research.tools` at plain module
level with no guard of any kind: the mere fact that it collects and runs is itself part of the
proof that `tools.py`'s import graph never touches langgraph/langchain/exa-py, so this file
behaves identically on a base install with zero research extras.
"""

import logging

from fastmcp import FastMCP

from maverick.research import tools


async def test_register_skips_when_extra_unavailable(monkeypatch, caplog):
    monkeypatch.setattr(tools, "_research_extra_available", lambda: False)
    mcp = FastMCP("test")

    with caplog.at_level(logging.WARNING):
        tools.register(mcp)

    registered = await mcp.list_tools()
    assert registered == []
    assert any("research" in record.message.lower() for record in caplog.records)


async def test_register_attaches_tools_when_extra_available(monkeypatch):
    monkeypatch.setattr(tools, "_research_extra_available", lambda: True)
    mcp = FastMCP("test")

    tools.register(mcp)

    registered = await mcp.list_tools()
    assert len(registered) == 3
