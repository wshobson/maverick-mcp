"""Smoke tests for the technical analysis MCP router.

These tests verify the router module loads and the technical-tools
registration callable is wired up, without requiring external
market-data services.
"""

from fastmcp import FastMCP


def test_technical_router_importable():
    """The technical router module imports and exposes a FastMCP instance."""
    from maverick_mcp.api.routers.technical import technical_router

    assert isinstance(technical_router, FastMCP)
    assert technical_router.name == "Technical_Analysis"


async def test_register_technical_tools_registers_on_server():
    """``register_technical_tools`` wires technical analysis tools onto
    a FastMCP server instance."""
    from maverick_mcp.api.routers.tool_registry import register_technical_tools

    server = FastMCP("TestServer")
    assert len(await server.list_tools()) == 0

    register_technical_tools(server)

    tools = await server.list_tools()
    assert len(tools) > 0
