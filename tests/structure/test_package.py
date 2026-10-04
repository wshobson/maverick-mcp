"""Structural checks for the new maverick package."""


def test_version_is_importable():
    import maverick

    assert maverick.__version__ == "1.1.0"


def verify_core_wheel_installation() -> None:
    """Called directly in a fresh core-only venv outside the source checkout."""
    import asyncio
    import importlib.util
    import json
    import os
    import socket
    import sys
    import tempfile
    from importlib.metadata import distribution
    from pathlib import Path
    from unittest.mock import patch

    from fastmcp import Client

    import maverick

    origin = Path(maverick.__file__).resolve()
    assert "site-packages" in origin.parts, origin
    assert Path(sys.prefix) in origin.parents, (origin, sys.prefix)
    direct_url = distribution("maverick-mcp-server").read_text("direct_url.json")
    assert not direct_url or not json.loads(direct_url).get("dir_info", {}).get(
        "editable"
    )
    for module in (
        "vectorbt",
        "sklearn",
        "langgraph",
        "exa_py",
        "langchain_core",
        "langchain_anthropic",
        "langchain_openai",
    ):
        assert importlib.util.find_spec(module) is None, module

    def block_network(*args, **kwargs):
        raise AssertionError("The core-wheel smoke must not open network connections")

    async def smoke():
        from maverick.platform.config import get_platform_settings
        from maverick.server.assembly import build_server

        assert get_platform_settings().database.url == database_url
        server = build_server()
        names = {tool.name for tool in await server.list_tools()}
        assert len(names) == 38, names
        assert "portfolio_get_my_portfolio" in names
        assert not any(name.startswith(("research_", "backtesting_")) for name in names)
        async with Client(server) as client:
            response = await client.call_tool("portfolio_get_my_portfolio", {})
        assert response.data["status"] == "success", response.data
        assert response.data["positions"] == [], response.data

    with tempfile.TemporaryDirectory() as temporary:
        database_path = Path(temporary) / "core-smoke.db"
        database_url = f"sqlite:///{database_path}"
        os.environ["DATABASE_URL"] = database_url
        os.environ.pop("POSTGRES_URL", None)
        os.environ.pop("DB_URL", None)
        os.environ["CI"] = "false"
        os.environ["GITHUB_ACTIONS"] = "false"
        for name in tuple(os.environ):
            if name.startswith(("REDIS_", "LLM_", "EXA_", "RESEARCH_", "SEARXNG_")):
                os.environ.pop(name)
        with patch.object(socket.socket, "connect", block_network):
            asyncio.run(smoke())
        assert database_path.is_file()
    print(
        f"Core wheel passed: {origin}; optional packages/tools absent; offline call succeeded"
    )


if __name__ == "__main__":
    import os
    import tempfile
    from unittest.mock import patch

    # Every CI run also verifies that inherited database URLs are overridden.
    # Nonexistent SQLite directories reproduce leaks without reaching a real DB.
    with tempfile.TemporaryDirectory() as inherited_directory:
        invalid_url = f"sqlite:///{inherited_directory}/missing/database.db"
        with patch.dict(
            os.environ,
            {
                "DATABASE_URL": invalid_url,
                "POSTGRES_URL": invalid_url,
                "CI": "false",
                "GITHUB_ACTIONS": "false",
            },
        ):
            verify_core_wheel_installation()
