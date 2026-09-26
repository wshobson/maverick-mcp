"""The environment `evals/tool_surface/agent_server.py` gives the Maverick server."""

from pathlib import Path

from evals.tool_surface import agent_server


def test_server_env_is_explicit() -> None:
    env = agent_server.server_env(
        {"HOME": "/h", "PATH": "/bin", "EXA_API_KEY": "x"},
        Path("/s/maverick.db"),
        Path("/s/cache.db"),
    )
    assert env == {
        "HOME": "/h",
        "PATH": "/bin",
        "DATABASE_URL": "sqlite:////s/maverick.db",
        "CACHE_SQLITE_PATH": "/s/cache.db",
    }
