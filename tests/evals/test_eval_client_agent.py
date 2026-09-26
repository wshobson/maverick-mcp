"""The maverick-eval-client agent file's frontmatter, as Claude Code reads it."""

from pathlib import Path

from evals.tool_surface import harness
from maverick.research import tools as research_tools

AGENT = Path(__file__).parents[2] / "evals/tool_surface/agent/maverick-eval-client.md"
RESEARCH = {harness.TOOL_PREFIX + tool.__name__ for tool in research_tools._TOOLS}


def _list_under(key: str) -> list[str]:
    """The `  - item` lines directly under a top-level frontmatter key."""
    lines = AGENT.read_text().split("---\n")[1].splitlines()
    items = []
    for line in lines[lines.index(f"{key}:") + 1 :]:
        if not line.startswith("  - "):
            break
        items.append(line.removeprefix("  - ").strip())
    return items


def test_mcp_servers_is_a_list() -> None:
    # Claude Code drops a mapping-form `mcpServers` when it loads the agent,
    # so the server never starts and every Maverick tool name is unrecognized.
    assert _list_under("mcpServers") == [f"{harness.SERVER_NAME}:"]


def test_research_tools_are_disallowed() -> None:
    # `tools` does not filter an inline server's tools; only `disallowedTools`
    # does. Without this the subagent gets the research tools the SDK harness
    # excludes.
    assert set(_list_under("disallowedTools")) == RESEARCH
    assert not RESEARCH & set(_list_under("tools"))
