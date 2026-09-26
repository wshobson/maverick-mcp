"""The MCP server behind the `maverick-eval-client` subagent.

Claude Code starts this for each subagent (an inline `mcpServers` entry in the
agent file) and stops it when the subagent finishes. It reads the current case
from the pointer file, seeds a fresh SQLite database for that case's data state
under $TMPDIR, and then replaces itself with the Maverick stdio server. The
server gets only PATH, HOME, and its database paths, and its working directory
is outside the repository, so it never loads the repository's `.env`.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
POINTER = HERE / ".agent_case.json"


def _server_environ() -> dict[str, str]:
    return {name: os.environ[name] for name in ("PATH", "HOME") if name in os.environ}


def main() -> None:
    case = json.loads(POINTER.read_text())
    # Drop every inherited variable (API keys, Redis settings) before any
    # maverick import can read them.
    kept = _server_environ()
    os.environ.clear()
    os.environ.update(kept)

    from evals.tool_surface import harness, seed

    scratch = Path(tempfile.mkdtemp(prefix="maverick-agent-eval-")).resolve()
    database = seed.build(case["data_state"], scratch / f"{case['data_state']}.db")
    env = harness.server_env(kept, database, scratch / "cache.db")
    print(
        f"agent_server: case {case['id']} ({case['data_state']}) using {scratch}",
        file=sys.stderr,
        flush=True,
    )
    os.chdir(scratch)
    os.execve(
        sys.executable,
        [sys.executable, "-m", "maverick.server", "--transport", "stdio"],
        env,
    )


if __name__ == "__main__":
    main()
