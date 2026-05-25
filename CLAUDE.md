# CLAUDE.md

This file is a Claude-specific entry point only. The canonical agent guidance is
in `AGENTS.md`, and durable project documentation is in `docs/`.

## Start Here

- `AGENTS.md`: repository guidelines, commands, MCP transport defaults, and
  safety notes.
- `docs/INDEX.md`: documentation map.
- `docs/CATALOG.md`: status of current, historical, archived, and deleted docs.
- `docs/runbooks/claude-desktop.md`: Claude Desktop setup.

## Claude Desktop Default

Prefer STDIO for Claude Desktop:

```json
{
  "mcpServers": {
    "maverick-mcp": {
      "command": "uv",
      "args": [
        "run",
        "python",
        "-m",
        "maverick_mcp.api.server",
        "--transport",
        "stdio"
      ],
      "cwd": "/path/to/maverick-mcp"
    }
  }
}
```

For bridge or remote workflows, run `make dev` and connect to
`http://localhost:8003/mcp/` with `mcp-remote`. SSE is legacy/debug only.

## Common Commands

```bash
uv sync --extra dev
make dev
make dev-stdio
make test
make lint
make typecheck
make docs-check
```

## Important Constraints

- This is a personal-use educational financial analysis server, not financial
  advice.
- Do not reintroduce auth, billing, or hosted SaaS scope without an explicit
  plan.
- Keep documentation changes cataloged in `docs/CATALOG.md`.
- Do not use `.claude/` files as the repository source of truth.
