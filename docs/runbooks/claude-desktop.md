# Claude Desktop And MCP Client Setup

## Recommended Claude Desktop Configuration

Use STDIO for Claude Desktop. It avoids an extra local bridge process and keeps
the client/server relationship simple.

Running the published package via `uvx` (no local checkout needed):

```json
{
  "mcpServers": {
    "maverick-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "maverick-mcp-server",
        "maverick-mcp",
        "--transport",
        "stdio"
      ]
    }
  }
}
```

Running from a local source checkout:

```json
{
  "mcpServers": {
    "maverick-mcp": {
      "command": "uv",
      "args": [
        "run",
        "python",
        "-m",
        "maverick.server",
        "--transport",
        "stdio"
      ],
      "cwd": "/path/to/maverick-mcp"
    }
  }
}
```

Config locations:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

Restart Claude Desktop after editing the config.

## Streamable HTTP With `mcp-remote`

Use this path when a client needs a local bridge or when testing the HTTP
transport.

```bash
make dev
```

Server endpoint:

```text
http://localhost:8003/mcp/
```

Bridge config:

```json
{
  "mcpServers": {
    "maverick-mcp": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://localhost:8003/mcp/"]
    }
  }
}
```

## Claude Code CLI

```bash
claude mcp add --transport http maverick-mcp http://localhost:8003/mcp/
```

STDIO is also valid for local development:

```bash
claude mcp add maverick-mcp uv run python -m maverick.server --transport stdio
```

## Windows `cwd` Workaround

Claude Desktop on Windows has a known bug where it ignores the `"cwd"`
configuration parameter when running a local checkout via `uv`, which can
crash the server with a `ModuleNotFoundError`. Prefer the `uvx` config above
on Windows; if you need a local checkout, wrap the command in `cmd.exe` to
force the directory change:

```json
{
  "mcpServers": {
    "maverick-mcp": {
      "command": "cmd.exe",
      "args": [
        "/c",
        "cd /d C:\\Path\\To\\maverick-mcp && uv run python -m maverick.server --transport stdio"
      ]
    }
  }
}
```
