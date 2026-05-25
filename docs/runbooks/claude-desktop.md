# Claude Desktop And MCP Client Setup

## Recommended Claude Desktop Configuration

Use STDIO for Claude Desktop. It avoids an extra local bridge process and keeps
the client/server relationship simple.

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
claude mcp add maverick-mcp uv run python -m maverick_mcp.api.server --transport stdio
```

## Legacy SSE

SSE remains available for legacy/debug clients:

```bash
make dev-sse
```

Endpoint:

```text
http://localhost:8003/sse/
```

Do not document SSE as the default Claude Desktop path.

## Windows `cwd` Workaround

If Claude Desktop ignores `cwd`, wrap the command:

```json
{
  "mcpServers": {
    "maverick-mcp": {
      "command": "cmd.exe",
      "args": [
        "/c",
        "cd /d C:\\Path\\To\\maverick-mcp && uv run python -m maverick_mcp.api.server --transport stdio"
      ]
    }
  }
}
```
