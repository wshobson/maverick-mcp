# MCP Client Setup

MaverickMCP is a standard MCP server with no client-specific behavior. Any
client that speaks the Model Context Protocol can use it. This runbook covers
the two transports and gives verified configuration for the common clients.

## Pick A Transport First

Everything below is one of two choices. Get this right and the per-client
config is mechanical.

| | STDIO | Streamable HTTP |
| --- | --- | --- |
| Who starts the server | The client, as a subprocess | You, ahead of time (`make dev`) |
| Endpoint | n/a | `http://localhost:8003/mcp` |
| Best for | Single local client, simplest setup | Several clients sharing one server, or remote access |
| Config shape | `command` + `args` | `url` |

STDIO is the default (`--transport stdio`) and the right choice for a single
local client: no port, no process to leave running, no network surface.

Use Streamable HTTP when more than one client should share a single server
process, or when you are connecting from somewhere other than this machine.

> [!IMPORTANT]
> The HTTP endpoint has **no trailing slash**: `http://localhost:8003/mcp`.
> Requesting `/mcp/` returns a `307` redirect to `/mcp`, and clients that do
> not follow redirects on `POST` will fail to register tools.

## Endpoint And Binding

```bash
make dev     # Streamable HTTP on http://localhost:8003/mcp
make dev-stdio   # STDIO on this terminal
```

The HTTP transport binds `127.0.0.1` by default, so it is reachable only from
this machine. To expose it on your LAN:

```bash
uv run python -m maverick.server --transport http --host 0.0.0.0 --port 8003
```

Only do that on a network you trust. The server has no authentication -- that
is deliberate for a personal-use tool (see `AGENTS.md`) -- so anyone who can
reach the port can call every tool, including the ones that write to your
portfolio and trade journal.

## Client Support At A Glance

| Client | STDIO | HTTP | Config location |
| --- | --- | --- | --- |
| Claude Desktop | Yes (incl. `.mcpb`) | Via `mcp-remote` | `claude_desktop_config.json` |
| Claude Code | Yes | Yes | `claude mcp add` |
| VS Code (Copilot) | Yes | Yes | `.vscode/mcp.json` |
| GitHub Copilot CLI | Yes | Yes | `~/.copilot/mcp-config.json` or `.mcp.json` |
| Codex CLI | Yes | Yes | `~/.codex/config.toml` or `.codex/config.toml` |
| Cursor | Yes | Yes | `~/.cursor/mcp.json` or `.cursor/mcp.json` |
| OpenCode | Yes | Yes | `~/.config/opencode/opencode.json` or `opencode.json` |
| Antigravity CLI | Yes | Yes | `~/.gemini/config/mcp_config.json` or `.agents/mcp_config.json` |
| Zed, LM Studio, Goose, Cline, Continue | Yes | Varies | Client-specific |

Ordered roughly by developer adoption (JetBrains Developer Ecosystem Survey,
May-July 2026). Clients not listed still work.

Clients not listed here still work. Give them either the STDIO command or the
HTTP endpoint in whatever shape their config expects.

## Claude Desktop

Claude Desktop's `claude_desktop_config.json` launches **local STDIO servers
only**. Use `uvx` to run the published package with no checkout:

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

From a local source checkout instead:

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

Fully quit and restart Claude Desktop after editing; the config is read-only at
startup.

> [!WARNING]
> Do **not** add `http://localhost:8003/mcp` as a Claude Desktop "custom
> connector". Custom connectors are brokered from Anthropic's cloud
> infrastructure rather than from your machine, so they cannot reach a server
> on your localhost. For Claude Desktop, local means STDIO or a `.mcpb`
> bundle.

### `.mcpb` Bundle

`make bundle` builds `dist/maverick-mcp.mcpb`, a one-click installable bundle
(Settings -> Extensions). It launches the PyPI-published package via `uvx`, so
it requires `uv` on the machine and only works once the package is published.
See `docs/runbooks/releasing.md`.

### Streamable HTTP Via `mcp-remote`

Claude Desktop's config file cannot express an HTTP server directly. `mcp-remote`
bridges it: the client still launches a STDIO subprocess, and that subprocess
forwards to your running HTTP server.

```bash
make dev
```

```json
{
  "mcpServers": {
    "maverick-mcp": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://localhost:8003/mcp"]
    }
  }
}
```

Claude Desktop is the main client that still needs this bridge. The clients
below all speak Streamable HTTP natively -- do not wrap them in `mcp-remote`.

### Windows `cwd` Workaround

Claude Desktop on Windows has a known bug where it ignores `"cwd"` when running
a local checkout via `uv`, crashing the server with `ModuleNotFoundError`.
Prefer the `uvx` config above. If you need a local checkout, wrap the command
in `cmd.exe` to force the directory change:

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

## Claude Code

HTTP, against an already-running `make dev`:

```bash
claude mcp add --transport http maverick-mcp http://localhost:8003/mcp
```

STDIO. Note the `--` separator: without it, `claude mcp add` consumes
`--transport stdio` as its own flag instead of passing it to the server.
`claude mcp add` has no `cwd` option, so point `uv` at the checkout with
`--directory`:

```bash
claude mcp add maverick-mcp -- \
  uv run --directory /path/to/maverick-mcp python -m maverick.server --transport stdio
```

Add `--scope user` to make the server available outside the current project.
Verify with `claude mcp list`.

## Cursor

`~/.cursor/mcp.json` (global) or `.cursor/mcp.json` (per project).

HTTP:

```json
{
  "mcpServers": {
    "maverick-mcp": {
      "url": "http://localhost:8003/mcp"
    }
  }
}
```

STDIO:

```json
{
  "mcpServers": {
    "maverick-mcp": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/maverick-mcp",
        "python",
        "-m",
        "maverick.server",
        "--transport",
        "stdio"
      ]
    }
  }
}
```

## VS Code (GitHub Copilot)

`.vscode/mcp.json` in the workspace, or the user profile `mcp.json`. Note the
key is `servers`, not `mcpServers`, and `type` is required.

HTTP:

```json
{
  "servers": {
    "maverick-mcp": {
      "type": "http",
      "url": "http://localhost:8003/mcp"
    }
  }
}
```

STDIO:

```json
{
  "servers": {
    "maverick-mcp": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "python", "-m", "maverick.server", "--transport", "stdio"],
      "cwd": "${workspaceFolder}"
    }
  }
}
```

## Codex CLI

`~/.codex/config.toml` (global) or `.codex/config.toml` (trusted projects). The
ChatGPT desktop app, Codex CLI, and the IDE extension share this config.

HTTP:

```toml
[mcp_servers.maverick-mcp]
url = "http://localhost:8003/mcp"
```

STDIO:

```toml
[mcp_servers.maverick-mcp]
command = "uv"
args = ["run", "python", "-m", "maverick.server", "--transport", "stdio"]
cwd = "/path/to/maverick-mcp"
startup_timeout_sec = 30
```

Or via the CLI:

```bash
codex mcp add maverick-mcp --url http://localhost:8003/mcp
codex mcp add maverick-mcp -- uv run --directory /path/to/maverick-mcp python -m maverick.server --transport stdio
```

Raise `startup_timeout_sec` if a cold `uvx` resolve times out on first launch;
the stock default is tight enough that a slow first resolve can look like a
broken server.

## Antigravity CLI

Antigravity CLI (`agy`) is Google's replacement for Gemini CLI, which stopped
serving Google AI Pro, Ultra, and free individual accounts on 2026-06-18.
Gemini CLI remains available on Gemini Code Assist Standard/Enterprise licenses
and paid API keys; if you are still on it, its config is
`~/.gemini/settings.json` with `httpUrl` for Streamable HTTP.

Antigravity keeps MCP servers in a dedicated file rather than inline in
settings:

- Global: `~/.gemini/config/mcp_config.json`
- Workspace: `.agents/mcp_config.json`

Remote servers use `serverUrl`. The legacy Gemini CLI keys `url` and `httpUrl`
are not used -- that is the one change that breaks a copied config.

```json
{
  "mcpServers": {
    "maverick-mcp": {
      "serverUrl": "http://localhost:8003/mcp"
    }
  }
}
```

STDIO:

```json
{
  "mcpServers": {
    "maverick-mcp": {
      "command": "uv",
      "args": ["run", "python", "-m", "maverick.server", "--transport", "stdio"],
      "cwd": "/path/to/maverick-mcp"
    }
  }
}
```

Or via the CLI, which writes the same file. Flags must precede the server name,
and `--` is required before a command that starts with `-`:

```bash
agy mcp add maverick-mcp http://localhost:8003/mcp
agy mcp add maverick-mcp -- uv run --directory /path/to/maverick-mcp python -m maverick.server --transport stdio
```

`agy mcp list` shows status; `/mcp` inside the prompt panel opens the
interactive manager.

## GitHub Copilot CLI

Config lives at `~/.copilot/mcp-config.json` (user scope) or `.mcp.json` in the
repository for project scope. Note the `tools` filter, which is specific to
Copilot: `["*"]` exposes every tool, or list tool names to narrow it.

HTTP:

```json
{
  "mcpServers": {
    "maverick-mcp": {
      "type": "http",
      "url": "http://localhost:8003/mcp",
      "tools": ["*"]
    }
  }
}
```

STDIO:

```json
{
  "mcpServers": {
    "maverick-mcp": {
      "type": "local",
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/maverick-mcp",
        "python",
        "-m",
        "maverick.server",
        "--transport",
        "stdio"
      ],
      "tools": ["*"]
    }
  }
}
```

Or via the CLI, which writes the same file. As with Claude Code, `--` is
required before a STDIO command:

```bash
copilot mcp add --transport http maverick-mcp http://localhost:8003/mcp
copilot mcp add maverick-mcp -- uv run --directory /path/to/maverick-mcp python -m maverick.server --transport stdio
```

`/mcp add` inside interactive mode opens a form for the same thing. The GitHub
MCP server is built in and needs no configuration.

## OpenCode

Config lives at `~/.config/opencode/opencode.json` (global) or `opencode.json`
in the project root; project config wins on conflicting keys. Servers go under
the `mcp` key, and each entry sets `type` to `remote` or `local`.

Remote (Streamable HTTP):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "maverick-mcp": {
      "type": "remote",
      "url": "http://localhost:8003/mcp",
      "enabled": true
    }
  }
}
```

Local (STDIO). Note that `command` is a single array holding the executable and
all of its arguments, unlike the `command` plus `args` split most clients use:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "maverick-mcp": {
      "type": "local",
      "command": [
        "uv",
        "run",
        "--directory",
        "/path/to/maverick-mcp",
        "python",
        "-m",
        "maverick.server",
        "--transport",
        "stdio"
      ],
      "enabled": true
    }
  }
}
```

`opencode mcp list` shows connection status. OpenCode counts MCP tools against
the model's context, so enable only the servers you need.

> [!NOTE]
> The above is the schema for the current release (verified on 1.18.21, the
> published `latest`). OpenCode's v2 documentation describes a different shape:
> servers nest under `mcp.servers`, and `enabled` is replaced by `disabled`.
> No 2.x is published yet, and 1.18.21 rejects the `mcp.servers` form with a
> config validation error, so use the schema above until v2 ships.

## Any Other Client

There is nothing client-specific to configure. Supply one of:

- **STDIO**: command `uv`, args
  `["run", "python", "-m", "maverick.server", "--transport", "stdio"]`, with the
  working directory set to your checkout. For the published package, command
  `uvx` with args
  `["--from", "maverick-mcp-server", "maverick-mcp", "--transport", "stdio"]`.
- **Streamable HTTP**: `http://localhost:8003/mcp` after `make dev`.

## Troubleshooting

**Tools do not appear.** Check the transport shape first. A client configured
for STDIO against a running HTTP server (or the reverse) fails silently in most
clients. Then confirm the URL has no trailing slash.

**`307 Temporary Redirect` in the logs.** You used `http://localhost:8003/mcp/`.
Drop the trailing slash.

**Server exits immediately under STDIO.** The client is likely resolving the
wrong working directory. Use an absolute path in `cwd`, or `uv run --directory
/absolute/path` for clients that have no `cwd` field.

**Nothing on port 8003.** `make dev` may have failed to bind. Run `make stop`,
then `make dev`, and check `make tail-log`.

**Connection refused from another machine.** The HTTP transport binds
`127.0.0.1`. Restart with `--host 0.0.0.0`, and read the warning in
[Endpoint And Binding](#endpoint-and-binding) before you do.

**Empty screening results.** Not a transport problem. There is no pre-seeded
ticker universe; see `docs/runbooks/database-setup.md`.
