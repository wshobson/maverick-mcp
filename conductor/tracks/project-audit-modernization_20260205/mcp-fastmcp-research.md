# MCP + FastMCP Research Notes

**Track ID:** project-audit-modernization_20260205
**Updated:** 2026-02-05

## Questions to Answer

- What are the current recommended MCP transports and capabilities?
- What has changed recently in MCP or major SDKs?
- What does the latest FastMCP support well (and what patterns are recommended)?
- What modernization opportunities apply directly to MaverickMCP?

## Findings

### MCP protocol (spec-level)

- MCP’s **Streamable HTTP** transport is now the recommended HTTP transport and explicitly **replaces the older HTTP+SSE transport** (from the 2024-11-05 protocol version).
  - The spec requires a **single MCP endpoint path** that supports **both POST and GET** (example: `https://example.com/mcp`).
  - Servers may optionally use SSE as part of Streamable HTTP for streaming server messages, but the transport itself is no longer “SSE-only”.
  - Sources: MCP transports (examples + requirements) and security warnings: https://modelcontextprotocol.io/specification/2025-11-25/basic/transports

- **Security guidance matters for local, unauthenticated servers**:
  - Servers **MUST validate the `Origin` header** to prevent DNS rebinding attacks.
  - When running locally, servers **SHOULD bind to localhost** (127.0.0.1) rather than `0.0.0.0`.
  - Source: MCP transports security warning: https://modelcontextprotocol.io/specification/2025-11-25/basic/transports

- Related ecosystem signal: the MCP TypeScript SDK had a high-severity advisory regarding DNS rebinding protection defaults for HTTP-based transports (fixed in `@modelcontextprotocol/sdk` 1.24.0), reinforcing that Origin/host validation is an ecosystem-wide concern.
  - Source: GitHub advisory (CVE-2025-66414): https://github.com/advisories/GHSA-w48q-cv73-mx4w

### FastMCP (Python SDK) – current state (Feb 2026)

- FastMCP is actively shipping in a stable **2.x** line.
  - As of **2026-02-05**, PyPI shows latest release **2.14.4** (uploaded **2026-01-22**).
  - Source: PyPI release history: https://pypi.org/project/fastmcp/

- FastMCP 2.14 introduces capabilities that are directly relevant to MaverickMCP’s “long-running tool call” profile:
  - **Protocol-native background tasks** (new in **2.14.0**) that allow clients to request background execution and poll/cancel without blocking.
    - Server-side docs: https://gofastmcp.com/servers/tasks
    - Client-side docs: https://gofastmcp.com/clients/tasks
  - **Task configuration** via `TaskConfig(mode="optional"|"required"|"forbidden")` to control whether background execution is allowed/required.
    - Note: docs mention additional task features (e.g., poll interval) as “new in 2.15.0”; verify availability when upgrading.

- HTTP transport APIs in FastMCP include helpers like `create_sse_app` and `create_streamable_http_app`.
  - Source: FastMCP HTTP docs: https://fastmcp.wiki/en/python-sdk/fastmcp-server-http

### Known transport sharp edges (recent)

- There have been real-world compatibility issues reported in FastMCP around SSE/tool fetching regressions (e.g., around v2.12.x) and route redirect/trailing-slash behavior behind proxies.
  - These reports suggest MaverickMCP should periodically re-validate whether local monkey-patches/hacks are still needed, especially after upgrading FastMCP.
  - Sources: FastMCP GitHub issues: https://github.com/jlowin/fastmcp/issues/1903 and https://github.com/jlowin/fastmcp/issues/1364

## Implications for MaverickMCP

- Consider upgrading/pinning to the latest stable `fastmcp` 2.14.x and **re-evaluating**:
  - the SSE trailing-slash monkey-patch in `maverick_mcp/api/server.py`
  - tool registration strategy (direct registry vs router mounting) given upstream fixes around naming/prefixing/redirect handling
- For tools like “deep research” and heavyweight backtests, FastMCP’s **background tasks** could replace ad-hoc timeout + logging patterns and enable better UX (progress + cancellability).
- If you want to lean into “agentic workflows,” FastMCP’s **sampling with tools** can potentially unify some of the in-repo orchestration logic with protocol-native patterns (subject to client support).
