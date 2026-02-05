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

- The MCP spec has a published changelog; notable modern changes include:
  - **Streamable HTTP transport** support (alongside SSE) and other protocol feature additions.
  - **Origin header validation** guidance to mitigate DNS rebinding attacks (relevant if you expose HTTP endpoints beyond localhost).
  - Structured additions like tool annotations / elicitation improvements in newer revisions.
  - Sources: MCP spec changelog and revision notes: https://modelcontextprotocol.io/specification/changelog and https://modelcontextprotocol.io/specification/2025-11-25

### FastMCP (Python SDK) – current state (Feb 2026)

- FastMCP is actively shipping; PyPI shows a current stable line in the **2.x** series with a **3.0 beta** available.
  - Latest stable: **2.14.5 (2026-02-03)**; pre-release: **3.0.0b1 (2026-01-20)**.
  - FastMCP docs explicitly mention v3 is in development and recommend pinning `<3` for now.
  - Sources: PyPI project page + releases, FastMCP docs: https://pypi.org/project/fastmcp/ and https://gofastmcp.com/

- FastMCP 2.14 introduces capabilities that are directly relevant to MaverickMCP’s “long-running tool call” profile:
  - **Protocol-native background tasks** (add `task=True` to async tools) for progress reporting without blocking clients.
  - **Sampling with tools** (`ctx.sample()` / `ctx.sample_step()`) to leverage client LLM capabilities for agentic workflows, including structured outputs.
  - Sources: FastMCP Updates (2.14.0, 2.14.1): https://gofastmcp.com/updates

- HTTP transport APIs in FastMCP include helpers like `create_sse_app` and `create_streamable_http_app`.
  - Source: FastMCP HTTP docs: https://fastmcp.wiki/en/python-sdk/fastmcp-server-http

### Known transport sharp edges (recent)

- There have been real-world compatibility issues reported in FastMCP around SSE/tool fetching regressions (e.g., around v2.12.x) and route redirect/trailing-slash behavior behind proxies.
  - These reports suggest MaverickMCP should periodically re-validate whether local monkey-patches/hacks are still needed, especially after upgrading FastMCP.
  - Sources: FastMCP GitHub issues: https://github.com/jlowin/fastmcp/issues/1903 and https://github.com/jlowin/fastmcp/issues/1364

## Implications for MaverickMCP

- Consider upgrading to the latest stable `fastmcp` 2.14.x and **re-evaluating**:
  - the SSE trailing-slash monkey-patch in `maverick_mcp/api/server.py`
  - tool registration strategy (direct registry vs router mounting) given upstream fixes around naming/prefixing/redirect handling
- For tools like “deep research” and heavyweight backtests, FastMCP’s **background tasks** could replace ad-hoc timeout + logging patterns and enable better UX (progress + cancellability).
- If you want to lean into “agentic workflows,” FastMCP’s **sampling with tools** can potentially unify some of the in-repo orchestration logic with protocol-native patterns (subject to client support).
