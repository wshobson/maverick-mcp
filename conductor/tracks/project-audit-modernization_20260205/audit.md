# Audit: MaverickMCP Project State

**Track ID:** project-audit-modernization_20260205
**Updated:** 2026-02-05

## Executive Summary

MaverickMCP is feature-rich and has a lot of working infrastructure (providers, caching, monitoring, agents, and a substantial test suite). The main maintainability risk is **architecture drift**: there are multiple parallel implementations and “compatibility” layers (e.g., `api/server.py` vs `api/api_server.py`, multiple router variants, and both interface-based providers and older concrete providers), which increases cognitive load and makes modernization harder.

The best path forward is to **pick a single “golden path”** for:

- server entrypoints + transports
- tool registration strategy
- provider abstractions
- where orchestration/agent logic lives

…then deprecate/remove (or quarantine) the alternative paths.

## Task 1.1 — Golden Path Decision (Entrypoint + Transport)

This repo currently supports multiple “ways to run” and multiple MCP transports (STDIO, SSE, Streamable HTTP). For long-term maintainability and client compatibility, the recommended “golden path” to standardize on is:

- **Primary entrypoint:** `python -m maverick_mcp.api.server` (one server module, multiple transports via `--transport`)
- **Primary Claude Desktop connection:** **STDIO transport** (Claude Desktop launches the server as a subprocess)
  - Rationale: eliminates the `mcp-remote` bridge layer, avoids HTTP/SSE edge cases, and keeps tool registration simplest.
- **Primary remote/client transport:** **Streamable HTTP** (for clients that connect over HTTP)
  - Rationale: aligns with modern MCP transport guidance and avoids SSE-specific proxy/trailing-slash pitfalls.
- **SSE transport:** keep as **debug/inspector-only** until proven stable and necessary for a specific client.
  - Rationale: this repo contains SSE-specific compatibility patches and documentation contradictions, which is a sign SSE is currently a sharp edge here.

Concrete “golden path” commands/config (to be codified in a follow-on implementation/doc-unification track):

- Claude Desktop (STDIO): `uv run python -m maverick_mcp.api.server --transport stdio`
- HTTP (Streamable): `uv run python -m maverick_mcp.api.server --transport streamable-http --host 127.0.0.1 --port 8003`

## What’s Working Well

- **Clear separation by domain area** via routers (technical, screening, portfolio, research, monitoring): `maverick_mcp/api/routers/`
- **Interface-driven provider layer** is clean and well-documented, enabling gradual migration: `maverick_mcp/providers/interfaces/`, `maverick_mcp/providers/implementations/`, `maverick_mcp/providers/factories/provider_factory.py`
- **Research agent foundations** use modern LangGraph patterns and have explicit tuning knobs (depth levels/personas) and resiliency primitives: `maverick_mcp/agents/deep_research.py`
- **Meaningful test investment**, including orchestration/integration/performance coverage: `tests/` and targeted in-memory MCP tests: `maverick_mcp/tests/`

## What’s Not Working Well

- **Too many entrypoints / server modes**: a large “everything” MCP server in `maverick_mcp/api/server.py` plus a separate simplified FastAPI server in `maverick_mcp/api/api_server.py` plus additional SSE/inspector variants in `maverick_mcp/api/`.
- **Compatibility hacks are embedded in core startup** (monkey-patching FastMCP SSE route handling, heavy warning suppression, and `print()` diagnostics): `maverick_mcp/api/server.py`
- **Multiple router variants** (`*_enhanced`, `*_ddd`, `*_parallel`) without a clear deprecation path creates duplication and ambiguity for “the right” implementation: `maverick_mcp/api/routers/`
- **Documentation drift / contradictions** around recommended transports (SSE vs streamable HTTP) and client compatibility: compare `CLAUDE.md` vs `maverick_mcp/README.md`
- **Global side effects in library modules** (e.g., `logging.basicConfig()`, `load_dotenv()`), which can interfere with consumers/tests and makes behavior harder to predict: `maverick_mcp/providers/stock_data.py`, `maverick_mcp/config/settings.py`
- **Partially-finished “registry” concepts** (LangChain ToolRegistry default tool init TODO): `maverick_mcp/langchain_tools/registry.py`

## Task 1.2 — Legacy / Non-Canonical Paths (What to Deprecate)

Once the golden path is adopted, the following should be explicitly labeled **legacy** (and then removed or quarantined in a follow-on implementation track), unless there is a strong client-driven reason to keep them:

### Server entrypoints / modes

- `maverick_mcp/api/api_server.py` (separate simplified FastAPI server)
- `maverick_mcp/api/simple_sse.py`, `maverick_mcp/api/inspector_sse.py`, `maverick_mcp/api/inspector_compatible_sse.py` (SSE variants)
- Any transport-specific “dev” guidance that contradicts the golden path (currently seen across `CLAUDE.md` and `maverick_mcp/README.md`)

### Transport-specific compatibility hacks

- The SSE trailing-slash monkey-patch in `maverick_mcp/api/server.py` should be treated as a **temporary compatibility shim**, not part of the long-term architecture.

### Router variants that aren’t part of canonical tool registration

The canonical tool registration today is `maverick_mcp/api/routers/tool_registry.py` (direct tool registration onto the main server to avoid client tool-name issues). Router variants not used by the registry should be treated as experiments/legacy:

- `maverick_mcp/api/routers/screening_ddd.py`
- `maverick_mcp/api/routers/screening_parallel.py`
- `maverick_mcp/api/routers/technical_ddd.py`
- Potentially `maverick_mcp/api/routers/data_enhanced.py` and `maverick_mcp/api/routers/health_enhanced.py` if the enhancements are not actually used by the registry and/or not part of the chosen architecture.

## Task 1.3 — Refactor Sequencing Plan (P0/P1/P2)

This is a repo-grounded sequencing plan intended to minimize risk and reduce ambiguity before larger refactors.

### P0 (clarity + stability first)

- **Codify the golden path** (entrypoint + transport + client setup) and reconcile contradictions in `CLAUDE.md` vs `maverick_mcp/README.md`.
- **Upgrade FastMCP within stable 2.x** and re-validate transport behavior (goal: remove SSE monkey-patch or confine it to an isolated shim).
- **Centralize startup side effects** (logging and environment loading) into a single bootstrap path (no module-level `basicConfig()` / `load_dotenv()` in library modules).
- **Declare canonical tool registration** (keep `tool_registry.py` as the only supported strategy unless/until client issues are proven resolved upstream).

### P1 (architecture consolidation)

- Unify server entrypoints: one supported server module + a thin CLI/wrapper; deprecate alternate modes.
- Modularize `maverick_mcp/api/server.py` by concern (bootstrap/config, transports, tool registration, resources/prompts, shutdown).
- Consolidate provider strategy (finish the interfaces/adapters migration or remove it; avoid a permanent hybrid).

### P2 (maintainability + hygiene)

- Consolidate test suite organization (`tests/` vs `maverick_mcp/tests/`) and formalize “fast vs slow” conventions (markers/targets).
- Address “unfinished registry/init” TODOs (e.g., `maverick_mcp/langchain_tools/registry.py`).

## Hotspots / Refactor Candidates

- `maverick_mcp/api/server.py`: very large, mixes concerns (bootstrapping, transport quirks, monitoring init, prompts, tools, resources, shutdown). Candidate for splitting into `server/bootstrap.py`, `server/transports.py`, `server/resources.py`, etc.
- `maverick_mcp/agents/deep_research.py`: very large; likely needs modularization (providers, prompt/policy, graph wiring, result shaping) to keep changes safe.
- `maverick_mcp/data/models.py`: large ORM surface area; consider splitting by bounded context (portfolio vs screening vs cache tables).
- `maverick_mcp/config/settings.py`: large config surface with mixed sources; likely needs consolidation to a single settings approach (and removal of module-level logging config).
- Tool registration strategy: `maverick_mcp/api/routers/tool_registry.py` + “register router tools directly” pattern vs any router mounting patterns; decide a single stable approach and retire the other.

## Test Coverage Notes

- There are **two test locations** (`tests/` and `maverick_mcp/tests/`). If both are intentional, document the distinction; otherwise, consolidate to one to reduce duplication and import-path confusion.
- Many tests appear “large” (multi-hundred/1k+ LOC integration/perf tests). That’s valuable, but also suggests a need for a small “fast suite” vs “slow suite” convention if not already present (e.g., markers, Make targets).
