# Modernization Roadmap

**Track ID:** project-audit-modernization_20260205
**Updated:** 2026-02-05

## Goals

- Reduce architecture drift by choosing and documenting a single “golden path”
- Make MCP transports/tool registration robust without embedded hacks
- Improve maintainability by modularizing the largest files and removing global side effects
- Align with current MCP/FastMCP best practices (transports, tooling, resource/prompt patterns)

## Proposed Sequencing

1. **Decide scope**: recommendations-only vs “quick wins” implementation in this track.
2. **Server/transport decision**: choose the primary transport(s) and primary entrypoint module(s).
3. **Tool registration strategy**: choose direct registration vs router mounting, and codify it.
4. **Architecture consolidation**: pick provider abstraction (interfaces/adapters) and deprecate alternatives.
5. **Modularize hotspots**: split `api/server.py` and `agents/deep_research.py` once architecture decisions are made.
6. **Cleanups**: remove module-level `basicConfig()` / `load_dotenv()` patterns; centralize startup config.

## Safe Wins (First Implementation Session)

These should be the first changes when you start implementation work (aim: low risk, high clarity):

1. **Choose the golden path** and encode it in docs + Make targets:
   - Decide whether Claude Desktop will be supported via `streamable-http` (likely) and/or SSE.
   - Reconcile the current contradiction between `CLAUDE.md` and `maverick_mcp/README.md`.
2. **Upgrade FastMCP within stable 2.x**, then re-test transports/tool registration:
   - Re-evaluate whether `maverick_mcp/api/server.py` still needs the SSE trailing-slash monkey-patch.
3. **Centralize startup side effects**:
   - Remove module-level `logging.basicConfig()` and `load_dotenv()` calls from library modules.
   - Keep all environment loading and logging configuration in one bootstrap path.
4. **Canonicalize routers**:
   - Choose canonical router modules (e.g., “enhanced” becomes default) and mark others legacy.
5. **Adopt background tasks** where appropriate:
   - Convert long-running tools (deep research, heavy backtests) to protocol-native background tasks to improve responsiveness and reduce timeouts.

### Safe Wins (Recommendations-Only for This Track)

This track does not implement changes, but the following are “first changes to ship” candidates for the follow-on implementation track:

- **Golden path = STDIO for Claude Desktop + Streamable HTTP for HTTP clients**, with `maverick_mcp.api.server` as the only supported entrypoint.
- **Bind localhost by default** for HTTP transports (align with MCP security guidance) and document explicit opt-in for `0.0.0.0`.
- **Remove or isolate transport shims** (SSE trailing-slash patch) behind a small compatibility layer, re-validated after a FastMCP upgrade.
- **Make tool registration canonical**: keep `routers/tool_registry.py` as the supported mechanism unless upstream + client behavior proves mounted routers are safe.

## Refactor Backlog

P0 (highest impact / lowest regret)

- Unify entrypoint(s): define a single supported way to run the MCP server and mark others as legacy (`maverick_mcp/api/server.py`, `maverick_mcp/api/api_server.py`, `maverick_mcp/api/simple_sse.py`, inspector variants).
- Remove/relocate global side effects (`logging.basicConfig`, `load_dotenv`) to a single startup/bootstrap location (`maverick_mcp/providers/stock_data.py`, `maverick_mcp/config/settings.py`).
- Establish “canonical routers” and deprecate variants (`maverick_mcp/api/routers/*_enhanced.py`, `*_ddd.py`, `*_parallel.py`) with a short migration plan.
- Upgrade `fastmcp` within the stable 2.x line and re-validate transport behavior to remove hacks (see `mcp-fastmcp-research.md`).
- Align HTTP transports with MCP spec guidance (notably Origin validation if exposed beyond localhost).

P1

- Modularize `maverick_mcp/api/server.py` (bootstrap vs resources/tools vs prompts vs shutdown).
- Modularize `maverick_mcp/agents/deep_research.py` (providers, policies, graph wiring, output formatting).
- Consolidate provider strategy: either (a) finish migrating to interfaces/adapters or (b) keep concrete providers and delete adapter layer. Current hybrid increases cost.
- Adopt FastMCP background tasks for long-running tools (deep research, parallel screening, backtests) to improve responsiveness and reduce timeouts.

P2

- Consolidate tests location / test suite tiers (fast vs slow), if not already formalized.
- Clean up unfinished registry/init TODOs: `maverick_mcp/langchain_tools/registry.py`.

## MCP/FastMCP Alignment Notes

See `mcp-fastmcp-research.md` for up-to-date research findings and the recommended transport/tool patterns to adopt.

## Task 3.2 — Follow-on Implementation Track Plan (Proposed)

Create a dedicated implementation track (separate from this recommendations-only track) that executes the modernization safely with TDD. Suggested shape:

### Phase A — Entrypoint + Transport consolidation (P0)

- Update docs/Make targets so the **golden path is unambiguous**.
- Ensure STDIO mode never writes non-protocol output to stdout (stderr-only logs).
- Ensure Streamable HTTP endpoint and path are documented and stable (e.g., `/mcp`).
- Add transport-focused tests/smoke checks:
  - tool list remains stable across reconnects
  - no redirects on HTTP paths used by clients

### Phase B — Startup side effects and configuration cleanup (P0)

- Move environment loading/logging configuration to a single bootstrap path.
- Add regression tests ensuring importing library modules has no global side effects (no `basicConfig`, no implicit dotenv loads).

### Phase C — Legacy quarantine + router canonicalization (P0/P1)

- Declare canonical routers + remove unused variants (or move to a `legacy/` namespace).
- Migrate any remaining “used-by-default” code paths to the canonical set.
- Add tests for tool name stability and for canonical router registration.

### Phase D — Long-running tools as background tasks (P1)

- Convert “deep research” and “heavy backtests” to FastMCP background tasks where it improves UX.
- Add tests around task lifecycle behavior (start/poll/cancel) and timeout expectations.

## Task 3.3 — Deprecation Plan (Alternate Paths)

Deprecation should be explicit and staged to avoid breaking existing users.

### Scope to deprecate

- Alternate server entrypoints/modes:
  - `maverick_mcp/api/api_server.py`
  - `maverick_mcp/api/simple_sse.py`, `maverick_mcp/api/inspector_sse.py`, `maverick_mcp/api/inspector_compatible_sse.py`
- Router variants not used by canonical tool registration:
  - `maverick_mcp/api/routers/screening_ddd.py`, `maverick_mcp/api/routers/screening_parallel.py`, `maverick_mcp/api/routers/technical_ddd.py`
  - Evaluate `data_enhanced.py` / `health_enhanced.py` and either (a) delete, (b) merge improvements, or (c) move to legacy.

### Staging

1. **Document** (no behavior change): label deprecated modules in docs and point to the golden path.
2. **Warn**: add runtime warnings when legacy entrypoints are invoked (where appropriate).
3. **Quarantine**: move deprecated modules under a `legacy/` namespace/package, keeping imports temporarily compatible via thin re-exports.
4. **Remove**: delete legacy modules once usage is near-zero and tests cover the golden path.
