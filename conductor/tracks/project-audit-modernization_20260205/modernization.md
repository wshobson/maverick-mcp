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
