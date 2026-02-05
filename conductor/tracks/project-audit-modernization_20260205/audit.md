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

## Hotspots / Refactor Candidates

- `maverick_mcp/api/server.py`: very large, mixes concerns (bootstrapping, transport quirks, monitoring init, prompts, tools, resources, shutdown). Candidate for splitting into `server/bootstrap.py`, `server/transports.py`, `server/resources.py`, etc.
- `maverick_mcp/agents/deep_research.py`: very large; likely needs modularization (providers, prompt/policy, graph wiring, result shaping) to keep changes safe.
- `maverick_mcp/data/models.py`: large ORM surface area; consider splitting by bounded context (portfolio vs screening vs cache tables).
- `maverick_mcp/config/settings.py`: large config surface with mixed sources; likely needs consolidation to a single settings approach (and removal of module-level logging config).
- Tool registration strategy: `maverick_mcp/api/routers/tool_registry.py` + “register router tools directly” pattern vs any router mounting patterns; decide a single stable approach and retire the other.

## Test Coverage Notes

- There are **two test locations** (`tests/` and `maverick_mcp/tests/`). If both are intentional, document the distinction; otherwise, consolidate to one to reduce duplication and import-path confusion.
- Many tests appear “large” (multi-hundred/1k+ LOC integration/perf tests). That’s valuable, but also suggests a need for a small “fast suite” vs “slow suite” convention if not already present (e.g., markers, Make targets).
