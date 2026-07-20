# Documentation Index

This directory is the repository knowledge base. Keep root agent files short and
use this index as the map to deeper, versioned sources of truth.

## Start Here

- `../AGENTS.md` - agent entry point: structure, commands, conventions, and
  safety notes.
- `CATALOG.md` - documentation inventory with current, historical, archived, and
  deleted status.
- `ARCHITECTURE.md` - package layout, service boundaries, and data flow.
- `runbooks/claude-desktop.md` - Claude Desktop and MCP transport setup.
- `testing/README.md` - test commands, markers, and focused-suite guidance.

## Current Product And Technical Docs

- `api/backtesting.md` - Backtesting MCP tools and examples.
- `features/portfolio.md` - Portfolio persistence, cost basis, P&L, and
  position-aware analysis behavior.
- `features/deep-research.md` - Research agent capabilities, providers, and
  configuration.
- `runbooks/database-setup.md` - SQLite/PostgreSQL setup and schema creation.
- `runbooks/self-contained-setup.md` - full local setup with market data.
- `runbooks/migrating-to-v1.md` - config/database migration from pre-v1.0
  installs.
- `runbooks/releasing.md` - the full publish sequence (PyPI, official MCP
  Registry, GHCR, third-party registries, `.mcpb` release asset); owner-run.
- `generated/registry/README.md` - ready-to-paste registry submission
  drafts (Docker MCP Catalog, Smithery, Glama, PulseMCP, mcp.so).

## Modernization

- `design-docs/2026-07-18-mcp-modernization.md` - approved v1.0 modernization
  design and migration plan.
- `exec-plans/completed/2026-07-18-phase-0-harness-and-cleanup.md` - Phase 0
  execution plan.
- `exec-plans/completed/2026-07-18-phase-1-platform-seam.md` - Phase 1
  execution plan (platform seam).
- `exec-plans/completed/2026-07-19-phase-2-market-data-domain.md` - Phase 2
  execution plan (market data domain).
- `exec-plans/completed/2026-07-19-phase-3-screening-domain.md` - Phase 3
  execution plan (screening domain and technical core).
- `exec-plans/completed/2026-07-19-phase-4-portfolio-domain.md` - Phase 4
  execution plan (portfolio domain).
- `exec-plans/completed/2026-07-19-phase-5-technical-domain.md` - Phase 5
  execution plan (technical domain completion).
- `exec-plans/completed/2026-07-19-phase-6-backtesting-extra.md` - Phase 6
  execution plan (backtesting extra).
- `exec-plans/completed/2026-07-20-phase-7-research-extra.md` - Phase 7
  execution plan (research extra).
- `exec-plans/completed/2026-07-20-phase-8-server-cutover.md` - Phase 8
  execution plan (server assembly and cutover).
- `exec-plans/active/2026-07-20-phase-9-distribution.md` - Phase 9
  execution plan (distribution and registry rollout).
- `exec-plans/tech-debt-tracker.md` - known debt, one line each.
- `product-specs/index.md` - product spec index, empty until the tool surface
  is curated.
- `generated/README.md` - marker for script-generated docs.
- `QUALITY_SCORE.md` - per-area quality grades.
- `RELIABILITY.md` - reliability state and gaps.
- `SECURITY.md` - engineering security posture.

## Testing Docs

- `testing/README.md` - canonical test guide.
- `testing/in-memory.md` - FastMCP in-memory testing patterns.
- `testing/integration.md` - integration and orchestration test notes.
- `testing/exa-research.md` - Exa/research provider test strategy.
- `testing/speed.md` - research speed and timeout validation.

## Historical Or Tool-Owned Context

- `../conductor/` - historical Conductor planning and workflow context.
- `superpowers/` - historical Superpowers specs and plans.

These folders are cataloged but are not the current product documentation
unless a current doc links to a specific artifact.

## Hygiene Rules

- Do not let root files become the project encyclopedia.
- When behavior changes, update the nearest source-of-truth doc in the same
  change.
- Prefer small linked documents over a single long instruction file.
- Delete stale docs after preserving current facts; Git history is the archive.
- If a rule must not drift, encode it in tests, scripts, or CI.
- Run `make docs-check` after adding, moving, or deleting Markdown/text docs.
