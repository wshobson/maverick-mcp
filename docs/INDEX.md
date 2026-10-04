# Documentation Index

This directory is the repository knowledge base. Keep root agent files short and
use this index as the map to deeper, versioned sources of truth.

Each entry says **when** to read it, so you can load only what the task needs
rather than the whole knowledge base up front.

## Start Here

- `../AGENTS.md` - agent entry point: structure, commands, conventions, and
  safety notes.
- `CATALOG.md` - documentation inventory with current, historical, archived, and
  deleted status.
- `../ARCHITECTURE.md` - package layout, service boundaries, and data flow.
  Read before changing domain layering or adding a package.
- `runbooks/mcp-clients.md` - transports and per-client MCP setup. Read when
  connecting any client, changing transports, or debugging tool registration.
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
- `generated/release-notes/v1.1.0.md` - release notes for v1.1.0.

- `references/readme-search-research.md` - DataForSEO evidence and README wording decisions.

## Designs, Plans, And Engineering State

- `design-docs/2026-10-04-project-review.md` - full review, merged maintenance
  queue, reproduced correctness findings, and distribution gates.
- `exec-plans/active/2026-10-04-correctness-and-reliability.md` - proposed
  prioritized implementation plan with regression checks for the review findings.

- `design-docs/2026-07-18-mcp-modernization.md` - approved v1.0 modernization
  design and migration plan.
- `design-docs/2026-09-05-open-items-remediation.md` - approved design for
  the 2026-09 open-items remediation, FastMCP 4 migration, and SearXNG
  research backend. Shipped in v1.1.0 except the owner-gated publishing.
- `design-docs/2026-09-13-pandas-ta-removal.md` - approved design for
  removing pandas-ta and moving numpy, numba, pandas, and vectorbt.
  Shipped in #271.
- `exec-plans/active/2026-07-20-phase-9-distribution.md` - Phase 9
  execution plan (distribution and registry rollout).
- `exec-plans/active/2026-09-05-open-items-remediation.md` - execution plan
  for the 2026-09 open-items remediation.
- `exec-plans/tech-debt-tracker.md` - known debt, one line each.
- `product-specs/index.md` - product spec index; no specs written yet.
- `generated/README.md` - what lives in `generated/` (hand-written registry
  drafts and release notes).
- `QUALITY_SCORE.md` - per-area quality grades.
- `RELIABILITY.md` - reliability state and gaps.
- `SECURITY.md` - engineering security posture.

## Testing Docs

- `testing/README.md` - canonical test guide, including integration/external
  test policy and timeouts.
- `testing/in-memory.md` - FastMCP in-memory testing patterns.
- `testing/exa-research.md` - Exa/research provider test strategy.
- `../evals/tool_surface/README.md` - tool-surface trace harness for error
  analysis. Read before recording or reviewing eval traces (`make eval-*`).

## Historical Or Tool-Owned Context

- `superpowers/` - historical Superpowers specs and plans.
- `exec-plans/completed/` - completed execution plans: Phases 0 to 8 of the
  v1.0 modernization, the pandas-ta removal, and the 2026-09-26 tech-debt
  sweep. Read when you need to know why a domain is shaped the way it is.

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
