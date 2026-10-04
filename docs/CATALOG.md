# Documentation Catalog

Status labels:

- `current`: source of truth for active behavior.
- `historical`: useful context, but not the active source of truth.
- `archived`: retained only for reference.
- `deleted`: intentionally removed because it was stale, fabricated, redundant,
  or superseded.

## Current

| Path | Status | Owner | Notes |
| --- | --- | --- | --- |
| `../AGENTS.md` | current | agents | Canonical agent entry point. |
| `../CLAUDE.md` | current | agents | Symlink to `../AGENTS.md`. Not a separate source of truth. |
| `../README.md` | current | project | User-facing overview and quick start. |
| `../CONTRIBUTING.md` | current | project | Contributor workflow. |
| `../SECURITY.md` | current | project | Vulnerability reporting and security guidance. |
| `../CODE_OF_CONDUCT.md` | current | project | Community standards. |
| `INDEX.md` | current | docs | Documentation entry point. |
| `CATALOG.md` | current | docs | Inventory and cleanup state. |
| `../ARCHITECTURE.md` | current | engineering | Package layout and system boundaries. Root-level entry point alongside `AGENTS.md`. |
| `../evals/tool_surface/README.md` | current | engineering | Tool-surface traces: an in-session Claude Code subagent runs fixed cases on the Maverick tools for error analysis, no scoring. |
| `../evals/tool_surface/agent/maverick-eval-client.md` | current | engineering | Claude Code subagent definition for in-session eval traces; `make eval-agent-install` copies it into `.claude/agents/`. |
| `../evals/tool_surface/failure_modes.md` | current | engineering | Draft failure-mode taxonomy from the reviewer's notes on the 2026-09-26 trace runs: where each fix belongs and which evaluator each mode gets. |
| `../evals/tool_surface/judges/acts-on-a-guess.md` | current | engineering | Judge prompt for the "acts on a guess instead of asking" failure mode, run by in-session subagents and scored by `evals/tool_surface/judge.py`. |
| `design-docs/2026-07-18-mcp-modernization.md` | current | engineering | Approved v1.0 modernization design and migration plan. |
| `design-docs/2026-09-05-open-items-remediation.md` | current | engineering | Approved design for the 2026-09 open-items remediation, FastMCP 4 migration, and SearXNG research backend. |
| `design-docs/2026-09-13-pandas-ta-removal.md` | current | engineering | Approved design for removing pandas-ta and moving numpy, numba, pandas, and vectorbt. |
| `exec-plans/active/2026-07-20-phase-9-distribution.md` | current | engineering | Phase 9 execution plan (distribution and registry rollout). |
| `exec-plans/active/2026-09-05-open-items-remediation.md` | current | engineering | Execution plan for the 2026-09 open-items remediation (25 tasks: triage, contributor fixes, deps, FastMCP 4, SearXNG, v1.1.0). |
| `design-docs/2026-10-04-project-review.md` | current | engineering | Review evidence, maintenance outcomes, financial/runtime findings, and distribution gates. |
| `exec-plans/active/2026-10-04-correctness-and-reliability.md` | current | engineering | Proposed correctness and reliability implementation plan; no roadmap fixes claimed complete. |
| `exec-plans/tech-debt-tracker.md` | current | engineering | Known debt, one line each. |
| `product-specs/index.md` | current | product | Product spec index; no specs written yet. |
| `generated/README.md` | current | docs | Describes `generated/`: hand-written registry drafts and release notes. |
| `generated/registry/README.md` | current | docs | Index of registry submission drafts (Phase 9 Task 3). |
| `generated/registry/docker-mcp-catalog.md` | current | docs | Docker MCP Catalog PR draft (server.yaml + PR body). |
| `generated/registry/glama.md` | current | docs | Glama submission draft. |
| `generated/registry/pulsemcp.md` | current | docs | PulseMCP submission draft (submission mechanism unconfirmed). |
| `generated/registry/mcp-so.md` | current | docs | mcp.so submission draft. |
| `generated/registry/smithery.yaml` | current | docs | Smithery config draft (YAML, not scanned by the docs-catalog checker; listed here for completeness). |
| `generated/release-notes/v1.1.0.md` | current | engineering | Release notes for v1.1.0 (used by `gh release create --notes-file`). |
| `QUALITY_SCORE.md` | current | engineering | Per-area quality grades. |
| `RELIABILITY.md` | current | engineering | Reliability state and gaps. |
| `SECURITY.md` | current | engineering | Engineering security posture. |
| `api/backtesting.md` | current | engineering | Backtesting API reference. |
| `features/portfolio.md` | current | product/engineering | Portfolio persistence and cost-basis behavior. |
| `features/deep-research.md` | current | engineering | Research agent behavior and configuration. |
| `runbooks/mcp-clients.md` | current | operations | Transports and per-client MCP setup (Claude Desktop, Claude Code, VS Code, GitHub Copilot CLI, Codex CLI, Cursor, OpenCode, Antigravity CLI). Config verified against vendor docs and live CLIs 2026-09-26. |
| `runbooks/database-setup.md` | current | operations | Database setup and schema creation (no migrations). |
| `runbooks/self-contained-setup.md` | current | operations | Full local setup. |
| `runbooks/migrating-to-v1.md` | current | operations | Config/database migration guide from pre-v1.0 installs. |
| `runbooks/releasing.md` | current | operations | Owner-run publish sequence: PyPI, official MCP Registry, GHCR, third-party registries, `.mcpb` release asset. |
| `testing/README.md` | current | engineering | Canonical test commands and marker policy. |
| `testing/in-memory.md` | current | engineering | FastMCP in-memory test patterns. |
| `testing/exa-research.md` | current | engineering | Exa/research provider test strategy. |
| `references/llm-documentation-hygiene.md` | current | docs | Agent-legible documentation rules. |

## Historical

| Path | Status | Notes |
| --- | --- | --- |
| `superpowers/` | historical | Historical specs and plans. Current designs live under `design-docs/` and execution plans under `exec-plans/active/`. |
| `exec-plans/completed/2026-07-18-phase-0-harness-and-cleanup.md` | historical | Phase 0 execution plan (harness scaffold and cleanup). |
| `exec-plans/completed/2026-07-18-phase-1-platform-seam.md` | historical | Phase 1 execution plan (platform seam). |
| `exec-plans/completed/2026-07-19-phase-2-market-data-domain.md` | historical | Phase 2 execution plan (market data domain). |
| `exec-plans/completed/2026-07-19-phase-3-screening-domain.md` | historical | Phase 3 execution plan (screening domain and technical core). |
| `exec-plans/completed/2026-07-19-phase-4-portfolio-domain.md` | historical | Phase 4 execution plan (portfolio domain). |
| `exec-plans/completed/2026-07-19-phase-5-technical-domain.md` | historical | Phase 5 execution plan (technical domain completion). |
| `exec-plans/completed/2026-07-19-phase-6-backtesting-extra.md` | historical | Phase 6 execution plan (backtesting extra). |
| `exec-plans/completed/2026-07-20-phase-7-research-extra.md` | historical | Phase 7 execution plan (research extra). |
| `exec-plans/completed/2026-07-20-phase-8-server-cutover.md` | historical | Phase 8 execution plan (server assembly and cutover). |
| `exec-plans/completed/2026-09-13-pandas-ta-removal.md` | historical | Execution plan for the pandas-ta removal (4 tasks: feature engineering, dependency drop, pandas 3, bookkeeping). |
| `exec-plans/completed/2026-09-26-tech-debt-sweep.md` | historical | Completed plan for the 2026-09-26 tech-debt sweep (unused dependencies, tracker items, Claude workflows on Opus 5.5; PRs #283 to #288). |

## Deleted Or Consolidated

| Old path | Status | Replacement |
| --- | --- | --- |
| `../PLANS.md` | deleted | Removed as unrelated Rust parser placeholder content. |
| `../DATABASE_SETUP.md` | deleted | `runbooks/database-setup.md` |
| `BACKTESTING.md` | deleted | `api/backtesting.md` |
| `COST_BASIS_SPECIFICATION.md` | deleted | `features/portfolio.md` |
| `PORTFOLIO.md` | deleted | `features/portfolio.md` |
| `PORTFOLIO_PERSONALIZATION_PLAN.md` | deleted | `features/portfolio.md` |
| `SETUP_SELF_CONTAINED.md` | deleted | `runbooks/self-contained-setup.md` |
| `deep_research_agent.md` | deleted | `features/deep-research.md` |
| `exa_research_testing_strategy.md` | deleted | `testing/exa-research.md` |
| `speed_testing_framework.md` | deleted | `testing/README.md` |
| `../scripts/INSTALLATION_GUIDE.md` | deleted | `runbooks/migrating-to-v1.md` |
| `../scripts/README_TIINGO_LOADER.md` | deleted | `runbooks/migrating-to-v1.md` |
| `../tests/README.md` | deleted | `testing/README.md` |
| `../tests/integration/README.md` | deleted | `testing/README.md` |
| `../maverick_mcp/tests/README_INMEMORY_TESTS.md` | deleted | `testing/in-memory.md` |
| `../maverick_mcp/README.md` | deleted | `../ARCHITECTURE.md` |
| `runbooks/tiingo-loader.md` | deleted | `runbooks/migrating-to-v1.md`; the Tiingo bulk data loader and its scripts were removed at the v1.0.0 cutover (`maverick_mcp` deletion). Market data now comes from `yfinance` with no API key required. |
| `runbooks/claude-desktop.md` | deleted | `runbooks/mcp-clients.md`; renamed and broadened. The server is client-agnostic, so the runbook is organized by transport with per-client config rather than around Claude Desktop. |
| `../conductor/` | deleted | Conductor is no longer used on this project. The `conductor/` planning scaffold was removed; Git history is the archive. |
| `ARCHITECTURE.md` | deleted | `../ARCHITECTURE.md`; moved to the repository root so the architecture map sits beside `AGENTS.md` as a top-level entry point. |
| `../GEMINI.md` | deleted | `../AGENTS.md`; it was a pure pointer with no unique content. |
| `../CLAUDE.md` (regular file) | deleted | Replaced by a symlink to `../AGENTS.md`. Its unique rules were folded into `AGENTS.md` so there is one agent entry point. |
| `testing/integration.md` | deleted | `testing/README.md`; no integration-marked tests or `tests/integration/` directory exist, so its remaining facts moved into the testing guide. |
| `testing/speed.md` | deleted | `testing/README.md`; the speed benchmark targets and script it documented no longer exist, so its remaining timeout facts moved into the testing guide. |

## Allowlisted Non-Documentation Text

No entries. `scripts/requirements_tiingo.txt` (the Tiingo loader's pinned
dependency file) was removed at the v1.0.0 cutover along with the loader
scripts it supported.
