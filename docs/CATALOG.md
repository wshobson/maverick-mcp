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
| `../CLAUDE.md` | current | agents | Claude-specific pointer to canonical docs. |
| `../GEMINI.md` | current | agents | Gemini-specific pointer to canonical docs. |
| `../README.md` | current | project | User-facing overview and quick start. |
| `../CONTRIBUTING.md` | current | project | Contributor workflow. |
| `../SECURITY.md` | current | project | Vulnerability reporting and security guidance. |
| `../CODE_OF_CONDUCT.md` | current | project | Community standards. |
| `INDEX.md` | current | docs | Documentation entry point. |
| `CATALOG.md` | current | docs | Inventory and cleanup state. |
| `ARCHITECTURE.md` | current | engineering | Package layout and system boundaries. |
| `design-docs/2026-07-18-mcp-modernization.md` | current | engineering | Approved v1.0 modernization design and migration plan. |
| `api/backtesting.md` | current | engineering | Backtesting API reference. |
| `features/portfolio.md` | current | product/engineering | Portfolio persistence and cost-basis behavior. |
| `features/deep-research.md` | current | engineering | Research agent behavior and configuration. |
| `runbooks/claude-desktop.md` | current | operations | MCP client setup and transport guidance. |
| `runbooks/database-setup.md` | current | operations | Database setup, migrations, and seeding. |
| `runbooks/self-contained-setup.md` | current | operations | Full local setup. |
| `runbooks/tiingo-loader.md` | current | operations | Tiingo loader setup and usage. |
| `testing/README.md` | current | engineering | Canonical test commands and marker policy. |
| `testing/in-memory.md` | current | engineering | FastMCP in-memory test patterns. |
| `testing/integration.md` | current | engineering | Integration test guidance. |
| `testing/exa-research.md` | current | engineering | Exa/research provider test strategy. |
| `testing/speed.md` | current | engineering | Research speed validation. |
| `references/llm-documentation-hygiene.md` | current | docs | Agent-legible documentation rules. |

## Historical

| Path | Status | Notes |
| --- | --- | --- |
| `../conductor/` | historical | Tool-owned planning context. Keep cataloged, but do not treat as the current product docs. |
| `superpowers/` | historical | Historical specs and plans. Current plans should live under `docs/plans/` or a new approved plan location. |

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
| `speed_testing_framework.md` | deleted | `testing/speed.md` |
| `../scripts/INSTALLATION_GUIDE.md` | deleted | `runbooks/tiingo-loader.md` |
| `../scripts/README_TIINGO_LOADER.md` | deleted | `runbooks/tiingo-loader.md` |
| `../tests/README.md` | deleted | `testing/README.md` |
| `../tests/integration/README.md` | deleted | `testing/integration.md` |
| `../maverick_mcp/tests/README_INMEMORY_TESTS.md` | deleted | `testing/in-memory.md` |
| `../maverick_mcp/README.md` | deleted | `ARCHITECTURE.md` |

## Allowlisted Non-Documentation Text

| Path | Status | Notes |
| --- | --- | --- |
| `../scripts/requirements_tiingo.txt` | current | Dependency input for the Tiingo loader, not prose documentation. |
