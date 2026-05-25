# Repository Guidelines

## Project Overview

MaverickMCP is a personal-use FastMCP server for local financial analysis in
Claude Desktop and other MCP clients. It provides stock data, technical
analysis, screening, portfolio tracking, backtesting, research, signals,
journaling, watchlists, and risk dashboards.

This project is for educational and informational use only. It is not financial
advice, tax advice, or a trading system.

## Project Structure

- `maverick_mcp/api/`: FastMCP server entrypoints and routers.
- `maverick_mcp/services/`: service-layer domains for signals, screening,
  journal, watchlist, and risk.
- `maverick_mcp/domain/`: domain entities and value objects.
- `maverick_mcp/data/`: SQLAlchemy models, database helpers, and data access.
- `maverick_mcp/providers/`: market, stock, macro, and optional data providers.
- `maverick_mcp/agents/`: research, supervisor, and orchestration agents.
- `maverick_mcp/backtesting/`: VectorBT backtesting engine and strategies.
- `tests/`: primary pytest suite.
- `scripts/`: local setup, migrations, data loading, and utility scripts.
- `conductor/`: historical/tool-owned Conductor planning context.
- `docs/`: canonical project documentation and catalog.

## Documentation Map

Keep this file concise. Treat it as the table of contents for future agents.
Durable detail belongs in `docs/`.

- `docs/INDEX.md`: start here for the documentation structure.
- `docs/CATALOG.md`: status of current, historical, archived, and deleted docs.
- `docs/ARCHITECTURE.md`: package layout, service boundaries, and data flow.
- `docs/runbooks/claude-desktop.md`: Claude Desktop and MCP transport setup.
- `docs/runbooks/database-setup.md`: SQLite/PostgreSQL setup and migrations.
- `docs/features/portfolio.md`: portfolio persistence and cost-basis behavior.
- `docs/features/deep-research.md`: research agent and provider behavior.
- `docs/api/backtesting.md`: backtesting API reference.
- `docs/testing/README.md`: canonical testing guide.
- `docs/references/llm-documentation-hygiene.md`: documentation hygiene rules.

## Build, Test, And Development Commands

```bash
uv sync --extra dev
make dev          # Streamable HTTP server on port 8003
make dev-stdio    # STDIO transport for Claude Desktop
make dev-sse      # Legacy/debug SSE transport
make stop

make test         # Unit tests only by default
make test-all     # Includes integration/slow/external markers
make lint
make typecheck
make check
make docs-check
```

Use `uv run pytest ...` for focused test runs. The default pytest config
excludes `integration`, `slow`, and `external` tests.

## Coding Style

- Python 3.12 only.
- Use Ruff formatting and linting; line length is 88.
- Keep exports typed and avoid broad rewrites.
- Prefer existing service/router/domain patterns over new abstractions.
- Use `Decimal` for financial arithmetic; do not introduce float-based cost
  basis or P&L calculations.

## MCP Transport Defaults

- Claude Desktop: prefer direct STDIO via `make dev-stdio` or the `uv run`
  command in `docs/runbooks/claude-desktop.md`.
- Streamable HTTP: default local server transport for bridge/remote workflows at
  `http://localhost:8003/mcp/`.
- SSE: legacy/debug only; do not document it as the preferred Claude Desktop
  path.

## Testing Guidelines

- Add or update focused tests when behavior changes.
- Avoid real network calls in unit tests.
- Mark external-provider tests with `external`; require explicit API keys.
- Use the docs catalog checker when moving, deleting, or adding Markdown/text
  files.

## Safety And Configuration

- Do not commit `.env`, API keys, database dumps, cache artifacts, or generated
  secrets.
- Required market data key for normal use: `TIINGO_API_KEY` or the Tiingo token
  variable used by the loader runbook.
- Optional research keys: `EXA_API_KEY`, `TAVILY_API_KEY`, and
  `OPENROUTER_API_KEY`.
- Keep authentication/billing complexity out of the local personal-use path
  unless a future plan explicitly changes that scope.
