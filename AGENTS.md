# Repository Guidelines

## Project Overview

MaverickMCP is a personal-use FastMCP server for local financial analysis in
Claude Desktop and other MCP clients. It provides stock data, technical
analysis, screening, portfolio tracking, backtesting, research, watchlists,
a trade journal, and a risk dashboard.

This project is for educational and informational use only. It is not financial
advice, tax advice, or a trading system.

As of v1.0.0, the entire server lives in `maverick/`. The legacy
`maverick_mcp/` package was deleted at the v1.0 cutover. If you are carrying
config or a database forward from a pre-v1.0 install, read
`docs/runbooks/migrating-to-v1.md`.

## Project Structure

- `maverick/platform/`: the shared seam -- database, cache, HTTP resilience,
  telemetry, and the BYOK LLM factory. The only place that reads env vars.
- `maverick/market_data/`, `technical/`, `screening/`, `portfolio/`: core
  domains, each `types.py` -> `config.py` -> `data.py` -> `service.py` ->
  `tools.py`. Import contracts and structural tests enforce that layering.
- `maverick/backtesting/`, `research/`: optional-extra domains
  (`[backtesting]`, `[research]`); each degrades to zero registered tools
  with one warning when its extra is absent.
- `maverick/server/`: FastMCP assembly (`assembly.py`), the CLI entry point
  (`app.py`), and prompts (`prompts.py`). Nothing imports `maverick.server`.
- `tests/`: primary pytest suite, mirroring the domain tree plus
  `tests/structure` (layering/naming checks) and `tests/server`.
- `scripts/`: local utility scripts (currently just indicator fixtures).
- `conductor/`: historical/tool-owned Conductor planning context.
- `docs/`: canonical project documentation and catalog.

## Documentation Map

Keep this file concise. Treat it as the table of contents for future agents.
Durable detail belongs in `docs/`.

- `docs/INDEX.md`: start here for the documentation structure.
- `docs/CATALOG.md`: status of current, historical, archived, and deleted docs.
- `docs/ARCHITECTURE.md`: package layout, service boundaries, and data flow.
- `docs/runbooks/claude-desktop.md`: Claude Desktop and MCP transport setup.
- `docs/runbooks/database-setup.md`: SQLite/PostgreSQL setup.
- `docs/runbooks/migrating-to-v1.md`: config/database migration from pre-v1.0.
- `docs/features/portfolio.md`: portfolio persistence and cost-basis behavior.
- `docs/features/deep-research.md`: research agent and provider behavior.
- `docs/api/backtesting.md`: backtesting API reference.
- `docs/testing/README.md`: canonical testing guide.
- `docs/references/llm-documentation-hygiene.md`: documentation hygiene rules.
- `docs/exec-plans/tech-debt-tracker.md`: known debt, one line each. Add a
  line when you find debt; remove the line when you remove the debt.

## Build, Test, And Development Commands

```bash
uv sync --extra dev                        # core + dev tooling
uv sync --extra dev --extra backtesting --extra research  # full tool surface

make dev          # Streamable HTTP server on port 8003
make dev-stdio    # STDIO transport for Claude Desktop
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

- Python 3.12+.
- Use Ruff formatting and linting; line length is 88.
- Keep exports typed and avoid broad rewrites.
- Prefer existing domain layering (`types -> config -> data -> service ->
  tools`) over new abstractions; run `uv run lint-imports` after touching
  imports.
- Use `Decimal` for financial arithmetic; do not introduce float-based cost
  basis or P&L calculations.

## MCP Transport Defaults

- Claude Desktop: prefer direct STDIO via `make dev-stdio` or the `uv run`
  command in `docs/runbooks/claude-desktop.md`.
- Streamable HTTP: default local server transport for bridge/remote workflows
  at `http://localhost:8003/mcp/`.
- SSE does not exist in this server; do not add it back without an explicit
  design decision.

## Testing Guidelines

- Add or update focused tests when behavior changes.
- Avoid real network calls in unit tests.
- Mark external-provider tests with `external`; require explicit API keys.
- Use the docs catalog checker when moving, deleting, or adding Markdown/text
  files.

## Safety And Configuration

- Do not commit `.env`, API keys, database dumps, cache artifacts, or generated
  secrets.
- No API key is required for core tools (market data, technical analysis,
  screening, portfolio): `yfinance` is the default data source.
- Optional keys: `EXA_API_KEY` (research web search) and `LLM_PROVIDER` +
  `LLM_API_KEY` + `LLM_MODEL` (BYOK LLM for research and
  `backtesting_parse_strategy`). See `.env.example` for the full list.
- Keep authentication/billing complexity out of the local personal-use path
  unless a future plan explicitly changes that scope.
