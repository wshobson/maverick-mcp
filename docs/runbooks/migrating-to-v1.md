# Migrating To v1.0

v1.0.0 replaced the legacy `maverick_mcp/` package with `maverick/`: a
smaller, curated tool surface behind a simplified configuration seam. The
legacy package was deleted entirely at cutover (Phase 8 of the
modernization). This runbook covers what changes for anyone carrying a
pre-v1.0 `.env` file or database forward.

## Package And Install

- PyPI package renamed to `maverick-mcp-server` (console script:
  `maverick-mcp`, entry point `maverick.server.app:main`).
- New install: `pip install "maverick-mcp-server[backtesting,research]"` or
  `uvx --from maverick-mcp-server maverick-mcp --transport stdio`.
- Run invocation changed from `python -m maverick_mcp.api.server` to
  `python -m maverick.server` (same `--transport stdio|http` flags).
- SSE transport is gone. Only `stdio` and streamable `http` exist now.

## Environment Variable Mapping

The legacy server auto-detected an LLM vendor from whichever of several
per-vendor keys was set (`OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, ...). v1.0 replaces that with one explicit BYOK seam:

| Legacy | v1.0 replacement |
| --- | --- |
| `OPENROUTER_API_KEY` (or any other vendor key), auto-detected | `LLM_PROVIDER=openrouter` (or `anthropic`/`openai`/`openai_compatible`) plus `LLM_API_KEY=<key>` |
| Vendor-specific model env vars | `LLM_MODEL=<model name>` |
| N/A (no override existed) | `LLM_BASE_URL` -- required only for `LLM_PROVIDER=openai_compatible`, optional override otherwise |
| N/A | `LLM_TEMPERATURE` (default `0.0`) |

Set all three of `LLM_PROVIDER`/`LLM_API_KEY`/`LLM_MODEL` together, or leave
all unset to run without LLM-backed tools (`research_*` and
`backtesting_parse_strategy` register but return a clear "not configured"
error; every other tool is unaffected).

`EXA_API_KEY` is unchanged -- it still gates web search for the research
domain and is independent of the LLM seam above.

## Dead Environment Variables

Remove these from your `.env`; nothing in `maverick/` reads them:

- `TIINGO_API_KEY`, `TIINGO_API_TOKEN` -- the Tiingo bulk data loader and its
  scripts were deleted at cutover. Market data now comes from `yfinance`
  with no API key required. See [`database-setup.md`](database-setup.md).
- `ADANOS_API_KEY` and any `ADANOS_*` variant -- Adanos sentiment retired;
  `research_analyze_sentiment` is the sentiment surface now.
- `FRED_API_KEY` -- the macro/FRED data provider was not ported (zero live
  consumers in the legacy code); there is currently no macro domain.
- `TAVILY_API_KEY` -- `TavilySearchProvider` was legacy dead code (never
  actually instantiated); Exa is the only search provider.
- Auth/billing remnants: `AUTH_ENABLED`, `API_KEY_*`, `JWT_*`, session/OAuth
  variables, `SENTRY_DSN`, `ALLOWED_ORIGINS`, `RATE_LIMIT_PER_IP`,
  `MAINTENANCE_MODE`. The server has no authentication, billing, CORS
  handling, or hosted-SaaS surface by design, and none of those variables
  are read anywhere in `maverick/`.
- Generic app/API scaffolding vars from the old `.env.example`:
  `APP_NAME`, `ENVIRONMENT`, `API_VERSION`, `API_HOST`, `API_PORT`,
  `API_DEBUG`. There is no REST API layer in the new server.

See `.env.example` for the current, complete, code-verified variable list.

## Database: No More Alembic

Alembic and its migration scripts were deleted at cutover. Every domain that
owns tables now creates its schema idempotently via
`maverick.platform.db.ensure_schema` (SQLAlchemy `create_all`) the first
time its service runs. There is nothing to run manually, and no migration
history to reconcile.

### Inert Legacy Tables

If you point `DATABASE_URL`/`POSTGRES_URL` at a database from a pre-v1.0
install, the old tables are still there -- the new server does not drop
anything. Two categories:

- **Reused verbatim, no action needed**: watchlists (`watchlists`,
  `watchlist_items`) and the trade journal (`journal_entries`,
  `strategy_performance`) kept their exact legacy table and column names on
  purpose, so existing rows carry forward automatically with zero migration
  step.
- **Inert, safe to drop manually**: everything else from the legacy schema
  -- for example `mcp_portfolios`/`mcp_portfolio_positions` (portfolio
  positions moved to new `pf_portfolios`/`pf_positions` tables with a fresh
  average-cost-basis ledger), the legacy `mcp_maverick_*` screening tables
  (replaced by `scr_results`), `alembic_version`, and any signal/alert or
  health-monitoring tables. The new server never reads or writes these; they
  just take up disk space. Drop them by hand if you want a clean database,
  or leave them -- they cause no errors.

If you were relying on legacy portfolio positions (`mcp_portfolios`), there
is no automatic migration into the new `pf_positions` ledger; re-add your
positions with `portfolio_add_position`.

## Tool Surface Changes

The legacy server exposed roughly 119 tools, 9 prompts, and 11 resources.
v1.0 ships a curated surface: 37 core tools, 12 `backtesting_*` tools
(`[backtesting]` extra), 3 `research_*` tools (`[research]` extra), 3
prompts, and 1 resource (`portfolio://my-holdings`). What ported and what
didn't:

**Ported** (renamed to the new `<domain>_<verb>` convention, e.g.
`fetch_stock_data` -> `market_data_get_price_history`):

- All core market data, technical analysis, and screening tools.
- Portfolio position tracking, plus the risk dashboard, watchlist, and
  trade journal (previously separate routers) into `portfolio_*`.
- Backtesting: rule-based and ML strategies, optimization, walk-forward
  analysis, Monte Carlo simulation.
- Research: comprehensive research, company research, and sentiment
  analysis collapsed from 9 legacy `research_*`/`agents_*` tools into 3.

**Retired, with no replacement** (see the Phase 8 exec plan's decision log
for the full rationale on each):

- The signal engine and alerting tools (8 tools + `signals://recent`) --
  persistent alerting needs a long-running daemon, which this server is
  not.
- Health/system monitoring tools and resources (8 tools + 3 resources) --
  there is no HTTP `/health` endpoint or ops dashboard surface; "did it
  register tools" is the health check for a personal server.
- Introspection tools (`discover_capabilities`, `list_all_strategies`,
  `get_strategy_help`) -- native MCP tool discovery plus
  `backtesting_list_strategies` cover this.
- `get_decision_log`/`get_tool_registry_status` -- exposed legacy
  rate-limiter/decision-logger internals that no longer exist.
- News sentiment (Tiingo-backed) and Adanos sentiment -- superseded by
  `research_analyze_sentiment`.
- `data_get_cached_price_data` (caching is a platform internal now) and
  `get_economic_calendar` (legacy already returned a hardcoded empty list).
- `get_upcoming_catalysts` -- read exclusively from a legacy table that had
  zero writers anywhere in the codebase; always empty in practice.
- The hardcoded 15-ticker `get_watchlist` demo tool -- superseded by the
  real, persistent `portfolio_watchlist_*` tools.
- The 9 legacy prompts and 3 legacy `stock://*` resources -- superseded by
  the 3 new curated prompts and the `portfolio://my-holdings` resource.

## Checklist

1. Reinstall: `pip install "maverick-mcp-server[backtesting,research]"` (or
   `uvx --from maverick-mcp-server maverick-mcp`).
2. Update your MCP client config to the new run command (see
   [`claude-desktop.md`](claude-desktop.md)).
3. Rewrite your `.env` against the current `.env.example`: drop dead vars,
   add `LLM_PROVIDER`/`LLM_API_KEY`/`LLM_MODEL` if you use research tools.
4. Point `DATABASE_URL` at your existing database if you want to keep
   watchlists, journal entries, and cached market data; re-add portfolio
   positions manually since those tables did not carry over automatically.
5. Start the server once (`make dev-stdio` or the equivalent client
   command) to let `ensure_schema` create the new tables.
