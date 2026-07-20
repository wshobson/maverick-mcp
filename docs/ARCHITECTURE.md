# Architecture

MaverickMCP is a local FastMCP server. The entire system lives in the
`maverick/` package: a shared platform seam, six business domains, and a
`server/` package that assembles them. There is no other package; the legacy
`maverick_mcp/` tree was deleted at the v1.0.0 cutover.

## Runtime Shape

```text
MCP clients (Claude Desktop, Claude Code, mcp-remote bridges, ...)
  -> maverick.server (FastMCP assembly, stdio/http transports)
  -> per-domain tools -> services -> data/providers
  -> maverick.platform (db, cache, http, telemetry, llm)
```

## Package Layout

```text
maverick/
├── platform/          # cross-cutting seam; the only place that reads env vars
│   ├── config.py       #   PlatformSettings: database, redis, cache, http, telemetry
│   ├── db.py            #   SQLAlchemy engine/session helpers, ensure_schema
│   ├── cache.py         #   tiered cache: memory, then Redis or SQLite
│   ├── http.py          #   outbound HTTP client, circuit breaker, rate limiter
│   ├── telemetry.py      #   structured logging setup
│   ├── serde.py           #   DataFrame/dict (de)serialization for the cache
│   └── llm.py              #   BYOK chat-model factory (LLM_PROVIDER/...)
├── market_data/        # quotes, history, fundamentals, market overview
├── technical/           # RSI/MACD/support-resistance/full technical analysis
├── screening/            # Maverick bullish/bearish/supply-demand screens
├── portfolio/             # positions, risk dashboard, watchlist, trade journal
├── backtesting/            # [backtesting] extra: VectorBT engine + strategies
├── research/                # [research] extra: LangGraph research workflow
└── server/                   # FastMCP assembly, CLI entry point, prompts
```

## The Platform Seam

`maverick/platform/` is the only shared dependency every domain may import,
and the only place in the codebase that reads environment variables
(`os.getenv`, wrapped by `platform.config._clean_env` and friends). Domain
`config.py` modules read their own env vars through those same helpers, but
`maverick.platform.config.get_platform_settings()` owns process-wide
concerns: the database engine, Redis/cache settings, the outbound HTTP
client's retry/circuit-breaker/rate-limit knobs, and log setup.
`maverick.platform.llm` adds a second, narrower seam used only by
`research` and `backtesting_parse_strategy`: a BYOK (bring-your-own-key)
chat-model factory gated on `LLM_PROVIDER`/`LLM_API_KEY`/`LLM_MODEL`.

## The Six Domains

Each domain follows the same forward-only layer order: `types.py` ->
`config.py` -> `data.py` (when the domain owns tables) -> `service.py` ->
`tools.py`. A domain may import another domain only at the service layer
(e.g. `screening`, `portfolio`, and `technical` each inject the one shared
`MarketDataService` instance built at assembly time). Cross-cutting concerns
enter only through `platform/`.

- `market_data/`: quote/history/fundamentals/market-overview reads, backed
  by `yfinance` (no API key required) with an optional Capital Companion
  tier and a finviz fallback for market movers.
- `technical/`: RSI, MACD, support/resistance, and full technical analysis
  built on `market_data`'s price history.
- `screening/`: Maverick bullish, bearish, and supply/demand screens;
  computes and persists its own snapshot tables.
- `portfolio/`: position tracking with average-cost-basis, the risk
  dashboard, watchlists, and the trade journal. The largest domain; several
  sibling services (`PortfolioService`, `JournalService`) share one engine.
- `backtesting/` (`[backtesting]` extra): VectorBT-powered engine, 12
  rule-based strategy templates plus 8 ML strategy classes, optimization,
  walk-forward analysis, and Monte Carlo simulation.
- `research/` (`[research]` extra): a sequential LangGraph workflow --
  plan, search via Exa, validate/score sources, synthesize with a BYOK LLM.

Extras degrade gracefully: with `[backtesting]`/`[research]` not installed,
each domain's `tools.register()` logs one warning and registers zero tools
instead of raising, so a base install still boots and serves the other
domains normally.

## Server Assembly

`maverick/server/assembly.py::build_server()` is the single place that wires
everything together, in a fixed order (see its docstring for the full
rationale):

1. `platform.config.get_platform_settings()` resolves one shared `Engine`
   and one shared `Cache`.
2. `MarketDataService` is constructed first (its schema creation is eager;
   every other domain's is lazy via `ensure_schema`).
3. `ScreeningService`, `PortfolioService`, and `TechnicalService` each
   receive that one `MarketDataService` instance.
4. `JournalService` is portfolio's standalone sibling (its own engine, own
   tables), wired into `portfolio.tools.configure`'s optional
   `journal_service` parameter.
5. `BacktestingService`/`ResearchService` are constructed, and their
   heavy-dependency modules imported, only when their extras are installed.
6. Per domain: `configure(...)` then `register(mcp)`.

`maverick/server/app.py` is the CLI entry point (`python -m maverick.server`
/ the `maverick-mcp` console script): it parses `--transport`, builds the
server, and runs it. Nothing imports `maverick.server` -- it sits above
every domain in the import graph.

## MCP Surface

- **37 core tools**: `market_data_*` (7), `screening_*` (6),
  `portfolio_*` (20, including risk dashboard, watchlist, and journal),
  `technical_*` (4).
- **12 `backtesting_*` tools** (`[backtesting]` extra).
- **3 `research_*` tools** (`[research]` extra).
- **3 prompts**: `analyze_stock`, `review_portfolio`, and
  `run_backtest_workflow` (registered only with `[backtesting]`).
- **1 resource**: `portfolio://my-holdings`, a passive AI-context snapshot
  of the default portfolio.

Every tool declares `readOnlyHint: true` unless it mutates state (adding,
removing, or clearing positions/watchlists/journal entries). Text fetched
from third parties (news, search results) is untrusted input and is
returned to the client labeled as data, never blended into instructions.

## Data And Cache

- SQLite is the default local path (`DATABASE_URL` unset -> `sqlite:///maverick.db`).
- PostgreSQL is supported via `DATABASE_URL`/`POSTGRES_URL` for larger local
  datasets.
- Every domain that owns tables calls `platform.db.ensure_schema`, which
  creates missing tables idempotently on first use. There is no migration
  framework (Alembic was legacy-only and did not carry over); schema
  changes are additive `create_all` calls.
- Redis is optional (`REDIS_HOST` presence enables it); the cache falls back
  to an in-memory tier, then SQLite, when Redis is unavailable or disabled.

## MCP Transports

- STDIO is the default and the preferred Claude Desktop path
  (`maverick-mcp --transport stdio` / `python -m maverick.server`).
- Streamable HTTP at `http://localhost:8003/mcp/` is the `make dev`
  transport for bridge (`mcp-remote`) and remote workflows.
- SSE does not exist in the new server; it was deleted at the v1.0.0
  cutover along with its monkey-patches.

See `runbooks/claude-desktop.md` for concrete client configuration.

## Historical Context

The `conductor/` and `docs/superpowers/` folders, and everything under
`docs/exec-plans/` and `docs/design-docs/`, contain historical planning and
migration context from the v1.0 rebuild. They explain why the domains are
shaped the way they are, but the current source of truth starts at
`docs/INDEX.md` and this file.
