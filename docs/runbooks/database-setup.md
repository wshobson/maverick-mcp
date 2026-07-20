# Database Setup

MaverickMCP defaults to local SQLite and also supports PostgreSQL for larger
local datasets. Database setup is local personal-use infrastructure, not
hosted SaaS setup.

## Quick Setup

There is no separate setup script. Every domain that owns tables calls
`maverick.platform.db.ensure_schema` the first time its service is used,
which creates missing tables idempotently (`CREATE TABLE IF NOT EXISTS`
semantics via SQLAlchemy `create_all`). Just start the server:

```bash
uv sync --extra dev
cp .env.example .env
make dev-stdio   # or: make dev
```

The database file (or configured PostgreSQL database) is created and
populated with schema on first tool call. There is no migration framework;
schema changes are additive, and there is nothing to run manually.

## Default SQLite

```bash
export DATABASE_URL=sqlite:///maverick.db
make dev-stdio
```

SQLite is the lowest-friction path and is enough for normal local MCP use.

## PostgreSQL

```bash
createdb maverick
export DATABASE_URL=postgresql://localhost/maverick
make dev-stdio
```

Use PostgreSQL when loading larger local datasets or when you want database
behavior closer to a production-style deployment. `ensure_schema` creates
the same tables on Postgres as it does on SQLite; no separate migration step
exists for either backend.

## No Bulk Data Seeding

Earlier versions of this project shipped a Tiingo-backed bulk loader that
pre-seeded a fixed S&P 500 universe. That loader and its scripts were
removed at the v1.0.0 cutover (see
[`migrating-to-v1.md`](migrating-to-v1.md)). The current server has no
pre-seeded universe:

- Market data (quotes, price history, fundamentals) comes from `yfinance` on
  demand, with no API key required. Calling `market_data_get_price_history`
  or `market_data_get_quote` for a ticker registers that symbol in the local
  `md_stocks` table as a side effect.
- The screening domain (`screening_run_screens`) computes its Maverick
  bullish/bearish/supply-demand screens over whatever symbols are already
  known locally (the same `md_stocks` table). Fetch price history for the
  tickers you care about before running a screen for meaningful coverage;
  there is no S&P 500-wide default universe on a fresh install.

## Claude Desktop After Setup

Prefer STDIO for Claude Desktop. See `claude-desktop.md`.

For HTTP bridge testing:

```bash
make dev
```

## Troubleshooting

- Missing/empty database file: it is created on first tool call; if it looks
  wrong, delete it and let `ensure_schema` recreate it on the next call
  (SQLite only -- do not do this against a PostgreSQL database with data you
  want to keep).
- Empty screening results: fetch price history for a few tickers via
  `market_data_get_price_history`/`market_data_get_price_history_batch`
  first, then call `screening_run_screens`.
- Redis unavailable: leave Redis variables unset; the app falls back to
  in-memory/SQLite caching.
- Carrying data forward from a pre-v1.0 install: see
  [`migrating-to-v1.md`](migrating-to-v1.md) for the inert-legacy-tables
  note.
