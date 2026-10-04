# Database Setup

MaverickMCP defaults to local SQLite and also supports PostgreSQL for larger
local datasets. Database setup is local personal-use infrastructure, not
hosted SaaS setup.

## Quick Setup

There is no separate setup script. Every domain that owns tables calls
`maverick.platform.db.ensure_schema` the first time its service is used,
which creates missing tables idempotently (`CREATE TABLE IF NOT EXISTS`
semantics via SQLAlchemy `create_all`) and adds missing nullable columns to
existing tables. Just start the server:

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

## Docker data storage

Build the current source and mount a named volume at `/data`, as shown in the
[README Docker instructions](../../README.md#docker-with-persistent-data).
The image uses `sqlite:////data/maverick.db` and `/data/maverick_cache.db`.
Copying `.env.example` unchanged does not override these defaults. Set
`DATABASE_URL` explicitly to use PostgreSQL or another database location.
If an older Docker configuration uses `POSTGRES_URL`, rename that setting to
`DATABASE_URL` before replacing the container. The fallback `POSTGRES_URL`
is used only when `DATABASE_URL` is absent, and the corrected image supplies
a `DATABASE_URL` default.

The container runs as UID/GID 1000. A named volume inherits ownership from the
image's `/data` directory on first use. A bind-mounted host directory must be
writable by that user. Keep the application and its virtual environment under
`/app`; do not mount a data volume over that directory.

### Move data from an older container

The published `1.1.0` image stores SQLite files under `/app`. Back up the
portfolio database before stopping a container started with `--rm`, because
stopping it also removes its writable filesystem. Use SQLite's backup API to
copy a consistent database while the old container is still running.

Replace `OLD_CONTAINER` with the existing container name. Run this against each
user database you intentionally configured, after checking its path.

```bash
docker exec OLD_CONTAINER python -c "import sqlite3; source = sqlite3.connect('file:/app/maverick.db?mode=ro', uri=True); backup = sqlite3.connect('/app/maverick-backup.db'); source.backup(backup); backup.close(); source.close()"
docker cp OLD_CONTAINER:/app/maverick-backup.db ./maverick-backup.db
```

Check the copied database before removing the old container. Restore the backup
as `maverick.db` inside a new, empty data volume owned by UID/GID 1000, then
start the corrected image with that volume. Keep the backup until holdings,
watchlists, and journal entries have been checked in the replacement container.
The cache can be rebuilt; it is separate from user portfolio data.

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
  or `market_data_get_price_history_batch` for a ticker registers that symbol
  in the local `md_stocks` table as a side effect. `market_data_get_quote`
  does not.
- The screening domain (`screening_run_screens`) computes its Maverick
  bullish/bearish/supply-demand screens over whatever symbols are already
  known locally (the same `md_stocks` table). Fetch price history for the
  tickers you care about before running a screen for meaningful coverage;
  there is no S&P 500-wide default universe on a fresh install. With no
  symbols known locally, `screening_run_screens` returns an error that says
  to fetch price history first.

## Connecting A Client After Setup

Prefer STDIO for a single local client. See `mcp-clients.md`.

For HTTP bridge testing:

```bash
make dev
```

## Troubleshooting

- Missing/empty database file: it is created on first tool call. Check the
  configured path and preserve a backup before changing an existing database.
  Do not delete a database to troubleshoot missing holdings.
- Empty screening results: fetch price history for a few tickers via
  `market_data_get_price_history`/`market_data_get_price_history_batch`
  first, then call `screening_run_screens`.
- Redis unavailable: leave Redis variables unset; the app falls back to
  in-memory/SQLite caching.
- Carrying data forward from a pre-v1.0 install: see
  [`migrating-to-v1.md`](migrating-to-v1.md) for the inert-legacy-tables
  note.


## In-memory SQLite

`sqlite:///:memory:` and `sqlite://` retain one connection per engine. Sessions,
raw connections, and schema operations take turns using that connection so an
uncommitted transaction cannot leak into another caller. File SQLite databases
continue to use separate connections.

Anonymous memory databases are local to one engine and process. Closing a
session preserves the data; disposing the engine, invalidating its connection,
or exiting the process loses it. A later schema setup creates an empty database.
Do not hold one connection while requesting another from the same memory engine.
Async engines belong to the event loop that created them. The CI configuration
uses an in-memory database unless a test supplies an explicit URL.


## Adjusted-history freshness

Stored adjusted prices refresh on the next access after 24 hours. There is no
background refresh or instant corporate-action feed. Requesting a larger range
or a provisional current-day bar triggers a refresh sooner. The fetch covers
the complete stored/requested union so old and new bars share one adjustment
basis. Long stored histories can therefore require larger provider requests.

Daily history uses NYSE sessions for US symbols, including class-share forms
such as `BRK.B` and `BRK-B`. Supported exchange suffixes select these calendars:

| Yahoo suffix | Calendar | Calendar timezone |
| --- | --- | --- |
| `.L` | LSE | Europe/London |
| `.T` | JPX | Asia/Tokyo |
| `.TO` | TSX | Canada/Eastern |
| `.AX` | ASX | Australia/Sydney |
| `.HK` | HKEX | Asia/Shanghai |
| `.DE` | XETR | Europe/Berlin |

Other exchange suffixes return an unsupported-calendar error before fetching or
changing history state. The limit applies to history and tools that consume it;
quote and fundamentals lookups still pass through to Yahoo Finance. Additional
exchanges need a verified calendar mapping before history can validate their
completed sessions.

The current date in the selected exchange timezone remains provisional, even
after the scheduled close; a later-date refresh can mark it fresh. Leading
pre-listing gaps are accepted. Absent leading dates are not memoized, so a
request starting before listing may fetch again.
Without previously stored prices, a leading provider omission cannot be
distinguished from a pre-listing gap using OHLCV alone.

Failed or incomplete responses produce an error without changing the previous
snapshot or its freshness. A response superseded by a newer refresh reservation
also errors with a retry instruction. These errors do not disable future reads.
An entirely empty initial range returns an empty frame and remains eligible for
refresh. Existing stored dates may not disappear from an accepted snapshot.

Two nullable columns, `history_refreshed_at` and `history_generation`, are added
to `md_stocks` through the existing additive schema setup. Old bars and company
metadata remain intact until a complete refresh succeeds. No portfolio,
watchlist, or journal tables are rewritten by this change.
