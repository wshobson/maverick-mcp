# Database Setup

MaverickMCP defaults to local SQLite and also supports PostgreSQL for larger
local datasets. Database setup is local personal-use infrastructure, not hosted
SaaS setup.

## Quick Setup

```bash
uv sync --extra dev
cp .env.example .env
./scripts/setup_database.sh
```

The setup script creates the database schema, seeds sample data, and prepares
screening data for local use.

## Default SQLite

```bash
export DATABASE_URL=sqlite:///maverick_mcp.db
./scripts/setup_database.sh
```

SQLite is the lowest-friction development path and is enough for normal local
MCP use.

## PostgreSQL

```bash
createdb maverick_mcp
export DATABASE_URL=postgresql://localhost/maverick_mcp
./scripts/run-migrations.sh upgrade
```

Use PostgreSQL when loading larger market datasets or when you need closer
parity with production-style database behavior.

## Manual Setup

```bash
python scripts/migrate_db.py
python scripts/seed_db.py
python scripts/test_seeded_data.py
```

## Migrations

```bash
./scripts/run-migrations.sh upgrade
alembic current
alembic history
```

## Seed Verification

```bash
sqlite3 maverick_mcp.db "SELECT COUNT(*) FROM mcp_stocks;"
python scripts/test_seeded_data.py
```

## Claude Desktop After Setup

Prefer STDIO for Claude Desktop. See `claude-desktop.md`.

For HTTP bridge testing:

```bash
make dev
curl http://localhost:8003/health
```

## Troubleshooting

- Missing database file: rerun `./scripts/setup_database.sh`.
- Missing screening rows: rerun `python scripts/seed_db.py` or the Tiingo loader.
- Migration mismatch: check `alembic current` and rerun
  `./scripts/run-migrations.sh upgrade`.
- Redis unavailable: leave Redis variables unset; the app should use fallback
  caching.
