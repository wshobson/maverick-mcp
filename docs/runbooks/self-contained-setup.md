# Self-Contained Local Setup

This runbook sets up MaverickMCP as a local personal-use financial analysis MCP
server.

## Prerequisites

- Python 3.12
- `uv`
- TA-Lib system library
- Tiingo API key for market data
- Redis optional
- PostgreSQL optional; SQLite works for development

## Install

```bash
uv sync --extra dev
cp .env.example .env
```

Add at least:

```bash
TIINGO_API_KEY=your_tiingo_key
```

Optional research keys:

```bash
EXA_API_KEY=your_exa_key
TAVILY_API_KEY=your_tavily_key
OPENROUTER_API_KEY=your_openrouter_key
```

## Database

SQLite:

```bash
export DATABASE_URL=sqlite:///maverick_mcp.db
./scripts/setup_database.sh
```

PostgreSQL:

```bash
createdb maverick_mcp
export DATABASE_URL=postgresql://localhost/maverick_mcp
./scripts/run-migrations.sh upgrade
```

## Load Market Data

Quick sample:

```bash
python scripts/load_tiingo_data.py \
  --symbols AAPL,MSFT,GOOGL,AMZN,NVDA \
  --years 2 \
  --calculate-indicators \
  --run-screening
```

S&P 500 sample:

```bash
python scripts/load_tiingo_data.py --sp500 --years 2 --calculate-indicators --run-screening
```

## Start MCP Server

```bash
make dev-stdio
```

For HTTP bridge workflows:

```bash
make dev
curl http://localhost:8003/health
```

## Validate

```bash
make test
make lint
make typecheck
make docs-check
```

For external-provider validation, set the relevant API keys and run explicit
focused tests instead of relying on the default unit-test command.
