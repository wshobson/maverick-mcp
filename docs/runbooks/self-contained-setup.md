# Self-Contained Local Setup

This runbook sets up MaverickMCP as a local personal-use financial analysis MCP
server, from a source checkout, with every optional extra enabled.

## Prerequisites

- Python 3.12+
- `uv`
- Redis optional
- PostgreSQL optional; SQLite works out of the box

No market-data API key is required: core tools (quotes, price history,
fundamentals, technical analysis, screening) run entirely on `yfinance`.
TA-Lib is not required either -- the backtesting extra uses `pandas-ta`, a
pure-Python dependency, so there is no system library to compile.

## Install

```bash
git clone https://github.com/wshobson/maverick-mcp.git
cd maverick-mcp
uv sync --extra dev --extra backtesting --extra research
cp .env.example .env
```

Optional BYOK LLM keys for research and `backtesting_parse_strategy` (leave
unset to run without them):

```bash
LLM_PROVIDER=anthropic
LLM_API_KEY=your_provider_key
LLM_MODEL=claude-sonnet-4-5
```

Optional research search key:

```bash
EXA_API_KEY=your_exa_key
```

## Database

No setup script is needed; schema is created on first use. See
[`database-setup.md`](database-setup.md) for detail.

SQLite (default):

```bash
export DATABASE_URL=sqlite:///maverick.db
```

PostgreSQL:

```bash
createdb maverick
export DATABASE_URL=postgresql://localhost/maverick
```

## Start MCP Server

```bash
make dev-stdio
```

For HTTP bridge workflows:

```bash
make dev
```

The server is now available at `http://localhost:8003/mcp/` (streamable
HTTP) or over stdio, per `--transport`.

## Bring In Market Data

There is no bulk seed step. Call market-data tools for the tickers you care
about (this also registers them for screening):

```text
"Get the price history for AAPL, MSFT, and NVDA"
"Run the Maverick bullish screen"
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
