# Tech Stack

## Primary Language

- **Python 3.12**

## Frameworks & Libraries

### MCP + API

| Library   | Purpose                                |
| --------- | -------------------------------------- |
| fastmcp   | MCP server framework                   |
| mcp       | MCP protocol primitives                |
| fastapi   | HTTP API framework (server transport)  |
| uvicorn   | ASGI server (development)              |
| gunicorn  | ASGI process manager (optional/prod)   |

### Data + Analysis

| Library    | Purpose                               |
| ---------- | ------------------------------------- |
| pandas     | Data manipulation and time series     |
| numpy      | Numerical computations                |
| pandas-ta  | Technical indicators                  |
| TA-Lib     | Additional indicators / TA functions  |
| vectorbt   | Backtesting and strategy evaluation   |

### Providers

| Library  | Purpose                               |
| -------- | ------------------------------------- |
| tiingo   | Primary market data API               |
| yfinance | Supplemental / fallback market data   |
| fredapi  | Macro data (optional)                 |
| httpx    | Async HTTP client                     |

### AI / Research (Optional)

| Library                  | Purpose                                |
| ------------------------ | -------------------------------------- |
| langchain / langgraph    | Research and agent orchestration       |
| anthropic / openai       | LLM providers                          |
| exa-py                   | Web search provider (optional)         |

### Persistence + Caching

| Library     | Purpose                                |
| ----------- | -------------------------------------- |
| sqlalchemy  | ORM / DB access                        |
| alembic     | DB migrations                           |
| aiosqlite   | Async SQLite driver (default DB)       |
| asyncpg     | Async Postgres driver (optional)       |
| redis       | Cache backend (optional)               |

## Database Defaults

- **SQLite** (default) - simple local persistence
- **PostgreSQL** (optional) - set `DATABASE_URL` to enable
- **Redis** (optional) - set `REDIS_HOST`/`REDIS_PORT` (or URL if supported)

## Development Tools

| Tool     | Purpose                |
| -------- | ---------------------- |
| uv       | Package management     |
| ruff     | Linting + formatting   |
| pyright  | Type checking          |
| pytest   | Testing                |

## Local Commands (Makefile)

```bash
make dev        # Start server (SSE transport)
make test       # Unit tests (fast)
make test-all   # All tests (incl. integration)
make check      # Lint + typecheck
```

