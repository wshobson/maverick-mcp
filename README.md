<!-- mcp-name: io.github.wshobson/maverick-mcp -->

# MaverickMCP: stock market MCP server for local analysis

[![CI](https://github.com/wshobson/maverick-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/wshobson/maverick-mcp/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

MaverickMCP is an open-source stock market MCP server for stock analysis,
portfolio tracking, and Python backtesting. It connects an AI assistant to
Yahoo Finance data through `yfinance`, with no API key required for core tools.
The server runs on your computer and stores your portfolio, watchlists, and
trade journal in a local database.

MCP means Model Context Protocol, the standard that lets an AI assistant call
external tools. Use MaverickMCP with clients such as Claude Desktop, Codex,
Cursor, and VS Code. Optional packages add backtesting and financial research
with a language model and web search.

MaverickMCP is for educational and informational use. It does not place orders
or provide financial advice. Market data can be delayed, incomplete, or wrong.

## Features

| Area | What you can do |
| --- | --- |
| Market data and technical analysis | Get quotes, historical prices, fundamentals, RSI, MACD, and observed support and resistance levels. |
| Stock screening | Run bullish, bearish, and supply/demand screens over symbols you have loaded. |
| Portfolio tracking | Record positions, calculate average cost and profit or loss, review risk, manage watchlists, and keep a trade journal. |
| Optional backtesting and research | Test strategies on historical data, compare results, or research companies and sectors with source citations. |

The current source has 38 core tools, 12 optional backtesting tools, and 3
optional research tools. The published `v1.1.0` release has 37 core tools and
predates the current correctness fixes. Follow the source installation below
for the behavior documented here.

## Quick start

Install [Python 3.12 or later](https://www.python.org/downloads/) and
[uv](https://docs.astral.sh/uv/getting-started/installation/). SQLite is included.
Redis and PostgreSQL are optional.

### Installation

First, clone the repository and install the core tools.

```bash
git clone https://github.com/wshobson/maverick-mcp.git
cd maverick-mcp
uv sync
cp .env.example .env
```

Second, add optional packages if you need backtesting or research.

```bash
uv sync --extra backtesting --extra research
```

Third, connect your MCP client using the command below. Replace the path with
the absolute path to your checkout. A client using STDIO starts the server
itself, so no separate server process is needed.

```bash
uv run --directory /absolute/path/to/maverick-mcp maverick-mcp --transport stdio
```

The PyPI name `maverick-mcp-server` belongs to an unrelated project while the
[name-transfer request](https://github.com/pypi/support/issues/12150) is pending.
Do not install that name from PyPI. The existing release can be run from its
Git tag, but it does not include the changes described for current source.

```bash
uvx --from "git+https://github.com/wshobson/maverick-mcp@v1.1.0" maverick-mcp --transport stdio
```

### Connect your MCP client

Choose STDIO for one local client, or Streamable HTTP for clients that share a
server process. For HTTP, run `make dev` and use `http://localhost:8003/mcp`.
The endpoint has no trailing slash. `/mcp/` redirects and can break registration.

A client that uses a `mcpServers` JSON configuration can launch the local checkout
with the following entry.

```json
{
  "mcpServers": {
    "maverick-mcp": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/absolute/path/to/maverick-mcp",
        "maverick-mcp",
        "--transport",
        "stdio"
      ]
    }
  }
}
```

Configuration formats vary by client. The [MCP client setup guide](docs/runbooks/mcp-clients.md)
covers Claude Desktop, Claude Code, Codex, Cursor, VS Code, GitHub Copilot CLI,
OpenCode, and other clients. Use the guide for the exact file location and keys.

HTTP binds to `127.0.0.1` by default. The server has no authentication, so keep
it local unless you have configured separate access controls.

### Docker with persistent data

Build the current source to include the fixes documented here. The existing
published `1.1.0` image predates them.

```bash
cp .env.example .env
docker build -t maverick-mcp:local .
docker run --rm --name maverick-mcp \
  -p 127.0.0.1:8003:8000 \
  --env-file .env \
  --mount source=maverick-data,target=/data \
  maverick-mcp:local
```

The named volume keeps your database and cache when the container is replaced.
The image stores them at `/data/maverick.db` and `/data/maverick_cache.db` and
runs as user 1000. A bind-mounted directory must be writable by that user.
For PostgreSQL, set `DATABASE_URL` explicitly. The `POSTGRES_URL` fallback
does not override the image's `DATABASE_URL` default.
The image includes both optional packages.

If you used an older container, back up its databases under `/app` before
removing it. See [Docker data storage and migration](docs/runbooks/database-setup.md#docker-data-storage)
for backup instructions and PostgreSQL overrides. Do not mount a volume over
`/app`, which contains the installed application.

## Tools

The tables list tool names as they appear to an MCP client. Tools marked
"mutates" change stored data or cache state. A portfolio entry or journal
trade is a local record and does not submit an order to a broker.

### Market data

| Tool | Description |
| --- | --- |
| `market_data_get_price_history` | OHLCV price history for a ticker, stored locally and refreshed on access. |
| `market_data_get_price_history_batch` | Price history for multiple tickers at once. |
| `market_data_get_quote` | A single quote, cached briefly. Returns an error when Yahoo has no price for the ticker (delisted or unknown). |
| `market_data_get_stock_fundamentals` | Valuation, financials, and trading stats. |
| `market_data_get_market_overview` | Indices, sector performance, top movers, and volatility. |
| `market_data_get_chart_links` | Static external chart links for a ticker. |
| `market_data_clear_market_cache` | Clear cached quotes (mutates cache state). |

### Technical analysis

| Tool | Description |
| --- | --- |
| `technical_get_rsi_analysis` | RSI reading and signal label. |
| `technical_get_macd_analysis` | MACD reading, signal label, and crossover state. |
| `technical_get_support_resistance` | Observed price extrema over the requested history. |
| `technical_get_full_technical_analysis` | Trend, outlook, and technical indicators. |

### Stock screening

| Tool | Description |
| --- | --- |
| `screening_get_bullish` | Top Maverick bullish-momentum results, latest snapshot. |
| `screening_get_bearish` | Top bearish setup results, latest snapshot. |
| `screening_get_supply_demand` | Top supply/demand breakout results, latest snapshot. |
| `screening_get_all` | Latest snapshot across all three screens. |
| `screening_get_by_criteria` | Bullish results filtered by arbitrary criteria. |
| `screening_run_screens` | Recompute one screen (or all three) and persist it (mutates). |

Fetch price history for the symbols you want to screen, then run a screen.
A quote lookup does not add a symbol to the screening universe. A new database
has no default stock universe. See the [database setup guide](docs/runbooks/database-setup.md).

### Portfolio tracking, watchlists, and trade journal

| Tool | Description |
| --- | --- |
| `portfolio_add_position` | Add/average into a position (mutates). |
| `portfolio_get_my_portfolio` | Portfolio snapshot with P&L using current available quotes. |
| `portfolio_remove_position` | Remove shares from a position (mutates). |
| `portfolio_clear_portfolio` | Remove every position; requires `confirm=True` (mutates). |
| `portfolio_risk_adjusted_analysis` | ATR-based position sizing/stop/target. |
| `portfolio_compare_tickers` | Ticker comparison, using your portfolio when tickers are omitted. |
| `portfolio_correlation_analysis` | Correlation matrix and diversification metrics. |
| `portfolio_get_risk_dashboard` | Total value, sector exposure, and risk metrics. |
| `portfolio_check_position_risk` | Pre-trade risk check for a hypothetical trade. |
| `portfolio_get_regime_adjusted_sizing` | Position size scaled by detected market regime. |
| `portfolio_get_risk_alerts` | Current sector/position/portfolio risk alerts. |
| `portfolio_watchlist_list` | List watchlist IDs and names. |
| `portfolio_watchlist_create` | Create a named watchlist (mutates). |
| `portfolio_watchlist_add` | Add a ticker to a watchlist (mutates). |
| `portfolio_watchlist_remove` | Remove a ticker from a watchlist (mutates). |
| `portfolio_watchlist_brief` | Quotes and analysis for the symbols on a watchlist. |
| `portfolio_journal_add_trade` | Log a new open trade; optional ISO `entry_date` records a past trade (mutates). |
| `portfolio_journal_close_trade` | Close an open trade; profit or loss calculated automatically (mutates). |
| `portfolio_journal_list_trades` | List journal trades, optionally filtered. |
| `portfolio_journal_review` | Full detail for a single journal trade. |
| `portfolio_get_strategy_performance` | Strategy performance analytics, with optional comparison. |

Tools that accept an optional ticker list use your portfolio when it is omitted.
See [portfolio behavior and precision](docs/features/portfolio.md) for details.

### Python backtesting (`backtesting` extra)

| Tool | Description |
| --- | --- |
| `backtesting_run_backtest` | Run a single-strategy backtest: metrics, trades, analysis. |
| `backtesting_optimize_strategy` | Grid-search a strategy's parameters. |
| `backtesting_walk_forward_analysis` | Rolling optimize/test windows to gauge robustness. |
| `backtesting_monte_carlo_simulation` | Bootstrap-resample trades for a return/drawdown distribution. |
| `backtesting_compare_strategies` | Backtest multiple strategies on the same symbol and rank them. |
| `backtesting_list_strategies` | List every rule-based strategy template with default parameters. |
| `backtesting_backtest_portfolio` | Backtest one strategy across multiple symbols. |
| `backtesting_parse_strategy` | Parse a natural-language description into a strategy + parameters (BYOK LLM). |
| `backtesting_run_ml_strategy_backtest` | Backtest a machine learning strategy (predictor, adaptive, ensemble, or regime). |
| `backtesting_train_ml_predictor` | Train a random-forest ML predictor for trading signals. |
| `backtesting_analyze_market_regimes` | Detect bear/sideways/bull regimes for a symbol. |
| `backtesting_create_strategy_ensemble` | Backtest a weighted ensemble of base strategies. |

The extra includes 12 strategy templates plus machine learning models and
strategies. Install it with `uv sync --extra backtesting`. Without the extra,
no `backtesting_*` tools are registered. See the [backtesting API reference](docs/api/backtesting.md)
for metric definitions, strategy parameters, and output limits.

### Financial research (`research` extra)

| Tool | Description |
| --- | --- |
| `research_run_comprehensive` | Web research on a financial topic, with source citations. |
| `research_analyze_company` | Company research, with source citations. |
| `research_analyze_sentiment` | Market sentiment analysis for a topic or sector. |

Install with `uv sync --extra research`, then configure a language model and
Exa or SearXNG web search. Without the extra, no `research_*` tools are registered.
Research returns source citations and an explicit error when no usable evidence
remains. See [research setup and behavior](docs/features/deep-research.md).

## Configuration

Use environment variables or a `.env` file. The [.env.example](.env.example)
file lists the supported settings.

| Setting | Purpose |
| --- | --- |
| `DATABASE_URL` | Select SQLite or PostgreSQL. Local processes default to `sqlite:///maverick.db`; containers default to `sqlite:////data/maverick.db`. |
| `REDIS_HOST` | Enable Redis caching. Otherwise the server uses memory and SQLite. |
| `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_MODEL` | Configure the language model for research and natural-language strategy parsing. |
| `LLM_BASE_URL` | Set a custom endpoint, required for `openai_compatible`. |
| `LLM_TEMPERATURE` | Optionally override the model's sampling temperature. Omit it to use the provider default. |
| `EXA_API_KEY` | Enable Exa web search for research. |
| `RESEARCH_SEARCH_BACKEND`, `SEARXNG_BASE_URL` | Use `searxng` with a server that supports JSON responses instead of Exa. |

Supported model providers are `anthropic`, `openai`, `openrouter`, and
`openai_compatible`. Core tools do not require a model API key. Optional research
can incur charges from your model and search providers.

## Usage examples

Once connected, ask your assistant to use the tools. Include the dates, symbols,
and prices needed for a request.

- "Get RSI and MACD for NVDA, then show the observed price range for the last 90 days."
- "Fetch one year of AAPL and MSFT price history, then run the bullish screen."
- "List my watchlists and show my portfolio with current prices."
- "Backtest the SMA crossover strategy on SPY from January 1, 2024 to January 1, 2025."

The `analyze_stock` and `review_portfolio` prompts provide guided workflows.
Installing the backtesting extra also adds `run_backtest_workflow`. The
`portfolio://my-holdings` resource exposes a snapshot of your default portfolio.

## Common questions

### Does the Yahoo Finance connection need an API key?

No. Core market data uses `yfinance`. Data availability and delays depend on
Yahoo Finance, and requests can fail or be rate limited. Market movers use
finviz. Neither source is an execution feed for placing orders.

### Why does stock screening return no results?

A new database has no stock universe. Fetch price history for your chosen
tickers, then call `screening_run_screens`. A quote request alone does not
register a ticker for screening.

### Why are backtesting or research tools missing?

Install the corresponding extra in the environment your MCP client starts,
then restart the server. Research also needs a configured model and search
backend. See the [research setup guide](docs/features/deep-research.md).

### Why does an HTTP client fail to connect?

Use `http://localhost:8003/mcp`, without a trailing slash, and check that the
server is running. A STDIO client should launch its own process instead.
See [client troubleshooting](docs/runbooks/mcp-clients.md).

## Development

Install the development tools and optional packages before running all checks.

```bash
uv sync --extra dev --extra backtesting --extra research
make test
make lint
make typecheck
make docs-check
```

`make test` excludes tests marked `integration`, `slow`, or `external`.
See the [testing guide](docs/testing/README.md) for focused tests and integration
checks. Run provider tests only with the required credentials and consent.

Read the [architecture](ARCHITECTURE.md) before adding a tool, and keep business
logic in its domain service. The [contributing guide](CONTRIBUTING.md) describes
the review process. Current work and remaining limitations are recorded in the
[documentation index](docs/INDEX.md) and [debt tracker](docs/exec-plans/tech-debt-tracker.md).

## Help and license

Report problems in [GitHub issues](https://github.com/wshobson/maverick-mcp/issues)
with the command, version, and error message. Remove credentials and personal
portfolio data from logs before sharing them. Use the [security policy](SECURITY.md)
for vulnerability reports.

MaverickMCP uses [FastMCP](https://github.com/jlowin/fastmcp),
[yfinance](https://github.com/ranaroussi/yfinance),
[vectorbt](https://github.com/polakowo/vectorbt), and
[LangGraph](https://github.com/langchain-ai/langgraph). The project is licensed
under the [MIT License](LICENSE).

## Disclaimer

MaverickMCP provides educational information, not financial, investment, or tax
advice. Historical backtests and technical indicators do not predict future
returns. Market data and generated research can contain errors. Verify the
underlying sources before using an analysis to make a decision.
