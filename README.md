# MaverickMCP - Personal Stock Analysis MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastMCP](https://img.shields.io/badge/FastMCP-3-green.svg)](https://github.com/jlowin/fastmcp)
[![GitHub Stars](https://img.shields.io/github/stars/wshobson/maverick-mcp?style=social)](https://github.com/wshobson/maverick-mcp)
[![GitHub Issues](https://img.shields.io/github/issues/wshobson/maverick-mcp)](https://github.com/wshobson/maverick-mcp/issues)
[![GitHub Forks](https://img.shields.io/github/forks/wshobson/maverick-mcp?style=social)](https://github.com/wshobson/maverick-mcp/network/members)

**MaverickMCP** is a personal-use FastMCP server that provides financial data
analysis, technical indicators, stock screening, and portfolio tracking tools
directly to your Claude Desktop interface. Built for individual traders and
investors, it runs entirely on your own machine with no authentication or
billing complexity.

Core tools need no API key: market data comes from `yfinance`. Two optional
extras add more: `[backtesting]` (VectorBT-powered strategy backtesting) and
`[research]` (LangGraph-based deep research, bring-your-own LLM key).

## Skip the setup — hosted version

Self-hosting MaverickMCP means Python, uv, and MCP client config (Redis and a
research LLM key are optional). If you just want the analysis, [Capital Companion](https://capitalcompanion.ai)
is the hosted product built on the same engine: AI technical analysis, trade-plan
review sheets with outcome tracking, and price alerts. **25 free analyses,
no credit card.**

Self-hosting instructions continue below.

## Why MaverickMCP?

**Key Benefits:**

- **No Setup Complexity**: `make dev` gets the server running; no database
  migrations, no seed scripts, no API key required for core tools.
- **Modern Python Tooling**: Built with `uv` for fast dependency management.
- **Claude Desktop Integration**: Native MCP support via stdio or streamable
  HTTP.
- **37 Core Tools**: Market data, technical analysis, screening, portfolio
  tracking with a risk dashboard, watchlists, and a trade journal.
- **Optional Extras**: 12 backtesting tools and 3 research tools, each fully
  opt-in via `pip install`/`uv sync` extras.
- **Smart Caching**: Tiered cache (memory, then Redis or SQLite) with
  graceful fallback when Redis isn't running.
- **Open Source**: MIT licensed.

## Features

- **Stock Data Access**: Historical and real-time quotes with intelligent
  caching (`yfinance`, no API key required).
- **Technical Analysis**: RSI, MACD, support/resistance, and a combined
  full-analysis tool.
- **Stock Screening**: Maverick bullish, bearish, and supply/demand
  strategies, computed over the tickers you've already queried.
- **Portfolio Tracking**: Positions with average cost-basis, live P&L, a
  risk dashboard, watchlists, and a trade journal.
- **Backtesting** (`[backtesting]` extra): VectorBT engine, 12 rule-based
  strategy templates plus 8 ML strategy classes, optimization, walk-forward
  analysis, and Monte Carlo simulation.
- **Research** (`[research]` extra): LangGraph-based deep research over
  companies, sectors, and market sentiment, backed by Exa web search and a
  bring-your-own LLM.
- **Multi-Transport Support**: Streamable HTTP and STDIO.

## Quick Start

### Prerequisites

- **Python 3.12+**: Core runtime environment
- **[uv](https://docs.astral.sh/uv/)**: Modern Python package manager (recommended)
- Redis (optional, for enhanced caching)
- PostgreSQL or SQLite (optional, for data persistence; SQLite is the default)

#### Installing uv (Recommended)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternative: via pip
pip install uv
```

### Installation

#### Option 1: Run without installing (uvx)

```bash
# Runs the published maverick-mcp-server package via uvx, invoking its
# maverick-mcp console script
uvx --from maverick-mcp-server maverick-mcp --transport stdio
```

#### Option 2: pip install

```bash
pip install "maverick-mcp-server[backtesting,research]"
maverick-mcp --transport stdio
```

Drop `[backtesting,research]` for a smaller, core-only install (37 tools,
no backtesting/research tools registered).

#### Option 3: From source with uv (for development)

```bash
# Clone the repository
git clone https://github.com/wshobson/maverick-mcp.git
cd maverick-mcp

# Install dependencies and create virtual environment in one command
uv sync --extra dev
# Or, for the full tool surface:
uv sync --extra dev --extra backtesting --extra research

# Copy environment template
cp .env.example .env
# Configure DATABASE_URL / LLM_PROVIDER / EXA_API_KEY as needed (all optional)
```

### Start the Server

```bash
# One command to start everything
make dev

# The server is now running with:
# - Streamable HTTP endpoint: http://localhost:8003/mcp/
```

### Connect to Claude Desktop

**Recommended: STDIO connection**

Claude Desktop works best with direct STDIO for local use:

```json
{
  "mcpServers": {
    "maverick-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "maverick-mcp-server",
        "maverick-mcp",
        "--transport",
        "stdio"
      ]
    }
  }
}
```

Running from a local source checkout instead:

```json
{
  "mcpServers": {
    "maverick-mcp": {
      "command": "uv",
      "args": [
        "run",
        "python",
        "-m",
        "maverick.server",
        "--transport",
        "stdio"
      ],
      "cwd": "/path/to/maverick-mcp"
    }
  }
}
```

> [!WARNING]
> **Windows Claude Desktop Users**
> Claude Desktop on Windows currently has a bug where it ignores the `"cwd"` configuration parameter, which can cause the server to crash with a `ModuleNotFoundError` when running via `uv`. 
> 
> To bypass this, wrap the command in `cmd.exe` to force the directory change:
> ```json
> "maverick-mcp": {
>   "command": "cmd.exe",
>   "args": [
>     "/c",
>     "cd /d C:\\Path\\To\\maverick-mcp && uv run python -m maverick.server --transport stdio"
>   ]
> }
> ```

**Config File Location:**
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

Always restart Claude Desktop after making configuration changes.

**Alternative: Streamable HTTP with `mcp-remote`**

Start the server:

```bash
make dev
```

Then configure a bridge:

```json
{
  "mcpServers": {
    "maverick-mcp": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://localhost:8003/mcp/"]
    }
  }
}
```

That's it! MaverickMCP tools will now be available in your Claude Desktop interface.

#### Cursor IDE

**Streamable HTTP bridge**:

```json
{
  "mcpServers": {
    "maverick-mcp": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://localhost:8003/mcp/"]
    }
  }
}
```

**Config Location**: Cursor → Settings → MCP Servers

#### Claude Code CLI

**HTTP transport**:

```bash
claude mcp add --transport http maverick-mcp http://localhost:8003/mcp/
```

**STDIO transport**:

```bash
claude mcp add maverick-mcp uv run python -m maverick.server --transport stdio
```

#### Windsurf IDE

**Streamable HTTP bridge**:

```json
{
  "mcpServers": {
    "maverick-mcp": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://localhost:8003/mcp/"]
    }
  }
}
```

**Config Location**: Windsurf → Settings → Advanced Settings → MCP Servers

#### Why mcp-remote is Needed

The `mcp-remote` tool bridges clients that launch local STDIO commands to a
server that is already running over HTTP:

- **Without mcp-remote**: Client tries STDIO → Server expects HTTP → Connection fails
- **With mcp-remote**: Client uses STDIO → mcp-remote converts to HTTP → Server receives HTTP → Success

## Tools

MaverickMCP registers **37 core tools** with a base install. Two optional
extras add more. Every tool is read-only (`readOnlyHint: true`) unless noted
otherwise. Full behavior detail lives in `docs/ARCHITECTURE.md`,
`docs/features/portfolio.md`, `docs/features/deep-research.md`, and
`docs/api/backtesting.md`.

### Market Data (7)

| Tool | Description |
| --- | --- |
| `market_data_get_price_history` | OHLCV price history for a ticker, smart-cached. |
| `market_data_get_price_history_batch` | Price history for multiple tickers at once. |
| `market_data_get_quote` | A single quote, TTL-cached. |
| `market_data_get_stock_fundamentals` | Valuation, financials, and trading stats. |
| `market_data_get_market_overview` | Indices, sector performance, top movers, and volatility. |
| `market_data_get_chart_links` | Static external chart links for a ticker. |
| `market_data_clear_market_cache` | Clear cached quotes (mutates cache state). |

### Technical Analysis (4)

| Tool | Description |
| --- | --- |
| `technical_get_rsi_analysis` | RSI reading and signal label. |
| `technical_get_macd_analysis` | MACD reading, signal label, and crossover state. |
| `technical_get_support_resistance` | Support/resistance levels. |
| `technical_get_full_technical_analysis` | Full technical analysis: trend, outlook, every indicator. |

### Screening (6)

| Tool | Description |
| --- | --- |
| `screening_get_bullish` | Top Maverick bullish-momentum results, latest snapshot. |
| `screening_get_bearish` | Top bearish setup results, latest snapshot. |
| `screening_get_supply_demand` | Top supply/demand breakout results, latest snapshot. |
| `screening_get_all` | Latest snapshot across all three screens. |
| `screening_get_by_criteria` | Bullish results filtered by arbitrary criteria. |
| `screening_run_screens` | Recompute one screen (or all three) and persist it (mutates). |

Screens run over the local universe of tickers you've already queried via
market-data tools; there is no pre-seeded S&P 500 database. See
`docs/runbooks/database-setup.md`.

### Portfolio (20)

| Tool | Description |
| --- | --- |
| `portfolio_add_position` | Add/average into a position (mutates). |
| `portfolio_get_my_portfolio` | Full portfolio snapshot with live P&L. |
| `portfolio_remove_position` | Remove shares from a position (mutates). |
| `portfolio_clear_portfolio` | Remove every position; requires `confirm=True` (mutates). |
| `portfolio_risk_adjusted_analysis` | ATR-based position sizing/stop/target. |
| `portfolio_compare_tickers` | Side-by-side ticker comparison (auto-uses your portfolio). |
| `portfolio_correlation_analysis` | Correlation matrix and diversification metrics. |
| `portfolio_get_risk_dashboard` | Total value, sector exposure, and risk metrics. |
| `portfolio_check_position_risk` | Pre-trade risk check for a hypothetical trade. |
| `portfolio_get_regime_adjusted_sizing` | Position size scaled by detected market regime. |
| `portfolio_get_risk_alerts` | Current sector/position/portfolio risk alerts. |
| `portfolio_watchlist_create` | Create a named watchlist (mutates). |
| `portfolio_watchlist_add` | Add a ticker to a watchlist (mutates). |
| `portfolio_watchlist_remove` | Remove a ticker from a watchlist (mutates). |
| `portfolio_watchlist_brief` | Intelligence brief for every symbol on a watchlist. |
| `portfolio_journal_add_trade` | Log a new open trade (mutates). |
| `portfolio_journal_close_trade` | Close an open trade; PnL computed automatically (mutates). |
| `portfolio_journal_list_trades` | List journal trades, optionally filtered. |
| `portfolio_journal_review` | Full detail for a single journal trade. |
| `portfolio_get_strategy_performance` | Strategy performance analytics, with optional comparison. |

All analysis tools auto-detect your portfolio positions when no explicit
tickers are supplied. See `docs/features/portfolio.md` for the cost-basis
method and precision rules.

### Backtesting (12, `[backtesting]` extra)

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
| `backtesting_run_ml_strategy_backtest` | Backtest an ML-enhanced strategy (adaptive, ensemble, regime-aware). |
| `backtesting_train_ml_predictor` | Train a random-forest ML predictor for trading signals. |
| `backtesting_analyze_market_regimes` | Detect bear/sideways/bull regimes for a symbol. |
| `backtesting_create_strategy_ensemble` | Backtest a weighted ensemble of base strategies. |

12 rule-based strategy templates plus 8 ML strategy classes. Install with
`uv sync --extra backtesting` or `pip install "maverick-mcp-server[backtesting]"`.
Absent the extra, the server still boots and registers zero
`backtesting_*` tools.

### Research (3, `[research]` extra)

| Tool | Description |
| --- | --- |
| `research_run_comprehensive` | Comprehensive web-search-backed research on a financial topic. |
| `research_analyze_company` | Comprehensive research on a specific company. |
| `research_analyze_sentiment` | Market sentiment analysis for a topic or sector. |

Requires `EXA_API_KEY` (web search) plus a configured BYOK LLM
(`LLM_PROVIDER`/`LLM_API_KEY`/`LLM_MODEL`; see [Configuration](#configuration)).
Install with `uv sync --extra research` or
`pip install "maverick-mcp-server[research]"`. Absent the extra, the server
still boots and registers zero `research_*` tools.

## Resources

- `portfolio://my-holdings` - a passive AI-context snapshot of your default
  portfolio, automatically available to the assistant.

## Prompts

- `analyze_stock(ticker)` - full technical + screening workflow for one ticker.
- `review_portfolio(portfolio_name)` - portfolio + risk review workflow.
- `run_backtest_workflow(ticker, strategy)` - strategy backtesting workflow
  (registered only with the `[backtesting]` extra).

## Configuration

Configure MaverickMCP via `.env` file or environment variables. See
`.env.example` for the complete, code-verified list.

**Essential Settings:**

- `DATABASE_URL` - PostgreSQL connection or `sqlite:///maverick.db` for SQLite (default).
- `REDIS_HOST` - enables Redis caching when set; caching falls back to
  in-memory/SQLite otherwise.
- `LOG_LEVEL` - Logging verbosity (default: `INFO`).

No API key is required to run the core server; stock data comes from `yfinance`.

**Optional (research extra, bring your own key):**

- `LLM_PROVIDER` - `anthropic`, `openai`, `openrouter`, or `openai_compatible`.
- `LLM_API_KEY` - API key for the configured `LLM_PROVIDER`.
- `LLM_MODEL` - Model name for the configured `LLM_PROVIDER`.
- `LLM_BASE_URL` - Base URL override, required when `LLM_PROVIDER=openai_compatible`.
- `LLM_TEMPERATURE` - Sampling temperature (default: `0.0`).
- `EXA_API_KEY` - Web search for the research tools (get at [exa.ai](https://exa.ai)).

Migrating an older `.env` (legacy `OPENROUTER_API_KEY`-style auto-detection,
`TIINGO_API_KEY`, etc.)? See `docs/runbooks/migrating-to-v1.md`.

## Usage Examples

Once connected to Claude Desktop, use natural language:

### Technical Analysis

```
"Show me the RSI and MACD analysis for NVDA"
"Identify support and resistance levels for MSFT"
"Get full technical analysis for AAPL"
```

### Screening

```
"Run the Maverick bullish screen"
"Show me the top supply/demand breakout setups"
```

### Portfolio

```
"Add 10 shares of AAPL I bought at $150.50"
"Show me my portfolio with current prices"
"Analyze correlation in my portfolio"  # Auto-detects your positions
"Get my risk dashboard"
"Add AAPL to my watchlist"
```

### Backtesting (`[backtesting]` extra)

```
"Run a backtest on AAPL using the momentum strategy for the last 6 months"
"Compare mean reversion vs trend following strategies on SPY"
"Optimize the RSI strategy parameters for TSLA"
```

### Research (`[research]` extra)

```
"Research the current state of the AI semiconductor industry"
"Provide comprehensive research on NVDA"
"Analyze market sentiment for the energy sector"
```

## Development

### Commands

```bash
make dev          # Start server (streamable HTTP transport)
make dev-stdio    # Start server (STDIO transport)
make stop         # Stop services

make test              # Unit tests (fast, default marker filter)
make test-all           # All tests, including integration/slow/external
make test-specific TEST=test_name
make test-watch         # Auto-run tests on file changes

make lint         # ruff check + lint-imports
make format       # ruff format + ruff check --fix
make typecheck     # pyright
make check          # lint + typecheck
make docs-check      # validate the documentation catalog
```

```bash
# Using uv directly
uv run pytest                 # Unit tests only
uv run pytest --cov=maverick  # With coverage
uv run pytest -m ""           # All tests (requires PostgreSQL/Redis for some)

uv run ruff check .    # Linting
uv run ruff format .   # Formatting
uv run ty check .      # Type checking (Astral's ty)
```

## Docker (Optional)

For containerized deployment:

```bash
# Copy and configure environment
cp .env.example .env

# Using uv in Docker (recommended for faster builds)
docker build -t maverick-mcp-server .
docker run -p 8003:8000 --env-file .env maverick-mcp-server

# Or start with docker-compose
docker-compose up -d
```

**Note**: The Dockerfile uses `uv` for fast dependency installation. The
image ships the `[backtesting]` and `[research]` extras by default; drop
`--extra backtesting --extra research` from the `uv sync` line in the
Dockerfile for a smaller, core-only image. There is no HTTP `/health`
endpoint or `HEALTHCHECK` -- this is an MCP server, not a REST API.

## Troubleshooting

### Common Issues

**Tools Disappearing in Claude Desktop:**
- **Solution**: Ensure the streamable HTTP endpoint has a trailing slash: `http://localhost:8003/mcp/`
- The 307 redirect from `/mcp` to `/mcp/` causes tool registration to fail
- Always use the exact configuration with trailing slash shown above

**Research Tool Timeouts:**
- Research tools have adaptive timeouts (120s-600s) based on requested depth
- Deep research may take several minutes depending on complexity
- Monitor progress in server logs with `make tail-log`

**Research Tools Not Available:**
- Ensure the `research` extra is installed: `pip install "maverick-mcp-server[research]"`
- Ensure `LLM_PROVIDER`, `LLM_API_KEY`, and `LLM_MODEL` are set in `.env`
- Ensure `EXA_API_KEY` is set for web search

**Backtesting Tools Not Available:**
- Ensure the `backtesting` extra is installed: `pip install "maverick-mcp-server[backtesting]"`

**Empty screening results:**
- There is no pre-seeded universe; fetch price history for the tickers you
  care about first (`market_data_get_price_history`), then run
  `screening_run_screens`. See `docs/runbooks/database-setup.md`.

```bash
# Common development issues
make tail-log          # View server logs
make stop              # Stop services if ports are in use
make clean             # Clean up cache files

# Quick fixes:
# Port 8003 in use → make stop
# Redis connection refused → brew services start redis / unset REDIS_HOST
# Tests failing → make test (unit tests only)
```

## Extending MaverickMCP

Add custom financial analysis tools with simple decorators, following the
same pattern used throughout `maverick/`:

```python
@mcp.tool()
def my_custom_indicator(ticker: str, period: int = 14):
    """Calculate custom technical indicator."""
    # Your analysis logic here
    return {"ticker": ticker, "signal": "buy", "confidence": 0.85}
```

## Getting Help

For issues or questions:

1. **Check Documentation**: Start with this README, [AGENTS.md](AGENTS.md),
   and `docs/INDEX.md`.
2. **Search Issues**: Look through existing [GitHub issues](https://github.com/wshobson/maverick-mcp/issues)
3. **Report Bugs**: Create a new [issue](https://github.com/wshobson/maverick-mcp/issues/new) with details
4. **Request Features**: Suggest improvements via GitHub issues
5. **Contribute**: See our [Contributing Guide](CONTRIBUTING.md) for development setup

## Acknowledgments

MaverickMCP builds on these excellent open-source projects:

- **[FastMCP](https://github.com/jlowin/fastmcp)** - MCP framework powering the server
- **[yfinance](https://github.com/ranaroussi/yfinance)** - Market data access
- **[VectorBT](https://github.com/polakowo/vectorbt)** - Backtesting engine (`[backtesting]` extra)
- **[LangGraph](https://github.com/langchain-ai/langgraph)** - Research workflow orchestration (`[research]` extra)
- **[pandas](https://pandas.pydata.org/)** & **[NumPy](https://numpy.org/)** - Data analysis
- The entire Python open-source community

## License

MIT License - see [LICENSE](LICENSE) file for details. Free to use for personal and commercial purposes.

## Support

If you find MaverickMCP useful:

- Star the repository
- Report bugs via GitHub issues
- Suggest features
- Improve documentation

---

Built for traders and investors. Happy Trading!

[![Verified on MseeP](https://mseep.ai/badge.svg)](https://mseep.ai/app/4678fb21-9034-4446-8628-6508f753d140)

**Read the full build guide**: [How to Build an MCP Stock Analysis Server](https://sethhobson.com/2025/08/how-to-build-an-mcp-stock-analysis-server/)

## Disclaimer

<sub>**This software is for educational and informational purposes only. It is NOT financial advice.**</sub>

<sub>**Investment Risk Warning**: Past performance does not guarantee future results. All investments carry risk of loss, including total loss of capital. Technical analysis and screening results are not predictive of future performance. Market data may be delayed, inaccurate, or incomplete.</sub>

<sub>**No Professional Advice**: This tool provides data analysis, not investment recommendations. Always consult with a qualified financial advisor before making investment decisions. The developers are not licensed financial advisors or investment professionals. Nothing in this software constitutes professional financial, investment, legal, or tax advice.</sub>

<sub>**Data and Accuracy**: Market data provided by third-party sources (Yahoo Finance, and optionally Capital Companion/finviz for market movers). Data may contain errors, delays, or omissions. Technical indicators are mathematical calculations based on historical data. No warranty is made regarding data accuracy or completeness.</sub>

<sub>**Regulatory Compliance**: US Users - This software is not registered with the SEC, CFTC, or other regulatory bodies. International Users - Check local financial software regulations before use. Users are responsible for compliance with all applicable laws and regulations. Some features may not be available in certain jurisdictions.</sub>

<sub>**Limitation of Liability**: Developers disclaim all liability for investment losses or damages. Use this software at your own risk. No guarantee is made regarding software availability or functionality.</sub>

<sub>By using MaverickMCP, you acknowledge these risks and agree to use the software for educational purposes only.</sub>
