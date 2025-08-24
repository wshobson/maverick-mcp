# MaverickMCP - Personal Stock Analysis MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastMCP](https://img.shields.io/badge/FastMCP-2.0-green.svg)](https://github.com/jlowin/fastmcp)
[![GitHub Stars](https://img.shields.io/github/stars/wshobson/maverick-mcp?style=social)](https://github.com/wshobson/maverick-mcp)
[![GitHub Issues](https://img.shields.io/github/issues/wshobson/maverick-mcp)](https://github.com/wshobson/maverick-mcp/issues)
[![GitHub Forks](https://img.shields.io/github/forks/wshobson/maverick-mcp?style=social)](https://github.com/wshobson/maverick-mcp/network/members)

**MaverickMCP** is a personal-use FastMCP 2.0 server that provides professional-grade financial data analysis, technical indicators, and portfolio optimization tools directly to your Claude Desktop interface. Built for individual traders and investors, it offers comprehensive stock analysis capabilities without any authentication or billing complexity.

The server comes pre-seeded with all 520 S&P 500 stocks and provides advanced screening recommendations across multiple strategies. It runs locally with HTTP/SSE/STDIO transport options for seamless integration with Claude Desktop and other MCP clients.

## 🌟 Why MaverickMCP?

MaverickMCP provides professional-grade financial analysis tools directly within your Claude Desktop interface. Perfect for individual traders and investors who want comprehensive stock analysis capabilities without the complexity of expensive platforms or commercial services.

**🚀 Key Benefits:**

- **No Setup Complexity**: Simple `make dev` command gets you running (or `uv sync` + `make dev`)
- **Modern Python Tooling**: Built with `uv` for lightning-fast dependency management
- **Claude Desktop Integration**: Native MCP support for seamless AI-powered analysis
- **Comprehensive Analysis**: 29+ financial tools covering technical indicators, screening, and portfolio optimization
- **Smart Caching**: Redis-powered performance with graceful fallbacks
- **Fast Development**: Hot reload, smart error handling, and parallel processing
- **Open Source**: MIT licensed, community-driven development
- **Educational Focus**: Perfect for learning financial analysis and MCP development

## Features

- **🚀 Pre-seeded Database**: 520 S&P 500 stocks with comprehensive screening recommendations
- **🚀 Fast Development**: Comprehensive Makefile, smart error handling, hot reload, and parallel processing
- **Stock Data Access**: Historical and real-time stock data with intelligent caching
- **Technical Analysis**: 20+ indicators including SMA, EMA, RSI, MACD, Bollinger Bands, and more
- **Stock Screening**: Multiple strategies (Maverick Bullish/Bearish, Trending Breakouts) with parallel processing
- **Portfolio Tools**: Correlation analysis, returns calculation, and optimization
- **Market Data**: Sector performance, market movers, and earnings information
- **Smart Caching**: Redis-powered performance with automatic fallback to in-memory storage
- **Database Support**: SQLAlchemy integration with PostgreSQL/SQLite (defaults to SQLite)
- **Multi-Transport Support**: HTTP, SSE, and STDIO transports for all MCP clients

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+**: Core runtime environment
- **[uv](https://docs.astral.sh/uv/)**: Modern Python package manager (recommended)
- Redis (optional, for enhanced caching)
- PostgreSQL or SQLite (optional, for data persistence)

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

#### Option 1: Using uv (Recommended - Fastest)

```bash
# Clone the repository
git clone https://github.com/wshobson/maverick-mcp.git
cd maverick-mcp

# Install dependencies and create virtual environment in one command
uv sync

# Copy environment template
cp .env.example .env
# Add your Tiingo API key (free at tiingo.com)
```

#### Option 2: Using pip (Traditional)

```bash
# Clone the repository
git clone https://github.com/wshobson/maverick-mcp.git
cd maverick-mcp

# Create virtual environment and install
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .

# Copy environment template
cp .env.example .env
# Add your Tiingo API key (free at tiingo.com)
```

### Start the Server

```bash
# One command to start everything (includes S&P 500 data seeding on first run)
make dev

# The server is now running with:
# - HTTP endpoint: http://localhost:8000/mcp
# - SSE endpoint: http://localhost:8000/sse
# - 520 S&P 500 stocks pre-loaded with screening data
```

### Connect to Claude Desktop

Claude Desktop uses STDIO to communicate with mcp-remote, which then connects to your HTTP server:

```json
{
  "mcpServers": {
    "maverick-mcp": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://localhost:8000/mcp"]
    }
  }
}
```

**Alternate option: Direct STDIO Connection (Development Only)**

Claude Desktop directly connects via STDIO without any HTTP layer:

```json
{
  "mcpServers": {
    "maverick-mcp": {
      "command": "uv",
      "args": [
        "run",
        "python",
        "-m",
        "maverick_mcp.api.server",
        "--transport",
        "stdio"
      ],
      "cwd": "/path/to/maverick-mcp"
    }
  }
}
```

> **Note**: The `mcp-remote` package bridges Claude Desktop's STDIO-only support to HTTP/SSE servers. For native remote server support, use [Claude.ai web interface](https://claude.ai/settings/integrations) instead of Claude Desktop.

That's it! MaverickMCP tools will now be available in your Claude Desktop interface.

### Connect to Other MCP Clients

> ⚠️ **Transport Compatibility Warning**: Different MCP clients support different transport methods. Using the wrong configuration will result in connection failures. Please use the exact configuration for your client.

#### Transport Compatibility Matrix

| MCP Client         | STDIO | HTTP | SSE | Notes                                        |
| ------------------ | ----- | ---- | --- | -------------------------------------------- |
| **Claude Desktop** | ✅    | ❌   | ❌  | STDIO-only, requires mcp-remote for HTTP/SSE |
| **Cursor IDE**     | ✅    | ❌   | ✅  | Supports STDIO and SSE                       |
| **Claude Code**    | ✅    | ✅   | ✅  | Supports all transports                      |
| **Continue.dev**   | ✅    | ❌   | ✅  | Supports STDIO and SSE                       |
| **Windsurf IDE**   | ✅    | ❌   | ✅  | Supports STDIO and SSE                       |
| **Goose CLI**      | ✅    | ❌   | ✅  | Supports STDIO and SSE                       |

#### Claude Desktop (Most Popular) - STDIO Only

**⚠️ Important**: Claude Desktop ONLY supports STDIO transport. It cannot directly connect to HTTP or SSE servers and requires the `mcp-remote` bridge tool.

**For HTTP Server Connection (Recommended)**:

```json
{
  "mcpServers": {
    "maverick-mcp": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://localhost:8000/mcp"]
    }
  }
}
```

**For Direct STDIO (Development Only)**:

```json
{
  "mcpServers": {
    "maverick-mcp": {
      "command": "uv",
      "args": [
        "run",
        "python",
        "-m",
        "maverick_mcp.api.server",
        "--transport",
        "stdio"
      ]
    }
  }
}
```

**Config Location**:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

#### Cursor IDE - STDIO and SSE

**Option 1: STDIO (via mcp-remote)**:

```json
{
  "mcpServers": {
    "maverick-mcp": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://localhost:8000/mcp"]
    }
  }
}
```

**Option 2: Direct SSE**:

```json
{
  "mcpServers": {
    "maverick-mcp": {
      "url": "http://localhost:8000/sse"
    }
  }
}
```

**Config Location**: Cursor → Settings → MCP Servers

#### Claude Code CLI - All Transports

**HTTP Transport (Recommended)**:

```bash
claude mcp add --transport http maverick-mcp http://localhost:8000/mcp
```

**SSE Transport (Legacy)**:

```bash
claude mcp add --transport sse maverick-mcp http://localhost:8000/sse
```

**STDIO Transport (Development)**:

```bash
claude mcp add maverick-mcp uv run python -m maverick_mcp.api.server --transport stdio
```

#### Continue.dev - STDIO and SSE

**Option 1: STDIO (via mcp-remote)**:

```json
{
  "experimental": {
    "modelContextProtocolServer": {
      "transport": {
        "type": "stdio",
        "command": "npx",
        "args": ["-y", "mcp-remote", "http://localhost:8000/mcp"]
      }
    }
  }
}
```

**Option 2: Direct SSE**:

```json
{
  "mcpServers": {
    "maverick-mcp": {
      "url": "http://localhost:8000/sse"
    }
  }
}
```

**Config Location**: `~/.continue/config.json`

#### Windsurf IDE - STDIO and SSE

**Option 1: STDIO (via mcp-remote)**:

```json
{
  "mcpServers": {
    "maverick-mcp": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://localhost:8000/mcp"]
    }
  }
}
```

**Option 2: Direct SSE**:

```json
{
  "mcpServers": {
    "maverick-mcp": {
      "serverUrl": "http://localhost:8000/sse"
    }
  }
}
```

**Config Location**: Windsurf → Settings → Advanced Settings → MCP Servers

#### Why mcp-remote is Needed

The `mcp-remote` tool bridges the gap between STDIO-only clients (like Claude Desktop) and HTTP/SSE servers. Without it, these clients cannot connect to remote MCP servers:

- **Without mcp-remote**: Client tries STDIO → Server expects HTTP → Connection fails
- **With mcp-remote**: Client uses STDIO → mcp-remote converts to HTTP → Server receives HTTP → Success

## Available Tools

MaverickMCP provides 29+ financial analysis tools organized into focused categories, with access to pre-seeded S&P 500 screening data:

### Development Commands

```bash
# Start the server (one command!)
make dev

# Alternative startup methods
./scripts/start-backend.sh --dev    # Script-based startup
./tools/fast_dev.sh                 # Ultra-fast startup (< 3 seconds)
uv run python tools/hot_reload.py   # Auto-restart on file changes

# Server will be available at:
# - HTTP endpoint: http://localhost:8000/mcp (recommended for Claude Desktop via mcp-remote)
# - SSE endpoint: http://localhost:8000/sse (legacy compatibility)
# - Health check: http://localhost:8000/health
```

### Testing

```bash
# Quick test commands
make test              # Run unit tests (5-10 seconds)
make test-specific TEST=test_name  # Run specific test
make test-watch        # Auto-run tests on file changes

# Using uv (recommended)
uv run pytest                 # Unit tests only
uv run pytest --cov=maverick_mcp  # With coverage
uv run pytest -m ""           # All tests (requires PostgreSQL/Redis)

# Alternative: Direct pytest (if activated in venv)
pytest                 # Unit tests only
pytest --cov=maverick_mcp  # With coverage
pytest -m ""           # All tests (requires PostgreSQL/Redis)
```

### Code Quality

```bash
# Quick quality commands
make lint              # Check code quality (ruff)
make format            # Auto-format code (ruff)
make typecheck         # Run type checking (ty)

# Using uv (recommended)
uv run ruff check .    # Linting
uv run ruff format .   # Formatting
uv run ty check .      # Type checking (Astral's modern type checker)

# Alternative: Direct commands (if activated in venv)
ruff check .           # Linting
ruff format .          # Formatting
ty check .             # Type checking

# Ultra-fast one-liner (no installation needed)
uvx ty check .         # Run ty directly without installing
```

## Configuration

Configure MaverickMCP via `.env` file or environment variables:

**Essential Settings:**

- `REDIS_HOST`, `REDIS_PORT` - Redis cache (optional, defaults to localhost:6379)
- `DATABASE_URL` - PostgreSQL connection or `sqlite:///maverick_mcp.db` for SQLite (default)
- `LOG_LEVEL` - Logging verbosity (INFO, DEBUG, ERROR)
- S&P 500 data automatically seeds on first startup

**Required API Keys:**

- `TIINGO_API_KEY` - Stock data provider (free tier available at [tiingo.com](https://tiingo.com))

**Optional API Keys:**

- `OPENAI_API_KEY` - For AI-powered analysis features
- `ANTHROPIC_API_KEY` - Alternative LLM provider
- `FRED_API_KEY` - Federal Reserve economic data

**Performance:**

- `CACHE_ENABLED=true` - Enable Redis caching
- `CACHE_TTL_SECONDS=3600` - Cache duration

## Tools

MaverickMCP provides 29+ financial analysis tools organized by category, with pre-seeded S&P 500 data:

### Stock Data Tools

- `fetch_stock_data` - Get historical stock data with intelligent caching
- `fetch_stock_data_batch` - Fetch data for multiple tickers simultaneously
- `get_news_sentiment` - Analyze news sentiment for any ticker
- `clear_cache` / `get_cache_info` - Cache management utilities

### Technical Analysis Tools

- `get_rsi_analysis` - RSI calculation with buy/sell signals
- `get_macd_analysis` - MACD analysis with trend identification
- `get_support_resistance` - Identify key price levels
- `get_full_technical_analysis` - Comprehensive technical analysis
- `get_stock_chart_analysis` - Visual chart generation

### Portfolio Tools

- `risk_adjusted_analysis` - Risk-based position sizing
- `compare_tickers` - Side-by-side ticker comparison
- `portfolio_correlation_analysis` - Correlation matrix analysis

### Stock Screening Tools (Pre-seeded with S&P 500)

- `get_maverick_stocks` - Bullish momentum screening from 520 S&P 500 stocks
- `get_maverick_bear_stocks` - Bearish setup identification from pre-analyzed data
- `get_trending_breakout_stocks` - Strong uptrend phase screening with supply/demand analysis
- `get_all_screening_recommendations` - Combined screening results across all strategies
- Database includes comprehensive screening data updated regularly

### Market Data Tools

- Market overview, sector performance, earnings calendars
- Economic indicators and Federal Reserve data
- Real-time market movers and sentiment analysis

## Resources

- `stock://{ticker}` - Latest year of stock data
- `stock://{ticker}/{start_date}/{end_date}` - Custom date range
- `stock_info://{ticker}` - Basic stock information

## Prompts

- `stock_analysis(ticker)` - Comprehensive stock analysis prompt
- `market_comparison(tickers)` - Compare multiple stocks
- `portfolio_optimization(tickers, risk_profile)` - Portfolio optimization guidance

## Docker (Optional)

For containerized deployment:

```bash
# Copy and configure environment
cp .env.example .env

# Using uv in Docker (recommended for faster builds)
docker build -t maverick_mcp .
docker run -p 8000:8000 --env-file .env maverick_mcp

# Or start with docker-compose
docker-compose up -d
```

**Note**: The Dockerfile uses `uv` for fast dependency installation and smaller image sizes.

## Troubleshooting

```bash
# Common development issues
make tail-log          # View server logs
make stop              # Stop services if ports are in use
make clean             # Clean up cache files

# Quick fixes:
# Port 8000 in use → make stop
# Redis connection refused → brew services start redis
# Tests failing → make test (unit tests only)
# Slow startup → ./tools/fast_dev.sh
# Missing S&P 500 data → uv run python scripts/seed_sp500.py
```

## Extending MaverickMCP

Add custom financial analysis tools with simple decorators:

```python
@mcp.tool()
def my_custom_indicator(ticker: str, period: int = 14):
    """Calculate custom technical indicator."""
    # Your analysis logic here
    return {"ticker": ticker, "signal": "buy", "confidence": 0.85}

@mcp.resource("custom://analysis/{ticker}")
def custom_analysis(ticker: str):
    """Custom analysis resource."""
    # Your resource logic here
    return f"Custom analysis for {ticker}"
```

## Development Tools

### Quick Development Workflow

```bash
make dev               # Start everything
make stop              # Stop services
make tail-log          # Follow server logs
make test              # Run tests quickly
make experiment        # Test custom analysis scripts
```

### Smart Error Handling

MaverickMCP includes helpful error diagnostics:

- DataFrame column case sensitivity → Shows correct column name
- Connection failures → Provides specific fix commands
- Import errors → Shows exact install commands
- Database issues → Suggests SQLite fallback

### Fast Development Options

- **Hot Reload**: `uv run python tools/hot_reload.py` - Auto-restart on changes
- **Fast Startup**: `./tools/fast_dev.sh` - < 3 second startup
- **Quick Testing**: `uv run python tools/quick_test.py --test stock` - Test specific features
- **Experiment Harness**: Drop .py files in `tools/experiments/` for auto-execution

### Performance Features

- **Parallel Screening**: 4x faster stock analysis with ProcessPoolExecutor
- **Smart Caching**: `@quick_cache` decorator for instant re-runs
- **Optimized Tests**: Unit tests complete in 5-10 seconds

## Getting Help

For issues or questions:

1. **📖 Check Documentation**: Start with this README and [CLAUDE.md](CLAUDE.md)
2. **🔍 Search Issues**: Look through existing [GitHub issues](https://github.com/wshobson/maverick-mcp/issues)
3. **🐛 Report Bugs**: Create a new [issue](https://github.com/wshobson/maverick-mcp/issues/new) with details
4. **💡 Request Features**: Suggest improvements via GitHub issues
5. **🤝 Contribute**: See our [Contributing Guide](CONTRIBUTING.md) for development setup

## Recent Updates

### Personal Use Optimization

- **No Authentication Required**: Removed all authentication/billing complexity for personal use
- **Pre-seeded S&P 500 Database**: 520 stocks with comprehensive screening recommendations
- **Simplified Architecture**: Clean, focused codebase for core stock analysis functionality
- **Multi-Transport Support**: HTTP, SSE, and STDIO for all MCP clients

### Development Experience Improvements

- **Comprehensive Makefile**: One command (`make dev`) starts everything including database seeding
- **Smart Error Handling**: Automatic fix suggestions for common issues
- **Fast Development**: < 3 second startup with `./tools/fast_dev.sh`
- **Parallel Processing**: 4x speedup for stock screening operations
- **Enhanced Tooling**: Hot reload, experiment harness, quick testing

### Technical Improvements

- **Modern Tooling**: Migrated to uv and ty for faster dependency management and type checking
- **Market Data**: Improved fallback logic and async support
- **Caching**: Smart Redis caching with graceful in-memory fallback
- **Database**: SQLite default with PostgreSQL option for enhanced performance

## Acknowledgments

MaverickMCP builds on these excellent open-source projects:

- **[FastMCP](https://github.com/jlowin/fastmcp)** - MCP framework powering the server
- **[yfinance](https://github.com/ranaroussi/yfinance)** - Market data access
- **[TA-Lib](https://github.com/mrjbq7/ta-lib)** - Technical analysis indicators
- **[pandas](https://pandas.pydata.org/)** & **[NumPy](https://numpy.org/)** - Data analysis
- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern web framework
- The entire Python open-source community

## License

MIT License - see [LICENSE](LICENSE) file for details. Free to use for personal and commercial purposes.

## Support

If you find MaverickMCP useful:

- ⭐ Star the repository
- 🐛 Report bugs via GitHub issues
- 💡 Suggest features
- 📖 Improve documentation

---

Built for traders and investors. Happy Trading!

## Disclaimer

<sub>**This software is for educational and informational purposes only. It is NOT financial advice.**</sub>

<sub>**Investment Risk Warning**: Past performance does not guarantee future results. All investments carry risk of loss, including total loss of capital. Technical analysis and screening results are not predictive of future performance. Market data may be delayed, inaccurate, or incomplete.</sub>

<sub>**No Professional Advice**: This tool provides data analysis, not investment recommendations. Always consult with a qualified financial advisor before making investment decisions. The developers are not licensed financial advisors or investment professionals. Nothing in this software constitutes professional financial, investment, legal, or tax advice.</sub>

<sub>**Data and Accuracy**: Market data provided by third-party sources (Tiingo, Yahoo Finance, FRED). Data may contain errors, delays, or omissions. Technical indicators are mathematical calculations based on historical data. No warranty is made regarding data accuracy or completeness.</sub>

<sub>**Regulatory Compliance**: US Users - This software is not registered with the SEC, CFTC, or other regulatory bodies. International Users - Check local financial software regulations before use. Users are responsible for compliance with all applicable laws and regulations. Some features may not be available in certain jurisdictions.</sub>

<sub>**Limitation of Liability**: Developers disclaim all liability for investment losses or damages. Use this software at your own risk. No guarantee is made regarding software availability or functionality.</sub>

<sub>By using MaverickMCP, you acknowledge these risks and agree to use the software for educational purposes only.</sub>
