# MaverickMCP - Personal Stock Analysis MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastMCP](https://img.shields.io/badge/FastMCP-2.0-green.svg)](https://github.com/jlowin/fastmcp)
[![GitHub Stars](https://img.shields.io/github/stars/wshobson/maverick-mcp?style=social)](https://github.com/wshobson/maverick-mcp)
[![GitHub Issues](https://img.shields.io/github/issues/wshobson/maverick-mcp)](https://github.com/wshobson/maverick-mcp/issues)
[![GitHub Forks](https://img.shields.io/github/forks/wshobson/maverick-mcp?style=social)](https://github.com/wshobson/maverick-mcp/network/members)

**MaverickMCP** is a personal-use FastMCP 2.0 server that provides professional-grade financial data analysis, technical indicators, and portfolio optimization tools directly to your Claude Desktop interface. Built for individual traders and investors, it offers comprehensive stock analysis capabilities without any authentication or billing complexity.

The server runs locally with stdio transport for seamless Claude Desktop integration, focusing on simplicity and core functionality for personal stock analysis needs.


## 🌟 Why MaverickMCP?

MaverickMCP provides professional-grade financial analysis tools directly within your Claude Desktop interface. Perfect for individual traders and investors who want comprehensive stock analysis capabilities without the complexity of expensive platforms or commercial services.

**🚀 Key Benefits:**

- **No Setup Complexity**: Simple `make dev` command gets you running
- **Claude Desktop Integration**: Native MCP support for seamless AI-powered analysis
- **Comprehensive Analysis**: 29 financial tools covering technical indicators, screening, and portfolio optimization
- **Smart Caching**: Redis-powered performance with graceful fallbacks
- **Fast Development**: Hot reload, smart error handling, and parallel processing
- **Open Source**: MIT licensed, community-driven development
- **Educational Focus**: Perfect for learning financial analysis and MCP development

## Features

- **🚀 Fast Development**: Comprehensive Makefile, smart error handling, hot reload, and parallel processing
- **Stock Data Access**: Historical and real-time stock data with intelligent caching
- **Technical Analysis**: 20+ indicators including SMA, EMA, RSI, MACD, Bollinger Bands, and more
- **Stock Screening**: Multiple strategies (Bullish/Bearish momentum, Trending Breakouts) with parallel processing
- **Portfolio Tools**: Correlation analysis, returns calculation, and optimization
- **Market Data**: Sector performance, market movers, and earnings information
- **News Analysis**: Stock news sentiment and fundamental data
- **Smart Caching**: Redis-powered performance with automatic fallback to in-memory storage
- **Database Support**: SQLAlchemy integration with PostgreSQL/SQLite
- **Claude Desktop Ready**: Native MCP stdio transport for seamless AI integration

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- Redis (optional, for enhanced caching)
- PostgreSQL or SQLite (optional, for data persistence)

### Installation

```bash
# Clone the repository
git clone https://github.com/wshobson/maverick-mcp.git
cd maverick-mcp

# Install dependencies (uv recommended for speed)
uv sync

# Or use standard pip
pip install -e .

# Copy environment template  
cp .env.example .env
# Add your Tiingo API key (free at tiingo.com)
```

### Start the Server

```bash
# One command to start everything
make dev

# The server is now running and ready for Claude Desktop!
```

### Connect to Claude Desktop

Add this configuration to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "maverick-mcp": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://localhost:8000/sse"]
    }
  }
}
```

That's it! MaverickMCP tools will now be available in your Claude Desktop interface.

## Available Tools

MaverickMCP provides 29 financial analysis tools organized into focused categories:

### Development Commands

```bash
# Start the server (one command!)
make dev

# Alternative startup methods
./scripts/start-backend.sh --dev    # Script-based startup
./tools/fast_dev.sh                 # Ultra-fast startup (< 3 seconds)
python tools/hot_reload.py          # Auto-restart on file changes

# Server will be available at:
# - SSE endpoint: http://localhost:8000/sse (for Claude Desktop via mcp-remote)
# - Health check: http://localhost:8000/health
```

### Testing

```bash
# Quick test commands
make test              # Run unit tests (5-10 seconds)
make test-specific TEST=test_name  # Run specific test
make test-watch        # Auto-run tests on file changes

# Manual pytest commands
pytest                 # Unit tests only
pytest --cov=maverick_mcp  # With coverage
pytest -m ""           # All tests (requires PostgreSQL/Redis)
```

### Code Quality

```bash
# Quick quality commands
make lint              # Check code quality
make format            # Auto-format code
make typecheck         # Run type checking

# Manual commands
ruff check .           # Linting
ruff format .          # Formatting
pyright                # Type checking
```

## Configuration

Configure MaverickMCP via `.env` file or environment variables:

**Essential Settings:**

- `REDIS_HOST`, `REDIS_PORT` - Redis cache (optional, defaults to localhost:6379)
- `DATABASE_URL` - PostgreSQL connection or `sqlite:///maverick.db` for SQLite
- `LOG_LEVEL` - Logging verbosity (INFO, DEBUG, ERROR)

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

MaverickMCP provides 29 financial analysis tools organized by category:

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

### Stock Screening Tools

- `get_maverick_stocks` - Bullish momentum screening
- `get_maverick_bear_stocks` - Bearish setup identification
- `get_trending_breakout_stocks` - Strong uptrend phase screening
- `get_all_screening_recommendations` - Combined screening results

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

# Start with docker-compose
docker-compose up -d

# Or build and run manually
docker build -t maverick_mcp .
docker run -p 8000:8000 --env-file .env maverick_mcp
```

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

- **Hot Reload**: `python tools/hot_reload.py` - Auto-restart on changes
- **Fast Startup**: `./tools/fast_dev.sh` - < 3 second startup
- **Quick Testing**: `python tools/quick_test.py --test stock` - Test specific features
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

### Development Experience Improvements

- **Comprehensive Makefile**: One command (`make dev`) starts everything
- **Smart Error Handling**: Automatic fix suggestions for common issues
- **Fast Development**: < 3 second startup with `./tools/fast_dev.sh`
- **Parallel Processing**: 4x speedup for stock screening operations
- **Enhanced Tooling**: Hot reload, experiment harness, quick testing

### Technical Improvements

- **Type Checker**: Migrated from mypy to pyright for better performance
- **Market Data**: Improved fallback logic and async support
- **Caching**: Smart Redis caching with graceful in-memory fallback

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
