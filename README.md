# MaverickMCP - Personal Stock Analysis MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastMCP](https://img.shields.io/badge/FastMCP-2.0-green.svg)](https://github.com/jlowin/fastmcp)
[![GitHub Stars](https://img.shields.io/github/stars/wshobson/maverick-mcp?style=social)](https://github.com/wshobson/maverick-mcp)
[![GitHub Issues](https://img.shields.io/github/issues/wshobson/maverick-mcp)](https://github.com/wshobson/maverick-mcp/issues)
[![GitHub Forks](https://img.shields.io/github/forks/wshobson/maverick-mcp?style=social)](https://github.com/wshobson/maverick-mcp/network/members)

**MaverickMCP** is a personal-use FastMCP 2.0 server that provides professional-grade financial data analysis, technical indicators, and portfolio optimization tools directly to your Claude Desktop interface. Built for individual traders and investors, it offers comprehensive stock analysis capabilities without any authentication or billing complexity.

The server comes pre-seeded with all 520 S&P 500 stocks and provides advanced screening recommendations across multiple strategies. It runs locally with HTTP/SSE/STDIO transport options for seamless integration with Claude Desktop and other MCP clients.

## Why MaverickMCP?

MaverickMCP provides professional-grade financial analysis tools directly within your Claude Desktop interface. Perfect for individual traders and investors who want comprehensive stock analysis capabilities without the complexity of expensive platforms or commercial services.

**Key Benefits:**

- **No Setup Complexity**: Simple `make dev` command gets you running (or `uv sync` + `make dev`)
- **Modern Python Tooling**: Built with `uv` for lightning-fast dependency management
- **Claude Desktop Integration**: Native MCP support for seamless AI-powered analysis
- **Comprehensive Analysis**: 29+ financial tools covering technical indicators, screening, and portfolio optimization
- **Smart Caching**: Redis-powered performance with graceful fallbacks
- **Fast Development**: Hot reload, smart error handling, and parallel processing
- **Open Source**: MIT licensed, community-driven development
- **Educational Focus**: Perfect for learning financial analysis and MCP development

## Features

- **Pre-seeded Database**: 520 S&P 500 stocks with comprehensive screening recommendations
- **Advanced Backtesting**: VectorBT-powered engine with 15+ built-in strategies and ML algorithms
- **Fast Development**: Comprehensive Makefile, smart error handling, hot reload, and parallel processing
- **Stock Data Access**: Historical and real-time stock data with intelligent caching
- **Technical Analysis**: 20+ indicators including SMA, EMA, RSI, MACD, Bollinger Bands, and more
- **Stock Screening**: Multiple strategies (Maverick Bullish/Bearish, Trending Breakouts) with parallel processing
- **Portfolio Tools**: Correlation analysis, returns calculation, and optimization
- **Market Data**: Sector performance, market movers, and earnings information
- **Smart Caching**: Redis-powered performance with automatic fallback to in-memory storage
- **Database Support**: SQLAlchemy integration with PostgreSQL/SQLite (defaults to SQLite)
- **Multi-Transport Support**: HTTP, SSE, and STDIO transports for all MCP clients

## Quick Start

### Prerequisites

- **Python 3.12+**: Core runtime environment
- **[uv](https://docs.astral.sh/uv/)**: Modern Python package manager (recommended)
- **TA-Lib**: Technical analysis library for advanced indicators
- Redis (optional, for enhanced caching)
- PostgreSQL or SQLite (optional, for data persistence)

#### Installing TA-Lib

TA-Lib is required for technical analysis calculations.

**macOS and Linux (Homebrew):**
```bash
brew install ta-lib
```

**Windows (Multiple Options):**

**Option 1: Conda/Anaconda (Recommended - Easiest)**
```bash
conda install -c conda-forge ta-lib
```

**Option 2: Pre-compiled Wheels**
1. Download the appropriate wheel for your Python version from:
   - [cgohlke/talib-build releases](https://github.com/cgohlke/talib-build/releases)
   - Choose the file matching your Python version (e.g., `TA_Lib-0.4.28-cp312-cp312-win_amd64.whl` for Python 3.12 64-bit)
2. Install using pip:
```bash
pip install path/to/downloaded/TA_Lib-X.X.X-cpXXX-cpXXX-win_amd64.whl
```

**Option 3: Alternative Pre-compiled Package**
```bash
pip install TA-Lib-Precompiled
```

**Option 4: Build from Source (Advanced)**
If other methods fail, you can build from source:
1. Install Microsoft C++ Build Tools
2. Download and extract ta-lib C library to `C:\ta-lib`
3. Build using Visual Studio tools
4. Run `pip install ta-lib`

**Verification:**
Test your installation:
```bash
python -c "import talib; print(talib.__version__)"
```

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
# - HTTP endpoint: http://localhost:8003/mcp/
# - SSE endpoint: http://localhost:8003/sse/
# - 520 S&P 500 stocks pre-loaded with screening data
```

### Connect to Claude Desktop

**Recommended: SSE Connection (Stable and Reliable)**

This configuration provides stable tool registration and prevents tools from disappearing:

```json
{
  "mcpServers": {
    "maverick-mcp": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://localhost:8003/sse/"]
    }
  }
}
```

> **Important**: Note the trailing slash in `/sse/` - this is REQUIRED to prevent redirect issues!

**Config File Location:**
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

**Why This Configuration Works Best:**
- Stable tool registration - tools don't disappear after initial connection
- Reliable connection management through SSE transport
- Proper session persistence for long-running analysis tasks
- All 29+ financial tools available consistently

**Alternative: Direct STDIO Connection (Development Only)**

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

> **Important**: Always **restart Claude Desktop** after making configuration changes. The SSE configuration via mcp-remote has been tested and confirmed to provide stable, persistent tool access without connection drops.

That's it! MaverickMCP tools will now be available in your Claude Desktop interface.

#### Claude Desktop (Most Popular) - Recommended Configuration

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
      "args": ["-y", "mcp-remote", "http://localhost:8003/sse/"]
    }
  }
}
```

**Option 2: Direct SSE**:

```json
{
  "mcpServers": {
    "maverick-mcp": {
      "url": "http://localhost:8003/sse/"
    }
  }
}
```

**Config Location**: Cursor → Settings → MCP Servers

#### Claude Code CLI - All Transports

**HTTP Transport (Recommended)**:

```bash
claude mcp add --transport http maverick-mcp http://localhost:8003/mcp/
```

**SSE Transport (Alternative)**:

```bash
claude mcp add --transport sse maverick-mcp http://localhost:8003/sse/
```

**STDIO Transport (Development)**:

```bash
claude mcp add maverick-mcp uv run python -m maverick_mcp.api.server --transport stdio
```

#### Windsurf IDE - STDIO and SSE

**Option 1: STDIO (via mcp-remote)**:

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

**Option 2: Direct SSE**:

```json
{
  "mcpServers": {
    "maverick-mcp": {
      "serverUrl": "http://localhost:8003/sse/"
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

MaverickMCP provides 39+ financial analysis tools organized into focused categories, including advanced AI-powered research agents:

### Development Commands

```bash
# Start the server (one command!)
make dev

# Alternative startup methods
./scripts/start-backend.sh --dev    # Script-based startup
./tools/fast_dev.sh                 # Ultra-fast startup (< 3 seconds)
uv run python tools/hot_reload.py   # Auto-restart on file changes

# Server will be available at:
# - HTTP endpoint: http://localhost:8003/mcp/ (streamable-http - use with mcp-remote)
# - SSE endpoint: http://localhost:8003/sse/ (SSE - direct connection only, not mcp-remote)
# - Health check: http://localhost:8003/health
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

- `OPENROUTER_API_KEY` - **Strongly Recommended for Research**: Access to 400+ AI models with intelligent cost optimization (40-60% cost savings)
- `EXA_API_KEY` - **Recommended for Research**: Web search capabilities for comprehensive research
- `OPENAI_API_KEY` - Direct OpenAI access (fallback)
- `ANTHROPIC_API_KEY` - Direct Anthropic access (fallback)
- `FRED_API_KEY` - Federal Reserve economic data
- `TAVILY_API_KEY` - Alternative web search provider

**Performance:**

- `CACHE_ENABLED=true` - Enable Redis caching
- `CACHE_TTL_SECONDS=3600` - Cache duration

## Usage Examples

### Backtesting Example

Once connected to Claude Desktop, you can use natural language to run backtests:

```
"Run a backtest on AAPL using the momentum strategy for the last 6 months"

"Compare the performance of mean reversion vs trend following strategies on SPY"

"Optimize the RSI strategy parameters for TSLA with walk-forward analysis"

"Show me the Sharpe ratio and maximum drawdown for a portfolio of tech stocks using the adaptive ML strategy"

"Generate a detailed backtest report for the ensemble strategy on the S&P 500 sectors"
```

### Technical Analysis Example

```
"Show me the RSI and MACD analysis for NVDA"

"Identify support and resistance levels for MSFT"

"Get full technical analysis for the top 5 momentum stocks"
```

### Portfolio Management Example (NEW)

```
"Add 10 shares of AAPL I bought at $150.50"

"Show me my portfolio with current prices"

"Compare my portfolio holdings"  # No tickers needed!

"Analyze correlation in my portfolio"  # Auto-detects your positions

"Remove 5 shares of MSFT"
```

### Portfolio Optimization Example

```
"Optimize a portfolio of AAPL, GOOGL, MSFT, and AMZN for maximum Sharpe ratio"

"Calculate the correlation matrix for my tech portfolio"

"Analyze the risk-adjusted returns for energy sector stocks"
```

## Tools

MaverickMCP provides 39+ financial analysis tools organized by category, including advanced AI-powered research agents:

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

### Portfolio Management Tools (NEW) - Personal Portfolio Tracking

- `portfolio_add_position` - Add or update positions with automatic cost basis averaging
- `portfolio_get_my_portfolio` - View portfolio with live P&L calculations
- `portfolio_remove_position` - Remove partial or full positions
- `portfolio_clear_portfolio` - Clear all positions with safety confirmation

**Portfolio Features:**
- **Persistent Storage**: Track your actual holdings with cost basis
- **Automatic Averaging**: Cost basis updates automatically on repeat purchases
- **Live P&L**: Real-time unrealized gains/losses on all positions
- **Portfolio Resource**: `portfolio://my-holdings` provides AI context automatically
- **Multi-Portfolio Support**: Track multiple portfolios (IRA, 401k, taxable, etc.)
- **Fractional Shares**: Full support for partial share positions

### Portfolio Analysis Tools

- `risk_adjusted_analysis` - Risk-based position sizing with position awareness
- `compare_tickers` - Side-by-side ticker comparison (auto-uses your portfolio)
- `portfolio_correlation_analysis` - Correlation matrix analysis (auto-analyzes your holdings)

**Smart Integration:**
- All analysis tools auto-detect your portfolio positions
- No need to manually enter tickers you already own
- Position-aware recommendations (averaging up/down, profit taking)

### Stock Screening Tools (Pre-seeded with S&P 500)

- `get_maverick_stocks` - Bullish momentum screening from 520 S&P 500 stocks
- `get_maverick_bear_stocks` - Bearish setup identification from pre-analyzed data
- `get_trending_breakout_stocks` - Strong uptrend phase screening with supply/demand analysis
- `get_all_screening_recommendations` - Combined screening results across all strategies
- Database includes comprehensive screening data updated regularly

### Advanced Research Tools (NEW) - AI-Powered Deep Analysis

- `research_comprehensive` - Full parallel research with multiple AI agents (7-256x faster)
- `research_company` - Company-specific deep research with financial analysis
- `analyze_market_sentiment` - Multi-source sentiment analysis with confidence tracking
- `coordinate_agents` - Multi-agent supervisor for complex research orchestration

**Research Features:**
- **Parallel Execution**: 7-256x speedup with intelligent agent orchestration
- **Adaptive Timeouts**: 120s-600s based on research depth and complexity
- **Smart Model Selection**: Automatic selection from 400+ models via OpenRouter
- **Cost Optimization**: 40-60% cost reduction through intelligent model routing
- **Early Termination**: Confidence-based early stopping to save time and costs
- **Content Filtering**: High-credibility source prioritization
- **Error Recovery**: Circuit breakers and comprehensive error handling

### Backtesting Tools (NEW) - Production-Ready Strategy Testing

- `run_backtest` - Execute backtests with VectorBT engine for any strategy
- `compare_strategies` - A/B testing framework for strategy comparison
- `optimize_strategy` - Walk-forward optimization with parameter tuning
- `analyze_backtest_results` - Comprehensive performance metrics and risk analysis
- `get_backtest_report` - Generate detailed HTML reports with visualizations

**Backtesting Features:**
- **15+ Built-in Strategies**: Including ML-powered adaptive, ensemble, and regime-aware algorithms
- **VectorBT Integration**: High-performance vectorized backtesting engine
- **Parallel Processing**: 7-256x speedup for multi-strategy evaluation
- **Advanced Metrics**: Sharpe, Sortino, Calmar ratios, maximum drawdown, win rate
- **Walk-Forward Optimization**: Out-of-sample testing and validation
- **Monte Carlo Simulations**: Robustness testing with confidence intervals
- **Multi-Timeframe Support**: From 1-minute to monthly data
- **Custom Strategy Development**: Easy-to-use templates for custom strategies

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

## Test Examples - Validate All Features

Test the comprehensive research capabilities and parallel processing improvements with these examples:

### Core Research Features

1. **Basic Research with Timeout Protection**
   ```
   "Research the current state of the AI semiconductor industry and identify the top 3 investment opportunities"
   ```
   - Tests: Basic research, adaptive timeouts, industry analysis

2. **Comprehensive Company Research with Parallel Agents**
   ```
   "Provide comprehensive research on NVDA including fundamental analysis, technical indicators, competitive positioning, and market sentiment using multiple research approaches"
   ```
   - Tests: Parallel orchestration, multi-agent coordination, company research

3. **Cost-Optimized Quick Research**
   ```
   "Give me a quick overview of AAPL's recent earnings and stock performance"
   ```
   - Tests: Intelligent model selection, cost optimization, quick analysis

### Performance Testing

4. **Parallel Performance Benchmark**
   ```
   "Research and compare MSFT, GOOGL, and AMZN simultaneously focusing on cloud computing revenue growth"
   ```
   - Tests: Parallel execution speedup (7-256x), multi-company analysis

5. **Deep Research with Early Termination**
   ```
   "Conduct exhaustive research on Tesla's autonomous driving technology and its impact on the stock valuation"
   ```
   - Tests: Deep research depth, confidence tracking, early termination (0.85 threshold)

### Error Handling & Recovery

6. **Error Recovery and Circuit Breaker Test**
   ```
   "Research 10 penny stocks with unusual options activity and provide risk assessments for each"
   ```
   - Tests: Circuit breaker activation, error handling, fallback mechanisms

7. **Supervisor Agent Coordination**
   ```
   "Analyze the renewable energy sector using both technical and fundamental analysis approaches, then synthesize the findings into actionable investment recommendations"
   ```
   - Tests: Supervisor routing, agent coordination, result synthesis

### Advanced Features

8. **Sentiment Analysis with Content Filtering**
   ```
   "Analyze market sentiment for Bitcoin and cryptocurrency stocks over the past week, filtering for high-credibility sources only"
   ```
   - Tests: Sentiment analysis, content filtering, source credibility

9. **Timeout Stress Test**
   ```
   "Research the entire S&P 500 technology sector companies and rank them by growth potential"
   ```
   - Tests: Timeout management, large-scale analysis, performance under load

10. **Multi-Modal Research Integration**
    ```
    "Research AMD using technical analysis, then find recent news about their AI chips, analyze competitor Intel's position, and provide a comprehensive investment thesis with risk assessment"
    ```
    - Tests: All research modes, integration, synthesis, risk assessment

### Bonus Edge Case Tests

11. **Empty/Invalid Query Handling**
    ```
    "Research [intentionally leave blank or use symbol that doesn't exist like XYZABC]"
    ```
    - Tests: Error messages, helpful fix suggestions

12. **Token Budget Optimization**
    ```
    "Provide the most comprehensive possible analysis of the entire semiconductor industry including all major players, supply chain dynamics, geopolitical factors, and 5-year projections"
    ```
    - Tests: Progressive token allocation, budget management, depth vs breadth

### Expected Performance Metrics

When running these tests, you should observe:
- **Parallel Speedup**: 7-256x faster for multi-entity queries
- **Response Times**: Simple queries ~10s, complex research 30-120s
- **Cost Efficiency**: 60-80% reduction vs premium-only models
- **Confidence Scores**: Early termination when confidence > 0.85
- **Error Recovery**: Graceful degradation without crashes
- **Model Selection**: Automatic routing to optimal models per task

## Docker (Optional)

For containerized deployment:

```bash
# Copy and configure environment
cp .env.example .env

# Using uv in Docker (recommended for faster builds)
docker build -t maverick_mcp .
docker run -p 8003:8003 --env-file .env maverick_mcp

# Or start with docker-compose
docker-compose up -d
```

**Note**: The Dockerfile uses `uv` for fast dependency installation and smaller image sizes.

## Troubleshooting

### Common Issues

**Tools Disappearing in Claude Desktop:**
- **Solution**: Ensure SSE endpoint has trailing slash: `http://localhost:8003/sse/`
- The 307 redirect from `/sse` to `/sse/` causes tool registration to fail
- Always use the exact configuration with trailing slash shown above

**Research Tool Timeouts:**
- Research tools have adaptive timeouts (120s-600s)
- Deep research may take 2-10 minutes depending on complexity
- Monitor progress in server logs with `make tail-log`

**OpenRouter Not Working:**
- Ensure `OPENROUTER_API_KEY` is set in `.env`
- Check API key validity at [openrouter.ai](https://openrouter.ai)
- System falls back to direct providers if OpenRouter unavailable

```bash
# Common development issues
make tail-log          # View server logs
make stop              # Stop services if ports are in use
make clean             # Clean up cache files

# Quick fixes:
# Port 8003 in use → make stop
# Redis connection refused → brew services start redis
# Tests failing → make test (unit tests only)
# Slow startup → ./tools/fast_dev.sh
# Missing S&P 500 data → uv run python scripts/seed_sp500.py
# Research timeouts → Check logs, increase timeout settings
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

1. **Check Documentation**: Start with this README and [CLAUDE.md](CLAUDE.md)
2. **Search Issues**: Look through existing [GitHub issues](https://github.com/wshobson/maverick-mcp/issues)
3. **Report Bugs**: Create a new [issue](https://github.com/wshobson/maverick-mcp/issues/new) with details
4. **Request Features**: Suggest improvements via GitHub issues
5. **Contribute**: See our [Contributing Guide](CONTRIBUTING.md) for development setup

## Recent Updates

### Production-Ready Backtesting Framework (NEW)

- **VectorBT Integration**: High-performance vectorized backtesting engine for institutional-grade performance
- **15+ Built-in Strategies**: Including ML-powered adaptive, ensemble, and regime-aware algorithms
- **Parallel Processing**: 7-256x speedup for multi-strategy evaluation and optimization
- **Advanced Analytics**: Comprehensive metrics including Sharpe, Sortino, Calmar ratios, and drawdown analysis
- **Walk-Forward Optimization**: Out-of-sample testing with automatic parameter tuning
- **Monte Carlo Simulations**: Robustness testing with confidence intervals
- **LangGraph Workflow**: Multi-agent orchestration for intelligent strategy selection and validation
- **Production Features**: Database persistence, batch processing, and HTML reporting

### Advanced Research Agents

- **Parallel Research Execution**: Achieved 7-256x speedup (exceeded 2x target) with intelligent agent orchestration
- **Adaptive Timeout Protection**: Dynamic timeouts (120s-600s) based on research depth and complexity
- **Intelligent Model Selection**: OpenRouter integration with 400+ models, 40-60% cost reduction
- **Comprehensive Error Handling**: Circuit breakers, retry logic, and graceful degradation
- **Early Termination**: Confidence-based stopping to optimize time and costs
- **Content Filtering**: High-credibility source prioritization for quality results
- **Multi-Agent Orchestration**: Supervisor pattern for complex research coordination

### Performance Improvements

- **Parallel Agent Execution**: Increased concurrent agents from 4 to 6
- **Optimized Semaphores**: BoundedSemaphore for better resource management  
- **Reduced Rate Limiting**: Delays decreased from 0.5s to 0.05s
- **Batch Processing**: Improved throughput for multiple research tasks
- **Smart Caching**: Redis-powered with in-memory fallback

### Testing & Quality

- **84% Test Coverage**: 93 tests with comprehensive coverage
- **Zero Linting Errors**: Fixed 947 issues for clean codebase
- **Full Type Annotations**: Complete type coverage for research components
- **Error Recovery Testing**: Comprehensive failure scenario coverage

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

- Star the repository
- Report bugs via GitHub issues
- Suggest features
- Improve documentation

---

Built for traders and investors. Happy Trading!

[![MSeeP.ai Security Assessment Badge](https://mseep.net/pr/wshobson-maverick-mcp-badge.png)](https://mseep.ai/app/wshobson-maverick-mcp)

**Read the full build guide**: [How to Build an MCP Stock Analysis Server](https://sethhobson.com/2025/08/how-to-build-an-mcp-stock-analysis-server/)

## Disclaimer

<sub>**This software is for educational and informational purposes only. It is NOT financial advice.**</sub>

<sub>**Investment Risk Warning**: Past performance does not guarantee future results. All investments carry risk of loss, including total loss of capital. Technical analysis and screening results are not predictive of future performance. Market data may be delayed, inaccurate, or incomplete.</sub>

<sub>**No Professional Advice**: This tool provides data analysis, not investment recommendations. Always consult with a qualified financial advisor before making investment decisions. The developers are not licensed financial advisors or investment professionals. Nothing in this software constitutes professional financial, investment, legal, or tax advice.</sub>

<sub>**Data and Accuracy**: Market data provided by third-party sources (Tiingo, Yahoo Finance, FRED). Data may contain errors, delays, or omissions. Technical indicators are mathematical calculations based on historical data. No warranty is made regarding data accuracy or completeness.</sub>

<sub>**Regulatory Compliance**: US Users - This software is not registered with the SEC, CFTC, or other regulatory bodies. International Users - Check local financial software regulations before use. Users are responsible for compliance with all applicable laws and regulations. Some features may not be available in certain jurisdictions.</sub>

<sub>**Limitation of Liability**: Developers disclaim all liability for investment losses or damages. Use this software at your own risk. No guarantee is made regarding software availability or functionality.</sub>

<sub>By using MaverickMCP, you acknowledge these risks and agree to use the software for educational purposes only.</sub>
