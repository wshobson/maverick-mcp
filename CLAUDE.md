# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with the MaverickMCP codebase.

**🚀 QUICK START**: Run `make dev` to start the server. Connect with Claude Desktop using `mcp-remote`. See "Claude Desktop Setup" section below.

## Project Overview

MaverickMCP is a personal stock analysis MCP server built for Claude Desktop. It provides:

- Pre-seeded database with all 520 S&P 500 stocks and screening recommendations
- Real-time and historical stock data access with intelligent caching
- Advanced technical analysis tools (RSI, MACD, Bollinger Bands, etc.)
- Multiple stock screening strategies (Maverick Bullish/Bearish, Supply/Demand Breakouts)
- Portfolio optimization and correlation analysis
- Market and macroeconomic data integration
- SQLAlchemy-based database integration with SQLite default (PostgreSQL optional)
- Redis caching for high performance (optional)
- Clean, personal-use architecture without authentication complexity

## Project Structure

- `maverick_mcp/`
  - `api/`: MCP server implementation
    - `server.py`: Main FastMCP server (simple stock analysis mode)
    - `routers/`: Domain-specific routers for organized tool groups
  - `config/`: Configuration and settings
  - `core/`: Core financial analysis functions
  - `data/`: Data handling, caching, and database models
  - `providers/`: Stock, market, and macro data providers
  - `utils/`: Development utilities and performance optimizations
  - `tests/`: Comprehensive test suite
  - `validation/`: Request/response validation
- `tools/`: Development tools for faster workflows
- `docs/`: Architecture documentation
- `scripts/`: Startup and utility scripts
- `Makefile`: Central command interface

## Environment Setup

1. **Prerequisites**:

   - **Python 3.12+**: Core runtime environment
   - **[uv](https://docs.astral.sh/uv/)**: Modern Python package manager (recommended)
   - Redis server (optional, for enhanced caching performance)
   - PostgreSQL (optional, SQLite works fine for personal use)

2. **Installation**:

   ```bash
   # Clone the repository
   git clone https://github.com/wshobson/maverick-mcp.git
   cd maverick-mcp

   # Install dependencies using uv (recommended - fastest)
   uv sync

   # Or use traditional pip
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e .

   # Set up environment
   cp .env.example .env
   # Add your Tiingo API key (required)
   ```

3. **Required Configuration** (add to `.env`):

   ```
   # Required - Stock data provider (free tier available)
   TIINGO_API_KEY=your-tiingo-key
   ```

4. **Optional Configuration** (add to `.env`):

   ```
   # OpenRouter API (strongly recommended for research - access to 400+ AI models with intelligent cost optimization)
   OPENROUTER_API_KEY=your-openrouter-key
   
   # Web Search API (recommended for research features)
   EXA_API_KEY=your-exa-key

   # Enhanced data providers (optional)
   FRED_API_KEY=your-fred-key

   # Database (optional - uses SQLite by default)
   DATABASE_URL=postgresql://localhost/maverick_mcp

   # Redis (optional - works without caching)
   REDIS_HOST=localhost
   REDIS_PORT=6379
   ```

   **Get a free Tiingo API key**: Sign up at [tiingo.com](https://tiingo.com) - free tier includes 500 requests/day.

   **OpenRouter API (Recommended)**: Sign up at [openrouter.ai](https://openrouter.ai) for access to 400+ AI models with intelligent cost optimization. The system automatically selects optimal models based on task requirements.

## Quick Start Commands

### Essential Commands (Powered by Makefile)

```bash
# Start the MCP server
make dev              # One command to start everything

# Development
make backend          # Start backend server only
make tail-log         # Follow logs in real-time
make stop             # Stop all services

# Testing
make test             # Run unit tests (5-10 seconds)
make test-watch       # Auto-run tests on file changes
make test-cov         # Run with coverage report

# Code Quality
make lint             # Check code quality
make format           # Auto-format code
make typecheck        # Run type checking
make check            # Run all checks

# Database
make migrate          # Run database migrations
make setup            # Initial setup

# Utilities
make clean            # Clean up generated files
make redis-start      # Start Redis (if using caching)

# Quick shortcuts
make d                # Alias for make dev
make t                # Alias for make test
make l                # Alias for make lint
make c                # Alias for make check
```

## Claude Desktop Setup

### Connection Methods

**✅ RECOMMENDED**: Claude Desktop works best with the **SSE endpoint via mcp-remote bridge**. This configuration has been tested and **prevents tools from disappearing** after initial connection.

#### Method A: SSE Server with mcp-remote Bridge (Recommended - Stable)

This is the **tested and proven method for Claude Desktop** - provides stable tool registration:

1. **Start the SSE server**:
   ```bash
   make dev  # Runs SSE server on port 8003
   ```

2. **Configure with mcp-remote bridge**:
   Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

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

   > **CRITICAL**: Note the trailing slash in `/sse/` - this is REQUIRED to prevent 307 redirect issues that cause tools to disappear!

**Why This Configuration Works Best**:
- ✅ **Prevents Tool Disappearing**: Tools remain available throughout your session (trailing slash is critical!)
- ✅ **Stable Connection**: SSE transport provides consistent communication
- ✅ **Session Persistence**: Maintains connection state for complex analysis workflows
- ✅ **All 35+ Tools Available**: Reliable access to all financial and research tools
- ✅ **Tested and Confirmed**: This exact configuration has been verified to work

#### Method B: HTTP Streamable Server with mcp-remote Bridge (Alternative)
   
1. **Start the HTTP Streamable server**:
   ```bash
   make dev  # Runs HTTP streamable server on port 8003
   ```

2. **Configure with mcp-remote bridge**:
   Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

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

**Benefits**:
- ✅ Uses HTTP Streamable transport
- ✅ Alternative to SSE endpoint
- ✅ Supports remote access

#### Method C: Remote via Claude.ai (Alternative)
   
   For native remote server support, use [Claude.ai web interface](https://claude.ai/settings/integrations) instead of Claude Desktop.

3. **Restart Claude Desktop** and test with: "Show me technical analysis for AAPL"

### Other Popular MCP Clients

> ⚠️ **Critical Transport Warning**: MCP clients have specific transport limitations. Using incorrect configurations will cause connection failures. Always verify which transports your client supports.

#### Transport Compatibility Matrix

| MCP Client           | STDIO | HTTP | SSE | Optimal Method                                |
|----------------------|-------|------|-----|-----------------------------------------------|
| **Claude Desktop**   | ❌    | ❌   | ✅  | **SSE via mcp-remote** (stable, tested)      |
| **Cursor IDE**       | ✅    | ❌   | ✅  | SSE and STDIO supported                       |
| **Claude Code CLI**  | ✅    | ✅   | ✅  | All transports supported                      |
| **Continue.dev**     | ✅    | ❌   | ✅  | SSE and STDIO supported                       |
| **Windsurf IDE**     | ✅    | ❌   | ✅  | SSE and STDIO supported                       |

#### Claude Desktop (Most Commonly Used)

**✅ TESTED CONFIGURATION**: Use SSE endpoint with mcp-remote bridge - prevents tools from disappearing and ensures stable connection.

**Configuration Location:**
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

**SSE Connection with mcp-remote (Tested and Stable):**

1. Start the server:
   ```bash
   make dev  # Starts SSE server on port 8003
   ```

2. Configure Claude Desktop:
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
   
   > **CRITICAL**: Always include the trailing slash (`/sse/`) to prevent 307 redirects!

**Important**: This exact configuration has been tested and confirmed to prevent the common issue where tools appear initially but then disappear from Claude Desktop.

**Restart Required:** Always restart Claude Desktop after config changes.

#### Cursor IDE - SSE and STDIO Support

**Option 1: Direct SSE (Recommended):**
```json
{
  "mcpServers": {
    "maverick-mcp": {
      "url": "http://localhost:8003/sse/"
    }
  }
}
```

**Location:** Cursor → Settings → MCP Servers

#### Claude Code CLI - Full Transport Support

**SSE Transport (Recommended):**
```bash
claude mcp add --transport sse maverick-mcp http://localhost:8003/sse/
```

**HTTP Transport (Alternative):**
```bash
claude mcp add --transport http maverick-mcp http://localhost:8003/mcp/
```

**STDIO Transport (Development only):**
```bash
claude mcp add maverick-mcp uv run python -m maverick_mcp.api.server --transport stdio
```

#### Continue.dev - SSE and STDIO Support

**Option 1: Direct SSE (Recommended):**
```json
{
  "mcpServers": {
    "maverick-mcp": {
      "url": "http://localhost:8003/sse/"
    }
  }
}
```

**Option 2: SSE via mcp-remote (Alternative):**
```json
{
  "experimental": {
    "modelContextProtocolServer": {
      "transport": {
        "type": "stdio",
        "command": "npx",
        "args": ["-y", "mcp-remote", "http://localhost:8003/sse/"]
      }
    }
  }
}
```

**Location:** `~/.continue/config.json`

#### Windsurf IDE - SSE and STDIO Support

**Option 1: Direct SSE (Recommended):**
```json
{
  "mcpServers": {
    "maverick-mcp": {
      "serverUrl": "http://localhost:8003/sse/"
    }
  }
}
```

**Option 2: SSE via mcp-remote (Alternative):**
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

**Location:** Windsurf → Settings → Advanced Settings → MCP Servers

### How It Works

**Connection Architecture:**
- **STDIO Mode (Optimal for Claude Desktop)**: Direct subprocess communication - fastest, most reliable
- **Streamable-HTTP Endpoint**: `http://localhost:8003/` - For remote access via mcp-remote bridge
- **SSE Endpoint**: `http://localhost:8003/sse/` - For other clients with native SSE support

> **Key Finding**: Direct STDIO is the optimal transport for Claude Desktop. HTTP/SSE require the mcp-remote bridge tool, adding latency and complexity. SSE is particularly problematic as it's incompatible with mcp-remote (GET vs POST mismatch).

**Transport Limitations by Client:**
- **Claude Desktop**: STDIO-only, cannot directly connect to HTTP/SSE
- **Most Other Clients**: Support STDIO + SSE (but not HTTP)
- **Claude Code CLI**: Full transport support (STDIO, HTTP, SSE)

**mcp-remote Bridge Tool:**
- **Purpose**: Converts STDIO client calls to HTTP/SSE server requests
- **Why Needed**: Bridges the gap between STDIO-only clients and HTTP/SSE servers
- **Connection Flow**: Client (STDIO) ↔ mcp-remote ↔ HTTP/SSE Server
- **Installation**: `npx mcp-remote <server-url>`

**Key Transport Facts:**
- **STDIO**: All clients support this for local connections
- **HTTP**: Only Claude Code CLI supports direct HTTP connections
- **SSE**: Cursor, Continue.dev, Windsurf support direct SSE connections  
- **Claude Desktop Limitation**: Cannot connect to HTTP/SSE without mcp-remote bridge

**Alternatives for Remote Access:**
- Use Claude.ai web interface for native remote server support (no mcp-remote needed)

## Key Features

### Stock Analysis

- Historical price data with database caching
- Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- Support/resistance levels
- Volume analysis and patterns

### Stock Screening (Pre-seeded S&P 500 Data)

- **Maverick Bullish**: High momentum stocks with strong technicals from 520 S&P 500 stocks
- **Maverick Bearish**: Weak setups for short opportunities with pre-analyzed data
- **Supply/Demand Breakouts**: Stocks in confirmed uptrend phases with technical breakout patterns
- All screening data is pre-calculated and stored in database for instant results

### Portfolio Analysis

- Portfolio optimization using Modern Portfolio Theory
- Risk analysis and correlation matrices
- Performance metrics and comparisons

### Market Data

- Real-time quotes and market indices
- Sector performance analysis
- Economic indicators from FRED API

## Available Tools

All tools are organized into logical groups (35+ tools total):

### Data Tools (`/data/*`) - S&P 500 Pre-seeded

- `get_stock_data` - Historical price data with database caching
- `get_stock_info` - Company information from pre-seeded S&P 500 database
- `get_multiple_stocks_data` - Batch data fetching with optimized queries

### Technical Analysis (`/technical/*`)

- `calculate_sma`, `calculate_ema` - Moving averages
- `calculate_rsi` - Relative Strength Index
- `calculate_macd` - MACD indicator
- `calculate_bollinger_bands` - Bollinger Bands
- `get_full_technical_analysis` - Complete analysis suite

### Screening (`/screening/*`) - Pre-calculated Results

- `get_maverick_recommendations` - Bullish momentum stocks from S&P 500 database
- `get_maverick_bear_recommendations` - Bearish setups with pre-analyzed data
- `get_trending_breakout_recommendations` - Supply/demand breakout candidates from 520 stocks
- All screening results are pre-calculated and stored for instant access

### Advanced Research Tools (`/research/*`) - NEW AI-Powered Analysis

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

### Portfolio Analysis (`/portfolio/*`)

- `optimize_portfolio` - Portfolio optimization
- `analyze_portfolio_risk` - Risk assessment
- `calculate_correlation_matrix` - Asset correlations

### Market Data

- `get_market_overview` - Indices, sectors, market breadth
- `get_watchlist` - Sample portfolio with real-time data

## Development Commands

### Running the Server

```bash
# Development mode (recommended)
make dev                    # Uses Makefile for full setup

# Alternative direct commands
# Streamable-HTTP transport (FastMCP 2.0 standard - use with mcp-remote)
uv run python -m maverick_mcp.api.server --transport streamable-http --port 8003

# SSE transport (direct connections only, not mcp-remote)
uv run python -m maverick_mcp.api.server --transport sse --port 8003

# STDIO transport (development)
uv run python -m maverick_mcp.api.server  # Defaults to stdio

# Script-based startup
./scripts/dev.sh           # Includes additional setup
```

### Testing

```bash
# Quick testing
make test                  # Unit tests only (5-10 seconds)
make test-specific TEST=test_name  # Run specific test
make test-watch           # Auto-run on changes

# Using uv (recommended)
uv run pytest                    # Manual pytest execution
uv run pytest --cov=maverick_mcp # With coverage
uv run pytest -m integration    # Integration tests (requires PostgreSQL/Redis)

# Alternative: Direct pytest (if activated in venv)
pytest                    # Manual pytest execution
pytest --cov=maverick_mcp # With coverage
pytest -m integration    # Integration tests (requires PostgreSQL/Redis)
```

### Code Quality

```bash
# Automated quality checks
make format               # Auto-format with ruff
make lint                 # Check code quality with ruff
make typecheck            # Type check with ty (Astral's modern type checker)
make check                # Run all checks

# Using uv (recommended)
uv run ruff check .       # Linting
uv run ruff format .      # Formatting
uv run ty check .         # Type checking (Astral's modern type checker)

# Ultra-fast one-liner (no installation needed)
uvx ty check .            # Run ty directly without installing

# Alternative: Direct commands (if activated in venv)
ruff check .             # Linting
ruff format .            # Formatting
ty check .               # Type checking
```

## Configuration

### Database Options

**SQLite (Default - No Setup Required, includes S&P 500 data)**:

```bash
# Uses SQLite automatically with S&P 500 data seeding on first run
make dev
```

**PostgreSQL (Optional - Better Performance)**:

```bash
# In .env file
DATABASE_URL=postgresql://localhost/maverick_mcp

# Create database
createdb maverick_mcp
make migrate
```

### Caching Options

**No Caching (Default)**:

- Works out of the box, uses in-memory caching

**Redis Caching (Optional - Better Performance)**:

```bash
# Install and start Redis
brew install redis
brew services start redis

# Or use make command
make redis-start

# Server automatically detects Redis and uses it
```

## Code Guidelines

### General Principles

- Python 3.12+ with modern features
- Type hints for all functions
- Google-style docstrings for public APIs
- Comprehensive error handling
- Performance-first design with caching

### Financial Analysis

- Use pandas_ta for technical indicators
- Document all financial calculations
- Validate input data ranges
- Cache expensive computations
- Use vectorized operations for performance

### MCP Integration

- Register tools with `@mcp.tool()` decorator
- Return JSON-serializable results
- Implement graceful error handling
- Use database caching for persistence
- Follow FastMCP 2.0 patterns

## Troubleshooting

### Common Issues

**Server won't start**:

```bash
make stop          # Stop any running processes
make clean         # Clean temporary files
make dev           # Restart
```

**Port already in use**:

```bash
lsof -i :8003      # Find what's using port 8003
make stop          # Stop MaverickMCP services
```

**Redis connection errors** (optional):

```bash
brew services start redis    # Start Redis
# Or disable caching by not setting REDIS_HOST
```

**Database errors**:

```bash
# Use SQLite (no setup required)
unset DATABASE_URL
make dev

# Or fix PostgreSQL
createdb maverick_mcp
make migrate
```

**Claude Desktop not connecting**:

1. Verify server is running: `lsof -i :8003` (check if port 8003 is in use)
2. Check `claude_desktop_config.json` syntax and correct port (8003)
3. **Use the tested SSE configuration with trailing slash**: `http://localhost:8003/sse/` with mcp-remote
4. Restart Claude Desktop completely
5. Test with: "Get AAPL stock data"

**Tools appearing then disappearing**:

1. **CRITICAL FIX**: Ensure SSE endpoint has trailing slash: `http://localhost:8003/sse/`
2. The 307 redirect from `/sse` to `/sse/` causes tool registration to fail
3. **Use the recommended SSE configuration** (prevents this issue)
4. Ensure you're using the exact configuration shown above with trailing slash
5. **Avoid STDIO direct connections** for stable operation
6. The SSE + mcp-remote setup with trailing slash has been tested and prevents tool disappearing

**Research Tool Issues**:

1. **Timeouts**: Research tools have adaptive timeouts (120s-600s)
2. Deep research may take 2-10 minutes depending on complexity
3. Monitor progress in server logs with `make tail-log`
4. Ensure `OPENROUTER_API_KEY` and `EXA_API_KEY` are set for full functionality

**Missing S&P 500 screening data**:

```bash
# Manually seed S&P 500 database if needed
uv run python scripts/seed_sp500.py
```

### Performance Tips

- **Use Redis caching** for better performance
- **PostgreSQL over SQLite** for larger datasets
- **Parallel screening** is enabled by default (4x speedup)
- **Parallel research** achieves 7-256x speedup with agent orchestration
- **In-memory caching** reduces API calls
- **Smart model selection** reduces costs by 40-60% with OpenRouter

## Quick Testing

Test the server is working:

```bash
# Test server is running
lsof -i :8003

# Test MCP endpoint (after connecting with mcp-remote)
# Use Claude Desktop with: "List available tools"
```

## Recent Updates

### Advanced Research Agents (NEW - Major Feature Release)

- **Parallel Research Execution**: Achieved 7-256x speedup (exceeded 2x target) with intelligent agent orchestration
- **Adaptive Timeout Protection**: Dynamic timeouts (120s-600s) based on research depth and complexity
- **Intelligent Model Selection**: OpenRouter integration with 400+ models, 40-60% cost reduction
- **Comprehensive Error Handling**: Circuit breakers, retry logic, and graceful degradation
- **Early Termination**: Confidence-based stopping to optimize time and costs
- **Content Filtering**: High-credibility source prioritization for quality results
- **Multi-Agent Orchestration**: Supervisor pattern for complex research coordination
- **New Research Tools**: `research_comprehensive`, `research_company`, `analyze_market_sentiment`, `coordinate_agents`

### Performance Improvements

- **Parallel Agent Execution**: Increased concurrent agents from 4 to 6
- **Optimized Semaphores**: BoundedSemaphore for better resource management
- **Reduced Rate Limiting**: Delays decreased from 0.5s to 0.05s
- **Batch Processing**: Improved throughput for multiple research tasks
- **Smart Caching**: Redis-powered with in-memory fallback
- **Stock Screening**: 4x faster with parallel processing

### Testing & Quality

- **84% Test Coverage**: 93 tests with comprehensive coverage
- **Zero Linting Errors**: Fixed 947 issues for clean codebase
- **Full Type Annotations**: Complete type coverage for research components
- **Error Recovery Testing**: Comprehensive failure scenario coverage

### Personal Use Optimization

- **No Authentication/Billing**: Completely removed for personal use simplicity
- **Pre-seeded S&P 500 Database**: 520 stocks with comprehensive screening data on first startup
- **Simplified Architecture**: Clean, focused codebase without commercial complexity
- **Multi-Transport Support**: HTTP, SSE, and STDIO for all MCP clients
- **SQLite Default**: No database setup required, PostgreSQL optional for performance

### AI/LLM Integration

- **OpenRouter Integration**: Access to 400+ AI models with intelligent cost optimization
- **Smart Model Selection**: Automatic model selection based on task requirements (sentiment analysis, market research, technical analysis)
- **Cost-Efficient by Default**: Prioritizes cost-effectiveness while maintaining quality, 40-60% cost savings over premium-only approaches
- **Multiple Model Support**: Claude Opus 4.1, Claude Sonnet 4, Claude 3.5 Haiku, GPT-5, GPT-5 Nano, Gemini 2.5 Pro, DeepSeek R1, and more

### Developer Experience

- Comprehensive Makefile for all common tasks
- Smart error handling with automatic fix suggestions
- Hot reload development mode
- Extensive test suite with quick unit tests
- Type checking with ty (Astral's extremely fast type checker) for better IDE support

## Additional Resources

- **Architecture docs**: `docs/` directory
- **Test examples**: `tests/` directory
- **Development tools**: `tools/` directory
- **Example scripts**: `scripts/` directory

For detailed technical information and advanced usage, see the full documentation in the `docs/` directory.

---

**Note**: This project is designed for personal use. It provides powerful stock analysis tools for Claude Desktop with pre-seeded S&P 500 data, without the complexity of multi-user systems, authentication, or billing. The database automatically seeds with 520 S&P 500 stocks and screening recommendations on first startup.
