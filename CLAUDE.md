# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with the MaverickMCP codebase.

**🚀 QUICK START**: Run `make dev` to start the server. Connect with Claude Desktop using `mcp-remote`. See "Claude Desktop Setup" section below.

## Project Overview

MaverickMCP is a personal stock analysis MCP server built for Claude Desktop. It provides:

- Real-time and historical stock data access with intelligent caching
- Advanced technical analysis tools (RSI, MACD, Bollinger Bands, etc.)
- Multiple stock screening strategies (momentum, bearish, trending breakout)
- Portfolio optimization and correlation analysis
- Market and macroeconomic data integration
- SQLAlchemy-based database integration for persistent storage
- Redis caching for high performance (optional)
- Clean, simple architecture focused on stock analysis

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

   - **Python 3.11+**: Core runtime environment
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
   # Enhanced data providers (optional)
   FRED_API_KEY=your-fred-key

   # Database (optional - uses SQLite by default)
   DATABASE_URL=postgresql://localhost/maverick_mcp

   # Redis (optional - works without caching)
   REDIS_HOST=localhost
   REDIS_PORT=6379
   ```

   **Get a free Tiingo API key**: Sign up at [tiingo.com](https://tiingo.com) - free tier includes 500 requests/day.

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

**Important Limitation**: Claude Desktop only supports STDIO transport in its local configuration file. To connect to HTTP/SSE servers, you must use the `mcp-remote` bridge.

1. **Start the server**:

   ```bash
   make dev  # Server runs with both HTTP and SSE endpoints
   ```

2. **Configure Claude Desktop**:

   **Method A: HTTP Transport via mcp-remote (Recommended)**
   
   Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

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

   **Method B: SSE Transport via mcp-remote (Legacy)**
   
   For SSE endpoint:

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

   **Method C: Direct STDIO (Development - No HTTP Layer)**
   
   For development without HTTP server:

   ```json
   {
     "mcpServers": {
       "maverick-mcp": {
         "command": "uv",
         "args": ["run", "python", "-m", "maverick_mcp.api.server"],
         "cwd": "/path/to/maverick-mcp"
       }
     }
   }
   ```

   **Method D: Remote via Claude.ai (Alternative)**
   
   For native remote server support, use [Claude.ai web interface](https://claude.ai/settings/integrations) instead of Claude Desktop.

3. **Restart Claude Desktop** and test with: "Show me technical analysis for AAPL"

### Other Popular MCP Clients

> ⚠️ **Critical Transport Warning**: MCP clients have specific transport limitations. Using incorrect configurations will cause connection failures. Always verify which transports your client supports.

#### Transport Compatibility Matrix

| MCP Client           | STDIO | HTTP | SSE | Direct Config Support                         |
|----------------------|-------|------|-----|-----------------------------------------------|
| **Claude Desktop**   | ✅    | ❌   | ❌  | STDIO-only, requires mcp-remote for HTTP/SSE |
| **Cursor IDE**       | ✅    | ❌   | ✅  | STDIO and SSE supported                       |
| **Claude Code CLI**  | ✅    | ✅   | ✅  | All transports supported                      |
| **Continue.dev**     | ✅    | ❌   | ✅  | STDIO and SSE supported                       |
| **Windsurf IDE**     | ✅    | ❌   | ✅  | STDIO and SSE supported                       |

#### Claude Desktop (Most Commonly Used)

**⚠️ CRITICAL**: Claude Desktop ONLY supports STDIO transport. It cannot directly connect to HTTP or SSE endpoints.

**Configuration Location:**
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

**For HTTP Server (Recommended):**
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

**For Direct STDIO (Development Only):**
```json
{
  "mcpServers": {
    "maverick-mcp": {
      "command": "uv",
      "args": ["run", "python", "-m", "maverick_mcp.api.server", "--transport", "stdio"],
      "cwd": "/path/to/maverick-mcp"
    }
  }
}
```

**Restart Required:** Always restart Claude Desktop after config changes.

#### Cursor IDE - STDIO and SSE Support

**Option 1: STDIO via mcp-remote (Recommended):**
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

**Option 2: Direct SSE:**
```json
{
  "mcpServers": {
    "maverick-mcp": {
      "url": "http://localhost:8000/sse"
    }
  }
}
```

**Location:** Cursor → Settings → MCP Servers

#### Claude Code CLI - Full Transport Support

**HTTP Transport (Modern Standard):**
```bash
claude mcp add --transport http maverick-mcp http://localhost:8000/mcp
```

**SSE Transport (Legacy Compatibility):**
```bash
claude mcp add --transport sse maverick-mcp http://localhost:8000/sse
```

**STDIO Transport (Development):**
```bash
claude mcp add maverick-mcp uv run python -m maverick_mcp.api.server --transport stdio
```

#### Continue.dev - STDIO and SSE Support

**Option 1: STDIO via mcp-remote:**
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

**Option 2: Direct SSE:**
```json
{
  "mcpServers": {
    "maverick-mcp": {
      "url": "http://localhost:8000/sse"
    }
  }
}
```

**Location:** `~/.continue/config.json`

#### Windsurf IDE - STDIO and SSE Support

**Option 1: STDIO via mcp-remote:**
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

**Option 2: Direct SSE:**
```json
{
  "mcpServers": {
    "maverick-mcp": {
      "serverUrl": "http://localhost:8000/sse"
    }
  }
}
```

**Location:** Windsurf → Settings → Advanced Settings → MCP Servers

### How It Works

**Server Architecture:**
- **HTTP Endpoint** (Modern): `http://localhost:8000/mcp` - FastMCP 2.0 standard
- **SSE Endpoint** (Legacy): `http://localhost:8000/sse` - Server-Sent Events compatibility  
- **STDIO Mode**: Direct subprocess communication for development

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
- Host MCP server with proper authentication for web access

## Key Features

### Stock Analysis

- Historical price data with database caching
- Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- Support/resistance levels
- Volume analysis and patterns

### Stock Screening

- **Maverick Bullish**: High momentum stocks with strong technicals
- **Maverick Bearish**: Weak setups for short opportunities
- **Trending Breakout**: Stocks in confirmed uptrend phases

### Portfolio Analysis

- Portfolio optimization using Modern Portfolio Theory
- Risk analysis and correlation matrices
- Performance metrics and comparisons

### Market Data

- Real-time quotes and market indices
- Sector performance analysis
- Economic indicators from FRED API

## Available Tools

All tools are organized into logical groups:

### Data Tools (`/data/*`)

- `get_stock_data` - Historical price data
- `get_stock_info` - Company information
- `get_multiple_stocks_data` - Batch data fetching

### Technical Analysis (`/technical/*`)

- `calculate_sma`, `calculate_ema` - Moving averages
- `calculate_rsi` - Relative Strength Index
- `calculate_macd` - MACD indicator
- `calculate_bollinger_bands` - Bollinger Bands
- `get_full_technical_analysis` - Complete analysis suite

### Screening (`/screening/*`)

- `get_maverick_recommendations` - Bullish momentum stocks
- `get_maverick_bear_recommendations` - Bearish setups
- `get_trending_breakout_recommendations` - Breakout candidates

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
# HTTP transport (FastMCP 2.0 standard)
uv run python -m maverick_mcp.api.server --transport http --port 8000

# SSE transport (legacy compatibility)
uv run python -m maverick_mcp.api.server --transport sse --port 8000

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

**SQLite (Default - No Setup Required)**:

```bash
# Uses SQLite automatically - no configuration needed
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

- Python 3.11+ with modern features
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
lsof -i :8000      # Find what's using port 8000
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

1. Verify server is running: `lsof -i :8000` (check if port 8000 is in use)
2. Check `claude_desktop_config.json` syntax and correct port (8000)
3. Restart Claude Desktop completely
4. Test with: "Get AAPL stock data"

### Performance Tips

- **Use Redis caching** for better performance
- **PostgreSQL over SQLite** for larger datasets
- **Parallel screening** is enabled by default (4x speedup)
- **In-memory caching** reduces API calls

## Quick Testing

Test the server is working:

```bash
# Test server is running
lsof -i :8000

# Test MCP endpoint (after connecting with mcp-remote)
# Use Claude Desktop with: "List available tools"
```

## Recent Updates

### Personal Use Optimization

- Removed authentication system - no login required
- Removed credit/billing system - unlimited usage
- Simplified server architecture
- Focus on core stock analysis functionality
- Modern Claude Desktop integration via `mcp-remote`

### Performance Improvements

- 4x faster stock screening with parallel processing
- Smart caching with Redis fallback to memory
- Optimized database queries and indexes
- Fast startup options for development

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

**Note**: This project is designed for personal use. It provides powerful stock analysis tools for Claude Desktop without the complexity of multi-user systems, authentication, or billing.
