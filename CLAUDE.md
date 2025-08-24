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

**Claude Desktop** (Most Commonly Used)
Claude Desktop is the most popular MCP client but has STDIO-only limitations:

**Configuration Location:**
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

**Connection Options:**
```json
{
  "mcpServers": {
    "maverick-mcp-http": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://localhost:8000/mcp"]
    },
    "maverick-mcp-sse": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://localhost:8000/sse"]
    },
    "maverick-mcp-direct": {
      "command": "uv",
      "args": ["run", "python", "-m", "maverick_mcp.api.server"],
      "cwd": "/path/to/maverick-mcp"
    }
  }
}
```

**Restart Required:** Always restart Claude Desktop after config changes.

**Cursor IDE**
Cursor has native MCP support through its settings:

1. Open Cursor → Settings → MCP Servers
2. Add server configuration:
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
3. Restart Cursor to apply changes

**Claude Code (CLI Tool)**
Anthropic's official CLI tool with excellent MCP support:

```bash
# Add HTTP server (recommended)
claude mcp add --transport http maverick-mcp http://localhost:8000/mcp

# Add SSE server (legacy)
claude mcp add --transport sse maverick-mcp http://localhost:8000/sse

# Add direct STDIO server (development)
claude mcp add --scope user maverick-stdio \
  uv run python -m maverick_mcp.api.server

# List configured servers
claude mcp list
```

Or manually edit `~/.claude.json`:
```json
{
  "mcpServers": {
    "maverick-mcp": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://localhost:8000/mcp"],
      "env": {}
    }
  }
}
```

**Continue.dev (VS Code Extension)**
Continue was the first MCP client with full protocol support:

1. Install Continue extension in VS Code
2. Edit `~/.continue/config.json`:
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
3. Use `@MCP` in Continue chat to access tools

**Windsurf IDE**
Similar to Cursor, with native MCP support:
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

**Key Transport Notes:**
- Most clients support STDIO transport only in local config
- HTTP/SSE servers require `mcp-remote` bridge for most clients
- Claude Code CLI has the most comprehensive transport support
- Continue.dev offers excellent MCP integration with VS Code

### How It Works

- **MCP Server**: Runs locally with multiple endpoints:
  - HTTP (FastMCP 2.0 standard): `http://localhost:8000/mcp`
  - SSE (legacy compatibility): `http://localhost:8000/sse`
  - STDIO: Direct connection for development
- **mcp-remote**: Third-party bridge tool that connects STDIO-only clients to HTTP/SSE servers
- **Claude Desktop Limitation**: Only supports STDIO transport in local config
- **Connection Flow**: MCP Client (STDIO) ↔ mcp-remote ↔ HTTP/SSE Server
- **Client Support**: Multiple MCP clients available (Claude Desktop, Cursor, Claude Code, Continue.dev, Windsurf)
- **Transport Limitation**: Most clients only support STDIO locally, requiring mcp-remote for HTTP/SSE
- **Alternative**: Use Claude.ai web interface for native remote server support
- **No authentication**: Simple personal use - no login required

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
