# Maverick-MCP Directory Structure

## Overview

**⚠️ IMPORTANT FINANCIAL DISCLAIMER**: This software is for educational and informational purposes only. It is NOT financial advice. Always consult with a qualified financial advisor before making investment decisions.

The Maverick-MCP package is organized into the following modules:

- **core/**: Core client functionality and rate limiting
- **api/**: MCP API server and client
- **tools/**: Financial analysis tools
- **providers/**: Data providers for stocks, market, and macro data
- **data/**: Data handling utilities, including caching
- **config/**: Configuration constants and settings
- **cli/**: Command-line interface tools
- **examples/**: Example scripts and usage patterns

## Module Details

### core/

- `client.py` - Base Anthropic client implementation with rate limiting
- `rate_limiter.py` - Anthropic API rate limiter

### api/

- `mcp_client.py` - MCP protocol client implementation
- `server.py` - FastMCP server implementation

### tools/

- `portfolio_manager.py` - Portfolio management and optimization tools

### providers/

- `stock_data.py` - Stock data provider utilities
- `market_data.py` - Market data provider utilities
- `macro_data.py` - Macroeconomic data provider utilities

### data/

- `cache.py` - Cache implementation (Redis and in-memory)

### config/

- `constants.py` - Configuration constants and environment variable handling

### cli/

- `server.py` - Server CLI implementation

### examples/

- Various example scripts showing how to use the Maverick-MCP tools

## Usage

**Personal Use Only**: This server is designed for individual educational use with Claude Desktop.

To start the Maverick-MCP server:

```bash
# Recommended: Use the Makefile
make dev

# Alternative: Direct FastMCP server
python -m maverick_mcp.api.server --transport streamable-http --port 8003

# Development mode with hot reload
./scripts/dev.sh
```

Note: The server will start using streamable-http transport on port 8003. The streamable-http transport is compatible with mcp-remote, while SSE transport is not (SSE requires GET requests but mcp-remote sends POST requests).

When the server starts, you can access it at:

- http://localhost:8003

You can also start the server programmatically:

```python
from maverick_mcp.api.server import mcp

# Start the server with SSE transport
# NOTE: All financial analysis tools include appropriate disclaimers
mcp.run(transport="sse")
```

## Financial Analysis Tools

MaverickMCP provides comprehensive financial analysis capabilities:

### Stock Data Tools
- Historical price data with intelligent caching
- Real-time quotes and market data
- Company information and fundamentals

### Technical Analysis Tools  
- 20+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Support and resistance level identification
- Trend analysis and pattern recognition

### Portfolio Tools
- Risk assessment and correlation analysis
- Portfolio optimization using Modern Portfolio Theory
- Position sizing and risk management

### Screening Tools
- Momentum-based stock screening
- Breakout pattern identification
- Custom filtering and ranking systems

**All tools include appropriate financial disclaimers and are for educational purposes only.**
