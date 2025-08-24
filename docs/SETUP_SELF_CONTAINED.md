# MaverickMCP Self-Contained Setup Guide

⚠️ **IMPORTANT FINANCIAL DISCLAIMER**: This software is for educational and informational purposes only. It is NOT financial advice. Past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions.

This guide explains how to set up MaverickMCP as a completely self-contained system for personal-use financial analysis with Claude Desktop.

## Overview

MaverickMCP is now fully self-contained and doesn't require any external database dependencies. All data is stored in its own PostgreSQL database with the `mcp_` prefix for all tables to avoid conflicts.

## Prerequisites

- Python 3.11+
- PostgreSQL 14+ (or SQLite for development)
- Redis (optional, for caching)
- Tiingo API account (free tier available)

## Quick Start

### 1. Clone and Install

```bash
# Clone the repository
git clone https://github.com/wshobson/maverick-mcp.git
cd maverick-mcp

# Install dependencies using uv (recommended)
uv sync

# Or use pip
pip install -e .
```

### 2. Configure Environment

Create a `.env` file with your configuration:

```bash
# Database Configuration (MaverickMCP's own database)
MCP_DATABASE_URL=postgresql://user:password@localhost/maverick_mcp
# Or use SQLite for development
# MCP_DATABASE_URL=sqlite:///maverick_mcp.db

# Tiingo API Configuration (required for data loading)
TIINGO_API_TOKEN=your_tiingo_api_token_here

# Redis Configuration (optional)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
CACHE_ENABLED=true

# Personal Use Configuration
# Note: Authentication and billing systems have been removed for simplicity
# This version is designed for local personal use only
```

### 3. Create Database

```bash
# Create PostgreSQL database
createdb maverick_mcp

# Or use existing PostgreSQL
psql -U postgres -c "CREATE DATABASE maverick_mcp;"
```

### 4. Run Migrations

```bash
# Initialize Alembic (if not already done)
alembic init alembic

# Run all migrations to create schema
alembic upgrade head

# Verify migration
alembic current
```

The migration creates the following self-contained tables:
- `mcp_stocks` - Master stock information
- `mcp_price_cache` - Historical price data
- `mcp_maverick_stocks` - Maverick screening results
- `mcp_maverick_bear_stocks` - Bear market screening
- `mcp_supply_demand_breakouts` - Supply/demand analysis
- `mcp_technical_cache` - Technical indicator cache
- `mcp_users`, `mcp_api_keys`, etc. - Authentication tables

### 5. Load Initial Data

#### Option A: Quick Start (Top 10 S&P 500)

```bash
# Load 2 years of data for top 10 S&P 500 stocks
python scripts/load_tiingo_data.py \
    --symbols AAPL MSFT GOOGL AMZN NVDA META TSLA LLY V UNH \
    --years 2 \
    --calculate-indicators \
    --run-screening
```

#### Option B: Full S&P 500

```bash
# Load S&P 500 stocks with screening
python scripts/load_tiingo_data.py \
    --sp500 \
    --years 2 \
    --calculate-indicators \
    --run-screening
```

#### Option C: Custom Symbols

```bash
# Load specific symbols with custom date range
python scripts/load_tiingo_data.py \
    --symbols AAPL MSFT GOOGL \
    --start-date 2022-01-01 \
    --end-date 2024-01-01 \
    --calculate-indicators \
    --run-screening
```

### 6. Start the Server

```bash
# Recommended: Use the Makefile
make dev

# Alternative: Direct FastMCP server
python -m maverick_mcp.api.server --transport sse --port 8000

# Development mode with hot reload
./scripts/dev.sh
```

### 7. Verify Setup

```bash
# Check health endpoint
curl http://localhost:8000/health

# Test a simple query (if using MCP client)
echo '{"method": "tools/list"}' | nc localhost 8000

# Or use the API directly
curl http://localhost:8000/api/data/stock/AAPL
```

## Database Schema

The self-contained schema uses the `mcp_` prefix for all tables:

```sql
-- Stock Data Tables
mcp_stocks                  -- Master stock information
├── stock_id (UUID PK)
├── ticker_symbol
├── company_name
├── sector
└── ...

mcp_price_cache            -- Historical OHLCV data
├── price_cache_id (UUID PK)
├── stock_id (FK -> mcp_stocks)
├── date
├── open_price, high_price, low_price, close_price
└── volume

-- Screening Tables
mcp_maverick_stocks        -- Momentum screening
├── id (PK)
├── stock_id (FK -> mcp_stocks)
├── combined_score
└── technical indicators...

mcp_maverick_bear_stocks   -- Bear screening
├── id (PK)
├── stock_id (FK -> mcp_stocks)
├── score
└── bearish indicators...

mcp_supply_demand_breakouts -- Supply/demand analysis
├── id (PK)
├── stock_id (FK -> mcp_stocks)
├── momentum_score
└── accumulation metrics...

mcp_technical_cache        -- Flexible indicator storage
├── id (PK)
├── stock_id (FK -> mcp_stocks)
├── indicator_type
└── values...
```

## Data Loading Details

### Rate Limiting

Tiingo API has a rate limit of 2400 requests/hour. The loader automatically handles this:

```python
# Configure in load_tiingo_data.py
MAX_CONCURRENT_REQUESTS = 5  # Parallel requests
RATE_LIMIT_DELAY = 1.5       # Seconds between requests
```

### Resume Capability

The loader saves progress and can resume interrupted loads:

```bash
# Start loading (creates checkpoint file)
python scripts/load_tiingo_data.py --sp500 --years 2

# If interrupted, resume from checkpoint
python scripts/load_tiingo_data.py --resume
```

### Technical Indicators Calculated

- **Trend**: SMA (20, 50, 150, 200), EMA (21)
- **Momentum**: RSI, MACD, Relative Strength Rating
- **Volatility**: ATR, ADR (Average Daily Range)
- **Volume**: 30-day average, volume ratio

### Screening Algorithms

1. **Maverick Screening** (Momentum)
   - Price > EMA21 > SMA50 > SMA200
   - Momentum Score > 70
   - Combined score calculation

2. **Bear Screening** (Weakness)
   - Price < EMA21 < SMA50
   - Momentum Score < 30
   - Negative MACD

3. **Supply/Demand Screening** (Accumulation)
   - Price > SMA50 > SMA150 > SMA200
   - Momentum Score > 80
   - Volume confirmation

## Troubleshooting

### Database Connection Issues

```bash
# Test PostgreSQL connection
psql -U user -d maverick_mcp -c "SELECT 1;"

# Use SQLite for testing
export MCP_DATABASE_URL=sqlite:///test.db
```

### Tiingo API Issues

```bash
# Test Tiingo API token
curl -H "Authorization: Token YOUR_TOKEN" \
     "https://api.tiingo.com/api/test"

# Check rate limit status
curl -H "Authorization: Token YOUR_TOKEN" \
     "https://api.tiingo.com/account/usage"
```

### Migration Issues

```bash
# Check current migration
alembic current

# Show migration history
alembic history

# Downgrade if needed
alembic downgrade -1

# Re-run specific migration
alembic upgrade 010_self_contained_schema
```

### Data Loading Issues

```bash
# Check checkpoint file
cat tiingo_load_progress.json

# Clear checkpoint to start fresh
rm tiingo_load_progress.json

# Load single symbol for testing
python scripts/load_tiingo_data.py \
    --symbols AAPL \
    --years 1 \
    --calculate-indicators
```

## Performance Optimization

### Database Indexes

The migration creates optimized indexes for common queries:

```sql
-- Price data lookups
CREATE INDEX mcp_price_cache_stock_id_date_idx 
ON mcp_price_cache(stock_id, date);

-- Screening queries
CREATE INDEX mcp_maverick_stocks_combined_score_idx 
ON mcp_maverick_stocks(combined_score DESC);

-- Supply/demand filtering
CREATE INDEX mcp_supply_demand_breakouts_ma_filter_idx 
ON mcp_supply_demand_breakouts(close_price, sma_50, sma_150, sma_200);
```

### Connection Pooling

Configure in `.env`:

```bash
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=10
DB_POOL_TIMEOUT=30
DB_POOL_RECYCLE=3600
```

### Caching with Redis

Enable Redis caching for better performance:

```bash
CACHE_ENABLED=true
CACHE_TTL_SECONDS=300
REDIS_HOST=localhost
REDIS_PORT=6379
```

## Personal Use Deployment

### 1. Use Local Database

```bash
# Use SQLite for simplicity
MCP_DATABASE_URL=sqlite:///maverick_mcp.db

# Or PostgreSQL for better performance
MCP_DATABASE_URL=postgresql://user:password@localhost/maverick_mcp
```

### 2. Connect with Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

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

### 5. Set Up Daily Data Updates

Create a cron job for daily updates:

```bash
# Add to crontab
0 1 * * * /path/to/venv/bin/python /path/to/scripts/load_tiingo_data.py \
    --sp500 --years 0.1 --calculate-indicators --run-screening
```

## API Usage Examples

### Fetch Stock Data

```python
from maverick_mcp.data.models import Stock, PriceCache, get_db

# Get historical data
with get_db() as session:
    df = PriceCache.get_price_data(
        session, 
        "AAPL", 
        "2023-01-01", 
        "2024-01-01"
    )
```

### Run Screening

```python
from maverick_mcp.data.models import MaverickStocks

# Get top momentum stocks
with get_db() as session:
    top_stocks = MaverickStocks.get_top_stocks(session, limit=20)
    for stock in top_stocks:
        print(f"{stock.stock}: Score {stock.combined_score}")
```

### Using MCP Tools

```bash
# List available tools
curl -X POST http://localhost:8000/mcp \
    -H "Content-Type: application/json" \
    -d '{"method": "tools/list"}'

# Get screening results
curl -X POST http://localhost:8000/mcp \
    -H "Content-Type: application/json" \
    -d '{
        "method": "tools/call",
        "params": {
            "name": "get_maverick_stocks",
            "arguments": {"limit": 10}
        }
    }'
```

## Next Steps

1. **Load More Data**: Expand beyond S&P 500 to Russell 3000
2. **Add More Indicators**: Implement additional technical indicators
3. **Custom Screening**: Create your own screening algorithms
4. **Web Dashboard**: Deploy the maverick-mcp-web frontend
5. **API Integration**: Build applications using the MCP protocol

## Support

- GitHub Issues: [Report bugs or request features](https://github.com/wshobson/maverick-mcp/issues)
- Discussions: [Join community discussions](https://github.com/wshobson/maverick-mcp/discussions)
- Documentation: [Read the full docs](https://github.com/wshobson/maverick-mcp)

---

*MaverickMCP is now completely self-contained and ready for deployment!*