# MaverickMCP Database Setup

This guide explains how to set up and seed the SQLite database for MaverickMCP with sample stock data.

## Quick Start

### 1. Run Complete Setup (Recommended)

```bash
# Set your database URL (optional - defaults to SQLite)
export DATABASE_URL=sqlite:///maverick_mcp.db

# Run the complete setup script
./scripts/setup_database.sh
```

This will:

- ✅ Create SQLite database with all tables
- ✅ Seed with 40 sample stocks (AAPL, MSFT, GOOGL, etc.)
- ✅ Populate with 1,370+ price records
- ✅ Generate sample screening results (Maverick, Bear, Supply/Demand)
- ✅ Add technical indicators cache

### 2. Manual Step-by-Step Setup

```bash
# Step 1: Create database tables
python scripts/migrate_db.py

# Step 2: Seed with sample data (no API key required)
python scripts/seed_db.py

# Step 3: Test the setup
python scripts/test_seeded_data.py
```

## Database Configuration

### Default Configuration (SQLite)

- **Database**: `sqlite:///maverick_mcp.db`
- **Location**: Project root directory
- **No setup required**: Works out of the box

### PostgreSQL (Optional)

```bash
# Set environment variable
export DATABASE_URL=postgresql://localhost/maverick_mcp

# Create PostgreSQL database
createdb maverick_mcp

# Run migration
python scripts/migrate_db.py
```

## Sample Data Overview

### Stocks Included (40 total)

- **Large Cap**: AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, BRK-B, JNJ, V
- **Growth**: AMD, CRM, SHOP, ROKU, ZM, DOCU, SNOW, PLTR, RBLX, U
- **Value**: KO, PFE, XOM, CVX, JPM, BAC, WMT, PG, T, VZ
- **Small Cap**: UPST, SOFI, OPEN, WISH, CLOV, SPCE, LCID, RIVN, BYND, PTON

### Generated Data

- **1,370+ Price Records**: 200 days of historical data for 10 stocks
- **24 Maverick Stocks**: Bullish momentum recommendations
- **16 Bear Stocks**: Bearish setups with technical indicators
- **16 Supply/Demand Breakouts**: Accumulation breakout candidates
- **600 Technical Indicators**: RSI, SMA cache for analysis

## Testing MCP Tools

After seeding, test that the screening tools work:

```bash
python scripts/test_seeded_data.py
```

Expected output:

```
✅ Found 5 Maverick recommendations
  1. PTON - Score: 100
  2. BYND - Score: 100
  3. RIVN - Score: 100

✅ Found 5 Bear recommendations
  1. MSFT - Score: 37
  2. JNJ - Score: 32
  3. TSLA - Score: 32

✅ Total screening results across all categories: 56
```

## Using with Claude Desktop

After database setup, start the MCP server:

```bash
# Start the server
make dev

# Or manually
uvicorn maverick_mcp.api.server:app --host 0.0.0.0 --port 8003
```

Then connect with Claude Desktop using `mcp-remote`:

```json
{
  "mcpServers": {
    "maverick-mcp": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://localhost:8003/mcp"]
    }
  }
}
```

Test with prompts like:

- "Show me the top maverick stock recommendations"
- "Get technical analysis for AAPL"
- "Find bearish stocks with high RSI"

## Database Schema

### Core Tables

- **mcp_stocks**: Stock symbols and company information
- **mcp_price_cache**: Historical OHLCV price data
- **mcp_technical_cache**: Calculated technical indicators

### Screening Tables

- **mcp_maverick_stocks**: Bullish momentum screening results
- **mcp_maverick_bear_stocks**: Bearish setup screening results
- **mcp_supply_demand_breakouts**: Breakout pattern screening results

## Troubleshooting

### Database Connection Issues

```bash
# Check database exists
ls -la maverick_mcp.db

# Test SQLite connection
sqlite3 maverick_mcp.db "SELECT COUNT(*) FROM mcp_stocks;"
```

### No Screening Results

```bash
# Verify data was seeded
sqlite3 maverick_mcp.db "
SELECT
  (SELECT COUNT(*) FROM mcp_stocks) as stocks,
  (SELECT COUNT(*) FROM mcp_price_cache) as prices,
  (SELECT COUNT(*) FROM mcp_maverick_stocks) as maverick;
"
```

### MCP Server Connection

```bash
# Check server is running
curl http://localhost:8003/health

# Check MCP endpoint
curl http://localhost:8003/mcp/capabilities
```

## Advanced Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=sqlite:///maverick_mcp.db

# Optional: Enable debug logging
LOG_LEVEL=debug

# Optional: Redis caching
REDIS_HOST=localhost
REDIS_PORT=6379
```

### Custom Stock Lists

Edit `scripts/seed_db.py` and modify `SAMPLE_STOCKS` to include your preferred stock symbols.

### Production Setup

- Use PostgreSQL for better performance
- Enable Redis caching
- Set up proper logging
- Configure rate limiting

---

✅ **Database ready!** Your MaverickMCP instance now has a complete SQLite database with sample stock data and screening results.
