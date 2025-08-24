# Tiingo Data Loader Installation Guide

This guide will help you set up and use the comprehensive Tiingo data loader for Maverick-MCP.

## 📋 What You Get

The Tiingo data loader provides:

- **Comprehensive Data Loading**: Fetch stock metadata, OHLCV price data from Tiingo API
- **Technical Indicators**: 50+ indicators calculated using pandas-ta
- **Screening Algorithms**: Built-in Maverick, Bear Market, and Supply/Demand screens
- **Progress Tracking**: Resume interrupted loads with checkpoint files
- **Performance Optimized**: Async operations with rate limiting and batch processing
- **Production Ready**: Error handling, logging, and database optimization

## 🚀 Quick Start

### 1. Check Your Setup
```bash
cd /path/to/maverick-mcp
python3 scripts/validate_setup.py
```

This will show you exactly what needs to be installed or configured.

### 2. Install Dependencies
```bash
# Install required Python packages
pip install -r scripts/requirements_tiingo.txt

# Or install individually:
pip install aiohttp pandas pandas-ta sqlalchemy psycopg2-binary
```

### 3. Get Tiingo API Token
1. Sign up at [tiingo.com](https://www.tiingo.com) (free tier gives 2400 requests/hour)
2. Get your API token from the dashboard
3. Set environment variable:
```bash
export TIINGO_API_TOKEN=your_token_here
```

### 4. Configure Database
```bash
# Set your database URL
export DATABASE_URL=postgresql://user:password@localhost/maverick_mcp

# Or use existing environment variables
export POSTGRES_URL=postgresql://user:password@localhost/maverick_mcp
```

### 5. Verify Setup
```bash
python3 scripts/validate_setup.py
```
You should see "🎉 Setup validation PASSED!"

## 📊 Usage Examples

### Load Sample Stocks
```bash
# Load 5 popular stocks with 2 years of data
python3 scripts/load_tiingo_data.py --symbols AAPL,MSFT,GOOGL,AMZN,TSLA --years 2 --calculate-indicators
```

### Load S&P 500 (Top 100)
```bash
# Load top 100 S&P 500 stocks with screening
python3 scripts/load_tiingo_data.py --sp500 --years 1 --run-screening
```

### Load from File
```bash
# Create symbol file
echo -e "AAPL\nMSFT\nGOOGL\nTSLA\nNVDA" > my_stocks.txt

# Load from file
python3 scripts/load_tiingo_data.py --file my_stocks.txt --calculate-indicators --run-screening
```

### Interactive Examples
```bash
# Run guided examples
python3 scripts/load_example.py
```

## 🏗️ Architecture

### Files Created
- **`load_tiingo_data.py`**: Main comprehensive data loader script
- **`tiingo_config.py`**: Configuration settings and symbol lists  
- **`load_example.py`**: Interactive examples and tutorials
- **`validate_setup.py`**: Setup validation and dependency checking
- **`test_tiingo_loader.py`**: Unit tests and validation
- **`requirements_tiingo.txt`**: Python package requirements
- **`README_TIINGO_LOADER.md`**: Comprehensive documentation

### Data Flow
1. **Fetch Metadata**: Get stock information from Tiingo
2. **Load Prices**: Download historical OHLCV data
3. **Calculate Indicators**: Compute 50+ technical indicators
4. **Store Data**: Bulk insert into Maverick-MCP database tables
5. **Run Screens**: Execute screening algorithms
6. **Track Progress**: Save checkpoints for resume capability

### Database Tables
- **`mcp_stocks`**: Basic stock information
- **`mcp_price_cache`**: Historical OHLCV price data  
- **`mcp_technical_cache`**: Calculated technical indicators
- **`mcp_maverick_stocks`**: Momentum screening results
- **`mcp_maverick_bear_stocks`**: Bear market screening results
- **`mcp_supply_demand_breakouts`**: Supply/demand pattern results

## ⚙️ Configuration Options

### Environment Variables
```bash
# Required
export TIINGO_API_TOKEN=your_token
export DATABASE_URL=postgresql://user:pass@localhost/db

# Optional
export DB_POOL_SIZE=20
export DB_ECHO=false
export ENVIRONMENT=development
```

### Symbol Sources
- **S&P 500**: `--sp500` (top 100) or `--sp500-full` (all 500)
- **Custom**: `--symbols AAPL,MSFT,GOOGL`
- **File**: `--file symbols.txt`
- **All Supported**: `--supported` (3000+ symbols)

### Performance Tuning
- **Batch Size**: `--batch-size 100` (default: 50)
- **Concurrency**: `--max-concurrent 10` (default: 5)
- **Date Range**: `--years 5` or `--start-date 2020-01-01`

### Processing Options
- **Technical Indicators**: `--calculate-indicators` (default: on)
- **Screening**: `--run-screening` (run after data load)
- **Resume**: `--resume` (continue from checkpoint)

## 📈 Technical Indicators

### Trend Indicators
- Simple Moving Averages (SMA 20, 50, 150, 200)
- Exponential Moving Average (EMA 21)
- Average Directional Index (ADX 14)

### Momentum Indicators  
- Relative Strength Index (RSI 14)
- MACD (12, 26, 9)
- Stochastic Oscillator (14, 3, 3)
- Relative Strength Rating vs Market

### Volatility Indicators
- Average True Range (ATR 14)
- Bollinger Bands (20, 2.0)
- Average Daily Range percentage

### Volume Indicators
- Volume Moving Averages
- Volume Ratio vs Average
- Volume-Weighted Average Price (VWAP)

### Custom Indicators
- Price Momentum (10, 20 period)
- Bollinger Band Squeeze Detection
- Position vs Moving Averages

## 🔍 Screening Algorithms

### Maverick Momentum Screen
**Criteria:**
- Price > 21-day EMA
- EMA-21 > SMA-50  
- SMA-50 > SMA-200
- Relative Strength > 70
- Volume > 500K daily

**Scoring:** 0-10 points based on strength of signals

### Bear Market Screen
**Criteria:**
- Price < 21-day EMA
- EMA-21 < SMA-50
- Relative Strength < 30
- High volume on declines

**Use Case:** Short candidates or stocks to avoid

### Supply/Demand Breakout Screen  
**Criteria:**
- Price > SMA-50 and SMA-200
- Strong relative strength (>60)
- Accumulation patterns
- Institutional buying signals

**Use Case:** Stocks with strong fundamental demand

## 🚨 Troubleshooting

### Common Issues

#### 1. Missing Dependencies
```bash
# Error: ModuleNotFoundError: No module named 'aiohttp'
pip install -r scripts/requirements_tiingo.txt
```

#### 2. API Rate Limiting
```bash
# Reduce concurrency if getting rate limited
python3 scripts/load_tiingo_data.py --symbols AAPL --max-concurrent 2
```

#### 3. Database Connection Issues
```bash
# Test database connection
python3 -c "
from maverick_mcp.data.models import SessionLocal
with SessionLocal() as session:
    print('Database connection OK')
"
```

#### 4. Memory Issues
```bash
# Reduce batch size for large loads
python3 scripts/load_tiingo_data.py --sp500 --batch-size 25 --max-concurrent 3
```

#### 5. Checkpoint File Corruption
```bash
# Remove corrupted checkpoint and restart
rm load_progress.json
python3 scripts/load_tiingo_data.py --symbols AAPL,MSFT
```

### Getting Help
1. **Validation Script**: `python3 scripts/validate_setup.py`
2. **Check Logs**: `tail -f tiingo_data_loader.log`
3. **Test Individual Components**: `python3 scripts/test_tiingo_loader.py`
4. **Interactive Examples**: `python3 scripts/load_example.py`

## 🎯 Best Practices

### For Development
```bash
# Start small for testing
python3 scripts/load_tiingo_data.py --symbols AAPL,MSFT --years 0.5 --batch-size 10

# Use checkpoints for large loads
python3 scripts/load_tiingo_data.py --sp500 --checkpoint-file dev_progress.json
```

### For Production
```bash
# Higher performance settings
python3 scripts/load_tiingo_data.py --sp500-full \
    --batch-size 100 \
    --max-concurrent 10 \
    --years 2 \
    --run-screening

# Schedule regular updates
# Add to crontab: 0 18 * * 1-5 /path/to/load_script.sh
```

### For Resume Operations
```bash
# Always use checkpoints for large operations
python3 scripts/load_tiingo_data.py --supported --checkpoint-file full_load.json

# If interrupted, resume with:
python3 scripts/load_tiingo_data.py --resume --checkpoint-file full_load.json
```

## 📊 Performance Benchmarks

**Typical Loading Times (on modern hardware):**
- 10 symbols, 1 year: 2-3 minutes
- 100 symbols, 2 years: 15-20 minutes  
- 500 symbols, 2 years: 1-2 hours
- 3000+ symbols, 2 years: 6-12 hours

**Rate Limits:**
- Tiingo Free: 2400 requests/hour
- Recommended: 5 concurrent requests max
- With indicators: ~1.5 seconds per symbol

## 🔗 Integration

### With Maverick-MCP API
The loaded data is immediately available through:
- `/api/v1/stocks` - Stock metadata
- `/api/v1/prices/{symbol}` - Price data
- `/api/v1/technical/{symbol}` - Technical indicators
- `/api/v1/screening/*` - Screening results

### With MCP Tools
- `get_stock_analysis` - Uses loaded data
- `run_screening` - Operates on cached data
- `portfolio_analysis` - Leverages technical indicators

### Custom Workflows
```python
# Example: Load data then run custom analysis
from scripts.load_tiingo_data import TiingoDataLoader
from maverick_mcp.data.models import SessionLocal

async with TiingoDataLoader() as loader:
    await loader.load_symbol_data("AAPL", "2023-01-01")

with SessionLocal() as session:
    # Your custom analysis here
    pass
```

## 🎉 Success!

Once setup is complete, you should be able to:

1. ✅ Load market data from Tiingo efficiently
2. ✅ Calculate comprehensive technical indicators  
3. ✅ Run sophisticated screening algorithms
4. ✅ Resume interrupted loads seamlessly
5. ✅ Access all data through Maverick-MCP APIs
6. ✅ Build custom trading strategies

**Next Steps:**
- Explore the interactive examples: `python3 scripts/load_example.py`
- Read the full documentation: `scripts/README_TIINGO_LOADER.md`
- Set up automated daily updates
- Customize screening algorithms for your strategy