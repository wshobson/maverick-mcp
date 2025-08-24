# Tiingo Data Loader for Maverick-MCP

A comprehensive, production-ready data loader for fetching market data from Tiingo API and storing it in the Maverick-MCP database with technical indicators and screening algorithms.

## Features

### 🚀 Core Capabilities
- **Async Operations**: High-performance async data fetching with configurable concurrency
- **Rate Limiting**: Built-in rate limiting to respect Tiingo's 2400 requests/hour limit
- **Progress Tracking**: Resume capability with checkpoint files for interrupted loads
- **Error Handling**: Exponential backoff retry logic with comprehensive error handling
- **Batch Processing**: Efficient batch processing with configurable batch sizes

### 📊 Data Processing
- **Technical Indicators**: 50+ technical indicators using pandas-ta
- **Screening Algorithms**: Built-in Maverick, Bear Market, and Supply/Demand screens
- **Database Optimization**: Bulk inserts with connection pooling
- **Data Validation**: Comprehensive data validation and cleaning

### 🎛️ Flexible Configuration
- **Multiple Symbol Sources**: S&P 500, custom files, individual symbols, or all Tiingo-supported tickers
- **Date Range Control**: Configurable date ranges with year-based shortcuts
- **Processing Options**: Enable/disable technical indicators and screening
- **Performance Tuning**: Adjustable concurrency, batch sizes, and timeouts

## Installation

### Prerequisites
1. **Tiingo API Token**: Sign up at [tiingo.com](https://www.tiingo.com) and get your API token
2. **Database**: Ensure Maverick-MCP database is set up and accessible
3. **Python Dependencies**: pandas-ta, aiohttp, SQLAlchemy, and other requirements

### Setup
```bash
# Set your Tiingo API token
export TIINGO_API_TOKEN=your_token_here

# Set database URL (if different from default)
export DATABASE_URL=postgresql://user:pass@localhost/maverick_mcp

# Make scripts executable
chmod +x scripts/load_tiingo_data.py
chmod +x scripts/load_example.py
```

## Usage Examples

### 1. Load S&P 500 Stocks (Top 100)
```bash
# Load 2 years of data with technical indicators
python scripts/load_tiingo_data.py --sp500 --years 2 --calculate-indicators

# Load with screening algorithms
python scripts/load_tiingo_data.py --sp500 --years 1 --run-screening
```

### 2. Load Specific Symbols
```bash
# Load individual stocks
python scripts/load_tiingo_data.py --symbols AAPL,MSFT,GOOGL,AMZN,TSLA --years 3

# Load with custom date range
python scripts/load_tiingo_data.py --symbols AAPL,MSFT --start-date 2020-01-01 --end-date 2023-12-31
```

### 3. Load from File
```bash
# Create a symbol file
echo -e "AAPL\nMSFT\nGOOGL\nAMZN\nTSLA" > my_symbols.txt

# Load from file
python scripts/load_tiingo_data.py --file my_symbols.txt --calculate-indicators --run-screening
```

### 4. Resume Interrupted Load
```bash
# If a load was interrupted, resume from checkpoint
python scripts/load_tiingo_data.py --resume --checkpoint-file load_progress.json
```

### 5. Performance-Optimized Load
```bash
# High-performance loading with larger batches and more concurrency
python scripts/load_tiingo_data.py --sp500 --batch-size 100 --max-concurrent 10 --no-checkpoint
```

### 6. All Supported Tickers
```bash
# Load all Tiingo-supported symbols (this will take a while!)
python scripts/load_tiingo_data.py --supported --batch-size 50 --max-concurrent 8
```

## Command Line Options

### Symbol Selection (Required - choose one)
- `--symbols AAPL,MSFT,GOOGL` - Comma-separated list of symbols
- `--file symbols.txt` - Load symbols from file (one per line or comma-separated)
- `--sp500` - Load S&P 500 symbols (top 100 most liquid)
- `--sp500-full` - Load full S&P 500 (500 symbols)
- `--supported` - Load all Tiingo-supported symbols
- `--resume` - Resume from checkpoint file

### Date Range Options
- `--years 2` - Number of years of historical data (default: 2)
- `--start-date 2020-01-01` - Custom start date (YYYY-MM-DD)
- `--end-date 2023-12-31` - Custom end date (YYYY-MM-DD, default: today)

### Processing Options
- `--calculate-indicators` - Calculate technical indicators (default: True)
- `--no-indicators` - Skip technical indicator calculations
- `--run-screening` - Run screening algorithms after data loading

### Performance Options
- `--batch-size 50` - Batch size for processing (default: 50)
- `--max-concurrent 5` - Maximum concurrent requests (default: 5)

### Database Options
- `--create-tables` - Create database tables if they don't exist
- `--database-url` - Override database URL

### Progress Tracking
- `--checkpoint-file load_progress.json` - Checkpoint file location
- `--no-checkpoint` - Disable checkpoint saving

## Configuration

The loader can be customized through the `tiingo_config.py` file:

```python
from scripts.tiingo_config import TiingoConfig, get_config_for_environment

# Get environment-specific config
config = get_config_for_environment('production')

# Customize settings
config.max_concurrent_requests = 10
config.default_batch_size = 100
config.maverick_min_momentum_score = 80.0
```

### Available Configurations
- **Rate Limiting**: Requests per hour, retry settings
- **Technical Indicators**: Periods for RSI, SMA, EMA, MACD, etc.
- **Screening Criteria**: Minimum momentum scores, volume thresholds
- **Database Settings**: Batch sizes, connection pooling
- **Symbol Lists**: Predefined lists for different strategies

## Technical Indicators Calculated

The loader calculates 50+ technical indicators including:

### Trend Indicators
- **SMA**: 20, 50, 150, 200-period Simple Moving Averages
- **EMA**: 21-period Exponential Moving Average
- **ADX**: Average Directional Index (14-period)

### Momentum Indicators
- **RSI**: Relative Strength Index (14-period)
- **MACD**: Moving Average Convergence Divergence (12,26,9)
- **Stochastic**: Stochastic Oscillator (14,3,3)
- **Momentum Score**: Relative Strength vs Market

### Volatility Indicators
- **ATR**: Average True Range (14-period)
- **Bollinger Bands**: 20-period with 2 standard deviations
- **ADR**: Average Daily Range percentage

### Volume Indicators
- **Volume SMA**: 30-period volume average
- **Volume Ratio**: Current vs average volume
- **VWAP**: Volume Weighted Average Price

### Custom Indicators
- **Momentum**: 10 and 20-period price momentum
- **BB Squeeze**: Bollinger Band squeeze detection
- **Price Position**: Position relative to moving averages

## Screening Algorithms

### Maverick Momentum Screen
Identifies stocks with strong upward momentum:
- Price above 21-day EMA
- EMA-21 above SMA-50
- SMA-50 above SMA-200
- Relative Strength Rating > 70
- Minimum volume thresholds

### Bear Market Screen  
Identifies stocks in downtrends:
- Price below 21-day EMA
- EMA-21 below SMA-50
- Relative Strength Rating < 30
- High volume on down moves

### Supply/Demand Breakout Screen
Identifies accumulation patterns:
- Price above SMA-50 and SMA-200
- Strong relative strength (RS > 60)
- Institutional accumulation signals
- Volume dry-up followed by expansion

## Progress Tracking & Resume

The loader automatically saves progress to a checkpoint file:

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "total_symbols": 100,
  "processed_symbols": 75,
  "successful_symbols": 73,
  "completed_symbols": ["AAPL", "MSFT", ...],
  "failed_symbols": ["BADTICKER", "ANOTHERBAD"],
  "errors": [...],
  "elapsed_time": 3600
}
```

To resume an interrupted load:
```bash
python scripts/load_tiingo_data.py --resume --checkpoint-file load_progress.json
```

## Error Handling

### Automatic Retry Logic
- **Exponential Backoff**: 1s, 2s, 4s delays between retries
- **Rate Limit Handling**: Automatic delays when rate limited
- **Connection Errors**: Automatic retry with timeout handling

### Error Reporting
- **Detailed Logging**: All errors logged with context
- **Error Tracking**: Failed symbols tracked in checkpoint
- **Graceful Degradation**: Continue processing other symbols on individual failures

## Performance Optimization

### Database Optimizations
- **Bulk Inserts**: Use PostgreSQL's UPSERT for efficiency
- **Connection Pooling**: Reuse database connections
- **Batch Processing**: Process multiple symbols together
- **Index Creation**: Automatically create performance indexes

### Memory Management
- **Streaming Processing**: Process data in chunks to minimize memory usage
- **Garbage Collection**: Explicit cleanup of large DataFrames
- **Connection Limits**: Prevent connection exhaustion

### Monitoring
```bash
# Monitor progress in real-time
tail -f tiingo_data_loader.log

# Check database stats
python scripts/load_tiingo_data.py --database-stats

# Monitor system resources
htop  # CPU and memory usage
iotop # Disk I/O usage
```

## Examples and Testing

### Interactive Examples
```bash
# Run interactive examples
python scripts/load_example.py
```

The example script provides:
1. Load sample stocks (5 symbols)
2. Load sector stocks (technology)  
3. Resume interrupted load demonstration
4. Database statistics viewer

### Testing Different Configurations
```bash
# Test with small dataset
python scripts/load_tiingo_data.py --symbols AAPL,MSFT --years 1 --batch-size 10

# Test screening only (no new data)
python scripts/load_tiingo_data.py --symbols AAPL --years 0.1 --run-screening

# Test resume functionality
python scripts/load_tiingo_data.py --symbols AAPL,MSFT,GOOGL,AMZN,TSLA --batch-size 2
# Interrupt with Ctrl+C, then resume:
python scripts/load_tiingo_data.py --resume
```

## Troubleshooting

### Common Issues

#### 1. API Token Issues
```bash
# Check if token is set
echo $TIINGO_API_TOKEN

# Test API access
curl -H "Authorization: Token $TIINGO_API_TOKEN" \
     "https://api.tiingo.com/tiingo/daily/AAPL"
```

#### 2. Database Connection Issues
```bash
# Check database URL
echo $DATABASE_URL

# Test database connection
python -c "from maverick_mcp.data.models import SessionLocal; print('DB OK')"
```

#### 3. Rate Limiting
If you're getting rate limited frequently:
- Reduce `--max-concurrent` (default: 5)
- Increase `--batch-size` to reduce total requests
- Consider upgrading to Tiingo's paid plan

#### 4. Memory Issues
For large loads:
- Reduce `--batch-size` 
- Reduce `--max-concurrent`
- Monitor memory usage with `htop`

#### 5. Checkpoint Corruption
```bash
# Remove corrupted checkpoint
rm load_progress.json

# Start fresh
python scripts/load_tiingo_data.py --symbols AAPL,MSFT --no-checkpoint
```

### Performance Benchmarks

Typical performance on modern hardware:
- **Small Load (10 symbols, 1 year)**: 2-3 minutes
- **Medium Load (100 symbols, 2 years)**: 15-20 minutes
- **Large Load (500 symbols, 2 years)**: 1-2 hours
- **Full Load (3000+ symbols, 2 years)**: 6-12 hours

## Integration with Maverick-MCP

The loaded data integrates seamlessly with Maverick-MCP:

### API Endpoints
The data is immediately available through Maverick-MCP's API endpoints:
- `/api/v1/stocks` - Stock information
- `/api/v1/prices/{symbol}` - Price data  
- `/api/v1/technical/{symbol}` - Technical indicators
- `/api/v1/screening/maverick` - Maverick stock screen results

### MCP Tools
Use the loaded data in MCP tools:
- `get_stock_analysis` - Comprehensive stock analysis
- `run_screening` - Run custom screens
- `get_technical_indicators` - Retrieve calculated indicators
- `portfolio_analysis` - Analyze portfolio performance

## Advanced Usage

### Custom Symbol Lists
Create sector-specific or strategy-specific symbol files:

```bash
# Create growth stock list
cat > growth_stocks.txt << EOF
TSLA
NVDA
AMZN
GOOGL
META
NFLX
CRM
ADBE
EOF

# Load growth stocks
python scripts/load_tiingo_data.py --file growth_stocks.txt --run-screening
```

### Automated Scheduling
Set up daily data updates with cron:

```bash
# Add to crontab (crontab -e)
# Daily update at 6 PM EST (after market close)
0 18 * * 1-5 cd /path/to/maverick-mcp && python scripts/load_tiingo_data.py --sp500 --years 0.1 --run-screening >> /var/log/tiingo_updates.log 2>&1

# Weekly full reload on weekends  
0 2 * * 6 cd /path/to/maverick-mcp && python scripts/load_tiingo_data.py --sp500 --years 2 --calculate-indicators --run-screening >> /var/log/tiingo_weekly.log 2>&1
```

### Integration with CI/CD
```yaml
# GitHub Actions workflow
name: Update Market Data
on:
  schedule:
    - cron: '0 18 * * 1-5'  # 6 PM EST weekdays

jobs:
  update-data:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Update market data
        env:
          TIINGO_API_TOKEN: ${{ secrets.TIINGO_API_TOKEN }}
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
        run: |
          python scripts/load_tiingo_data.py --sp500 --years 0.1 --run-screening
```

## Support and Contributing

### Getting Help
- **Documentation**: Check this README and inline code comments
- **Logging**: Enable debug logging for detailed troubleshooting
- **Examples**: Use the example script to understand usage patterns

### Contributing
To add new features or fix bugs:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

### Feature Requests
Common requested features:
- Additional data providers (Alpha Vantage, Yahoo Finance)
- More technical indicators
- Custom screening algorithms
- Real-time data streaming
- Portfolio backtesting integration