# Tiingo Data Loader

The Tiingo loader fetches stock metadata and OHLCV data, calculates indicators,
and stores the results in the MaverickMCP database.

## Prerequisites

- Tiingo API token.
- Database configured with `DATABASE_URL` or default SQLite.
- Dependencies installed with `uv sync --extra dev`.

The loader-specific requirements file remains at
`../../scripts/requirements_tiingo.txt` for direct script usage.

## Environment

```bash
export TIINGO_API_TOKEN=your_token_here
export DATABASE_URL=sqlite:///maverick_mcp.db
```

Some app paths use `TIINGO_API_KEY`; keep both variables aligned if you are
switching between loader scripts and server runtime.

## Validate Setup

```bash
python scripts/validate_setup.py
```

## Load Examples

Specific symbols:

```bash
python scripts/load_tiingo_data.py \
  --symbols AAPL,MSFT,GOOGL,AMZN,TSLA \
  --years 2 \
  --calculate-indicators
```

S&P 500 sample:

```bash
python scripts/load_tiingo_data.py --sp500 --years 1 --run-screening
```

Custom file:

```bash
printf "AAPL\nMSFT\nGOOGL\n" > my_symbols.txt
python scripts/load_tiingo_data.py --file my_symbols.txt --calculate-indicators --run-screening
```

Resume from checkpoint:

```bash
python scripts/load_tiingo_data.py --resume --checkpoint-file load_progress.json
```

## Common Options

- `--symbols AAPL,MSFT` - comma-separated symbols.
- `--file symbols.txt` - load symbols from a file.
- `--sp500` - load the default S&P 500 subset.
- `--sp500-full` - load the full S&P 500.
- `--years 2` - trailing years of history.
- `--start-date YYYY-MM-DD` and `--end-date YYYY-MM-DD` - explicit range.
- `--calculate-indicators` - calculate technical indicators.
- `--run-screening` - run Maverick/Bear/Supply-Demand screens.
- `--batch-size 50` and `--max-concurrent 5` - tune throughput.

## Output And Recovery

The loader writes checkpoint files for interrupted runs. Use `--resume` with the
same checkpoint file to avoid refetching completed symbols.

Use smaller batches or lower concurrency if Tiingo rate limits or local database
contention appear.
