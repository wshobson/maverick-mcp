# Tech debt tracker

One line per item. Remove the line in the same change that removes the debt.

| Item | Where | Phase to fix |
| --- | --- | --- |
| `server.json` declares the PyPI package `maverick-mcp-server`, which this project has never published: the name is held by another account until pypi/support#12150 resolves | repo root | distribution |
| MCP Apps chart rendering | `maverick/` | deferred |
| Tasks extension for long-running backtests | `maverick/backtesting/` | deferred |
| Macro (FRED) port deferred; zero live consumers today; no macro domain exists yet | not ported | macro port |
| screening change-history (legacy pipeline) not ported; revisit if wanted | `maverick/screening/` | deferred |
| `server.json`'s `oci` package declares `stdio` transport, but the image's default command serves Streamable HTTP on `0.0.0.0:8000` | repo root, `Dockerfile` | distribution |
| `backtesting_backtest_portfolio` returns every trade of every symbol: five symbols over five years of `sma_cross` is about 71,000 characters (42,000 of them trades), near an MCP client's output limit | `maverick/backtesting/tools_support.py` | backtesting |
| `backtesting_create_strategy_ensemble` skips a symbol whose backtest raises or has fewer than 100 bars, and every symbol after the first five, without naming them in the result | `maverick/backtesting/service_ml.py` | backtesting |
| On a fresh install the screener has no universe until price history has been fetched for each ticker (eval q07); `screening_run_screens` now returns an error that explains how to add symbols, and a built-in default universe is an open product decision | `maverick/screening/` | screening |
| `portfolio_risk_adjusted_analysis` sizes every position at `account × 1% × risk factor` ($500 at the defaults) regardless of the ATR stop distance, reports that whole position as `max_risk_amount`, and returns a constant `confidence_score` (eval b06 re-run) | `maverick/portfolio/analysis.py` | portfolio |
| `evals/tool_surface/seed.py` records position cost bases and a closed journal trade far below the real prices on those dates (NVDA at 131.75 on 2025-10-21, when it traded near 180), so seeded gains are overstated (eval b15 re-run) | `evals/tool_surface/seed.py` | evals |
| `correlation_analysis` picks off-diagonal values by value (`corr.values != 1`), not by position, so a real correlation of exactly 1.0 is dropped and a diagonal of 0.9999999999999998 is kept | `maverick/portfolio/analysis.py` | portfolio |
| No tool lists watchlists by name and id, so a client cannot confirm which one to write to, and `portfolio_watchlist_brief` returns success with no items for an id that does not exist (eval c15) | `maverick/portfolio/` | portfolio |
| `market_data_get_stock_fundamentals` carries no dividend fields (yield, rate, payout ratio) although yfinance's `info` has them, so a dividend question cannot be answered from Maverick (eval c17) | `maverick/market_data/` | market data |
| Source archives ship tests/Makefile without required evals/tools/scripts sources; extracted-sdist tests fail, and CI lacks a real core-only wheel install lane (review R10) | `pyproject.toml`, `.github/workflows/ci.yml` | correctness task 13 |
