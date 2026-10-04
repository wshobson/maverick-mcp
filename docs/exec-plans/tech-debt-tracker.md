# Tech debt tracker

One line per item. Remove the line in the same change that removes the debt.

| Item | Where | Phase to fix |
| --- | --- | --- |
| `server.json` declares the PyPI package `maverick-mcp-server`, which this project has never published: the name is held by another account until pypi/support#12150 resolves | repo root | distribution |
| MCP Apps chart rendering | `maverick/` | deferred |
| Tasks extension for long-running backtests | `maverick/backtesting/` | deferred |
| Macro (FRED) port deferred; zero live consumers today; no macro domain exists yet | not ported | macro port |
| screening change-history (legacy pipeline) not ported; revisit if wanted | `maverick/screening/` | deferred |
| Every backtest trade's `duration` is `""`: `_extract_trades` reads a `Duration` column that vectorbt 1.x `records_readable` does not have | `maverick/backtesting/engine.py` | backtesting |
| `get_llm()` always passes `LLM_TEMPERATURE` (default `0.0`) to `ChatAnthropic`; Claude models released after Claude Opus 4.6 accept only `1.0` and return a 400 for any other value, so research and `backtesting_parse_strategy` fail on them unless the user sets `LLM_TEMPERATURE=1.0` | `maverick/platform/llm.py` | research |
| `server.json`'s `oci` package declares `stdio` transport, but the image's default command serves Streamable HTTP on `0.0.0.0:8000` | repo root, `Dockerfile` | distribution |
| `backtesting_backtest_portfolio` returns every trade of every symbol: five symbols over five years of `sma_cross` is about 71,000 characters (42,000 of them trades), near an MCP client's output limit | `maverick/backtesting/tools_support.py` | backtesting |
| `backtesting_create_strategy_ensemble` skips a symbol whose backtest raises or has fewer than 100 bars, and every symbol after the first five, without naming them in the result | `maverick/backtesting/service_ml.py` | backtesting |
| On a fresh install the screener has no universe until price history has been fetched for each ticker (eval q07); `screening_run_screens` now returns an error that explains how to add symbols, and a built-in default universe is an open product decision | `maverick/screening/` | screening |
| `portfolio_risk_adjusted_analysis` sizes every position at `account × 1% × risk factor` ($500 at the defaults) regardless of the ATR stop distance, reports that whole position as `max_risk_amount`, and returns a constant `confidence_score` (eval b06 re-run) | `maverick/portfolio/analysis.py` | portfolio |
| Backtest annualized metrics assume a 365-day year on trading-day bars: SPY sma_cross 2020 to 2024 reports `annual_return` 10.9% against a 7.4% compound rate, and Sharpe, Sortino, and Calmar are inflated the same way (eval q15 re-run) | `maverick/backtesting/engine.py` | backtesting |
| A strategy with no losing trades gets `profit_factor` 0, because `_safe_float` turns vectorbt's infinity into 0, and the analysis then lists "unprofitable trades" as a weakness (eval b15 re-run) | `maverick/backtesting/engine.py` | backtesting |
| `evals/tool_surface/seed.py` records position cost bases and a closed journal trade far below the real prices on those dates (NVDA at 131.75 on 2025-10-21, when it traded near 180), so seeded gains are overstated (eval b15 re-run) | `evals/tool_surface/seed.py` | evals |
| Stored price bars are never rewritten (including the first partial bar of the current session, which never becomes final): yfinance returns split- and dividend-adjusted prices, but `write_price_bars` keeps a date it already stored, so after a split a long-lived database mixes unadjusted and adjusted prices around the split date | `maverick/market_data/data.py` | market data |
| `correlation_analysis` picks off-diagonal values by value (`corr.values != 1`), not by position, so a real correlation of exactly 1.0 is dropped and a diagonal of 0.9999999999999998 is kept | `maverick/portfolio/analysis.py` | portfolio |
| Two concurrent price-history fetches for an uncached ticker race in `write_price_bars` (read existing dates, then insert), and the second fails with `UNIQUE constraint failed: md_price_bars.stock_id, md_price_bars.date`; parallel technical calls surface it as a tool error (eval c16). Conflict-safe upserts must also preserve later price corrections | `maverick/market_data/data.py` | market data |
| No tool lists watchlists by name and id, so a client cannot confirm which one to write to, and `portfolio_watchlist_brief` returns success with no items for an id that does not exist (eval c15) | `maverick/portfolio/` | portfolio |
| `market_data_get_stock_fundamentals` carries no dividend fields (yield, rate, payout ratio) although yfinance's `info` has them, so a dividend question cannot be answered from Maverick (eval c17) | `maverick/market_data/` | market data |
| Research response envelopes drop citations, and competitive-analysis focus names do not match the router (review R8/R9) | `maverick/research/service.py`, `agents/graph.py` | correctness task 12 |
| Source archives ship tests/Makefile without required evals/tools/scripts sources; extracted-sdist tests fail, and CI lacks a real core-only wheel install lane (review R10) | `pyproject.toml`, `.github/workflows/ci.yml` | correctness task 13 |
