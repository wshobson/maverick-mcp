# Tech debt tracker

One line per item. Remove the line in the same change that removes the debt.

| Item | Where | Phase to fix |
| --- | --- | --- |
| `server.json` declares the PyPI package `maverick-mcp-server`, which this project has never published: the name is held by another account until pypi/support#12150 resolves | repo root | distribution |
| Daily-history validation has calendars for US stocks and `.L`, `.T`, `.TO`, `.AX`, `.HK`, `.DE` only; other exchange suffixes need verified mappings before history and dependent tools can accept them | `maverick/market_data/service.py` | market data |
| MCP Apps chart rendering | `maverick/` | deferred |
| Tasks extension for long-running backtests | `maverick/backtesting/` | deferred |
| Macro (FRED) port deferred; zero live consumers today; no macro domain exists yet | not ported | macro port |
| screening change-history (legacy pipeline) not ported; revisit if wanted | `maverick/screening/` | deferred |
| `server.json`'s `oci` package declares `stdio` transport, but the image's default command serves Streamable HTTP on `0.0.0.0:8000` | repo root, `Dockerfile` | distribution |
| On a fresh install the screener has no universe until price history has been fetched for each ticker (eval q07); `screening_run_screens` now returns an error that explains how to add symbols, and a built-in default universe is an open product decision | `maverick/screening/` | screening |
| `market_data_get_stock_fundamentals` carries no dividend fields (yield, rate, payout ratio) although yfinance's `info` has them, so a dividend question cannot be answered from Maverick (eval c17) | `maverick/market_data/` | market data |
