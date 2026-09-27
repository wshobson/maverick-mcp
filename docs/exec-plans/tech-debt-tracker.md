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
| `evals/tool_surface/seed.py` writes a screening snapshot with fixture closes that disagree with live quotes, so traces that compare the two report stale data (eval b06) | `evals/tool_surface/seed.py` | evals |
