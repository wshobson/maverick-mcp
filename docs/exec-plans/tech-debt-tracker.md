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
| `get_llm()` always passes `LLM_TEMPERATURE` to `ChatAnthropic`; Claude Opus 4.7 and later reject it at any value (400) and Claude Sonnet 5 at any value but the default `1.0`, and langchain-anthropic 1.7.1 strips it only for `claude-fable-5*`, so Opus 4.7+ cannot back research or `backtesting_parse_strategy` and Sonnet 5 needs `LLM_TEMPERATURE=1.0` | `maverick/platform/llm.py` | research |
| `server.json`'s `oci` package declares `stdio` transport, but the image's default command serves Streamable HTTP on `0.0.0.0:8000` | repo root, `Dockerfile` | distribution |
