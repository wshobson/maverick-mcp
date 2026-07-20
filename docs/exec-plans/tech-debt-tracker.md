# Tech debt tracker

One line per item. Remove the line in the same change that removes the debt.

| Item | Where | Phase to fix |
| --- | --- | --- |
| `setup.py` duplicates hatchling and parses pyproject by hand | repo root | packaging |
| Wheel build uses `include = ["*.py"]` instead of explicit packages | `pyproject.toml` | packaging |
| `server.json` declares only remote transports and no package installs | repo root | distribution |
| Dockerfile is single-stage and ships build toolchain in the final image | `Dockerfile` | distribution |
| Default pytest filter deselects 664 tests; review the marker policy | `pyproject.toml` | cutover |
| MCP Apps chart rendering | new server | deferred |
| Tasks extension for long-running backtests | new server | deferred |
| Two typecheckers disagree: CI gates on ty, make check runs pyright; retire one or document ty as the gate | `Makefile`, `.github/workflows/ci.yml` | maintainer decision |
| Macro (FRED) port deferred; zero live consumers today; no macro domain exists yet | not ported | macro port |
| Tier-3 mover fallback runs without breaker/retry (documented last-resort trade-off) | `maverick/market_data/fetchers.py` | deferred |
| Capital Companion tier uses `request_with_retry` without breaker and creates a client per call; align with `request_resilient` at server assembly | `maverick/market_data/fetchers.py` | cutover |
| `get_quotes` is untested-in-production surface (no tool consumes it) | `maverick/market_data/service.py` | cutover |
| screening change-history (legacy pipeline) not ported; revisit if wanted | new server | deferred |
| run_screen executes rubrics on the event loop; wrap in to_thread if universe_max grows | `maverick/screening/service.py` | deferred |
| `pf_positions.total_cost` Numeric(20,4) would round >4dp fractional-share totals on Postgres (SQLite unaffected); revisit if Postgres adopted | `maverick/portfolio/data.py` | deferred |
| service_ml.py, ensemble.py, online_learning.py, and feature_engineering.py at 499-500/500 line cap; split before next addition | `maverick/backtesting/service_ml.py`, `maverick/backtesting/strategies/ml/ensemble.py`, `maverick/backtesting/strategies/ml/online_learning.py`, `maverick/backtesting/strategies/ml/feature_engineering.py` | deferred |
| exa search_and_contents() deprecated upstream -- migrate to search() post-cutover | `maverick/research/providers/exa.py` | deferred |
