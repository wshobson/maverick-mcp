# Tech debt tracker

One line per item. Remove the line in the same change that removes the debt.

| Item | Where | Phase to fix |
| --- | --- | --- |
| `server.json` declares only remote transports and no package installs | repo root | distribution |
| Dockerfile is single-stage and ships build toolchain in the final image | `Dockerfile` | distribution |
| MCP Apps chart rendering | new server | deferred |
| Tasks extension for long-running backtests | new server | deferred |
| `ty check` clean over `maverick/` but ~147 diagnostics under `tests/`; tests are outside the gate | `tests/` | deferred |
| Macro (FRED) port deferred; zero live consumers today; no macro domain exists yet | not ported | macro port |
| screening change-history (legacy pipeline) not ported; revisit if wanted | new server | deferred |
| run_screen executes rubrics on the event loop; wrap in to_thread if universe_max grows | `maverick/screening/service.py` | deferred |
| `pf_positions.total_cost` Numeric(20,4) would round >4dp fractional-share totals on Postgres (SQLite unaffected); revisit if Postgres adopted | `maverick/portfolio/data.py` | deferred |
