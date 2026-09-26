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
| `ty check` clean over `maverick/` but ~147 diagnostics under `tests/`; tests are outside the gate | `tests/` | deferred |
| Macro (FRED) port deferred; zero live consumers today; no macro domain exists yet | not ported | macro port |
| screening change-history (legacy pipeline) not ported; revisit if wanted | new server | deferred |
| run_screen executes rubrics on the event loop; wrap in to_thread if universe_max grows | `maverick/screening/service.py` | deferred |
| `pf_positions.total_cost` Numeric(20,4) would round >4dp fractional-share totals on Postgres (SQLite unaffected); revisit if Postgres adopted | `maverick/portfolio/data.py` | deferred |
| Lock carries pandas 3.0.5 / numpy 2.5.3 / vectorbt 1.1.0 while the floors stay at pandas>=2.3.3 / numpy>=2.2.6 / vectorbt>=1.0.0 and CI installs --frozen, so the floor combination never runs; a floor install also resurfaces numpy's generic-timedelta DeprecationWarning through vectorbt 1.0 | `pyproject.toml` | dependencies |
| Core `openai` and `anthropic` dependencies have no importer in `maverick/`; only `[research]` packages use them (langchain-openai, langchain-anthropic, exa-py), so a base install carries both SDKs for nothing | `pyproject.toml` | dependencies |
