# Tech debt tracker

One line per item. Remove the line in the same change that removes the debt.

| Item | Where | Phase to fix |
| --- | --- | --- |
| `is_auth_enabled` and `AUTH_ENABLED` survive in the provider config interface | `maverick_mcp/providers/` | cutover |
| `setup.py` duplicates hatchling and parses pyproject by hand | repo root | packaging |
| Wheel build uses `include = ["*.py"]` instead of explicit packages | `pyproject.toml` | packaging |
| `server.json` declares only remote transports and no package installs | repo root | distribution |
| Dockerfile is single-stage and ships build toolchain in the final image | `Dockerfile` | distribution |
| Two agent abstractions exist (`agents/` and `workflows/agents/`) | legacy tree | research port |
| Five LLM and search vendors are reachable from research paths | `providers/llm_factory.py` | research port |
| Default pytest filter deselects 664 tests; review the marker policy | `pyproject.toml` | cutover |
| MCP Apps chart rendering | new server | deferred |
| Tasks extension for long-running backtests | new server | deferred |
| `test_in_memory_server.py` hangs reading the `health://` resource via the in-memory client; quarantined `integration`, no root cause yet | `maverick_mcp/tests/` | cutover |
| `test_models_functional.py` fixture bypasses lazy schema creation and fails on a fresh CI database; needs a fixture rewrite; quarantined `integration` | `maverick_mcp/tests/` | cutover |
| `test_mcp_tool_fixes.py` is a vacuous duplicate of `test_mcp_tool_fixes_pytest.py`; deletion candidate | `maverick_mcp/tests/` | cutover |
| `application/commands/` is unimported by production code | `maverick_mcp/application/` | cutover |
| Two typecheckers disagree: CI gates on ty, make check runs pyright; retire one or document ty as the gate | `Makefile`, `.github/workflows/ci.yml` | maintainer decision |
| `tests/utils/test_quick_cache.py::test_cache_speedup` is a wall-clock timing flake (asserts >100x speedup) | `tests/utils/` | cutover |
| `config/database.py` and `config/database_self_contained.py` are dead pool config, not ported to `maverick/platform/db.py` | `maverick_mcp/config/` | cutover |
| Root `logging_config.py` is dead | `maverick_mcp/logging_config.py` | cutover |
| `utils/quick_cache.py` is unused | `maverick_mcp/utils/quick_cache.py` | cutover |
| Five parallel logging systems collapse into `maverick/platform/telemetry.py` | legacy tree | cutover |
| Three circuit-breaker implementations collapse into `maverick/platform/http.py` | legacy tree | cutover |
| `next(get_db())` session leak | `maverick_mcp/api/server.py`, `maverick_mcp/api/routers/portfolio.py` | portfolio port |
| Macro (FRED) port deferred; zero live consumers today | `maverick_mcp/providers/macro_data.py`, `maverick_mcp/providers/interfaces/macro_data.py` | macro port |
| Legacy market-data deletion inventory for cutover: provider interfaces/factories/implementations/mocks for market, stock, and macro data; both `dependencies.py` modules; the duplicate `infrastructure/data_fetching` + `infrastructure/caching` + `domain/stock_analysis` stack; dead provider methods | `maverick_mcp/providers/`, `maverick_mcp/infrastructure/data_fetching/`, `maverick_mcp/infrastructure/caching/`, `maverick_mcp/domain/stock_analysis/`, `maverick_mcp/dependencies.py`, `maverick_mcp/providers/dependencies.py` | cutover |
| Legacy duplicate tool names collapse at server assembly (e.g. `fetch_stock_data` vs `data_fetch_stock_data`) | `maverick_mcp/api/server.py`, `maverick_mcp/api/routers/tool_registry.py` | cutover |
| `docs/testing/in-memory.md` shows `result.text` but fastmcp 3.3.1 returns `result.data` (doc drift) | `docs/testing/in-memory.md` | cutover |
| Tier-3 mover fallback runs without breaker/retry (documented last-resort trade-off) | `maverick/market_data/fetchers.py` | deferred |
| Capital Companion tier uses `request_with_retry` without breaker and creates a client per call; align with `request_resilient` at server assembly | `maverick/market_data/fetchers.py` | cutover |
| `get_quotes` is untested-in-production surface (no tool consumes it) | `maverick/market_data/service.py` | cutover |
| `providers/optimized_stock_data.py` is a screening-provider duplicate kept alive only by the performance_* tool family; retires with it at cutover | `maverick_mcp/providers/` | cutover |
| screening change-history (legacy pipeline) not ported; revisit if wanted | new server | deferred |
| legacy mcp_maverick_* tables, scripts/run_stock_screening.py, and the screening_pipeline tools retire at cutover; scr_results is the successor | legacy tree | cutover |
| run_screen executes rubrics on the event loop; wrap in to_thread if universe_max grows | `maverick/screening/service.py` | deferred |
| legacy mcp_portfolios/mcp_portfolio_positions retire at cutover; optional one-shot migration script decision at cutover | legacy tree | cutover |
| risk_dashboard must consume PortfolioService reads when ported | `maverick_mcp/api/routers/risk_dashboard.py` | cutover |
| `pf_positions.total_cost` Numeric(20,4) would round >4dp fractional-share totals on Postgres (SQLite unaffected); revisit if Postgres adopted | `maverick/portfolio/data.py` | deferred |
| legacy core/technical_analysis.py + technical routers + visualization.py + stock_helpers chain retire at cutover | `maverick_mcp/core/technical_analysis.py`, `maverick_mcp/api/routers/technical.py`, `maverick_mcp/api/routers/technical_enhanced.py`, `maverick_mcp/core/visualization.py`, `maverick_mcp/utils/stock_helpers.py` | cutover |
| unwired validation models in maverick_mcp/validation/technical.py die with the legacy routers | `maverick_mcp/validation/technical.py` | cutover |
| dead registry.get_tool('get_technical_indicators') references in legacy agents (falls back to mock tools) | `maverick_mcp/agents/technical_analysis.py`, `maverick_mcp/agents/market_analysis.py` | cutover or research port |
| legacy maverick_mcp/backtesting + router + visualization retire at cutover | `maverick_mcp/backtesting/`, `maverick_mcp/api/routers/backtesting.py`, `maverick_mcp/core/visualization.py` | cutover (Phase 8) |
| requires-python <3.13 pin blocked by core ta-lib until cutover | `pyproject.toml` | cutover |
| matplotlib/seaborn removal blocked by legacy visualization until cutover | `pyproject.toml` | cutover |
| service_ml.py, ensemble.py, online_learning.py, and feature_engineering.py at 499-500/500 line cap; split before next addition | `maverick/backtesting/service_ml.py`, `maverick/backtesting/strategies/ml/ensemble.py`, `maverick/backtesting/strategies/ml/online_learning.py`, `maverick/backtesting/strategies/ml/feature_engineering.py` | deferred |
