# Phase 3: Screening Domain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `maverick/screening/` (query and compute) on a new slim pure-Python `maverick/technical/` indicator core, plus the tracked platform cleanup, so a fresh install can compute real screens with zero native dependencies.

**Architecture:** Phase 3 of `docs/design-docs/2026-07-18-mcp-modernization.md`, reconciling the spec's ordering: the technical domain's indicator core lands first (a slim slice: pure pandas/numpy functions with golden tests against recorded pandas-ta output), because the screening rubrics consume it. Screening follows the market-data domain template exactly: layer contracts, injectable everything, honest fakes, no network in tests. The 2026-07-19 screening recon mapped the legacy surface; the only real algorithm lives in `scripts/run_stock_screening.py` (a talib CLI script nothing calls automatically), so this phase brings compute into the domain and ends the "empty tables on fresh install" failure mode.

**Tech Stack:** Python 3.12, pydantic, SQLAlchemy 2, pandas/numpy (pure-Python indicators), maverick.platform, maverick.market_data (price history), pytest with TDD. pandas-ta is used ONLY inside a fixture-recording script, never imported by `maverick/` or by tests at runtime.

## Global Constraints

- No new dependencies. No talib, no numba, no pandas-ta imports anywhere under `maverick/` or `tests/`.
- `maverick/` files stay under 500 lines each.
- Env reading only in `config.py` modules via the platform helpers.
- No network in tests; screening tests build price frames by hand or via small deterministic generators.
- Full gate before every commit: `uv run pytest tests/technical/ tests/screening/ tests/market_data/ tests/platform/ tests/structure/ -q`, `make lint`, `git ls-files '*.py' | xargs uv run ruff format --check`, `uv run ty check maverick maverick_mcp/services maverick_mcp/domain`. All green.
- Commit after every task; flip the task's checkboxes in the same commit; stage explicitly; never `git add -A`.
- Prose in docs follows the plain style: short sentences, no em dashes.

## Decision log

- 2026-07-19: Spec-order reconciliation. A slim `maverick/technical/` indicator core (sma, ema, rsi, macd, atr) lands in this phase because screening needs it. The full technical domain (tool surface, remaining indicators) stays a later phase.
- 2026-07-19: Screening owns compute AND query. The legacy server only queried loader-populated tables; the real algorithm was an orphan CLI script. `run_*_screen()` service methods port the script's rubrics (thresholds preserved as env-backed config defaults) using pure-Python indicators.
- 2026-07-19: New tables (`scr_*`), not the legacy `mcp_maverick_*`. Existing legacy rows are not migrated; the new domain computes fresh snapshots. Recorded as a cutover note.
- 2026-07-19: The screening pipeline (runs/changes/scheduled_jobs) is NOT ported. It was never wired to a scheduler and `run_screen` had no production caller. A tech-debt row records "change history" as a possible future feature.
- 2026-07-19: `get_screening_by_criteria`'s `sector` parameter is dropped, not ported. It was a documented no-op in the legacy tool.
- 2026-07-19: The `momentum_score` rename is finished. No "formerly rs_rating" comments come forward.
- 2026-07-19: Dead legacy screening code with zero live callers is deleted THIS phase (not at cutover): `application/screening/`, `infrastructure/screening/`, `domain/screening/`, `providers/optimized_stock_data.py`, `providers/dependencies.py`, `providers/factories/provider_factory.py`, `providers/implementations/stock_data_adapter.py`, `providers/interfaces/stock_data.py`, `providers/mocks/mock_stock_data.py`. Grep-verified before deletion; the domain rules worth keeping (score thresholds 50/40/70, limits max 100 / default 20) are carried into `ScreeningSettings`.
- 2026-07-19: Platform cleanup: `Cache` gains an injectable `redis_settings` parameter, closing the tracked test-isolation debt row.

## Layer contracts (Task 3 and Task 5 encode these)

```
maverick.technical: indicators.py (pure functions; imports nothing from domains)
maverick.screening: tools -> service -> {data, screens} -> config -> types
cross-domain: screening service/data/screens MAY import maverick.technical and maverick.market_data (service level); platform imports no domain; no domain imports maverick_mcp.
```

---

### Task 1: platform cleanup — Cache redis_settings injection

**Files:**
- Modify: `maverick/platform/cache.py`
- Test: `tests/platform/test_cache.py` (additions only)
- Modify: `docs/exec-plans/tech-debt-tracker.md` (remove the closed row)

**Interfaces:**
- `Cache.__init__(settings: CacheSettings | None = None, redis_client=None, redis_settings: RedisSettings | None = None)` — `redis_settings` defaults to `get_platform_settings().redis` (current behavior preserved); an injected `RedisSettings(enabled=False, ...)` guarantees the SQLite tier regardless of the process environment.

- [x] **Step 1: Write the failing test** — with `REDIS_HOST` monkeypatched into the env (simulating a dev box's `.env`), `Cache(settings=..., redis_settings=RedisSettings(enabled=False))` must build the SQLite tier (`cache.sqlite is not None`, `cache.redis is None` or equivalent). Also: default behavior unchanged (no `redis_settings` -> reads platform settings).
- [x] **Step 2: RED.** **Step 3: Implement** (thread the parameter through tier selection; do not change tier logic). **Step 4: GREEN + full gate.** Remove the tracker row in the same commit.
- [x] **Step 5: Commit** `fix(platform): allow injecting redis settings into Cache`.

---

### Task 2: delete the dead legacy screening slice

**Files:**
- Delete: `maverick_mcp/application/screening/` (whole package; `application/` dir itself if empty after), `maverick_mcp/infrastructure/screening/`, `maverick_mcp/domain/screening/`, `maverick_mcp/providers/optimized_stock_data.py`, `maverick_mcp/providers/dependencies.py`, `maverick_mcp/providers/factories/provider_factory.py`, `maverick_mcp/providers/implementations/stock_data_adapter.py`, `maverick_mcp/providers/interfaces/stock_data.py`, `maverick_mcp/providers/mocks/mock_stock_data.py`
- Delete: any test files whose ONLY subject is the deleted modules (grep-discovered)

- [x] **Step 1: Verify zero live callers** — for each module, grep `maverick_mcp/ tests/ scripts/` for imports, excluding self-references and the other deleted modules. Known allowed fallout: `tools/performance_monitoring.py` imports `OptimizedStockDataProvider` — inspect it; if that tool is itself registered nowhere live (check tool_registry and server.py), delete it too and note it; if it IS live, stop and report NEEDS_CONTEXT. **Resolution:** `optimized_stock_data.py` IS live (backs the registered `performance_get_system_performance_health` MCP tool via `tools/performance_monitoring.py` -> `api/routers/performance.py` -> `register_performance_tools` -> `register_all_router_tools` -> `server.py`). Removed from the deletion list per controller decision; tracked in `docs/exec-plans/tech-debt-tracker.md` for cutover instead. The other 8 modules had zero live callers.
- [x] **Step 2: Delete, fix `__init__.py` re-exports the deletions orphan** (factories/implementations/interfaces/mocks packages keep their other members).
- [x] **Step 3: Full gate plus `make test`** — expect the suite to shrink only by deleted-module tests; everything else green.
- [x] **Step 4: Commit** `refactor: delete dead legacy screening slice (zero live callers)`.

---

### Task 3: technical indicator core

**Files:**
- Create: `maverick/technical/__init__.py`, `maverick/technical/indicators.py`
- Create: `scripts/record_indicator_fixtures.py` (runs pandas-ta ONCE to record goldens; not imported by anything)
- Create: `tests/technical/__init__.py`, `tests/technical/fixtures/indicator_goldens.json`, `tests/technical/test_indicators.py`
- Modify: `pyproject.toml` (import-linter: add `maverick.technical` to root packages handling if needed; contract "technical imports no domain": forbidden `maverick.technical` -> `maverick.market_data | maverick.screening`)

**Interfaces:**
- Pure functions on `pd.Series`/`pd.DataFrame`: `sma(close, period) -> pd.Series`; `ema(close, period) -> pd.Series` (pandas ewm, adjust=False); `rsi(close, period=14) -> pd.Series` (Wilder smoothing); `macd(close, fast=12, slow=26, signal=9) -> pd.DataFrame` (columns macd, signal, histogram); `atr(high, low, close, period=14) -> pd.Series` (Wilder). All tz-naive in, NaN-headed warmup out (matching pandas-ta's shape).

- [x] **Step 1: Record the goldens** — write and run `scripts/record_indicator_fixtures.py`: builds one deterministic OHLCV frame (seeded numpy RNG, 300 rows) plus one real-shaped frame (a hardcoded 60-row constant list in the script), runs pandas-ta's `sma/ema/rsi/macd/atr`, and writes inputs and expected outputs (last 200 values each, NaNs as nulls) to `tests/technical/fixtures/indicator_goldens.json`. Commit the fixture; the script stays for regeneration but is never imported.
- [x] **Step 2: Write the failing tests** — `test_indicators.py` loads the goldens and asserts each pure function matches the recorded pandas-ta output with `rtol=1e-9` (allclose over the non-NaN region, and NaN positions identical). Plus edge tests: period longer than series -> all-NaN; constant series -> RSI converges to a defined value without division errors.
- [x] **Step 3: RED.** **Step 4: Implement** the five functions in pure pandas/numpy. Wilder smoothing for rsi/atr must match pandas-ta's implementation (ewm with alpha=1/period, adjust=False) — the goldens are the arbiter.
- [x] **Step 5: GREEN + contract + full gate.** `uv run lint-imports` gains the technical-independence contract.
- [x] **Step 6: Commit** `feat(technical): add pure-python indicator core with pandas-ta goldens`.

---

### Task 4: screening skeleton and layer contracts

**Files:**
- Create: `maverick/screening/__init__.py` + docstring-only `types.py`, `config.py`, `data.py`, `screens.py`, `service.py`, `tools.py`
- Modify: `pyproject.toml` (layers contract for screening mirroring market_data's, with `data | screens` as the sibling tier; extend the platform-independence contract's forbidden list with `maverick.screening` and `maverick.technical`)
- Test: `tests/screening/__init__.py`, `tests/screening/test_layers.py` (same shutil/lint-imports pattern as market_data's)

- [x] Steps mirror Phase 2 Task 1 exactly: skeleton, contracts, verification test, prove-it-fails with a temporary reverse import, full gate, commit `feat(screening): add domain skeleton with enforced layer contracts`.

---

### Task 5: screening types

**Files:** `maverick/screening/types.py`; `tests/screening/test_types.py`

**Interfaces (pydantic):**
- `ScreeningResult(symbol, screen: str, date_analyzed: str, close: float, combined_score: int, momentum_score: float | None, indicators: dict[str, float | None], flags: dict[str, bool], reason: str)` — one shape for all three screens; screen is "bullish" | "bearish" | "supply_demand".
- `AllScreeningResults(bullish: list[ScreeningResult], bearish: list[ScreeningResult], supply_demand: list[ScreeningResult], date_analyzed: str | None)`.
- `ScreenRun(screen: str, symbols_screened: int, symbols_qualified: int, date_analyzed: str, duration_seconds: float)`.
- `ScreeningCriteria(min_momentum_score: float | None = None, min_volume: int | None = None, max_price: float | None = None, min_combined_score: int | None = None)` (no sector — dropped per decision log).
- [ ] TDD steps as in Phase 2: concrete tests (round-trip, composition), RED, implement, GREEN + gate, commit `feat(screening): add payload types`.

---

### Task 6: screening config

**Files:** `maverick/screening/config.py`; `tests/screening/test_config.py`

**Interfaces:** `ScreeningSettings` with env-backed fields defaulting to the legacy literals (env names in parens): `bullish_min_score: int = 50` (SCR_BULLISH_MIN_SCORE), `bear_min_score: int = 40` (SCR_BEAR_MIN_SCORE), `min_history_days: int = 200` (SCR_MIN_HISTORY_DAYS), `volume_surge_multiplier: float = 1.5`, `volume_decline_multiplier: float = 1.2`, `atr_contraction_multiplier: float = 0.8`, `rsi_overbought: float = 80.0`, `rsi_oversold: float = 30.0`, `default_limit: int = 20`, `max_limit: int = 100`, `universe_max: int = 200` (SCR_UNIVERSE_MAX; cap on symbols per run); plus `get_screening_settings()` / `reset_screening_settings()` using the platform helpers.
- [ ] TDD steps mirroring market_data's config task (defaults, env overrides, singleton/reset). Commit `feat(screening): add domain settings`.

---

### Task 7: screening persistence

**Files:** `maverick/screening/data.py`; `tests/screening/test_data.py`

**Interfaces:**
- `METADATA`; one table `scr_results` (id PK, screen TEXT, symbol TEXT, date_analyzed DATE, close NUMERIC(12,4), combined_score INT, momentum_score NUMERIC(5,2) nullable, indicators JSON, flags JSON, reason TEXT; unique (screen, symbol, date_analyzed); index (screen, date_analyzed, combined_score)). One table for all screens beats three near-identical legacy tables.
- `replace_screen_snapshot(session, screen, date_analyzed, rows: list[ScreeningResult]) -> int` (delete-then-insert for that screen+date, matching legacy upsert-by-date semantics).
- `read_top(session, screen, limit, min_combined_score=None, min_momentum_score=None) -> list[ScreeningResult]` (latest date_analyzed for that screen, ordered by combined_score desc) and `read_latest_all(session) -> AllScreeningResults`.
- `read_by_criteria(session, criteria: ScreeningCriteria, limit) -> list[ScreeningResult]` (bullish screen, all criteria AND-ed) — ONE query implementation, closing the legacy dual-path risk.
- [ ] TDD with tmp SQLite via platform.db: snapshot replace-idempotence (same date twice -> same row count), top-N ordering and filters, criteria filters, latest-all across screens, empty states. Commit `feat(screening): add results persistence`.

---

### Task 8: screen rubrics

**Files:** `maverick/screening/screens.py`; `tests/screening/test_screens.py`

**Interfaces:** pure functions taking a price DataFrame (yfinance-cased OHLCV, tz-naive) and `ScreeningSettings`, returning `ScreeningResult | None`:
- `score_bullish(symbol, df, settings)` — the legacy rubric: +25 each close>sma50/sma150/sma200, +25 MA alignment (sma50>sma150>sma200), +10 volume > surge_multiplier x 30d avg, +10 rsi < rsi_overbought, qualifies at combined_score >= bullish_min_score. (The legacy "+15 Uptrend pattern" point is folded into the MA-alignment criterion; note it in the docstring.)
- `score_bearish(symbol, df, settings)` — +20 close<sma50, +20 close<sma200, +15 rsi < rsi_oversold else +10 if rsi < 40, +15 macd < signal, +20 volume > decline_multiplier x avg on a down day, +10 atr < atr_contraction_multiplier x 20d avg atr; qualifies at bear_min_score.
- `score_supply_demand(symbol, df, settings)` — the 5-criteria boolean gate (close>sma150 and >sma200; sma150>sma200; sma200 rising over ~22 bars; sma50>sma150>sma200... wait, encode exactly: close>sma150, close>sma200, sma150>sma200, sma200 trending up over 22 bars, sma50>sma150 and sma50>sma200, close>sma50); when gated in: momentum-style ordering fields (breakout_strength: +25 volume>1.2x avg, +25 close within 25% of 252d high) mapped into combined_score.
- Each returns None when history < settings.min_history_days or when it does not qualify. Indicators come from `maverick.technical.indicators`. A `reason` string is generated by ONE helper shared across screens (closing the legacy duplicate reason-generator debt).
- [ ] TDD: hand-crafted frames that hit each rubric branch (a strong-uptrend frame scoring exactly the expected points; a frame one criterion short of the threshold returning None; a bear setup; a supply-demand qualifying frame; a 100-row frame returning None on history). Assert exact combined_score values — the rubric is the contract. Commit `feat(screening): add pure screen rubrics`.

---

### Task 9: screening service and tools

**Files:** `maverick/screening/service.py`, `maverick/screening/tools.py`; `tests/screening/test_service.py`, `tests/screening/test_tools.py`

**Interfaces:**
- `ScreeningService(engine, market_data: MarketDataService, settings=None, universe_fn: Callable[[], list[str]] | None = None)` — `universe_fn` defaults to reading distinct symbols from the market-data domain's stock table (via its public data API; add a small public helper there if needed — authorized); injectable for tests.
- Query methods (sync-wrapped async like market_data): `get_bullish(limit, min_score=None)`, `get_bearish(...)`, `get_supply_demand(limit, min_momentum_score=None)`, `get_all()`, `get_by_criteria(criteria, limit)` — thin over data.py, each returning the typed payloads.
- Compute: `run_screen(screen: str) -> ScreenRun` and `run_all_screens() -> dict[str, ScreenRun]` — universe (capped at settings.universe_max) -> `market_data.get_price_history` per symbol (sequential with a small asyncio.Semaphore, e.g. 4) -> rubric -> `replace_screen_snapshot`. Symbols whose history fetch fails are skipped and counted, never fatal.
- Tools (`configure(service)` + `register(mcp)`, all `screening_` prefixed): `screening_get_bullish(limit=20, min_score=None)`, `screening_get_bearish(...)`, `screening_get_supply_demand(limit=20, min_momentum_score=None)`, `screening_get_all()`, `screening_get_by_criteria(min_momentum_score=None, min_volume=None, max_price=None, min_combined_score=None, limit=20)` — all `readOnlyHint: True`; `screening_run_screens(screen: str | None = None)` with `readOnlyHint: False, destructiveHint: False, idempotentHint: True`. Error-payload semantics identical to market_data tools. Limits clamped to settings.max_limit.
- [ ] TDD: service tests with a stub MarketDataService (deterministic frames per symbol: one bullish qualifier, one bear qualifier, one too-short history) asserting run_all persists the right rows and counts, queries return them, and re-running the same day replaces (not duplicates); tool tests with a stub service (shapes, clamping, error payloads, unconfigured RuntimeError path — with the conftest resets pattern from market_data: add `tests/screening/conftest.py` with breaker/settings resets); an in-memory fastmcp Client round-trip for `screening_get_bullish`. Commit in two steps if cleaner (service then tools), each with the full gate.

---

### Task 10: close-out

**Files:** exports in `maverick/screening/__init__.py` and `maverick/technical/__init__.py`; `docs/QUALITY_SCORE.md`; `docs/exec-plans/tech-debt-tracker.md`; `docs/CATALOG.md`; `docs/INDEX.md`; move plan to completed/

- [ ] Exports with `__all__` + smoke imports. QUALITY_SCORE rows for `maverick/technical/` and `maverick/screening/` (A grades with one-line whys); update the legacy `application/`/`domain` rows to note the screening slice deletion. Tech-debt updates: remove the Cache row (closed in Task 1); add "screening change-history (legacy pipeline) not ported; revisit if wanted"; add "legacy mcp_maverick_* tables and scripts/run_stock_screening.py retire at cutover; scr_results is the successor". Plan decision-log addendum if execution deviated. Full verification including `make test` and `make docs-check`; move plan; CATALOG/INDEX paths; commit `docs: complete phase 3 (screening domain)`; push; `gh run watch --exit-status` in the foreground.
