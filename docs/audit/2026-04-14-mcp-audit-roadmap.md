# MaverickMCP — Audit & Optimization Roadmap

**Date:** 2026-04-14
**Branch:** `feature/resilience-and-sast`
**Scope:** MCP protocol hygiene, stale-data RCA, four-axis optimization sweep, phase-level roadmap
**Artifact type:** audit + roadmap. No code changes.

---

## Executive Summary (≤150 words)

**Top 3 findings across all stages:**

1. **CLAUDE.md documents Tiingo as the price provider; the code actually uses `yfinance`.** Tiingo is wired only for ticker listings (`providers/market_data.py:315`). Every fix direction for the stale-data bug assumes yfinance.
2. **Three independent staleness mechanisms stack**, and the combination—not any single one—is the likely user-visible bug: (a) `PriceCache.bulk_insert_price_data` uses **insert-or-skip**, never updating an existing `(stock_id, date)` row; (b) default `end_date = datetime.now(UTC)` is normalised against the **NYSE market calendar** and silently rolled back to the last trading day, so that truncated value becomes the cache's durable "newest"; (c) `quick_cache` TTL (300s) persists across a long-lived STDIO process with no hit/miss visibility.
3. **Tool surface is larger than CLAUDE.md claims (95 vs "60+"), 22 of them live in `server.py` outside `tool_registry.py`** (the documented single source of truth), and **0/72 router tools set `description=`** on `@mcp.tool`.

**Stale-data root cause:** High-confidence: insert-or-skip + NYSE trading-day adjustment. Verification steps in Stage 2.

**Roadmap at a glance:**
- **Phase 1** — Stale-data fix & data-freshness guarantees
- **Phase 2** — MCP protocol hygiene (tool descriptions, registry unification, error envelope)
- **Phase 3** — Router & utility consolidation with deprecation shims (no tool loss)
- **Phase 4** — Finish WIP: bound resilience, complete health tests, gate dep-smoke
- **Phase 5** — Observability & performance hardening

---

## Stage 1 — MCP Protocol & Best-Practices Sweep

Lens: FastMCP 2.0 patterns + `build-mcp-server` skill conventions. Evidence is file-path-anchored.

### Critical

- **22 `@mcp.tool` decorators live directly in `server.py` (`api/server.py:845–1401`)**, bypassing `tool_registry.py`. CLAUDE.md and `docs/ARCHITECTURE.md` name `tool_registry.py` as the sole registration point. Any tool-loss CI gate that iterates the registry will **not** see these 22; any consolidation effort that assumes "the registry is the truth" will miss them. These are mostly demo/overview tools (`get_user_portfolio_summary`, `get_watchlist`, `get_market_overview`, etc.).
- **Error envelope is not unified.** `api/error_handling.py:393–398` defines `create_error_handlers()` but `server.py` never invokes it. Routers return `dict[str, error]` ad hoc in ~80 call sites across `portfolio.py`, `data.py`, `research.py`, mixed with bare exceptions and dataclass returns. LLMs cannot rely on a stable error shape for tool-result handling.
- **`0/72` router-layer `@mcp.tool` decorators set `description=`.** 18/72 set `name=`; 54/72 rely on function name only. FastMCP exposes the docstring as the tool description, but many docstrings say *what the function returns* rather than *when an LLM should pick it* over a sibling (e.g., `fetch_stock_data` vs `fetch_stock_data_enhanced` vs `fetch_stock_data_batch`).

### High

- **`list[str]` violations still ship.** `scripts/check_mcp_list_types.py` is wired into `make check` (Makefile:162) and `docs/runbooks/mcp-client-serialization.md` mandates `StrList`/`OptionalStrList`, but `api/routers/screening_parallel.py` declares bare `symbols: list[str]` in two places. Either the checker has a scoping bug or the rule is being bypassed.
- **Readiness endpoint never enforces `min_tools`.** `server.py:350–365` returns `{ready: true, tool_count: N}` at 200 regardless of `N`. `scripts/smoke_test_dev.sh:100–104` enforces a client-side floor, which means a server that boots with half its routers failing still passes `/health/ready`.
- **`custom_route` is gated by `hasattr`.** `server.py:332` guards the readiness route registration on `hasattr(_fastmcp_instance, "custom_route")`. If the guard fires under a transport that doesn't support custom routes, there is no fallback — readiness probes silently 404.

### Medium

- **Docstring → schema fidelity is weak.** Spot-checked `portfolio.portfolio_add_position`, `research.research_company`, `data.fetch_stock_data`, `backtesting.run_backtest`, `signals.create_alert`. Rating: 1 Good, 2 Adequate, 2 Poor. None describe tool-selection criteria. None document return-key contracts.
- **`with_error_handling` decorator (`error_handling.py:402`) is defined but unused.** Context extraction is dead.
- **OTEL version guard (`scripts/check_otel_versions.py`) is correct** and the runbook (`docs/runbooks/otel-protobuf-crash.md`) is accurate. No finding — affirming this part of the WIP.

### Low

- `smoke_test_dev.sh` uses a 90s polling budget with no backoff/jitter — flakes on a cold CI runner.
- Resource usage field rename (`process_cpu_percent` vs legacy `cpu_percent`) in `monitoring/health_monitor.py:210` lacks a schema test; if `health_enhanced._get_resource_usage()` diverges, the alerting branch breaks silently.

### Tool inventory (load-bearing for Phase 3 "no tool loss" constraint)

| Router / module | `@mcp.tool` count | Notes |
|---|---:|---|
| `tool_registry.py` | 15 | Registration entrypoint |
| `server.py` (inline) | 22 | **Outside registry** |
| `backtesting.py` | 14 | All 14 `@mcp.tool()` bare |
| `health_tools.py` | 8 | All 8 bare |
| `signals.py` | 7 | Named |
| `journal.py` | 6 | Named |
| `watchlist.py` | 5 | Named |
| `risk_dashboard.py` | 4 | Named |
| `screening_pipeline.py` | 4 | Named |
| `research.py` | 3 | Bare |
| `intelligent_backtesting.py` | 3 | Bare |
| `introspection.py` | 3 | — |
| **Total** | **~94–95** | CLAUDE.md claim of "60+" is stale |

> **Validation gate (Stage 1):** Findings grouped by severity with file-path evidence. Confidence: **High** for counts and registration location; **Medium** for qualitative docstring ratings (sampled, not exhaustive). Proceed.

---

## Stage 2 — Stale-Data Root-Cause Analysis

This section is the audit's center of gravity. I traced the data path hop-by-hop, cite evidence at each hop, then form ranked hypotheses with verification steps.

### Hop 0 (pre-requisite finding): documentation/reality drift

`CLAUDE.md` names Tiingo as the price-data provider. **The code uses `yfinance`.** Tiingo appears only in `providers/market_data.py:315` as `tiingo_client.list_tickers(...)` — **ticker metadata listing only, not price data**. Every price fetch in `providers/stock_data.py:188, 237, 554` calls yfinance. This is not the stale-data bug, but it changes the fix surface: Tiingo tier/endpoint behavior is **not** a suspect; yfinance's behavior is.

### Hop 1 — Upstream (yfinance)

`providers/stock_data.py:554–573`:

```python
if end_date is None:
    end_date = datetime.now(UTC).strftime("%Y-%m-%d")
...
if not self._is_trading_day(end_dt):
    last_trading = self._get_last_trading_day(end_dt)
```

- Default `end_date` uses **UTC wall-clock date**, not US/Eastern. For callers in Asia or on servers whose UTC offset rolls into "tomorrow" before NYSE closes, "today UTC" is off by one.
- NYSE calendar adjustment (`mcal.get_calendar("NYSE")` at `stock_data.py:56`) silently rolls `end_dt` back to the last trading day **before** the cache decision.
- yfinance itself has no explicit tier-based lag for EOD; intraday is on a separate code path not on the stale-data hot path.

**Staleness mechanisms originating here:**
- UTC vs Eastern boundary bug: near midnight UTC on a Monday, `end_date` can resolve to a Sunday date that the calendar then rolls to Friday.
- The "last trading day" adjustment is the value that gets **persisted as the cache's newest row**. Once persisted, it never advances until `end_dt > cached_end` **and** there's a trading day in between — see Hop 2.

**Confirm/rule out:** print `end_date` computed inside `get_stock_data` around 06:00 UTC on a Monday vs 14:00 UTC on the same Monday. If they differ and the earlier one rolls to Friday, this hop contributes.

### Hop 2 — Provider cache-vs-network decision

`providers/stock_data.py:155–178`:

```python
if end_dt > cached_end:
    if self._is_trading_day_between(cached_end, end_dt):
        missing_end_trading = self._get_trading_days(cached_end + timedelta(days=1), end_dt)
        if len(missing_end_trading) > 0:
            missing_ranges.append(...)
if not missing_ranges:
    logger.info(f"Cache hit! Returning {len(cached_df)} cached records for {symbol}")
    return cached_df.loc[mask]
```

- Decision is "no missing trading days between cached end and requested end → serve cache."
- This interacts with Hop 1: if `end_dt` has already been rolled back to the last trading day, `cached_end == end_dt` in most weekend/holiday cases, and `missing_ranges` is empty. **Cache wins unconditionally once the first weekend passes.**
- On Monday, if the user runs the tool before NYSE close, yfinance may return the latest EOD as Friday (Monday EOD not yet final); that Friday value gets persisted. Tuesday call with `end_dt = Tuesday` will produce `_is_trading_day_between(Friday, Tuesday) = True` so it tries to fetch — but see Hop 3.

**Confirm/rule out:** temporarily log `cached_end`, `end_dt`, and `missing_ranges` in `get_stock_data`. If a persistent stale session shows `cached_end < end_dt but missing_ranges == []` — calendar logic drift. If `cached_end == end_dt` even on a weekday evening — Hop 1 rolling end back.

### Hop 3 — SQLAlchemy `PriceCache`

`data/models.py:412–491` and `bulk_insert_price_data` (near lines 340–400):

```python
existing_query = session.query(PriceCache.date).filter(
    PriceCache.stock_id == stock.stock_id,
    PriceCache.date.in_(dates_to_check),
)
existing_dates = {row[0] for row in existing_query.all()}
...
if date_val in existing_dates:
    continue
```

- Schema: `date = Column(Date, nullable=False)` — date-only, no timezone.
- Write path is **insert-or-skip, not upsert**. **Any row already in the table is never overwritten**, even if yfinance later revises it (dividend adjustments, corporate actions, an earlier write of a provisional / mid-session bar).
- This is the single most dangerous mechanism. A provisional/partial Monday bar written mid-day becomes a permanent fixture.

**Confirm/rule out:**
```sql
SELECT stock_id, date, close, updated_at FROM mcp_price_cache
WHERE symbol = 'AAPL' AND date >= date('now', '-7 days')
ORDER BY date DESC;
```
Compare `close` against a known-good source (any financial site). Any diverging row after the user's observed "days-old" window is a smoking gun for this hypothesis.

### Hop 4 — Redis

Redis is **optional** and used via `providers/implementations/cache_adapter.py`. Grep surfaces no explicit TTL on the price-data path within the adapter; the `quick_cache` decorator is in-memory, not Redis. If the user has Redis enabled with a long default TTL, it could layer additional staleness on top of Hop 3.

**Confirm/rule out:** check `.env` for `REDIS_HOST` and inspect Redis with `redis-cli --scan --pattern 'price*' | head` + `TTL <key>`. If Redis is disabled (likely for personal-use SQLite default), rule out.

### Hop 5 — `quick_cache`

`utils/quick_cache.py:63` sets default TTL 300s; `get_sync`/`set_sync` are time-based with per-process `OrderedDict`.

- STDIO transport (Claude Desktop's recommended mode) creates a **long-lived server process**. A cached value from 09:30 persists until 09:35. This is minutes of lag, not days — but it compounds whenever Hop 3 is already serving stale.
- Cache hit/miss counters exist (`QuickCache.hits`/`misses`) but `get_stats()` is never called. **Cache effectiveness is invisible** — a key Phase 5 issue.

**Confirm/rule out:** expose the counters temporarily and observe hit rate for `get_stock_data` keys across a session.

### Hop 6 — Tool response shaping

`api/routers/data.py:40–92` — `fetch_stock_data` is thin; returns the DataFrame JSON-serialised as-is, no `.iloc[:-1]`, no "drop partial bar" logic. **Not a staleness source.**

### Cross-hop interactions (how the bug presents as "days old")

The user-visible "days old" is not any single hop; it's:

1. Friday evening: user queries → yfinance → writes Friday bar to `PriceCache`. `quick_cache` holds for 5 min.
2. Saturday: user queries. `end_dt` normalised to Friday. `cached_end == end_dt`. `missing_ranges = []`. **Cache hit. Correct but static.**
3. Sunday: same.
4. Monday pre-open: `end_dt = Monday`, rolled back to Friday (markets not yet open per calendar *is_trading_day* semantics on some mcal versions / at certain hours). Same path as weekend.
5. Monday post-open intraday: `end_dt = Monday`, `_is_trading_day_between(Friday, Monday) = True` for Monday only. Missing range resolves; yfinance returns data through Friday (Monday EOD not available). Code attempts `bulk_insert` — but Friday already exists → **skip**. No new row for Monday because yfinance had nothing final for Monday yet. So cache still ends at Friday.
6. Monday after close: yfinance now has final Monday EOD. `missing_ranges` detects Monday. `bulk_insert` tries Friday (skip) and Monday (new row — written). Good.
7. **But if step 6 hit the `PriceCache` while Monday bar was still provisional (yfinance quirk), the provisional row is written and then Hop 3's insert-or-skip guarantees it is never corrected.** From this point the user sees "days-old" until the table is cleaned.

### Ranked hypotheses

**H1 — Insert-or-skip semantics make `PriceCache` a write-once-per-date cache (HIGH confidence).**
Evidence: `bulk_insert_price_data` `if date_val in existing_dates: continue`. Any provisional write (including seed data) is immortal.

**H2 — Default `end_date` uses UTC and is silently rolled back by the NYSE calendar adjustment (MEDIUM-HIGH confidence).**
Evidence: `stock_data.py:562` + `:568–573`. Explains weekend flatness of the cache and any single-day boundary miss.

**H3 — Pre-seeded S&P 500 data is the "days-old" surface for screening/data tools because the seed rows were written once and H1 ensures they never refresh (MEDIUM confidence).**
Evidence: CLAUDE.md promises pre-seeded screening recommendations; `scripts/seed_sp500.py` presumed to write into `PriceCache` (confirm). If the user's "out of date by days" is appearing in screening outputs rather than live-price outputs, this hypothesis dominates. **Distinguishing question for the user: does the staleness appear in `get_stock_data` for a non-S&P ticker, or only in screening/S&P outputs?**

### Verification plan (≤10 min)

1. **SQL staleness check** (30s): `SELECT max(date) FROM mcp_price_cache WHERE stock_id = (SELECT stock_id FROM mcp_stocks WHERE ticker_symbol='AAPL');` vs today's date. If max(date) ≥ N days old on a weekday — H1/H2/H3 all viable.
2. **Upsert probe** (2 min): manually `UPDATE mcp_price_cache SET close = 9999.99 WHERE ticker_symbol='AAPL' AND date = '<recent>';`, then call `get_stock_data("AAPL", ...)` → if 9999.99 comes back, H1 is confirmed (no refresh path exists).
3. **Timezone probe** (2 min): at 01:00 UTC on a weekday (== 21:00 ET prior weekday, post-close), invoke `get_stock_data("AAPL")` with no `end_date`. Log computed `end_date`. Repeat at 14:00 UTC. If they resolve to different anchors and one rolls back to Friday on a Monday morning — H2 confirmed.
4. **Seed origin probe** (1 min): `grep -n 'PriceCache\|bulk_insert' scripts/seed_*.py` to see whether seeding writes into `PriceCache` with fixed dates.
5. **quick_cache visibility** (3 min): temporarily log `QuickCache.hits` / `.misses` on each `get_stock_data` call. If hit-rate > 90% on cold queries, the in-memory cache is masking upstream reads.

### Proposed fix direction (direction only, not code)

- **Hop 3**: change `bulk_insert_price_data` semantics to **upsert on `(stock_id, date)`** when the source is live (guard against overwriting hand-corrected rows only if a provenance column exists; otherwise always overwrite). Add an `updated_at` column and revisit any row older than `N` trading days on read.
- **Hops 1–2**: normalise "today" against the **US/Eastern** timezone, not UTC, and make the calendar-adjustment a **pre-fetch clamp** rather than a **post-cache-decision clamp**. The request's `end_dt` should reflect what the user asked for; the cache should serve whatever rows fit within it.
- **Hop 5**: make `quick_cache` TTL **a function of the request's end-date vs "today"**: short TTL when `end_date == today` (seconds), long TTL when asking for pure history (hours). Expose hit/miss counters via a lightweight `/metrics` or structured log.
- **Hop 3 (seed)**: add a "seed freshness" flag per stock; on read, if `max(date) < today − N` and stock is in the live universe, force a refresh through the provider.

> **Validation gate (Stage 2):** I have ranked hypotheses with explicit confidence, each backed by quoted code with file:line. A skeptical reviewer would ask: "you haven't proven it's H1 vs H3." Correct — the verification steps above exist precisely to distinguish. **Proceed only after Phase 1 kick-off step 1 reproduces at least one of the three hypotheses.**

---

## Stage 3 — Four-Axis Optimization Sweep

### Axis 1 — Tool-surface quality

- **[C]** 0/72 router tools set `description=`; 28 of 72 are fully bare `@mcp.tool()`. Evidence: `backtesting.py` (14), `health_tools.py` (8), `research.py` (3), `intelligent_backtesting.py` (3).
- **[H]** Backtesting tools accept `str | int | None` for numeric params (`backtesting.py:15–27`, `fast_period` etc.), forcing downstream coercion in the hot loop.
- **[H]** Data router return types are undocumented (`data.py`, `data_enhanced.py`): callers can't predict `dict[str, list]` vs DataFrame JSON split vs domain model.
- **[M]** CLAUDE.md claims "60+ tools"; actual is ~95. Doc drift.

### Axis 2 — Code architecture

- **[C]** **Screening router quadruplication.** `screening.py` (238 LOC, 5 fns), `screening_ddd.py` (485 LOC, 3 fns), `screening_parallel.py` (296 LOC, 2 fns), `screening_pipeline.py` (170 LOC, 4 tools). ~1200 LOC with minimal reuse. Each is a separate implementation of overlapping concerns.
- **[C]** **Circuit-breaker triplication.** `utils/circuit_breaker.py` (946 LOC), `utils/circuit_breaker_decorators.py` (329 LOC), `utils/circuit_breaker_services.py` (326 LOC). Each redefines `CircuitBreakerConfig`, metrics, and state enums.
- **[H]** **Data router duplication.** `data.py` and `data_enhanced.py` each define ~7 functions with near-identical names. Both are registered into the MCP surface. LLMs see both.
- **[H]** **Technical router triplication.** `technical.py` (458), `technical_ddd.py` (174), `technical_enhanced.py` (429). Same pattern.
- **[H]** **DDD layer is near-unused.** `domain/` has 39 files (entities/services/value_objects); routers import `domain` only ~4 times. DDD is shelfware.
- **[M]** `server.py` (89-line WIP diff) is still the hot spot for boot-time logic — registers routers, defines 22 inline tools, and handles transport dispatch. No try/except around `include_router()` calls — a single router import failure crashes the entire server.
- **[M]** `utils/` has 35+ files; dead-code sweep warranted (`logging_example.py`, overlapping `logging.py` / `mcp_logging.py` / `structured_logger.py`).

### Axis 3 — Runtime performance (single-user path only)

- **[H]** **Quick-cache hit rate is unobservable** in production. `QuickCache.hits`/`misses` exist; `get_stats()` is never called anywhere (`utils/quick_cache.py:29–48`). Operators can't tell whether caching is helping.
- **[H]** **Backtesting `str | int` coercion inside a vectorised path** (`vectorbt_engine.py` is 946 LOC) forces per-call type checks in the hot loop.
- **[M]** **No explicit batch load** for screening-result joins with stock info. If `get_maverick_recommendations` iterates results and lazily loads each `Stock`, N+1 on every call — worth measuring with SQL echo.
- **[M]** **Startup time**: `server.py` performs all router registrations synchronously at import time; any heavy pandas/vectorbt import cost is paid up front, including for STDIO clients that may make one query and exit.
- **[L]** No synchronous `requests.` / `time.sleep` inside async defs on the spot-checked hot paths. Async hygiene is reasonable.

### Axis 4 — Protocol hygiene gaps beyond Stage 1

- **[M]** **`include_router()` has no error boundary.** If any router import fails at boot, the entire server crashes with no partial-degradation mode. Given 28 router files and 22 inline tools, blast radius is wide.
- **[M]** **Pydantic error messages are unhelpful** for the `str | int | None` union: a malformed numeric string yields `Input should be a valid integer OR valid string` — confusing for LLM clients.
- **[L]** Commented-out `print()` calls in `server.py:398–406`. Flagged for removal so they can't be reintroduced by mistake in an STDIO path.

### Top 5 "if I could fix only 5 things"

1. **Fix the stale-data root cause** (Phase 1). Nothing else matters if tools return wrong data.
2. **Add `description=` to every `@mcp.tool`** with a "when to use" sentence — LLM tool selection depends on it.
3. **Unify tool registration** — bring the 22 `server.py` inline tools under `tool_registry.py` via a new `register_demo_tools` module.
4. **Expose `quick_cache` stats** — one log line per N requests, or a `/metrics` JSON; cache effectiveness must be measurable.
5. **Pick a screening canonical implementation** and deprecate the other three with shims — a 1200-LOC reduction with zero tool-surface loss.

> **Validation gate (Stage 3):** Each axis has ≥3 findings; each finding has file:line evidence and severity. Proceed.

---

## Stage 4 — Phase-Level Roadmap

Five phases. Order = Phase 1 (stale-data) fixed by user mandate; Phases 2–5 ordered by **impact × certainty ÷ risk**. No `@mcp.tool` is removed without a deprecation shim. Effort is honest person-weeks, not optimistic.

---

### Phase 1 — Stale-Data Fix & Data-Freshness Guarantees

**Goal:** Queries that request "today" for any ticker return data no older than the most recent completed US/Eastern trading day, always, across STDIO and HTTP transports.

**Scope (in):**
- Run Stage 2 verification plan steps 1–5 against the user's live environment. Record which hypothesis fires.
- Convert `bulk_insert_price_data` from insert-or-skip to upsert on `(stock_id, date)` with explicit `updated_at`.
- Normalise default `end_date` computation to US/Eastern, and move the NYSE calendar clamp to occur **after** the cache decision, not before.
- Add an `age` predicate to the cache-hit path: "if `cached_end.date()` < last-completed-trading-day at call time, force refresh for the tail window, regardless of `missing_ranges` emptiness."
- Make `quick_cache` TTL end-date-aware: seconds for `end_date == today`, hours for pure history.
- Add a one-line structured log per `get_stock_data` call: `{symbol, cached_end, requested_end, served_from, age_days}`.
- Audit `scripts/seed_sp500.py` (and any `seed_*`): if it writes into `PriceCache`, confirm rows are tagged with a provenance flag or guaranteed to upsert under the new write path.

**Scope (out):** Broader cache architecture rewrite (Phase 5); Redis TTL strategy for non-price data (Phase 5); changing the upstream provider away from yfinance (not in scope).

**Success criteria:**
- Verification plan step 2 (manual row clobber) reflects in tool output within one query.
- A newly booted STDIO server, queried 30 seconds after NYSE close on a weekday, returns the current trading day's close.
- Structured log shows `age_days == 0` for a same-day query in ≥95 % of calls.
- No row in `mcp_price_cache` has `max(date) < current_trading_day - 1` for an actively queried ticker.

**Validation steps:**
- `uv run pytest tests/providers/test_stock_data_simple.py tests/utils/test_quick_cache.py -v`
- `make check`
- Manual smoke: run the five Stage 2 verification steps; all should confirm the chosen hypothesis is now dead.
- New unit test: upsert behaviour (insert-then-overwrite-same-date returns the overwritten value).
- New unit test: timezone-boundary default `end_date` at UTC midnight vs 06:00 UTC.

**Affected areas:** `maverick_mcp/providers/stock_data.py`, `maverick_mcp/providers/optimized_stock_data.py`, `maverick_mcp/data/models.py` (`PriceCache` + `bulk_insert_price_data`), `maverick_mcp/data/cache.py`, `maverick_mcp/utils/quick_cache.py`, `scripts/seed_sp500.py` (audit only).

**Risk:** **Medium** — changing write semantics of `PriceCache` risks accidentally clobbering hand-curated rows. Mitigated by adding `updated_at` and a provenance check before overwrite.
**Effort:** **1.5–2 person-weeks** (most of it is verification + regression testing, not code).

**Tool-preservation note:** **No `@mcp.tool` is removed.** `get_stock_data`, `get_multiple_stocks_data`, `get_stock_info`, and every screening tool keep exact signatures. Behaviour changes; surface does not.

**Dependencies:** None; this is the fulcrum.

---

### Phase 2 — MCP Protocol Hygiene

**Goal:** Every MCP tool has a reliable description, a predictable error envelope, and a single canonical registration path. LLM clients can pick tools correctly without reverse-engineering.

**Scope (in):**
- Add `description=` (2–3 sentences: *what*, *when*, *return shape*) to all 94+ `@mcp.tool` decorators. Enforce via a `make check` script analogous to `check_mcp_list_types.py`.
- Unify error envelope: pick one (raised `MaverickError` subclass → `create_error_handlers()` converts to a stable JSON shape) and migrate routers. Keep `handle_api_error` as the single entrypoint.
- Move the 22 inline `@mcp.tool` functions from `server.py` into a new `api/routers/demo.py` (or appropriate router) and register via `tool_registry.py`.
- Wire `create_error_handlers()` into `server.py` startup.
- Harden readiness: enforce `min_tools` threshold server-side in `/health/ready`.
- Fix the `screening_parallel.py` bare-`list[str]` violations and confirm `check_mcp_list_types.py` scope includes all routers.
- Add a try/except around each `include_router()` in `server.py` so a single bad router doesn't kill the server — degrade to partial surface with a loud error log.

**Scope (out):** Router consolidation (Phase 3); tool renaming (Phase 3).

**Success criteria:**
- `scripts/check_mcp_descriptions.py` (new) passes in `make check`.
- `tool_registry.py` is the sole registration call site (`server.py` has zero `@mcp.tool`).
- A router raising on import produces a 200 response at `/health` and 503 at `/health/ready` with the failed router named.
- Error-envelope unit test: every error path returns `{error: {type, message, trace_id}}`.

**Validation steps:** `make check`; `uv run pytest tests/test_api_*.py -v`; manual transport smoke (`make dev-stdio`, `make dev`, `make dev-sse`).

**Affected areas:** `maverick_mcp/api/server.py`, `maverick_mcp/api/routers/*.py`, `maverick_mcp/api/error_handling.py`, `scripts/check_mcp_*.py` (new).

**Risk:** **Low** — mechanical changes, largely additive. The inline-tools move is the one non-trivial bit and is pure refactor with import-test coverage.
**Effort:** **2 person-weeks**.

**Tool-preservation note:** **No `@mcp.tool` is removed or renamed.** Inline-tools move preserves names; descriptions are additive.

**Dependencies:** Phase 1 complete (so error-envelope and log changes land on a correct data path).

---

### Phase 3 — Router & Utility Consolidation with Deprecation Shims

**Goal:** Collapse the four screening variants, two data variants, three technical variants, and three `circuit_breaker` modules into single canonical implementations — without removing any exposed tool.

**Scope (in):**
- **Screening:** pick canonical (likely `screening_pipeline.py` + the repository/DDD patterns from `screening_ddd.py`), port parallel features as flags. Keep `screening.py`, `screening_ddd.py`, `screening_parallel.py` tool names as thin delegators that call the canonical implementation. Mark delegators `deprecated` in descriptions with a removal-target phase (Phase 5).
- **Data:** merge `data.py` + `data_enhanced.py` around a single handler with optional enhancement params. Keep both tool names; deprecate `_enhanced`-suffixed variants.
- **Technical:** same pattern.
- **Circuit breaker:** one `CircuitBreaker` class with optional adapter modules. Remove duplicate config/enum definitions.
- Drop truly-dead utility files after grep confirms zero importers (`utils/logging_example.py` and siblings).
- Stand up an architectural invariant check: `scripts/check_router_variants.py` fails if new `*_enhanced` / `*_parallel` / `*_ddd` variants are added without an approved pattern.

**Scope (out):** Actual removal of deprecated tools (Phase 5); DDD layer resuscitation vs deletion (Phase 5, explicitly deferred — needs a product decision first).

**Success criteria:**
- Router LOC drops ≥30 %. Tool surface count unchanged (+/- 0).
- All four screening tools return identical results for identical inputs before and after.
- `utils/circuit_breaker*.py` collapses to one module + adapters.
- `make check` passes; no new import cycles.

**Validation steps:** `make test`; `uv run pytest tests/integration -v`; golden-file test pinning tool outputs pre- and post-consolidation.

**Affected areas:** `api/routers/screening*.py`, `api/routers/data*.py`, `api/routers/technical*.py`, `utils/circuit_breaker*.py`, `application/`, `domain/`, `infrastructure/` (read-only audit).

**Risk:** **Medium-High** — consolidation touches many call sites; risk is regressing a tool's output contract. Mitigated by golden-file tests and keeping thin delegators rather than deleting tools.
**Effort:** **3 person-weeks**.

**Tool-preservation note:** **Every `@mcp.tool` name remains registered.** Bodies are replaced with `return _canonical_implementation(...)` delegators. Descriptions gain a `[DEPRECATED — use X in Phase 5]` prefix but tools continue to work.

**Dependencies:** Phase 2 (description convention in place so deprecation notices have a home).

---

### Phase 4 — Finish the Resilience & SAST WIP

**Goal:** The `feature/resilience-and-sast` branch ships as intended, not as a 60 %-done patch.

**Scope (in):**
- **Bound VectorBTEngine retries** (`vectorbt_engine.py:181–207`): add `max_retries` and exponential backoff; confirm `@circuit_breaker` threshold is correct.
- **Complete `tests/test_health_monitor_alerting.py`**: cover memory sustained-duration branches (`_high_memory_since` None/set transition), `_handle_high_memory_usage`, disk thresholds. Assert alert-fired state, not "no exception."
- **Gate `.github/workflows/dep-smoke.yml`** as a required check in branch protection so a failed OTEL/protobuf version guard blocks merges.
- **Decide on B608 SAST posture**: either land `# nosec` annotations with per-line justification (not blanket), or refactor the flagged SQL to parameterised form. `docs/security/DEPENDENCY_AUDIT.md` has to reconcile the two paths.
- **Drop commented `print()` calls** in `server.py:398–406`.

**Scope (out):** Redis/cache-level resilience (Phase 5); comprehensive rate-limiting (deferred — not needed at single-user scale).

**Success criteria:**
- `_fetch_with_retry` has a hard retry cap; test proves it surrenders rather than looping.
- Health-monitor test file exercises every branch of `monitoring/health_monitor.py:206–230`.
- CI: a deliberately-broken `opentelemetry-*` version in a PR fails the check and blocks merge.
- `make check` + a new `uv run pytest tests/test_health_monitor_alerting.py -v` passes and covers the full state machine.

**Validation steps:** `make check`; `uv run pytest tests/test_health_monitor_alerting.py -v`; open a test PR with a broken OTEL pin to confirm the gate blocks.

**Affected areas:** `maverick_mcp/backtesting/vectorbt_engine.py`, `maverick_mcp/monitoring/health_monitor.py`, `tests/test_health_monitor_alerting.py`, `.github/workflows/dep-smoke.yml`, GitHub branch protection (external), `docs/security/DEPENDENCY_AUDIT.md`.

**Risk:** **Low** — closing gaps on known work.
**Effort:** **1 person-week**.

**Tool-preservation note:** No `@mcp.tool` signatures touched.

**Dependencies:** Phase 1 lands first so that resilience claims on the data path are validated against a non-buggy baseline. Otherwise bounded retries could mask the stale-data issue.

---

### Phase 5 — Observability, Performance, Deprecation Cleanup

**Goal:** Cache effectiveness is measurable; deprecated tools removed safely; N+1 and startup regressions caught.

**Scope (in):**
- Expose `quick_cache` hit/miss counters via a structured log and an optional `/metrics` JSON endpoint.
- Add per-tool call-count and latency logging (single log line per call).
- Surface a tool-level N+1 check: SQL echo test harness for `get_maverick_recommendations`, `watchlist_brief`, portfolio reads.
- Review `PriceCache` indexes (`stock_id`, `date`) and add composite index if missing.
- Lazy-load heavy imports (vectorbt, pandas-heavy modules) behind first-use, not server startup.
- **Remove the deprecated tool delegators** introduced in Phase 3. Announce via `CHANGELOG`, keep `v0.x` tag snapshot for rollback. Update `CLAUDE.md` tool-count claims.
- DDD layer decision: either wire it meaningfully (port ≥3 router flows to go through `domain/`/`application/`) or delete it. Pick one; document the choice.

**Scope (out):** Multi-tenant scaling, distributed caching, auth (all out of personal-use scope).

**Success criteria:**
- Hit-rate log line visible in `make tail-log`.
- SQL-echo test shows no N+1 hotspots on the top 5 tools by call volume.
- Server startup ≤1.5s on STDIO (measured before/after).
- `CLAUDE.md` tool count matches reality post-deprecation.
- DDD decision documented in `docs/ARCHITECTURE.md`.

**Validation steps:** `make check`; new perf-smoke test measuring startup time; targeted `uv run pytest` for N+1 tests; manual tool-count reconciliation against `tool_registry.py`.

**Affected areas:** `utils/quick_cache.py`, `utils/monitoring*.py`, `data/models.py`, `api/server.py`, `api/routers/*` (delegator removal), `domain/`, `application/`, `infrastructure/`, `CLAUDE.md`, `docs/ARCHITECTURE.md`.

**Risk:** **Medium** — tool removal is irreversible. Mitigated by keeping a release tag immediately prior.
**Effort:** **2 person-weeks**.

**Tool-preservation note:** **This is the only phase that removes tools.** Removals are limited to tools explicitly deprecated in Phase 3, after one release cycle of deprecation warnings. Before removal, each tool's deletion must be logged in the PR with: (a) the canonical replacement, (b) the Phase 3 commit that added the delegator, (c) evidence from log telemetry that nothing is calling the deprecated name. If any deprecated tool shows calls in the prior week of telemetry, **it is kept** — do not proceed until zero-use is confirmed.

**Dependencies:** Phases 1–4 complete. Phase 3's deprecation window must have elapsed before Phase 5 removals.

---

## Open Questions for the User

1. **Is the "out of date by days" you observed appearing in live-price tools (`get_stock_data` on a non-S&P ticker) or in screening tools (`get_maverick_recommendations`)?** This distinguishes H1/H2 from H3 in Stage 2 and shortens Phase 1 scoping.
2. **Is Redis actually enabled in your deployment?** If so, its TTL on price keys (if any) is a fourth staleness layer to fold into Phase 1.
3. **Where does `scripts/seed_sp500.py` source its prices from, and does it write into `PriceCache`?** If it writes `PriceCache` rows, Phase 1's upsert work has to cover the seed path explicitly; otherwise the seed is metadata-only and we can skip that audit.
4. **Production-agentic commit `5c4137a`** touched 9 domains — do you have a list of which of those domains are considered "load-bearing" vs "experimental"? Affects Phase 3 consolidation priorities.
5. **DDD layer intent**: is the `application/` + `domain/` + `infrastructure/` split a finished pattern you want the codebase to converge on, or an exploratory prototype that can be deleted? Phase 5's final scope depends on this.
6. **Tooling for deprecation telemetry**: do you have any existing tool-call logging beyond `tail-log`, or does Phase 5 need to stand that up from scratch?

---

*End of audit.*
