# Correctness and Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the reproduced data-loss, historical-analysis, research, and delivery defects before expanding the product.

**Architecture:** Preserve the existing domain layers and public interfaces except for the explicitly proposed response additions below. Put database coordination in the platform/data layers, financial arithmetic in domain logic, and output limits at the MCP boundary. Deliver independently reviewable fixes rather than one cross-domain rewrite.

**Tech Stack:** Python 3.12+, FastMCP 4, Pydantic, SQLAlchemy, SQLite/PostgreSQL, Redis, pandas, vectorbt, LangGraph, pytest, Ruff, ty, uv, Docker, GitHub Actions.

**Spec:** [Project review and correctness priorities](../../design-docs/2026-10-04-project-review.md). Read that review and `AGENTS.md` before execution.

**Status:** Proposed; no implementation steps have been executed. Baseline maintenance ends at `4a9c61c`, with 1,334 offline tests passing and zero open Dependabot alerts reported on October 4. Reconfirm the execution baseline rather than treating those counts as permanent.

## Global Constraints

- Keep the personal-use, local-server scope; defer new product features until the correctness work passes its acceptance checks.
- Keep Python `>=3.12`, existing dependency floors, domain import contracts, and the 500-line Python-file limit; no dependency upgrades are part of these tasks.
- Use `Decimal` for financial arithmetic; preserve existing holdings, watchlists, and journal data. Never infer a destructive migration from this plan.
- Preserve graceful optional-extra behavior: core installation registers no backtesting/research tools and performs no paid calls.
- Unit tests use injected providers, synthetic prices, disposable databases, and captured HTTP requests. No live providers or paid evaluations without separate authorization.
- New APIs, fields, policies, and constants labeled **proposed** below become requirements only when this plan is accepted; existing signatures otherwise stay unchanged.
- Use Codex subagents matched to database, quantitative, runtime, research, or delivery work; there is no vendor-specific executor requirement.
- Implement only the assigned task; preserve unrelated changes. Commit each verified task separately and obtain independent review before merging.
- Update the nearest feature/API/runbook documentation and remove resolved debt in the same implementation change; catalog added documentation and run `make docs-check`.
- Do not publish a package, image, tag, registry entry, or bundle from this plan. PyPI ownership/provenance and Docker catalog command/pin remain separate release gates.

## Review Focus

- Two independent service instances write the same previously absent portfolio/position: both acknowledged mutations must survive, including rollback and cancellation; task 1.
- Non-finite, negative, fractional, and sub-cent journal inputs must fail cleanly or retain precision without modifying a valid trade; task 2.
- Changing or appending future bars, then reusing the same strategy instance, must not change earlier features/signals; tasks 3–4.
- A partial market session, corporate-action adjustment, or expired-between-reads cache key must not become indefinitely fresh data; tasks 6 and 9.
- Empty evidence, no-loss/no-trade metrics, and unavailable optional packages must remain explicit and JSON-safe; tasks 7, 12, and 13.

---

## File Ownership and Execution Order

Existing `maverick/portfolio/{service,data,ledger}.py` owns holdings transactions; `service_journal.py`/`journal.py` owns journal writes. Backtesting strategy files own causality, `engine.py` owns metric conventions, and `tools_support.py` owns response truncation. Platform `cache.py`, `http.py`, and `db.py` own shared resilience. Keep these responsibilities intact.

Create only the focused test files explicitly identified below, plus task-local helpers if the file-size gate requires them; do not introduce a general persistence or strategy framework. File lists name the primary owners; update directly affected type consumers when an explicitly proposed field changes.

Recommended groups for separate branches/PRs:

1. Start independent tasks **1, 2, 3, 4, 5, 9, 10, 12, 13** in parallel when specialist capacity permits; schedule **11** immediately after 1. Tasks 1/11 share `db.py`; tasks 2/8/15 share portfolio types; coordinate ownership rather than concurrently editing those files.
2. Do **6** after 1/11 establish database behavior; do **7** after 3/4, and **8** after 1/2. Task **14** can run independently, then integrate against task 6's history behavior.
3. Do **15** after 3/7/12/14 so response contracts are stable; do **16** after 2/6/7/8/14/15, rerunning only affected cases.
4. Each task follows red → green → focused checks → independent review → commit. Finish with the combined gate below; do not serialize unrelated work merely to preserve task numbering.

Product decisions: task 12's insufficient-evidence contract needs explicit review before implementation because current tests accept successful zero-source research. A default screening universe remains deferred; do not silently seed one. Other explicitly proposed contracts below are reviewed as part of accepting this plan.

## Proposed Contract Acceptance

Confirm these public behavior changes when accepting the plan; they are not claims about today's interfaces:

| Task | Contract requiring acceptance |
| --- | --- |
| 2 | Positive finite unit prices/shares, `long|short` sides, at least four decimal places of price precision, and rejection of an exit date before entry. Repeated close already raises. |
| 5–6 | Container databases move to `/data`; completed adjusted history refreshes on access once 24 hours old, and failed consistency refreshes return errors. |
| 7 | Daily annualization uses 252 trading periods; undefined profit factors use nullable values and status fields throughout metrics, ranking, and analysis. |
| 8, 14 | Risk confidence becomes explicitly unavailable; support/resistance represents observed extrema over the stated window. |
| 10, 12 | Exhausted retryable HTTP statuses raise; research returns citations, and unset LLM temperature delegates to provider defaults. |
| 15 | MCP trade lists are bounded with omission counts; watchlist discovery and ensemble skip reasons become visible. |

The insufficient-evidence decision in task 12 remains a separate gate even if the other tasks are accepted. None of these changes authorizes automated trading, a new data vendor, a new model vendor, or broader hosted-service scope.

### Task 1: Serialize portfolio mutations

**Files:** Modify `maverick/portfolio/service.py`, `maverick/portfolio/data.py`, `maverick/platform/db.py`; test `tests/portfolio/test_service.py`, `tests/portfolio/test_data.py`, `tests/platform/test_db.py`; document `docs/features/portfolio.md`.
**Interfaces:** Preserve `PortfolioService.add_position(...) -> PositionPayload`, `remove_position(...) -> RemoveResult`, and `clear_portfolio(...) -> int`. Existing `get_or_create_portfolio(session, user_id, name) -> uuid.UUID` must remain safe for first creation.

- [ ] **Red:** Add `test_concurrent_adds_preserve_every_acknowledged_share`: initialize 10 shares at $100, concurrently add 1 at $100 and 2 at $100 through two service instances; `assert final.shares == Decimal("13")`. Also test first creation, mixed add/remove, clear/write ordering, and transaction rollback; coordinate before transaction entry so the test cannot deadlock on the intended lock.
- [ ] Run `uv run pytest tests/portfolio/test_service.py -k concurrent -v`; confirm failure is lost state or creation conflict, not a test timeout.
- [ ] **Green:** Serialize the complete read/modify/write transaction: SQLite obtains its write reservation before reading; PostgreSQL locks the portfolio row before reading positions. Make first-portfolio insertion conflict-safe. Keep sector/network lookups outside the transaction, and apply the same serialization boundary to add, remove, and clear.
- [ ] Run `uv run pytest tests/portfolio/test_service.py tests/portfolio/test_data.py tests/platform/test_db.py -v`; verify cancellation cannot release coordination while a worker-thread transaction is still running. Add a disposable PostgreSQL integration check for the row-lock path; do not claim that path verified solely from SQLite.
- [ ] Review for cross-process safety, Decimal cost basis, rollback, and lock ordering; document supported concurrency and commit `fix: serialize portfolio mutations`.

### Task 2: Validate journal inputs and preserve price precision

**Files:** Modify `maverick/portfolio/service_journal.py`, `maverick/portfolio/journal.py`, `maverick/portfolio/tools_journal.py`, and types if required; test `tests/portfolio/test_service_journal.py`, `tests/portfolio/test_journal.py`, `tests/portfolio/test_tools.py`; document `docs/features/portfolio.md`.
**Interfaces:** Keep `JournalService.add_trade(symbol, side, entry_price: Decimal, shares: Decimal, ...) -> JournalEntryPayload` and `close_trade(entry_id, exit_price: Decimal, ...) -> JournalEntryPayload`.

- [ ] **Red:** Add `test_subcent_trade_preserves_ten_dollar_profit`: 10,000 long shares entered at `Decimal("0.004")`, exited at `Decimal("0.005")`; `assert Decimal(str(closed.pnl)) == Decimal("10.00")`. Repeat at `0.0041`/`0.0051`; assert stored entry/exit prices retain at least four decimal places of input precision.
- [ ] Add parameterized invalid-input tests for side outside `long|short`, shares/price ≤ 0, NaN/infinity, and exit before entry; assert a `ValueError` and unchanged row counts/trade state. Run `uv run pytest tests/portfolio/test_service_journal.py -v` to record the failures.
- [ ] **Green:** Validate at the service boundary before writing, normalize documented side spelling, and retain Decimal price arithmetic through realized-P&L calculation. Quantize the resulting money amount, not unit prices; legacy storage conversions stay at the persistence boundary. Do not silently rewrite historical journal rows or change column types without a separately reviewed migration.
- [ ] Run `uv run pytest tests/portfolio/test_service_journal.py tests/portfolio/test_journal.py tests/portfolio/test_tools.py -v`; cover long/short symmetry, fractional shares, repeated close, and strategy-performance recomputation.
- [ ] Review precision and backwards-readable storage; document validation/rounding and commit `fix: preserve journal price precision and validate trades`.

### Task 3: Make ensemble weights causal

**Files:** Modify `maverick/backtesting/strategies/ml/ensemble.py`, `maverick/backtesting/strategies/ml/ensemble_voting.py`; test `tests/backtesting/test_ml_ensemble.py`.
**Interfaces:** Preserve `StrategyEnsemble.generate_signals(data: DataFrame) -> tuple[Series, Series]` and the existing strategy/weighting parameters. Weight history must be internal, indexed by the bars it can affect.

- [ ] **Red:** Add `test_future_suffix_cannot_change_prefix_signals` using real SMA/RSI/MACD templates and 300 deterministic bars; mutate only the final 100 and assert both first-200 signal Series are equal with `pd.testing.assert_series_equal`. Repeat for performance/volatility/equal weighting and a reused instance.
- [ ] Run `uv run pytest tests/backtesting/test_ml_ensemble.py -k 'prefix or reuse' -v`; confirm the historical signal mismatch.
- [ ] **Green:** Compute weights at each existing rebalance boundary from returns strictly before that boundary, using the configured lookback. Apply those weights only to that boundary's following segment; use equal weights before sufficient history. Reset run-local return/weight state and remove the full-series return accumulation that feeds earlier rebalances.
- [ ] Run `uv run pytest tests/backtesting/test_ml_ensemble.py tests/backtesting/test_service.py -v`; assert normalized finite weights and unchanged prefix signals when data is appended, with no global vectorbt settings mutation.
- [ ] Review signal/return lag and segment endpoints independently; document the convention in `docs/api/backtesting.md` and commit `fix: calculate ensemble weights without future data`.

### Task 4: Remove future information from feature warm-up

**Files:** Modify `maverick/backtesting/strategies/ml/feature_engineering.py`; inspect consumers in `ml_predictor.py` and `online_features.py`; test `tests/backtesting/test_ml_feature_engineering.py`, `tests/backtesting/test_ml_adaptive.py`.
**Interfaces:** Preserve `FeatureExtractor.extract_all_features(data: DataFrame) -> DataFrame`, feature names, index alignment, and finite numeric output.

- [ ] **Red:** Add `test_feature_prefix_is_independent_of_future_bars`: extract from a 30-bar prefix and a 100-bar extension; `pd.testing.assert_frame_equal(short, long.iloc[:30])`. Assert the unavailable first SMA-50 ratio is zero in both, and cover internal missing bars and zero volume.
- [ ] Run `uv run pytest tests/backtesting/test_ml_feature_engineering.py -k prefix -v`; confirm the backward-fill mismatch.
- [ ] **Green:** Replace backward fill with past-only forward fill followed by the existing zero fallback; ensure no feature normalization is fitted across future/test rows. Keep forward targets exclusively in training-label construction.
- [ ] Run `uv run pytest tests/backtesting/test_ml_feature_engineering.py tests/backtesting/test_ml_adaptive.py -v`; verify reproducible repeated extraction, empty/short input behavior, and train/test isolation.
- [ ] Review all fill/shift/window directions in the touched feature path; commit `fix: keep feature warm-up causal`.

### Task 5: Persist Docker user data across container replacement

**Files:** Modify `Dockerfile`, `README.md`, `.env.example`, `docs/runbooks/database-setup.md`; create `tests/structure/test_docker_persistence.py` for configuration assertions.
**Interfaces:** **Proposed container-only defaults:** `DATABASE_URL=sqlite:////data/maverick.db`, `CACHE_SQLITE_PATH=/data/maverick_cache.db`; retain existing local-process defaults and HTTP command.

- [ ] **Red:** Assert Docker defaults target `/data` and the documented quick start mounts `maverick-data:/data`; copy `.env.example` unchanged to `.env`, execute that exact `--env-file .env` command, and assert actual database/cache paths stay under `/data` rather than being overridden to `/app`.
- [ ] Run `uv run pytest tests/structure/test_docker_persistence.py -v`; expect missing defaults/mount assertions to fail.
- [ ] **Green:** Create `/data` owned by non-root UID/GID 1000, set container defaults, and document the named volume in every quick-start command. Change `.env.example`'s relative `DATABASE_URL` assignment to a commented local example so an unchanged copied file preserves image defaults; an intentional PostgreSQL value still overrides them. Do not mount over `/app`/the venv. Document backup/copy of existing `/app/*.db` before replacement.
- [ ] Build a local image and use a uniquely named disposable volume to write a portfolio, watchlist, and journal trade; stop/remove only the test container, recreate it with that volume, and assert all three survive. Confirm configured PostgreSQL and bind-mount overrides still work.
- [ ] Record the smoke-test commands/results, clean up only test-owned resources, and commit `fix: persist Docker databases in a mounted data directory`.

### Task 6: Refresh adjusted history and upsert concurrent price bars

**Files:** Modify `maverick/market_data/data.py`, `maverick/market_data/service.py`; test `tests/market_data/test_data.py`, `tests/market_data/test_service.py`; document `ARCHITECTURE.md` and `docs/runbooks/database-setup.md`.
**Interfaces:** Preserve `write_price_bars(session, symbol, df) -> int` and `MarketDataService.get_price_history(symbol, start, end) -> pd.DataFrame`. **Proposed policy:** on the next access after 24 hours, refresh completed adjusted history; no background refresh is introduced. An incomplete current-session bar is never treated as final.

- [ ] **Red:** Add tests for simultaneous initial fetches, a revised same-day close, and a split crossing a newly requested session while cached history is less than 24 hours old; assert one row per `(stock_id, date)` and a uniformly adjusted series. Also race different-range refreshes with the older response completing last; it must not overwrite the newer snapshot. Legacy rows without freshness metadata must refresh.
- [ ] Run `uv run pytest tests/market_data/test_data.py tests/market_data/test_service.py -v`; pin insert-conflict and stale-row failures with injected clock/calendar/provider.
- [ ] **Green:** Use SQLite/PostgreSQL conflict-safe upserts and proposed nullable stock fields `history_refreshed_at`/`history_generation`. Fetch the complete stored/requested union whenever stale, expanding coverage, or fetching a current-session bar, even before the TTL expires; this establishes one adjustment basis. Reserve generations in short transactions and atomically reject stale-generation commits. Advance freshness only after a complete refresh, and never mark the current-session bar final.
- [ ] Run the focused suites; cover provider failure/partial responses, legacy columns, holidays, exclusive end dates, and concurrent refreshes. Do not label legitimate pre-listing gaps a failed refresh. Preserve a prior complete snapshot on failure, do not advance freshness, and allow the next request to retry; never combine newly adjusted overlaps with retained older values on another basis or permanently disable history after one partial result.
- [ ] Review the 24-hour freshness limitation, upsert row-count semantics, and large-range refresh cost; commit `fix: refresh adjusted history with conflict-safe writes`.

### Task 7: Correct annualization, undefined profit factors, and durations

**Files:** Modify `maverick/backtesting/engine.py`, `maverick/backtesting/types.py`, `maverick/backtesting/analysis.py` and direct metric consumers; test `tests/backtesting/test_engine.py`, `tests/backtesting/test_analysis.py`, `tests/backtesting/test_types.py`.
**Interfaces:** Keep `run_backtest(...) -> BacktestResult` and `optimize_parameters(...) -> OptimizationResult`. **Proposed:** daily trading-bar returns use a 252-session annualization basis; undefined `profit_factor` becomes JSON `null` with `profit_factor_status` of `finite`, `no_losses`, `no_trades`, or `no_realized_pnl` for breakeven-only trades.

- [ ] **Red:** Add a 252-bar, fee/slippage-free fixture gaining exactly 10%; `assert metrics.annual_return == pytest.approx(0.10)`. Independently calculate Sharpe/Sortino and check both regular backtests and optimization use the same basis.
- [ ] Add profitable/no-loss, finite-profitable, no-trade, and breakeven-only candidates to a real `optimization_metric="profit_factor"` test; assert that order, with the last two below valid evidence, and distinct nullable statuses. Add a January 2–5 trade asserting `duration == "3 days 00:00:00"`; run `uv run pytest tests/backtesting/test_engine.py -v` and observe failures.
- [ ] **Green:** Pass annualization locally through both vectorbt paths; do not change global settings. Update `_get_metric_value` and optimization sorting as well as report serialization: positive-profit/no-loss candidates rank above finite ratios, while no-trade/breakeven-only candidates rank last. Propagate nullable values/status through simplified metrics and analysis without numeric null comparisons. Derive duration from timestamps; open duration is elapsed through the last observed bar.
- [ ] Run `uv run pytest tests/backtesting/test_engine.py tests/backtesting/test_analysis.py tests/backtesting/test_types.py tests/backtesting/test_optimization.py -v`; assert `json.dumps(payload, allow_nan=False)` succeeds, no-loss cases are not labeled unprofitable, and ranking follows the stated order including breakeven-only candidates.
- [ ] Review units and compatibility of nullable fields, update `docs/api/backtesting.md`, and commit `fix: report backtest metrics with explicit financial conventions`.

### Task 8: Align ATR sizing, risk labels, and correlations

**Files:** Modify `maverick/portfolio/analysis.py`, `maverick/portfolio/types.py` if required; test `tests/portfolio/test_analysis.py`, `tests/portfolio/test_tools.py`; document `docs/features/portfolio.md`.
**Interfaces:** Preserve `risk_adjusted_analysis(market_data, settings, ticker, risk_level) -> RiskAnalysis` and `correlation_analysis(...) -> CorrelationResult`. **Proposed:** risk budget stays `account × 1% × risk_level/100`; whole-share sizing is capped by account cash and risk budget divided by returned stop distance.

- [ ] **Red:** With account $100,000, price $100, risk level 50, and ATR $2, assert stop $97, target $103, 166 shares, position value $16,600, max risk $498, and reward/risk 1.0. Doubling ATR must reduce shares; reject non-finite/non-positive prices or stop distances.
- [ ] Add two distinct perfectly correlated return Series and a perturbed floating-point diagonal; assert their off-diagonal 1.0 is retained and every diagonal excluded. Run `uv run pytest tests/portfolio/test_analysis.py -v` to capture failures.
- [ ] **Green:** Calculate sizing and reported cash risk from one Decimal stop-distance calculation, then convert at the response boundary. Derive reward/risk from returned target and stop. Replace the fabricated confidence number with proposed `confidence_score: null` and an explanation that this is a sizing heuristic; use a positional diagonal mask for correlation.
- [ ] Run the focused analysis/tool suites; cover risk levels 0/100, insufficient history, zero ATR, fractional prices, constant-return symbols, and cash caps.
- [ ] Review mathematical units and response labels; commit `fix: align ATR sizing and correlation outputs with their inputs`.

### Task 9: Preserve Redis expiry on memory backfill

**Files:** Modify `maverick/platform/cache.py`; test `tests/platform/test_cache.py`.
**Interfaces:** Keep `RedisTier.get_with_expiry(key) -> tuple[bytes, float] | None` and `Cache.get(key) -> Any`; no public cache API change.

- [ ] **Red:** Add tests for Redis PTTL values 400, 0, -2, and -1; the 400-ms entry must not outlive 0.4 seconds in memory, 0/-2 must not be backfilled, and -1 must not invent a week-long expiry. Include expiry between GET and TTL reads.
- [ ] Run `uv run pytest tests/platform/test_cache.py -v`; reproduce the 604800-second backfill from a TTL-zero key.
- [ ] **Green:** Prefer `pttl` for real Redis clients. Treat nonpositive expiry as no backfill/miss as appropriate; for a nonexpiring external key, serve without adding an invented memory lifetime. Keep the documented fake-client fallback only when no TTL method exists, and update test doubles to exercise real-client semantics.
- [ ] Run the cache suite with a fake clock scoped to cache code; assert normal quote/overview TTLs, Redis outages, and SQLite backfill behavior remain intact.
- [ ] Review races and units; commit `fix: preserve Redis expiry when warming memory cache`.

### Task 10: Count exhausted HTTP error statuses as breaker failures

**Files:** Modify `maverick/platform/http.py` and, only if needed, `maverick/research/providers/searxng.py`; test `tests/platform/test_http.py`, `tests/platform/test_http_recovery.py`, `tests/research/test_searxng.py`.
**Interfaces:** Preserve `request_with_retry(...) -> httpx.Response`, including its last-response behavior. **Proposed composed behavior:** `request_resilient(...)` raises `httpx.HTTPStatusError` after exhausting statuses 429/500/502/503/504, inside the breaker callback.

- [ ] **Red:** Add `test_exhausted_503_opens_breaker`: threshold one and retries zero; first call raises `HTTPStatusError`, second raises `CircuitOpenError`, and `assert transport_calls == 1`. Parameterize retryable statuses; a 403 remains available for SearXNG's JSON-configuration hint.
- [ ] Run `uv run pytest tests/platform/test_http.py -k exhausted -v`; confirm the closed-breaker failure.
- [ ] **Green:** Raise for an exhausted retryable response within `_attempt`, before breaker completion; preserve transport errors and callers' handling of nonretryable statuses. Do not regress the merged generation/cancellation logic.
- [ ] Run `uv run pytest tests/platform/test_http.py tests/platform/test_http_recovery.py tests/research/test_searxng.py -v`; verify failed half-open probes reopen, healthy probes close, and `wait_for` deadlines use a real event-loop clock.
- [ ] Review affected callers and retry budget; commit `fix: count exhausted HTTP statuses in circuit breakers`.

### Task 11: Keep in-memory SQLite schema across connections

**Files:** Modify `maverick/platform/db.py`; test `tests/platform/test_db.py`, `tests/platform/test_config.py`, `tests/server/test_assembly.py`.
**Interfaces:** Preserve `create_engine_from_settings(settings) -> Engine` and `create_async_engine_from_settings(settings) -> AsyncEngine`; task 1's transaction guarantees must also hold for supported memory usage.

- [ ] **Red:** Add sync/async tests that create a table, close the connection, and read it through the next connection; `assert rows == [(1,)]`. Cover `sqlite:///:memory:`, `sqlite://`, and the CI-selected default.
- [ ] Run `uv run pytest tests/platform/test_db.py -k memory -v`; confirm missing-table failures.
- [ ] **Green:** Detect true SQLite memory URLs using SQLAlchemy URL parsing and use `StaticPool` with foreign-key enforcement; retain file-SQLite and PostgreSQL behavior. Coordinate memory transactions per engine, shared by all service instances rather than per service, because a shared connection does not supply transaction isolation; reconcile with task 1.
- [ ] Run the database/config/assembly suites; verify separate engines remain isolated, sync/async foreign keys work, and two simultaneous writes cannot share an active transaction unsafely.
- [ ] Review pool lifecycle and supported memory concurrency; commit `fix: preserve in-memory SQLite state across sessions`.

### Task 12: Preserve research evidence and honor requested behavior

**Files:** Modify `maverick/research/{types,service,service_support}.py`, `maverick/research/agents/graph.py`, `maverick/platform/llm.py`; test `tests/research/test_service.py`, `tests/research/test_agents_graph.py`, `tests/platform/test_llm.py`; create `tests/platform/test_llm_anthropic_client.py` if request-level coverage needs a separate file.
**Interfaces:** Preserve the three research service methods. **Proposed additions:** `citations: list[SourceCitation]` on every public success envelope; optional `LLMSettings.temperature: float | None`, where unset means omit the SDK parameter and an explicit setting remains an explicit override.

- [ ] **Red:** Feed a `ResearchReport` with three known URLs through each service method, parameterizing typed and dictionary citations; `assert result.citations == [SourceCitation.model_validate(c) for c in report.citations]`. Spy on the competitive subagent and assert it runs exactly once with `include_competitive_analysis=True`, zero times when false.
- [ ] Add offline captured-request tests for configured Anthropic models: unset temperature is absent from JSON, explicit supported `1.0` is preserved, and OpenAI-family configuration remains covered. Run the focused research/LLM suites and record failures; check current official provider constraints at implementation time without sending paid requests.
- [ ] **Green:** Carry typed citations through envelopes/tool serialization, align competitive focus values with exact router keys, and omit unspecified temperature at the factory boundary. Update `.env.example` and `docs/features/deep-research.md` to describe effective configuration and evidence fields.
- [ ] **Decision gate:** Obtain explicit product review of proposed `ResearchError(error_type="insufficient_evidence")` when no sources survive validation. If accepted, add failing all-provider-failure/all-rejected-source tests, stop before synthesis, and implement that typed error; otherwise record the unresolved contract and do not describe it as fixed.
- [ ] Run `uv run pytest tests/research tests/platform/test_llm.py tests/platform/test_llm_openai_client.py tests/platform/test_llm_anthropic_client.py -v` (omit the new filename only if request tests were placed in the existing module). Independently review evidence propagation, configuration errors, and no paid calls; commit the accepted changes.

### Task 13: Verify real core installations and complete source archives

**Files:** Modify `pyproject.toml`, `.github/workflows/ci.yml`, `tools/check_docs_catalog.py`, `tests/structure/test_package.py`, `tests/structure/test_docs_catalog.py`; create `tests/structure/test_sdist.py`; document `docs/testing/README.md`.
**Interfaces:** Installed wheel remains `maverick` only; source archives include source/configuration needed by shipped tests and Makefile targets. **Proposed:** `make docs-check` works without Git metadata by enumerating the extracted source tree using the explicit archive scope; repository mode retains tracked-file checking.

- [ ] **Red:** Build/extract an sdist outside the checkout, then collect `tests/evals` and run `tests/structure/test_docs_catalog.py`; assert they succeed without editable installs, `PYTHONPATH`, or access to repository files. Assert archives exclude `.env`/secrets, databases, caches, `.claude`, `.agent_case.json`, and generated eval runs/traces.
- [ ] Run `uv build` and the extracted-source checks in a disposable environment; record missing `evals/`, `tools/`, or Makefile-required scripts/configuration as failures.
- [ ] **Green:** Extend the explicit sdist allowlist for required source assets/root documentation; never package `.git`. Add a no-Git source-tree discovery mode to the docs checker with equivalent catalog/link checks and regressions for missing links/uncataloged files. Add a CI job installing the core-only built wheel into a fresh venv outside the checkout, asserting zero optional tools and a successful offline core call.
- [ ] Add a separate extracted-sdist CI job for collection/docs/build checks. Verify wheel imports resolve into site-packages and optional dependencies are genuinely absent; a mocked `find_spec` check is insufficient.
- [ ] Run packaging tests and existing lint/type/import gates; independently inspect both archives, preserve release gates, and commit `test: validate core wheels and extracted source archives`.

### Task 14: Give support/resistance observed-data semantics

**Files:** Modify `maverick/technical/analysis.py`, `maverick/technical/service.py`, `maverick/technical/types.py`, tool documentation; test `tests/technical/test_analysis.py`, `tests/technical/test_service.py`, `tests/technical/test_tools.py`.
**Interfaces:** Preserve `TechnicalService.get_support_resistance(ticker, days=None) -> LevelsResult`. **Proposed:** return only observed low/high levels from the requested calendar lookback; with `days=None`, retain `sr_lookback` bars as the default. Add `method="observed_range"` and `bars_analyzed` to `LevelsResult`.

- [ ] **Red:** Use deterministic bars with an older extreme inside a 90-day request but outside a 20-day request; assert returned levels differ, every level comes from observed prices, and no synthetic ±5%/±10% levels appear.
- [ ] Run `uv run pytest tests/technical/test_analysis.py tests/technical/test_service.py -v`; confirm fixed-window and synthetic-level failures.
- [ ] **Green:** Pass the actual requested window to level analysis independently of indicator warm-up padding; return observed extrema with method/window metadata. Empty or insufficient data must produce a clear existing error path, not invented prices.
- [ ] Run the technical analysis/service/tool suites; verify full-analysis output uses the same convention and that omitted `days` retains the documented default.
- [ ] Review the term support/resistance as an observed range heuristic, update user-facing descriptions, and commit `fix: derive support and resistance from the requested history`.

### Task 15: Bound tool responses and expose omitted entities

**Files:** Modify `maverick/backtesting/{tools_support,service_ml,types}.py`, `maverick/portfolio/{watchlist,service_watchlist,service,tools,types}.py`; test `tests/backtesting/test_tools_support.py`, `tests/backtesting/test_service.py`, `tests/portfolio/test_watchlist.py`, `tests/portfolio/test_tools.py`, `tests/server/test_assembly.py`.
**Interfaces:** **Proposed:** retain at most 20 trade records per nested backtest result in MCP output, with `trades_total`/`trades_returned`/`trades_truncated`; full service results remain intact. Add `PortfolioService.list_watchlists() -> list[WatchlistPayload]` and read-only tool `portfolio_watchlist_list() -> dict[str, Any]`; add ensemble `skipped_symbols` entries with symbol/reason.

- [ ] **Red:** Build a five-symbol large-trade fixture; assert every MCP trade list has at most 20 records, total counts remain exact, and the source model is unchanged. Assert a new list tool exposes watchlist IDs/names and unknown IDs return an error rather than an empty success.
- [ ] Add an ensemble test with one provider failure, one short-history symbol, and six valid requested symbols; assert every omitted symbol is named with `fetch_failed`, `insufficient_history`, or `symbol_limit`. Run the focused backtesting/portfolio tool suites to verify failures.
- [ ] **Green:** Limit trades only in the existing nested response conversion; retain the existing 60-point series limit. Implement watchlist listing through existing layers and existence checks before briefing. Report skips at their service decision points; do not infer them from missing final results.
- [ ] Run the named tests plus server assembly; update registered tool counts, annotations, and `docs/api/backtesting.md`/`docs/features/portfolio.md`. Assert strict JSON serialization and bounded fixture size without claiming a universal client token limit.
- [ ] Review additive response fields and deterministic output ordering; commit separately for backtesting output/skips and watchlist discovery if reviewers need independent acceptance.

### Task 16: Repair evaluation fixtures and preserve human labels

**Files:** Modify `evals/tool_surface/seed.py`, `tests/evals/test_seed.py`, and `evals/tool_surface/README.md`; inspect existing case files and `failure_modes.md` before selecting reruns. Do not edit `annotations.json` or historical trace data.
**Interfaces:** Preserve `build(state: str, path: Path) -> Path` and `build_all(directory: Path) -> dict[str, Path]`; seed construction remains offline and deterministic.

- [ ] **Red:** Add provenance expectations for every position/journal price/date, and ledger assertions derived with Decimal from those records; demonstrate the old NVDA examples do not satisfy the corrected fixture's documented basis. Do not assert market accuracy from an invented historical price.
- [ ] Resolve fixture prices from attributable dated source data, or explicitly mark them synthetic and ensure associated prompts/expected interpretation say so. Store the selected evidence/basis alongside the fixture definition; this is the fixture acceptance decision before changing expectations.
- [ ] **Green:** Correct seed records and offline expected arithmetic together. Run `uv run pytest tests/evals/test_seed.py -v`; build every state twice in temporary directories and verify equivalent contents, including task 2's journal behavior.
- [ ] Read `evals/tool_surface/README.md` before any `make eval-*` command. Prepare the smallest affected case list and its data/model provenance; run live/model traces only with the relevant authorization and preserve the existing harness unless separately redesigned.
- [ ] Leave batch-3 and new verdicts/annotations to the owner in the review UI. Agents may draft failure-mode groupings linked to the owner's notes, never invent labels. Commit fixture/docs/tests; record unperformed trace or human-review gates explicitly.

## Combined Acceptance and Handoff

- [ ] For each completed task, retain the red-test evidence, focused green result, independent review outcome, and its commit; do not mark a deferred product decision complete.
- [ ] Run `uv run pytest --timeout=60`, `make lint`, `make typecheck`, `uv run lint-imports`, `uv lock --check`, and `make docs-check` on the integrated branch; all must pass. CI retains the longer existing backtesting test timeout where declared.
- [ ] Run the new built-wheel core lane, extracted-sdist lane, disposable Docker persistence smoke, and supported database-concurrency checks. Report unavailable Docker/PostgreSQL checks as unverified rather than inferring success.
- [ ] Recheck tasks 3–4 with suffix-mutation/prefix-invariance tests, tasks 1–2 against acknowledged writes and Decimal arithmetic, and task 15 against serialized MCP output rather than service objects alone.
- [ ] Update the review/debt/reliability documents to distinguish fixed, verified, unverified, and deferred items. Keep the screening-universe decision and external publishing blockers open until their actual acceptance criteria are met.

The recommended implementation method is specialist Codex subagents with fresh independent reviewers per task and a final combined review. This document authorizes no execution by itself; the current deliverable is the proposed plan.
