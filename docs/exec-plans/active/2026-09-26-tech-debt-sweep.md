# Tech-debt sweep (2026-09-26) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clear every actionable line in `docs/exec-plans/tech-debt-tracker.md`, drop the dependencies nothing uses, and move the Claude GitHub workflows to Opus 5.5.

**Architecture:** Six pull requests. The workflow PR and this plan land first. Four independent PRs then run in parallel worktrees (dependencies and Dockerfile, dead market-data code, backtesting fixes, screening and portfolio fixes). The last PR brings `tests/` under the `ty` gate after the others merge, because it touches test files everywhere.

**Tech Stack:** Python 3.12, uv, hatchling, FastMCP 4, SQLAlchemy 2, pytest, ty, import-linter, Docker.

**Spec:** the audit recorded in the "Decisions" section below (Seth asked on 2026-09-26 to remove tech debt and unused dependencies, with full authority).

## Decisions

### Dependency audit

`deptry` (run over `maverick/`) and a manual check of `tests/`, `scripts/`, `Makefile`, `.github/`, and `Dockerfile` found these results on main at `97dca15`.

| Package | Where declared | Decision | Reason |
| --- | --- | --- | --- |
| uvicorn, python-multipart | core | remove | Never imported; fastmcp and mcp declare them. |
| aiofiles, psutil, cryptography, certifi, pytz | core | remove | Never imported; still installed transitively where another package needs them. |
| anthropic, openai | core | remove | Never imported; only `[research]` packages use them (langchain-anthropic, langchain-openai, exa-py). |
| greenlet | core and dev | remove, and declare `sqlalchemy[asyncio]` instead | SQLAlchemy 2.0 only pulls greenlet on x86_64 and aarch64, not on macOS arm64, so the async engine needs it; the extra says why. |
| hiredis | core | remove, and declare `redis[hiredis]` instead | redis-py loads it at runtime when present; the extra says why. |
| psycopg2-binary, aiosqlite, asyncpg | core | keep | Loaded by SQLAlchemy from the URL scheme: the sync engine uses psycopg2 for `postgresql://`, and `maverick/platform/db.py` rewrites URLs to `sqlite+aiosqlite` and `postgresql+asyncpg`. |
| aiosqlite, asyncpg | dev | remove | Duplicates of the core entries. |
| pytest-timeout, testcontainers, vcrpy, watchdog, bandit, safety, types-requests, types-pytz | dev | remove | No test, Makefile target, or workflow uses them. Removing `safety` also removes `nltk` and its open advisory GHSA-8mgp-746c-j5xp. |
| pytest, pytest-asyncio, pytest-cov, pytest-xdist, ruff, ty | dev | keep | Used by pytest config, `make test-cov`, `make test-parallel`, `make lint`, `make typecheck`. |
| numba, scipy | `[backtesting]` | remove | Never imported; vectorbt and scikit-learn declare them. |
| langchain, langchain-community | `[research]` | remove | Never imported. |
| pydantic | not declared | add to core | Imported directly by every domain's `config.py` and `types.py`. |
| langchain-core | not declared | add to `[research]` | Imported directly by the research agents and the strategy parser. |

### Tracker line dispositions

| Tracker line | Disposition |
| --- | --- |
| `setup.py` duplicates hatchling | Stale: `setup.py` no longer exists. Delete the line (Task 1). |
| Wheel build uses `include = ["*.py"]` | Stale: the wheel already declares `packages = ["maverick"]`. Delete the line (Task 1). |
| `server.json` has no package installs | Keep. The PyPI name `maverick-mcp-server` is held by another account until pypi/support#12150 resolves. |
| Dockerfile is single-stage | Fix (Task 2). |
| Default pytest filter deselects 664 tests | Stale: the filter deselects 0 of 1,179 tests since the legacy suite was deleted. Delete the line (Task 1). |
| MCP Apps chart rendering, Tasks extension, Macro (FRED) port, screening change-history | Keep. These are unbuilt features, not debt; building them is new scope. |
| `ty` clean over `maverick/` but not `tests/` | Fix (Task 9). |
| Tier-3 mover fallback runs without breaker or retry | Delete the line. The docstring on `_build_yfinance_tier` records why: routing through the breaker's `asyncio.Lock` from a worker thread deadlocked. It is a deliberate design, not debt (Task 3). |
| Capital Companion tier uses `request_with_retry` without breaker | Fix by deleting the tier. capitalcompanion.ai has served a static sunset page since 2026-09-19, so `/gainers`, `/losers`, and `/most-active` no longer exist (Task 3). |
| `get_quotes` has no consumer | Fix by deleting it (Task 4). |
| `run_screen` scores rubrics on the event loop | Fix (Task 5). |
| `pf_positions.total_cost` Numeric(20,4) | Fix (Task 6). |
| Three backtesting files at the 500-line cap | Fix (Task 8). |
| Regime detector fallback overwrites the requested method | Fix (Task 7). |
| Lock carries pandas 3.0.5, numpy 2.5.3, vectorbt 1.1.0 above the floors | Fix by raising the floors to the tested versions (Task 1). |
| Core `openai` and `anthropic` have no importer (added by #282) | Fix (Task 1). |

## Global Constraints

- Python `>=3.12`; ruff line length 88; `uv run lint-imports` must keep all contracts.
- Files under `maverick/` stay under the 500-line cap (`tests/structure/test_harness_rules.py`).
- Use `Decimal` for all money arithmetic.
- Unit tests make no external network calls.
- Remove a tracker line in the same PR that removes its debt.
- Every PR passes: `make lint`, `make typecheck`, `uv run lint-imports`, `make docs-check`, `make test`, and `uv lock --check`.
- Do not edit `docs/CATALOG.md` or this plan from a task branch; the controller owns both.

## Review Focus

1. A base install with no extras must still start and register the 37 core tools after the core SDK removals (Task 1 adds the check).
2. On macOS arm64, `greenlet` must still resolve with no platform marker once `sqlalchemy[asyncio]` replaces the direct pin (Task 1 checks the lock).
3. The Docker image must start as the non-root user, answer an MCP `initialize` over HTTP, list 52 tools, and write its SQLite database without uv or build tools present (Task 2).
4. A `.env` that still sets `CAPITAL_COMPANION_API_KEY` must not break startup, and movers must still come from finviz, then the yfinance batch (Task 3).
5. A regime detector that fell back to `threshold` on small data must retry its requested method on a later, larger fit (Task 7).

---

## PR 0: Claude workflows to Opus 5.5, and this plan

### Task 0: Move both Claude workflows to Opus 5.5

**Files:**
- Modify: `.github/workflows/claude.yml`, `.github/workflows/claude-code-review.yml`
- Create: this plan; Modify: `docs/CATALOG.md` (one row)

- [ ] **Step 1:** Replace `--model claude-opus-4-8` with `--model claude-opus-5-5` in both workflows.
- [ ] **Step 2:** Re-pin `anthropics/claude-code-action` from `787c5a0` (2026-05-23) to the commit of release `v1.0.235`, keeping the `# v1.0.235` comment style.
- [ ] **Step 3:** Run `make docs-check`. Expected: passes with the new catalog row.
- [ ] **Step 4:** Commit, open the PR, merge after CI.

## PR 1: Dependencies and Dockerfile

### Task 1: Remove unused dependencies and raise the tested floors

**Files:**
- Modify: `pyproject.toml`, `uv.lock`, `docs/exec-plans/tech-debt-tracker.md`, and any doc or comment that names a removed package
- Test: `tests/structure/test_harness_rules.py`

- [ ] **Step 1:** Edit `pyproject.toml` per the dependency audit table. Add `pydantic` and `langchain-core` with the currently locked versions as floors. Raise `numpy>=2.5.3`, `pandas>=3.0.5`, `vectorbt>=1.1.0`.
- [ ] **Step 2:** Run `uv lock` (no `--upgrade`). Expected: the lock diff only removes packages; no remaining package changes version. `greenlet` stays in the lock and `sqlalchemy[asyncio]` requires it with no platform marker; `nltk` is gone.
- [ ] **Step 3:** Add `test_removed_dependencies_stay_removed` to `tests/structure/test_harness_rules.py`, in the style of `test_pandas_ta_is_not_a_dependency_or_an_import`. Assert that no removed package name appears in `pyproject.toml`'s dependency lists.
- [ ] **Step 4:** Base-install check. Create a fresh venv, install the project with no extras, run `python -m maverick.server --help`, and count registered tools through the server assembly. Expected: 37. With `--extra backtesting --extra research`: 52.
- [ ] **Step 5:** Remove these tracker lines: `setup.py`, wheel `include`, pytest 664, lock-floor divergence, core `openai`/`anthropic`.
- [ ] **Step 6:** Run the full gate. Expected: green, test count 1,183 plus the new structural test.
- [ ] **Step 7:** Commit (`build: drop unused dependencies and raise floors to the tested versions`).

### Task 2: Multi-stage Dockerfile

**Files:**
- Modify: `Dockerfile`; also `docker-compose.yml` if it overrides the command with `uv run`
- Modify: `docs/exec-plans/tech-debt-tracker.md` (remove the Dockerfile line)

- [ ] **Step 1:** Record the current image size: `docker build -t maverick:before .` and `docker image ls maverick:before`.
- [ ] **Step 2:** Rewrite as two stages from the same `python:3.12-slim` base, so the venv's interpreter path matches. The builder copies a pinned `uv` binary from `ghcr.io/astral-sh/uv`, then runs `uv sync --frozen --no-dev --no-editable --extra backtesting --extra research` into `/app/.venv`. The runtime stage has no build-essential, libpq-dev, python3-dev, curl, or uv. It keeps the `io.modelcontextprotocol.server.name` label and the non-root `maverick` user. It sets `PATH=/app/.venv/bin:$PATH` and runs `CMD ["python", "-m", "maverick.server", "--transport", "http", "--host", "0.0.0.0", "--port", "8000"]`.
- [ ] **Step 3:** Build and verify. First, `docker run --rm maverick:after python -m maverick.server --help` exits 0. Second, run detached on `-p 8003:8000`, POST an MCP `initialize` then `tools/list` to `http://127.0.0.1:8003/mcp`, and get 52 tools. Third, as the `maverick` user inside the container, create the default SQLite database (for example by calling the portfolio schema setup) and confirm it succeeds.
- [ ] **Step 4:** Record the before and after sizes in the commit message. Commit (`build: split the Dockerfile into builder and runtime stages`).

## PR 2: Dead market-data code

### Task 3: Delete the Capital Companion mover tier

**Files:**
- Modify: `maverick/market_data/fetchers.py`, `maverick/market_data/config.py`, `tests/market_data/test_fetchers.py`, `tests/market_data/test_config.py`, `tests/market_data/test_service.py`, `.env.example`, `README.md` (the Capital Companion sentence near line 27), and any `ARCHITECTURE.md`, `docs/`, or `server.json` mention of `CAPITAL_COMPANION_API_KEY`
- Modify: `docs/exec-plans/tech-debt-tracker.md` (remove the Capital Companion line and the tier-3 line)

- [ ] **Step 1:** Delete `fetch_capital_companion`, `_CAPITAL_COMPANION_*`, `_build_capital_companion_tier`, the `external_client` parameter and `_from_external` on `MoverFetcher`, and the `ExternalClientFn` alias if nothing else uses it. `MoverFetcher` becomes two tiers: finviz, then the yfinance batch. Update its docstring and `build_mover_fetcher`'s. Drop the `settings` parameter from `MoverFetcher` and `build_mover_fetcher` only if nothing else reads it.
- [ ] **Step 2:** Delete `capital_companion_api_key` and `_resolve_capital_companion_api_key` from `MarketDataSettings`.
- [ ] **Step 3:** Delete the Capital Companion tests and update the `MoverFetcher` tests to the two-tier order. Add `test_settings_ignore_a_leftover_capital_companion_key`: with `CAPITAL_COMPANION_API_KEY` set in the environment, `MarketDataSettings()` constructs without error.
- [ ] **Step 4:** Run `uv run pytest tests/market_data -q`, then the full gate. Expected: green.
- [ ] **Step 5:** Commit (`refactor(market-data): delete the Capital Companion mover tier; the service is gone`).

### Task 4: Delete `MarketDataService.get_quotes`

**Files:**
- Modify: `maverick/market_data/service.py`, `tests/market_data/test_service.py`, `docs/exec-plans/tech-debt-tracker.md`

- [ ] **Step 1:** Delete `get_quotes` and its two tests (`test_get_quotes_happy_path_over_two_symbols`, `test_get_quotes_fails_fast_when_one_symbol_raises`). Keep `Quote` and `get_quote`.
- [ ] **Step 2:** Full gate, remove the tracker line, commit (`refactor(market-data): delete get_quotes; no tool consumes it`).

## PR 3: Screening and portfolio fixes

### Task 5: Score screening rubrics off the event loop

**Files:**
- Modify: `maverick/screening/service.py`, `tests/screening/test_service.py`, `docs/exec-plans/tech-debt-tracker.md`

- [ ] **Step 1:** Write `test_run_screen_scores_rubrics_off_the_event_loop_thread`. Patch one rubric to record `threading.get_ident()`, run `run_screen`, and assert the recorded id differs from the event-loop thread's id.
- [ ] **Step 2:** Run it. Expected: FAIL (the ids match).
- [ ] **Step 3:** Move the per-symbol rubric scoring in `run_screen` into a sync helper awaited through `asyncio.to_thread`. Data fetching stays async.
- [ ] **Step 4:** Run `uv run pytest tests/screening -q`. Expected: PASS. Remove the tracker line and commit (`perf(screening): score rubrics in a worker thread`).

### Task 6: Store portfolio cost columns at the ledger's precision

**Files:**
- Modify: `maverick/portfolio/data.py`, `tests/portfolio/test_data.py`, `docs/features/portfolio.md`, `docs/exec-plans/tech-debt-tracker.md`

- [ ] **Step 1:** Read how the ledger computes and quantizes `average_cost_basis` (now `Numeric(12, 4)`) and `total_cost` (now `Numeric(20, 4)`). Choose each column's scale so every Decimal the ledger can produce round-trips unchanged on Postgres. If the ledger does not quantize, it must: add quantization in the ledger at a documented scale.
- [ ] **Step 2:** Write `test_cost_columns_hold_the_ledger_scale`. Assert that each cost column's `type.scale` is at least the ledger's quantization scale, and that a fractional-share position (8dp shares, 4dp price) round-trips its `total_cost` exactly through `ensure_schema` plus a write and a read on SQLite.
- [ ] **Step 3:** Change the column types. Run `uv run pytest tests/portfolio -q`. Expected: PASS.
- [ ] **Step 4:** In `docs/features/portfolio.md`, add the `ALTER TABLE pf_positions ALTER COLUMN ... TYPE NUMERIC(p, s)` statements for existing Postgres databases. `ensure_schema` only creates missing tables. Remove the tracker line and commit (`fix(portfolio): store cost columns at the ledger's precision`).

## PR 4: Backtesting fixes

### Task 7: Regime detector retries its requested method

**Files:**
- Modify: `maverick/backtesting/strategies/ml/regime_detector.py`, `tests/backtesting/` (the regime detector test module), `docs/exec-plans/tech-debt-tracker.md`

- [ ] **Step 1:** Write `test_fallback_then_larger_fit_retries_requested_method`. Build a detector with a statistical method, fit it on data too small for that method (it falls back to `threshold`), then fit it on data large enough. Assert the effective method after the second fit is the requested method.
- [ ] **Step 2:** Run it. Expected: FAIL (it stays `threshold`).
- [ ] **Step 3:** Keep the requested method in its own attribute (`self.requested_method`). The fallback sets only the effective method, and each `fit_regimes` call starts from the requested method.
- [ ] **Step 4:** Run `uv run pytest tests/backtesting -q`. Expected: PASS. Remove the tracker line and commit (`fix(backtesting): let a fallen-back regime detector refit its requested method`).

### Task 8: Split the three files at the line cap

**Files:**
- Modify: `maverick/backtesting/service_ml.py` (490 lines), `maverick/backtesting/strategies/ml/ensemble.py` (499), `maverick/backtesting/strategies/ml/online_learning.py` (500)
- Create: one cohesive sibling module per file, named for what moves (snake_case)
- Modify: `pyproject.toml` import-linter contracts if a new module must be listed; `docs/exec-plans/tech-debt-tracker.md`

- [ ] **Step 1:** For each file, move one cohesive block of helpers into the new module, as a pure move with no behavior change. Each original file ends at or below 420 lines.
- [ ] **Step 2:** Run `uv run lint-imports` and `uv run pytest tests/backtesting -q`. Expected: all contracts kept, and the golden and characterization tests pass unchanged.
- [ ] **Step 3:** Remove the tracker line and commit (`refactor(backtesting): split the three modules at the line cap`).

## PR 5: `tests/` under the type gate (after PRs 1 to 4 merge)

### Task 9: Clear `ty` diagnostics in `tests/` and gate them

**Files:**
- Modify: files under `tests/` as needed; `Makefile` (`typecheck` target); `.github/workflows/ci.yml` (the `ty check` step); `docs/testing/README.md` if it describes the gate; `docs/exec-plans/tech-debt-tracker.md`

- [ ] **Step 1:** Run `uv run ty check tests` and record the count (143 on `97dca15`).
- [ ] **Step 2:** Fix each diagnostic with real types first: annotations, narrowing, correct fixture types. Use `# ty: ignore[<rule>]` with a short reason only where a test deliberately passes a wrong type or ty is wrong.
- [ ] **Step 3:** Change the Makefile target to `ty check maverick tests`. Change the CI step to the same scope and rename it `ty check (maverick/ and tests/)`.
- [ ] **Step 4:** Run `make typecheck` (expected: 0 diagnostics) and the full gate. Remove the tracker line and commit (`test: bring tests/ under the ty gate`).

## Close-out

- [ ] Move this plan to `docs/exec-plans/completed/` and update its `docs/CATALOG.md` row.
- [ ] Confirm the tracker holds only the kept lines: `server.json`, MCP Apps, Tasks extension, Macro port, screening change-history.
