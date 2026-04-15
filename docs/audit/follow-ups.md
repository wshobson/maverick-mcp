# Audit Roadmap — Follow-up Issue Ladder

Concrete, track-able work items for the phases that were **deliberately
deferred** by the session that shipped commits `39b665a`, `a8c92fa`, and
`e01be56`. File these as GitHub issues (or your tracker equivalent) in
the order below; the order preserves the impact × certainty ÷ risk
ranking from the source audit document.

Each item includes:
- **Evidence** — where the audit flagged this.
- **Scope** — what "done" looks like.
- **Success criteria** — how to verify in CI or by inspection.
- **Effort (honest)** — person-days, not optimistic.

The source of truth for context is
[`docs/audit/2026-04-14-mcp-audit-roadmap.md`](./2026-04-14-mcp-audit-roadmap.md).

---

## Phase 2 — MCP Protocol Hygiene

### 2.1 Add `description=` to every `@mcp.tool` decorator

**Evidence:** `scripts/check_mcp_descriptions.py` currently flags 34
tools with no `description=` and a docstring first line under 8 words.
Decorator location split across `tool_registry.py` (15), `server.py`
inline (22), `backtesting.py` (14), `health_tools.py` (8), and
`research.py` (3).

**Scope:**
- Rewrite each flagged tool's decorator to set `description=` with a
  2–3-sentence statement of *what it does*, *when to pick it over
  similarly-named siblings*, and *what return keys to expect*.
- Promote the canonical text from the function docstring so the two
  cannot drift.
- Flip `scripts/check_mcp_descriptions.py` to `--strict` in `make
  check` once zero findings remain.

**Success criteria:**
- `uv run python scripts/check_mcp_descriptions.py --strict` exits 0.
- `make check` includes the strict variant.

**Effort:** **3–4 person-days** — this is mostly writing copy and
reviewing it with the LLM-tool-selection use-case in mind.

---

### 2.2 Migrate the 22 inline `@mcp.tool` functions from `server.py`

**Evidence:** Audit Stage 1 finding *"22 inline tools bypass the
tool_registry single source of truth"*. Any tool-loss CI gate that
walks the registry misses these.

**Scope:**
- Create `maverick_mcp/api/routers/demo.py` (or `composite.py`).
- Move the inline tool bodies — they all delegate to router
  implementations via `from maverick_mcp.api.routers.X import fn as _fn`.
- Add `register_demo_tools(mcp)` in the new file; call from
  `tool_registry.register_all_router_tools`.
- Add a **golden-file test** per migrated tool: invoke the tool with a
  canonical input pre-migration, capture the JSON response, then assert
  byte-equivalent after migration.
- Keep the exact same tool names so clients do not see breakage.

**Success criteria:**
- `server.py` has zero `@mcp.tool` / `@mcp.resource` decorators.
- `ripgrep '@mcp\.tool' maverick_mcp/api/server.py` returns nothing.
- Each migrated tool has a golden-file fixture under
  `tests/fixtures/migrated_tool_responses/`.
- The total tool count reported by `list_tools()` before and after is
  identical.

**Effort:** **5–7 person-days** — the migration is mechanical; the
golden-file harness is where the time goes.

---

### 2.3 Unify the error envelope

**Evidence:** `api/error_handling.py:393–398` defines
`create_error_handlers()` but `server.py` never invokes it. Routers
return `dict[str, error]` in ad-hoc shapes across ~80 call sites.

**Scope:**
- Call `create_error_handlers(mcp)` in server startup.
- Migrate routers to raise `MaverickError` subclasses on failure instead
  of returning `{"error": "..."}` dicts.
- Add a contract test: every `@mcp.tool` error path returns a JSON
  object with fields `{error: {type: str, message: str, trace_id: str}}`.

**Success criteria:**
- `tests/api/test_error_envelope.py` (new) asserts the shape on a
  canonical set of error scenarios.
- `grep -rn '"error":' maverick_mcp/api/routers/` returns close to zero
  (allowing a handful of legitimate `{"error_count": N}` style fields
  that aren't error envelopes).

**Effort:** **4–5 person-days**.

---

## Phase 3 — Router & Utility Consolidation

### 3.1 Consolidate screening variants

**Evidence:** `scripts/check_router_variants.py` flags
`screening.py`, `screening_ddd.py`, `screening_parallel.py`,
`screening_pipeline.py`. Audit Stage 3 finding *"four implementations
of overlapping concerns, ~1200 LOC with no reuse"*.

**Scope:**
- Pick a canonical implementation (recommendation in audit: start from
  `screening_pipeline.py` + the repository pattern from
  `screening_ddd.py`).
- Port parallel-execution features from `screening_parallel.py` behind
  a flag.
- Keep all `@mcp.tool` names in all four files, with bodies replaced by
  `return _canonical_impl(...)` delegators; add
  `deprecated=True` to the decorator description so clients see the
  marker.
- Add golden-file tests per tool (same as 2.2).

**Success criteria:**
- Total LOC in the four files drops ≥ 30%.
- All four original tool names still register.
- Golden-file suite passes pre- and post-consolidation.

**Effort:** **8–10 person-days**. High regression risk — this is the
largest single deferred item.

---

### 3.2 Consolidate `technical` + `data` + `health` variants

Same pattern as 3.1, applied to:
- `technical.py` / `technical_ddd.py` / `technical_enhanced.py`
- `data.py` / `data_enhanced.py`
- `health.py` / `health_enhanced.py` / `health_tools.py`

**Effort:** **6–8 person-days combined**.

---

### 3.3 Collapse the three `circuit_breaker` modules

**Evidence:** Audit Stage 3 finding *"Circuit breaker triplication:
946 + 329 + 326 LOC with overlapping `CircuitBreakerConfig`, metrics,
state enums."*

**Scope:**
- One canonical `CircuitBreaker` class.
- Adapter modules (`circuit_breaker_decorators`, `circuit_breaker_services`)
  re-export from it without redefining the core types.

**Success criteria:**
- A single definition of `CircuitBreakerConfig` in the repo.
- `import CircuitBreaker` from the old three module names still works
  (shim).

**Effort:** **3 person-days**.

---

### 3.4 Flip `check_router_variants` to `--strict`

After 3.1–3.2 land: change `make check` to run
`scripts/check_router_variants.py --strict` so a regression (a new
`_enhanced` / `_parallel` / `_ddd` / `_pipeline` variant) fails the
build.

**Effort:** **0.5 person-day** (mostly verifying no legitimate
variants remain).

---

## Tooling & Infrastructure

### T.1 Wire the `dep-smoke` workflow as a required check

**Evidence:** `docs/runbooks/otel-protobuf-crash.md` §"CI gate —
making this a blocking check".

**Scope:**
- GitHub → Settings → Branches → branch protection rule for `main`.
- Add `dep-smoke / smoke` to **Require status checks to pass before
  merging**.
- Open a test PR that breaks the OTEL version pin; confirm merge is
  blocked while the workflow is red.

**Effort:** **0.5 person-day**. **Not code — GitHub UI action**.

---

### T.2 Re-enable coverage by default (beartype unblock)

**Evidence:** Validation run showed `pytest --cov` crashing with
`ImportError: cannot import name 'claw_state' from partially initialized
module 'beartype.claw._clawstate'` under the default
`sys.settrace`-based coverage core.

**Scope:**
- ✅ **Done** in the current session (commit-in-progress): Makefile
  `test-cov` target sets `COVERAGE_CORE=sysmon`; `pyproject.toml`
  `[tool.coverage.run]` omits `*/beartype/*`.
- **Remaining:** document in `CONTRIBUTING.md` (or equivalent) that
  local `pytest --cov` invocations outside `make test-cov` should also
  set `COVERAGE_CORE=sysmon` if they hit the issue.

**Effort:** **0.5 person-day** for documentation.

---

### T.3 Resolve DDD layer intent

**Evidence:** Audit Stage 3 finding *"domain/ has 39 files; routers
import domain only ~4 times — shelfware or unfinished pattern?"*

**Scope:**
- Decide: wire it meaningfully (port ≥ 3 router flows through
  `application/ → domain/ → infrastructure/`) OR delete it.
- Document the decision in `docs/ARCHITECTURE.md`.

**Effort:** **1 person-day to decide; 3–5 days to execute** depending
on direction.

---

### T.4 Real dependency-vulnerability gating

**Evidence:** Validation run: `pip-audit` ran against uv's global
Python, not `.venv`. `dep-smoke` workflow only asserts version
alignment, not vulnerability absence.

**Scope:**
- Add a `pip-audit` step to `.github/workflows/dep-smoke.yml` that runs
  against the project's actual resolved set (via `uv export` +
  `pip-audit -r -`).
- Fail on any High/Critical CVE in a direct dependency.

**Effort:** **1 person-day**.

---

## Phase 1 — Production-Verification Preflight

### V.1 Run `scripts/verify_phase1_fix.py` on a live instance

**Evidence:** Phase 1 is verified **in unit tests**, not in a running
deployment.

**Scope:**
- On the target environment with a populated `PriceCache`, run
  `uv run python scripts/verify_phase1_fix.py --ticker AAPL`.
- Script clobbers a recent row to `9999.99` and re-invokes
  `get_stock_data` — the value must not survive.
- Exit 0 = upsert live end-to-end.

**Why this is last, not first:** the unit-test suite pins the upsert
semantics, but insert-or-skip-vs-upsert is the kind of behaviour that
can be subtly broken by e.g. a misconfigured Postgres index or a stale
migration. This script is a 30-second behavioural proof.

**Effort:** **0.1 person-day** — it's a single command.

---

## Priority order (first-in → last-in)

1. **V.1** — prove the Phase 1 fix is alive (0.1 pd).
2. **T.1** — lock the dep-smoke gate (0.5 pd, non-code).
3. **2.1** — add tool descriptions (3–4 pd).
4. **3.3** — collapse circuit breakers (3 pd, low-risk warmup for 3.1).
5. **2.3** — unify error envelope (4–5 pd).
6. **3.1** — screening consolidation (8–10 pd — the hard one).
7. **2.2** — inline-tools migration (5–7 pd).
8. **3.2** — technical/data/health consolidation (6–8 pd).
9. **T.3** — DDD decision + execution (1–5 pd).
10. **T.4** — pip-audit CI gate (1 pd).
11. **3.4** — flip variant check to strict (0.5 pd).
12. **T.2** — coverage doc cleanup (0.5 pd).

Total: **~35–47 person-days** of deferred work, matching the audit's
original 6–8 person-week estimate for Phases 2+3+5.
