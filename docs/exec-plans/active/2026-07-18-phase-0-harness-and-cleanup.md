# Phase 0: Harness and Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the enforcement harness (docs tree, import contracts, structural tests, CI gates) and delete the dead code, so later phases port code into a constrained structure.

**Architecture:** Phase 0 of the design in `docs/design-docs/2026-07-18-mcp-modernization.md`. It creates an empty `maverick/` package with mechanical guardrails around it, and removes the code the 2026-07-18 audit confirmed dead. No behavior visible to users changes.

**Tech Stack:** Python 3.12, uv, pytest, ruff, ty, import-linter, GitHub Actions.

## Global Constraints

- Python floor is 3.12. Do not change `requires-python` in this phase.
- `make test`, `make lint`, and `make docs-check` must pass after every task.
- The only new dependency allowed in this phase is `import-linter` (dev).
- Every doc added, moved, or deleted needs a row in `docs/CATALOG.md`.
- Commit after every task. Do not batch tasks into one commit.
- Do not add code to `maverick_mcp/` beyond the removals in this plan.
- Prose in new docs follows the plain style: short sentences, no em dashes.

## Decision log

Record outcomes here as tasks complete.

- 2026-07-18 Task 1: PyPI returned 200, "maverick-mcp" is taken (registered by an unrelated project). Fallback candidates: `maverick-mcp-server`, `maverickmcp`. Stopping for a maintainer decision before the packaging phase.

---

### Task 1: Verify the PyPI package name

**Files:**
- Modify: `docs/exec-plans/active/2026-07-18-phase-0-harness-and-cleanup.md` (this file, decision log)

**Interfaces:**
- Produces: a recorded decision on the distribution name that the packaging phase depends on.

- [ ] **Step 1: Check whether `maverick-mcp` is taken on PyPI**

Run: `curl -s -o /dev/null -w "%{http_code}" https://pypi.org/pypi/maverick-mcp/json`
Expected: `404` means the name is free. `200` means it is taken.

- [ ] **Step 2: Record the result**

Append one line to the decision log above, e.g.:
`- 2026-07-18 Task 1: PyPI returned 404, "maverick-mcp" is available.`

If the name is taken, record the fallback candidates `maverick-mcp-server` and
`maverickmcp`, and stop for a maintainer decision before the packaging phase.
Do not register anything on PyPI in this phase.

- [ ] **Step 3: Commit**

```bash
git add docs/exec-plans/active/2026-07-18-phase-0-harness-and-cleanup.md
git commit -m "docs: record PyPI name availability for maverick-mcp"
```

---

### Task 2: Delete the confirmed-dead modules

The 2026-07-18 audit confirmed each file below has zero references from
production code or tests. Deleting them is safe when the full suite still
passes.

**Files:**
- Delete: `maverick_mcp/backtesting/ab_testing.py`
- Delete: `maverick_mcp/backtesting/retraining_pipeline.py`
- Delete: `maverick_mcp/backtesting/strategies/ml_strategies.py`
- Delete: `maverick_mcp/api/connection_manager.py`
- Delete: `maverick_mcp/infrastructure/connection_manager.py`
- Delete: `maverick_mcp/infrastructure/sse_optimizer.py`
- Delete: `maverick_mcp/api/routers/intelligent_backtesting.py`
- Delete: `maverick_mcp/api/inspector_sse.py`
- Delete: `maverick_mcp/api/inspector_compatible_sse.py`
- Delete: `maverick_mcp/api/simple_sse.py`
- Delete: `maverick_mcp/api/openapi_config.py`
- Delete: `maverick_mcp/application/screening/dtos.py`
- Delete: `maverick_mcp/providers/optimized_screening.py`
- Delete: `maverick_mcp/providers/mocks/mock_persistence.py`
- Delete: `maverick_mcp/infrastructure/screening/repositories.py`
- Delete: `maverick_mcp/infrastructure/health/health_checker.py`
- Delete: `maverick_mcp/data/django_adapter.py`
- Delete: `maverick_mcp/monitoring/integration_example.py`
- Delete: `maverick_mcp/utils/resource_manager.py`
- Delete: `maverick_mcp/utils/tool_monitoring.py`
- Delete: `maverick_mcp/utils/monitoring_middleware.py`
- Delete: `maverick_mcp/utils/logging_example.py`
- Delete: `maverick_mcp/utils/logging_init.py`

**Interfaces:**
- Produces: a tree where every module is reachable, which Task 10's import
  contracts and later ports rely on.

- [x] **Step 1: Confirm each module is unreferenced**

Run this loop. It prints any live import of a module about to be deleted.

```bash
for m in ab_testing retraining_pipeline ml_strategies connection_manager \
         sse_optimizer intelligent_backtesting inspector_sse \
         inspector_compatible_sse simple_sse openapi_config \
         optimized_screening mock_persistence django_adapter \
         integration_example resource_manager tool_monitoring \
         monitoring_middleware logging_example logging_init; do
  grep -rn --include="*.py" -E "import ${m}\b|from .*${m} import" \
    maverick_mcp/ tests/ scripts/ | grep -v "maverick_mcp/.*/${m}.py:" || true
done
```

Expected: no output except comment lines or matches inside the files being
deleted themselves. `server.py` has a commented-out `sse_optimizer` import and
a `TYPE_CHECKING` reference to `infrastructure.connection_manager`. Remove
those two references in Step 2.

- [x] **Step 2: Remove the two dangling references in server.py**

In `maverick_mcp/api/server.py`, delete the `TYPE_CHECKING` import of
`MCPConnectionManager` (around line 146), the commented-out
`sse_optimizer` import (around line 344), the unused
`init_connection_management()` function (around lines 1611 to 1673), and the
commented-out call to it. Also delete `maverick_mcp/application/screening/`
re-exports of `dtos` if `application/screening/__init__.py` has any.

- [x] **Step 3: Delete the files**

```bash
git rm maverick_mcp/backtesting/ab_testing.py \
  maverick_mcp/backtesting/retraining_pipeline.py \
  maverick_mcp/backtesting/strategies/ml_strategies.py \
  maverick_mcp/api/connection_manager.py \
  maverick_mcp/infrastructure/connection_manager.py \
  maverick_mcp/infrastructure/sse_optimizer.py \
  maverick_mcp/api/routers/intelligent_backtesting.py \
  maverick_mcp/api/inspector_sse.py \
  maverick_mcp/api/inspector_compatible_sse.py \
  maverick_mcp/api/simple_sse.py \
  maverick_mcp/api/openapi_config.py \
  maverick_mcp/application/screening/dtos.py \
  maverick_mcp/providers/optimized_screening.py \
  maverick_mcp/providers/mocks/mock_persistence.py \
  maverick_mcp/infrastructure/screening/repositories.py \
  maverick_mcp/infrastructure/health/health_checker.py \
  maverick_mcp/data/django_adapter.py \
  maverick_mcp/monitoring/integration_example.py \
  maverick_mcp/utils/resource_manager.py \
  maverick_mcp/utils/tool_monitoring.py \
  maverick_mcp/utils/monitoring_middleware.py \
  maverick_mcp/utils/logging_example.py \
  maverick_mcp/utils/logging_init.py
```

- [x] **Step 4: Verify the suite and lints still pass**

Run: `make test`
Expected: about 900 passed, 4 skipped, 664 deselected. Zero errors.

Run: `make lint`
Expected: pass. If ruff reports unused imports created by the deletions,
remove those import lines and rerun.

- [x] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: delete confirmed-dead modules (phase 0 cleanup)"
```

---

### Task 3: Delete the zombie CQRS layer

The audit found `application/queries/`, `application/dto/`, and
`api/dependencies/` are imported only by tests, never by production code.
The tests that exercise them test nothing that runs.

**Files:**
- Delete: `maverick_mcp/application/queries/get_technical_analysis.py`
- Delete: `maverick_mcp/application/dto/technical_analysis_dto.py`
- Delete: `maverick_mcp/api/dependencies/stock_analysis.py`
- Delete: `maverick_mcp/api/dependencies/technical_analysis.py`
- Delete: the test files found in Step 1.

**Interfaces:**
- Produces: one data-access pattern in the legacy tree, so ports in later
  phases do not copy the abandoned one.

- [x] **Step 1: List the test files that import the zombie layer**

```bash
grep -rln --include="*.py" \
  -e "application.queries" -e "application.dto" -e "api.dependencies" \
  tests/ maverick_mcp/tests/
```

Expected: a short list of test files. Review each match to confirm the import
is for the zombie layer and not a coincidental name.

- [x] **Step 2: Delete the zombie modules and their tests**

```bash
git rm maverick_mcp/application/queries/get_technical_analysis.py \
  maverick_mcp/application/dto/technical_analysis_dto.py \
  maverick_mcp/api/dependencies/stock_analysis.py \
  maverick_mcp/api/dependencies/technical_analysis.py
git rm <each test file found in step 1>
```

If `application/queries/__init__.py`, `application/dto/__init__.py`, or
`api/dependencies/__init__.py` re-export the deleted names, remove those
lines. If a package directory becomes empty except `__init__.py`, delete the
directory too.

- [x] **Step 3: Verify**

Run: `make test`
Expected: pass, with a lower collected count because the deleted tests are gone.

- [x] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: delete zombie CQRS layer and its orphaned tests"
```

---

### Task 4: Remove the auth remnants

The server has no authentication, but two routers still check OAuth scopes
and the mock config plumbs a JWT secret. Remove all of it.

**Files:**
- Modify: `maverick_mcp/api/routers/technical.py:19,192-212`
- Modify: `maverick_mcp/api/routers/technical_enhanced.py:18,128-139,276`
- Modify: `maverick_mcp/providers/mocks/mock_config.py:37,107-110`
- Modify: `maverick_mcp/providers/factories/config_factory.py:63,125-127`

**Interfaces:**
- Produces: a codebase where `grep -rn get_access_token maverick_mcp/`
  returns nothing, which the security doc in Task 7 asserts.

- [x] **Step 1: Remove the premium gate from technical.py**

Delete line 19 (`from fastmcp.server.dependencies import get_access_token`).
Replace the block below (lines 192 to 212) with nothing.

```python
        # Access authentication context if available (optional for this tool)
        # This demonstrates optional authentication - tool works without auth
        # but provides enhanced features for authenticated users
        has_premium = False
        try:
            access_token = get_access_token()
            if access_token is None:
                raise ValueError("No access token available")

            # Log authenticated user
            logger.info(
                f"Technical analysis requested by authenticated user: {access_token.client_id}",
                extra={"scopes": access_token.scopes},
            )

            # Check for premium features based on scopes
            has_premium = "premium:access" in access_token.scopes
            logger.info(f"Has premium: {has_premium}")
        except Exception:
            # Authentication is optional for this tool
            logger.debug("Technical analysis requested by unauthenticated user")
```

Then run `grep -n has_premium maverick_mcp/api/routers/technical.py` and
delete any remaining line that reads or reports `has_premium`.

- [x] **Step 2: Remove the auth step from technical_enhanced.py**

Delete line 18 (the `get_access_token` import). Delete the auth-check block:

```python
    # Step 1: Check authentication (optional)
    tool_logger.step("auth_check", "Checking authentication context")
    has_premium = False
    try:
        access_token = get_access_token()
        if access_token and "premium:access" in access_token.scopes:
            has_premium = True
            logger.info(
                f"Premium user accessing technical analysis: {access_token.client_id}"
            )
    except Exception:
        logger.debug("Unauthenticated user accessing technical analysis")
```

Delete the `"has_premium": has_premium,` line from `analysis_metadata`
(around line 276). Run
`grep -n has_premium maverick_mcp/api/routers/technical_enhanced.py` and
delete any remaining reference.

- [x] **Step 3: Remove the JWT plumbing from the mocks**

In `maverick_mcp/providers/mocks/mock_config.py`, delete the
`"JWT_SECRET_KEY": "test-secret-key",` default (line 37) and the
`get_jwt_secret_key` method (lines 107 to 110).

In `maverick_mcp/providers/factories/config_factory.py`, delete the
`"JWT_SECRET_KEY": "test-secret-key",` default (line 63) and the
`get_jwt_secret_key` method (lines 125 to 127).

If an interface in `maverick_mcp/providers/interfaces/` declares
`get_jwt_secret_key`, delete the declaration too. Find it with
`grep -rn get_jwt_secret_key maverick_mcp/providers/interfaces/`.

Leave `is_auth_enabled` and `AUTH_ENABLED` alone. They are part of a wider
interface and are tracked in the tech-debt tracker instead.

- [x] **Step 4: Verify nothing remains and the suite passes**

Run: `grep -rn "get_access_token\|JWT_SECRET_KEY\|has_premium" --include="*.py" maverick_mcp/`
Expected: no output.

Run: `make test`
Expected: pass. If a test asserted on `has_premium` or the JWT mock, delete
that assertion or test, because it tested removed behavior.

- [x] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: remove vestigial auth and JWT remnants"
```

---

### Task 5: Triage and adopt the in-package test tree

`maverick_mcp/tests/` holds about 171 test functions that no make target or
CI job collects. A plain `uv run pytest maverick_mcp/tests` hung during
planning, so treat every file as suspect until it proves itself under a
timeout.

**Files:**
- Modify: `pyproject.toml:110` (testpaths)
- Modify: files under `maverick_mcp/tests/` that need markers.

**Interfaces:**
- Produces: a single collected test tree that CI runs, which Task 12 relies on.

- [ ] **Step 1: Collect without running**

Run: `uv run pytest maverick_mcp/tests --collect-only -q | tail -3`
Expected: about 171 tests collected, zero collection errors. Fix any
collection error first, because it would break `make test` for everyone.

- [ ] **Step 2: Run each file with a timeout and record the outcome**

```bash
for f in maverick_mcp/tests/test_*.py; do
  echo "=== $f"
  uv run pytest "$f" -q --timeout 60 -m "not integration and not slow and not external" 2>&1 | tail -2
done
```

Record per file: pass, fail, or hang (timeout). The `pytest-timeout` plugin
is already a dev dependency because `--timeout` is used in CI.

- [ ] **Step 3: Quarantine the files that hang or fail**

For each hanging or externally dependent file, add a module-level marker under
the imports so the default filter excludes it:

```python
import pytest

pytestmark = pytest.mark.integration  # requires a running server or network
```

For each file with genuine small failures (e.g. a stale assertion), fix the
test if the fix is obvious in under ten minutes. Otherwise mark the file
`integration` and add one line describing it to
`docs/exec-plans/tech-debt-tracker.md`.

- [ ] **Step 4: Add the tree to testpaths**

In `pyproject.toml`, change:

```toml
testpaths = ["tests"]
```

to:

```toml
testpaths = ["tests", "maverick_mcp/tests"]
```

- [ ] **Step 5: Verify the default suite passes and does not hang**

Run: `timeout 900 uv run pytest -q 2>&1 | tail -3`
Expected: completes well under the timeout, all collected tests pass, and the
collected count is higher than before this task.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "test: collect maverick_mcp/tests in the default suite with quarantine markers"
```

---

### Task 6: Fix the stale FastMCP claims in the README

**Files:**
- Modify: `README.md:5,10`

- [ ] **Step 1: Update the badge and the intro sentence**

Line 5, change the badge from `FastMCP-2.0` to `FastMCP-3`:

```markdown
[![FastMCP](https://img.shields.io/badge/FastMCP-3-green.svg)](https://github.com/jlowin/fastmcp)
```

Line 10, change `personal-use FastMCP 2.0 server` to `personal-use FastMCP
server`. The rest of the README is repositioned at cutover, not now.

- [ ] **Step 2: Verify and commit**

Run: `make docs-check`
Expected: pass.

```bash
git add README.md
git commit -m "docs: fix stale FastMCP 2.0 claims in README"
```

---

### Task 7: Scaffold the docs knowledge base

**Files:**
- Create: `docs/exec-plans/completed/.gitkeep`
- Create: `docs/exec-plans/tech-debt-tracker.md`
- Create: `docs/product-specs/index.md`
- Create: `docs/generated/README.md`
- Create: `docs/QUALITY_SCORE.md`
- Create: `docs/RELIABILITY.md`
- Create: `docs/SECURITY.md`
- Modify: `docs/CATALOG.md`, `docs/INDEX.md`

**Interfaces:**
- Produces: the docs tree that `AGENTS.md` (Task 8) points into.

- [ ] **Step 1: Create the tech-debt tracker**

Create `docs/exec-plans/tech-debt-tracker.md`:

```markdown
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
```

- [ ] **Step 2: Create the product-specs index and generated README**

`docs/product-specs/index.md`:

```markdown
# Product specs

No product specs exist yet. Add one file per user-facing behavior when the
new server's tool surface is curated, and list it here.
```

`docs/generated/README.md`:

```markdown
# Generated docs

Files in this directory are produced by scripts, not written by hand. Do not
edit them directly. The tool catalog generator lands with the new server.
```

- [ ] **Step 3: Create the quality score**

Create `docs/QUALITY_SCORE.md` with the audit-derived grades:

```markdown
# Quality score

Grades reflect the 2026-07-18 audit. Update the grade in the same change
that changes the code, and note why.

| Area | Grade | Why |
| --- | --- | --- |
| `services/` | B+ | Clean domain style, event bus, typed. The template for ports. |
| `data/` | B | Clean single-user models. `models.py` is 1,922 lines. |
| `backtesting/` | B- | Self-contained, guarded imports. Dead experiments removed in phase 0. |
| `providers/` | C | Two competing patterns (fat classes and interfaces/factories). |
| `api/server.py` | D | God-module: transports, middleware, tools, lifecycle in one file. |
| `api/routers/` | C- | Per-router FastMCP instances never mounted; hand re-registration. |
| `agents/`, `workflows/` | C | Two parallel agent abstractions, heavy vendor surface. |
| `application/`, `api/dependencies/` | removed | Zombie layer deleted in phase 0. |
| `maverick/` (new) | A | Empty and enforced. Keep it that way as code arrives. |
```

- [ ] **Step 4: Create the reliability doc**

Create `docs/RELIABILITY.md`:

```markdown
# Reliability

Current state and known gaps. Update when behavior changes.

## What exists

- Circuit breakers around external data providers.
- Rate limiting per tool category, strictest on research tools.
- Graceful-shutdown hooks for the scheduler, event bus, and caches.
- Health endpoints at `/health`, `/health/ready`, and `/health/live`.

## Known gaps

- Tool registration failures are logged and swallowed, so a broken router
  silently drops its tools. The new server fails fast instead.
- The legacy SSE transport needs a monkey-patch to serve trailing slashes.
  SSE is not carried into the new server.
```

- [ ] **Step 5: Create the security doc**

Create `docs/SECURITY.md`:

```markdown
# Security posture

Engineering rules for this codebase. Vulnerability reporting lives in the
root `SECURITY.md`.

- The server has no authentication by design. It is a local, single-user
  tool. Remote deployment is out of scope until a design doc reopens it.
- Text fetched from third parties (news, filings, web search) is untrusted
  input. Tools return it labeled as data. Never blend it into instructions,
  tool descriptions, or prompts.
- Tool annotations are UX hints, not security guarantees.
- No secrets in tool output, logs, or error messages. API keys live in
  environment variables and never leave the process.
- Integrations with third-party data services require maintainer review of
  the provider itself, not just the code. See PR #209 for the precedent.
```

- [ ] **Step 6: Catalog everything**

Add rows to the Current table in `docs/CATALOG.md`:

```markdown
| `design-docs/2026-07-18-mcp-modernization.md` | current | engineering | Approved v1.0 modernization design and migration plan. |
| `exec-plans/active/2026-07-18-phase-0-harness-and-cleanup.md` | current | engineering | Phase 0 execution plan. |
| `exec-plans/tech-debt-tracker.md` | current | engineering | Known debt, one line each. |
| `product-specs/index.md` | current | product | Product spec index, empty until the tool surface is curated. |
| `generated/README.md` | current | docs | Marker for script-generated docs. |
| `QUALITY_SCORE.md` | current | engineering | Per-area quality grades. |
| `RELIABILITY.md` | current | engineering | Reliability state and gaps. |
| `SECURITY.md` | current | engineering | Engineering security posture. |
```

(The design-doc row already exists; keep it and add the rest.) Add matching
lines to `docs/INDEX.md` under a new `## Modernization` section.

- [ ] **Step 7: Verify and commit**

Run: `make docs-check`
Expected: pass, with a higher tracked-file count.

```bash
git add docs/
git commit -m "docs: scaffold the knowledge-base tree (phase 0 harness)"
```

---

### Task 8: Point AGENTS.md at the new structure

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Add a modernization section and the new-code rule**

After the `## Project Overview` section, insert:

```markdown
## Modernization In Progress

A v1.0 rebuild is underway. Read
`docs/design-docs/2026-07-18-mcp-modernization.md` before structural work.

- `maverick/` is the new package. Code lands there through the exec plans in
  `docs/exec-plans/active/`, and import contracts and structural tests
  enforce its layering. Run `uv run lint-imports` and `make test` before
  committing.
- `maverick_mcp/` is the legacy package. It still serves users. Fix bugs
  there, but do not add features or new modules.
- `docs/exec-plans/tech-debt-tracker.md` lists known debt. Add a line when
  you find debt; remove the line when you remove the debt.
```

In the `## Project Structure` list, add one line at the top:

```markdown
- `maverick/`: the v1.0 package (in migration; see Modernization In Progress).
```

- [ ] **Step 2: Verify and commit**

Run: `make docs-check`
Expected: pass.

```bash
git add AGENTS.md
git commit -m "docs: point AGENTS.md at the modernization structure"
```

---

### Task 9: Create the empty maverick package

**Files:**
- Create: `maverick/__init__.py`
- Create: `maverick/py.typed`
- Test: `tests/structure/test_package.py`

**Interfaces:**
- Produces: importable `maverick` package with `maverick.__version__`.

- [ ] **Step 1: Write the failing test**

Create `tests/structure/__init__.py` (empty) and
`tests/structure/test_package.py`:

```python
"""Structural checks for the new maverick package."""


def test_version_is_importable():
    import maverick

    assert maverick.__version__ == "1.0.0.dev0"
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/structure/test_package.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'maverick'`.

- [ ] **Step 3: Create the package**

`maverick/__init__.py`:

```python
"""Maverick MCP v1.0 package.

Code arrives here phase by phase per
docs/design-docs/2026-07-18-mcp-modernization.md. Import contracts forbid
importing the legacy maverick_mcp package from here.
"""

__version__ = "1.0.0.dev0"
```

Create `maverick/py.typed` as an empty file.

- [ ] **Step 4: Run the test again**

Run: `uv run pytest tests/structure/test_package.py -q`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add maverick/ tests/structure/
git commit -m "feat: create empty maverick package for the v1.0 migration"
```

---

### Task 10: Enforce import contracts

**Files:**
- Modify: `pyproject.toml` (dev deps, new `[tool.importlinter]` section)
- Modify: `Makefile` (lint target)

**Interfaces:**
- Produces: `uv run lint-imports` as a passing command, used by Task 12's CI.

- [ ] **Step 1: Add the dev dependency**

Run: `uv add --dev import-linter`
Expected: `import-linter` appears in `[dependency-groups]` or the dev extra,
and `uv.lock` updates.

- [ ] **Step 2: Add the contracts**

Append to `pyproject.toml`:

```toml
[tool.importlinter]
root_packages = ["maverick", "maverick_mcp"]

[[tool.importlinter.contracts]]
name = "The new package never imports the legacy package"
type = "forbidden"
source_modules = ["maverick"]
forbidden_modules = ["maverick_mcp"]

[[tool.importlinter.contracts]]
name = "The legacy package never imports the new package"
type = "forbidden"
source_modules = ["maverick_mcp"]
forbidden_modules = ["maverick"]
```

- [ ] **Step 3: Verify the contracts pass, then prove they can fail**

Run: `uv run lint-imports`
Expected: `Contracts: 2 kept, 0 broken.`

Create `maverick/_violation.py` containing `import maverick_mcp  # noqa`.
Run: `uv run lint-imports`
Expected: FAIL reporting the forbidden import.
Delete `maverick/_violation.py` and rerun.
Expected: `Contracts: 2 kept, 0 broken.`

- [ ] **Step 4: Wire it into make lint**

In the `Makefile`, find the `lint` target and add this line to its recipe:

```make
	uv run lint-imports
```

Run: `make lint`
Expected: ruff and lint-imports both pass.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock Makefile
git commit -m "build: enforce import contracts between maverick and maverick_mcp"
```

---

### Task 11: Add the structural tests

**Files:**
- Test: `tests/structure/test_harness_rules.py`

**Interfaces:**
- Consumes: the `maverick/` package from Task 9.
- Produces: structural rules every later port must satisfy.

- [ ] **Step 1: Write the structural tests**

Create `tests/structure/test_harness_rules.py`:

```python
"""Mechanical rules for the maverick package.

Each failure message says how to fix the violation, so an agent that trips
a rule can correct itself without reading this file's history.
"""

import re
from pathlib import Path

MAVERICK = Path(__file__).resolve().parents[2] / "maverick"
MAX_LINES = 500
ENV_ALLOWED = ("config.py",)
ENV_ALLOWED_DIRS = ("platform",)


def _py_files():
    return [p for p in MAVERICK.rglob("*.py") if "__pycache__" not in p.parts]


def test_files_stay_under_the_size_cap():
    oversized = {
        str(p): n
        for p in _py_files()
        if (n := len(p.read_text().splitlines())) > MAX_LINES
    }
    assert not oversized, (
        f"Files over {MAX_LINES} lines: {oversized}. Split the file by "
        "responsibility (types, config, data, service, tools) instead of "
        "raising the cap."
    )


def test_env_access_only_in_config_or_platform():
    pattern = re.compile(r"os\.getenv|os\.environ")
    offenders = [
        str(p)
        for p in _py_files()
        if pattern.search(p.read_text())
        and p.name not in ENV_ALLOWED
        and not any(d in p.parts for d in ENV_ALLOWED_DIRS)
    ]
    assert not offenders, (
        f"Environment access outside config/platform: {offenders}. Read the "
        "value in the domain's config.py and pass it in as a parameter."
    )


def test_module_names_are_snake_case():
    bad = [
        str(p)
        for p in _py_files()
        if not re.fullmatch(r"[a-z_][a-z0-9_]*\.py", p.name)
    ]
    assert not bad, (
        f"Module names must be lowercase snake_case: {bad}. Rename the file."
    )
```

- [ ] **Step 2: Run and verify the rules pass on the empty package**

Run: `uv run pytest tests/structure/ -q`
Expected: all pass.

- [ ] **Step 3: Prove each rule can fail**

Create `maverick/BadName.py` with `import os\nX = os.getenv("HOME")` and 501
blank lines. Run `uv run pytest tests/structure/ -q` and expect all three
tests to fail with their remediation messages. Delete `maverick/BadName.py`
and rerun. Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/structure/test_harness_rules.py
git commit -m "test: add structural rules for the maverick package"
```

---

### Task 12: Wire the gates into CI

**Files:**
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `uv run lint-imports` (Task 10), `tests/structure/` (Tasks 9, 11),
  the widened testpaths (Task 5).

- [ ] **Step 1: Add lint-imports to the lint job**

In the `lint` job of `.github/workflows/ci.yml`, after the ruff steps, add:

```yaml
      - name: Import contracts
        run: uv run lint-imports
```

- [ ] **Step 2: Make type checking strict on the new package**

In the `typecheck` job, the blocking step currently scopes strict checking to
`maverick_mcp/services` and `maverick_mcp/domain`. Add `maverick` to that
command so the step reads, e.g.:

```yaml
      - name: Type check (strict scope)
        run: uv run ty check maverick maverick_mcp/services maverick_mcp/domain
```

Keep the informational full-package run as is.

- [ ] **Step 3: Verify locally what CI will run**

```bash
uv run lint-imports
uv run ty check maverick maverick_mcp/services maverick_mcp/domain
uv run pytest -q -m "not integration and not slow and not external" --timeout 60 -x
```

Expected: all pass.

- [ ] **Step 4: Commit, push, and watch CI**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: gate on import contracts, structural tests, and strict typing for maverick"
git push
gh run watch --exit-status
```

Expected: the run completes green.

---

### Task 13: Close out the phase

**Files:**
- Move: `docs/exec-plans/active/2026-07-18-phase-0-harness-and-cleanup.md`
  to `docs/exec-plans/completed/`
- Modify: `docs/CATALOG.md` (update the plan's path)

- [ ] **Step 1: Full verification**

```bash
make check
make test
make docs-check
```

Expected: all pass.

- [ ] **Step 2: Move the plan to completed and update the catalog**

```bash
git mv docs/exec-plans/active/2026-07-18-phase-0-harness-and-cleanup.md \
  docs/exec-plans/completed/
```

Update the plan's row in `docs/CATALOG.md` to the new path. Run
`make docs-check` and expect pass.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "docs: complete phase 0 (harness and cleanup)"
git push
```
