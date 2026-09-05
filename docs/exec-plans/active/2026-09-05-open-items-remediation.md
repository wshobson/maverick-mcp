# Open Items Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task, inline in the controlling session. Steps use checkbox (`- [ ]`) syntax for tracking. For the 2026-09-05 session the owner directed inline execution; any sub-agent runs on Fable 5.1 only (no Opus or Sonnet workers), and there are no agent teams and no Workflow-tool runs. **Tasks marked (P) are public, mostly irreversible actions under the owner's identity (a release tag, a PyPI publish, registry submissions). Do NOT run a (P) task without a fresh, explicit go-ahead from the owner for that task.**

**Goal:** Clear every issue and pull request open on 2026-09-05, migrate the server to FastMCP 4 so it serves the `2026-07-28` protocol revision, add a self-hosted SearXNG research backend, and prepare the v1.1.0 release.

**Architecture:** Six workstreams executed as 25 tasks. Triage replies and CI approvals first; the three contributor bug fixes and the five safe dependency bumps land in parallel; then a single FastMCP 4 migration PR; then a SearXNG provider PR built on the existing `WebSearchProvider` seam; then version bump and release notes, with the owner-gated publish steps last. Vault and repository documentation are updated as each workstream closes.

**Tech Stack:** Python 3.12, uv, FastMCP 4 (MCP Python SDK v2), pydantic 2, httpx, pytest with `asyncio_mode = "auto"`, ruff, ty, import-linter, `gh` CLI, `@hasmcp/mcp-spec-test` 0.1.5 via npx.

**Spec:** `docs/design-docs/2026-09-05-open-items-remediation.md`

## Global Constraints

- Python 3.12 or later. Ruff formatting and linting with line length 88. Domain layering `types -> config -> data -> service -> tools`; run `uv run lint-imports` after touching imports.
- The full gate for any code change is `make check` (ruff check, ruff format check, import contracts, `ty check maverick`) followed by `uv run pytest --timeout=60`. The suite includes `tests/structure`, which enforces a 500-line cap on every file under `maverick/`, forbids `os.getenv`/`os.environ` outside `config.py` files and `maverick/platform/`, and requires snake_case module names.
- `make docs-check` after adding, moving, or deleting any Markdown or text doc; every new doc needs a row in `docs/CATALOG.md`.
- No live network calls in unit tests.
- `Decimal` for financial arithmetic (nothing in this plan changes financial arithmetic).
- Every change to `main` goes through a pull request, squash-merged with `gh pr merge --squash`.
- Version floors from the spec: `fastmcp>=4.0.3`; the direct `mcp` dependency is removed; `httpx>=0.28.1` stays and is now load-bearing.
- Public actions approved by the owner on 2026-09-05: approving CI runs; commenting on, merging, and closing the PRs and issues named in this plan; pushing rebases and fixups to the contributor's branches. Not approved, each needing its own go-ahead: the `v1.1.0` tag, PyPI trusted-publishing configuration, any publish (PyPI, MCP Registry, GHCR image, bundle upload), edits to the Docker catalog PR, and third-party registry submissions.
- Repository root: `/home/wshobson/workspace/major7apps/maverick-mcp`. Worktrees go beside it (`../maverick-mcp-<name>`). Scratch output goes in the session scratchpad, never in the repository.
- `gh` is authenticated as `wshobson` with `repo` and `workflow` scopes.
- Commit messages and PR descriptions created while executing this plan from a Claude Code session end with that session's attribution line (the executing session supplies it). The fork-run approval policy on the repository is `first_time_contributors`; approving a run does not change the policy.

## Sequencing

1. Tasks 1 to 6 (triage) run first and take minutes.
2. Tasks 7 to 9 (contributor PRs, serial) and Task 10 (dependency batch) run in parallel: they touch disjoint files.
3. Tasks 11 to 15 (FastMCP 4) start after Task 10 merges, to avoid lock-file conflicts.
4. Tasks 16 to 20 (SearXNG) start after Task 15 merges.
5. Task 21 (release prep) starts after Task 20 merges. Tasks 22 and 23 wait for the owner.
6. Task 24 (vault) runs after each workstream closes; Task 25 closes the plan.

---

## Workstream 0: triage

### Task 1: Merge the design and plan docs branch

**Files:**
- Already on branch `docs/2026-09-05-remediation-design`: `docs/design-docs/2026-09-05-open-items-remediation.md`, `docs/CATALOG.md`, `docs/INDEX.md`
- Create: `docs/exec-plans/active/2026-09-05-open-items-remediation.md` (this file)
- Modify: `docs/CATALOG.md` (row for this plan), `docs/INDEX.md` (line for this plan)

- [x] **Step 1: Confirm the branch state and the docs check**

Run:
```bash
git status --short && git branch --show-current && make docs-check
```
Expected: no uncommitted changes, branch `docs/2026-09-05-remediation-design`, `Documentation catalog check passed`.

- [x] **Step 2: Push and open the PR**

Run:
```bash
git push -u origin docs/2026-09-05-remediation-design
gh pr create --title "docs: 2026-09-05 open-items remediation design and plan" --body "Approved design and the implementation plan for clearing the open issues and PRs as of 2026-09-05: contributor bug fixes (#242, #243, #244), the dependabot batch, the FastMCP 4 migration that unblocks #235, a SearXNG research backend for #186, and the v1.1.0 release prep. Docs only."
```
Expected: a PR URL. CI runs lint, docs catalog, type check, and unit tests; all four must pass (docs-only change).

- [x] **Step 3: Merge and return to main**

Run:
```bash
gh pr merge --squash --delete-branch
git checkout main && git pull --ff-only
```
Expected: `main` contains both docs; `git log --oneline -1` shows the squash commit.

### Task 2: Approve the pending CI runs on PRs 242, 243, and 244

**Files:** none (GitHub Actions state only)

- [x] **Step 1: List the runs waiting for approval**

Run:
```bash
gh api 'repos/wshobson/maverick-mcp/actions/runs?status=action_required&per_page=50' --jq '.workflow_runs[] | select(.name == "CI") | "\(.id) head=\(.head_branch) sha=\(.head_sha[0:7])"'
```
Expected (the newest run per branch is the one to approve; older shas are superseded):
```
33318171228 head=fix/backtesting-regime-analysis sha=6f583fd
33317697832 head=fix/backtesting-ensemble-strategy-mapping sha=612814d
33317566423 head=fix/backtesting-portfolio-drawdown sha=d54401d
```

- [x] **Step 2: Approve the three current CI runs**

Run:
```bash
for id in 33317566423 33317697832 33318171228; do
  gh api -X POST "repos/wshobson/maverick-mcp/actions/runs/$id/approve" && echo "approved $id"
done
```
Expected: three `approved <id>` lines with no error body. If the API answers 422 for an id, the run was superseded; re-run Step 1 and approve the id it lists for that branch.

- [x] **Step 3: Confirm the runs started**

Run:
```bash
for pr in 242 243 244; do echo "== PR $pr =="; gh pr checks $pr; done
```
Expected: each PR lists `Lint (ruff)`, `Docs catalog`, `Type check (ty, baseline)`, and `Unit tests (pytest)` as pending or running. The `Claude Code Review` workflow stays skipped for forks by design.

### Task 3: Put PR 249 on hold

**Files:** none

- [x] **Step 1: Post the hold comment**

Run:
```bash
gh pr comment 249 --body "On hold. This lock bump also moves \`fastmcp\` from 3.3.1 to 4.0.0: FastMCP 3.x pins \`mcp<2\`, so resolving \`mcp\` 2.1.1 pulled the major framework upgrade in silently. FastMCP 4 lands as a deliberate migration with a raised version floor instead (tracked on #235). This PR closes once that merges."
```
Expected: the comment URL.

### Task 4: Close #241 with a reply

**Files:** none

- [x] **Step 1: Reply and close**

Run:
```bash
gh issue close 241 --reason "not planned" --comment "Thanks for the note. The README keeps its links to installation and configuration material only, so I will not be adding a pointer. The tools are MIT-licensed and anyone is welcome to build a forecasting agent on top of them. Closing."
```
Expected: `Closed issue #241`.

### Task 5: Close #254 with a reply

**Files:** none

- [x] **Step 1: Reply and close**

Run:
```bash
gh issue close 254 --reason "completed" --comment "Thank you for the offer. MaverickMCP is a local, personal-use server by design: it runs on the user's own machine, has no network authentication surface, and reads keys only from environment variables. The model is written up in SECURITY.md and docs/SECURITY.md. I will pass on the scoping call. If you find a concrete vulnerability, a report through GitHub Security Advisories is welcome and gets a response within 48 hours. Closing as answered."
```
Expected: `Closed issue #254`.

### Task 6: Refresh the root security policy and record the regime-detector debt

**Files:**
- Modify: `SECURITY.md` (whole file)
- Modify: `docs/exec-plans/tech-debt-tracker.md` (one new row at the end of the table)

- [x] **Step 1: Branch**

Run:
```bash
git checkout -b docs/security-policy-refresh main
```

- [x] **Step 2: Replace SECURITY.md**

Write the file with exactly this content:

```markdown
# Security Policy

MaverickMCP is a personal-use, local MCP server. It has no network
authentication by design: it runs on the user's own machine, binds its HTTP
transport to `127.0.0.1` unless started with `--host 0.0.0.0`, and reads API
keys only from environment variables. The engineering rules that follow from
that model live in `docs/SECURITY.md`.

## Reporting a vulnerability

Please do not report security vulnerabilities through public GitHub issues.

Report them through
[GitHub Security Advisories](https://github.com/wshobson/maverick-mcp/security/advisories/new).
Include the affected file paths, a commit or tag, reproduction steps, and the
impact you observed. You should receive a response within 48 hours.

## Supported versions

| Version | Supported |
| ------- | --------- |
| 1.x     | yes       |
| < 1.0   | no        |

Versions before 1.0.0 shipped the legacy `maverick_mcp` package, which was
deleted at the v1.0.0 cutover and receives no fixes.

## Security model

- Local, single-user deployment. Remote deployment, authentication, and
  billing are out of scope until a design document reopens them.
- No secrets in the repository, in tool output, or in logs. Keys live in
  environment variables; see `.env.example`.
- Text fetched from third parties (market data, filings, web search) is
  untrusted input and is returned to the client labeled as data.
- Tool annotations such as `readOnlyHint` are hints for clients, not
  security guarantees.
- Dependencies are kept current through Dependabot, and `safety` is part of
  the dev extra.

## Security checklist for pull requests

- No hardcoded secrets or credentials.
- Input validated with Pydantic models at the tool boundary.
- Errors do not leak secrets or internal paths.
- API keys flow only through environment variables.
- No vulnerable dependencies introduced.
- Personal-use security model maintained: no auth or billing surface.
```

- [x] **Step 3: Add the tech-debt row**

Append this row to the table in `docs/exec-plans/tech-debt-tracker.md` (after the last existing row):

```markdown
| Regime detector fallback overwrites the requested method (`self.method = "threshold"`), so a detector held by `RegimeAwareStrategy` never retries the statistical fit on later, larger data; store the requested method separately and allow a refit | `maverick/backtesting/strategies/ml/regime_detector.py` | deferred |
```

- [x] **Step 4: Check, commit, PR, merge**

Run:
```bash
make docs-check
git add SECURITY.md docs/exec-plans/tech-debt-tracker.md
git commit -m "docs: refresh the security policy for v1.x and record regime-detector fallback debt"
git push -u origin docs/security-policy-refresh
gh pr create --title "docs: refresh security policy for v1.x" --body "The root SECURITY.md still listed 0.1.x as supported and described legacy features (secure headers, audit logging) that v1.0.0 deleted. Rewrites it around the personal-use model and points at docs/SECURITY.md for engineering rules. Also records the regime-detector fallback debt surfaced while reviewing #244."
gh pr checks --watch && gh pr merge --squash --delete-branch
git checkout main && git pull --ff-only
```
Expected: `Documentation catalog check passed`; CI green; the squash commit on `main`.

---

## Workstream 1: land the contributor fixes

The three PRs merge in order 242, 243, 244. Each PR's author allows maintainer edits, and after 242 merges the author is no longer a first-time contributor, so later pushes start CI without approval.

The local gate for a PR head, used in Tasks 7 to 9:

```bash
uv sync --extra dev --extra backtesting --extra research --frozen
make check
uv run pytest --timeout=60
```

### Task 7: Review and merge PR 242 (portfolio drawdown, fixes #245)

**Files:** none modified locally; the PR changes `maverick/backtesting/service.py`, `maverick/backtesting/tools.py`, `maverick/backtesting/types.py`, `docs/api/backtesting.md`, `tests/backtesting/test_service.py`.

- [x] **Step 1: Wait for CI**

Run:
```bash
gh pr checks 242 --watch
```
Expected: all four CI jobs pass. If a job fails, read it with `gh run view <id> --log-failed` and report the failure on the PR rather than merging.

- [x] **Step 2: Confirm the sign convention that makes `min()` correct**

Run:
```bash
sed -n '210,220p' maverick/backtesting/engine.py
grep -n 'abs(max_dd)\|abs(metrics.max_drawdown)' maverick/backtesting/engine.py maverick/backtesting/analysis.py
```
Expected: `_extract_metrics` sets `max_drawdown=_safe_float(portfolio.max_drawdown)`, which is vectorbt's signed drawdown (zero or negative), and every consumer wraps it in `abs(...)`. So the most negative value is the worst constituent and `min()` is right. If any producer emitted positive drawdowns, stop and request a normalization step on the PR instead of merging.

- [x] **Step 3: Review the diff**

Run:
```bash
gh pr diff 242
```
Check: the only code change is `max(...)` to `min(...)` at `maverick/backtesting/service.py:358` plus the comment; the three tests stub `_run_single_backtest` and assert the worst, order independence, and zero; the docs describe per-symbol averaging and worst-constituent drawdown. Nothing else.

- [x] **Step 4: Run the gate on the PR head**

Run:
```bash
gh pr checkout 242
uv sync --extra dev --extra backtesting --extra research --frozen && make check && uv run pytest --timeout=60
git checkout main
```
Expected: `All checks passed!` and a green pytest run.

- [x] **Step 5: Merge and thank the author**

Run:
```bash
gh pr merge 242 --squash --body "Fixes #245: report the worst (most negative) constituent drawdown."
gh pr comment 242 --body "Merged. Thank you for the careful report and fix; #245 closes with it."
gh issue view 245 --json state --jq .state
git checkout main && git pull --ff-only
```
Expected: `CLOSED` for #245.

### Task 8: Rebase and merge PR 243 (ensemble strategy mapping, fixes #246)

**Files:** none modified by hand; the rebase replays the contributor's two commits onto `main`.

- [x] **Step 1: Tell the author before touching the branch**

Run:
```bash
gh pr comment 243 --body "Rebasing this onto main now that #242 landed (both PRs touch docs/api/backtesting.md and tests/backtesting/test_service.py). No content changes to your commits."
```

- [x] **Step 2: Rebase onto main**

Run:
```bash
gh pr checkout 243
git rebase main
```
Expected: a clean replay of two commits. The two PRs edit different regions of the shared files (242 adds the portfolio paragraph near line 321 of the docs and tests before the ensemble section; 243 adds the ensemble paragraph near line 581 and tests after `test_create_strategy_ensemble_calls_symbols_in_order_sequentially`). If git still reports a conflict, open the file, keep both sides in order (242's block first), remove the markers, `git add` the file, and `git rebase --continue`.

- [x] **Step 3: Run the gate and push**

Run:
```bash
uv sync --extra dev --extra backtesting --extra research --frozen && make check && uv run pytest --timeout=60
wc -l maverick/backtesting/service_ml.py
git push --force-with-lease
git checkout main
```
Expected: gate green; `service_ml.py` at 490 lines; the push lands on the fork branch `fix/backtesting-ensemble-strategy-mapping`.

- [x] **Step 4: Review the diff**

Run:
```bash
gh pr diff 243
```
Check: `TemplateStrategy` validates against `templates.STRATEGY_TEMPLATES`, exposes `name`, `description`, `get_default_parameters()`, and dispatches `generate_signals` through `signal_dispatch.generate_signals(data, self.strategy_type, self.parameters)`; `create_strategy_ensemble` builds `TemplateStrategy(name)` per requested name. The `if not instances` guard is now unreachable (`base_strategies or [...]` never yields an empty list); leave it, it is harmless.

- [x] **Step 5: Wait for CI, merge, thank**

Run:
```bash
gh pr checks 243 --watch
gh pr merge 243 --squash --body "Fixes #246: rsi and macd ensemble members run their own signal logic and keep distinct result keys."
gh pr comment 243 --body "Merged. Thank you; #246 closes with it."
gh issue view 246 --json state --jq .state
git checkout main && git pull --ff-only
```
Expected: `CLOSED`.

### Task 9: Rebase PR 244 with two fixups and merge (regime fallback, fixes #247)

**Files:**
- Modify (on the contributor's branch): `maverick/backtesting/service_ml.py` (the comment above `method=detector.method`), `tests/backtesting/test_ml_regime_aware.py` (two detector constructions)

- [x] **Step 1: Tell the author**

Run:
```bash
gh pr comment 244 --body "Rebasing onto main after #243 and adding one fixup commit: (1) maverick/ is under a 500-line-per-file structural rule (tests/structure/test_harness_rules.py), and the multi-line comment above method=detector.method pushed service_ml.py to 505 lines on the pre-rebase base, so it becomes a one-line trailing comment; (2) the two GaussianMixture tests pass random_state=0 so they cannot flake on an unlucky initialization. Everything else is yours as submitted."
```

- [x] **Step 2: Rebase onto main**

Run:
```bash
gh pr checkout 244
git rebase main
```
Expected: clean replay of two commits (244's `service_ml.py` hunk is the `method=` line near 382; 243's hunk is the ensemble block near 420, so they do not overlap). On a conflict in the docs or tests, keep both sides in order and continue as in Task 8.

- [x] **Step 3: Apply the comment fixup**

In `maverick/backtesting/service_ml.py`, replace this block inside `analyze_market_regimes`:

```python
                # `detector.method`, not the requested `method`: `fit_regimes`
                # silently switches to "threshold" when there isn't enough
                # data to fit a genuine statistical model (see
                # `MarketRegimeDetector.fit_regimes`) -- reporting the
                # originally-requested value here would misrepresent what was
                # actually used.
                method=detector.method,
```

with:

```python
                method=detector.method,  # what was used; fit_regimes may fall back
```

- [x] **Step 4: Apply the seed fixup**

In `tests/backtesting/test_ml_regime_aware.py`, inside `TestRegimeProbabilities`, change both occurrences of

```python
        det = MarketRegimeDetector(method="hmm", n_regimes=3, lookback_period=50)
```

(in `test_well_separated_regimes_produce_non_uniform_probabilities` and `test_probability_lookup_failure_falls_back_to_one_hot_not_uniform`) to

```python
        det = MarketRegimeDetector(
            method="hmm", n_regimes=3, lookback_period=50, random_state=0
        )
```

Leave `test_hmm_method_is_gaussian_mixture_not_hidden_markov_model` as is; it never fits.

- [x] **Step 5: Run the gate, commit the fixup, push**

Run:
```bash
wc -l maverick/backtesting/service_ml.py
uv sync --extra dev --extra backtesting --extra research --frozen && make check && uv run pytest --timeout=60
git add maverick/backtesting/service_ml.py tests/backtesting/test_ml_regime_aware.py
git commit -m "fix(backtesting): keep service_ml under the line cap and seed the mixture tests"
git push --force-with-lease
git checkout main
```
Expected: 490 lines; gate green; three commits on the PR.

- [x] **Step 6: Review the diff**

Run:
```bash
gh pr diff 244
```
Check: `_fall_back_to_threshold_method` is the only path that sets `is_fitted = True` without a fit (six call sites: four early returns and two exception handlers); `get_regime_probabilities` returns the model posterior or `_one_hot_via_threshold_fallback`, never `np.ones(n) / n`; the clamp `min(regime, n_regimes - 1)` only changes behavior for `n_regimes < 3`; `analyze_market_regimes` reports `detector.method`.

- [x] **Step 7: Wait for CI, merge, thank**

Run:
```bash
gh pr checks 244 --watch
gh pr merge 244 --squash --body "Fixes #247: regime fallback reports method=threshold and one-hot probabilities instead of a fabricated uniform vector."
gh pr comment 244 --body "Merged with the two fixups described above. Thank you; #247 closes with it."
gh issue view 247 --json state --jq .state
git checkout main && git pull --ff-only
```
Expected: `CLOSED`.

---

## Workstream 2: dependency batch

### Task 10: Verify the five safe bumps together, then merge them

**Files:** none kept; a scratch branch that is deleted at the end.

- [x] **Step 1: Build the combined lock on a scratch branch**

Run:
```bash
git checkout -b scratch/deps-2026-09 main
uv lock --upgrade-package langchain-anthropic --upgrade-package redis --upgrade-package greenlet --upgrade-package uvicorn --upgrade-package nltk
grep -A1 -E '^name = "(fastmcp|mcp|langchain-anthropic|redis|greenlet|uvicorn|nltk)"$' uv.lock | grep -E 'name|version' | paste - -
```
Expected versions: fastmcp 3.3.1 and mcp 1.28.1 unchanged; langchain-anthropic 1.7.0, redis 8.1.0, greenlet 3.5.5, uvicorn 0.52.4, nltk 3.10.3. If fastmcp or mcp moved, stop: the resolver found a path to the major upgrade and Task 11 must go first.

- [x] **Step 2: Run the gate on the combined lock**

Run:
```bash
uv sync --extra dev --extra backtesting --extra research --frozen && make check && uv run pytest --timeout=60
```
Expected: green.

- [x] **Step 3: Discard the scratch branch**

Run:
```bash
git checkout main && git branch -D scratch/deps-2026-09 && git checkout -- uv.lock 2>/dev/null; git status --short
```
Expected: clean tree.

- [x] **Step 4: Merge the five PRs one at a time**

Run, waiting for each merge before the next:
```bash
for pr in 248 250 251 252 253; do
  gh pr checks $pr --watch
  gh pr merge $pr --squash
  echo "merged $pr"
done
```
If a merge reports a conflict on `uv.lock`, comment `@dependabot rebase` on that PR, wait for the new run, and merge it.

- [x] **Step 5: Confirm main**

Run:
```bash
git checkout main && git pull --ff-only
gh pr list --state open --author app/dependabot
gh run list --branch main --limit 1
```
Expected: only PR 249 remains from dependabot (plus any brand-new bumps it opened when the cap freed up); the latest `main` run is green. A new dependabot PR for `fastmcp` 4.x, if it appears, gets the same hold comment as Task 3 and closes with Task 15.

---

## Workstream 3: FastMCP 4 migration

One PR from branch `chore/fastmcp-4` in a worktree. The project's exposure is small: eight `from fastmcp import FastMCP` imports, `FastMCP(name=...)`, `mcp.tool(name=..., annotations={...})`, `mcp.prompt(name=...)`, `mcp.resource(uri)`, `mcp.run(transport=...)`, and the in-memory `Client(mcp)` in tests. Nothing uses `Context`, sampling, roots, elicitation, tasks, proxies, mounts, `serializer=`, `exclude_args=`, or `McpError`.

### Task 11: Raise the floor and re-lock

**Files:**
- Modify: `pyproject.toml` (dependencies block, lines 7 to 16)
- Modify: `uv.lock` (regenerated)

- [x] **Step 1: Create the worktree**

Run:
```bash
git -C /home/wshobson/workspace/major7apps/maverick-mcp worktree add ../maverick-mcp-fastmcp4 -b chore/fastmcp-4 main
cd /home/wshobson/workspace/major7apps/maverick-mcp-fastmcp4
```

- [x] **Step 2: Edit the dependency block**

In `pyproject.toml`, replace

```toml
    # Core MCP and server dependencies
    "fastmcp>=3.3.1",
    "mcp>=1.28.1",
    "uvicorn>=0.48.0",
    "python-multipart>=0.0.31",
    "aiofiles>=25.1.0",
    "httpx>=0.28.1",
```

with

```toml
    # Core MCP and server dependencies. FastMCP 4 owns the MCP SDK version
    # (no direct `mcp` pin: nothing here imports it). FastMCP 4 no longer
    # depends on httpx, and maverick/platform/http.py imports it directly,
    # so the httpx pin below is load-bearing.
    "fastmcp>=4.0.3",
    "uvicorn>=0.48.0",
    "python-multipart>=0.0.31",
    "aiofiles>=25.1.0",
    "httpx>=0.28.1",
```

- [x] **Step 3: Re-lock and inspect the resolution**

Run:
```bash
uv lock
grep -A1 -E '^name = "(fastmcp|fastmcp-slim|mcp|mcp-types|pydantic|starlette|httpx|httpx2)"$' uv.lock | grep -E 'name|version' | paste - -
```
Expected: fastmcp 4.0.3 (or newer 4.x), fastmcp-slim at the same version, mcp 2.1.x, mcp-types present, pydantic 2.12 or newer, starlette 1.0.1 or newer, httpx still present (direct), httpx2 present (FastMCP's).

- [x] **Step 4: Sync and run the gate as-is**

Run:
```bash
uv sync --extra dev --extra backtesting --extra research --frozen
make check && uv run pytest --timeout=60 2>&1 | tail -15
```
Expected: green. Deprecation warnings mentioning `readOnlyHint` or `FastMCPDeprecationWarning` are expected at this step and go away in Task 12.

- [x] **Step 5: Commit**

Run:
```bash
git add pyproject.toml uv.lock
git commit -m "build: move to FastMCP 4 (fastmcp>=4.0.3) and drop the unused direct mcp pin"
```

### Task 12: Turn the camelCase bridge off and migrate annotations to snake_case

**Files:**
- Modify: `tests/conftest.py` (add a session fixture)
- Modify: `maverick/market_data/tools.py:12-17`, `maverick/screening/tools.py:10-15`, `maverick/portfolio/tools.py:20-43`, `maverick/technical/tools.py:9`, `maverick/backtesting/tools_support.py:24`, `maverick/research/tools.py:51`
- Modify: `tests/portfolio/test_tools.py` (annotation reads)
- Modify: `tests/server/test_assembly.py` (one new wire-format test)

**Interfaces:**
- Produces: annotation dicts keyed `read_only_hint`, `destructive_hint`, `idempotent_hint`, `open_world_hint`; the suite runs with `fastmcp.settings.mcp_camelcase_compat = False`.

- [x] **Step 1: Write the failing setup: disable the bridge for the whole suite**

Append to `tests/conftest.py`:

```python

import fastmcp
import pytest


@pytest.fixture(autouse=True, scope="session")
def _camelcase_bridge_off():
    """FastMCP 4 bridges camelCase reads of MCP SDK v2 models (`readOnlyHint`
    for `read_only_hint`) with a deprecation warning. Run the suite with the
    bridge off so any stale camelCase read fails as an `AttributeError` now,
    before FastMCP removes the shim."""
    fastmcp.settings.mcp_camelcase_compat = False
    yield
```

- [x] **Step 2: Run the portfolio annotation tests to verify they fail**

Run:
```bash
uv run pytest tests/portfolio/test_tools.py -k "marks" --timeout=60 2>&1 | tail -20
```
Expected: FAIL with `AttributeError: ... readOnlyHint` on the `tool.annotations.readOnlyHint` reads. If they pass instead, print `uv run python -c "import fastmcp; print(fastmcp.settings.mcp_camelcase_compat)"`; it must print `False`. If the attribute does not exist, check `uv run python -c "import fastmcp; print([f for f in type(fastmcp.settings).model_fields if 'camel' in f])"` and use the field it names.

- [x] **Step 3: Migrate the dict keys and the test reads**

Run:
```bash
sed -i -E 's/"readOnlyHint"/"read_only_hint"/g; s/"destructiveHint"/"destructive_hint"/g; s/"idempotentHint"/"idempotent_hint"/g; s/"openWorldHint"/"open_world_hint"/g' \
  maverick/market_data/tools.py maverick/screening/tools.py maverick/portfolio/tools.py \
  maverick/technical/tools.py maverick/backtesting/tools_support.py maverick/research/tools.py
sed -i -E 's/\.readOnlyHint\b/.read_only_hint/g; s/\.destructiveHint\b/.destructive_hint/g; s/\.idempotentHint\b/.idempotent_hint/g; s/\.openWorldHint\b/.open_world_hint/g' \
  tests/portfolio/test_tools.py
grep -rn -E 'readOnlyHint|destructiveHint|idempotentHint|openWorldHint' maverick/ tests/ 
```
Expected: the final grep shows only prose mentions (the docstring in `maverick/research/tools.py` lines 15 to 21 that describes the wire names). No dict keys and no attribute reads remain. For reference, `maverick/portfolio/tools.py` now reads:

```python
_READ_ONLY_ANNOTATIONS = {"read_only_hint": True}
_ADD_ANNOTATIONS = {
    "read_only_hint": False,
    "destructive_hint": False,
    "idempotent_hint": False,
}
_REMOVE_ANNOTATIONS = {
    "read_only_hint": False,
    "destructive_hint": True,
    "idempotent_hint": False,
}
_CLEAR_ANNOTATIONS = {
    "read_only_hint": False,
    "destructive_hint": True,
    "idempotent_hint": True,
}
```

- [x] **Step 4: Add a wire-format test**

Append to `tests/server/test_assembly.py`:

```python


async def test_tool_annotations_serialize_camel_case_on_the_wire():
    """Annotations are declared with SDK v2 snake_case field names; the MCP wire
    format is camelCase. Pin both sides so a future rename cannot drop the hint."""
    mcp = build_server()
    async with Client(mcp) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
    tool = tools["market_data_get_chart_links"]
    assert tool.annotations is not None
    assert tool.annotations.read_only_hint is True
    wire = tool.annotations.model_dump(by_alias=True, exclude_none=True)
    assert wire["readOnlyHint"] is True
```

- [x] **Step 5: Run the full gate**

Run:
```bash
make check && uv run pytest --timeout=60 -W error::DeprecationWarning 2>&1 | tail -15
```
Expected: green with no deprecation warnings escalated to errors. If a third-party library emits an unrelated `DeprecationWarning`, rerun without `-W` and confirm no `FastMCPDeprecationWarning` appears in the warnings summary.

- [x] **Step 6: Commit**

Run:
```bash
git add tests/conftest.py tests/server/test_assembly.py tests/portfolio/test_tools.py \
  maverick/market_data/tools.py maverick/screening/tools.py maverick/portfolio/tools.py \
  maverick/technical/tools.py maverick/backtesting/tools_support.py maverick/research/tools.py
git commit -m "refactor: declare tool annotations with SDK v2 snake_case names and run tests with the camelCase bridge off"
```

### Task 13: Walk the upgrade checklist, smoke both transports, update the README badge

**Files:**
- Modify: `README.md:7` (badge)
- Modify only if the smoke test contradicts it: `docs/runbooks/mcp-clients.md:25-28`

- [x] **Step 1: Prove the removed and changed APIs are unused**

Run:
```bash
grep -rn -E 'as_proxy|import_server|\.mount\(|add_tool_transformation|remove_tool\(|serializer=|exclude_args=|sse_read_timeout|ctx\.(sample|elicit|list_roots)|sampling_handler|McpError|ErrorData|fastmcp\.server\.(proxy|openapi|apps)|fastmcp\.(tools\.tool|resources\.resource|prompts\.prompt)|task=True|TaskConfig|except httpx\.' maverick/ tests/ || echo "no matches: nothing on the removed list is in use"
```
Expected: `no matches: ...`. Record the command and its output in the PR description.

- [x] **Step 2: Smoke the stdio transport with a negotiated client**

Run:
```bash
uv run python - <<'EOF'
import asyncio
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

async def main() -> None:
    transport = StdioTransport(
        command="uv", args=["run", "python", "-m", "maverick.server", "--transport", "stdio"]
    )
    async with Client(transport) as client:
        tools = await client.list_tools()
        prompts = await client.list_prompts()
        resources = await client.list_resources()
        print(f"tools={len(tools)} prompts={len(prompts)} resources={len(resources)}")
        result = await client.call_tool("market_data_get_chart_links", {"ticker": "AAPL"})
        print("call ok:", result.data["status"])

asyncio.run(main())
EOF
```
Expected: `tools=52 prompts=3 resources=1` (37 core, 12 backtesting, 3 research with both extras installed) and `call ok: success`.

- [x] **Step 3: Smoke the HTTP transport and the trailing-slash behavior**

Run:
```bash
(uv run python -m maverick.server --transport http --port 8003 > /tmp/claude-1000/-home-wshobson-workspace-major7apps-maverick-mcp/9eed80ae-44bf-437b-86f4-ec3903826144/scratchpad/http-smoke.log 2>&1 &) ; sleep 4
uv run python - <<'EOF'
import asyncio
from fastmcp import Client

async def main() -> None:
    async with Client("http://127.0.0.1:8003/mcp") as client:
        tools = await client.list_tools()
        print("http tools:", len(tools))

asyncio.run(main())
EOF
curl -s -o /dev/null -w 'POST /mcp/ -> %{http_code}\n' -X POST -H 'Content-Type: application/json' -d '{}' http://127.0.0.1:8003/mcp/
pkill -f "maverick.server" || true
```
Expected: `http tools: 52` and `POST /mcp/ -> 307`. If the trailing-slash request no longer answers 307, edit the note at `docs/runbooks/mcp-clients.md:25-28` and the troubleshooting lines at 487 to 492 to state the observed status.

- [x] **Step 4: Update the README badge**

Run:
```bash
sed -i 's#badge/FastMCP-3-green.svg#badge/FastMCP-4-green.svg#' README.md
grep -n 'FastMCP-4' README.md
```
Expected: line 7 shows the FastMCP 4 badge.

- [x] **Step 5: Gate and commit**

Run:
```bash
make check && make docs-check && uv run pytest --timeout=60 2>&1 | tail -3
git add README.md docs/runbooks/mcp-clients.md
git commit -m "docs: FastMCP 4 badge and transport notes"
```
(`git add` of the runbook is a no-op when it was not edited.)

### Task 14: Run the conformance suite and settle #235

**Files:** none in the repository; two reports in the scratchpad.

- [x] **Step 1: Run both protocol revisions**

Run from the worktree root:
```bash
S=/tmp/claude-1000/-home-wshobson-workspace-major7apps-maverick-mcp/9eed80ae-44bf-437b-86f4-ec3903826144/scratchpad
npx -y @hasmcp/mcp-spec-test@0.1.5 -c "uv run python -m maverick.server --transport stdio" --spec-version 2026-07-28 > "$S/conformance-2026-07-28.md" 2>&1; echo "exit=$?"
npx -y @hasmcp/mcp-spec-test@0.1.5 -c "uv run python -m maverick.server --transport stdio" --spec-version 2025-11-25 > "$S/conformance-2025-11-25.md" 2>&1; echo "exit=$?"
grep -E '^\*\*Verdict|^\| (Passed|Failed|Not verified)' "$S"/conformance-*.md
```
Expected: both reports render a verdict table. The target is `Verdict: conformant` on both, with `server/discover` checks passing on `2026-07-28`.

- [x] **Step 2: Fix anything the server owns**

For each failed check, decide whether it is server code (this repository) or the framework (FastMCP or the SDK). Server-owned failures get a fix in this branch with a test; framework-owned failures get recorded with the check name and the upstream project. A version-less `tools/list` refusal, `server/discover` errors, or result-envelope field problems are framework behavior; a tool schema or resource read problem is ours.

- [x] **Step 3: Attach the reports to #235**

Run:
```bash
S=/tmp/claude-1000/-home-wshobson-workspace-major7apps-maverick-mcp/9eed80ae-44bf-437b-86f4-ec3903826144/scratchpad
{ echo "Conformance after the FastMCP 4 migration (fastmcp $(uv run python -c 'import importlib.metadata as m; print(m.version("fastmcp"))'), mcp-spec-test 0.1.5), run on \`$(git rev-parse --short HEAD)\`:"; echo; echo "## 2026-07-28"; echo; cat "$S/conformance-2026-07-28.md"; echo; echo "## 2025-11-25"; echo; cat "$S/conformance-2025-11-25.md"; } > "$S/issue-235-comment.md"
gh issue comment 235 --body-file "$S/issue-235-comment.md"
```

- [x] **Step 4: Close or keep open with the residual list**

If both verdicts are conformant, run:
```bash
gh issue close 235 --reason completed --comment "Closing: the server now runs on FastMCP 4 and passes the 2026-07-28 and 2025-11-25 suites (reports above). Thanks for the report and the tool."
```
Otherwise leave #235 open and post one comment listing each residual failed check with its owner (this repository or the upstream project and its tracking link).

### Task 15: Open, review, and merge the migration PR

**Files:** none new.

- [x] **Step 1: Push and open the PR**

Run:
```bash
git push -u origin chore/fastmcp-4
gh pr create --title "build: migrate to FastMCP 4" --body-file - <<'EOF'
Moves the server to FastMCP 4 (MCP Python SDK v2), which serves the 2026-07-28 protocol revision alongside 2025-11-25 and unblocks #235. Supersedes #249, whose lock bump pulled the major upgrade in silently.

- `fastmcp>=4.0.3`; the direct `mcp` pin is gone (nothing imported it); `httpx` stays pinned directly because the platform HTTP seam imports it and FastMCP 4 no longer depends on it.
- Tool annotations are declared with SDK v2 snake_case field names; the wire format stays camelCase (pinned by a new assembly test). The test suite runs with the camelCase compatibility bridge off.
- Upgrade-guide checklist: no removed or changed API in use (grep in the task log). Stdio and HTTP smoke tests pass; `/mcp/` still answers 307.
- Conformance reports attached to #235.
EOF
```

- [x] **Step 2: Wait for CI and merge**

Run:
```bash
gh pr checks --watch
gh pr merge --squash --delete-branch
git checkout main && git pull --ff-only
git worktree remove ../maverick-mcp-fastmcp4
```

- [x] **Step 3: Close PR 249**

Run:
```bash
gh pr view 249 --json state --jq .state
```
If it prints `OPEN` (dependabot did not close it on its own), run:
```bash
gh pr close 249 --comment "Superseded by the FastMCP 4 migration, which carries mcp 2.x through fastmcp>=4.0.3."
```

---

## Workstream 4: SearXNG research backend (#186)

One PR from branch `feat/searxng-backend` in a worktree created from `main` after Task 15 merges. Test-first throughout. Design note: the spec placed the shared scoring helpers in `providers/base.py`; this plan puts them in a new `providers/scoring.py` so `base.py` keeps its single responsibility (timeouts and health), and the spec's one sentence is updated in Task 1's branch to match.

```bash
git -C /home/wshobson/workspace/major7apps/maverick-mcp worktree add ../maverick-mcp-searxng -b feat/searxng-backend main
cd /home/wshobson/workspace/major7apps/maverick-mcp-searxng
uv sync --extra dev --extra backtesting --extra research --frozen
```

### Task 16: Lift the financial-relevance scoring out of the Exa provider

**Files:**
- Create: `maverick/research/providers/scoring.py`
- Modify: `maverick/research/providers/exa.py` (imports, constants, three helper methods)
- Test: `tests/research/test_scoring.py` (new)

**Interfaces:**
- Produces: `extract_domain(url: str) -> str`, `is_authoritative_source(url: str) -> bool`, `financial_relevance(*, url: str, text: str | None, title: str | None, published_date: str | None, financial_domains: list[str] | None = None) -> float`, and the constants `FINANCIAL_DOMAINS`, `AUTHORITATIVE_DOMAINS`, `FINANCIAL_KEYWORDS`. Task 18 consumes the three functions.

- [x] **Step 1: Write the failing tests**

Create `tests/research/test_scoring.py`:

```python
"""Tests for `maverick.research.providers.scoring`, the relevance helpers shared by
every search provider."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from maverick.research.providers.scoring import (
    extract_domain,
    financial_relevance,
    is_authoritative_source,
)


def test_extract_domain_lowercases_and_strips_www():
    assert extract_domain("https://www.Reuters.com/markets/x") == "reuters.com"


def test_extract_domain_of_empty_input_is_empty():
    assert extract_domain("") == ""


def test_authoritative_source_matches_the_domain_list():
    assert is_authoritative_source("https://www.sec.gov/edgar") is True
    assert is_authoritative_source("https://example.com") is False


def test_financial_relevance_tiers_domains():
    def score(url: str) -> float:
        return financial_relevance(url=url, text=None, title=None, published_date=None)

    assert score("https://sec.gov/x") == pytest.approx(0.4)
    assert score("https://reuters.com/x") == pytest.approx(0.3)
    assert score("https://fool.com/x") == pytest.approx(0.2)
    assert score("https://example.com/x") == pytest.approx(0.0)


def test_financial_relevance_caps_keyword_bonus_and_adds_title_bonus():
    text = "earnings revenue profit dividend valuation analyst forecast guidance"
    score = financial_relevance(
        url="https://example.com", text=text, title="Quarterly earnings", published_date=None
    )
    # Eight keyword hits cap at 0.3; the title term adds 0.1.
    assert score == pytest.approx(0.4)


def test_financial_relevance_recency_bonus():
    recent = (datetime.now(UTC) - timedelta(days=5)).isoformat()
    older = (datetime.now(UTC) - timedelta(days=60)).isoformat()
    assert financial_relevance(
        url="https://example.com", text=None, title=None, published_date=recent
    ) == pytest.approx(0.1)
    assert financial_relevance(
        url="https://example.com", text=None, title=None, published_date=older
    ) == pytest.approx(0.05)


def test_financial_relevance_ignores_unparseable_dates():
    assert (
        financial_relevance(
            url="https://example.com", text=None, title=None, published_date="not a date"
        )
        == 0.0
    )


def test_financial_relevance_honors_a_custom_domain_list():
    score = financial_relevance(
        url="https://example.com/x",
        text=None,
        title=None,
        published_date=None,
        financial_domains=["example.com"],
    )
    assert score == pytest.approx(0.2)
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/research/test_scoring.py --timeout=60`
Expected: FAIL at import with `ModuleNotFoundError: No module named 'maverick.research.providers.scoring'`.

- [x] **Step 3: Create the scoring module**

Create `maverick/research/providers/scoring.py`:

```python
"""Financial-relevance scoring shared by every search provider. Third-layer sibling: imports nothing from the domain.

Moved out of `exa.py` so `SearXNGProvider` and `ExaSearchProvider` score and
sort results identically. The domain lists, keyword list, and weights are
verbatim from the legacy Exa provider. `ExaSearchProvider` keeps thin method
wrappers around these functions so its callers and tests are unchanged.
"""

from __future__ import annotations

from datetime import UTC, datetime
from urllib.parse import urlparse

# Financial-specific domain preferences for better results.
FINANCIAL_DOMAINS = [
    "sec.gov",
    "edgar.sec.gov",
    "investor.gov",
    "bloomberg.com",
    "reuters.com",
    "wsj.com",
    "ft.com",
    "marketwatch.com",
    "yahoo.com/finance",
    "finance.yahoo.com",
    "morningstar.com",
    "fool.com",
    "seekingalpha.com",
    "investopedia.com",
    "barrons.com",
    "cnbc.com",
    "nasdaq.com",
    "nyse.com",
    "finra.org",
    "federalreserve.gov",
    "treasury.gov",
    "bls.gov",
]

AUTHORITATIVE_DOMAINS = [
    "sec.gov",
    "edgar.sec.gov",
    "federalreserve.gov",
    "treasury.gov",
    "bloomberg.com",
    "reuters.com",
    "wsj.com",
    "ft.com",
]

FINANCIAL_KEYWORDS = [
    "earnings",
    "revenue",
    "profit",
    "financial",
    "quarterly",
    "annual",
    "sec filing",
    "10-k",
    "10-q",
    "balance sheet",
    "income statement",
    "cash flow",
    "dividend",
    "market cap",
    "valuation",
    "analyst",
    "forecast",
    "guidance",
    "ebitda",
    "eps",
    "pe ratio",
]

_TOP_TIER_DOMAINS = ["sec.gov", "edgar.sec.gov", "federalreserve.gov"]
_HIGH_QUALITY_DOMAINS = ["bloomberg.com", "reuters.com", "wsj.com", "ft.com"]
_TITLE_TERMS = ["financial", "earnings", "quarterly", "annual", "sec"]

# Scoring weights, verbatim from legacy.
_DOMAIN_SCORE_TOP_TIER = 0.4
_DOMAIN_SCORE_HIGH_QUALITY = 0.3
_DOMAIN_SCORE_OTHER = 0.2
_KEYWORD_SCORE_PER_MATCH = 0.05
_KEYWORD_SCORE_MAX = 0.3
_TITLE_SCORE = 0.1
_RECENCY_SCORE_30D = 0.1
_RECENCY_SCORE_90D = 0.05


def extract_domain(url: str) -> str:
    """Return the lowercased host of `url` without a `www.` prefix, or `""`."""
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def is_authoritative_source(url: str) -> bool:
    """Whether `url` is from an authoritative financial source."""
    return extract_domain(url) in AUTHORITATIVE_DOMAINS


def financial_relevance(
    *,
    url: str,
    text: str | None,
    title: str | None,
    published_date: str | None,
    financial_domains: list[str] | None = None,
) -> float:
    """Score a search result's financial relevance from 0.0 to 1.0.

    Domain tier, keyword density in `text` (capped), a title bonus, and a
    recency bonus for ISO `published_date` values within 30 or 90 days.
    """
    domains = financial_domains if financial_domains is not None else FINANCIAL_DOMAINS
    score = 0.0

    domain = extract_domain(url)
    if domain in domains:
        if domain in _TOP_TIER_DOMAINS:
            score += _DOMAIN_SCORE_TOP_TIER
        elif domain in _HIGH_QUALITY_DOMAINS:
            score += _DOMAIN_SCORE_HIGH_QUALITY
        else:
            score += _DOMAIN_SCORE_OTHER

    if text:
        text_lower = text.lower()
        keyword_matches = sum(1 for keyword in FINANCIAL_KEYWORDS if keyword in text_lower)
        score += min(keyword_matches * _KEYWORD_SCORE_PER_MATCH, _KEYWORD_SCORE_MAX)

    if title:
        title_lower = title.lower()
        if any(term in title_lower for term in _TITLE_TERMS):
            score += _TITLE_SCORE

    if published_date:
        try:
            date_str = str(published_date)
            if date_str.endswith("Z"):
                date_str = date_str.replace("Z", "+00:00")
            pub_date = datetime.fromisoformat(date_str)
            days_old = (datetime.now(UTC) - pub_date).days
            if days_old <= 30:
                score += _RECENCY_SCORE_30D
            elif days_old <= 90:
                score += _RECENCY_SCORE_90D
        except (ValueError, AttributeError, TypeError):
            pass

    return min(score, 1.0)
```

- [x] **Step 4: Run the scoring tests to verify they pass**

Run: `uv run pytest tests/research/test_scoring.py --timeout=60`
Expected: 8 passed.

- [x] **Step 5: Point the Exa provider at the shared helpers**

In `maverick/research/providers/exa.py`:

1. Replace the import block

```python
import asyncio
import logging
from datetime import UTC
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from maverick.platform.config import HttpSettings
from maverick.platform.http import CircuitOpenError, get_breaker
from maverick.research.config import ResearchSettings
from maverick.research.providers.base import WebSearchError, WebSearchProvider
```

with

```python
import asyncio
import logging
from typing import TYPE_CHECKING, Any

from maverick.platform.config import HttpSettings
from maverick.platform.http import CircuitOpenError, get_breaker
from maverick.research.config import ResearchSettings
from maverick.research.providers.base import WebSearchError, WebSearchProvider
from maverick.research.providers.scoring import (
    FINANCIAL_DOMAINS,
    extract_domain,
    financial_relevance,
    is_authoritative_source,
)
```

2. Delete the module-level `_FINANCIAL_DOMAINS`, `_AUTHORITATIVE_DOMAINS`, and `_FINANCIAL_KEYWORDS` lists and the eight scoring constants (`_DOMAIN_SCORE_TOP_TIER`, `_DOMAIN_SCORE_HIGH_QUALITY`, `_DOMAIN_SCORE_OTHER`, `_KEYWORD_SCORE_PER_MATCH`, `_KEYWORD_SCORE_MAX`, `_TITLE_SCORE`, `_RECENCY_SCORE_30D`, `_RECENCY_SCORE_90D`). Keep `_EXCLUDED_DOMAINS`, `_FINANCIAL_QUERY_TERMS`, `_CONTENT_CHARS`, `_RAW_CONTENT_CHARS`, and `_DEFAULT_SCORE`.

3. In `__init__`, change `self.financial_domains = list(_FINANCIAL_DOMAINS)` to `self.financial_domains = list(FINANCIAL_DOMAINS)`.

4. Replace the three methods `_calculate_financial_relevance`, `_extract_domain`, and `_is_authoritative_source` (everything from `def _calculate_financial_relevance` to the end of the file) with:

```python
    def _calculate_financial_relevance(self, result: ExaResult | Any) -> float:
        """Calculate financial relevance score for a search result (0.0 to 1.0)."""
        return financial_relevance(
            url=result.url or "",
            text=getattr(result, "text", None),
            title=getattr(result, "title", None),
            published_date=getattr(result, "published_date", None),
            financial_domains=self.financial_domains,
        )

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        return extract_domain(url)

    def _is_authoritative_source(self, url: str) -> bool:
        """Check if URL is from an authoritative financial source."""
        return is_authoritative_source(url)
```

- [x] **Step 6: Verify nothing dangling remains and the Exa tests still pass**

Run:
```bash
grep -n -E '_FINANCIAL_DOMAINS|_AUTHORITATIVE_DOMAINS|_FINANCIAL_KEYWORDS|_DOMAIN_SCORE|_KEYWORD_SCORE|_TITLE_SCORE|_RECENCY_SCORE|urlparse|UTC' maverick/research/providers/exa.py || echo "clean"
uv run ruff check maverick/research/providers/ && uv run ruff format --check maverick/research/providers/
uv run pytest tests/research/test_providers.py tests/research/test_scoring.py --timeout=60
uv run lint-imports
```
Expected: `clean`, ruff clean, all tests pass, `0 broken` contracts.

- [x] **Step 7: Commit**

Run:
```bash
git add maverick/research/providers/scoring.py maverick/research/providers/exa.py tests/research/test_scoring.py
git commit -m "refactor(research): share financial-relevance scoring across search providers"
```

### Task 17: Add the search backend settings

**Files:**
- Modify: `maverick/research/config.py`
- Test: `tests/research/test_config.py`

**Interfaces:**
- Produces: `ResearchSettings.search_backend: Literal["exa", "searxng"]` (env `RESEARCH_SEARCH_BACKEND`, default `"exa"`) and `ResearchSettings.searxng_base_url: str | None` (env `SEARXNG_BASE_URL`, trailing slash stripped, `http://` or `https://` required). Tasks 19 and 20 consume both.

- [x] **Step 1: Write the failing tests**

In `tests/research/test_config.py`, extend the env list and add tests:

```python
_ENV_VARS = (
    "EXA_API_KEY",
    "RESEARCH_DEFAULT_DEPTH",
    "RESEARCH_DEFAULT_MAX_SOURCES",
    "RESEARCH_DEFAULT_TIMEFRAME",
    "RESEARCH_SENTIMENT_DEFAULT_TIMEFRAME",
    "RESEARCH_SEARCH_BACKEND",
    "SEARXNG_BASE_URL",
)
```

and append:

```python


def test_search_backend_defaults_to_exa_with_no_searxng_url():
    s = ResearchSettings()

    assert s.search_backend == "exa"
    assert s.searxng_base_url is None


def test_search_backend_env_override_normalizes_the_url(monkeypatch):
    monkeypatch.setenv("RESEARCH_SEARCH_BACKEND", "searxng")
    monkeypatch.setenv("SEARXNG_BASE_URL", "http://localhost:8080/")

    s = ResearchSettings()

    assert s.search_backend == "searxng"
    assert s.searxng_base_url == "http://localhost:8080"


def test_invalid_search_backend_fails_fast(monkeypatch):
    monkeypatch.setenv("RESEARCH_SEARCH_BACKEND", "bing")

    with pytest.raises(ValidationError):
        ResearchSettings()


def test_searxng_base_url_requires_an_http_scheme(monkeypatch):
    monkeypatch.setenv("SEARXNG_BASE_URL", "localhost:8080")

    with pytest.raises(ValidationError):
        ResearchSettings()
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/research/test_config.py --timeout=60`
Expected: the four new tests FAIL (`AttributeError: 'ResearchSettings' object has no attribute 'search_backend'` and no `ValidationError` raised).

- [x] **Step 3: Add the settings**

In `maverick/research/config.py`:

1. Change the imports to

```python
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_validator

from maverick.platform.config import _clean_env, _env_float, _env_int, _env_str
from maverick.research.types import ResearchDepth
```

2. After `_resolve_exa_api_key`, add

```python

SearchBackend = Literal["exa", "searxng"]
"""Which web search provider backs the research tools. `exa` needs `EXA_API_KEY`;
`searxng` needs `SEARXNG_BASE_URL` (a self-hosted instance with the JSON format
enabled) and no key."""
```

3. Directly after the `exa_api_key` field inside `ResearchSettings`, add

```python
    search_backend: SearchBackend = Field(
        default_factory=lambda: _env_str("RESEARCH_SEARCH_BACKEND", "exa"),
        validate_default=True,
    )
    searxng_base_url: str | None = Field(
        default_factory=lambda: _clean_env("SEARXNG_BASE_URL"),
        validate_default=True,
    )

    @field_validator("searxng_base_url")
    @classmethod
    def _normalize_searxng_base_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip().rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("SEARXNG_BASE_URL must start with http:// or https://")
        return value
```

4. Append two lines to the module docstring's list of env-backed fields, after the `search_timeout_failure_threshold` bullet:

```
- `search_backend` (`RESEARCH_SEARCH_BACKEND`, default `exa`) and
  `searxng_base_url` (`SEARXNG_BASE_URL`): the 2026-09 SearXNG backend
  selection (#186). See `providers/searxng.py`.
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/research/test_config.py --timeout=60`
Expected: all pass, including `test_defaults_are_zero_config`.

- [x] **Step 5: Commit**

Run:
```bash
git add maverick/research/config.py tests/research/test_config.py
git commit -m "feat(research): add RESEARCH_SEARCH_BACKEND and SEARXNG_BASE_URL settings"
```

### Task 18: Implement the SearXNG provider

**Files:**
- Create: `maverick/research/providers/searxng.py`
- Modify: `maverick/research/providers/__init__.py:1` (docstring names the new provider)
- Test: `tests/research/test_searxng.py` (new)

**Interfaces:**
- Consumes: `extract_domain`, `financial_relevance`, `is_authoritative_source` from Task 16; `WebSearchProvider`, `WebSearchError` from `providers/base.py`; `create_client`, `request_resilient`, `CircuitOpenError` from `maverick.platform.http`; `HttpSettings`, `get_platform_settings` from `maverick.platform.config`.
- Produces: `SearXNGProvider(base_url: str, *, settings: ResearchSettings | None = None, time_range: str | None = None, http_settings: HttpSettings | None = None, transport: httpx.AsyncBaseTransport | None = None)` with `search(query, num_results=10, timeout_budget=None) -> list[dict[str, Any]]`, attributes `base_url` and `time_range`; and `time_range_for(timeframe: str | None) -> str | None`. Task 19 consumes both.

- [x] **Step 1: Write the failing tests**

Create `tests/research/test_searxng.py`:

```python
"""Tests for `maverick.research.providers.searxng`. No network: every request is
answered by an `httpx.MockTransport` handed to the provider."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from maverick.platform.config import HttpSettings
from maverick.research.config import ResearchSettings
from maverick.research.providers.base import WebSearchError
from maverick.research.providers.searxng import SearXNGProvider, time_range_for

# No retries, no backoff sleeps, no rate limiting: these tests cover the
# provider's own behavior, not the platform resilience policy.
_FAST_HTTP = HttpSettings(
    retries=0, backoff_base_seconds=0.0, rate_limit_per_second=1000.0
)


def _provider(handler: Any, **kwargs: Any) -> SearXNGProvider:
    kwargs.setdefault("settings", ResearchSettings())
    return SearXNGProvider(
        "http://searx.local:8080/",
        http_settings=_FAST_HTTP,
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def _json(payload: dict[str, Any], status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=payload)


def test_time_range_mapping():
    assert time_range_for("1d") == "day"
    assert time_range_for("7d") == "week"
    assert time_range_for("1w") == "week"
    assert time_range_for("30d") == "month"
    assert time_range_for("1m") == "month"
    assert time_range_for("1y") == "year"
    assert time_range_for("3m") is None
    assert time_range_for(None) is None


async def test_search_sends_json_format_and_normalizes_results():
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["params"] = dict(request.url.params)
        return _json(
            {
                "results": [
                    {
                        "url": "https://example.com/a",
                        "title": "A",
                        "content": "plain",
                        "publishedDate": "",
                        "score": 0.5,
                    },
                    {
                        "url": "https://www.sec.gov/filing",
                        "title": "Quarterly earnings",
                        "content": "revenue and earnings",
                        "score": 0.2,
                    },
                ]
            }
        )

    provider = _provider(handler, time_range="month")

    results = await provider.search("AAPL earnings", num_results=10)

    assert captured["path"] == "/search"
    assert captured["params"] == {
        "q": "AAPL earnings",
        "format": "json",
        "categories": "general",
        "language": "en",
        "time_range": "month",
    }
    # Sorted by financial relevance: the SEC filing (domain tier + keywords +
    # title term) outranks the plain result despite its lower raw score.
    assert [r["url"] for r in results] == [
        "https://www.sec.gov/filing",
        "https://example.com/a",
    ]
    sec, plain = results
    assert sec["provider"] == "searxng"
    assert sec["domain"] == "sec.gov"
    assert sec["is_authoritative"] is True
    assert sec["content"] == sec["raw_content"] == "revenue and earnings"
    assert sec["score"] == 0.2
    assert sec["financial_relevance"] == pytest.approx(0.6)
    assert plain["score"] == 0.5
    assert plain["author"] == ""
    assert plain["published_date"] == ""
    assert provider.is_healthy() is True
    assert provider._failure_count == 0


async def test_search_defaults_missing_fields():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json({"results": [{"url": "https://example.com/x"}]})

    results = await _provider(handler).search("q")

    assert results == [
        {
            "url": "https://example.com/x",
            "title": "No Title",
            "content": "",
            "raw_content": "",
            "published_date": "",
            "score": 0.7,
            "financial_relevance": 0.0,
            "provider": "searxng",
            "author": "",
            "domain": "example.com",
            "is_authoritative": False,
        }
    ]


async def test_search_truncates_to_num_results_and_content_length():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json(
            {
                "results": [
                    {"url": f"https://example.com/{i}", "content": "x" * 3000}
                    for i in range(5)
                ]
            }
        )

    results = await _provider(handler).search("q", num_results=2)

    assert len(results) == 2
    assert len(results[0]["content"]) == 2000


async def test_search_omits_time_range_when_none():
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["params"] = dict(request.url.params)
        return _json({"results": []})

    await _provider(handler, time_range=None).search("q")

    assert "time_range" not in captured["params"]


async def test_forbidden_explains_how_to_enable_json():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="Forbidden")

    provider = _provider(handler)

    with pytest.raises(WebSearchError, match=r"formats: \[html, json\]"):
        await provider.search("q")
    assert provider._failure_count == 1


async def test_non_json_body_is_reported_with_the_hint():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>search page</html>")

    with pytest.raises(WebSearchError, match="format=json"):
        await _provider(handler).search("q")


async def test_other_http_errors_name_the_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="nope")

    with pytest.raises(WebSearchError, match="HTTP 404"):
        await _provider(handler).search("q")


async def test_transport_error_wraps_into_web_search_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    provider = _provider(handler)

    with pytest.raises(WebSearchError, match="SearXNG search failed"):
        await provider.search("q")
    assert provider._failure_count == 1
    assert provider.is_healthy() is True


async def test_provider_disables_itself_after_repeated_failures():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("refused")

    provider = _provider(handler)

    for _ in range(6):  # base.py's _MAX_NON_TIMEOUT_FAILURES
        with pytest.raises(WebSearchError):
            await provider.search("q")
    assert provider.is_healthy() is False

    with pytest.raises(WebSearchError, match="disabled due to repeated failures"):
        await provider.search("q")
    assert calls == 6


async def test_open_circuit_breaker_short_circuits_subsequent_calls():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("refused")

    settings = ResearchSettings(
        search_circuit_breaker_failure_threshold=1,
        search_circuit_breaker_recovery_seconds=9999.0,
    )
    provider = _provider(handler, settings=settings)

    with pytest.raises(WebSearchError, match="SearXNG search failed"):
        await provider.search("first")
    with pytest.raises(WebSearchError, match="SearXNG search failed"):
        await provider.search("second")
    assert calls == 1
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/research/test_searxng.py --timeout=60`
Expected: FAIL at import with `ModuleNotFoundError: No module named 'maverick.research.providers.searxng'`.

- [x] **Step 3: Create the provider**

Create `maverick/research/providers/searxng.py`:

```python
"""SearXNG web search provider: a self-hosted, keyless backend for the research tools. Third-layer sibling: imports platform, config, and the provider base.

SearXNG (https://docs.searxng.org) is a self-hosted metasearch engine with a
JSON API at `GET {base_url}/search?q=...&format=json`. Most instances ship
with the `json` format disabled; enabling it is a one-line change to the
instance's `settings.yml` (`search: formats: [html, json]`). A 403 from the
instance is reported with that instruction instead of a generic failure.

Requests go through `maverick.platform.http.request_resilient`, so the shared
per-name rate limiter, circuit breaker, and retry policy apply, with the
breaker thresholds taken from `ResearchSettings` exactly as `exa.py` does.
The base class's provider-level health gate runs on top. Results normalize
to the dict shape `ExaSearchProvider` returns so the research agents cannot
tell the backends apart; SearXNG returns snippets rather than page text, so
`raw_content` equals `content`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from maverick.platform.config import HttpSettings, get_platform_settings
from maverick.platform.http import CircuitOpenError, create_client, request_resilient
from maverick.research.config import ResearchSettings
from maverick.research.providers.base import WebSearchError, WebSearchProvider
from maverick.research.providers.scoring import (
    extract_domain,
    financial_relevance,
    is_authoritative_source,
)

logger = logging.getLogger(__name__)

_BREAKER_NAME = "searxng_search"
_CONTENT_CHARS = 2000
_DEFAULT_SCORE = 0.7

# SearXNG `time_range` values keyed by the research timeframe strings the
# service passes around (`ResearchSettings.default_timeframe` and friends).
_TIME_RANGES = {
    "1d": "day",
    "7d": "week",
    "1w": "week",
    "30d": "month",
    "1m": "month",
    "1y": "year",
}

_JSON_DISABLED_HINT = (
    "SearXNG answered {status} to a format=json request. Enable the JSON "
    "format on the instance: in settings.yml set `search: formats: [html, json]`."
)


def time_range_for(timeframe: str | None) -> str | None:
    """Map a research timeframe (`"1m"`, `"7d"`, ...) to a SearXNG `time_range`.

    Returns `None` for timeframes SearXNG cannot express (`"3m"`), in which case
    the request omits the parameter and the instance applies no recency filter.
    """
    if timeframe is None:
        return None
    return _TIME_RANGES.get(timeframe.strip().lower())


class SearXNGProvider(WebSearchProvider):
    """Keyless search provider backed by a self-hosted SearXNG instance."""

    def __init__(
        self,
        base_url: str,
        *,
        settings: ResearchSettings | None = None,
        time_range: str | None = None,
        http_settings: HttpSettings | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        # SearXNG has no API key; the base class stores one but never reads it.
        super().__init__("", settings=settings)
        self.base_url = base_url.rstrip("/")
        self.time_range = time_range
        self._http_settings = http_settings
        self._transport = transport
        logger.info("Initialized SearXNGProvider for %s", self.base_url)

    def _request_settings(self, timeout_seconds: float) -> HttpSettings:
        """Per-search HTTP settings: the platform's retry and rate policy, the
        research breaker thresholds, and the adaptive per-query timeout."""
        base = self._http_settings or get_platform_settings().http
        return HttpSettings(
            timeout_seconds=timeout_seconds,
            retries=base.retries,
            backoff_base_seconds=base.backoff_base_seconds,
            rate_limit_per_second=base.rate_limit_per_second,
            breaker_failure_threshold=self._settings.search_circuit_breaker_failure_threshold,
            breaker_recovery_seconds=self._settings.search_circuit_breaker_recovery_seconds,
        )

    async def search(
        self, query: str, num_results: int = 10, timeout_budget: float | None = None
    ) -> list[dict[str, Any]]:
        """Search the instance and return results in the shared provider shape."""
        if not self.is_healthy():
            logger.warning("SearXNG provider is unhealthy - skipping search")
            raise WebSearchError("SearXNG provider disabled due to repeated failures")

        search_timeout = self._calculate_timeout(query, timeout_budget)
        http_settings = self._request_settings(search_timeout)
        params: dict[str, Any] = {
            "q": query,
            "format": "json",
            "categories": "general",
            "language": "en",
        }
        if self.time_range is not None:
            params["time_range"] = self.time_range

        try:
            async with create_client(http_settings, transport=self._transport) as client:
                response = await asyncio.wait_for(
                    request_resilient(
                        _BREAKER_NAME,
                        client,
                        "GET",
                        f"{self.base_url}/search",
                        settings=http_settings,
                        params=params,
                    ),
                    timeout=search_timeout,
                )
            results = self._normalize_response(response)[:num_results]
        except TimeoutError:
            self._record_failure("timeout")
            raise WebSearchError(
                f"SearXNG search timed out after {search_timeout:.1f} seconds"
            )
        except CircuitOpenError as e:
            self._record_failure("error")
            raise WebSearchError(f"SearXNG search failed: {e}") from e
        except WebSearchError:
            self._record_failure("error")
            raise
        except Exception as e:
            self._record_failure("error")
            raise WebSearchError(f"SearXNG search failed: {e}") from e

        self._record_success()
        return results

    def _normalize_response(self, response: httpx.Response) -> list[dict[str, Any]]:
        """Convert a SearXNG JSON response into the shared result shape."""
        if response.status_code == 403:
            raise WebSearchError(_JSON_DISABLED_HINT.format(status=403))
        if response.status_code >= 400:
            raise WebSearchError(f"SearXNG returned HTTP {response.status_code}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise WebSearchError(
                _JSON_DISABLED_HINT.format(status=response.status_code)
            ) from exc
        items = payload.get("results", []) if isinstance(payload, dict) else []

        results: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "")
            title = str(item.get("title") or "No Title")
            content = str(item.get("content") or "")[:_CONTENT_CHARS]
            published_date = str(item.get("publishedDate") or "")
            raw_score = item.get("score")
            score = (
                float(raw_score)
                if isinstance(raw_score, int | float) and not isinstance(raw_score, bool)
                else _DEFAULT_SCORE
            )
            results.append(
                {
                    "url": url,
                    "title": title,
                    "content": content,
                    "raw_content": content,
                    "published_date": published_date,
                    "score": score,
                    "financial_relevance": financial_relevance(
                        url=url,
                        text=content,
                        title=title,
                        published_date=published_date or None,
                    ),
                    "provider": "searxng",
                    "author": "",
                    "domain": extract_domain(url),
                    "is_authoritative": is_authoritative_source(url),
                }
            )

        results.sort(key=lambda x: (x["financial_relevance"], x["score"]), reverse=True)
        return results
```

- [x] **Step 4: Name the provider in the package docstring**

In `maverick/research/providers/__init__.py`, change the first line to:

```python
"""Web search providers (`WebSearchProvider`, `ExaSearchProvider`, `SearXNGProvider`). Third-layer sibling: imports config and types.
```

- [x] **Step 5: Run the tests to verify they pass**

Run:
```bash
uv run pytest tests/research/test_searxng.py tests/research/test_providers.py --timeout=60
uv run ruff check maverick/research tests/research && uv run ruff format --check maverick/research tests/research
uv run lint-imports
```
Expected: all pass; ruff clean; `0 broken`. The financial-relevance assertion of `0.6` is the SEC domain tier (0.4) plus two keyword hits (`revenue`, `earnings`: 0.1) plus the title term `quarterly` (0.1). If ruff format rewrites a line, apply `uv run ruff format maverick/research tests/research` and re-run.

- [x] **Step 6: Commit**

Run:
```bash
git add maverick/research/providers/searxng.py maverick/research/providers/__init__.py tests/research/test_searxng.py
git commit -m "feat(research): add a SearXNG search provider on the platform HTTP seam"
```

### Task 19: Select the backend in the service and name it in the prerequisite check

**Files:**
- Modify: `maverick/research/service.py` (imports, `_build_default_agent`, `_configuration_error`, one new helper)
- Modify: `maverick/research/service_support.py` (`configuration_problem`)
- Test: `tests/research/test_service.py`

**Interfaces:**
- Consumes: `SearXNGProvider`, `time_range_for` (Task 18); `ResearchSettings.search_backend`, `ResearchSettings.searxng_base_url` (Task 17).
- Produces: `configuration_problem(*, search_backend: str, search_configured: bool, llm_provider: str | None, valid_llm_providers: str)`; `ResearchService._search_configured() -> bool`.

- [x] **Step 1: Write the failing tests**

Append to `tests/research/test_service.py` (the module already imports `ResearchSettings`, `ResearchService`, `ResearchError`, `pytest`, and defines `FakeAgent`, `_fixture_report`, `_configured_settings`, and the `configured_llm` fixture):

```python


# ---------------------------------------------------------------------------
# Search backend selection (SearXNG, #186)
# ---------------------------------------------------------------------------


async def test_run_comprehensive_errors_when_searxng_not_configured(configured_llm):
    service = ResearchService(settings=ResearchSettings(search_backend="searxng"))

    result = await service.run_comprehensive("AAPL outlook")

    assert isinstance(result, ResearchError)
    assert result.error_type == "not_configured"
    assert "SearXNG search backend not configured" in result.error
    assert result.model_extra is not None
    assert result.model_extra["details"]["searxng_base_url"].startswith("Missing")


async def test_searxng_backend_does_not_require_an_exa_key(configured_llm):
    service = ResearchService(
        settings=_configured_settings(
            exa_api_key=None,
            search_backend="searxng",
            searxng_base_url="http://searx.local:8080",
        ),
        agent_factory=lambda **_kw: FakeAgent(report=_fixture_report()),
    )

    result = await service.run_comprehensive("AAPL outlook")

    assert not isinstance(result, ResearchError)


async def test_default_agent_factory_picks_the_searxng_provider(monkeypatch):
    from maverick.research.providers.searxng import SearXNGProvider

    captured: dict[str, Any] = {}

    class _StubAgent:
        def __init__(self, *, search_clients, persona, default_depth) -> None:
            captured["clients"] = search_clients

    monkeypatch.setattr("maverick.research.service.DeepResearchAgent", _StubAgent)
    service = ResearchService(
        settings=ResearchSettings(
            search_backend="searxng", searxng_base_url="http://searx.local:8080"
        )
    )

    service._build_default_agent(persona="moderate", default_depth="basic")

    (client,) = captured["clients"]
    assert isinstance(client, SearXNGProvider)
    assert client.base_url == "http://searx.local:8080"
    assert client.time_range == "month"  # default_timeframe "1m"


async def test_default_agent_factory_picks_exa_by_default(monkeypatch):
    from maverick.research.providers.exa import ExaSearchProvider

    captured: dict[str, Any] = {}

    class _StubAgent:
        def __init__(self, *, search_clients, persona, default_depth) -> None:
            captured["clients"] = search_clients

    monkeypatch.setattr("maverick.research.service.DeepResearchAgent", _StubAgent)
    service = ResearchService(settings=_configured_settings())

    service._build_default_agent(persona="moderate", default_depth="basic")

    (client,) = captured["clients"]
    assert isinstance(client, ExaSearchProvider)
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/research/test_service.py -k "searxng or picks" --timeout=60`
Expected: the four new tests FAIL (the not-configured test reports the Exa message; the factory test raises the Exa assertion).

- [x] **Step 3: Generalize the prerequisite check**

In `maverick/research/service_support.py`, replace `configuration_problem` with:

```python
def configuration_problem(
    *,
    search_backend: str,
    search_configured: bool,
    llm_provider: str | None,
    valid_llm_providers: str,
) -> tuple[str, dict[str, Any]] | None:
    """Return `(message, details)` for the first missing prerequisite (the selected
    search backend, then the BYOK LLM), or `None` when both are configured. See
    `service.py`'s module docstring, "Configuration errors" section."""
    if not search_configured:
        if search_backend == "searxng":
            return (
                "Research functionality unavailable - SearXNG search backend not configured",
                {
                    "required_configuration": (
                        "SEARXNG_BASE_URL is required when RESEARCH_SEARCH_BACKEND=searxng"
                    ),
                    "searxng_base_url": (
                        "Missing (configure SEARXNG_BASE_URL environment variable)"
                    ),
                    "setup_instructions": (
                        "Point SEARXNG_BASE_URL at a SearXNG instance with the json "
                        "format enabled (settings.yml: search.formats includes json), "
                        "or set RESEARCH_SEARCH_BACKEND=exa with EXA_API_KEY"
                    ),
                },
            )
        return (
            "Research functionality unavailable - Exa search provider not configured",
            {
                "required_configuration": "Exa search provider API key is required",
                "exa_api_key": "Missing (configure EXA_API_KEY environment variable)",
                "setup_instructions": "Get a free API key from: Exa (exa.ai)",
            },
        )
    if llm_provider is None:
        return (
            "Research functionality unavailable - no LLM configured",
            {
                "required_configuration": (
                    f"Set LLM_PROVIDER (one of: {valid_llm_providers}) plus "
                    "LLM_API_KEY and LLM_MODEL"
                ),
            },
        )
    return None
```

- [x] **Step 4: Select the provider in the service**

In `maverick/research/service.py`:

1. Add the import after the `ExaSearchProvider` import:

```python
from maverick.research.providers.base import WebSearchProvider
from maverick.research.providers.searxng import SearXNGProvider, time_range_for
```

2. Replace `_build_default_agent` with:

```python
    def _build_default_agent(
        self, *, persona: Persona, default_depth: ResearchDepth
    ) -> DeepResearchAgent:
        search: WebSearchProvider
        if self._settings.search_backend == "searxng":
            assert self._settings.searxng_base_url is not None, (
                "_build_default_agent must only run after _configuration_error confirms "
                "searxng_base_url is set"
            )
            search = SearXNGProvider(
                self._settings.searxng_base_url,
                settings=self._settings,
                time_range=time_range_for(self._settings.default_timeframe),
            )
        else:
            assert self._settings.exa_api_key is not None, (
                "_build_default_agent must only run after _configuration_error confirms "
                "exa_api_key is set"
            )
            search = ExaSearchProvider(
                self._settings.exa_api_key.get_secret_value(), settings=self._settings
            )
        return DeepResearchAgent(
            search_clients=[search], persona=persona, default_depth=default_depth
        )

    def _search_configured(self) -> bool:
        """Whether the selected search backend has what it needs."""
        if self._settings.search_backend == "searxng":
            return self._settings.searxng_base_url is not None
        return self._settings.exa_api_key is not None
```

3. In `_configuration_error`, replace the `configuration_problem(` call with:

```python
        problem = configuration_problem(
            search_backend=self._settings.search_backend,
            search_configured=self._search_configured(),
            llm_provider=provider.value if provider is not None else None,
            valid_llm_providers=_VALID_LLM_PROVIDERS,
        )
```

4. In the module docstring, change the sentence `The default factory (\`_build_default_agent\`) constructs a real \`ExaSearchProvider\` +` to `The default factory (\`_build_default_agent\`) constructs a real \`ExaSearchProvider\` (or \`SearXNGProvider\`, per \`ResearchSettings.search_backend\`) +`.

- [x] **Step 5: Run the research suite and the gate**

Run:
```bash
uv run pytest tests/research --timeout=60
make check
```
Expected: all research tests pass, including the pre-existing Exa not-configured tests; `All checks passed!`.

- [x] **Step 6: Commit**

Run:
```bash
git add maverick/research/service.py maverick/research/service_support.py tests/research/test_service.py
git commit -m "feat(research): select the search backend from settings and name it in the not-configured error"
```

### Task 20: Document the backend, declare the variables, merge, close #186

**Files:**
- Modify: `.env.example` (after the `EXA_API_KEY` line), `docs/features/deep-research.md` (Configuration section), `README.md:492-493`, `README.md:531`, `README.md:656`, `server.json` (environment variable arrays)

- [x] **Step 1: .env.example**

After the line `# EXA_API_KEY=your_exa_api_key_here`, add:

```
# Or point research at a self-hosted SearXNG instance instead of Exa (no key).
# The instance must serve JSON: settings.yml -> search: formats: [html, json]
# RESEARCH_SEARCH_BACKEND=exa  # exa | searxng
# SEARXNG_BASE_URL=http://localhost:8080
```

- [x] **Step 2: Feature doc**

In `docs/features/deep-research.md`, change the Configuration intro sentence

```
Research needs two independent things configured: a search provider (Exa)
and a BYOK LLM.
```

to

```
Research needs two independent things configured: a search backend (Exa by
default, or a self-hosted SearXNG instance) and a BYOK LLM.
```

and replace the whole `### Search provider` section (its heading through the blank line before `### BYOK LLM`) with:

````markdown
### Search backend

The default backend is Exa:

```bash
EXA_API_KEY=your_exa_key
```

To use a self-hosted SearXNG instance instead, with no API key, select the
backend and point it at the instance:

```bash
RESEARCH_SEARCH_BACKEND=searxng
SEARXNG_BASE_URL=http://localhost:8080
```

The instance must serve the JSON API. Most SearXNG installs ship with it
disabled; enable it in the instance's `settings.yml`:

```yaml
search:
  formats:
    - html
    - json
```

A 403 from the instance is reported by the research tools with that
instruction. SearXNG returns result snippets rather than page text, so
`raw_content` equals `content` for its results, and the recency window
follows `RESEARCH_DEFAULT_TIMEFRAME` (`1m` maps to SearXNG's `month`
range). The selected backend is the one "is research configured" gate for
search: `EXA_API_KEY` for Exa, `SEARXNG_BASE_URL` for SearXNG.

````

- [x] **Step 3: README**

Replace

```
Requires `EXA_API_KEY` (web search) plus a configured BYOK LLM
(`LLM_PROVIDER`/`LLM_API_KEY`/`LLM_MODEL`; see [Configuration](#configuration)).
```

with

```
Requires a search backend (`EXA_API_KEY` for Exa, the default, or
`RESEARCH_SEARCH_BACKEND=searxng` plus `SEARXNG_BASE_URL` for a self-hosted
SearXNG instance) and a configured BYOK LLM
(`LLM_PROVIDER`/`LLM_API_KEY`/`LLM_MODEL`; see [Configuration](#configuration)).
```

Replace

```
- `EXA_API_KEY` - Web search for the research tools (get at [exa.ai](https://exa.ai)).
```

with

```
- `EXA_API_KEY` - Web search for the research tools (get at [exa.ai](https://exa.ai)).
- `RESEARCH_SEARCH_BACKEND` - `exa` (default) or `searxng`.
- `SEARXNG_BASE_URL` - Base URL of a self-hosted SearXNG instance with the JSON format enabled; used when the backend is `searxng`.
```

Replace

```
- Ensure `EXA_API_KEY` is set for web search
```

with

```
- Ensure `EXA_API_KEY` is set for web search, or `RESEARCH_SEARCH_BACKEND=searxng` with `SEARXNG_BASE_URL`
```

- [x] **Step 4: server.json**

Run:
```bash
uv run python - <<'EOF'
import json
from pathlib import Path

entries = [
    {
        "name": "RESEARCH_SEARCH_BACKEND",
        "description": "Optional research search backend: exa (default) or searxng.",
        "isRequired": False,
        "isSecret": False,
    },
    {
        "name": "SEARXNG_BASE_URL",
        "description": "Optional base URL of a self-hosted SearXNG instance with the JSON format enabled; used when RESEARCH_SEARCH_BACKEND=searxng.",
        "isRequired": False,
        "isSecret": False,
    },
]
path = Path("server.json")
data = json.loads(path.read_text())
for package in data["packages"]:
    if "environmentVariables" in package:
        package["environmentVariables"].extend(entries)
for remote in data["remotes"]:
    remote["environmentVariables"].extend(entries)
path.write_text(json.dumps(data, indent=2) + "\n")
print("server.json updated")
EOF
uv run --with jsonschema python3 - <<'EOF'
import json, urllib.request, jsonschema
data = json.load(open("server.json"))
schema = json.loads(urllib.request.urlopen(data["$schema"]).read())
jsonschema.validate(data, schema)
print("server.json is valid")
EOF
```
Expected: `server.json updated` and `server.json is valid`.

- [x] **Step 5: Gate, commit, PR, merge**

Run:
```bash
make check && make docs-check && uv run pytest --timeout=60 2>&1 | tail -3
git add .env.example docs/features/deep-research.md README.md server.json
git commit -m "docs(research): document the SearXNG backend and declare its variables"
git push -u origin feat/searxng-backend
gh pr create --title "feat(research): self-hosted SearXNG search backend" --body-file - <<'EOF'
Closes #186.

- `SearXNGProvider` on the platform HTTP seam (rate limiter, breaker, retry), keyless, JSON API, results normalized to the Exa shape; a 403 from an instance with the json format disabled is reported with the settings.yml fix.
- `RESEARCH_SEARCH_BACKEND=exa|searxng` and `SEARXNG_BASE_URL` settings; the service builds the selected provider and the not-configured error names the right variable.
- Financial-relevance scoring lifted into `providers/scoring.py`, shared by both providers.
- Stubbed-HTTP tests only; docs, .env.example, and server.json updated.
EOF
gh pr checks --watch && gh pr merge --squash --delete-branch
git checkout main && git pull --ff-only
git worktree remove ../maverick-mcp-searxng
```

- [x] **Step 6: Reply on #186**

Run (substitute the merged PR number):
```bash
gh issue comment 186 --body "Shipped in #<PR>. Set RESEARCH_SEARCH_BACKEND=searxng and SEARXNG_BASE_URL=http://your-instance:8080 (with the json format enabled on the instance) and the research tools run without an Exa key. Details in docs/features/deep-research.md. Thanks for the idea and the offer to contribute; domain presets and multi-step retrieval stay as follow-ups if there is demand."
gh issue view 186 --json state --jq .state
```
Expected: `CLOSED` (the `Closes #186` line in the PR closes it on merge; if it prints `OPEN`, run `gh issue close 186 --reason completed`).

---

## Workstream 5: v1.1.0 release and distribution close-out

### Task 21: Version bump, release notes, runbook refresh

**Files:**
- Modify: `pyproject.toml:3`, `maverick/__init__.py:8`, `tests/structure/test_package.py`, `server.json` (three version fields), `docs/runbooks/releasing.md`
- Create: `docs/generated/release-notes/v1.1.0.md`
- Modify: `docs/CATALOG.md` (row for the release notes), `docs/INDEX.md` (line for the release notes)

- [x] **Step 1: Branch**

Run:
```bash
git checkout -b release/v1.1.0-prep main
```

- [x] **Step 2: Bump the version in code and metadata**

Run:
```bash
sed -i 's/^version = "1.0.0"$/version = "1.1.0"/' pyproject.toml
sed -i 's/^__version__ = "1.0.0.dev0"$/__version__ = "1.1.0"/' maverick/__init__.py
sed -i 's/assert maverick.__version__ == "1.0.0.dev0"/assert maverick.__version__ == "1.1.0"/' tests/structure/test_package.py
uv run python - <<'EOF'
import json
from pathlib import Path

path = Path("server.json")
data = json.loads(path.read_text())
data["version"] = "1.1.0"
for package in data["packages"]:
    package["version"] = "1.1.0"
    if package["registryType"] == "oci":
        package["identifier"] = "ghcr.io/wshobson/maverick-mcp:1.1.0"
path.write_text(json.dumps(data, indent=2) + "\n")
print("server.json at 1.1.0")
EOF
grep -n '"version"\|"identifier"' server.json
uv lock
git diff --stat
```
Expected: `pyproject.toml`, `maverick/__init__.py`, `tests/structure/test_package.py`, `server.json`, and `uv.lock` (the project's own version entry) changed; every version field reads 1.1.0.

- [x] **Step 3: Write the release notes**

Create `docs/generated/release-notes/v1.1.0.md`:

```markdown
## maverick-mcp-server v1.1.0

The first release published end to end by the tag-triggered workflow: PyPI,
the official MCP Registry, and the GHCR image.

### Fixes (backtesting)

Three bugs reported and fixed by @A1-NWS-Dev1:

- `backtesting_backtest_portfolio` reported the mildest constituent drawdown
  as the portfolio `max_drawdown`; it now reports the worst (#245, #242).
- `backtesting_create_strategy_ensemble` built SMA-crossover variants for
  `"rsi"` and `"macd"` and collapsed the ensemble onto one `"SMA Crossover"`
  key; each member now runs its own signal logic under its own template name
  (#246, #243). Result keys are template display names such as
  `"RSI Mean Reversion"`.
- `backtesting_analyze_market_regimes` could report `method: "hmm"` and a
  fabricated uniform probability vector after a silent fallback; it now
  reports `method: "threshold"` when it falls back and returns one-hot
  probabilities in that case. `method="hmm"` fits a Gaussian mixture, not a
  Hidden Markov Model; the name is kept for compatibility (#247, #244).

### FastMCP 4

The server runs on FastMCP 4 (MCP Python SDK v2) and serves the
`2026-07-28` protocol revision alongside `2025-11-25` (#235). Tool
annotations are declared with SDK v2 field names; the wire format is
unchanged. `fastmcp>=4.0.3` is the new floor; the direct `mcp` dependency
is gone.

### SearXNG research backend

Research can run against a self-hosted SearXNG instance instead of Exa, with
no API key: set `RESEARCH_SEARCH_BACKEND=searxng` and `SEARXNG_BASE_URL`
(#186, requested by @Josephur). The instance must enable the JSON format;
see `docs/features/deep-research.md`.

### Dependencies

langchain-anthropic 1.7.0, redis 8.1.0, greenlet 3.5.5, uvicorn 0.52.4,
nltk 3.10.3 (dev only, security fixes).

### Install

```bash
uvx --from maverick-mcp-server==1.1.0 maverick-mcp --transport stdio
pip install "maverick-mcp-server[backtesting,research]==1.1.0"
```
```

- [x] **Step 4: Catalog the release notes**

Add to `docs/CATALOG.md` in the Current table, after the `generated/registry/smithery.yaml` row:

```markdown
| `generated/release-notes/v1.1.0.md` | current | engineering | Release notes for v1.1.0 (used by `gh release create --notes-file`). |
```

Add to `docs/INDEX.md` under "Current Product And Technical Docs", after the `generated/registry/README.md` bullet:

```markdown
- `generated/release-notes/v1.1.0.md` - release notes for v1.1.0.
```

- [x] **Step 5: Refresh the release runbook**

Run:
```bash
python3 - <<'EOF'
from pathlib import Path

path = Path("docs/runbooks/releasing.md")
text = path.read_text()
old = (
    "v1.0.0 is already tagged and released on GitHub\n"
    "(`gh release list` shows `v1.0.0`). The steps below take that release the"
)
new = (
    "v1.0.0 was released on GitHub only and never reached PyPI. v1.1.0 is the\n"
    "first tag the publish workflow carries end to end. The steps below take that release the"
)
assert old in text, "runbook intro changed; edit by hand"
text = text.replace(old, new)
text = text.replace("v1.0.0 GitHub release", "v1.1.0 GitHub release")
text = text.replace("git push origin v1.0.0", "git push origin v1.1.0")
text = text.replace("maverick-mcp:1.0.0", "maverick-mcp:1.1.0")
text = text.replace('"version": "1.0.0"', '"version": "1.1.0"')
text = text.replace("gh release upload v1.0.0", "gh release upload v1.1.0")
text = text.replace("Publishing 1.0.0 is a one-time", "Publishing 1.1.0 is a one-time")
path.write_text(text)
print("runbook refreshed")
EOF
grep -n '1\.0\.0' docs/runbooks/releasing.md
```
Expected: the remaining `1.0.0` mentions are historical ("v1.0.0 was released on GitHub only", the `.mcpb` prose). Read each remaining line and change it only if it instructs a command.

- [x] **Step 6: Gate, commit, PR, merge**

Run:
```bash
make check && make docs-check && uv run pytest --timeout=60 2>&1 | tail -3
git add pyproject.toml maverick/__init__.py tests/structure/test_package.py server.json uv.lock docs/generated/release-notes/v1.1.0.md docs/CATALOG.md docs/INDEX.md docs/runbooks/releasing.md
git commit -m "release: prepare v1.1.0 (version bump, release notes, runbook)"
git push -u origin release/v1.1.0-prep
gh pr create --title "release: prepare v1.1.0" --body "Version 1.1.0 in pyproject, the package, server.json, and the structure test; release notes under docs/generated/release-notes; the release runbook generalized from v1.0.0. No tag is pushed by this PR."
gh pr checks --watch && gh pr merge --squash --delete-branch
git checkout main && git pull --ff-only
```

### Task 22 (P): Tag v1.1.0 and publish (owner go-ahead required)

**Files:** `README.md:95-97` after PyPI is live.

Stop here and ask the owner. This task pushes a public tag and, through `publish.yml`, publishes to PyPI (permanent), the MCP Registry, and GHCR.

- [ ] **Step 1 (owner): Configure PyPI trusted publishing**

The owner, on pypi.org: Account → Publishing → "Add a new pending publisher" (the project does not exist on PyPI yet): PyPI project name `maverick-mcp-server`, owner `wshobson`, repository `maverick-mcp`, workflow `publish.yml`, environment `pypi`. Then confirm the GitHub environment exists:

```bash
gh api repos/wshobson/maverick-mcp/environments --jq '.environments[].name'
```
Expected: `pypi` in the list. If not, create it: repository Settings → Environments → New environment → `pypi` (no protection rules needed).

- [ ] **Step 2: Create the release, which pushes the tag and starts the workflow**

Run:
```bash
gh release create v1.1.0 --target main --title "maverick-mcp-server v1.1.0" --notes-file docs/generated/release-notes/v1.1.0.md
sleep 10
gh run list --workflow Publish --limit 1
```
Then `gh run watch <run-id>` until it finishes. Expected: `build`, `publish-pypi`, `publish-mcp-registry`, and `publish-ghcr` all succeed. If `publish-pypi` fails with `invalid-publisher`, the pending publisher fields do not match the workflow; fix them on PyPI and re-run the job with `gh run rerun <run-id> --failed`.

- [ ] **Step 3: Verify PyPI and the console script from a clean environment**

Run:
```bash
curl -s https://pypi.org/pypi/maverick-mcp-server/json | python3 -c 'import sys, json; d = json.load(sys.stdin); print("pypi latest:", d["info"]["version"])'
uvx --refresh --from maverick-mcp-server==1.1.0 maverick-mcp --help | head -5
```
Expected: `pypi latest: 1.1.0` and the CLI usage text.

- [ ] **Step 4: Verify the registry listing and the image**

Follow `docs/runbooks/releasing.md` Step 2 for the registry verification command, and run:
```bash
docker run --rm ghcr.io/wshobson/maverick-mcp:1.1.0 --help | head -3
```
Expected: the CLI usage text from the container.

- [ ] **Step 5: Remove the "not yet published" README note**

In `README.md`, delete the three-line block starting `> **Note:** v1.0.0 is not yet published to PyPI` and the blank line after it; change the headings `#### Option 1: Run without installing (uvx, once published)` to `#### Option 1: Run without installing (uvx)` and `#### Option 2: pip install (once published)` to `#### Option 2: pip install`. Then:

```bash
git checkout -b docs/pypi-live main
git add README.md && git commit -m "docs: maverick-mcp-server is on PyPI"
git push -u origin docs/pypi-live
gh pr create --title "docs: package is on PyPI" --body "Removes the install-from-source caveat now that v1.1.0 is published."
gh pr checks --watch && gh pr merge --squash --delete-branch
git checkout main && git pull --ff-only
```

### Task 23 (P): Bundle, Docker catalog, third-party registries (owner go-ahead required)

**Files:** `docs/generated/registry/docker-mcp-catalog.md:77` (commit pin)

- [ ] **Step 1: Build, validate, and attach the bundle**

Run:
```bash
make bundle
npx -y @anthropic-ai/mcpb validate dist/manifest.json
gh release upload v1.1.0 dist/maverick-mcp.mcpb --clobber
gh release view v1.1.0 --json assets --jq '.assets[].name'
```
Expected: the validator reports the manifest valid and the asset list shows `maverick-mcp.mcpb`. The bundle launches `uvx --from maverick-mcp-server==1.1.0`, which works now that PyPI exists.

- [ ] **Step 2: Update the Docker MCP Catalog submission**

Run:
```bash
git rev-parse v1.1.0
```
Replace the commit on line 77 of `docs/generated/registry/docker-mcp-catalog.md` with that hash and change its trailing comment to `# v1.1.0 tag commit`. Then, in the owner's fork of `docker/mcp-registry` that backs PR #4490, update the server entry to the same commit and version, push, and comment on the PR asking for review. Commit the draft change here via a small docs PR (`docs: point the Docker catalog draft at v1.1.0`).

- [ ] **Step 3: File the remaining submissions**

Using the drafts in `docs/generated/registry/` (Smithery, Glama, PulseMCP, mcp.so), each under the owner's account. Record on `docs/exec-plans/active/2026-07-20-phase-9-distribution.md` which ones were filed and their URLs, then move that plan to `docs/exec-plans/completed/` (update its `docs/CATALOG.md` and `docs/INDEX.md` rows) in the same docs PR as Step 2.

---

## Workstream 6: memory and documentation

### Task 24: Vault updates

**Files (in `/home/wshobson/vaults/Maverick/`):**
- Modify: `projects/maverick-mcp.md`, `TODO.md`, `people/_index.md`, `_changelog.md`
- Create: `decisions/2026-09-05-fastmcp-4-migration.md`, `decisions/2026-09-05-searxng-research-backend.md`, `people/a1-nws-dev1.md`, `people/josephur.md`

Run this task in pieces as workstreams close; the changelog gets one line per write.

- [ ] **Step 1: Project page entries**

Append to `projects/maverick-mcp.md` before the `## Working agreements` heading, one entry per completed workstream, each dated. Write the numbers from the actual command outputs (the pytest summary line and the conformance verdict), not estimates:

```markdown

2026-09-05: open-items sweep, design at `docs/design-docs/2026-09-05-open-items-remediation.md`, plan under `docs/exec-plans/`. Triage: #241 (marketing pitch) and #254 (security-audit offer) closed with replies; the mcp 2.x dependabot PR (#249) held because it silently carried fastmcp 3.3.1 -> 4.0.0. Root SECURITY.md refreshed for v1.x. Contributor A1-NWS-Dev1's three backtesting fixes merged in order (#242 drawdown min(), #243 TemplateStrategy for ensembles, #244 honest regime fallback), closing #245/#246/#247; #244 needed a rebase plus two maintainer fixups (line cap, seeded mixture tests). Five dependabot bumps merged after a combined local lock check.

2026-09-05 (later): FastMCP 4 migration merged (fastmcp>=4.0.3, direct mcp pin dropped, annotations declared snake_case, test suite runs with the camelCase bridge off). Conformance on 2026-07-28 and 2025-11-25 recorded on #235; see the decision record. SearXNG research backend merged for #186 (RESEARCH_SEARCH_BACKEND, SEARXNG_BASE_URL, provider on the platform HTTP seam, scoring shared via providers/scoring.py). v1.1.0 prepared; publish steps remain owner-gated (PyPI trusted publishing still unconfigured as of the design).
```

Then bump the frontmatter `updated:` to the date of the last entry written.

- [ ] **Step 2: Decision records**

Create `decisions/2026-09-05-fastmcp-4-migration.md`:

```markdown
---
type: decision
status: active
tags:
  - maverick-mcp
  - mcp
updated: 2026-09-05
---
# Maverick MCP moves to FastMCP 4 as a deliberate migration, not a lock bump

Seth approved the migration on 2026-09-05 as part of the open-items design. FastMCP 4.0.0 went stable on 2026-08-31 (4.0.3 on 2026-09-05) on the MCP Python SDK v2, which serves the 2026-07-28 protocol revision; that was the stated trigger on issue #235 since 2026-08-26.

Decision: raise the floor to `fastmcp>=4.0.3` in its own PR; drop the direct `mcp` pin (nothing imports it; FastMCP owns the SDK version); keep `httpx` pinned directly because the platform HTTP seam imports it and FastMCP 4 no longer depends on it; declare tool annotations with SDK v2 snake_case names; run the test suite with FastMCP's camelCase compatibility bridge off so stale reads fail now. Dependabot's #249, which reached fastmcp 4.0.0 through the lock file alone, was held and superseded.

Why: the exposure was small (constructor, decorators, `run(transport=...)`, in-memory `Client`), CI already passed on 4.0.0 through #249, and a major framework version deserves an explicit floor and a conformance run rather than an incidental lock change.

Verification: full gate, stdio and HTTP smoke tests, and `@hasmcp/mcp-spec-test` 0.1.5 on both revisions; the reports are attached to #235.

See [[maverick-mcp]].
```

Create `decisions/2026-09-05-searxng-research-backend.md`:

```markdown
---
type: decision
status: active
tags:
  - maverick-mcp
  - research
updated: 2026-09-05
---
# Research gets a keyless SearXNG backend beside Exa, selected by one setting

Seth approved the backend on 2026-09-05, scoping it to the first PR he had outlined on issue #186 on 2026-08-15: a `SearXNGProvider` on the existing `WebSearchProvider` seam, a `RESEARCH_SEARCH_BACKEND=exa|searxng` setting with `SEARXNG_BASE_URL`, the prerequisite check relaxed to "the selected backend is configured", and stubbed-HTTP tests only.

Design choices: requests go through the platform HTTP seam (rate limiter, breaker, retry) with breaker thresholds from research settings, matching Exa; results normalize to Exa's dict shape so the agents cannot tell backends apart; the financial-relevance scoring moved to `providers/scoring.py` and is shared; a 403 from an instance with the JSON format disabled is reported with the `settings.yml` fix; the recency window follows `RESEARCH_DEFAULT_TIMEFRAME`. Domain presets and multi-step retrieval are deferred until asked for.

Why now: the request had a scoped design and an offered contribution that never arrived; the seam survived the v1.0 rewrite intact, so the cost was one provider module plus settings.

See [[maverick-mcp]] and [[josephur]].
```

- [ ] **Step 3: People pages**

Create `people/a1-nws-dev1.md`:

```markdown
---
type: person
tags: [maverick-mcp, open-source]
aliases: ["A1-NWS-Dev1"]
updated: 2026-09-05
---

# A1-NWS-Dev1

- GitHub contributor to [[maverick-mcp]]. On 2026-08-30 filed three precise
  backtesting bug reports (#245 drawdown aggregation, #246 ensemble strategy
  mapping, #247 regime fallback) with paired PRs (#242, #243, #244), each
  with regression tests, docs, and same-day follow-ups to review comments.
- Works on Windows 11 / Python 3.12 and did not run `tests/structure`, so the
  500-line cap caught #244; otherwise high-quality, template-conformant
  submissions. Worth inviting to future backtesting work.
```

Create `people/josephur.md`:

```markdown
---
type: person
tags: [maverick-mcp, open-source]
aliases: ["Josephur"]
updated: 2026-09-05
---

# Josephur

- GitHub user who requested a self-hosted SearXNG research backend for
  [[maverick-mcp]] (#186, 2026-05-26) and offered to contribute. Runs local
  LLM stacks with a self-hosted SearXNG instance. The backend shipped in
  2026-09; see [[2026-09-05-searxng-research-backend]].
```

Add two rows to the table in `people/_index.md`:

```markdown
| [[a1-nws-dev1]] | open-source contributor, Maverick MCP | Filed and fixed three backtesting bugs (2026-08-30). |
| [[josephur]] | open-source requester, Maverick MCP | Asked for the SearXNG research backend (#186). |
```

and bump that file's `updated:`.

- [ ] **Step 4: Open loops**

In `TODO.md`: when #235 closes, delete the line `- [ ] [[maverick-mcp]] — revisit issue #235 once FastMCP 4 ships stable (needs mcp 2.x)`. Under `## Everything else`, add while it is still true:

```markdown
- [ ] [[maverick-mcp]] — configure PyPI trusted publishing for publish.yml (pending publisher for maverick-mcp-server, environment pypi), then tag v1.1.0; the .mcpb bundle and the MCP Registry listing wait on it (plan Task 22)
```

- [ ] **Step 5: Changelog lines**

Append one line per write, in this shape:

```
2026-09-05 [claude] projects/maverick-mcp.md — open-items sweep entries (triage, contributor PRs, deps, FastMCP 4, SearXNG, v1.1.0 prep)
2026-09-05 [claude] decisions/2026-09-05-fastmcp-4-migration.md — new decision record
2026-09-05 [claude] decisions/2026-09-05-searxng-research-backend.md — new decision record
2026-09-05 [claude] people/a1-nws-dev1.md, people/josephur.md, people/_index.md — contributor pages added
2026-09-05 [claude] TODO.md — #235 loop closed; PyPI trusted-publishing loop added
```

### Task 25: Close the plan

**Files:**
- Move: `docs/exec-plans/active/2026-09-05-open-items-remediation.md` to `docs/exec-plans/completed/`
- Modify: `docs/CATALOG.md`, `docs/INDEX.md` (paths)

- [ ] **Step 1: Confirm the exit criteria**

Run:
```bash
gh issue list --state open
gh pr list --state open
gh run list --branch main --limit 1
```
Expected: no open PRs; the only open issues are ones this plan deliberately left open with a recorded blocker (#235 if residual failures were upstream); the latest `main` run is green.

- [ ] **Step 2: Move the plan and fix the catalog**

Run:
```bash
git checkout -b docs/close-remediation-plan main
git mv docs/exec-plans/active/2026-09-05-open-items-remediation.md docs/exec-plans/completed/2026-09-05-open-items-remediation.md
sed -i 's#exec-plans/active/2026-09-05-open-items-remediation.md#exec-plans/completed/2026-09-05-open-items-remediation.md#' docs/CATALOG.md docs/INDEX.md
make docs-check
git add -A docs/
git commit -m "docs: close the 2026-09-05 open-items remediation plan"
git push -u origin docs/close-remediation-plan
gh pr create --title "docs: close the 2026-09-05 remediation plan" --body "Every task in the plan is done or handed to the owner; moving it to completed."
gh pr checks --watch && gh pr merge --squash --delete-branch
git checkout main && git pull --ff-only
```

---

## Execution addenda (2026-09-05)

Deviations and lessons from executing Tasks 1 to 21 inline in one session.

1. **Local gates and a leaked key.** The executing shell exported a real `EXA_API_KEY`, so the "Exa not configured" research test silently built a live agent and made network calls. PR #257 (not in the plan) makes the research conftest scrub the research and BYOK variables before every test; gates before it ran with `env -u EXA_API_KEY`. CI never had a key and was never affected.
2. **`gh pr checkout` has no `-q` flag**; a chain using it skipped the whole PR 242 gate once before the retry.
3. **Force-pushing a contributor's branch.** `gh pr checkout` configures the fork as a bare URL, so `--force-with-lease` has no remote-tracking ref ("stale info"). Use `--force-with-lease=<branch>:<contributor head sha>` from `gh pr view --json headRefOid`.
4. **Dependency batch.** `uv lock --upgrade-package` resolved langchain-anthropic 1.7.1 while dependabot's PR pinned 1.7.0; the merged lock carries dependabot's 1.7.0. The gate passed on both.
5. **camelCase reads lived in six test modules**, not only `tests/portfolio/test_tools.py`: market data, screening, technical, backtesting, research, and portfolio. The bridge-off fixture found them; the plan's grep had been truncated.
6. **Removed-API grep** matched `except httpx.TransportError` in `maverick/platform/http.py`. That guards our own outbound client, not anything handed to FastMCP; no change.
7. **Conformance harness.** `@hasmcp/mcp-spec-test` runs its tests with the working directory set to the npm package root, so `uv run python -m maverick.server` there resolves no project; the suite's own client ran whatever venv `VIRTUAL_ENV` pointed at, and its bundled official-SDK client started a bare interpreter and reported "Connection closed". Run it as `env -u VIRTUAL_ENV npx -y @hasmcp/mcp-spec-test@0.1.5 -c "uv run --directory /abs/path python -m maverick.server --transport stdio" --spec-version <rev>`.
8. **Conformance results (FastMCP 4.0.3, mcp 2.1.1).** 2025-11-25 conformant on stdio and HTTP. 2026-07-28 over HTTP: 31 passed, 2 failed (negotiated-version response header not echoed; version-less request answered 400), 7 not verified. Over stdio the SDK serves the handshake era only. FastMCP 3, run the same way, fails the 2025-11-25 `CallToolResult` content check (#234's finding). #235 stays open as the tracker for the three upstream items. Reports are attached to #235.
9. **Scoring helpers** live in `maverick/research/providers/scoring.py`, not `providers/base.py`; the spec was updated to match.
10. **Factory tests** in `tests/research/test_service.py` compare against the provider classes held by `maverick.research.service`: `tests/research/test_providers.py` re-imports the provider modules, so `isinstance` against a fresh import fails. The first Task 19 commit landed with that failure because `pytest | tail` hid the exit code; it was amended. Use `set -o pipefail` in gate chains.
11. **Worktree sequencing.** The SearXNG worktree was cut from the FastMCP 4 branch before #258 merged and replayed onto `main` with `git rebase --onto main <fastmcp-4 tip> feat/searxng-backend`.
12. **Release runbook.** Option A told the owner to leave the PyPI trusted publisher's environment blank while `publish.yml` runs its PyPI job in the GitHub environment `pypi`; that mismatch is the likely cause of the 2026-07-20 `invalid-publisher` failure. Corrected in #260: the publisher must name environment `pypi`.
13. **Task 22 outcome.** The tag and GitHub release were created and the GHCR image published, but the PyPI job failed twice with `invalid-publisher`. Root cause: a PyPI project named `maverick-mcp-server` already exists under another account, created by an unrelated product (Day-AI-Labs' agent runtime) that released 0.1.3 to 0.1.6 on 2026-05-29 to 05-31 and removed everything on 2026-06-10; the pending-publisher form reports "This project already exists". The owner chose a PEP 541 transfer request over a rename (filed as https://github.com/pypi/support/issues/12150); until it lands, no install instruction points at the PyPI name (README, release notes, runbooks updated; the v1.0.0 `.mcpb` asset, which launched that name, was removed). Resume with `gh workflow run publish.yml -f confirm=publish` after adding the publisher on the transferred project.

Merged this session: #255, #256, #242, #243, #244, #248, #250, #251, #252, #253, #257, #258, #259, #260. Closed: #241, #254, #245, #246, #247, #186, #249. Open by design: #235 (upstream tracker). Owner-gated and not started: Tasks 22 and 23.

## Spec coverage

| Spec section | Tasks |
| --- | --- |
| WS0 triage (CI approval, #249 hold, #241, #254, SECURITY.md) | 2, 3, 4, 5, 6 |
| WS1 contributor fixes in order 242, 243, 244 with review checks and fixups | 7, 8, 9 |
| WS1 tech-debt line for the regime fallback | 6 |
| WS2 dependency batch with a combined local check | 10 |
| WS3 floor bump, drop `mcp` pin, `httpx` note | 11 |
| WS3 snake_case annotations, bridge off | 12 |
| WS3 upgrade checklist, transport smoke, docs | 13 |
| WS3 conformance and #235 | 14 |
| WS3 merge and #249 | 15 |
| WS4 provider, settings, service, prerequisite, tests, docs, #186 | 16, 17, 18, 19, 20 |
| WS5 version bump, release notes, runbook | 21 |
| WS5 owner-gated publish steps | 22, 23 |
| WS6 vault and repository docs | 1, 24, 25 |
