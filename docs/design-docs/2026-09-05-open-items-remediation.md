# Open items remediation and modernization design

Date: 2026-09-05
Status: approved (design approved by Seth Hobson in session on 2026-09-05)
Owner: Seth Hobson

This document is the approved design for clearing every issue and pull
request open on 2026-09-05 and for the two modernization items they unlock:
the FastMCP 4 migration and a self-hosted SearXNG research backend. The
implementation plan derived from it lives in `docs/exec-plans/active/` once
written.

## Summary

Nine pull requests and seven issues were open on 2026-09-05. Three PRs from
one external contributor fix confirmed bugs in the backtesting domain, but
none has run CI: the repository requires maintainer approval for workflow
runs from first-time contributors, and the runs sit in `action_required`.
Six dependabot PRs are green. One of them, the `mcp` 2.1.1 bump, silently
moves `fastmcp` from 3.3.1 to 4.0.0 in the lock file. FastMCP 4 went stable
on 2026-08-31, which unblocks the spec-conformance issue that was waiting on
it. The package was never published to PyPI, so the release's `.mcpb` bundle
cannot launch and the official registry publish remains blocked. Two issues
are unsolicited offers. One is a feature request whose design the owner
already scoped in a reply.

The work runs as six workstreams: triage replies, landing the contributor
fixes, the dependency batch, the FastMCP 4 migration, the SearXNG backend,
and a v1.1.0 release with the distribution close-out.

## Goals

- Every open issue and PR reaches a terminal state: merged, closed with a
  reply, or left open with a precise blocker recorded on it.
- The three backtesting bugs (#245, #246, #247) are fixed on `main` with
  regression tests, credited to the contributor who fixed them.
- The server runs on FastMCP 4 and serves the `2026-07-28` protocol
  revision, verified with the same conformance tool that filed #235.
- Research works without an Exa key when a user points it at a self-hosted
  SearXNG instance (#186).
- v1.1.0 ships with release notes, and the distribution steps that only the
  owner can perform are listed with their exact prerequisites.

## Non-goals

- No auth, billing, or hosted scope. The security-offer reply restates the
  local personal-use model; it does not expand it.
- No real Hidden Markov Model for regime detection. PR 244 documents the
  `hmm` misnomer; replacing the estimator is a separate decision.
- No SearXNG-specific research modes (domain presets, multi-step retrieval).
  The first backend PR is provider, settings, prerequisite check, and tests.
- No third-party promotional links in the README (#241).
- No new orchestration machinery. Work runs inline; see "Execution model".

## Findings, verified on 2026-09-05

### Contributor pull requests 242, 243, 244

Author A1-NWS-Dev1 opened three PRs on 2026-08-30, each with a paired bug
issue and each with a follow-up commit pushed after CodeRabbit's review.

- **PR 242, fixes #245.** `backtest_portfolio` aggregates signed drawdowns
  with `max()`, which picks the mildest constituent. The fix is `min()` at
  `maverick/backtesting/service.py:358`, plus three regression tests that
  stub `_run_single_backtest`, plus docs stating the aggregation semantics.
  Sound.
- **PR 243, fixes #246.** `create_strategy_ensemble` built
  `SimpleMovingAverageStrategy` instances for `"rsi"` and `"macd"`, so all
  three members collapsed onto one `"SMA Crossover"` key. The fix adds
  `TemplateStrategy` in `service_support.py`, a `Strategy` wrapper that
  dispatches through the existing signal catalog and validates names
  eagerly. Result keys become template display names, which is what the
  ensemble already keyed on. One nit: the `if not instances` guard is now
  unreachable. Sound.
- **PR 244, fixes #247.** `fit_regimes` set `is_fitted = True` on four
  early-return branches without fitting the scaler or model, so
  `get_regime_probabilities` swallowed `NotFittedError` and fabricated a
  uniform vector while `method` still reported the request. The fix routes
  every fallback through one helper that sets `method = "threshold"`,
  returns one-hot vectors instead of uniform ones, reports
  `detector.method`, clamps the threshold label for `n_regimes < 3`, and
  documents that `hmm` fits a `GaussianMixture`. Two problems:
  - It pushes `service_ml.py` to 505 lines. `tests/structure` enforces a
    500-line cap, so the PR fails CI as submitted. The contributor ran only
    the backtesting and server suites.
  - The two new `hmm` tests fit an unseeded `GaussianMixture`. The detector
    accepts `random_state`; the tests should pass one.
  - Noted, not blocking: fallback overwrites `self.method` permanently, so a
    detector held by `RegimeAwareStrategy` never retries the statistical
    method. This predates the PR (the old code also set `is_fitted`), so it
    becomes a tech-debt line, not a review demand.

Shared facts: all three edit `docs/api/backtesting.md` and
`tests/backtesting/test_service.py`; 243 and 244 both edit `service_ml.py`.
They must land serially with rebases. All three allow maintainer edits.
CodeRabbit's "changes requested" is stale: it reviewed the first commit of
each PR and its hourly quota ran out before the follow-ups. The fork-run
approval policy is `first_time_contributors`; after the author's first PR
merges, later runs start without approval.

### Dependency updates

| PR | Change | Reach | Verdict |
| --- | --- | --- | --- |
| 248 | langchain-anthropic 1.4.6 to 1.7.0 | `[research]` extra | merge |
| 250 | redis 8.0.1 to 8.1.0 | core | merge |
| 251 | greenlet 3.5.1 to 3.5.5 | core and dev | merge |
| 252 | uvicorn 0.48.0 to 0.52.4 | core | merge |
| 253 | nltk 3.10.0 to 3.10.3 (CVE-2026-12252, CVE-2026-12841) | dev only, via `safety` | merge |
| 249 | mcp 1.28.1 to 2.1.1, and fastmcp 3.3.1 to 4.0.0 in the lock | core | hold, supersede |

All six pass lint, type check, unit tests, and the docs catalog on CI. PR
249 passes too, which means the code already runs on FastMCP 4.0.0, but a
major framework upgrade must land as a deliberate change with a raised
version floor, not as a lock-only bump.

### FastMCP 4 and issue 235

FastMCP 4.0.0 shipped on 2026-08-31 and 4.0.3 on 2026-09-05. It builds on
the MCP Python SDK v2 and serves the `2026-07-28` protocol revision,
including `server/discover`, which is exactly what the conformance report
on #235 found missing. The owner's 2026-08-26 reply on #235 named the
FastMCP 4 stable release as the trigger.

The server's exposure to the upgrade guide is small:

- Eight modules import `FastMCP`; the constructor is `FastMCP(name=...)`;
  registration is `mcp.tool(name=..., annotations={...})`,
  `mcp.prompt(name=...)`, and `mcp.resource(uri)`; the entry point calls
  `mcp.run(transport="stdio")` or `mcp.run(transport="http", host, port)`.
- Tests use the in-memory `Client(mcp)`.
- Nothing uses `Context`, sampling, roots, elicitation, tasks, proxies,
  mounts, `serializer=`, `exclude_args=`, or `McpError` construction.
- Annotations are plain dicts with camelCase keys (`readOnlyHint`,
  `destructiveHint`, `idempotentHint`, `openWorldHint`) in six modules, and
  `tests/portfolio/test_tools.py` reads `tool.annotations.readOnlyHint`
  and `.destructiveHint`. SDK v2 fields are snake_case; FastMCP 4 bridges
  camelCase reads with a `FastMCPDeprecationWarning` and offers
  `FASTMCP_MCP_CAMELCASE_COMPAT=false` to turn the bridge off.
- FastMCP 4 no longer depends on `httpx`. The platform HTTP seam imports
  `httpx`, and `pyproject.toml` declares `httpx>=0.28.1` directly, so that
  pin is now load-bearing.
- `pyproject.toml` declares `mcp>=1.28.1` directly, but no module imports
  `mcp`. The pin exists only because dependabot needed a target.
- `fastmcp.Client` now negotiates the newest era by default, so the
  in-memory tests exercise the sessionless `2026-07-28` path.

### Distribution

- `maverick-mcp-server` returns 404 on PyPI. PyPI trusted publishing was
  never configured; the 2026-07-20 publish run failed on it.
- The v1.0.0 release carries `maverick-mcp.mcpb` (802 bytes, 4 downloads).
  Its manifest launches `uvx --from maverick-mcp-server==1.0.0`, so it
  cannot work until PyPI exists.
- `publish.yml` runs `build`, then `publish-pypi` (environment `pypi`),
  then `publish-mcp-registry`; `publish-ghcr` depends only on `build`. A
  new tag therefore fails PyPI and skips the registry again until trusted
  publishing is configured, while GHCR publishes.
- Docker MCP Catalog PR docker/mcp-registry#4490 has been open and untouched
  since 2026-07-20.
- `server.json` and `pyproject.toml` both say 1.0.0.
- The root `SECURITY.md` lists 0.1.x as the supported version and describes
  legacy features (secure headers, audit logging) that v1.0.0 deleted.

### Non-code issues

- **#254** (2026-09-04): an unsolicited offer of a paid-style security
  audit with a free scoping call.
- **#241** (2026-08-28): a marketing pitch asking for a README pointer to a
  forecasting leaderboard.
- **#186** (2026-05-26, replied 2026-08-15): SearXNG as a research search
  backend. The reply scoped a first PR: a `SearXNGProvider` subclass, a
  backend-selection setting, a relaxed prerequisite check, and stubbed-HTTP
  tests. The requester offered to contribute; nothing has arrived.

## Design

### WS0: triage

1. Approve the pending CI runs on PRs 242, 243, and 244 (GitHub's "Approve
   and run", or `POST /repos/{owner}/{repo}/actions/runs/{id}/approve`).
2. Comment on PR 249: on hold because the lock bump silently upgrades
   fastmcp to 4.0.0; the migration lands as its own PR (WS3); this PR closes
   when that merges.
3. Reply on #241 and close it: thanks, README links are limited to install
   and configuration, anyone may build on the tools.
4. Reply on #254 and close it: the server is a local, personal-use tool with
   no network auth surface by design; vulnerability reports are welcome
   through GitHub Security Advisories per `SECURITY.md`; declining the
   scoping call.
5. Refresh the root `SECURITY.md`: supported versions become 1.x supported
   and below 1.0 unsupported; replace the legacy feature list with a pointer
   to `docs/SECURITY.md`; keep the reporting process. Small docs commit on
   `main`.

### WS1: land the contributor fixes

Order: 242, then 243, then 244. For each PR:

1. Wait for the approved CI run.
2. Review the diff against the findings above. Specific checks:
   - 242: confirm every producer of `BacktestMetrics.max_drawdown` uses the
     0-or-negative convention, so `min()` is the worst.
   - 243: confirm `TemplateStrategy` names match `STRATEGY_TEMPLATES` and
     that no other caller depended on the old collapsed key.
   - 244: confirm the fallback helper is the only path that sets
     `is_fitted` without a fit, and that the clamp cannot mask a real bug
     for `n_regimes >= 3`.
3. Run the full gate locally on the PR head: `make check` (ruff check, ruff
   format, import contracts, `ty`) and `uv run pytest` (which includes
   `tests/structure`).
4. Squash-merge with `gh pr merge --squash`, which keeps the contributor as
   the commit author. Thank the author on the PR.

Maintainer fixups, pushed to the contributor's branch with a comment first:

- 243 and 244 need a rebase onto `main` after the previous merge
  (`docs/api/backtesting.md`, `tests/backtesting/test_service.py`, and for
  244 also `service_ml.py`).
- 244: shorten the seven-line comment above `method=detector.method` in
  `service_ml.py` to one line, which lands the file at 499 lines under the
  500-line cap, and pass `random_state=0` to the detector in the two
  `GaussianMixture` tests.

Add one tech-debt line: the regime detector's fallback overwrites the
requested method; store the request separately and allow a refit.

Exit: three PRs merged, #245, #246, #247 closed by the `Fixes` keywords,
`main` CI green.

### WS2: dependency batch

1. On a scratch branch, run `uv lock --upgrade-package` for the five
   packages (langchain-anthropic, redis, greenlet, uvicorn, nltk) and
   confirm the lock still holds fastmcp 3.3.1 and mcp 1.28.1.
2. `uv sync --frozen --extra dev --extra backtesting --extra research`, then
   `make check` and `uv run pytest`.
3. Merge PRs 248, 250, 251, 252, 253 on GitHub, one at a time, using
   `@dependabot rebase` when a lock conflict appears. Discard the scratch
   branch.
4. Dependabot's five-PR cap then frees slots; any new bump it opens for
   `fastmcp` folds into WS3, and anything else waits for the next sweep.

Exit: five PRs merged, `main` CI green.

### WS3: FastMCP 4 migration

Branch `chore/fastmcp-4`, in a worktree. One PR.

1. `pyproject.toml`: raise `fastmcp>=4.0.3`; remove the direct `mcp`
   dependency; annotate the `httpx` pin as required by the platform HTTP
   seam now that FastMCP no longer brings it. Run `uv lock`; confirm
   pydantic 2.12 or later and starlette 1.0.1 or later resolve.
2. Replace the camelCase annotation dict keys with snake_case in
   `market_data/tools.py`, `screening/tools.py`, `portfolio/tools.py`,
   `technical/tools.py`, `backtesting/tools_support.py`, and
   `research/tools.py`. Update the reads in `tests/portfolio/test_tools.py`.
3. Turn the compatibility bridge off for the whole suite: an autouse
   session fixture in `tests/conftest.py` sets
   `fastmcp.settings.mcp_camelcase_compat = False`, so any remaining
   camelCase read fails as an `AttributeError`.
4. Walk the upgrade guide's checklist and record the result of each item in
   the PR description. Expected: no removed API in use.
5. Verify: `make check`, `uv run pytest`, then a transport smoke test:
   stdio via `uv run python -m maverick.server --transport stdio` with an
   `initialize` request, and HTTP via `make dev` with `POST /mcp` and a
   check that `/mcp/` still answers 307 (or update the runbook if not).
6. Conformance: run `npx @hasmcp/mcp-spec-test@0.1.5 -c "uv run python -m
   maverick.server --transport stdio" --spec-version 2026-07-28`, and again
   with `2025-11-25`. Fix anything the server owns. Attach both reports to
   #235. Close #235 if conformant; otherwise leave it open with the exact
   residual list and the upstream owner of each item.
7. Docs: update any FastMCP version or protocol mention in `README.md`,
   `docs/runbooks/mcp-clients.md`, and `docs/RELIABILITY.md`; add release
   notes; `make docs-check`.
8. Merge. Close PR 249 (dependabot closes it itself once `main` carries the
   newer version; close by hand if it does not).

Exit: floor at 4.0.3, gate green with the bridge off, conformance reports
attached, #235 and #249 closed.

### WS4: SearXNG research backend

Branch `feat/searxng-backend`, in a worktree, started after WS3 merges so it
is written against FastMCP 4. One PR. Test-first.

Provider, `maverick/research/providers/searxng.py`:

- `SearXNGProvider(WebSearchProvider)` takes `base_url` and `settings`. It
  passes an empty API key to the base class, which has no other use for it.
- `search()` keeps the `ExaSearchProvider.search` signature,
  `(query, num_results=10, timeout_budget=None)`. The agents call it with
  `num_results` and `timeout_budget` and nothing else; `timeout_budget`
  feeds the base class's `_calculate_timeout` exactly as Exa does.
- Requests go through the platform seam: `create_client()` plus
  `request_resilient("searxng", client, "GET", f"{base_url}/search",
  params=...)`, so the shared rate limiter, circuit breaker, and retry
  apply. Parameters: `q`, `format=json`, `categories=general`,
  `language=en`, and `time_range` in `day`, `week`, `month`, or `year`
  mapped from the provider timeframe; unmapped timeframes omit it.
- Results normalize to the same dict shape Exa produces (`url`, `title`,
  `content`, `raw_content`, `published_date`, `score`,
  `financial_relevance`, `provider="searxng"`, `author`, `domain`,
  `is_authoritative`). SearXNG returns snippets, so `raw_content` equals
  `content`. Missing fields default the way the Exa tests already require.
- The domain, authority, and financial-relevance helpers move from
  `ExaSearchProvider` methods to module-level functions in `providers/base.py`;
  the Exa methods become one-line wrappers so its tests stay untouched.
- A 403 or a non-JSON body produces a `WebSearchError` that says the
  instance must enable the `json` format in its `search.formats` setting.
  Other transport errors wrap into `WebSearchError`, and the base class's
  failure and success counters are recorded as Exa does.
- `get_content()` stays unimplemented; no caller uses it.
- `providers/__init__.py` must still import without `exa_py` installed.

Settings, `maverick/research/config.py`:

- `search_backend: Literal["exa", "searxng"]` from `RESEARCH_SEARCH_BACKEND`,
  default `exa`.
- `searxng_base_url: str | None` from `SEARXNG_BASE_URL`, trailing slash
  stripped, `http` or `https` required.

Service and prerequisite check:

- `_build_default_agent` in `research/service.py` constructs the provider
  named by `search_backend`.
- `configuration_problem` in `research/service_support.py` takes the
  backend and whether it is configured, and names the right variable:
  `EXA_API_KEY` for Exa, `SEARXNG_BASE_URL` for SearXNG. The existing Exa
  payload keys stay as they are; the SearXNG payload uses
  `searxng_base_url` in place of `exa_api_key`.

Tests, no live network:

- `tests/research/test_providers.py`: normalization, time-range mapping,
  the JSON-disabled error, error wrapping, and breaker interplay, using an
  `httpx.MockTransport` passed through `create_client(transport=...)`.
- `tests/research/test_config.py`: both settings from the environment,
  including the trailing slash and the scheme check.
- `tests/research/test_service.py`: factory selection per backend, and the
  prerequisite message for each backend.

Docs: `docs/features/deep-research.md` (backends section, the SearXNG
`json` format note), `.env.example` (two variables), the README research
configuration table. Reply on #186 with the shipped configuration, credit
the requester, and close it.

Exit: with `RESEARCH_SEARCH_BACKEND=searxng`, `SEARXNG_BASE_URL`, and the
BYOK LLM variables set, the research tools register and run against the
stubbed instance; with neither backend configured, the not-configured error
names the active backend's variable.

### WS5: v1.1.0 release and distribution close-out

Prep, in the repository:

1. Bump 1.0.0 to 1.1.0 in `pyproject.toml` and in `server.json` (top-level
   version, both package entries, and the OCI tag). The `.mcpb` builder
   reads the version from `pyproject.toml`. Generalize the `v1.0.0`
   examples in `docs/runbooks/releasing.md` to a placeholder tag. Grep
   `docs/generated/registry/` for the old version.
2. Write release notes: the three bug fixes with credit, the FastMCP 4
   migration and `2026-07-28` support, the SearXNG backend with credit, the
   dependency refresh, and three behavior notes for users: ensemble result
   keys are now template display names, `analyze_market_regimes` may report
   `method: "threshold"` when it falls back, and portfolio `max_drawdown`
   now reports the worst constituent.
3. Update the README's "not yet published to PyPI" note only once PyPI is
   live.

Owner-gated, in order, each needing a separate go-ahead:

4. Configure PyPI trusted publishing for `publish.yml` with environment
   `pypi` (the steps are in `docs/runbooks/releasing.md`).
5. Tag `v1.1.0`. `publish.yml` then builds, publishes to PyPI, publishes to
   the MCP Registry through OIDC, and pushes the GHCR image. Verify with
   `uvx maverick-mcp-server==1.1.0 --help` in a clean environment.
6. `make bundle`, validate with the mcpb CLI, upload the bundle to the
   v1.1.0 release.
7. Update docker/mcp-registry#4490 to 1.1.0 and ask for review. File the
   Smithery, Glama, PulseMCP, and mcp.so submissions from the drafts.

If the owner defers PyPI, tagging still publishes GHCR and creates the
release; the notes then state that PyPI and the bundle are pending.

### WS6: memory and documentation

- Vault: dated entries on `projects/maverick-mcp.md` for each merged
  workstream; `TODO.md` closes the #235 loop and adds the PyPI
  trusted-publishing loop while it stays open; decision records
  `decisions/2026-09-05-fastmcp-4-migration.md` and
  `decisions/2026-09-05-searxng-research-backend.md`; people pages for the
  two contributors; one changelog line per write.
- Repository: `docs/INDEX.md` and `docs/CATALOG.md` rows for this design
  and the execution plan; the tech-debt line from WS1; the execution plan
  moves from `active` to `completed` when WS5 closes.

## Sequencing

1. WS0 runs first and takes minutes.
2. WS1 and WS2 run in parallel; the contributor PRs and the dependabot PRs
   touch disjoint files.
3. WS3 starts after WS2 merges, to avoid lock-file conflicts.
4. WS4 starts after WS3 merges.
5. WS5 prep starts after WS4 merges; its owner-gated steps wait for the
   owner.
6. WS6 runs continuously, with a final sweep after WS5.

## Execution model

Work runs inline in the controlling session, one workstream at a time,
test-first for WS3 and WS4. Sub-agents, when used at all, are plain
background sub-agents on the model the owner has designated for the session
(for the 2026-09-05 session: Fable 5.1 only, no Opus or Sonnet workers). No
agent teams and no Workflow-tool orchestration. WS3 and WS4 use git
worktrees. Every merge follows a full local gate run and a diff review by
the controller.

## Authorization boundary

Approved by the owner on 2026-09-05: approving CI runs; commenting on,
merging, and closing the PRs and issues named above; pushing rebases and
fixups to the contributor's branches; committing to `main` through the
usual PR flow.

Not yet approved, each asked for separately when reached: the `v1.1.0` tag,
PyPI trusted-publishing configuration, any publish (PyPI, MCP Registry,
GHCR image, bundle upload), edits to the Docker catalog PR, and third-party
registry submissions.

## Decisions

| Decision | Choice | Why |
| --- | --- | --- |
| #254 | Decline the call, welcome advisories | Personal-use scope, no auth surface; advisories already documented |
| #241 | Close, no README link | README links stay to install and configuration |
| Release version | 1.1.0 | SearXNG is a feature; FastMCP 4 is a platform change |
| PR 244 fixups | Maintainer pushes to the fork branch | Keeps momentum; the PR allows maintainer edits; comment first |
| Direct `mcp` pin | Remove | No import uses it; fastmcp owns the SDK version |
| Annotations | snake_case dict keys | Smallest diff; typed objects add imports for no gain |
| camelCase bridge | Off in the test suite | Surfaces every stale read now, before FastMCP removes the shim |
| Ensemble result keys | Keep display names | Inherent to `StrategyEnsemble`; changing it is a separate API decision |
| SearXNG helpers | Lift to module functions in `providers/base.py` | Two providers share scoring without duplicating it |
| Spec location | `docs/design-docs/` | The catalog marks `docs/superpowers/` historical |

## Risks

- **FastMCP 4 surprises at runtime.** CI already passes on 4.0.0 through PR
  249, and the exposure list above is short. Mitigation: the transport smoke
  test and the conformance run. Fallback: pin `fastmcp>=4.0.3,<5`.
- **Residual conformance failures owned upstream.** #235 stays open with a
  precise list rather than closing on a partial pass.
- **Maintainer pushes to a contributor's branch.** Comment before pushing;
  keep fixups minimal and separate from their commits.
- **Dependabot lock conflicts.** Merge one at a time; `@dependabot rebase`.
- **SearXNG instance variance.** Instances often ship with the `json` format
  disabled and vary in result fields. Mitigation: the explicit 403 message,
  defensive normalization, and tests for missing fields.
- **No headroom in `service_ml.py`.** After WS1 the file sits at 490 to 498
  lines under a 500-line cap. The existing tech-debt line already calls for
  a split before the next addition; WS4 does not touch it.

## Exit criteria

| Workstream | Done when |
| --- | --- |
| WS0 | CI approved on 242, 243, 244; 241 and 254 closed with replies; 249 has the hold comment; `SECURITY.md` refreshed |
| WS1 | 242, 243, 244 merged; #245, #246, #247 closed; `main` green |
| WS2 | 248, 250, 251, 252, 253 merged; `main` green |
| WS3 | `fastmcp>=4.0.3`; gate green with the bridge off; conformance reports on #235; #235 and #249 closed |
| WS4 | Provider, settings, prerequisite check, tests, docs merged; #186 closed |
| WS5 | 1.1.0 in the repo with release notes; owner-gated steps listed with their prerequisites, executed as authorized |
| WS6 | Vault and repository docs reflect every merge; execution plan filed |
