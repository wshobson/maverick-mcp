# Project review and correctness priorities — 2026-10-04

## Assessment

The domain boundaries, import contracts, typed tool surface, and offline test
coverage are strong. The next investment should be correctness and durable user
data. Passing tests and dependency updates do not establish that financial
outputs are accurate: this review reproduced lost portfolio writes and future
information changing historical backtest signals.

The findings below record the pre-fix review. All sixteen accepted implementation
tasks now have regression coverage and independent review; see the
[completed plan](../exec-plans/completed/2026-10-04-correctness-and-reliability.md)
for commits, verification, and remaining gates. Current behavior lives in the
feature/API/runbook documentation, and unresolved scope remains in the debt tracker.

## Review scope and maintenance completed

Reviewed the source at `a2519768ace5f988dcb77e74a96dd276d41badcb`, with
independent runtime/research, financial/data, and delivery/CI reviews. Checked
GitHub PRs, issues, security alerts, and the two external distribution blockers.
Maintenance landed through `4a9c61c65bfc7bc8d242fbfac3065af820366896`.

| PR | Change | Outcome |
| --- | --- | --- |
| #302 | langchain-core 1.6.3 → 1.6.5 | Merged |
| #303 | FastMCP / fastmcp-slim 4.0.3 → 4.0.10 | Merged |
| #304 | langchain-anthropic 1.7.1 → 1.7.4 | Merged; its core update overlaps #302 |
| #305 | pydantic 2.13.4 → 2.13.5, pydantic-core 2.46.4 → 2.46.5 | Merged |
| #306 | psycopg2-binary 2.9.12 → 2.9.13 | Merged |
| #307 | PyJWT 2.13.0 → 2.15.0 | Merged |
| #308 | urllib3 2.7.0 → 2.8.0 | Merged |
| #273 | Cancelled/stale circuit-breaker recovery | Corrected test fixture, then merged; closes #272 |

PR #273's `clock` fixture patched the shared `time.monotonic`, also freezing
asyncio's deadline clock. Its new `wait_for` test failed the three-second
watchdog. Maintainer commit `03023d7` replaces only `http.time` with a fake
namespace: the 18 HTTP/recovery tests pass, and the corrected head's CI passed
before merge. The production generation/cancellation changes passed independent
review.

All eight original PRs and issue #272 are closed. GitHub reported **zero open
Dependabot alerts** after the merges, down from 16. One advisory had no named
patched release but its affected range ended at PyJWT 2.13.0; the final alert
state, rather than an inferred release claim, is the evidence used here.
The combined dependency lock was verified before merging and is byte-identical
to the merged lock. This was a reviewed queue update, not an unrestricted
upgrade of every transitive package.

## New findings

P1 means possible silent data corruption or misleading historical results.
P2 means incorrect behavior under a supported input or deployment condition.
These findings are separate from the previously documented backlog below.

| ID | Priority | Finding and evidence | Source / next task |
| --- | --- | --- | --- |
| R1 | P1 | Concurrent purchases lose an update. Start with 10 AAPL shares at $100; two synchronized calls add 1 and 2, both succeed, but storage ends at 11 or 12 shares instead of 13. A transaction without locking does not serialize read/modify/write; removal uses the same pattern. | `maverick/portfolio/service.py:175`, `data.py:186`; task 1 |
| R2 | P1 | Ensemble weights learned from the complete input are applied to its whole history. Changing only the final 100 of 300 synthetic bars changed five earlier entry signals and four earlier exit signals in the unchanged 200-bar prefix, using the real SMA/RSI/MACD templates. | `maverick/backtesting/strategies/ml/ensemble.py:205`, `:293`; task 3 |
| R3 | P1 | Feature warm-up uses backward fill. Extending an unchanged 30-bar prefix to 100 bars changes its first-row SMA-50 ratio from 0 to about 1.08585. Prediction-time features therefore contain information from later test bars. | `maverick/backtesting/strategies/ml/feature_engineering.py:386`; task 4 |
| R4 | P1 | Docker quick start runs `--rm` without persistent storage. SQLite defaults to `/app/maverick.db`, so removing that container removes portfolios, watchlists, and the journal. Confirmed in the current and published-tag Dockerfiles; no destructive container test was performed. | `README.md:112`, `Dockerfile:45`, `.env.example:11`; task 5 |
| R5 | P2 | A Redis TTL of zero or an expired-between-reads key falls back to the global TTL. A mock TTL-zero quote was backfilled into memory for 604800 seconds. | `maverick/platform/cache.py:276`; task 9 |
| R6 | P2 | Exhausted HTTP 503 responses return normally from the breaker's callback and count as successes. Three mocked requests at failure threshold one left the breaker closed. | `maverick/platform/http.py`, `request_resilient`; task 10 |
| R7 | P2 | In-memory SQLite uses `NullPool`; each connection has a separate database. Creating a table and opening the next connection loses the schema. Configuration can select this URL under CI. | `maverick/platform/db.py:84`; task 11 |
| R8 | P2 | Research service envelopes discard `report.citations`. A fake-backed graph produced three SEC citations but the public response contained no citations or source URLs. | `maverick/research/service.py:277` and company/sentiment methods; task 12 |
| R9 | P2 | Competitive-analysis focus values do not match the router's exact strings. Requesting it routes to validation while result metadata claims it was included. | `maverick/research/agents/graph.py:293`, `:395`; task 12 |
| R10 | P2 | Source archives include tests but omit required `evals/` and `tools/` sources. Extracted-sdist eval-test collection and docs-catalog tests fail; Makefile targets also reference omitted scripts. | `pyproject.toml:74`, `tests/structure/test_docs_catalog.py:34`; task 13 |
| R11 | P2 | Journal tools accept invalid sides and negative quantities, treat unknown sides as short, and quantize prices to cents. A long trade of 10,000 shares from $0.004 to $0.005 records $100 profit instead of $10. | `maverick/portfolio/service_journal.py:107`, `:151`; task 2 |

The September remediation plan also retained obsolete publisher setup and an
unqualified publish dispatch. Those instructions are corrected to point at the
canonical release runbook in this documentation change.

## Existing debt confirmed and ordered

The [debt tracker](../exec-plans/tech-debt-tracker.md) remains the complete
inventory. The following items should precede feature expansion:

1. **Stored market data can be wrong indefinitely.** Existing price dates are
   never updated, including a current session's first partial daily bar and
   later corporate-action corrections. Parallel insertions also reproduce a
   unique-constraint failure. Fix freshness, adjustment consistency, and
   concurrency together in task 6.
2. **Backtest metrics misstate results.** A deterministic 252-bar, fee-free 10%
   gain annualizes above 14% under the current 365-bar convention. An all-winning
   strategy reports profit factor zero, and trade durations are empty. Task 7
   must define the annualization and JSON representation of undefined ratios.
3. **Risk output does not match its labels.** Sizing ignores the stop distance;
   reported risk is position notional, reward/risk does not match the returned
   target and stop, and confidence is constant. Perfect correlations are
   dropped by a value-based off-diagonal mask. Task 8 fixes these outputs.
4. **Support/resistance mixes synthetic percentages with observed levels**, and
   the requested history does not control the fixed analysis window. Task 14
   should return observed levels or explicitly identify a heuristic fallback.
5. **Research model compatibility remains configuration-dependent.** The LLM
   factory sends a temperature some supported models reject. Test request
   serialization without paid calls in task 12.
6. **Tool usability gaps:** portfolio-backtest trade payloads remain large;
   ensembles silently skip symbols; watchlists have no discovery tool and an
   unknown ID looks like an empty list. These belong in task 15.
7. **Evaluation quality:** seed cost bases need provenance; batch 3 awaits the
   owner's labels. Preserve those labels as human judgments. Repair fixtures,
   then rerun only relevant cases in task 16.

Research can also return a successful narrative after every search provider
fails. The plan proposes a typed insufficient-evidence result; this changes an
existing tested contract and needs explicit product review before that portion
is implemented.

## Delivery and verification

- Combined dependency updates: 1,325 tests passed, Ruff lint/format passed,
  14 import contracts kept, type checking passed, and `uv lock --check` passed.
- Final merged runtime at `4a9c61c`: **1,334 tests passed**. The review
  documentation PR supplies the durable CI record for the final documentation tree.
- Baseline docs catalog passed for 60 tracked documentation/text files.
- Built wheel: 104 files, no forbidden legacy, scratch, database, or secret
  paths. Rebuilding the wheel from the 263-file source archive succeeded.
- A wheel import-safety smoke test bypassed editable-install paths, blocked
  optional imports, assembled 37 core tools/two prompts, and called the
  chart-link tool. This is not a fresh dependency-resolved base installation.
- Financial reproductions used temporary SQLite databases, synthetic bars,
  and mocked providers. No live portfolio, paid model, real market-data,
  PostgreSQL, Redis server, or destructive Docker exercise was used.
- The first restricted local test run hit sandbox loopback/proxy/PATH issues;
  verification used the normal `uv` environment with loopback access. These
  environment failures were not treated as product regressions.

CI currently installs every extra. Add a real built-wheel core-install lane,
plus a source-archive validation lane, rather than relying only on simulated
missing imports. The historical all-A quality grades do not override this audit.

## Distribution blockers and deferred scope

As checked on October 4, 2026:

- [PyPI name transfer #12150](https://github.com/pypi/support/issues/12150)
  remains open, unchanged since September 5. PyPI, official registry, and the
  PyPI-launching bundle remain blocked. Do not point users at that package name
  until ownership and artifact provenance are verified.
- [Docker catalog PR #4490](https://github.com/docker/mcp-registry/pull/4490)
  remains open, unchanged since July 20. Its `uv run` command matches its old
  pinned image build; update command and pin together for the current image.
- The OCI manifest declares stdio while the image default serves HTTP.
- v1.1.0 has a GitHub release and GHCR image; its release has no bundle assets.
  No new package/image/tag/registry release was made during this review.

Defer chart apps, background backtest tasks, macro data, screen history,
dividend expansion, and a default screening universe until correctness and
release gates are closed. A default universe remains a product decision.
