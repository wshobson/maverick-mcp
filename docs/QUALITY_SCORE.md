# Quality score

Assessment as of 2026-10-04. These are review judgments, not coverage scores.
A means no material gap found in the reviewed scope; B means a bounded
correctness or verification gap; C means reproduced correctness defects;
D means reproduced silent accounting loss or future-data leakage. Passing
existing tests does not override a reproduced defect. Reassess each area when
its linked regressions and implementation fixes land.

The [project review](design-docs/2026-10-04-project-review.md) records evidence
and limitations; the [implementation plan](exec-plans/active/2026-10-04-correctness-and-reliability.md)
sets the acceptance checks. The former all-A grades described architectural
completion at the v1.0 cutover, not current financial correctness.

| Area | Grade | Current evidence and gap |
| --- | --- | --- |
| `maverick/platform/` | C | Fourteen import contracts hold; cancelled/stale breaker recovery fixed. Redis expiry, HTTP status accounting, and in-memory SQLite remain incorrect. |
| `maverick/market_data/` | C | Provider injection and normalized tool errors work; stored bars miss finalization/adjustment refresh and concurrent inserts can fail. |
| `maverick/technical/` | C | Golden-tested indicators; support/resistance mixes unlabeled percentage scenarios with observations and ignores the requested analysis window. |
| `maverick/screening/` | B | Enforced layering and tested rubric behavior; a fresh database needs an explicitly populated symbol universe. Default-universe choice remains deferred. |
| `maverick/portfolio/` | D | Decimal ledger exists, but concurrent mutations lose holdings. Journal validation/precision, risk sizing, and perfect-correlation treatment need correction. |
| `maverick/backtesting/` | D | Optional registration works; ensemble/feature generation leaks future information and annualization/no-loss metrics mislead. |
| `maverick/research/` | C | Provider abstraction and typed responses exist; citations are dropped, competitive routing is mismatched, and supported-model parameters need validation. |
| `maverick/server/` | B | Assembly/client tests pass; a built-wheel import-safety smoke passed, but a real core-only installation is not covered in CI. |
| Packaging and delivery | C | Wheel build and source-to-wheel rebuild pass; source archive tests lack their supporting files, Docker persistence is missing from quick start, and PyPI ownership remains blocked. |
