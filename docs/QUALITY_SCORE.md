# Quality score

Assessment after the October 4, 2026 correctness implementation and independent
review. Grades are judgments within the reviewed scope, not coverage percentages
or a guarantee of financial accuracy. A means the accepted offline contracts have
no remaining actionable finding; B means a product, provider, or delivery gate
remains. Earlier C/D grades documented reproduced defects before these fixes.

The [completed plan](exec-plans/completed/2026-10-04-correctness-and-reliability.md)
records commits and verification. The original [review](design-docs/2026-10-04-project-review.md)
remains historical evidence of the reproduced failures.

| Area | Grade | Current evidence and remaining limit |
| --- | --- | --- |
| `maverick/platform/` | A | Redis expiry, status-based breaker failures, and memory SQLite lifetime/cancellation have regressions; all 14 import contracts hold. |
| `maverick/market_data/` | B | SQLite/PostgreSQL upserts, snapshot generations, partial-response preservation, and additive migration verified. Provider accuracy, 24-hour refresh latency, and leading-gap ambiguity remain limits. |
| `maverick/technical/` | A | Golden indicator tests and requested-window observed levels pass; synthetic percentage levels removed. |
| `maverick/screening/` | B | Layering and rubric behavior tested; default-universe choice remains deferred. |
| `maverick/portfolio/` | A | Concurrent writes, journal precision/validation, ATR units, correlation diagonals, and watchlist discovery verified. Existing documented storage precision limits still apply. |
| `maverick/backtesting/` | B | Causal features/weights, 252-session metrics, nullable profit factors, durations, and bounded MCP responses verified. Strategy effectiveness and live execution behavior are not evaluated. |
| `maverick/research/` | B | Citations/routing/evidence errors and actual SDK request serialization tested offline. Live provider availability and answer quality are unverified. |
| `maverick/server/` | A | Assembly exposes 38 core or 53 full tools; real core-only installed-wheel smoke passes outside the checkout. |
| Packaging and delivery | B | Direct wheel and sdist exclusions, extracted-source checks, and container persistence verified. CI lanes added; PyPI ownership and registry command/image pairing remain separate release gates. |
