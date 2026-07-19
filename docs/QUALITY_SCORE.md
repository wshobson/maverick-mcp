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
| `application/`, `api/dependencies/`, `domain/` | removed | Zombie layer deleted in phase 0; the legacy screening slice (`application/screening/`, `domain/screening/`, `infrastructure/screening/`) deleted in phase 3. |
| `maverick/` (new) | A | Platform seam landed with full test coverage. Domains arrive next. |
| `maverick/market_data/` | A | First domain through the seam. Layer contracts enforced, 130+ tests, injectable fetchers. |
| `maverick/technical/` | A | Pure-Python indicators, golden-tested against pandas-ta at rtol=1e-9. |
| `maverick/screening/` | A | Query and compute domain; rubric scores are exact-tested; fresh installs can self-populate. |
