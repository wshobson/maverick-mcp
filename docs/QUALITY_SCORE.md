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
