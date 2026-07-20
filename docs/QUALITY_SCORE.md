# Quality score

Grades reflect the 2026-07-20 v1.0.0 cutover. The legacy `maverick_mcp/`
package (and its grades) is deleted; `maverick/` is the whole system now.
Update the grade in the same change that changes the code, and note why.

| Area | Grade | Why |
| --- | --- | --- |
| `maverick/platform/` | A | Platform seam: db, http (circuit breakers), config, telemetry, cache, serde, and (phase 7) `llm.py` -- a single explicit BYOK LLM seam (`LLM_PROVIDER`/`LLM_API_KEY`/`LLM_BASE_URL`/`LLM_MODEL`/`LLM_TEMPERATURE`) collapsing the legacy five-vendor auto-detection surface, with lazy provider imports so the module stays importable with no `langchain*` package installed. Full test coverage. |
| `maverick/market_data/` | A | First domain through the seam. Layer contracts enforced, 130+ tests, injectable fetchers. |
| `maverick/technical/` | A | Full domain: 8 golden-tested indicators, pure analysis rubrics with the legacy outlook bug fixed, a timeout-guarded service, and 4 canonical tools. |
| `maverick/screening/` | A | Query and compute domain; rubric scores are exact-tested; fresh installs can self-populate. |
| `maverick/portfolio/` | A | Decimal ledger ported from the tested domain layer; analyses on the seam; FK policy platform-owned. |
| `maverick/backtesting/` | A | Full domain behind the optional `[backtesting]` extra: 12 read-only tools (phase 7 adds `backtesting_parse_strategy` on the BYOK LLM seam), 12 rule-based templates + 8 ML strategy classes with golden/seeded tests, guarded package exports (base install always importable, extra-only members lazy), zero tools registered when the extra is absent. Store removed (YAGNI, no persisting caller found). |
| `maverick/research/` | A | Full domain behind the optional `[research]` extra: 3 curated tools (collapsed from 9 legacy `research_*`/`agents_*` tools) on a sequential LangGraph workflow, BYOK LLM seam, Exa search with financial scoring, guarded package exports (base install always importable, extra-only members lazy), zero tools registered when the extra is absent. Fixes two live legacy bugs (router timeout-wrapper key mismatch; dual dispatch on a `Command`-returning graph node) rather than porting them. |
| `maverick/server/` | A | Single assembly module (`assembly.py`) wires the shared engine/cache and every domain in one documented, deterministic order; `app.py` is a thin CLI entry point with no business logic; nothing imports it, so it sits cleanly above every domain. |
