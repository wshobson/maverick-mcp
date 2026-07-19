# Phase 4: Portfolio Domain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `maverick/portfolio/` — position ledger with strict Decimal cost-basis math, portfolio metrics with live prices, and the three portfolio-aware analyses (correlation, comparison, risk sizing) — as the fourth domain on the seam.

**Architecture:** Phase 4 of the modernization. Follows the screening template exactly (layer contracts, injectable everything, honest fakes, hardened write path). The 2026-07-19 portfolio recon is the map. The behavioral contract is `docs/features/portfolio.md` plus the legacy `maverick_mcp/domain/portfolio.py` dataclasses, which already embody the correct average-cost rules with proper quantization and 33 passing behavior tests; the legacy router's inline recomputation (which diverges from them) is NOT the reference.

**Tech Stack:** Python 3.12, pydantic, SQLAlchemy 2, Decimal end-to-end for money/shares (float only at the tools JSON boundary via str-mediated conversion), maverick.platform, maverick.market_data, maverick.technical, pytest with TDD.

## Global Constraints

- Same gates as Phase 3 (add `tests/portfolio/` to the pytest gate list). No new dependencies. Files under 500 lines. Env reads only in config.py. No network in tests. Commit per task with checkbox flips; stage explicitly. Plain prose in docs.
- Decimal discipline: `Decimal(str(x))` at every float ingress; quantization `0.0001` ROUND_HALF_UP for cost basis, `0.01` for money values; no float arithmetic on stored shares/costs anywhere in data/ledger/service.
- Write-path template: fetch-failure accounting, no destructive writes on fully-degraded input (inherited from the Phase 3 fixes).

## Decision log

- 2026-07-19: The ledger math ports from `maverick_mcp/domain/portfolio.py` (correct, quantized, tested), not from the router's inline recomputation. Its 33-test behavior suite ports with it.
- 2026-07-19: New tables `pf_portfolios` / `pf_positions` (same shapes as the legacy `mcp_portfolios` / `mcp_portfolio_positions`, including the one-row-per-ticker average-cost design). Legacy rows are not migrated automatically; a cutover note records the option of a one-shot migration script at cutover.
- 2026-07-19: Every tool exposes `portfolio_name` with the "My Portfolio" default (the legacy surface was inconsistent: only 2 of 6 unprefixed wrappers had it).
- 2026-07-19: One canonical `portfolio_*` tool name per behavior; the legacy dual registrations collapse at cutover as already tracked.
- 2026-07-19: `get_user_portfolio_summary` (a static capability descriptor misnamed as portfolio) and `tools/portfolio_manager.py` (dead float-math duplicate) are not ported; the latter is deleted this phase.
- 2026-07-19: risk_dashboard and watchlist stay out of this domain. risk_dashboard's direct table queries are a tracked cutover bridge item: it must consume `PortfolioService` reads when it ports.
- 2026-07-19: Analysis outputs (correlation, comparison, risk sizing) are advisory floats, as in legacy; stored ledger values remain Decimal.
- 2026-07-19: analysis.py added as its own layer between service and data/ledger when service exceeded the size cap; contract extended additively.
- 2026-07-19: naive purchase dates normalize to UTC at the service boundary, matching the persistence convention.

## Layer contract (Task 1 encodes)

```
maverick.portfolio: tools -> service -> {data, ledger} -> config -> types
ledger.py = pure Decimal position math (the ported domain dataclass logic).
Cross-domain: service may import maverick.market_data and maverick.technical. Platform-independence and technical-independence forbidden lists gain maverick.portfolio.
```

---

### Task 1: skeleton and contracts
Mirror the screening skeleton task: seven docstring-only files (`__init__`, types, config, data, ledger, service, tools), the layers contract with `data | ledger` siblings, extend both forbidden contracts with `maverick.portfolio`, `tests/portfolio/test_layers.py` (same pattern), fail-then-pass proof, full gate. Commit `feat(portfolio): add domain skeleton with enforced layer contracts`.

### Task 2: types
Pydantic models (Decimal fields where money/shares): `PositionPayload(ticker, shares: Decimal, average_cost_basis: Decimal, total_cost: Decimal, purchase_date: str, notes: str | None)`; `PositionWithPrice(PositionPayload + current_price: float | None, current_value: float | None, unrealized_pnl: float | None, unrealized_pnl_percent: float | None)`; `PortfolioMetrics(total_invested: Decimal, total_value: float | None, total_pnl: float | None, total_pnl_percent: float | None, position_count: int)`; `PortfolioSnapshot(user_id, name, positions: list[PositionWithPrice], metrics: PortfolioMetrics, as_of: str)`; `RemoveResult(ticker, shares_removed: Decimal, position_fully_closed: bool)`; `ComparisonResult`, `CorrelationResult`, `RiskAnalysis` (advisory-float shapes matching the legacy payload fields the recon documented: comparison has per-ticker metrics + performance/trend ranks + best_performer/strongest_trend + optional portfolio_context; correlation has matrix, high-correlation pairs > 0.7, hedges < -0.3, average correlation, diversification score; risk has position sizing, stop loss, entry scaling, targets, optional existing_position block). TDD: round-trips, Decimal preservation through model_validate, a `model_dump(mode="json")` check documenting the JSON boundary. Commit `feat(portfolio): add payload types`.

### Task 3: config
`PortfolioSettings`: `default_user_id: str = "default"`, `default_portfolio_name: str = "My Portfolio"` (PF_DEFAULT_PORTFOLIO_NAME), `correlation_default_days: int = 252` (PF_CORRELATION_DAYS), `correlation_min_rows: int = 30`, `compare_default_days: int = 90`, `risk_account_size: int = 100000` (PF_RISK_ACCOUNT_SIZE), `price_lookback_days: int = 7`, `history_pad_calendar_days: int = 400`, `max_shares: int = 10**9`, `max_price: int = 10**6`. Singleton/reset, platform helpers, TDD mirroring prior config tasks. Commit `feat(portfolio): add domain settings`.

### Task 4: ledger
`maverick/portfolio/ledger.py`: port the legacy domain dataclass logic onto the pydantic types — pure functions: `add_shares(position: PositionPayload | None, ticker, shares: Decimal, price: Decimal, purchase_date, notes) -> PositionPayload` (average-cost formula, 0.0001 ROUND_HALF_UP quantization, earliest-date-wins, total_cost from STORED total plus new lot); `remove_shares(position, shares: Decimal | None) -> tuple[PositionPayload | None, RemoveResult]` (full close on None or >= held; partial keeps basis, total_cost = shares x basis); `position_value(position, current_price: Decimal) -> tuple[Decimal, Decimal, Decimal]` (value, pnl, pnl_percent; 0.01 quantization; zero-cost-safe); `portfolio_metrics(positions, prices: dict[str, Decimal]) -> PortfolioMetrics`. TDD: port the legacy 33-test behavior matrix (adapt `tests/domain/test_portfolio_entities.py` — averaging, partial/full close, fractional shares, quantization edges, zero-cost safety, validation errors on non-positive inputs). The ported tests are the contract; exact Decimal values asserted. Commit `feat(portfolio): add decimal position ledger`.

### Task 5: persistence
`data.py`: tables `pf_portfolios` (uuid pk, user_id indexed, name, unique (user_id, name)) and `pf_positions` (uuid pk, portfolio_id FK cascade, ticker indexed, shares Numeric(20,8), average_cost_basis Numeric(12,4), total_cost Numeric(20,4), purchase_date tz DateTime, notes, unique (portfolio_id, ticker)); functions `get_or_create_portfolio(session, user_id, name) -> id`, `read_positions(session, portfolio_id) -> list[PositionPayload]`, `upsert_position(session, portfolio_id, position: PositionPayload) -> None` (insert-or-update by ticker), `delete_position(session, portfolio_id, ticker) -> bool`, `clear_positions(session, portfolio_id) -> int`. No math in this layer — it stores what the ledger computed. TDD with tmp SQLite: Decimal round-trip precision (Numeric columns back to exact Decimal via str), constraint behavior (dup ticker upserts, cascade delete), empty states. Commit `feat(portfolio): add persistence layer`.

### Task 6: service and tools
`service.py`: the API from the recon's conclusions — CRUD composing ledger + data inside single `session_scope` transactions; `get_portfolio`/`calculate_metrics` fetching current prices per position via `market_data.get_quote` (Semaphore(4); a failed quote leaves that position's price fields None and counts it — never fails the read); `correlation_analysis` and `compare_tickers` fanning out `market_data.get_price_history` with the calendar pad, auto-fill from holdings when tickers omitted (>= 2 required), correlation guard `>= correlation_min_rows` return rows, comparison using `maverick.technical.indicators.rsi` + a simple documented trend classification; `risk_adjusted_analysis` using `maverick.technical.indicators.atr` (length 20) with the legacy sizing formulas (account size from settings) and the existing-position P&L block computed via the LEDGER (not float mixing). Validation per legacy bounds (ticker alnum <= 10 chars; positive shares/price; settings max bounds) raising ValueError -> tools error payloads.
`tools.py`: the seven `portfolio_*` tools with uniform `portfolio_name` param and str-mediated Decimal ingress, `clear` requiring `confirm=True`; annotations: reads readOnlyHint True; add/remove/clear readOnlyHint False, destructiveHint True for clear and for remove-without-shares (full close), idempotentHint False for add; plus the `portfolio://my-holdings` resource registered in `register(mcp)`. Stub-service tests for shapes/errors/unconfigured path, ledger-integration service tests with a tmp engine + stub market data (deterministic quotes/frames incl. a failing symbol), in-memory Client round-trip. Two commits allowed: `feat(portfolio): add domain service` then `feat(portfolio): add MCP tool layer and resource`.

### Task 7: cleanup
Delete `maverick_mcp/tools/portfolio_manager.py` (grep-verify zero importers first; STOP if any). Full suite. Commit `refactor: delete dead portfolio_manager duplicate`.

### Task 8: close-out
Exports with `__all__` + smoke; QUALITY_SCORE row (`maverick/portfolio/` A: "Decimal ledger ported from the tested domain layer; analyses on the seam."); tech-debt rows: "legacy mcp_portfolios/mcp_portfolio_positions retire at cutover; optional one-shot migration script decision at cutover | legacy tree | cutover", "risk_dashboard must consume PortfolioService reads when ported | maverick_mcp/api/routers/risk_dashboard.py | cutover"; decision-log addenda for any execution deviations; full verification incl. make test/docs-check; move plan to completed/ with CATALOG/INDEX updates; commit `docs: complete phase 4 (portfolio domain)`; push; foreground CI watch.
