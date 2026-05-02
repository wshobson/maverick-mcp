# Maverick MCP Roadmap v2 — Stubs (post-v1, post-Phase-3.1/3.2)

> **Status:** Stub. Each section is a paragraph, not a plan. The goal
> is to start the next planning session warm rather than cold.

## Context

Roadmap v1 (`docs/superpowers/plans/2026-03-27-maverick-roadmap-v1.md`)
shipped through April 2026: service-layer infrastructure (`event_bus`,
`registry`, `scheduler`) plus all five domains
(`signals`, `screening`, `journal`, `watchlist`, `risk`).

The post-v1 forward plan (this directory's session at
`/Users/wshobson/.claude/plans/read-in-any-roadmaps-eventual-flamingo.md`)
delivered Phase 1 (CI bootstrap, doc accuracy, type-bug fixes), Phase 2.1/2.2/2.4
(dead-code purge, type-safety beachhead), and Phase 3.1/3.2
(signal notifier scaffold + `backtest_signal` MCP tool with parity
testing). Phase 2.3 (refactors of `deep_research.py` and
`llm_optimization.py`) was deliberately deferred for its own
characterization-test-driven PR series.

This doc captures what's natural to pick up *after* that work, so the
next plan doesn't start from a blank page.

## Candidate v2 features

### 1. Options analytics

A new `services/options/` domain that ingests an options chain
(strike, expiry, bid/ask, OI, IV, Greeks where the provider supplies
them) for a ticker and exposes MCP tools for chain queries, ATM
straddle pricing, simple Greeks (delta-neutral hedge ratios, dollar
delta), and IV rank vs trailing window. Reuses the v1 service-layer
pattern (DB models + service + thin router) and the same
`EnhancedStockDataProvider` for the underlying. Tiingo's free tier
does not include options; first-class integration likely requires
adding `polygon-api-client` or a similar provider behind a feature
flag.

### 2. Sector rotation

A read-only analytics service over the existing market data that
ranks sector ETFs (XLK, XLF, XLE, …) by trailing return and relative
strength vs SPY across 1m/3m/6m windows, surfacing leadership
rotations as MCP tools and as a `sectors://leadership` resource.
Stateless, no new domain models — basically a pandas computation on
top of the existing `get_stock_data` path. Cheap to ship and a
natural input for the v1 `RegimeDetector` to consume.

### 3. News-driven signal triggers

Wire the existing Exa research path (`agents/optimized_research.py`)
to the v1 signals event bus: a new `services/signals/news_trigger.py`
that runs scheduled research queries (using the `MaverickScheduler`)
for tickers on the watchlist, classifies the result into
`bullish` / `bearish` / `neutral` using the existing model-selection
infrastructure, and publishes a `signal.triggered` event with
`indicator: "news"` when sentiment crosses a threshold. Reuses the
Phase 3.1 notifier path so news-driven fires reach the same MCP
resource and webhook surfaces as price-driven ones.

## Carry-forwards from this session

These were called out as deferred in PR #161 / #162 and remain on
the table:

- **Phase 2.3 refactors** — `deep_research.py` (3,681 LOC) and
  `llm_optimization.py` (1,675 LOC). Plan: extract two seams from
  each, not a full rewrite. Add characterization tests against the
  current public functions before moving code.
- **Strict-zone burndown completion** — 8 ty diagnostics remain in
  `services/` + `domain/` after this session. Four are SQLAlchemy
  `Column[T]` friction that would benefit from a coordinated
  migration to `Mapped[T]` annotations on the model classes. Once
  at 0, flip `continue-on-error: true` to `false` in `ci.yml` to
  gate PRs.
- **Integration test for end-to-end signal delivery** — publish on
  the real `EventBus` singleton with Postgres-backed `SignalService`
  and assert the notifiers receive. Needs Postgres/Redis service
  containers in CI.
- **Integration test for `backtest_signal`** — create a real Signal
  via `SignalService` and run the full tool end-to-end with a
  vectorbt simulation. Mark `integration` and add to the deferred
  integration workflow.

## Out of scope (still)

- Re-introducing authentication / multi-user mode — explicitly
  removed per `CLAUDE.md`.
- Public hosting / SaaS — this remains a personal-use MCP server.
- Migrating off `pandas-ta 0.3.14b0` unless a specific bug forces
  it.
- Refactoring `data/models.py`, `api/server.py`,
  `tool_registry.py`, `backtesting/vectorbt_engine.py`, or
  `providers/stock_data.py` — high-churn, deferred indefinitely.
