# Maverick MCP Architecture — Roadmap v1

**Status:** current as of `4f5dc53 feat: Maverick MCP roadmap v1 — service layer + 5 feature domains` (PR #121, 2026-03-29).

This document describes the service-layer architecture introduced by Roadmap v1. Prior to this change the codebase was organized as a flat set of `@mcp.tool`-decorated functions in a handful of routers. Roadmap v1 adds a thin service layer, an async event bus for cross-domain communication, a job scheduler, and five feature domains built on top.

## Layering

```
┌─────────────────────────────────────────────────────────────────┐
│  FastMCP app (maverick_mcp/api/server.py)                       │
│  └── tool_registry.py  ← wires register_*_tools(mcp) per router │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Routers  (maverick_mcp/api/routers/*.py)                       │
│  Thin MCP adapters. No business logic. Translate MCP tool       │
│  calls → service-layer calls. Format responses as JSON dicts.   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Service layer  (per-domain Service classes, e.g.               │
│  JournalService, WatchlistService, ScreeningPipelineService)    │
│  All business logic lives here. Services publish events to the  │
│  event bus and may subscribe to events from other domains.      │
└─────────────────────────────────────────────────────────────────┘
                 │                              │
                 ▼                              ▼
┌──────────────────────────────┐   ┌──────────────────────────────┐
│  Event bus (async, in-proc)  │   │  Scheduler (apscheduler,     │
│  topic-based pub/sub for     │   │  AsyncIOScheduler on ASGI    │
│  cross-domain communication  │   │  loop)                       │
└──────────────────────────────┘   └──────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  Data layer — SQLAlchemy models, Redis cache, vector store      │
│  (`maverick_mcp/data/`, `maverick_mcp/memory/`)                 │
└─────────────────────────────────────────────────────────────────┘
```

## Event bus

Async, in-process pub/sub. Handlers are registered by **topic** (a dotted string like `"signals.triggered"` or `"screening.changed"`) and receive `(topic, data)`. The bus is the only sanctioned way for one domain to react to another — domains do **not** import each other's services directly.

Keep this contract strict:

- `async def handler(topic: str, data: dict) -> None` — no other signature.
- Published payloads are plain dicts, JSON-serializable, and versioned by topic.
- Handlers must be idempotent; the bus does not guarantee exactly-once delivery.

## Scheduler

`apscheduler`'s `AsyncIOScheduler`, started against the running ASGI event loop (not a transient init loop). Fix notes from the PR:

- Use `asyncio.get_running_loop()` to attach the scheduler; **never** `get_event_loop()`.
- Use `loop.create_task(...)`, not `asyncio.ensure_future(..., loop=...)` (deprecated).
- Job listings serialize `datetime` fields to ISO strings before returning over MCP.
- Event handlers bound via the scheduler still go through the event bus — they are scheduled emitters, not direct callers into other services.

## The five feature domains

Each domain follows the same pattern: SQLAlchemy models → `Service` class → `register_*_tools(mcp)` function → thin router file in `maverick_mcp/api/routers/`.

### 1. Signal Engine (`signals.py`)

CRUD + dispatch for user-defined alerts plus a market-regime detector. Emits `signals.triggered` events carrying the triggering indicator, symbol, and `price_at_trigger` (the actual close at evaluation time — **not** the indicator value itself — fixed in the PR).

### 2. Screening Pipeline (`screening_pipeline.py`)

Runs screeners on a schedule, diffs results against the previous run, and records symbol entries/exits as `ScreeningChange` rows. `get_screening_changes` exposes the diff history; subscribers (e.g. watchlist brief, risk dashboard) can react to changes.

### 3. Trade Journal (`journal.py`)

Models: `JournalEntry`, `StrategyPerformance`. `JournalService` owns `add_trade` / `close_trade` / `list_trades` / `get_trade`. `StrategyTracker` recomputes `expectancy` and `profit_factor` on every close. Six MCP tools wire this via `register_journal_tools`: `journal_add_trade`, `journal_close_trade`, `journal_list_trades`, `journal_trade_review`, `get_strategy_performance`, `get_strategy_comparison`. Covered by 21 unit tests.

### 4. Watchlist Intelligence (`watchlist.py`)

Models: `Watchlist`, `CatalystEvent`. `WatchlistService` + `CatalystTracker` expose watchlist CRUD and upcoming/past catalyst queries. The scheduler drives pre-market catalyst briefings by emitting into the event bus.

### 5. Risk Dashboard (`risk_dashboard.py`)

Portfolio-level risk analytics, position sizing, and threshold alerting. Subscribes to portfolio events (positions opened/closed) and signal events (regime changes) to re-evaluate risk in near-real time.

## Registration

`maverick_mcp/api/routers/tool_registry.py` is the one place where domain `register_*_tools(mcp)` functions are called. Adding a new domain means:

1. Write `mydomain.py` with `def register_mydomain_tools(mcp: FastMCP) -> None:`.
2. Import + call it in `tool_registry.py`.
3. Publish/subscribe via the event bus — do not reach into other domains directly.

## Conventions worth preserving

- **Typed list parameters**: `@mcp.tool()` functions must use `StrList` / `OptionalStrList` from `maverick_mcp.utils.mcp_types` for any `list[str]` argument. Bare `list[str]` breaks for clients that JSON-stringify array arguments (Claude Desktop via `mcp-remote`). Enforced by `scripts/check_mcp_list_types.py` (runs in `make check`). See `docs/runbooks/mcp-client-serialization.md`.
- **Event payloads are dicts**, not dataclasses or Pydantic models. Keep the bus framework-agnostic.
- **Scheduler jobs** must set `misfire_grace_time` and survive ASGI reloads cleanly — don't assume a long-lived loop.
- **No direct cross-domain imports.** If `risk_dashboard.py` needs something from `journal.py`, route through the event bus or add it to the portfolio service.

## Where to look next

- `docs/BACKTESTING.md` — VectorBT engine and strategy catalog.
- `docs/PORTFOLIO.md` — cost-basis and P&L engine that the risk dashboard reads from.
- `docs/deep_research_agent.md` — multi-agent research stack (separate from roadmap v1).
- `docs/security/DEPENDENCY_AUDIT.md` — dependency audit and license posture.
- `docs/runbooks/mcp-client-serialization.md` — why `list[str]` fails and how to use `StrList`.
