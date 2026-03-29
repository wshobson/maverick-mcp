# Maverick MCP Roadmap v1 — Design Spec

**Date**: 2026-03-27
**Status**: Approved
**Scope**: Sprint 1 milestone — 5 domains, ~25 new MCP tools, gateway architecture

---

## Vision

Maverick MCP evolves from a monolithic stock analysis server into a **thin intelligence gateway**. The MCP tool layer becomes a clean interface over a service layer that specialized backends (and eventually Maverick Bot) can consume directly.

**Target users**: Owner + small circle of self-hosting traders, each running their own instance.

**Future direction**: Maverick Bot for automated trading consumes the same services the MCP exposes. The MCP is one client of the service layer, not the service layer itself.

---

## Architecture

### Current State

```
Claude Desktop
    │
MCP Tools (86 tools, business logic mixed into routers)
    │
Providers (StockData, MarketData, MacroData, OpenRouter)
    │
Database (SQLite/PostgreSQL) + Cache (Redis optional)
```

### Target State

```
Claude Desktop / Maverick Bot (future)
        │
   MCP Tools (thin wrappers, ~111 tools)
        │
   Service Layer (business logic + orchestration)
        │
   ┌────┼────┬────────┬──────────┐
   │    │    │        │          │
Providers  EventBus  Scheduler  SignalEngine
   │                     │          │
 Tiingo/yfinance    APScheduler   EventBus
 FRED/OpenRouter    (in-process)  (publish signals)
 SQLite/Postgres
```

### Infrastructure Components

#### Event Bus

In-process async pub/sub using `asyncio.Queue`. No external dependencies.

```python
# Publish
await event_bus.publish("signal.triggered", {
    "symbol": "AAPL",
    "condition": "RSI < 30",
    "timestamp": "2026-03-27T14:30:00Z"
})

# Subscribe
@event_bus.subscribe("signal.triggered")
async def on_signal(event: dict): ...
```

**Event namespaces**:
- `signal.triggered` — a signal condition was met
- `signal.cleared` — a signal condition is no longer met
- `screening.entry` — stock entered a screening list
- `screening.exit` — stock exited a screening list
- `regime.changed` — market regime transitioned
- `trade.recorded` — a trade was logged to the journal
- `risk.alert` — a risk threshold was breached

Upgradeable to Redis pub/sub or a message broker when cross-process communication is needed.

#### Service Registry

Lightweight dict mapping service names to initialized instances. Created at server startup. Services register themselves. Consumers look up by name.

```python
registry = ServiceRegistry()
registry.register("signals", SignalService(event_bus, scheduler))
registry.register("screening", ScreeningPipelineService(event_bus, scheduler))
# ...
signal_svc = registry.get("signals")
```

#### Scheduler

APScheduler with `AsyncIOScheduler`. In-process, no external deps. Jobs persist to SQLite so they survive restarts. Configurable intervals per job.

Used by: Signal Engine (evaluate conditions every 5min), Screening Pipeline (run screens on schedule), Catalyst Tracker (refresh earnings dates daily).

---

## Domain 1: Signal Engine

**Covers**: Real-time alerts/signals, market regime detection.

### Signal Service

CRUD for signal definitions. Evaluation engine runs on a schedule (default: every 5 minutes) via APScheduler.

**Signal condition types**:
- Price crossover: above/below a level or moving average
- Indicator threshold: RSI, MACD histogram, Bollinger %B
- Volume spike: N standard deviations above average
- Regime change: market transitions from one state to another

**Condition DSL** — dict-based, not a custom language:
```python
{"indicator": "rsi", "operator": "lt", "value": 30, "period": 14}
{"indicator": "price", "operator": "crosses_above", "reference": "sma_200"}
{"indicator": "volume", "operator": "spike", "std_devs": 2.0}
```

**Evaluation flow**:
1. Scheduler fires evaluation job
2. Fetch latest data for all watched tickers (batch, deduplicated)
3. Run indicator calculations against each signal's conditions
4. For triggered signals: publish `signal.triggered` event, persist `SignalEvent`
5. For cleared signals (previously triggered, now false): publish `signal.cleared`

### Regime Detector

Classifies current market as **bull**, **bear**, **choppy**, or **transitional** using a composite of 4 factors:

1. **Market breadth** — advance/decline ratio, new highs vs new lows
2. **Volatility regime** — VIX level + VIX term structure (contango/backwardation)
3. **Trend strength** — SPY position relative to 50/200 SMA, slope of moving averages
4. **Momentum** — rate of change across market indices, breadth thrust indicators

Each factor votes bull/bear/choppy. Weighted ensemble produces regime + confidence score (0-1).

Publishes `regime.changed` events when the classification shifts.

### Stateful Condition Tracking

Crossover conditions (e.g., `crosses_above`) require knowing the previous evaluation state. Each `Signal` row stores a `previous_state` JSON field updated after every evaluation cycle. For a crossover: `{"previous_value": 195.30, "was_above": false}`. The evaluation engine compares current vs. previous to detect transitions.

### Database Models

- `Signal` — id, user label, ticker, condition (JSON), interval, active flag, previous_state (JSON, for stateful conditions), created_at
- `SignalEvent` — id, signal_id (FK), triggered_at, price_at_trigger, condition_snapshot (JSON)
- `RegimeEvent` — id, regime (bull/bear/choppy/transitional), confidence, drivers (JSON), detected_at, previous_regime

### MCP Tools (~7)

| Tool | Description |
|------|-------------|
| `create_signal` | Define a new signal condition on a ticker |
| `update_signal` | Modify an existing signal's conditions or interval |
| `list_signals` | List all active signals with last trigger time |
| `delete_signal` | Remove a signal |
| `check_signals_now` | Manually evaluate all signals immediately |
| `get_market_regime` | Current regime classification with confidence and drivers |
| `get_regime_history` | Regime shifts over the past N days (from `RegimeEvent` table) |

---

## Domain 2: Screening Pipeline

**Covers**: Automated screening with change detection.

### Pipeline Service

Wraps existing screening tools (`get_maverick_recommendations`, `get_maverick_bear_recommendations`, `get_trending_breakout_recommendations`) in a scheduler. Snapshots results after each run. Diffs against previous snapshot to detect entries and exits.

**Flow**:
1. Scheduler fires screening job (configurable, default: daily at market close)
2. Run configured screens
3. Snapshot results to `ScreeningRun`
4. Diff against previous run → produce `ScreeningChange` records
5. Publish `screening.entry` / `screening.exit` events for each change

### Database Models

- `ScreeningRun` — id, screen_name, run_at, result_count, results (JSON)
- `ScreeningChange` — id, run_id (FK), symbol, change_type (entry/exit/rank_change), screen_name, previous_rank (nullable int), new_rank (nullable int), detected_at

### Scheduling Mechanism

`schedule_screening` creates/modifies APScheduler jobs at runtime. Job definitions (screen name, cron expression) are persisted to a `ScheduledJob` table so they survive restarts. The scheduler reads this table on startup and recreates jobs.

### MCP Tools (~4)

| Tool | Description |
|------|-------------|
| `get_screening_changes` | What changed since last run — new entries, exits, rank changes |
| `get_screening_history` | How long has a stock been on a given screen |
| `schedule_screening` | Configure which screens run and how often |
| `get_screening_pipeline_status` | What's scheduled, when it last ran, next run time |

---

## Domain 3: Trade Journal

**Covers**: Trade journaling, strategy performance tracking.

### Journal Service

Record trades with full context: entry/exit prices, size, rationale, tags linking to strategies/screens, outcome. AI-powered trade review using existing OpenRouter research agents.

### Strategy Tracker

Aggregates journal entries by strategy tag. Computes: win rate, average gain/loss, expectancy (avg_win * win_rate - avg_loss * loss_rate), profit factor, R-multiple distribution.

**Integration with Screening Pipeline**: When a trade is tagged with a screen name (e.g., "maverick_bullish"), the system links it to screening history. Over time this builds a feedback loop showing which screens produce profitable trades.

### Database Models

- `JournalEntry` — id, symbol, side (long/short), entry_price, exit_price, shares, entry_date, exit_date, rationale (text), tags (JSON array), pnl, r_multiple, notes, status (open/closed)
- `StrategyPerformance` — id, strategy_tag, period, win_count, loss_count, total_pnl, avg_win, avg_loss, expectancy, profit_factor, updated_at (materialized rollup, recomputed via application-level event: when a `JournalEntry.status` changes to `closed`, the service recalculates the rollup for all affected strategy tags)

### MCP Tools (~6)

| Tool | Description |
|------|-------------|
| `journal_add_trade` | Log a trade with entry/exit, rationale, strategy tags |
| `journal_list_trades` | Filter by date, ticker, strategy, outcome |
| `journal_trade_review` | AI-powered analysis of a specific trade |
| `get_strategy_performance` | Win rate, expectancy, profit factor per strategy |
| `get_strategy_comparison` | Side-by-side comparison of all strategies |
| `get_trading_patterns` | AI analysis of journal: mistakes, best setups, patterns |

---

## Domain 4: Watchlist Intelligence

**Covers**: Smart watchlists, earnings/events calendar.

### Watchlist Service

Watchlists where each stock has a daily "intelligence score" — a composite of signal activity, catalyst proximity, technical setup quality, and screening status.

**Daily scoring factors** (each 0-100, weighted):
- Signal activity: how many signals are active/recently triggered
- Catalyst proximity: days until next earnings, ex-div, etc.
- Technical quality: setup score from existing technical analysis
- Screening status: on which screens, how recently entered

### Catalyst Tracker

Fetches earnings dates, ex-dividend dates from Tiingo/yfinance. Stores in `CatalystEvent` table. Refreshed daily via scheduler.

AI-powered impact assessment for upcoming catalysts using existing research agents: "AAPL earnings Thursday — consensus expects strong iPhone revenue, options imply 4% move."

### Database Models

- `Watchlist` — id, name, description, created_at
- `WatchlistItem` — id, watchlist_id (FK), symbol, added_at, notes
- `CatalystEvent` — id, symbol, event_type (earnings/ex_div/fda/other), event_date, description, impact_assessment (text, AI-generated)

### MCP Tools (5)

| Tool | Description |
|------|-------------|
| `watchlist_create` | Create a named watchlist |
| `watchlist_add` | Add tickers to a watchlist |
| `watchlist_remove` | Remove tickers from a watchlist |
| `watchlist_brief` | "What's interesting today?" — scored and ranked |
| `get_upcoming_catalysts` | Earnings, ex-div, events for watchlist or ticker list |

---

## Domain 5: Risk Dashboard

**Covers**: Enhanced portfolio-level risk management.

### Risk Service

Aggregates risk across the portfolio. Integrates with existing `PortfolioManager`, `RegimeDetector`, and `SignalEngine`.

**Risk metrics computed**:
- Portfolio VaR (95%, 99%) — parametric and historical
- Sector/industry concentration
- Pairwise correlation heat map (top correlated pairs)
- Maximum drawdown exposure given current positions
- Beta-weighted delta (portfolio sensitivity to SPY)

**Regime-aware sizing**: Position size recommendations adjust based on current market regime. Tighter sizing in choppy/bear regimes, standard in trending bull.

### Risk Alerts

Subscribes to events from other domains. Auto-generates alerts:
- Over-concentration in a single sector (default >30%, configurable)
- High correlation between top positions (default >0.8, configurable)
- Approaching user-defined max drawdown threshold
- Position size exceeds regime-adjusted recommendation

Thresholds are stored in a `RiskConfig` JSON column on the `UserPortfolio` model (existing). Configurable via a future settings tool; hardcoded defaults for Sprint 1.

Publishes `risk.alert` events.

**Portfolio scope**: Operates on one portfolio at a time (matching existing multi-portfolio support). The `portfolio_name` parameter selects which portfolio to analyze. Defaults to the user's primary portfolio.

### Database Models

- `RiskAlert` — id, portfolio_name, alert_type (concentration/correlation/drawdown/sizing), severity (warning/critical), message, details (JSON), created_at, acknowledged (bool)
- `RiskSnapshot` — id, portfolio_name, var_95, var_99, max_sector_pct, max_correlation, beta_weighted_delta, regime, snapshot_at

### MCP Tools (~4)

| Tool | Description |
|------|-------------|
| `get_portfolio_risk_dashboard` | Full risk view: VaR, concentration, correlation, drawdown |
| `get_position_risk_check` | Pre-trade risk check: "what happens if I add TSLA?" |
| `get_regime_adjusted_sizing` | Position size factoring in market regime |
| `get_risk_alerts` | Current risk flags across the portfolio |

---

## File Organization

**Note on existing `api/services/`**: The codebase has an existing `maverick_mcp/api/services/portfolio_service.py`. The new `maverick_mcp/services/` is a separate top-level package for domain services. Existing code in `api/services/` stays as-is for Sprint 1. Future sprints may migrate it into the new service layer.

**Note on existing tools**: The existing ~86 MCP tools are not refactored to thin wrappers in Sprint 1. Only new tools follow the thin-wrapper pattern. Refactoring existing tools is a future sprint task.

```
maverick_mcp/
  services/
    __init__.py
    event_bus.py              # async pub/sub
    registry.py               # service registration
    scheduler.py              # APScheduler wrapper
    signals/
      __init__.py
      service.py              # SignalService
      conditions.py           # condition evaluation engine
      regime.py               # RegimeDetector
      models.py               # Signal, SignalEvent
    screening/
      __init__.py
      pipeline.py             # ScreeningPipelineService
      models.py               # ScreeningRun, ScreeningChange
    journal/
      __init__.py
      service.py              # JournalService
      analytics.py            # StrategyTracker
      models.py               # JournalEntry, StrategyPerformance
    watchlist/
      __init__.py
      service.py              # WatchlistService
      catalysts.py            # CatalystTracker
      models.py               # Watchlist, WatchlistItem, CatalystEvent
    risk/
      __init__.py
      service.py              # RiskService, risk aggregation
      models.py               # RiskAlert, RiskSnapshot
  api/routers/
    signals.py                # MCP tools for signal engine
    screening_pipeline.py     # MCP tools for screening pipeline
    journal.py                # MCP tools for trade journal
    watchlist.py              # MCP tools for watchlist intelligence
    risk_dashboard.py         # MCP tools for risk dashboard
```

---

## Implementation Sequence

### Week 1: Foundation + Signal Engine + Screening Pipeline

| Day | Work |
|-----|------|
| 1-2 | Infrastructure: event bus, service registry, scheduler, DB migrations, wire into server startup |
| 3-4 | Signal Engine: condition DSL, evaluation loop, regime detector, MCP tools |
| 5   | Screening Pipeline: snapshot/diff logic, change detection, MCP tools |

### Week 2: Journal + Watchlist + Risk + Integration

| Day | Work |
|-----|------|
| 6-7 | Trade Journal: recording, strategy tracking, AI trade review, MCP tools |
| 8   | Watchlist Intelligence: scoring engine, catalyst tracker, MCP tools |
| 9   | Risk Dashboard: portfolio risk aggregation, regime-adjusted sizing, MCP tools |
| 10  | Integration pass: cross-domain event wiring, merge open community PRs (#118, #98), cleanup |

---

## Pre-Sprint Housekeeping

Merge these open PRs before starting:
- **#118** (tad3j): Fix unsupported debug argument in `mcp.run` for stdio transport
- **#98** (luisdeltoro): Fix rate limiting middleware registration on FastAPI app

These are legitimate bug fixes from community contributors that reduce friction.

---

## Maverick Bot Extension Points

The design creates explicit seams for Maverick Bot:

| Seam | How Bot Uses It |
|------|----------------|
| Event bus | Subscribe to `signal.triggered`, `screening.entry`, `regime.changed` |
| Service registry | Call services directly, bypass MCP tool layer |
| Trade journal | Log bot trades through same journal for human review |
| Risk service | Pre-trade risk check before execution |
| Signal engine | Bot trading rules = signal definitions with `auto_execute: true` flag (added to `Signal` model in Bot sprint, not Sprint 1) |

When Maverick Bot arrives, it's a new consumer of existing services — not a rewrite.

**New dependency**: APScheduler (`apscheduler>=4.0`) — add to `pyproject.toml` in Sprint 1 infrastructure phase.

---

## Open Issue Disposition

- **#119** (investment principles): Defer to post-sprint. Could integrate as enrichment context for trade journal AI reviews. Low priority.
- **#97** (GitMCP proxy bug): Not a server bug — it's about the GitMCP proxy behavior. Close with explanation.

---

## Success Criteria

Sprint is complete when:
1. All 5 domains have working MCP tools callable from Claude Desktop
2. Signal engine evaluates conditions on a schedule and surfaces triggers
3. Screening pipeline detects changes between runs
4. Trade journal records trades and computes strategy performance
5. Watchlist brief returns scored, ranked intelligence
6. Risk dashboard shows portfolio-level risk with regime awareness
7. Events flow between domains via the event bus
8. All new code has unit tests
9. Existing 93 tests still pass
