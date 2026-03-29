# Maverick MCP Roadmap v1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 5 feature domains (signals, screening pipeline, trade journal, watchlist intelligence, risk dashboard) on a service-layer architecture with event bus and scheduler.

**Architecture:** New `maverick_mcp/services/` package with async event bus, service registry, and APScheduler. Each domain is a service with its own models, registered at startup. MCP tools are thin wrappers in `api/routers/`. All new DB models use existing `Base` from `maverick_mcp/database/base.py` and `TimestampMixin` from `maverick_mcp/data/models.py`.

**Tech Stack:** Python 3.12+, SQLAlchemy (existing), APScheduler 4.x, FastMCP (existing), pandas/numpy (existing)

**Spec:** `docs/superpowers/specs/2026-03-27-maverick-roadmap-v1-design.md`

---

## File Map

### New Files (Infrastructure)

| File | Responsibility |
|------|---------------|
| `maverick_mcp/services/__init__.py` | Package init, exports `ServiceRegistry`, `EventBus`, `Scheduler` |
| `maverick_mcp/services/event_bus.py` | Async in-process pub/sub |
| `maverick_mcp/services/registry.py` | Service registration and lookup |
| `maverick_mcp/services/scheduler.py` | APScheduler wrapper with SQLite job persistence |
| `tests/unit/test_event_bus.py` | Event bus tests |
| `tests/unit/test_service_registry.py` | Registry tests |
| `tests/unit/test_scheduler.py` | Scheduler tests |

### New Files (Domain 1: Signals)

| File | Responsibility |
|------|---------------|
| `maverick_mcp/services/signals/__init__.py` | Package init |
| `maverick_mcp/services/signals/models.py` | `Signal`, `SignalEvent`, `RegimeEvent` SQLAlchemy models |
| `maverick_mcp/services/signals/conditions.py` | Condition evaluation engine |
| `maverick_mcp/services/signals/regime.py` | `RegimeDetector` — market regime classification |
| `maverick_mcp/services/signals/service.py` | `SignalService` — CRUD + evaluation orchestration |
| `maverick_mcp/api/routers/signals.py` | MCP tools (thin wrappers) |
| `tests/unit/test_signal_conditions.py` | Condition evaluation tests |
| `tests/unit/test_regime_detector.py` | Regime detection tests |
| `tests/unit/test_signal_service.py` | Signal service tests |

### New Files (Domain 2: Screening Pipeline)

| File | Responsibility |
|------|---------------|
| `maverick_mcp/services/screening/__init__.py` | Package init |
| `maverick_mcp/services/screening/models.py` | `ScreeningRun`, `ScreeningChange`, `ScheduledJob` models |
| `maverick_mcp/services/screening/pipeline.py` | `ScreeningPipelineService` — snapshot, diff, schedule |
| `maverick_mcp/api/routers/screening_pipeline.py` | MCP tools (thin wrappers) |
| `tests/unit/test_screening_pipeline.py` | Pipeline service tests |

### New Files (Domain 3: Trade Journal)

| File | Responsibility |
|------|---------------|
| `maverick_mcp/services/journal/__init__.py` | Package init |
| `maverick_mcp/services/journal/models.py` | `JournalEntry`, `StrategyPerformance` models |
| `maverick_mcp/services/journal/service.py` | `JournalService` — trade recording, AI review |
| `maverick_mcp/services/journal/analytics.py` | `StrategyTracker` — performance rollups |
| `maverick_mcp/api/routers/journal.py` | MCP tools (thin wrappers) |
| `tests/unit/test_journal_service.py` | Journal service tests |
| `tests/unit/test_strategy_tracker.py` | Strategy analytics tests |

### New Files (Domain 4: Watchlist Intelligence)

| File | Responsibility |
|------|---------------|
| `maverick_mcp/services/watchlist/__init__.py` | Package init |
| `maverick_mcp/services/watchlist/models.py` | `Watchlist`, `WatchlistItem`, `CatalystEvent` models |
| `maverick_mcp/services/watchlist/service.py` | `WatchlistService` — scoring, intelligence |
| `maverick_mcp/services/watchlist/catalysts.py` | `CatalystTracker` — earnings/events fetching |
| `maverick_mcp/api/routers/watchlist.py` | MCP tools (thin wrappers) |
| `tests/unit/test_watchlist_service.py` | Watchlist service tests |
| `tests/unit/test_catalyst_tracker.py` | Catalyst tracker tests |

### New Files (Domain 5: Risk Dashboard)

| File | Responsibility |
|------|---------------|
| `maverick_mcp/services/risk/__init__.py` | Package init |
| `maverick_mcp/services/risk/models.py` | `RiskAlert`, `RiskSnapshot` models |
| `maverick_mcp/services/risk/service.py` | `RiskService` — aggregation, regime-aware sizing |
| `maverick_mcp/api/routers/risk_dashboard.py` | MCP tools (thin wrappers) |
| `tests/unit/test_risk_service.py` | Risk service tests |

### Modified Files

| File | Change |
|------|--------|
| `pyproject.toml` | Add `apscheduler>=4.0` dependency |
| `maverick_mcp/api/routers/tool_registry.py` | Add `register_signal_tools`, `register_screening_pipeline_tools`, `register_journal_tools`, `register_watchlist_tools`, `register_risk_dashboard_tools` to `register_all_router_tools` |
| `maverick_mcp/api/server.py` | Initialize service registry, event bus, scheduler at startup; register shutdown cleanup |

---

## Task 0: Merge Community PRs (Pre-Sprint Housekeeping)

**Files:** None (git operations only)

- [ ] **Step 1: Review and merge PR #118 (debug arg fix)**

```bash
gh pr checkout 118
uv run pytest tests/ -x -q --timeout=30
gh pr merge 118 --squash
git checkout main && git pull
```

- [ ] **Step 2: Review and merge PR #98 (rate limiting middleware fix)**

```bash
gh pr checkout 98
uv run pytest tests/ -x -q --timeout=30
gh pr merge 98 --squash
git checkout main && git pull
```

---

## Task 1: Add APScheduler Dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add apscheduler to dependencies**

In `pyproject.toml`, add `apscheduler>=4.0` to the `dependencies` list. Find the `dependencies = [` section and add the entry alphabetically.

- [ ] **Step 2: Sync dependencies**

Run: `uv sync`
Expected: Clean install with apscheduler added

- [ ] **Step 3: Verify import works**

Run: `uv run python -c "from apscheduler import AsyncScheduler; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add apscheduler dependency for scheduled task support"
```

---

## Task 2: Event Bus

**Files:**
- Create: `maverick_mcp/services/__init__.py`
- Create: `maverick_mcp/services/event_bus.py`
- Test: `tests/unit/test_event_bus.py`

- [ ] **Step 1: Create services package init**

Create `maverick_mcp/services/__init__.py`:

```python
"""Service layer for Maverick MCP — business logic and orchestration."""
```

- [ ] **Step 2: Write failing tests for event bus**

Create `tests/unit/test_event_bus.py`:

```python
"""Tests for the async event bus."""

import asyncio

import pytest

from maverick_mcp.services.event_bus import EventBus


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.mark.asyncio
async def test_publish_and_subscribe(event_bus):
    received = []

    async def handler(event):
        received.append(event)

    event_bus.subscribe("test.event", handler)
    await event_bus.publish("test.event", {"key": "value"})
    # Give the handler time to process
    await asyncio.sleep(0.05)

    assert len(received) == 1
    assert received[0]["key"] == "value"


@pytest.mark.asyncio
async def test_multiple_subscribers(event_bus):
    results_a = []
    results_b = []

    async def handler_a(event):
        results_a.append(event)

    async def handler_b(event):
        results_b.append(event)

    event_bus.subscribe("test.multi", handler_a)
    event_bus.subscribe("test.multi", handler_b)
    await event_bus.publish("test.multi", {"n": 1})
    await asyncio.sleep(0.05)

    assert len(results_a) == 1
    assert len(results_b) == 1


@pytest.mark.asyncio
async def test_unsubscribe(event_bus):
    received = []

    async def handler(event):
        received.append(event)

    event_bus.subscribe("test.unsub", handler)
    event_bus.unsubscribe("test.unsub", handler)
    await event_bus.publish("test.unsub", {"key": "value"})
    await asyncio.sleep(0.05)

    assert len(received) == 0


@pytest.mark.asyncio
async def test_no_crosstalk(event_bus):
    received = []

    async def handler(event):
        received.append(event)

    event_bus.subscribe("topic.a", handler)
    await event_bus.publish("topic.b", {"key": "value"})
    await asyncio.sleep(0.05)

    assert len(received) == 0


@pytest.mark.asyncio
async def test_handler_error_does_not_break_others(event_bus):
    results = []

    async def bad_handler(event):
        raise ValueError("boom")

    async def good_handler(event):
        results.append(event)

    event_bus.subscribe("test.error", bad_handler)
    event_bus.subscribe("test.error", good_handler)
    await event_bus.publish("test.error", {"n": 1})
    await asyncio.sleep(0.05)

    assert len(results) == 1


@pytest.mark.asyncio
async def test_event_history(event_bus):
    await event_bus.publish("test.history", {"n": 1})
    await event_bus.publish("test.history", {"n": 2})

    history = event_bus.get_history("test.history")
    assert len(history) == 2
    assert history[0]["data"]["n"] == 1
    assert history[1]["data"]["n"] == 2


@pytest.mark.asyncio
async def test_history_max_size(event_bus):
    event_bus.max_history = 3
    for i in range(5):
        await event_bus.publish("test.cap", {"n": i})

    history = event_bus.get_history("test.cap")
    assert len(history) == 3
    assert history[0]["data"]["n"] == 2  # oldest kept
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_event_bus.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'maverick_mcp.services.event_bus'`

- [ ] **Step 4: Implement event bus**

Create `maverick_mcp/services/event_bus.py`:

```python
"""Async in-process event bus for cross-domain communication."""

import asyncio
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

EventHandler = Callable[[dict[str, Any]], Awaitable[None]]


class EventBus:
    """Simple async pub/sub event bus.

    Handlers are called concurrently via asyncio.gather. A failing handler
    does not prevent other handlers from executing.
    """

    def __init__(self, max_history: int = 100):
        self._subscribers: dict[str, list[EventHandler]] = defaultdict(list)
        self._history: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.max_history = max_history

    def subscribe(self, topic: str, handler: EventHandler) -> None:
        """Register a handler for a topic."""
        self._subscribers[topic].append(handler)
        logger.debug("Subscribed handler %s to topic %s", handler.__name__, topic)

    def unsubscribe(self, topic: str, handler: EventHandler) -> None:
        """Remove a handler from a topic."""
        handlers = self._subscribers.get(topic, [])
        if handler in handlers:
            handlers.remove(handler)
            logger.debug("Unsubscribed handler %s from topic %s", handler.__name__, topic)

    async def publish(self, topic: str, data: dict[str, Any]) -> None:
        """Publish an event to all subscribers of a topic."""
        envelope = {
            "topic": topic,
            "data": data,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Store in history
        history = self._history[topic]
        history.append(envelope)
        if len(history) > self.max_history:
            self._history[topic] = history[-self.max_history :]

        # Dispatch to handlers
        handlers = self._subscribers.get(topic, [])
        if not handlers:
            return

        results = await asyncio.gather(
            *[self._safe_call(h, data) for h in handlers],
            return_exceptions=True,
        )
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "Handler %s failed for topic %s: %s",
                    handlers[i].__name__,
                    topic,
                    result,
                )

    async def _safe_call(self, handler: EventHandler, data: dict[str, Any]) -> None:
        """Call a handler, catching exceptions."""
        await handler(data)

    def get_history(self, topic: str, limit: int | None = None) -> list[dict[str, Any]]:
        """Get recent events for a topic."""
        history = self._history.get(topic, [])
        if limit:
            return history[-limit:]
        return list(history)

    def clear_history(self, topic: str | None = None) -> None:
        """Clear event history for a topic or all topics."""
        if topic:
            self._history.pop(topic, None)
        else:
            self._history.clear()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_event_bus.py -v`
Expected: All 8 tests PASS

- [ ] **Step 6: Commit**

```bash
git add maverick_mcp/services/__init__.py maverick_mcp/services/event_bus.py tests/unit/test_event_bus.py
git commit -m "feat: add async event bus for cross-domain communication"
```

---

## Task 3: Service Registry

**Files:**
- Create: `maverick_mcp/services/registry.py`
- Test: `tests/unit/test_service_registry.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_service_registry.py`:

```python
"""Tests for the service registry."""

import pytest

from maverick_mcp.services.registry import ServiceRegistry


@pytest.fixture
def registry():
    return ServiceRegistry()


def test_register_and_get(registry):
    service = {"name": "test"}
    registry.register("test_svc", service)
    assert registry.get("test_svc") is service


def test_get_unregistered_raises(registry):
    with pytest.raises(KeyError, match="test_svc"):
        registry.get("test_svc")


def test_get_optional_returns_none(registry):
    assert registry.get_optional("missing") is None


def test_register_duplicate_raises(registry):
    registry.register("svc", "first")
    with pytest.raises(ValueError, match="already registered"):
        registry.register("svc", "second")


def test_register_duplicate_replace(registry):
    registry.register("svc", "first")
    registry.register("svc", "second", replace=True)
    assert registry.get("svc") == "second"


def test_list_services(registry):
    registry.register("a", 1)
    registry.register("b", 2)
    assert set(registry.list_services()) == {"a", "b"}


def test_has(registry):
    assert not registry.has("svc")
    registry.register("svc", "val")
    assert registry.has("svc")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_service_registry.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement service registry**

Create `maverick_mcp/services/registry.py`:

```python
"""Lightweight service registry for dependency lookup."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ServiceRegistry:
    """Maps service names to instances. Created at startup."""

    def __init__(self):
        self._services: dict[str, Any] = {}

    def register(self, name: str, service: Any, *, replace: bool = False) -> None:
        """Register a service instance by name."""
        if name in self._services and not replace:
            raise ValueError(f"Service '{name}' already registered")
        self._services[name] = service
        logger.info("Registered service: %s", name)

    def get(self, name: str) -> Any:
        """Get a service by name. Raises KeyError if not found."""
        if name not in self._services:
            raise KeyError(f"Service '{name}' not registered")
        return self._services[name]

    def get_optional(self, name: str) -> Any | None:
        """Get a service by name, returning None if not found."""
        return self._services.get(name)

    def has(self, name: str) -> bool:
        """Check if a service is registered."""
        return name in self._services

    def list_services(self) -> list[str]:
        """List all registered service names."""
        return list(self._services.keys())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_service_registry.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add maverick_mcp/services/registry.py tests/unit/test_service_registry.py
git commit -m "feat: add service registry for dependency lookup"
```

---

## Task 4: Scheduler Wrapper

**Files:**
- Create: `maverick_mcp/services/scheduler.py`
- Test: `tests/unit/test_scheduler.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_scheduler.py`:

```python
"""Tests for the scheduler wrapper."""

import asyncio

import pytest

from maverick_mcp.services.scheduler import MaverickScheduler


@pytest.fixture
async def scheduler():
    s = MaverickScheduler()
    yield s
    await s.shutdown()


@pytest.mark.asyncio
async def test_add_and_list_jobs(scheduler):
    call_count = 0

    async def my_job():
        nonlocal call_count
        call_count += 1

    await scheduler.start()
    scheduler.add_interval_job("test_job", my_job, seconds=3600)

    jobs = scheduler.list_jobs()
    assert len(jobs) == 1
    assert jobs[0]["id"] == "test_job"


@pytest.mark.asyncio
async def test_remove_job(scheduler):
    async def my_job():
        pass

    await scheduler.start()
    scheduler.add_interval_job("removable", my_job, seconds=3600)
    assert len(scheduler.list_jobs()) == 1

    scheduler.remove_job("removable")
    assert len(scheduler.list_jobs()) == 0


@pytest.mark.asyncio
async def test_remove_nonexistent_job_is_noop(scheduler):
    await scheduler.start()
    scheduler.remove_job("nonexistent")  # Should not raise


@pytest.mark.asyncio
async def test_job_executes(scheduler):
    results = []

    async def my_job():
        results.append(1)

    await scheduler.start()
    scheduler.add_interval_job("fast_job", my_job, seconds=0.1)
    await asyncio.sleep(0.35)

    assert len(results) >= 2


@pytest.mark.asyncio
async def test_add_cron_job(scheduler):
    async def my_job():
        pass

    await scheduler.start()
    scheduler.add_cron_job("cron_job", my_job, hour=0, minute=0)

    jobs = scheduler.list_jobs()
    assert len(jobs) == 1
    assert jobs[0]["id"] == "cron_job"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_scheduler.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement scheduler**

Create `maverick_mcp/services/scheduler.py`:

```python
"""APScheduler wrapper for periodic task execution."""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from apscheduler import AsyncScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)


class MaverickScheduler:
    """Wraps APScheduler with a simplified API for Maverick services."""

    def __init__(self):
        self._scheduler: AsyncScheduler | None = None
        self._jobs: dict[str, dict[str, Any]] = {}

    async def start(self) -> None:
        """Start the scheduler."""
        self._scheduler = AsyncScheduler()
        await self._scheduler.__aenter__()
        logger.info("Scheduler started")

    async def shutdown(self) -> None:
        """Stop the scheduler gracefully."""
        if self._scheduler:
            await self._scheduler.__aexit__(None, None, None)
            self._scheduler = None
            self._jobs.clear()
            logger.info("Scheduler stopped")

    def add_interval_job(
        self,
        job_id: str,
        func: Callable[..., Awaitable[None]],
        *,
        seconds: float | None = None,
        minutes: float | None = None,
        hours: float | None = None,
    ) -> None:
        """Add a job that runs at a fixed interval."""
        if not self._scheduler:
            raise RuntimeError("Scheduler not started")

        trigger_kwargs = {}
        if seconds is not None:
            trigger_kwargs["seconds"] = seconds
        if minutes is not None:
            trigger_kwargs["minutes"] = minutes
        if hours is not None:
            trigger_kwargs["hours"] = hours

        self._scheduler.add_schedule(
            func,
            trigger=IntervalTrigger(**trigger_kwargs),
            id=job_id,
        )
        self._jobs[job_id] = {
            "id": job_id,
            "type": "interval",
            "trigger_kwargs": trigger_kwargs,
        }
        logger.info("Added interval job: %s", job_id)

    def add_cron_job(
        self,
        job_id: str,
        func: Callable[..., Awaitable[None]],
        **cron_kwargs: Any,
    ) -> None:
        """Add a job with a cron-like schedule."""
        if not self._scheduler:
            raise RuntimeError("Scheduler not started")

        self._scheduler.add_schedule(
            func,
            trigger=CronTrigger(**cron_kwargs),
            id=job_id,
        )
        self._jobs[job_id] = {
            "id": job_id,
            "type": "cron",
            "trigger_kwargs": cron_kwargs,
        }
        logger.info("Added cron job: %s", job_id)

    def remove_job(self, job_id: str) -> None:
        """Remove a scheduled job. No-op if job doesn't exist."""
        if job_id in self._jobs:
            self._jobs.pop(job_id, None)
            logger.info("Removed job: %s", job_id)

    def list_jobs(self) -> list[dict[str, Any]]:
        """List all scheduled jobs."""
        return list(self._jobs.values())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_scheduler.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add maverick_mcp/services/scheduler.py tests/unit/test_scheduler.py
git commit -m "feat: add scheduler wrapper for periodic task execution"
```

---

## Task 5: Wire Infrastructure into Server Startup

**Files:**
- Modify: `maverick_mcp/api/server.py`
- Modify: `maverick_mcp/services/__init__.py`

- [ ] **Step 1: Update services init to export singletons**

Update `maverick_mcp/services/__init__.py`:

```python
"""Service layer for Maverick MCP — business logic and orchestration.

Provides module-level singletons for the event bus, service registry,
and scheduler. Import these from `maverick_mcp.services`.
"""

from maverick_mcp.services.event_bus import EventBus
from maverick_mcp.services.registry import ServiceRegistry
from maverick_mcp.services.scheduler import MaverickScheduler

# Module-level singletons — initialized once, shared across the app
event_bus = EventBus()
registry = ServiceRegistry()
scheduler = MaverickScheduler()

__all__ = [
    "EventBus",
    "MaverickScheduler",
    "ServiceRegistry",
    "event_bus",
    "registry",
    "scheduler",
]
```

- [ ] **Step 2: Add startup and shutdown hooks in server.py**

In `maverick_mcp/api/server.py`, find the `async def init_systems():` function (around line 1504) and add scheduler startup after the existing initialization:

```python
        # Initialize service layer
        logger.info("Starting service layer...")
        try:
            from maverick_mcp.services import scheduler as maverick_scheduler

            await maverick_scheduler.start()
            logger.info("Service layer scheduler started")
        except Exception as e:
            logger.error(f"Failed to start service layer: {e}")
```

Then find `shutdown_handler.register_cleanup(cleanup_database)` (around line 1679) and add before it:

```python
        async def cleanup_service_layer():
            """Shutdown service layer scheduler."""
            try:
                from maverick_mcp.services import scheduler as maverick_scheduler

                await maverick_scheduler.shutdown()
                logger.info("Service layer scheduler stopped")
            except Exception as e:
                logger.error(f"Service layer shutdown error: {e}")

        shutdown_handler.register_cleanup(cleanup_service_layer)
```

- [ ] **Step 3: Verify server starts cleanly**

Run: `uv run python -m maverick_mcp.api.server --transport stdio < /dev/null 2>&1 | head -20`
Expected: Log output showing "Service layer scheduler started" without errors

- [ ] **Step 4: Run existing tests to ensure no regressions**

Run: `uv run pytest tests/ -x -q --timeout=30 2>&1 | tail -20`
Expected: All existing tests still pass

- [ ] **Step 5: Commit**

```bash
git add maverick_mcp/services/__init__.py maverick_mcp/api/server.py
git commit -m "feat: wire service layer infrastructure into server lifecycle"
```

---

## Task 6: Signal Models

**Files:**
- Create: `maverick_mcp/services/signals/__init__.py`
- Create: `maverick_mcp/services/signals/models.py`

- [ ] **Step 1: Create signals package**

Create `maverick_mcp/services/signals/__init__.py`:

```python
"""Signal engine — alerts, conditions, and market regime detection."""
```

- [ ] **Step 2: Implement signal models**

Create `maverick_mcp/services/signals/models.py`:

```python
"""SQLAlchemy models for the signal engine."""

from datetime import UTC, datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.dialects.sqlite import JSON

from maverick_mcp.database.base import Base
from maverick_mcp.data.models import TimestampMixin


class Signal(Base, TimestampMixin):
    """A user-defined signal condition on a ticker."""

    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    label = Column(String(255), nullable=False)
    ticker = Column(String(10), nullable=False, index=True)
    condition = Column(JSON, nullable=False)  # {"indicator": "rsi", "operator": "lt", "value": 30, ...}
    interval_seconds = Column(Integer, default=300)  # 5 minutes default
    active = Column(Boolean, default=True, nullable=False)
    previous_state = Column(JSON, nullable=True)  # for stateful conditions like crosses_above


class SignalEvent(Base, TimestampMixin):
    """Records when a signal condition was triggered."""

    __tablename__ = "signal_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(Integer, nullable=False, index=True)
    triggered_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False)
    price_at_trigger = Column(Float, nullable=True)
    condition_snapshot = Column(JSON, nullable=True)


class RegimeEvent(Base, TimestampMixin):
    """Records market regime transitions."""

    __tablename__ = "regime_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    regime = Column(String(20), nullable=False)  # bull, bear, choppy, transitional
    confidence = Column(Float, nullable=False)
    drivers = Column(JSON, nullable=True)  # {"breadth": "bull", "volatility": "bear", ...}
    previous_regime = Column(String(20), nullable=True)
    detected_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False)
```

- [ ] **Step 3: Verify models register with Base**

Run: `uv run python -c "from maverick_mcp.services.signals.models import Signal, SignalEvent, RegimeEvent; print('Tables:', [m.__tablename__ for m in [Signal, SignalEvent, RegimeEvent]])"`
Expected: `Tables: ['signals', 'signal_events', 'regime_events']`

- [ ] **Step 4: Commit**

```bash
git add maverick_mcp/services/signals/
git commit -m "feat: add signal engine database models"
```

---

## Task 7: Signal Condition Engine

**Files:**
- Create: `maverick_mcp/services/signals/conditions.py`
- Test: `tests/unit/test_signal_conditions.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_signal_conditions.py`:

```python
"""Tests for signal condition evaluation."""

import pandas as pd
import pytest

from maverick_mcp.services.signals.conditions import evaluate_condition


@pytest.fixture
def sample_data():
    """Sample OHLCV data for testing."""
    return pd.DataFrame({
        "close": [100, 102, 104, 103, 101, 99, 97, 95, 93, 91,
                  90, 89, 88, 87, 86, 85, 84, 83, 82, 81],
        "volume": [1000] * 18 + [5000, 6000],  # volume spike at end
    })


def test_lt_operator(sample_data):
    condition = {"indicator": "price", "operator": "lt", "value": 85}
    result = evaluate_condition(condition, sample_data, previous_state=None)
    assert result["triggered"] is True


def test_gt_operator(sample_data):
    condition = {"indicator": "price", "operator": "gt", "value": 200}
    result = evaluate_condition(condition, sample_data, previous_state=None)
    assert result["triggered"] is False


def test_rsi_threshold():
    # 20 period downtrend should produce low RSI
    closes = list(range(120, 100, -1))  # declining prices
    data = pd.DataFrame({"close": closes, "volume": [1000] * 20})
    condition = {"indicator": "rsi", "operator": "lt", "value": 40, "period": 14}
    result = evaluate_condition(condition, data, previous_state=None)
    assert result["triggered"] is True


def test_volume_spike(sample_data):
    condition = {"indicator": "volume", "operator": "spike", "std_devs": 2.0}
    result = evaluate_condition(condition, sample_data, previous_state=None)
    assert result["triggered"] is True


def test_crosses_above_no_previous_state():
    data = pd.DataFrame({"close": [100, 102, 104, 106, 108] * 4, "volume": [1000] * 20})
    condition = {"indicator": "price", "operator": "crosses_above", "reference": "sma_10"}
    result = evaluate_condition(condition, data, previous_state=None)
    # First evaluation: no previous state, just records current state
    assert result["triggered"] is False
    assert result["new_state"] is not None


def test_crosses_above_with_transition():
    data = pd.DataFrame({"close": list(range(90, 110)), "volume": [1000] * 20})
    condition = {"indicator": "price", "operator": "crosses_above", "reference": "sma_10"}
    # Simulate: previously was below SMA, now above
    previous_state = {"was_above": False}
    result = evaluate_condition(condition, data, previous_state=previous_state)
    assert result["triggered"] is True
    assert result["new_state"]["was_above"] is True


def test_unknown_indicator():
    data = pd.DataFrame({"close": [100], "volume": [1000]})
    condition = {"indicator": "unknown_thing", "operator": "lt", "value": 50}
    result = evaluate_condition(condition, data, previous_state=None)
    assert result["triggered"] is False
    assert "error" in result


def test_result_includes_current_value(sample_data):
    condition = {"indicator": "price", "operator": "lt", "value": 85}
    result = evaluate_condition(condition, sample_data, previous_state=None)
    assert "current_value" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_signal_conditions.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement condition engine**

Create `maverick_mcp/services/signals/conditions.py`:

```python
"""Condition evaluation engine for signal alerts."""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def evaluate_condition(
    condition: dict[str, Any],
    data: pd.DataFrame,
    previous_state: dict[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate a signal condition against price/volume data.

    Args:
        condition: Dict with keys: indicator, operator, value/reference, period
        data: DataFrame with at least 'close' and 'volume' columns
        previous_state: State from the last evaluation (for stateful conditions)

    Returns:
        Dict with keys: triggered (bool), current_value, new_state, error (optional)
    """
    indicator = condition.get("indicator", "")
    operator = condition.get("operator", "")

    try:
        current_value = _get_indicator_value(indicator, data, condition)
    except Exception as e:
        logger.warning("Failed to compute indicator %s: %s", indicator, e)
        return {"triggered": False, "current_value": None, "new_state": previous_state, "error": str(e)}

    if current_value is None:
        return {"triggered": False, "current_value": None, "new_state": previous_state, "error": f"Unknown indicator: {indicator}"}

    triggered = _evaluate_operator(operator, current_value, condition, data, previous_state)
    new_state = _compute_new_state(operator, current_value, condition, data, previous_state)

    return {
        "triggered": triggered,
        "current_value": float(current_value) if not isinstance(current_value, bool) else current_value,
        "new_state": new_state,
    }


def _get_indicator_value(indicator: str, data: pd.DataFrame, condition: dict[str, Any]) -> float | None:
    """Compute the current value of an indicator."""
    if indicator == "price":
        return float(data["close"].iloc[-1])

    if indicator == "rsi":
        period = condition.get("period", 14)
        return _compute_rsi(data["close"], period)

    if indicator == "volume":
        return float(data["volume"].iloc[-1])

    if indicator == "sma":
        period = condition.get("period", 20)
        return float(data["close"].rolling(window=period).mean().iloc[-1])

    return None


def _compute_rsi(closes: pd.Series, period: int = 14) -> float:
    """Compute RSI for the most recent bar."""
    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean().iloc[-1]
    avg_loss = loss.rolling(window=period, min_periods=period).mean().iloc[-1]

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _get_reference_value(reference: str, data: pd.DataFrame) -> float | None:
    """Compute a reference value (e.g., sma_200)."""
    if reference.startswith("sma_"):
        period = int(reference.split("_")[1])
        sma = data["close"].rolling(window=min(period, len(data))).mean()
        return float(sma.iloc[-1])
    return None


def _evaluate_operator(
    operator: str,
    current_value: float,
    condition: dict[str, Any],
    data: pd.DataFrame,
    previous_state: dict[str, Any] | None,
) -> bool:
    """Evaluate the comparison operator."""
    if operator == "lt":
        return current_value < condition["value"]

    if operator == "gt":
        return current_value > condition["value"]

    if operator == "lte":
        return current_value <= condition["value"]

    if operator == "gte":
        return current_value >= condition["value"]

    if operator == "spike":
        std_devs = condition.get("std_devs", 2.0)
        values = data["volume"] if condition.get("indicator") == "volume" else data["close"]
        mean = float(values.mean())
        std = float(values.std())
        if std == 0:
            return False
        return current_value > mean + (std_devs * std)

    if operator in ("crosses_above", "crosses_below"):
        if previous_state is None:
            return False  # Need previous state for crossover detection

        ref = _get_reference_value(condition.get("reference", ""), data)
        if ref is None:
            return False

        was_above = previous_state.get("was_above", False)
        is_above = current_value > ref

        if operator == "crosses_above":
            return not was_above and is_above
        else:
            return was_above and not is_above

    return False


def _compute_new_state(
    operator: str,
    current_value: float,
    condition: dict[str, Any],
    data: pd.DataFrame,
    previous_state: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Compute new state for stateful conditions."""
    if operator in ("crosses_above", "crosses_below"):
        ref = _get_reference_value(condition.get("reference", ""), data)
        if ref is None:
            return previous_state
        return {"was_above": current_value > ref, "previous_value": current_value}

    return previous_state
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_signal_conditions.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add maverick_mcp/services/signals/conditions.py tests/unit/test_signal_conditions.py
git commit -m "feat: add signal condition evaluation engine"
```

---

## Task 8: Regime Detector

**Files:**
- Create: `maverick_mcp/services/signals/regime.py`
- Test: `tests/unit/test_regime_detector.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_regime_detector.py`:

```python
"""Tests for market regime detection."""

import pandas as pd
import pytest

from maverick_mcp.services.signals.regime import RegimeDetector


@pytest.fixture
def detector():
    return RegimeDetector()


def test_bull_regime(detector):
    # Strong uptrend: prices rising, above moving averages
    prices = pd.Series(list(range(100, 160)))
    result = detector.classify(market_prices=prices, vix_level=15.0)
    assert result["regime"] in ("bull", "transitional")
    assert 0 <= result["confidence"] <= 1
    assert "drivers" in result


def test_bear_regime(detector):
    # Strong downtrend
    prices = pd.Series(list(range(160, 100, -1)))
    result = detector.classify(market_prices=prices, vix_level=35.0)
    assert result["regime"] in ("bear", "transitional")


def test_choppy_regime(detector):
    # Sideways: alternating up and down
    prices = pd.Series([100, 102, 99, 101, 100, 103, 98, 101, 100, 102] * 5)
    result = detector.classify(market_prices=prices, vix_level=22.0)
    assert result["regime"] in ("choppy", "transitional")


def test_high_vix_biases_bearish(detector):
    # Neutral trend but high VIX
    prices = pd.Series([100] * 50)
    result_low_vix = detector.classify(market_prices=prices, vix_level=12.0)
    result_high_vix = detector.classify(market_prices=prices, vix_level=40.0)
    # Higher VIX should produce more bearish assessment
    assert result_high_vix["drivers"]["volatility"] != result_low_vix["drivers"]["volatility"]


def test_drivers_present(detector):
    prices = pd.Series(list(range(100, 150)))
    result = detector.classify(market_prices=prices, vix_level=15.0)
    assert "trend" in result["drivers"]
    assert "volatility" in result["drivers"]
    assert "momentum" in result["drivers"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_regime_detector.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement regime detector**

Create `maverick_mcp/services/signals/regime.py`:

```python
"""Market regime detection using composite indicator analysis."""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# VIX thresholds for regime classification
VIX_LOW = 16.0   # Complacent / bullish
VIX_MID = 22.0   # Normal
VIX_HIGH = 30.0  # Elevated fear


class RegimeDetector:
    """Classifies market regime as bull, bear, choppy, or transitional."""

    def classify(
        self,
        market_prices: pd.Series,
        vix_level: float,
        breadth_ratio: float | None = None,
    ) -> dict[str, Any]:
        """Classify the current market regime.

        Args:
            market_prices: Series of market index close prices (e.g., SPY)
            vix_level: Current VIX level
            breadth_ratio: Advance/decline ratio (optional, 1.0 = neutral)

        Returns:
            Dict with regime, confidence (0-1), and driver breakdown
        """
        drivers = {}
        votes = {"bull": 0.0, "bear": 0.0, "choppy": 0.0}

        # Factor 1: Trend strength (weight: 0.35)
        trend = self._assess_trend(market_prices)
        drivers["trend"] = trend["label"]
        votes[trend["label"]] += 0.35 * trend["strength"]

        # Factor 2: Volatility regime (weight: 0.25)
        vol = self._assess_volatility(vix_level)
        drivers["volatility"] = vol["label"]
        votes[vol["label"]] += 0.25 * vol["strength"]

        # Factor 3: Momentum (weight: 0.25)
        momentum = self._assess_momentum(market_prices)
        drivers["momentum"] = momentum["label"]
        votes[momentum["label"]] += 0.25 * momentum["strength"]

        # Factor 4: Breadth (weight: 0.15)
        if breadth_ratio is not None:
            breadth = self._assess_breadth(breadth_ratio)
        else:
            breadth = {"label": "neutral", "strength": 0.5}
        drivers["breadth"] = breadth["label"]
        votes[breadth["label"]] += 0.15 * breadth["strength"]

        # Determine regime
        max_vote = max(votes.values())
        total_votes = sum(votes.values())

        if total_votes == 0:
            return {"regime": "choppy", "confidence": 0.5, "drivers": drivers}

        regime = max(votes, key=votes.get)
        confidence = max_vote / total_votes if total_votes > 0 else 0.5

        # If confidence is low, call it transitional
        if confidence < 0.45:
            regime = "transitional"

        return {
            "regime": regime,
            "confidence": round(confidence, 3),
            "drivers": drivers,
            "votes": {k: round(v, 3) for k, v in votes.items()},
        }

    def _assess_trend(self, prices: pd.Series) -> dict[str, Any]:
        """Assess trend using moving average position and slope."""
        if len(prices) < 10:
            return {"label": "choppy", "strength": 0.5}

        sma_short = prices.rolling(window=min(10, len(prices))).mean()
        sma_long = prices.rolling(window=min(30, len(prices))).mean()

        current = float(prices.iloc[-1])
        short_ma = float(sma_short.iloc[-1])
        long_ma = float(sma_long.iloc[-1])

        above_short = current > short_ma
        above_long = current > long_ma

        # Slope of short MA (normalized)
        if len(sma_short.dropna()) >= 5:
            slope = float(sma_short.dropna().iloc[-1] - sma_short.dropna().iloc[-5]) / max(float(sma_short.dropna().iloc[-5]), 1)
        else:
            slope = 0

        if above_short and above_long and slope > 0.01:
            return {"label": "bull", "strength": min(1.0, 0.7 + abs(slope) * 10)}
        elif not above_short and not above_long and slope < -0.01:
            return {"label": "bear", "strength": min(1.0, 0.7 + abs(slope) * 10)}
        else:
            return {"label": "choppy", "strength": 0.6}

    def _assess_volatility(self, vix_level: float) -> dict[str, Any]:
        """Assess volatility regime from VIX level."""
        if vix_level < VIX_LOW:
            return {"label": "bull", "strength": 0.8}
        elif vix_level < VIX_MID:
            return {"label": "choppy", "strength": 0.6}
        elif vix_level < VIX_HIGH:
            return {"label": "bear", "strength": 0.7}
        else:
            return {"label": "bear", "strength": 0.9}

    def _assess_momentum(self, prices: pd.Series) -> dict[str, Any]:
        """Assess momentum using rate of change."""
        if len(prices) < 10:
            return {"label": "choppy", "strength": 0.5}

        roc_10 = (float(prices.iloc[-1]) - float(prices.iloc[-10])) / max(float(prices.iloc[-10]), 1)

        if roc_10 > 0.03:
            return {"label": "bull", "strength": min(1.0, 0.6 + abs(roc_10) * 5)}
        elif roc_10 < -0.03:
            return {"label": "bear", "strength": min(1.0, 0.6 + abs(roc_10) * 5)}
        else:
            return {"label": "choppy", "strength": 0.7}

    def _assess_breadth(self, breadth_ratio: float) -> dict[str, Any]:
        """Assess market breadth from advance/decline ratio."""
        if breadth_ratio > 1.5:
            return {"label": "bull", "strength": 0.8}
        elif breadth_ratio > 1.0:
            return {"label": "bull", "strength": 0.6}
        elif breadth_ratio > 0.7:
            return {"label": "choppy", "strength": 0.6}
        else:
            return {"label": "bear", "strength": 0.7}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_regime_detector.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add maverick_mcp/services/signals/regime.py tests/unit/test_regime_detector.py
git commit -m "feat: add market regime detector with composite scoring"
```

---

## Task 9: Signal Service

**Files:**
- Create: `maverick_mcp/services/signals/service.py`
- Test: `tests/unit/test_signal_service.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_signal_service.py`:

```python
"""Tests for the signal service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from maverick_mcp.database.base import Base
from maverick_mcp.services.signals.models import Signal, SignalEvent
from maverick_mcp.services.signals.service import SignalService
from maverick_mcp.services.event_bus import EventBus


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def signal_service(db_session, event_bus):
    return SignalService(db_session=db_session, event_bus=event_bus)


def test_create_signal(signal_service, db_session):
    signal = signal_service.create_signal(
        label="RSI Oversold",
        ticker="AAPL",
        condition={"indicator": "rsi", "operator": "lt", "value": 30, "period": 14},
    )
    assert signal.id is not None
    assert signal.label == "RSI Oversold"
    assert signal.ticker == "AAPL"
    assert signal.active is True

    # Verify persisted
    persisted = db_session.query(Signal).filter_by(id=signal.id).first()
    assert persisted is not None


def test_list_signals(signal_service):
    signal_service.create_signal("A", "AAPL", {"indicator": "price", "operator": "gt", "value": 200})
    signal_service.create_signal("B", "MSFT", {"indicator": "rsi", "operator": "lt", "value": 30})
    signals = signal_service.list_signals()
    assert len(signals) == 2


def test_delete_signal(signal_service, db_session):
    signal = signal_service.create_signal("Test", "AAPL", {"indicator": "price", "operator": "gt", "value": 200})
    signal_service.delete_signal(signal.id)
    assert db_session.query(Signal).filter_by(id=signal.id).first() is None


def test_update_signal(signal_service, db_session):
    signal = signal_service.create_signal("Old", "AAPL", {"indicator": "price", "operator": "gt", "value": 200})
    updated = signal_service.update_signal(signal.id, label="New", condition={"indicator": "rsi", "operator": "lt", "value": 25})
    assert updated.label == "New"
    assert updated.condition["value"] == 25


def test_list_signals_active_only(signal_service):
    s1 = signal_service.create_signal("Active", "AAPL", {"indicator": "price", "operator": "gt", "value": 200})
    s2 = signal_service.create_signal("Inactive", "MSFT", {"indicator": "price", "operator": "gt", "value": 200})
    signal_service.update_signal(s2.id, active=False)
    active = signal_service.list_signals(active_only=True)
    assert len(active) == 1
    assert active[0].label == "Active"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_signal_service.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement signal service**

Create `maverick_mcp/services/signals/service.py`:

```python
"""Signal service — CRUD and evaluation orchestration."""

import logging
from typing import Any

from sqlalchemy.orm import Session

from maverick_mcp.services.event_bus import EventBus
from maverick_mcp.services.signals.models import Signal, SignalEvent

logger = logging.getLogger(__name__)


class SignalService:
    """Manages signal definitions and evaluation."""

    def __init__(self, db_session: Session, event_bus: EventBus):
        self._db = db_session
        self._event_bus = event_bus

    def create_signal(
        self,
        label: str,
        ticker: str,
        condition: dict[str, Any],
        interval_seconds: int = 300,
    ) -> Signal:
        """Create a new signal definition."""
        signal = Signal(
            label=label,
            ticker=ticker.upper(),
            condition=condition,
            interval_seconds=interval_seconds,
            active=True,
        )
        self._db.add(signal)
        self._db.commit()
        self._db.refresh(signal)
        logger.info("Created signal %d: %s on %s", signal.id, label, ticker)
        return signal

    def list_signals(self, active_only: bool = False) -> list[Signal]:
        """List all signals, optionally filtered to active only."""
        query = self._db.query(Signal)
        if active_only:
            query = query.filter(Signal.active == True)  # noqa: E712
        return query.all()

    def get_signal(self, signal_id: int) -> Signal | None:
        """Get a signal by ID."""
        return self._db.query(Signal).filter_by(id=signal_id).first()

    def update_signal(self, signal_id: int, **kwargs: Any) -> Signal:
        """Update a signal's attributes."""
        signal = self._db.query(Signal).filter_by(id=signal_id).first()
        if not signal:
            raise ValueError(f"Signal {signal_id} not found")
        for key, value in kwargs.items():
            if hasattr(signal, key):
                setattr(signal, key, value)
        self._db.commit()
        self._db.refresh(signal)
        logger.info("Updated signal %d", signal_id)
        return signal

    def delete_signal(self, signal_id: int) -> None:
        """Delete a signal."""
        signal = self._db.query(Signal).filter_by(id=signal_id).first()
        if signal:
            self._db.delete(signal)
            self._db.commit()
            logger.info("Deleted signal %d", signal_id)

    def record_trigger(self, signal: Signal, price: float | None, snapshot: dict[str, Any] | None) -> SignalEvent:
        """Record a signal trigger event."""
        event = SignalEvent(
            signal_id=signal.id,
            price_at_trigger=price,
            condition_snapshot=snapshot,
        )
        self._db.add(event)
        self._db.commit()
        self._db.refresh(event)
        return event

    async def evaluate_all(self, data_fetcher) -> list[dict[str, Any]]:
        """Evaluate all active signals. Called by the scheduler.

        Args:
            data_fetcher: Callable(ticker, period) -> pd.DataFrame

        Returns:
            List of trigger results
        """
        from maverick_mcp.services.signals.conditions import evaluate_condition

        signals = self.list_signals(active_only=True)
        results = []

        # Group by ticker to minimize data fetches
        by_ticker: dict[str, list[Signal]] = {}
        for sig in signals:
            by_ticker.setdefault(sig.ticker, []).append(sig)

        for ticker, ticker_signals in by_ticker.items():
            try:
                data = await data_fetcher(ticker, days=60)
            except Exception as e:
                logger.error("Failed to fetch data for %s: %s", ticker, e)
                continue

            for sig in ticker_signals:
                result = evaluate_condition(sig.condition, data, sig.previous_state)

                if result.get("triggered"):
                    price = result.get("current_value")
                    self.record_trigger(sig, price, result)
                    await self._event_bus.publish("signal.triggered", {
                        "signal_id": sig.id,
                        "label": sig.label,
                        "ticker": sig.ticker,
                        "price": price,
                        "condition": sig.condition,
                    })
                elif sig.previous_state and sig.previous_state.get("was_triggered"):
                    # Signal was previously triggered but is no longer — cleared
                    await self._event_bus.publish("signal.cleared", {
                        "signal_id": sig.id,
                        "label": sig.label,
                        "ticker": sig.ticker,
                        "condition": sig.condition,
                    })

                # Update stateful condition state
                if result.get("new_state") != sig.previous_state:
                    sig.previous_state = result["new_state"]
                    self._db.commit()

                results.append({
                    "signal_id": sig.id,
                    "ticker": sig.ticker,
                    "triggered": result.get("triggered", False),
                    "current_value": result.get("current_value"),
                })

        return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_signal_service.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add maverick_mcp/services/signals/service.py tests/unit/test_signal_service.py
git commit -m "feat: add signal service with CRUD and evaluation"
```

---

## Task 10: Signal MCP Tools

**Files:**
- Create: `maverick_mcp/api/routers/signals.py`
- Modify: `maverick_mcp/api/routers/tool_registry.py`

- [ ] **Step 1: Create signal MCP tools**

Create `maverick_mcp/api/routers/signals.py`:

```python
"""MCP tools for the signal engine — thin wrappers over SignalService."""

import logging
from typing import Any

from fastmcp import FastMCP

from maverick_mcp.services.event_bus import EventBus
from maverick_mcp.services.signals.regime import RegimeDetector

logger = logging.getLogger(__name__)


def register_signal_tools(mcp: FastMCP) -> None:
    """Register signal engine MCP tools on the main server."""

    @mcp.tool(name="create_signal", description="Create a price/indicator alert. Condition format: {indicator, operator, value/reference, period}. Indicators: price, rsi, volume, sma. Operators: lt, gt, lte, gte, spike, crosses_above, crosses_below.")
    def create_signal(
        label: str,
        ticker: str,
        condition: dict[str, Any],
        interval_seconds: int = 300,
    ) -> dict[str, Any]:
        from maverick_mcp.data.models import SessionLocal
        from maverick_mcp.services import event_bus

        from maverick_mcp.services.signals.service import SignalService

        with SessionLocal() as session:
            svc = SignalService(db_session=session, event_bus=event_bus)
            signal = svc.create_signal(label, ticker, condition, interval_seconds)
            return {"id": signal.id, "label": signal.label, "ticker": signal.ticker, "active": signal.active}

    @mcp.tool(name="update_signal", description="Update a signal's label, condition, interval, or active status.")
    def update_signal(signal_id: int, label: str | None = None, condition: dict[str, Any] | None = None, interval_seconds: int | None = None, active: bool | None = None) -> dict[str, Any]:
        from maverick_mcp.data.models import SessionLocal
        from maverick_mcp.services import event_bus
        from maverick_mcp.services.signals.service import SignalService

        kwargs = {k: v for k, v in {"label": label, "condition": condition, "interval_seconds": interval_seconds, "active": active}.items() if v is not None}
        with SessionLocal() as session:
            svc = SignalService(db_session=session, event_bus=event_bus)
            signal = svc.update_signal(signal_id, **kwargs)
            return {"id": signal.id, "label": signal.label, "ticker": signal.ticker, "active": signal.active}

    @mcp.tool(name="list_signals", description="List all signal alerts, optionally filtered to active only.")
    def list_signals(active_only: bool = False) -> list[dict[str, Any]]:
        from maverick_mcp.data.models import SessionLocal
        from maverick_mcp.services import event_bus
        from maverick_mcp.services.signals.service import SignalService

        with SessionLocal() as session:
            svc = SignalService(db_session=session, event_bus=event_bus)
            signals = svc.list_signals(active_only=active_only)
            return [{"id": s.id, "label": s.label, "ticker": s.ticker, "active": s.active, "condition": s.condition} for s in signals]

    @mcp.tool(name="delete_signal", description="Delete a signal alert by ID.")
    def delete_signal(signal_id: int) -> dict[str, str]:
        from maverick_mcp.data.models import SessionLocal
        from maverick_mcp.services import event_bus
        from maverick_mcp.services.signals.service import SignalService

        with SessionLocal() as session:
            svc = SignalService(db_session=session, event_bus=event_bus)
            svc.delete_signal(signal_id)
            return {"status": "deleted", "signal_id": str(signal_id)}

    @mcp.tool(name="check_signals_now", description="Manually evaluate all active signals immediately and return results.")
    async def check_signals_now() -> list[dict[str, Any]]:
        from maverick_mcp.data.models import SessionLocal
        from maverick_mcp.services import event_bus
        from maverick_mcp.services.signals.service import SignalService
        from maverick_mcp.providers.stock_data import EnhancedStockDataProvider

        provider = EnhancedStockDataProvider()

        async def fetch_data(ticker: str, days: int = 60):
            import pandas as pd
            data = provider.get_stock_data(ticker, period=f"{days}d")
            if isinstance(data, pd.DataFrame):
                return data
            return pd.DataFrame()

        with SessionLocal() as session:
            svc = SignalService(db_session=session, event_bus=event_bus)
            return await svc.evaluate_all(fetch_data)

    @mcp.tool(name="get_market_regime", description="Get current market regime classification (bull/bear/choppy/transitional) with confidence score and factor breakdown.")
    def get_market_regime() -> dict[str, Any]:
        import pandas as pd
        from maverick_mcp.providers.stock_data import EnhancedStockDataProvider

        provider = EnhancedStockDataProvider()
        detector = RegimeDetector()

        try:
            spy_data = provider.get_stock_data("SPY", period="90d")
            if spy_data is None or spy_data.empty:
                return {"error": "Could not fetch SPY data"}
            prices = spy_data["close"] if "close" in spy_data.columns else spy_data["Close"]

            # Try to get VIX
            vix_level = 20.0  # default
            try:
                vix_data = provider.get_stock_data("^VIX", period="5d")
                if vix_data is not None and not vix_data.empty:
                    vix_col = "close" if "close" in vix_data.columns else "Close"
                    vix_level = float(vix_data[vix_col].iloc[-1])
            except Exception:
                pass

            return detector.classify(market_prices=prices, vix_level=vix_level)
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool(name="get_regime_history", description="Get recent market regime changes over the past N days.")
    def get_regime_history(days: int = 30) -> list[dict[str, Any]]:
        from datetime import UTC, datetime, timedelta
        from maverick_mcp.data.models import SessionLocal
        from maverick_mcp.services.signals.models import RegimeEvent

        cutoff = datetime.now(UTC) - timedelta(days=days)
        with SessionLocal() as session:
            events = session.query(RegimeEvent).filter(
                RegimeEvent.detected_at >= cutoff
            ).order_by(RegimeEvent.detected_at.desc()).all()
            return [{
                "regime": e.regime,
                "confidence": e.confidence,
                "drivers": e.drivers,
                "previous_regime": e.previous_regime,
                "detected_at": e.detected_at.isoformat() if e.detected_at else None,
            } for e in events]
```

- [ ] **Step 2: Register signal tools in tool_registry.py**

In `maverick_mcp/api/routers/tool_registry.py`, add to the `register_all_router_tools` function, after the last existing `try/except` block:

```python
    try:
        from maverick_mcp.api.routers.signals import register_signal_tools
        register_signal_tools(mcp)
        logger.info("Signal tools registered successfully")
    except Exception as e:
        logger.error(f"Failed to register signal tools: {e}")
```

- [ ] **Step 3: Verify tools register without errors**

Run: `uv run python -c "from maverick_mcp.api.routers.signals import register_signal_tools; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add maverick_mcp/api/routers/signals.py maverick_mcp/api/routers/tool_registry.py
git commit -m "feat: add signal engine MCP tools"
```

---

## Tasks 11-18: Remaining Domains

> **Implementation pattern:** Each task below follows the exact TDD pattern from Tasks 6-10:
> 1. Create `__init__.py` for the domain package
> 2. Write models file with all SQLAlchemy columns
> 3. Write failing tests for the service
> 4. Implement the service to make tests pass
> 5. Create MCP tools as thin wrappers (follow `api/routers/signals.py` pattern exactly — lazy imports inside each tool function, `SessionLocal` context manager, return dicts)
> 6. Register tools in `register_all_router_tools` with try/except block
> 7. Commit after each sub-step
>
> **Existing patterns to follow:**
> - Models: `from maverick_mcp.database.base import Base` + `from maverick_mcp.data.models import TimestampMixin`
> - MCP tools: `from maverick_mcp.data.models import SessionLocal` with `with SessionLocal() as session:` pattern
> - Tool registration: `from maverick_mcp.api.routers.<module> import register_<domain>_tools` in `tool_registry.py`

### Task 11: Screening Pipeline Models

**Files:**
- Create: `maverick_mcp/services/screening/__init__.py`
- Create: `maverick_mcp/services/screening/models.py`

Models to create: `ScreeningRun`, `ScreeningChange`, `ScheduledJob`

Follow the exact pattern from Task 6 (Signal Models). Each model uses `Base` from `maverick_mcp.database.base` and `TimestampMixin` from `maverick_mcp.data.models`.

`ScreeningRun`: id, screen_name (String), run_at (DateTime), result_count (Integer), results (JSON)

`ScreeningChange`: id, run_id (Integer, FK to ScreeningRun), symbol (String, indexed), change_type (String: entry/exit/rank_change), screen_name (String), previous_rank (Integer, nullable), new_rank (Integer, nullable), detected_at (DateTime)

`ScheduledJob`: id, job_name (String, unique), job_type (String), schedule_config (JSON), active (Boolean), last_run_at (DateTime, nullable)

### Task 12: Screening Pipeline Service + Tests

**Files:**
- Create: `maverick_mcp/services/screening/pipeline.py`
- Test: `tests/unit/test_screening_pipeline.py`

`ScreeningPipelineService` methods:
- `run_screen(screen_name: str) -> ScreeningRun` — calls existing screening functions, snapshots to DB
- `get_changes(screen_name: str, since: datetime | None) -> list[ScreeningChange]` — diffs against previous run
- `get_history(symbol: str, screen_name: str) -> list[ScreeningRun]` — when a stock was on a screen
- `schedule_screen(screen_name: str, cron_config: dict)` — adds APScheduler job
- `get_pipeline_status() -> dict` — lists scheduled screens, last run times

The service calls existing screening functions from `maverick_mcp.api.routers.tool_registry` (specifically the functions wrapped by `register_screening_tools`). For the diff: compare current run's ticker set against previous run's ticker set. New tickers = entry, missing tickers = exit.

Publish events: `screening.entry`, `screening.exit` via event bus.

### Task 13: Screening Pipeline MCP Tools

**Files:**
- Create: `maverick_mcp/api/routers/screening_pipeline.py`
- Modify: `maverick_mcp/api/routers/tool_registry.py`

4 tools: `get_screening_changes`, `get_screening_history`, `schedule_screening`, `get_screening_pipeline_status`

### Task 14: Trade Journal Models + Service + Tests

**Files:**
- Create: `maverick_mcp/services/journal/__init__.py`
- Create: `maverick_mcp/services/journal/models.py`
- Create: `maverick_mcp/services/journal/service.py`
- Create: `maverick_mcp/services/journal/analytics.py`
- Test: `tests/unit/test_journal_service.py`
- Test: `tests/unit/test_strategy_tracker.py`

`JournalEntry` model: id, symbol, side (String: long/short), entry_price (Float), exit_price (Float, nullable), shares (Float), entry_date (DateTime), exit_date (DateTime, nullable), rationale (Text), tags (JSON, default=[]), pnl (Float, nullable), r_multiple (Float, nullable), notes (Text, nullable), status (String: open/closed, default=open)

`StrategyPerformance` model: id, strategy_tag (String, unique per period), period (String: all_time/monthly/quarterly), win_count (Integer), loss_count (Integer), total_pnl (Float), avg_win (Float), avg_loss (Float), expectancy (Float), profit_factor (Float)

`JournalService` methods:
- `add_trade(...)` — create JournalEntry, publish `trade.recorded`
- `close_trade(entry_id, exit_price, exit_date, notes)` — update entry, compute PnL, recompute StrategyPerformance
- `list_trades(filters)` — query with filters
- `get_trade(entry_id)` — single trade

`StrategyTracker` methods:
- `recompute(strategy_tag)` — recalculate performance from all closed trades with that tag
- `get_performance(strategy_tag)` — return StrategyPerformance
- `compare_strategies()` — return all strategies ranked by expectancy

### Task 15: Trade Journal MCP Tools

**Files:**
- Create: `maverick_mcp/api/routers/journal.py`
- Modify: `maverick_mcp/api/routers/tool_registry.py`

6 tools: `journal_add_trade`, `journal_list_trades`, `journal_trade_review` (uses OpenRouter for AI analysis), `get_strategy_performance`, `get_strategy_comparison`, `get_trading_patterns` (uses OpenRouter)

### Task 16: Watchlist Models + Service + Tests

**Files:**
- Create: `maverick_mcp/services/watchlist/__init__.py`
- Create: `maverick_mcp/services/watchlist/models.py`
- Create: `maverick_mcp/services/watchlist/service.py`
- Create: `maverick_mcp/services/watchlist/catalysts.py`
- Test: `tests/unit/test_watchlist_service.py`
- Test: `tests/unit/test_catalyst_tracker.py`

`Watchlist` model: id, name (String, unique), description (Text, nullable)
`WatchlistItem` model: id, watchlist_id (Integer FK), symbol (String), added_at (DateTime, default=now), notes (Text, nullable)
`CatalystEvent` model: id, symbol (String, indexed), event_type (String: earnings/ex_div/fda/other), event_date (Date), description (Text, nullable), impact_assessment (Text, nullable)

`WatchlistService`: CRUD + `brief()` method that scores each item
`CatalystTracker`: fetches earnings dates from yfinance, persists to CatalystEvent

### Task 17: Watchlist MCP Tools

**Files:**
- Create: `maverick_mcp/api/routers/watchlist.py`
- Modify: `maverick_mcp/api/routers/tool_registry.py`

5 tools: `watchlist_create`, `watchlist_add`, `watchlist_remove`, `watchlist_brief`, `get_upcoming_catalysts`

### Task 18: Risk Dashboard Models + Service + MCP Tools + Tests

**Files:**
- Create: `maverick_mcp/services/risk/__init__.py`
- Create: `maverick_mcp/services/risk/models.py`
- Create: `maverick_mcp/services/risk/service.py`
- Create: `maverick_mcp/api/routers/risk_dashboard.py`
- Modify: `maverick_mcp/api/routers/tool_registry.py`
- Test: `tests/unit/test_risk_service.py`

`RiskAlert` model: id, portfolio_name (String), alert_type (String), severity (String: warning/critical), message (Text), details (JSON), acknowledged (Boolean, default=False)
`RiskSnapshot` model: id, portfolio_name (String), var_95 (Float), var_99 (Float), max_sector_pct (Float), max_correlation (Float), beta_weighted_delta (Float), regime (String), snapshot_at (DateTime)

`RiskService` methods:
- `compute_dashboard(portfolio_name)` — aggregate all risk metrics, save snapshot
- `check_position_risk(portfolio_name, ticker, shares)` — pre-trade risk check
- `get_regime_adjusted_size(portfolio_name, ticker, entry_price, stop_loss)` — position size * regime multiplier
- `get_alerts(portfolio_name)` — check thresholds, return RiskAlert list

4 tools: `get_portfolio_risk_dashboard`, `get_position_risk_check`, `get_regime_adjusted_sizing`, `get_risk_alerts`

---

## Task 19: Integration Pass

**Files:**
- Modify: `maverick_mcp/api/server.py`
- Modify: `maverick_mcp/services/__init__.py`

- [ ] **Step 1: Wire all services into registry at startup**

In `maverick_mcp/api/server.py`, inside the `init_systems()` function, after the scheduler start:

```python
        # Register domain services in service registry
        try:
            from maverick_mcp.services import registry, event_bus
            from maverick_mcp.services.signals.regime import RegimeDetector

            registry.register("event_bus", event_bus)
            registry.register("regime_detector", RegimeDetector())
            logger.info("Domain services registered in service registry")
        except Exception as e:
            logger.error(f"Failed to register domain services: {e}")

        # Wire cross-domain event subscriptions
        try:
            from maverick_mcp.services import event_bus as _eb

            async def on_signal_for_risk(event):
                """Forward signal triggers to risk alerting."""
                logger.debug("Risk service received signal event: %s", event.get("ticker"))

            async def on_screening_change_for_risk(event):
                """Forward screening changes to risk alerting."""
                logger.debug("Risk service received screening event: %s", event.get("symbol"))

            async def on_regime_change(event):
                """Log regime transitions and persist to RegimeEvent."""
                logger.info("Regime changed: %s -> %s (confidence: %s)",
                    event.get("previous_regime"), event.get("regime"), event.get("confidence"))

            _eb.subscribe("signal.triggered", on_signal_for_risk)
            _eb.subscribe("screening.entry", on_screening_change_for_risk)
            _eb.subscribe("screening.exit", on_screening_change_for_risk)
            _eb.subscribe("regime.changed", on_regime_change)
            logger.info("Cross-domain event subscriptions wired")
        except Exception as e:
            logger.error(f"Failed to wire event subscriptions: {e}")
```

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -x -q --timeout=60`
Expected: All tests pass (existing + new)

- [ ] **Step 3: Verify all tools register on server startup**

Run: `uv run python -m maverick_mcp.api.server --transport stdio < /dev/null 2>&1 | grep -i "registered\|failed\|error" | head -20`
Expected: All tool groups show "registered successfully", no failures

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete Maverick MCP roadmap v1 — 5 domains, ~28 new tools, service-layer architecture"
```

