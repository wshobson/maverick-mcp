"""Tests ensuring temporary event loops are restored correctly."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from maverick_mcp.api.server import health_resource, status_dashboard_resource
from maverick_mcp.backtesting.strategy_executor import (
    ExecutionContext,
    StrategyExecutor,
)
from maverick_mcp.utils.quick_cache import quick_cache


def _assert_loop_clean() -> None:
    """Assert that no closed event loop remains configured."""

    policy = asyncio.get_event_loop_policy()
    try:
        loop = policy.get_event_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        assert not loop.is_closed()
        asyncio.set_event_loop(None)


def test_health_resource_restores_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling the health resource twice should not leave a closed loop."""

    async def _stub_health() -> dict[str, Any]:
        return {"status": "healthy"}

    monkeypatch.setattr(
        "maverick_mcp.api.routers.health_enhanced._get_detailed_health_status",
        _stub_health,
    )

    result = health_resource()
    assert result["status"] == "healthy"
    second_result = health_resource()
    assert second_result["status"] == "healthy"

    _assert_loop_clean()


def test_status_dashboard_restores_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """The status dashboard resource should restore the previous loop."""

    async def _stub_dashboard() -> dict[str, Any]:
        return {"status": "ok"}

    monkeypatch.setattr(
        "maverick_mcp.monitoring.status_dashboard.get_dashboard_data",
        _stub_dashboard,
    )

    result = status_dashboard_resource()
    assert result["status"] == "ok"
    again = status_dashboard_resource()
    assert again["status"] == "ok"

    _assert_loop_clean()


def test_quick_cache_sync_wrapper_restores_loop() -> None:
    """Synchronous quick_cache wrapper should not leave a closed loop behind."""

    call_count = {"count": 0}

    @quick_cache(ttl_seconds=60)
    def _compute(value: int) -> int:
        call_count["count"] += 1
        return value * 2

    assert _compute(2) == 4
    assert _compute(2) == 4
    assert call_count["count"] == 1

    _assert_loop_clean()


def test_strategy_executor_sync_runner_restores_loop() -> None:
    """Running a backtest synchronously should restore the previous loop."""

    executor = StrategyExecutor(max_concurrent_strategies=1)

    class _DummyEngine:
        async def run_backtest(self, **_: Any) -> dict[str, Any]:
            return {"status": "ok"}

    context = ExecutionContext(
        strategy_id="test",
        symbol="AAPL",
        strategy_type="demo",
        parameters={},
        start_date="2024-01-01",
        end_date="2024-01-02",
    )

    engine = _DummyEngine()
    result = executor._run_backtest_sync(engine, context)
    assert result["status"] == "ok"

    _assert_loop_clean()

    executor._thread_pool.shutdown(wait=True)
