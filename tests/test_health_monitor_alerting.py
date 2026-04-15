"""Regression tests for health monitor alerting logic.

Covers two classes of bugs fixed together:

1. CPU alerts were measuring host-wide CPU via ``psutil.cpu_percent()``, which
   causes false positives on dev machines when unrelated apps spike. Alerts now
   fire on ``process_cpu_percent`` (this server's own CPU).

2. ``_handle_high_cpu_usage`` / ``_handle_high_memory_usage`` were
   docstring-only "sustained" — a single breach fired the alert. Alerts now
   require the breach to persist for
   ``ALERT_THRESHOLDS["high_cpu_duration"]`` /
   ``ALERT_THRESHOLDS["high_memory_duration"]`` before firing.

The disk branch is intentionally single-reading because disk fills slowly —
a single >90% observation is already actionable.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from maverick_mcp.api.routers.health_enhanced import ResourceUsage
from maverick_mcp.monitoring import health_monitor as hm_mod
from maverick_mcp.monitoring.health_monitor import ALERT_THRESHOLDS, HealthMonitor


def _make_usage(
    process_cpu: float = 1.0,
    host_cpu: float = 10.0,
    memory: float = 10.0,
    disk: float = 10.0,
) -> ResourceUsage:
    """Build a ResourceUsage whose fields default to benign values.

    Individual tests override only the dimension under test, so unrelated
    fields cannot accidentally trigger a different alert branch and create
    false-positive assertions.
    """
    return ResourceUsage(
        cpu_percent=host_cpu,
        process_cpu_percent=process_cpu,
        memory_percent=memory,
        disk_percent=disk,
        memory_used_mb=100.0,
        memory_total_mb=1000.0,
        disk_used_gb=10.0,
        disk_total_gb=100.0,
    )


@pytest.fixture
def monitor(monkeypatch: pytest.MonkeyPatch) -> HealthMonitor:
    """HealthMonitor with a controllable clock and stubbed resource reader."""
    m = HealthMonitor()
    # Controllable virtual clock for the sustained-duration gate.
    m._now = 0.0  # type: ignore[attr-defined]
    monkeypatch.setattr(hm_mod.time, "time", lambda: m._now)  # type: ignore[attr-defined]
    return m


@pytest.mark.asyncio
async def test_host_cpu_spike_does_not_alert(
    monitor: HealthMonitor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Host CPU at 100% with process CPU at 0% must not trigger an alert."""
    monkeypatch.setattr(
        "maverick_mcp.api.routers.health_enhanced._get_resource_usage",
        lambda: _make_usage(process_cpu=0.1, host_cpu=100.0),
    )
    # Run many checks spread across well past the duration threshold.
    for _ in range(5):
        await monitor._check_resource_usage()
        monitor._now += ALERT_THRESHOLDS["high_cpu_duration"]  # type: ignore[attr-defined]

    assert monitor._high_cpu_since is None
    assert "high_cpu" not in monitor.alerts_sent


@pytest.mark.asyncio
async def test_single_process_cpu_spike_does_not_alert(
    monitor: HealthMonitor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single process-CPU breach must start the timer but not fire an alert."""
    monkeypatch.setattr(
        "maverick_mcp.api.routers.health_enhanced._get_resource_usage",
        lambda: _make_usage(process_cpu=95.0),
    )
    await monitor._check_resource_usage()

    assert monitor._high_cpu_since == 0.0
    assert "high_cpu" not in monitor.alerts_sent


@pytest.mark.asyncio
async def test_sustained_process_cpu_fires_alert(
    monitor: HealthMonitor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Breach persisting past high_cpu_duration must fire exactly one alert."""
    monkeypatch.setattr(
        "maverick_mcp.api.routers.health_enhanced._get_resource_usage",
        lambda: _make_usage(process_cpu=95.0),
    )
    # First check starts the clock.
    await monitor._check_resource_usage()
    # Second check, just before threshold, still shouldn't alert.
    monitor._now = ALERT_THRESHOLDS["high_cpu_duration"] - 1  # type: ignore[attr-defined]
    await monitor._check_resource_usage()
    assert "high_cpu" not in monitor.alerts_sent

    # Cross the threshold.
    monitor._now = ALERT_THRESHOLDS["high_cpu_duration"] + 1  # type: ignore[attr-defined]
    await monitor._check_resource_usage()
    assert "high_cpu" in monitor.alerts_sent


@pytest.mark.asyncio
async def test_recovery_clears_breach_tracker(
    monitor: HealthMonitor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dropping below threshold must reset the tracker so the next spike re-starts the timer."""
    usage = {"process_cpu": 95.0}
    monkeypatch.setattr(
        "maverick_mcp.api.routers.health_enhanced._get_resource_usage",
        lambda: _make_usage(process_cpu=usage["process_cpu"]),
    )
    await monitor._check_resource_usage()
    assert monitor._high_cpu_since == 0.0

    usage["process_cpu"] = 10.0
    await monitor._check_resource_usage()
    assert monitor._high_cpu_since is None


# ─────────────────────────────────────────────────────────────────────────
# Memory alert branches
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_single_memory_breach_starts_timer_but_does_not_alert(
    monitor: HealthMonitor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single memory breach must start the sustained-duration timer, not fire."""
    monkeypatch.setattr(
        "maverick_mcp.api.routers.health_enhanced._get_resource_usage",
        lambda: _make_usage(memory=90.0),
    )
    handle = AsyncMock()
    monkeypatch.setattr(monitor, "_handle_high_memory_usage", handle)

    await monitor._check_resource_usage()

    assert monitor._high_memory_since == 0.0
    assert "high_memory" not in monitor.alerts_sent
    handle.assert_not_called()


@pytest.mark.asyncio
async def test_sustained_memory_breach_fires_alert(
    monitor: HealthMonitor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Memory breach persisting past high_memory_duration must call the handler."""
    monkeypatch.setattr(
        "maverick_mcp.api.routers.health_enhanced._get_resource_usage",
        lambda: _make_usage(memory=92.0),
    )
    handle = AsyncMock()
    monkeypatch.setattr(monitor, "_handle_high_memory_usage", handle)

    # First observation — timer starts.
    await monitor._check_resource_usage()
    assert monitor._high_memory_since == 0.0
    handle.assert_not_called()

    # Still inside the window — do not alert.
    monitor._now = ALERT_THRESHOLDS["high_memory_duration"] - 1  # type: ignore[attr-defined]
    await monitor._check_resource_usage()
    handle.assert_not_called()

    # Cross the threshold.
    monitor._now = ALERT_THRESHOLDS["high_memory_duration"] + 1  # type: ignore[attr-defined]
    await monitor._check_resource_usage()
    handle.assert_awaited_once()
    assert handle.await_args is not None
    # Must pass the current memory_percent, not the host CPU nor a stale value.
    (value_arg,) = handle.await_args.args
    assert value_arg == pytest.approx(92.0)


@pytest.mark.asyncio
async def test_memory_recovery_clears_tracker(
    monitor: HealthMonitor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dropping below the memory threshold resets the timer so the next breach re-starts."""
    state = {"memory": 90.0}
    monkeypatch.setattr(
        "maverick_mcp.api.routers.health_enhanced._get_resource_usage",
        lambda: _make_usage(memory=state["memory"]),
    )
    monkeypatch.setattr(monitor, "_handle_high_memory_usage", AsyncMock())

    await monitor._check_resource_usage()
    assert monitor._high_memory_since == 0.0

    state["memory"] = 20.0
    await monitor._check_resource_usage()
    assert monitor._high_memory_since is None


@pytest.mark.asyncio
async def test_host_memory_of_85_does_not_alert(
    monitor: HealthMonitor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The memory gate is strictly greater than 85; exactly 85 must not trigger."""
    monkeypatch.setattr(
        "maverick_mcp.api.routers.health_enhanced._get_resource_usage",
        lambda: _make_usage(memory=85.0),
    )
    handle = AsyncMock()
    monkeypatch.setattr(monitor, "_handle_high_memory_usage", handle)

    await monitor._check_resource_usage()

    assert monitor._high_memory_since is None
    handle.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────
# Disk alert branch (intentionally single-reading)
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_disk_above_90_fires_on_single_reading(
    monitor: HealthMonitor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disk fills slowly — a single >90% reading must call the handler immediately."""
    monkeypatch.setattr(
        "maverick_mcp.api.routers.health_enhanced._get_resource_usage",
        lambda: _make_usage(disk=92.5),
    )
    handle = AsyncMock()
    monkeypatch.setattr(monitor, "_handle_high_disk_usage", handle)

    await monitor._check_resource_usage()

    handle.assert_awaited_once()
    (value_arg,) = handle.await_args.args  # type: ignore[union-attr]
    assert value_arg == pytest.approx(92.5)


@pytest.mark.asyncio
async def test_disk_at_boundary_does_not_fire(
    monitor: HealthMonitor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disk gate is strictly >90, not >=; 90.0 exactly must not trigger."""
    monkeypatch.setattr(
        "maverick_mcp.api.routers.health_enhanced._get_resource_usage",
        lambda: _make_usage(disk=90.0),
    )
    handle = AsyncMock()
    monkeypatch.setattr(monitor, "_handle_high_disk_usage", handle)

    await monitor._check_resource_usage()

    handle.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────
# Multi-dimension interaction
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cpu_and_memory_tracked_independently(
    monitor: HealthMonitor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CPU and memory sustained-duration timers are independent.

    If CPU is breached but memory is fine, only the CPU timer advances; the
    memory tracker must stay ``None`` and vice-versa. Regresses a previous
    bug where a single shared ``_high_since`` conflated the two.
    """
    monkeypatch.setattr(
        "maverick_mcp.api.routers.health_enhanced._get_resource_usage",
        lambda: _make_usage(process_cpu=90.0, memory=10.0),
    )

    await monitor._check_resource_usage()

    assert monitor._high_cpu_since == 0.0
    assert monitor._high_memory_since is None


@pytest.mark.asyncio
async def test_resource_check_swallows_exceptions(
    monitor: HealthMonitor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_check_resource_usage`` logs and continues on failure — it must not crash the loop."""

    def _raising() -> ResourceUsage:
        raise RuntimeError("psutil exploded")

    monkeypatch.setattr(
        "maverick_mcp.api.routers.health_enhanced._get_resource_usage",
        _raising,
    )

    # Must not raise.
    await monitor._check_resource_usage()
    assert monitor._high_cpu_since is None
    assert monitor._high_memory_since is None
