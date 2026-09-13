"""Breaker recovery must survive cancellation and stale in-flight completions."""

import asyncio

import pytest

from maverick.platform import http
from maverick.platform.config import HttpSettings
from maverick.platform.http import CircuitBreaker, CircuitOpenError


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(http.time, "monotonic", lambda: now[0])
    return now


def _breaker(threshold=1):
    return CircuitBreaker(
        "recovery",
        HttpSettings(breaker_failure_threshold=threshold, breaker_recovery_seconds=60),
    )


async def _fail():
    raise RuntimeError("dependency down")


async def _healthy():
    return "up"


async def _open(breaker):
    with pytest.raises(RuntimeError):
        await breaker.call(_fail)
    assert breaker.state == "open"


class PendingCall:
    def __init__(self, *, fail=False):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.fail = fail

    async def __call__(self):
        self.started.set()
        await self.release.wait()
        if self.fail:
            raise RuntimeError("old failure")
        return "up"


async def test_cancelled_probe_reopens_and_recovers_after_a_new_window(clock):
    breaker = _breaker()
    await _open(breaker)
    clock[0] = 60
    pending = PendingCall()
    task = asyncio.create_task(breaker.call(pending))
    await pending.started.wait()
    assert breaker.state == "half_open"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert breaker.state == "open"
    clock[0] = 61
    with pytest.raises(CircuitOpenError) as error:
        await breaker.call(_healthy)
    assert error.value.seconds_until_half_open == 59
    clock[0] = 120
    assert await breaker.call(_healthy) == "up"
    assert breaker.state == "closed"


async def test_cancelled_closed_call_does_not_count_as_a_dependency_failure(clock):
    breaker = _breaker(threshold=2)
    pending = PendingCall()
    task = asyncio.create_task(breaker.call(pending))
    await pending.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(RuntimeError):
        await breaker.call(_fail)
    assert breaker.state == "closed"
    await _open(breaker)


async def test_cancelled_probe_cannot_undo_an_explicit_reset(clock):
    breaker = _breaker()
    await _open(breaker)
    clock[0] = 60
    pending = PendingCall()
    task = asyncio.create_task(breaker.call(pending))
    await pending.started.wait()
    breaker.reset()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert breaker.state == "closed"
    assert await breaker.call(_healthy) == "up"


async def test_old_success_cannot_clear_a_new_open_window(clock):
    breaker = _breaker()
    pending = PendingCall()
    task = asyncio.create_task(breaker.call(pending))
    await pending.started.wait()
    await _open(breaker)
    pending.release.set()
    assert await task == "up"
    assert breaker.state == "open"
    with pytest.raises(CircuitOpenError):
        await breaker.call(_healthy)


async def test_old_failure_cannot_reopen_a_successfully_recovered_breaker(clock):
    breaker = _breaker()
    pending = PendingCall(fail=True)
    task = asyncio.create_task(breaker.call(pending))
    await pending.started.wait()
    await _open(breaker)
    clock[0] = 60
    assert await breaker.call(_healthy) == "up"
    pending.release.set()
    with pytest.raises(RuntimeError, match="old failure"):
        await task
    assert breaker.state == "closed"


async def test_old_failure_cannot_displace_the_current_probe(clock):
    breaker = _breaker()
    old = PendingCall(fail=True)
    old_task = asyncio.create_task(breaker.call(old))
    await old.started.wait()
    await _open(breaker)
    clock[0] = 60
    probe = PendingCall()
    probe_task = asyncio.create_task(breaker.call(probe))
    await probe.started.wait()
    old.release.set()
    with pytest.raises(RuntimeError):
        await old_task
    assert breaker.state == "half_open"
    with pytest.raises(CircuitOpenError):
        await breaker.call(_healthy)
    probe.release.set()
    assert await probe_task == "up"
    assert breaker.state == "closed"


async def test_probe_result_cannot_undo_a_reset_and_new_failure(clock):
    breaker = _breaker()
    await _open(breaker)
    clock[0] = 60
    pending = PendingCall()
    task = asyncio.create_task(breaker.call(pending))
    await pending.started.wait()
    breaker.reset()
    await _open(breaker)
    pending.release.set()
    assert await task == "up"
    assert breaker.state == "open"


async def test_old_success_after_reset_cannot_clear_new_failure_count(clock):
    breaker = _breaker(threshold=2)
    pending = PendingCall()
    task = asyncio.create_task(breaker.call(pending))
    await pending.started.wait()
    breaker.reset()
    with pytest.raises(RuntimeError):
        await breaker.call(_fail)
    pending.release.set()
    assert await task == "up"
    await _open(breaker)
