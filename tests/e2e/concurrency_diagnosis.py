"""Opt-in diagnostics for the unchanged 15-second portfolio spawn barrier.

This is internal service evidence, not MCP or provider evidence. Child imports
are deferred so their progress and exceptions survive a failed parent barrier.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import faulthandler
import json
import multiprocessing
import os
import queue
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
THREAD_SETTINGS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def event(path: Path, stage: str, **fields: Any) -> None:
    """Append a timestamped process stage to diagnostic evidence."""
    with path.open("a") as output:
        output.write(
            json.dumps(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "monotonic": time.monotonic(),
                    "pid": os.getpid(),
                    "stage": stage,
                    **fields,
                }
            )
            + "\n"
        )


def process_add(url: str, shares: str, ready: Any, results: Any, evidence: str):
    """Same engine/schema/barrier/add sequence as the existing unit worker."""
    path = Path(evidence) / f"worker-{shares}.jsonl"
    engine = None
    with (Path(evidence) / f"worker-{shares}-python-stack.log").open("w") as stack:
        faulthandler.enable(file=stack)
        faulthandler.register(signal.SIGUSR1, file=stack, all_threads=True)
        event(path, "before-heavy-imports")
        try:
            from decimal import Decimal
            from unittest.mock import AsyncMock, patch

            event(path, "before-config-import")
            from maverick.platform.config import DatabaseSettings

            event(path, "before-db-import")
            from maverick.platform.db import create_engine_from_settings

            event(path, "before-service-import")
            from maverick.portfolio import service as service_module
            from maverick.portfolio.service import PortfolioService

            event(path, "imports-complete")
            event(path, "before-engine-init")
            engine = create_engine_from_settings(DatabaseSettings(url=url))
            event(path, "engine-ready")
            service_module.service_risk.resolve_sector = AsyncMock(return_value="Tech")
            original_add = service_module.add_shares

            def slow_add(*args: Any, **kwargs: Any):
                """Delay the write to preserve the original concurrency trigger."""
                time.sleep(0.1)
                return original_add(*args, **kwargs)

            async def run():
                """Initialize the schema, await the barrier, and record the write."""
                service = PortfolioService(engine, AsyncMock())
                event(path, "before-schema")
                await service._ensure_schema()
                event(path, "schema-ready")
                event(path, "before-barrier", timeout_seconds=15)
                ready.wait(timeout=15)
                event(path, "barrier-passed")
                await service.add_position(
                    "u", "p", "AAPL", Decimal(shares), Decimal("100")
                )
                event(path, "persisted", shares_added=shares)

            with patch.object(service_module, "add_shares", slow_add):
                asyncio.run(run())
            results.put({"pid": os.getpid(), "shares": shares, "result": "ok"})
        except BaseException as error:
            detail = {
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
            event(path, "exception", **detail)
            results.put({"pid": os.getpid(), "shares": shares, **detail})
        finally:
            if engine is not None:
                engine.dispose()
            event(path, "worker-exit")


def sample_worker(process: Any, evidence: Path) -> None:
    """Capture worker stacks without masking the original failure."""
    if not process.is_alive():
        return
    try:
        event(evidence / "parent.jsonl", "sampling-worker", child_pid=process.pid)
        if any(
            f'"pid": {process.pid},' in file.read_text()
            for file in evidence.glob("worker-*.jsonl")
        ):
            os.kill(process.pid, signal.SIGUSR1)
        if sys.platform == "darwin":
            with (evidence / f"sample-{process.pid}-command.log").open("w") as output:
                subprocess.run(
                    [
                        "/usr/bin/sample",
                        str(process.pid),
                        "1",
                        "-file",
                        str(evidence / f"sample-{process.pid}.log"),
                    ],
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=10,
                    check=False,
                )
    except (OSError, subprocess.TimeoutExpired) as error:
        # Sampling is secondary evidence; preserve the original barrier failure
        # and allow diagnose() to finish cleanup and write its summary.
        with contextlib.suppress(OSError):
            event(
                evidence / "parent.jsonl",
                "sampling-error",
                child_pid=process.pid,
                error=repr(error),
            )


async def diagnose(evidence: Path) -> dict[str, Any]:
    """Record worker stages and final state around the unchanged barrier."""
    from decimal import Decimal
    from unittest.mock import AsyncMock

    from maverick.platform.config import DatabaseSettings
    from maverick.platform.db import create_engine_from_settings
    from maverick.portfolio import service as service_module
    from maverick.portfolio.service import PortfolioService

    state = Path(tempfile.mkdtemp(prefix="maverick-concurrency-diagnosis-"))
    (state / ".env").write_text("")
    os.chdir(state)
    url = f"sqlite:///{state / 'concurrent.db'}"
    engine = create_engine_from_settings(DatabaseSettings(url=url))
    service_module.service_risk.resolve_sector = AsyncMock(return_value="Tech")
    service = PortfolioService(engine, AsyncMock())
    await service._ensure_schema()
    await service.add_position("u", "p", "AAPL", Decimal("10"), Decimal("100"))
    path = evidence / "parent.jsonl"
    event(path, "seeded", shares="10", state=str(state))
    context = multiprocessing.get_context("spawn")
    ready = context.Barrier(3)
    results = context.Queue()
    processes = [
        context.Process(
            target=process_add, args=(url, shares, ready, results, str(evidence))
        )
        for shares in ("1", "2")
    ]
    messages: list[dict[str, Any]] = []
    error = None
    try:
        for process in processes:
            process.start()
            event(path, "worker-started", child_pid=process.pid)
        event(path, "before-barrier", timeout_seconds=15)
        await asyncio.to_thread(ready.wait, 15)
        event(path, "barrier-passed")
        for _ in processes:
            messages.append(await asyncio.to_thread(results.get, True, 15))
        for process in processes:
            await asyncio.to_thread(process.join, 15)
    except Exception as failure:
        error = {"error": repr(failure), "traceback": traceback.format_exc()}
        event(path, "exception", **error)
        for process in processes:
            sample_worker(process, evidence)
    finally:
        # Drain on the failure path too: the original test hides these results.
        while True:
            try:
                messages.append(results.get_nowait())
            except queue.Empty:
                break
        for process in processes:
            if process.is_alive():
                event(path, "terminate-worker", child_pid=process.pid)
                process.terminate()
                process.join(timeout=5)
        while True:
            try:
                messages.append(results.get_nowait())
            except queue.Empty:
                break
        results.close()
        results.join_thread()
    positions = await service._read_positions("u", "p")
    engine.dispose()
    summary = {
        "label": "Internal service diagnostic; no MCP or providers",
        "barrier_timeout_seconds": 15,
        "thread_environment": {name: os.environ.get(name) for name in THREAD_SETTINGS},
        "state": str(state),
        "parent_error": error,
        "worker_results": messages,
        "workers": [
            {"pid": p.pid, "exitcode": p.exitcode, "alive": p.is_alive()}
            for p in processes
        ],
        "positions": [
            {"shares": str(p.shares), "total_cost": str(p.total_cost)}
            for p in positions
        ],
    }
    (evidence / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.evidence.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(asyncio.run(diagnose(output_dir)), indent=2))
