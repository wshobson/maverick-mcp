# Asyncio absorbs `SystemExit` from fire-and-forget tasks

**Symptom**: Graceful-shutdown code calls `sys.exit(exit_code)` but the
process keeps running. Orchestrators (systemd, Kubernetes) report a
0-code exit long after the intended shutdown, or hang until the liveness
probe kills them. Cleanup-failure signalling (non-zero exit on cleanup
error) silently stops working.

**Cause**: `sys.exit()` raises `SystemExit`, which is a `BaseException`
— *not* caught by ordinary `except Exception:` blocks, but *is* caught
by asyncio's Task machinery. When the raising coroutine is a
fire-and-forget task (scheduled via `loop.create_task(...)` without an
`await`), `SystemExit` is stored on the Task result and never
propagates to the interpreter's main thread. The task shows up in logs
only as

```text
Task exception was never retrieved
  future: <Task finished ... coro=<_async_shutdown() done ...> exception=SystemExit(1)>
```

…when the loop eventually closes. By then the orchestrator has already
given up or the process has been killed by a higher layer.

## Where this bit us

`maverick_mcp/utils/shutdown.py` — signal handlers schedule
`_async_shutdown` via `loop.create_task(...)` (see `_signal_handler`).
Calling `sys.exit(exit_code)` inside that coroutine silently broke the
PR-146 non-zero-exit guarantee. Unit tests passed because they
`await handler._async_shutdown(...)` directly and mocked `sys.exit`,
never exercising the real task path.

## The fix

Use `os._exit(exit_code)` in the shutdown helper. `os._exit` bypasses
Python's normal unwind machinery and terminates the process
immediately with the given code — no exception, no absorption, no
Task plumbing in the way. Flush logging and stdio first so the
post-mortem log line that explains the exit actually reaches disk.

See `maverick_mcp/utils/shutdown.py::_force_exit` for the canonical
implementation and
`tests/test_graceful_shutdown.py::test_force_exit_terminates_process_from_asyncio_task`
for the regression test that pins the invariant.

## When to use which exit primitive

| Primitive                                           | When                                                                                                                                                    |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sys.exit(code)`                                    | Main-thread top level, CLI scripts, synchronous test runners. Raises `SystemExit`; cooperative cleanup via `finally`/context managers still runs.       |
| `raise SystemExit(code)`                            | Same semantics as `sys.exit`. Prefer `sys.exit` for readability.                                                                                        |
| `os._exit(code)`                                    | When control is inside a fire-and-forget asyncio Task, a signal handler that can't unwind, or any context where Python's exception machinery can't reach the top level. **Skips `finally` and `atexit` handlers** — flush what you need first. |
| `loop.stop()` + exit in main after `run_until_complete` | When you genuinely want cooperative shutdown and can ensure a main-thread runner picks up the exit code after the loop stops.                          |

## How to spot this class of bug

The smell is: **the `sys.exit` is inside `async def` AND the coroutine
is scheduled via `create_task` rather than awaited to completion.**

Quick grep over async code:

```bash
grep -rn "sys.exit" --include='*.py' | grep -B3 "async def\|create_task"
```

Tests mocking `sys.exit` and `await`-ing the coroutine directly will
pass on the buggy code — because they bypass the very thing that
breaks. A real regression test needs a subprocess + asserting
`proc.returncode`.

## Related reading

- CPython asyncio source: `asyncio.tasks.Task.__step` — see where
  `BaseException` subclasses get stored on `_exception` rather than
  re-raised when `_must_cancel` is false.
- Python docs: [os._exit](https://docs.python.org/3/library/os.html#os._exit)
  is documented as "intended for cases ... where it is desired to exit
  immediately, for example, in the child process after a call to fork()."
  Fire-and-forget task exit is the same shape of problem.
