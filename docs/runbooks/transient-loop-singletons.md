# Runbook: Singletons created on transient event loops

**Status:** open follow-up. Filed 2026-04-15 after the SIGTERM-shutdown incident; see the RCA attached to PR #146.

## Symptom

During graceful shutdown, `cleanup_performance_systems()` logs:

```
ERROR - Error cleaning up performance systems: Event loop is closed
```

Other cleanup callbacks in the same phase succeed (health monitor, Redis
`close_cache`, scheduler, DB dispose). Only resources constructed on the
startup-time transient loop fail.

## Root cause

Server startup runs `asyncio.run(init_systems())` *before* `mcp.run()` in
`maverick_mcp/api/server.py` (grep for the `init_systems` call site).
`asyncio.run` creates a fresh loop, runs the coroutine, then **closes the
loop** as it returns. Any `aio` object constructed during that call
(`redis.asyncio.Redis` clients, aiohttp sessions, task groups) retains a
reference to the now-closed loop. When the much-later shutdown path
awaits `resource.close()` or `resource.aclose()`, the internal coroutine
tries to schedule on the old loop and raises `Event loop is closed`.

The same class of bug was raised by a PR #121 reviewer
(`discussion_r3004047826`) for the APScheduler instance. PR #146 moved
scheduler startup into the ASGI lifespan, but several singletons remain
on the `asyncio.run(init_systems())` path.

## Known affected singletons

| Object | Created where | Cleanup site | Behavior on shutdown |
|---|---|---|---|
| `redis_manager` (performance.py) | `init_systems()` → `initialize_performance_systems()` | `cleanup_performance_systems()` registered at `server.py:1885` | Fails with `Event loop is closed` |
| others TBD | — | — | Audit pending |

`close_cache` for the *other* Redis client works because it calls
`redis_client.close()` synchronously on a sync client — no loop binding.

## Fix 3 plan (scope for the follow-up PR)

1. Move `initialize_performance_systems()` out of `init_systems()` and
   into `_server_lifespan` alongside the scheduler startup
   (`server.py:298`). The server lifespan runs on uvicorn's long-lived
   loop, so any object constructed there outlives the request lifecycle
   correctly.
2. Audit the rest of `init_systems()` for additional `aio` singletons
   and relocate each one that holds loop state.
3. Add a lint/pre-commit guard: `asyncio.run(` outside entrypoints
   (`maverick_mcp/__main__.py`, scripts, tests) is a code smell — flag
   new occurrences in PR review.
4. Extend `tests/test_graceful_shutdown.py` with a subprocess test that
   starts the server, triggers SIGTERM, and asserts that
   `grep -q "Event loop is closed" backend.log` returns non-zero.

## Dependencies / ordering

- Do NOT land before PR #121's scheduler migration — the two refactors
  touch the same `_server_lifespan` startup block; landing them in the
  same window risks merge friction. PR #121 establishes the
  "construct singletons on the lifespan loop" pattern; Fix 3 applies it
  to the remaining singletons.

## Interim mitigation

The ASGI double-start RuntimeError (the *other* shutdown-race symptom
from the same incident) was fixed in PR #146 by
`ShutdownGateMiddleware` (`maverick_mcp/api/middleware/shutdown_gate.py`)
which returns 503 for new HTTP requests during drain. That stops the
cascade that was triggering the cleanup code path most often. The
`Event loop is closed` log line remains but is now cosmetic: the cleanup
callback logs the error and continues, and `GracefulShutdownHandler`
propagates a non-zero exit code so orchestrators still see the failure.

## Related

- `docs/runbooks/asyncio-systemexit.md` — companion runbook on
  `os._exit` vs `sys.exit` in async contexts.
- PR #146 RCA (see PR description / commit history, 2026-04-15).
- PR #121 review comment `discussion_r3004047826`.
