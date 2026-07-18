# Reliability

Current state and known gaps. Update when behavior changes.

## What exists

- Circuit breakers around external data providers.
- Rate limiting per tool category, strictest on research tools.
- Graceful-shutdown hooks for the scheduler, event bus, and caches.
- Health endpoints at `/health`, `/health/ready`, and `/health/live`.

## Known gaps

- Tool registration failures are logged and swallowed, so a broken router
  silently drops its tools. The new server fails fast instead.
- The legacy SSE transport needs a monkey-patch to serve trailing slashes.
  SSE is not carried into the new server.
