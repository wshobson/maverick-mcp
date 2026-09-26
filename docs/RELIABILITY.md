# Reliability

Current state and known gaps for `maverick/`, the whole system as of
v1.0.0. Update when behavior changes.

## What exists

- Per-service circuit breakers around outbound HTTP calls
  (`maverick.platform.http.get_breaker`), used by market data fetchers, the
  Exa search provider, and (through `request_resilient`) the SearXNG search
  provider.
- A per-service rate limiter (`DATA_PROVIDER_RATE_LIMIT`, default 5/s) on
  outbound HTTP requests made through
  `maverick.platform.http.request_resilient`. Only the SearXNG provider uses
  it today; yfinance and Exa calls do not pass through it.
- Extras degrade gracefully: with `[backtesting]`/`[research]` absent, each
  domain's `tools.register()` logs one warning and registers zero tools
  instead of raising, so a base install boots and serves the other domains.
- `maverick.server.app.main` catches any exception building the server and
  reports a clean one-line error plus a non-zero exit rather than a raw
  traceback -- the process's only top-level entry point.
- Tiered caching (memory, then Redis or SQLite) uses memory plus SQLite when
  Redis is not configured. A tier error on get/set/delete is logged and
  treated as a miss, so a configured but unreachable Redis degrades to the
  memory tier alone (the SQLite tier is not built when Redis is configured).

## Known gaps

- Tool registration failures inside a domain's `register(mcp)` are not
  individually caught; a broken domain fails server startup rather than
  degrading silently (a deliberate change from the legacy server's
  log-and-swallow behavior, made for a personal-use local server where a
  crash on startup is easier to diagnose than a quietly incomplete tool
  list).
- No HTTP `/health` endpoint exists. This is an MCP server, not a REST API;
  "did it register tools" (i.e. the client sees the expected tool list) is
  the health check for a personal-use server. Container orchestrators
  should use process liveness or an MCP-aware probe instead.
- No persistent alerting or scheduled background jobs (the legacy signal
  engine did not port; see `docs/runbooks/migrating-to-v1.md`). A stdio- or
  request-scoped MCP server has no long-running daemon to host one.
- The yfinance market-mover fallback (tier 2, a small liquid-stock scan)
  runs without breaker/retry protection by design. Routing it through the
  breaker from a worker thread deadlocked; `_build_yfinance_tier` in
  `maverick/market_data/fetchers.py` records why.
- A circuit breaker whose half-open probe is cancelled (for example by an
  `asyncio.wait_for` timeout) stays half-open, so every later call through
  that breaker fails with `CircuitOpenError` until the server restarts.
  `CircuitBreaker.call` in `maverick/platform/http.py` catches `Exception`,
  which does not include `asyncio.CancelledError`. A call admitted before the
  breaker opened can also close it when it completes late. Open as #272;
  draft fix in #273.
