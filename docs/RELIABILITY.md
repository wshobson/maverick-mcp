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
- Concurrent portfolio read/modify/write transactions can overwrite each other;
  a successful response does not guarantee that every purchase was preserved.
- Stored price bars are insert-only, so partial current-session bars and later
  corporate-action corrections are not refreshed; concurrent inserts can fail.
- Redis reads with TTL zero or an expired-between-reads key can extend a cached
  value to the global TTL when promoting it into the memory tier.
- Exhausted retryable HTTP statuses currently count as breaker successes.
  Cancellation/stale-result recovery was fixed in #273 on 2026-10-04, including
  the timeout regression's isolated fake clock; status-based failure accounting
  is a separate remaining defect.
- In-memory SQLite uses separate connections through NullPool, losing schema
  and records between connections.
- The documented Docker quick start does not persist the default SQLite data
  outside the removable container.

See [the 2026-10-04 review](design-docs/2026-10-04-project-review.md) for
reproductions, scope, and the ordered correctness plan. These are known defects,
not hypothetical risks; the default offline suite does not yet prevent them.
