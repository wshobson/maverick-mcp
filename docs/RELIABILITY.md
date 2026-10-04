# Reliability

Current source behavior and remaining limits, verified on October 4, 2026.
The published v1.1.0 release predates the correctness changes below.

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
- Adjusted history refreshes on access after 24 hours, or sooner for expanded
  coverage and provisional current-day bars. It is not an immediate corporate-action
  feed. Full-union refreshes can be expensive; pre-listing ranges can refetch.
- Market-provider availability and research answer quality are not established
  by offline tests. Final synthesis errors when no usable evidence survives.

## Verified correctness contracts

Portfolio mutations serialize across independent SQLite/PostgreSQL service
instances and processes. Memory SQLite retains one exclusively checked-out
connection; reset cancellation does not release it prematurely or block future
borrowers forever. A cancelled portfolio caller can still have a committed
worker transaction, so callers must inspect state before retrying.

History refreshes reserve generations and atomically replace complete adjusted
snapshots. Older responses cannot overwrite newer reservations. Partial or failed
responses preserve the prior snapshot and allow retry. Native upserts handle
concurrent date conflicts, and existing schemas gain only nullable metadata.

Redis promotion preserves remaining expiry, and the composed HTTP helper counts
exhausted retryable statuses as breaker failures. Docker defaults use `/data`;
the documented volume command survived a real container replacement with stored
holdings, watchlists, journal entries, and cache records intact.

The integrated offline suite passed 1,599 tests, with built-artifact checks and
14 PostgreSQL cases handled separately. All 28 combined SQLite/PostgreSQL cases
passed. A real core-only wheel installation and a Git-free extracted source
archive passed their distinct checks. See the
[completed correctness plan](exec-plans/completed/2026-10-04-correctness-and-reliability.md)
for exact evidence and limits. These checks do not establish live-provider
accuracy, human eval labels, or release publication.
