# Reliability

Current state and known gaps for `maverick/`, the whole system as of
v1.0.0. Update when behavior changes.

## What exists

- Per-service circuit breakers around outbound HTTP calls
  (`maverick.platform.http.get_breaker`), used by market data fetchers and
  the Exa search provider.
- A shared rate limiter (`DATA_PROVIDER_RATE_LIMIT`, default 5/s) on
  outbound HTTP requests via `maverick.platform.http.request_resilient`.
- Extras degrade gracefully: with `[backtesting]`/`[research]` absent, each
  domain's `tools.register()` logs one warning and registers zero tools
  instead of raising, so a base install boots and serves the other domains.
- `maverick.server.app.main` catches any exception building the server and
  reports a clean one-line error plus a non-zero exit rather than a raw
  traceback -- the process's only top-level entry point.
- Tiered caching (memory, then Redis or SQLite) degrades to in-memory/SQLite
  automatically when Redis is unavailable or disabled.

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
- The tier-3 market-mover fallback (a small liquid-stock scan) runs without
  breaker/retry protection; documented as a last-resort trade-off in
  `maverick/market_data/fetchers.py`.
