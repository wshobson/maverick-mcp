# Architecture

MaverickMCP is a local FastMCP server. The MCP layer should stay thin: routers
validate input, call services/providers, and return structured data with
financial disclaimers where appropriate.

## Runtime Shape

```text
MCP clients
  -> maverick_mcp.api.server / routers
  -> service layer and domain models
  -> providers, database, cache, and optional research APIs
```

## Package Layout

- `api/`: FastMCP server, routers, tool registry, health and monitoring routes.
- `services/`: durable service domains, including signals, screening pipeline,
  journal, watchlist, and risk dashboard.
- `domain/`: pure domain objects and business rules, including portfolio and
  screening concepts.
- `data/`: SQLAlchemy models, database session helpers, cache tables, and data
  performance utilities.
- `providers/`: external data-provider adapters for market, stock, macro, and
  optional sentiment/research data.
- `agents/`: research, supervisor, optimized research, and multi-agent
  orchestration components.
- `backtesting/`: VectorBT integration, strategies, strategy parsing, and
  backtest orchestration.
- `validation/`: request validation helpers for router-level tool inputs.
- `monitoring/`: health checks, metrics, and middleware.
- `tools/`: older local analysis helpers such as portfolio and risk utilities.

## Service Boundaries

- MCP tools should not accumulate durable business logic when a service-layer
  home exists.
- Shared business behavior belongs in `maverick_mcp/services/` or
  `maverick_mcp/domain/`.
- Provider-specific behavior belongs behind provider adapters.
- Long-running or scheduled behavior should use the existing scheduler/event-bus
  patterns instead of ad hoc background loops.

## Data And Cache

- SQLite is the default local path.
- PostgreSQL is supported for larger local datasets.
- Redis is optional; the app should degrade to in-memory caching when Redis is
  unavailable.
- Market data loading and seed flows live under `scripts/`.

## MCP Transports

- STDIO is the preferred Claude Desktop path.
- Streamable HTTP at `http://localhost:8003/mcp/` is the default `make dev`
  transport for bridge and remote workflows.
- SSE is retained for legacy/debug clients.

See `runbooks/claude-desktop.md` for concrete client configuration.

## Historical Context

The `conductor/` and `docs/superpowers/` folders contain historical planning
context. They can help explain why some service-layer domains exist, but the
current source of truth starts at `docs/INDEX.md` and this file.
