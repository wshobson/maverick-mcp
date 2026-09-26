# Testing Guide

The default test suite is pytest and is configured in `pyproject.toml`.

## Canonical Commands

```bash
make test         # unit tests only by default
make test-all     # includes integration, slow, and external markers
make test-specific TEST=name
make test-parallel
make test-cov
make lint
make typecheck
make check
make docs-check
```

Equivalent direct command:

```bash
uv run pytest -v
```

The default pytest `addopts` excludes:

- `integration`
- `slow`
- `external`

No test currently carries any of these three markers, so `make test-all`
collects the same tests as `make test`.

## Integration And External Tests

There is no `tests/integration/` directory. The nearest coverage runs in the
default unit suite: service tests for portfolio, screening, and market data
write to a tmp-file SQLite database, and `tests/server` builds the full server
and calls it through an in-memory FastMCP client. A future real-provider
research test is marked `external` and needs `EXA_API_KEY` (or
`RESEARCH_SEARCH_BACKEND=searxng` with `SEARXNG_BASE_URL`) plus
`LLM_PROVIDER`, `LLM_API_KEY`, and `LLM_MODEL`.

## Timeouts And Durations

There is no speed or benchmark suite.

- CI runs the unit suite with `--timeout=60` per test;
  `tests/backtesting/test_analysis.py` raises its own limit to 300s.
- The default `addopts` include `--durations=10`, so every run lists its 10
  slowest tests.
- Research timeouts are depth-scaled in `maverick/research/config.py`;
  `tests/research/test_service.py` pins the typed timeout error.

## Markers

- `unit`: fast isolated tests.
- `integration`: multi-component or database integration tests.
- `slow`: long-running tests.
- `external`: tests requiring real third-party APIs.
- `database`: tests requiring database access.
- `redis`: tests requiring Redis.

## Policy

- Unit tests should not make real network calls.
- External-provider tests must be opt-in and gated on API keys.
- Do not make the default unit suite depend on third-party availability.
- Prefer focused tests next to changed behavior.
- Use in-memory FastMCP patterns for MCP registration and tool behavior where
  possible.
- Update docs when commands, markers, or setup expectations change.

## Related Testing Docs

- `in-memory.md`
- `exa-research.md`
