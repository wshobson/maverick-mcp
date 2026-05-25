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
- Prefer focused tests next to changed behavior.
- Use in-memory FastMCP patterns for MCP registration and router behavior where
  possible.
- Update docs when commands, markers, or setup expectations change.

## Related Testing Docs

- `in-memory.md`
- `integration.md`
- `exa-research.md`
- `speed.md`
