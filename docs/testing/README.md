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

PostgreSQL concurrency and migration cases carry `integration`; the default
suite runs their SQLite counterparts.

## Integration And External Tests

Database tests live beside their domains. Portfolio and market-data concurrency
tests use independent engines, and portfolio tests also use separate processes.
Run both SQLite and PostgreSQL cases with:

```bash
uv run pytest tests/portfolio/test_service_concurrency.py tests/market_data/test_history_concurrency.py -m "integration or not integration"
```

The PostgreSQL fixture starts a disposable local cluster using `initdb` and
`pg_ctl`, with per-test databases and teardown. It skips when those executables
are unavailable or the host cannot safely run the cluster. These checks call
no market provider. In-memory SQLite tests cover retained schema, exclusive
connection use, cancellation, and disposal.

`tests/server` builds the server and calls it through an in-memory FastMCP client.
Research request tests use mocked providers or loopback request capture with
synthetic keys. A future external-provider test must use the `external` marker,
explicit credentials, and authorization.

## Built package checks

CI has separate `core-wheel` and `extracted-sdist` jobs in
[ci.yml](../../.github/workflows/ci.yml). The first builds a wheel, installs only
its core dependencies in a fresh environment outside the checkout, confirms
imports come from site-packages and optional packages are absent, then calls
an offline portfolio tool through MCP. It expects 38 core tools and no research
or backtesting tools.

The source-archive job builds and extracts an sdist outside the checkout, installs
it without editable imports, collects the shipped eval tests, checks documentation
without Git metadata, and rebuilds the packages offline. Archive tests verify
required source assets and exclude local secrets, databases, caches, pointer
files, and generated eval runs/results. They also inject private files into the
extracted tree and confirm that rebuilding still excludes them.

`tests/structure/test_sdist.py` needs built artifacts supplied through
`MAVERICK_SDIST` and `MAVERICK_WHEEL`; without those paths, artifact checks skip.
The CI workflow contains the complete reproducible commands. A passing ordinary
unit run alone does not establish these package-installation checks.

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

- [Process E2E report and reproduction](mcp-e2e-2026-10-04.md): real STDIO/HTTP,
  all-tool matrix, installed package/container, and separate live-provider evidence.
- `in-memory.md`
- `exa-research.md`
