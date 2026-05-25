# Integration Testing

Integration tests cover multi-component workflows such as orchestration,
portfolio persistence, database behavior, and external-provider paths.

## Commands

Default unit tests exclude integration:

```bash
make test
```

Run integration explicitly:

```bash
uv run pytest -m integration -v
uv run pytest tests/integration -v -m ""
```

Some older orchestration helpers under `tests/integration/` can also be run
directly when needed:

```bash
cd tests/integration
./run_integration_tests.sh
```

## Requirements

- Python 3.12 and dependencies installed through `uv`.
- Database service or test fixture appropriate to the selected test.
- Optional provider keys for external research paths:
  - `EXA_API_KEY`
  - `TAVILY_API_KEY`
  - `OPENROUTER_API_KEY`

## Policy

- Keep integration tests opt-in unless they are fast and deterministic.
- Mark real provider calls with `external`.
- Do not make the default unit suite depend on third-party availability.
- Capture generated result files under ignored output locations.
