# User Testing Knowledge — Maverick MCP BYOK

## Environment

- **Working directory:** `C:\Users\Janak Kalaria\Downloads\Droid.Claude\Linear\maverick-byok`
- **Branch:** `feat/byok-provider-endpoint`
- **Python:** 3.14.3 (numba/vectorbt/pandas_ta not installable)
- **OS:** Windows 10 (PowerShell default shell — use `;` not `&&` for command chaining)

## Testing Tools

All assertions in the validation contract specify `pytest` or `manual` as the testing tool. No browser or TUI testing is needed — this is a library/config PR.

### Running Tests

```bash
# Unit tests (reliable subset, avoids pandas_ta failures)
python -m pytest tests/unit/ tests/core/ tests/domain/ tests/providers/ tests/utils/ -m "not integration and not slow and not external" --tb=short -q

# Full test suite (has ~11 pre-existing pandas_ta failures to ignore)
python -m pytest tests/ -m "not integration and not slow and not external" --tb=short -q

# Specific test file
python -m pytest tests/test_llm_settings.py -v
python -m pytest tests/test_llm_factory.py -v

# Lint
ruff check .

# Format check
ruff format --check .
```

### Known Baseline Issues

- ~11 pre-existing pandas_ta failures (AttributeError: NoneType) — ignore these
- ~1300 collection errors from broad `tests/` — use `test-unit` command instead
- 73 pre-existing lint errors, 27 format issues — workers only need their own files clean

## Services

No services needed (`services: {}` in services.yaml). Unit tests use mocks only.

## Validation Concurrency

### Surface: pytest
- **Max concurrent validators:** 3
- **Rationale:** All assertions are pytest-based with monkeypatched env vars. No shared state, no database, no network. Tests are isolated by design.
- **Isolation:** Each flow validator can run its own pytest invocation against different test files without interference.

### Surface: manual
- **Max concurrent validators:** 1
- **Rationale:** Manual checks (grep, file inspection) don't conflict, but grouping them is efficient.

## Flow Validator Guidance: pytest

- Always use `monkeypatch` for env var manipulation — never set real environment variables
- Mock all LangChain chat models — never call real Anthropic/OpenAI APIs
- Tests run against the local repo clone, no external services needed
- Each validator should run its own test file(s) independently
- Use `-v` flag for detailed output and evidence collection

## Flow Validator Guidance: manual

- Use `grep`, `rg`, or file reading tools for manual checks
- Compare .env.example entries against settings.py Field definitions
- Verify documentation accuracy by cross-referencing with implementation code
