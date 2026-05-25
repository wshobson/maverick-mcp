# Exa And Research Provider Testing

Research-provider tests validate Exa/Tavily integration, timeout handling,
circuit breakers, and MCP research tool behavior.

## Main Coverage

- Provider initialization with and without API keys.
- Search timeout and failure handling.
- Research-agent orchestration across query depths.
- Specialized research paths for fundamental, technical, sentiment, and
  competitive analysis.
- MCP research tool responses.
- Performance and timeout budgets.

## Commands

Mocked/default tests:

```bash
uv run pytest tests/test_exa_research_integration.py -v
```

Real provider tests require API keys and should be run explicitly:

```bash
EXA_API_KEY=... uv run pytest -m external tests/test_exa_research_integration.py -v
```

## Provider Policy

- Mock provider responses in unit tests.
- Gate real API calls on environment variables.
- Prefer structured errors when providers are missing or unhealthy.
- Keep provider diagnostics in responses when it helps the user understand a
  degraded result.
