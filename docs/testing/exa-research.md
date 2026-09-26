# Exa And Research Provider Testing

Research-provider tests validate the Exa and SearXNG search providers,
source scoring, timeout handling, circuit breakers, and MCP research tool
behavior. They live in `tests/research/` and are fully mocked.

## Main Coverage

- Provider behavior with and without `exa_py` installed, and SearXNG
  requests answered by an `httpx.MockTransport` (`test_providers.py`,
  `test_searxng.py`).
- Search timeout and failure handling, including the provider health gate
  and open circuit breakers.
- Research-agent graph runs: the comprehensive, company, and sentiment entry
  points, persona prompts, and provider or LLM failures
  (`test_agents_graph.py`).
- Specialized research paths for fundamental, technical, sentiment, and
  competitive analysis (`test_agents_subagents.py`).
- Configuration errors for a missing search backend or LLM, and typed
  timeout errors (`test_service.py`).
- MCP research tool responses and the extra-absent registration path
  (`test_tools.py`, `test_tools_availability.py`).

## Commands

Mocked/default tests:

```bash
uv run pytest tests/research -v
```

There are no real-provider tests. `tests/research/conftest.py` scrubs
`EXA_API_KEY`, `RESEARCH_SEARCH_BACKEND`, `SEARXNG_BASE_URL`, and the `LLM_*`
variables before every test, so a shell with real keys cannot make these
tests call Exa or an LLM.

## Provider Policy

- Mock provider responses in unit tests.
- Gate real API calls on environment variables.
- Prefer structured errors when providers are missing or unhealthy.
- Keep provider diagnostics in responses when it helps the user understand a
  degraded result.
