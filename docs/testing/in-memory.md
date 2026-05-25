# In-Memory MCP Testing

In-memory FastMCP tests validate tools without starting external processes,
opening ports, or managing a server lifecycle.

## Use Cases

- Tool registration.
- Router isolation.
- Input validation.
- Error handling.
- Mocked provider behavior.
- Fast regression tests for MCP-facing behavior.

## Commands

```bash
uv run pytest maverick_mcp/tests/test_in_memory*.py -v
uv run pytest maverick_mcp/tests/test_in_memory_server.py -v
uv run pytest maverick_mcp/tests/test_in_memory_routers.py::TestTechnicalRouter -v
```

## Pattern

```python
async with Client(mcp) as client:
    result = await client.call_tool("tool_name", {"param": "value"})
    assert result.text is not None
```

## Guidelines

- Mock yfinance, Redis, external research providers, and network calls.
- Use in-memory SQLite when a database is needed.
- Assert both success and failure behavior.
- Keep tests deterministic and independent.
