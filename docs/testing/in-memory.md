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
uv run pytest tests/server -v
uv run pytest tests/portfolio tests/screening tests/technical tests/market_data -v
```

## Pattern

```python
async with Client(mcp) as client:
    result = await client.call_tool("tool_name", {"param": "value"})
    assert result.data is not None
```

## Guidelines

- Mock yfinance, Redis, external research providers, and network calls.
- Use in-memory SQLite when a database is needed.
- Assert both success and failure behavior.
- Keep tests deterministic and independent.
