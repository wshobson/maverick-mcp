# In-Memory Testing Guide for Maverick-MCP

This guide explains the in-memory testing patterns implemented for Maverick-MCP using FastMCP's testing capabilities.

## Overview

In-memory testing allows us to test the MCP server without:
- Starting external processes
- Making network calls
- Managing server lifecycle
- Dealing with port conflicts

This results in faster, more reliable tests that can run in any environment.

## Test Files

### 1. `test_in_memory_server.py`
Basic in-memory server tests covering:
- Health endpoint validation
- Stock data fetching
- Technical analysis tools
- Batch operations
- Input validation
- Error handling
- Resource management

### 2. `test_in_memory_routers.py`
Domain-specific router tests:
- Technical analysis router (RSI, MACD, support/resistance)
- Screening router (Maverick, Trending Breakout)
- Portfolio router (risk analysis, correlation)
- Data router (batch fetching, caching)
- Concurrent router operations

### 3. `test_advanced_patterns.py`
Advanced testing patterns:
- External dependency mocking (yfinance, Redis)
- Performance and load testing
- Error recovery patterns
- Integration scenarios
- Monitoring and metrics

## Running the Tests

### Run all in-memory tests:
```bash
pytest maverick_mcp/tests/test_in_memory*.py -v
```

### Run specific test file:
```bash
pytest maverick_mcp/tests/test_in_memory_server.py -v
```

### Run with coverage:
```bash
pytest maverick_mcp/tests/test_in_memory*.py --cov=maverick_mcp --cov-report=html
```

### Run specific test class:
```bash
pytest maverick_mcp/tests/test_in_memory_routers.py::TestTechnicalRouter -v
```

## Key Testing Patterns

### 1. In-Memory Database
```python
@pytest.fixture
def test_db():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    # Add test data...
    yield engine
```

### 2. Mock External Services
```python
@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    with patch('maverick_mcp.data.cache.RedisCache') as mock:
        cache_instance = Mock()
        # Configure mock behavior...
        yield cache_instance
```

### 3. FastMCP Client Testing
```python
async with Client(mcp) as client:
    result = await client.call_tool("tool_name", {"param": "value"})
    assert result.text is not None
```

### 4. Router Isolation
```python
test_mcp = FastMCP("TestServer")
test_mcp.mount("/technical", technical_router)
async with Client(test_mcp) as client:
    # Test only technical router
```

## Benefits

### 1. **Speed**
- No process startup overhead
- No network latency
- Instant test execution

### 2. **Reliability**
- No port conflicts
- No external dependencies
- Deterministic results

### 3. **Isolation**
- Each test runs in isolation
- No shared state between tests
- Easy to debug failures

### 4. **Flexibility**
- Easy to mock dependencies
- Test specific scenarios
- Control external service behavior

## Best Practices

### 1. Use Fixtures
Create reusable fixtures for common test setup:
```python
@pytest.fixture
def populated_db(test_db):
    """Database with test data."""
    # Add stocks, prices, etc.
    return test_db
```

### 2. Mock External APIs
Always mock external services:
```python
with patch('yfinance.download') as mock_yf:
    mock_yf.return_value = test_data
    # Run tests
```

### 3. Test Error Scenarios
Include tests for failure cases:
```python
mock_yf.side_effect = Exception("API Error")
# Verify graceful handling
```

### 4. Measure Performance
Use timing to ensure performance:
```python
start_time = time.time()
await client.call_tool("tool_name", params)
duration = time.time() - start_time
assert duration < 1.0  # Should complete in under 1 second
```

## Debugging Tests

### Enable logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Use pytest debugging:
```bash
pytest -vv --pdb  # Drop into debugger on failure
```

### Capture output:
```bash
pytest -s  # Don't capture stdout
```

## CI/CD Integration

These tests are perfect for CI/CD pipelines:

```yaml
# .github/workflows/test.yml
- name: Run in-memory tests
  run: |
    pytest maverick_mcp/tests/test_in_memory*.py \
      --cov=maverick_mcp \
      --cov-report=xml \
      --junit-xml=test-results.xml
```

## Extending the Tests

To add new test cases:

1. Choose the appropriate test file based on what you're testing
2. Use existing fixtures or create new ones
3. Follow the async pattern with `Client(mcp)`
4. Mock external dependencies
5. Assert both success and failure cases

Example:
```python
@pytest.mark.asyncio
async def test_new_feature(test_db, mock_redis):
    """Test description."""
    async with Client(mcp) as client:
        result = await client.call_tool("new_tool", {
            "param": "value"
        })
        
        assert result.text is not None
        data = eval(result.text)
        assert data["expected_key"] == "expected_value"
```

## Troubleshooting

### Common Issues:

1. **Import Errors**: Ensure maverick_mcp is installed: `pip install -e .`
2. **Async Warnings**: Use `pytest-asyncio` for async tests
3. **Mock Not Working**: Check patch path matches actual import
4. **Database Errors**: Ensure models are imported before `create_all()`

### Tips:

- Run tests in isolation first to identify issues
- Check fixture dependencies
- Verify mock configurations
- Use debugger to inspect test state

## Conclusion

These in-memory tests provide comprehensive coverage of Maverick-MCP functionality while maintaining fast execution and reliability. They demonstrate best practices for testing MCP servers and can be easily extended for new features.