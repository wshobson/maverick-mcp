# MaverickMCP Test Suite

This comprehensive test suite covers both **Phase 5.1 (End-to-End Integration Tests)** and **Phase 5.2 (Performance Testing Suite)** for the VectorBT backtesting system.

## Overview

The test suite includes:

### Phase 5.1: End-to-End Integration Tests
- **Complete workflow integration** from data fetch to results visualization
- **All 15 strategies testing** (9 traditional + 6 ML strategies)
- **Parallel execution capabilities** with concurrency testing
- **Cache behavior optimization** and performance validation
- **MCP tools integration** for Claude Desktop interaction
- **High-volume production scenarios** with 100+ symbols
- **Chaos engineering** for resilience testing

### Phase 5.2: Performance Testing Suite
- **Load testing** for 10, 50, and 100 concurrent users
- **Benchmark testing** against performance targets
- **Stress testing** for resource usage monitoring
- **Profiling** for bottleneck identification

## Test Structure

```
tests/
├── integration/           # Phase 5.1: Integration Tests
│   ├── test_full_backtest_workflow_advanced.py
│   ├── test_mcp_tools.py
│   ├── test_high_volume.py
│   └── test_chaos_engineering.py
├── performance/          # Phase 5.2: Performance Tests
│   ├── test_load.py
│   ├── test_benchmarks.py
│   ├── test_stress.py
│   └── test_profiling.py
├── conftest.py          # Shared fixtures and configuration
└── README.md           # This file
```

## Quick Start

### Running All Tests

```bash
# Run all integration and performance tests
make test-full

# Run only integration tests
pytest tests/integration/ -v

# Run only performance tests
pytest tests/performance/ -v
```

### Running Specific Test Categories

```bash
# Integration tests
pytest tests/integration/test_full_backtest_workflow_advanced.py -v
pytest tests/integration/test_mcp_tools.py -v
pytest tests/integration/test_high_volume.py -v
pytest tests/integration/test_chaos_engineering.py -v

# Performance tests
pytest tests/performance/test_load.py -v
pytest tests/performance/test_benchmarks.py -v
pytest tests/performance/test_stress.py -v
pytest tests/performance/test_profiling.py -v
```

## Test Categories Detailed

### Integration Tests

#### 1. Advanced Full Backtest Workflow (`test_full_backtest_workflow_advanced.py`)
- **All 15 strategies integration testing** (traditional + ML)
- **Parallel execution capabilities** with async performance
- **Cache behavior optimization** and hit rate validation
- **Database persistence integration** with PostgreSQL
- **Visualization integration** with chart generation
- **Error recovery mechanisms** across the workflow
- **Resource management** and cleanup testing
- **Memory leak prevention** validation

**Key Tests:**
- `test_all_15_strategies_integration()` - Tests all available strategies
- `test_parallel_execution_capabilities()` - Concurrent backtest execution
- `test_cache_behavior_and_optimization()` - Cache efficiency validation
- `test_resource_management_comprehensive()` - Memory and thread management

#### 2. MCP Tools Integration (`test_mcp_tools.py`)
- **All MCP tool registrations** for Claude Desktop
- **Tool parameter validation** and error handling
- **Tool response formats** and data integrity
- **Claude Desktop simulation** with realistic usage patterns
- **Performance and timeout handling** for MCP calls

**Key Tests:**
- `test_all_mcp_tools_registration()` - Validates all tools are registered
- `test_run_backtest_tool_comprehensive()` - Core backtesting tool validation
- `test_claude_desktop_simulation()` - Realistic usage pattern simulation

#### 3. High Volume Production Scenarios (`test_high_volume.py`)
- **Large symbol set backtesting** (100+ symbols)
- **Multi-year historical data** processing
- **Memory management under load** with leak detection
- **Concurrent user scenarios** simulation
- **Database performance under load** testing
- **Cache efficiency** with large datasets

**Key Tests:**
- `test_large_symbol_set_backtesting()` - 100+ symbol processing
- `test_concurrent_user_scenarios()` - Multi-user simulation
- `test_memory_management_large_datasets()` - Memory leak prevention

#### 4. Chaos Engineering (`test_chaos_engineering.py`)
- **API failures and recovery** mechanisms
- **Database connection drops** and reconnection
- **Cache failures and fallback** behavior
- **Circuit breaker behavior** under load
- **Network instability** injection
- **Memory pressure scenarios** testing
- **CPU overload situations** handling
- **Cascading failure recovery** validation

**Key Tests:**
- `test_api_failures_and_recovery()` - API resilience testing
- `test_circuit_breaker_behavior()` - Circuit breaker validation
- `test_cascading_failure_recovery()` - Multi-component failure handling

### Performance Tests

#### 1. Load Testing (`test_load.py`)
- **Concurrent user load testing** (10, 50, 100 users)
- **Response time and throughput** measurement
- **Memory usage under load** monitoring
- **Database performance** with multiple connections
- **System stability** under sustained load

**Performance Targets:**
- 10 users: ≥2.0 req/s, ≤5.0s avg response time
- 50 users: ≥5.0 req/s, ≤8.0s avg response time
- 100 users: ≥3.0 req/s, ≤15.0s avg response time

**Key Tests:**
- `test_concurrent_users_10()`, `test_concurrent_users_50()`, `test_concurrent_users_100()`
- `test_load_scalability_analysis()` - Performance scaling analysis
- `test_sustained_load_stability()` - Long-duration stability testing

#### 2. Benchmark Testing (`test_benchmarks.py`)
- **Backtest execution < 2 seconds** per backtest
- **Memory usage < 500MB** per backtest
- **Cache hit rate > 80%** efficiency
- **API failure rate < 0.1%** reliability
- **Database query performance < 100ms** speed
- **Response time SLA compliance** validation

**Key Benchmarks:**
- Execution time targets
- Memory efficiency targets
- Cache performance targets
- Database performance targets
- SLA compliance targets

**Key Tests:**
- `test_backtest_execution_time_benchmark()` - Speed validation
- `test_memory_usage_benchmark()` - Memory efficiency
- `test_cache_hit_rate_benchmark()` - Cache performance
- `test_comprehensive_benchmark_suite()` - Full benchmark report

#### 3. Stress Testing (`test_stress.py`)
- **Sustained load testing** (15+ minutes)
- **Memory leak detection** over time
- **CPU utilization monitoring** under stress
- **Database connection pool** exhaustion testing
- **File descriptor limits** testing
- **Queue overflow scenarios** handling

**Key Tests:**
- `test_sustained_load_15_minutes()` - Extended load testing
- `test_memory_leak_detection()` - Memory leak validation
- `test_cpu_stress_resilience()` - CPU stress handling
- `test_database_connection_stress()` - DB connection pool testing

#### 4. Profiling and Bottleneck Identification (`test_profiling.py`)
- **CPU profiling** with cProfile integration
- **Memory allocation hotspots** identification
- **Database query performance** analysis
- **I/O vs CPU-bound** operation analysis
- **Optimization recommendations** generation

**Key Tests:**
- `test_profile_backtest_execution()` - CPU bottleneck identification
- `test_profile_database_query_performance()` - DB query analysis
- `test_profile_memory_allocation_patterns()` - Memory optimization
- `test_comprehensive_profiling_suite()` - Full profiling report

## Performance Targets

### Execution Performance
- **Backtest execution**: < 2 seconds per backtest
- **Data loading**: < 0.5 seconds average
- **Database saves**: < 50ms average
- **Database queries**: < 20ms average

### Throughput Targets
- **Sequential**: ≥ 2.0 backtests/second
- **Concurrent**: ≥ 5.0 backtests/second
- **10 concurrent users**: ≥ 2.0 requests/second
- **50 concurrent users**: ≥ 5.0 requests/second

### Resource Efficiency
- **Memory usage**: < 500MB per backtest
- **Memory growth**: < 100MB/hour sustained
- **Cache hit rate**: > 80%
- **API failure rate**: < 0.1%

### Response Time SLA
- **50th percentile**: < 1.5 seconds
- **95th percentile**: < 3.0 seconds
- **99th percentile**: < 5.0 seconds
- **SLA compliance**: > 95% of requests

## Test Configuration

### Environment Setup
All tests use containerized PostgreSQL and Redis for consistency:

```python
# Automatic container setup in conftest.py
@pytest.fixture(scope="session")
def postgres_container():
    with PostgresContainer("postgres:15-alpine") as postgres:
        yield postgres

@pytest.fixture(scope="session")
def redis_container():
    with RedisContainer("redis:7-alpine") as redis:
        yield redis
```

### Mock Data Providers
Tests use optimized mock data providers for consistent, fast testing:

```python
# Realistic stock data generation
def generate_stock_data(symbol: str) -> pd.DataFrame:
    # 3 years of realistic OHLCV data
    # Different market regimes (bull, sideways, bear)
    # Deterministic but varied based on symbol hash
```

### Parallel Execution
Tests are designed for parallel execution where possible:

```python
# Concurrent backtest execution
async def run_parallel_backtests(symbols, strategies):
    semaphore = asyncio.Semaphore(8)  # Control concurrency
    tasks = [run_with_semaphore(backtest) for backtest in all_backtests]
    results = await asyncio.gather(*tasks, return_exceptions=True)
```

## Running Tests in CI/CD

### GitHub Actions Configuration
```yaml
- name: Run Integration Tests
  run: |
    pytest tests/integration/ -v --tb=short --timeout=600

- name: Run Performance Tests
  run: |
    pytest tests/performance/ -v --tb=short --timeout=1800
```

### Test Markers
```bash
# Run only fast tests
pytest -m "not slow"

# Run only integration tests
pytest -m integration

# Run only performance tests
pytest -m performance

# Run stress tests (extended duration)
pytest -m stress
```

## Expected Test Results

### Integration Test Results
- **Strategy Coverage**: All 15 strategies (9 traditional + 6 ML) tested
- **Success Rate**: ≥ 80% success rate across all tests
- **Parallel Efficiency**: ≥ 2x speedup with concurrent execution
- **Error Recovery**: Graceful handling of all failure scenarios

### Performance Test Results
- **Load Testing**: Successful handling of 100 concurrent users
- **Benchmark Compliance**: ≥ 80% of benchmarks passed
- **Stress Testing**: Stable operation under extended load
- **Profiling**: Identification of optimization opportunities

## Troubleshooting

### Common Issues

#### Test Timeouts
```bash
# Increase timeout for long-running tests
pytest tests/performance/test_stress.py --timeout=1800
```

#### Memory Issues
```bash
# Monitor memory usage during tests
pytest tests/integration/test_high_volume.py -s --tb=short
```

#### Database Connection Issues
```bash
# Check container status
docker ps | grep postgres
docker logs <container_id>
```

#### Performance Assertion Failures
Check the test output for specific performance metrics that failed and compare against targets.

### Debug Mode
```bash
# Run with detailed logging
pytest tests/ -v -s --log-cli-level=INFO

# Run specific test with profiling
pytest tests/performance/test_profiling.py::test_comprehensive_profiling_suite -v -s
```

## Contributing

When adding new tests:

1. **Follow the existing patterns** for fixtures and mocks
2. **Add appropriate performance assertions** with clear targets
3. **Include comprehensive logging** for debugging
4. **Document expected behavior** and performance characteristics
5. **Use realistic test data** that represents production scenarios

### Test Categories
- Mark integration tests with `@pytest.mark.integration`
- Mark slow tests with `@pytest.mark.slow`
- Mark performance tests with `@pytest.mark.performance`
- Mark stress tests with `@pytest.mark.stress`

## Results and Reporting

### Test Reports
All tests generate comprehensive reports including:
- Performance metrics and benchmarks
- Resource usage analysis
- Error rates and success rates
- Optimization recommendations

### Performance Dashboards
Key metrics are logged for dashboard visualization:
- Execution times and throughput
- Memory usage patterns
- Database performance
- Cache hit rates
- Error rates and recovery times

This comprehensive test suite ensures the MaverickMCP backtesting system meets all performance and reliability requirements for production use.