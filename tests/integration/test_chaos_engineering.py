"""
Chaos Engineering Tests for Resilience Testing.

This test suite covers:
- API failures and recovery mechanisms
- Database connection drops and reconnection
- Cache failures and fallback behavior
- Circuit breaker behavior under load
- Network timeouts and retries
- Memory pressure scenarios
- CPU overload situations
- External service outages
"""

import asyncio
import logging
import random
import threading
import time
from contextlib import ExitStack, contextmanager
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from maverick_mcp.backtesting import VectorBTEngine
from maverick_mcp.backtesting.persistence import BacktestPersistenceManager
from maverick_mcp.backtesting.strategies import STRATEGY_TEMPLATES

logger = logging.getLogger(__name__)


class ChaosInjector:
    """Utility class for injecting various types of failures."""

    @staticmethod
    @contextmanager
    def api_failure_injection(failure_rate: float = 0.3):
        """Inject API failures at specified rate."""
        original_get_stock_data = None

        def failing_get_stock_data(*args, **kwargs):
            if random.random() < failure_rate:
                if random.random() < 0.5:
                    raise ConnectionError("Simulated API connection failure")
                else:
                    raise TimeoutError("Simulated API timeout")
            return original_get_stock_data(*args, **kwargs) if original_get_stock_data else Mock()

        try:
            # Store original method and replace with failing version
            with patch.object(
                VectorBTEngine,
                'get_historical_data',
                side_effect=failing_get_stock_data
            ):
                yield
        finally:
            pass

    @staticmethod
    @contextmanager
    def database_failure_injection(failure_rate: float = 0.2):
        """Inject database failures at specified rate."""
        def failing_db_operation(*args, **kwargs):
            if random.random() < failure_rate:
                if random.random() < 0.33:
                    raise ConnectionError("Database connection lost")
                elif random.random() < 0.66:
                    raise Exception("Database query timeout")
                else:
                    raise Exception("Database lock timeout")
            return MagicMock()  # Return mock successful result

        try:
            with patch.object(
                BacktestPersistenceManager,
                'save_backtest_result',
                side_effect=failing_db_operation
            ):
                yield
        finally:
            pass

    @staticmethod
    @contextmanager
    def memory_pressure_injection(pressure_mb: int = 500):
        """Inject memory pressure by allocating large arrays."""
        pressure_arrays = []
        try:
            # Create memory pressure
            for _ in range(pressure_mb // 10):
                arr = np.random.random((1280, 1000))  # ~10MB each
                pressure_arrays.append(arr)
            yield
        finally:
            # Clean up memory pressure
            del pressure_arrays

    @staticmethod
    @contextmanager
    def cpu_load_injection(load_intensity: float = 0.8, duration: float = 5.0):
        """Inject CPU load using background threads."""
        stop_event = threading.Event()
        load_threads = []

        def cpu_intensive_task():
            """CPU-intensive task for load injection."""
            while not stop_event.is_set():
                # Perform CPU-intensive computation
                for _ in range(int(100000 * load_intensity)):
                    _ = sum(i ** 2 for i in range(100))
                time.sleep(0.01)  # Brief pause

        try:
            # Start CPU load threads
            num_threads = max(1, int(4 * load_intensity))  # Scale with intensity
            for _ in range(num_threads):
                thread = threading.Thread(target=cpu_intensive_task)
                thread.daemon = True
                thread.start()
                load_threads.append(thread)

            yield
        finally:
            # Stop CPU load
            stop_event.set()
            for thread in load_threads:
                thread.join(timeout=1.0)

    @staticmethod
    @contextmanager
    def network_instability_injection(delay_range: tuple = (0.1, 2.0), timeout_rate: float = 0.1):
        """Inject network instability with delays and timeouts."""
        async def unstable_network_call(original_func, *args, **kwargs):
            # Random delay
            delay = random.uniform(*delay_range)
            await asyncio.sleep(delay)

            # Random timeout
            if random.random() < timeout_rate:
                raise TimeoutError("Simulated network timeout")

            return await original_func(*args, **kwargs)

        # This is a simplified version - real implementation would patch actual network calls
        yield


class TestChaosEngineering:
    """Chaos engineering tests for system resilience."""

    @pytest.fixture
    async def resilient_data_provider(self):
        """Create data provider with built-in resilience patterns."""
        provider = Mock()

        async def resilient_get_data(symbol: str, *args, **kwargs):
            """Data provider with retry logic and fallback."""
            max_retries = 3
            retry_delay = 0.1

            for attempt in range(max_retries):
                try:
                    # Simulate data generation (can fail randomly)
                    if random.random() < 0.1:  # 10% failure rate
                        raise ConnectionError(f"API failure for {symbol}")

                    # Generate mock data
                    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
                    returns = np.random.normal(0.0008, 0.02, len(dates))
                    prices = 100 * np.cumprod(1 + returns)

                    return pd.DataFrame({
                        "Open": prices * np.random.uniform(0.99, 1.01, len(dates)),
                        "High": prices * np.random.uniform(1.00, 1.03, len(dates)),
                        "Low": prices * np.random.uniform(0.97, 1.00, len(dates)),
                        "Close": prices,
                        "Volume": np.random.randint(100000, 5000000, len(dates)),
                        "Adj Close": prices,
                    }, index=dates)

                except Exception:
                    if attempt == max_retries - 1:
                        # Final attempt failed, return minimal fallback data
                        logger.warning(f"All retries failed for {symbol}, using fallback data")
                        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
                        prices = np.full(len(dates), 100.0)
                        return pd.DataFrame({
                            "Open": prices,
                            "High": prices * 1.01,
                            "Low": prices * 0.99,
                            "Close": prices,
                            "Volume": np.full(len(dates), 1000000),
                            "Adj Close": prices,
                        }, index=dates)

                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff

        provider.get_stock_data.side_effect = resilient_get_data
        return provider

    async def test_api_failures_and_recovery(self, resilient_data_provider, benchmark_timer):
        """Test API failure scenarios and recovery mechanisms."""
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        strategy = "sma_cross"
        parameters = STRATEGY_TEMPLATES[strategy]["parameters"]

        # Test with different failure rates
        failure_scenarios = [
            {"name": "low_failure", "rate": 0.1},
            {"name": "moderate_failure", "rate": 0.3},
            {"name": "high_failure", "rate": 0.6},
        ]

        scenario_results = {}

        for scenario in failure_scenarios:
            with ChaosInjector.api_failure_injection(failure_rate=scenario["rate"]):
                with benchmark_timer() as timer:
                    results = []
                    failures = []

                    engine = VectorBTEngine(data_provider=resilient_data_provider)

                    for symbol in symbols:
                        try:
                            result = await engine.run_backtest(
                                symbol=symbol,
                                strategy_type=strategy,
                                parameters=parameters,
                                start_date="2023-01-01",
                                end_date="2023-12-31",
                            )
                            results.append(result)
                            logger.info(f"✓ {symbol} succeeded under {scenario['name']} conditions")

                        except Exception as e:
                            failures.append({"symbol": symbol, "error": str(e)})
                            logger.error(f"✗ {symbol} failed under {scenario['name']} conditions: {e}")

                execution_time = timer.elapsed
                success_rate = len(results) / len(symbols)
                recovery_rate = 1 - (scenario["rate"] * (1 - success_rate))  # Account for injected failures

                scenario_results[scenario["name"]] = {
                    "failure_rate_injected": scenario["rate"],
                    "success_rate_achieved": success_rate,
                    "recovery_effectiveness": recovery_rate,
                    "execution_time": execution_time,
                    "successful_backtests": len(results),
                    "failed_backtests": len(failures),
                }

                logger.info(
                    f"{scenario['name'].upper()} Failure Scenario:\n"
                    f"  • Injected Failure Rate: {scenario['rate']:.1%}\n"
                    f"  • Achieved Success Rate: {success_rate:.1%}\n"
                    f"  • Recovery Effectiveness: {recovery_rate:.1%}\n"
                    f"  • Execution Time: {execution_time:.2f}s"
                )

                # Assert minimum recovery effectiveness
                assert success_rate >= 0.5, f"Success rate too low for {scenario['name']}: {success_rate:.1%}"

        return scenario_results

    async def test_database_connection_drops(self, resilient_data_provider, db_session, benchmark_timer):
        """Test database connection drops and reconnection logic."""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        strategy = "rsi"
        parameters = STRATEGY_TEMPLATES[strategy]["parameters"]

        engine = VectorBTEngine(data_provider=resilient_data_provider)

        # Generate backtest results first
        backtest_results = []
        for symbol in symbols:
            result = await engine.run_backtest(
                symbol=symbol,
                strategy_type=strategy,
                parameters=parameters,
                start_date="2023-01-01",
                end_date="2023-12-31",
            )
            backtest_results.append(result)

        # Test database operations under chaos
        with ChaosInjector.database_failure_injection(failure_rate=0.3):
            with benchmark_timer() as timer:
                persistence_results = []
                persistence_failures = []

                # Attempt to save results with intermittent database failures
                for result in backtest_results:
                    retry_count = 0
                    max_retries = 3

                    while retry_count < max_retries:
                        try:
                            with BacktestPersistenceManager(session=db_session) as persistence:
                                backtest_id = persistence.save_backtest_result(
                                    vectorbt_results=result,
                                    execution_time=2.0,
                                    notes=f"Chaos test - {result['symbol']}",
                                )
                                persistence_results.append({
                                    "symbol": result["symbol"],
                                    "backtest_id": backtest_id,
                                    "retry_count": retry_count,
                                })
                                break  # Success, break retry loop

                        except Exception as e:
                            retry_count += 1
                            if retry_count >= max_retries:
                                persistence_failures.append({
                                    "symbol": result["symbol"],
                                    "error": str(e),
                                    "retry_count": retry_count,
                                })
                            else:
                                await asyncio.sleep(0.1 * retry_count)  # Exponential backoff

            persistence_time = timer.elapsed

        # Analyze results
        persistence_success_rate = len(persistence_results) / len(backtest_results)
        avg_retries = np.mean([r["retry_count"] for r in persistence_results]) if persistence_results else 0

        # Test recovery by attempting to retrieve saved data
        retrieval_successes = 0
        if persistence_results:
            for saved_result in persistence_results:
                try:
                    with BacktestPersistenceManager(session=db_session) as persistence:
                        retrieved = persistence.get_backtest_by_id(saved_result["backtest_id"])
                        if retrieved:
                            retrieval_successes += 1
                except Exception as e:
                    logger.error(f"Retrieval failed for {saved_result['symbol']}: {e}")

        retrieval_success_rate = retrieval_successes / len(persistence_results) if persistence_results else 0

        logger.info(
            f"Database Connection Drops Test Results:\n"
            f"  • Backtest Results: {len(backtest_results)}\n"
            f"  • Persistence Successes: {len(persistence_results)}\n"
            f"  • Persistence Failures: {len(persistence_failures)}\n"
            f"  • Persistence Success Rate: {persistence_success_rate:.1%}\n"
            f"  • Average Retries: {avg_retries:.1f}\n"
            f"  • Retrieval Success Rate: {retrieval_success_rate:.1%}\n"
            f"  • Total Time: {persistence_time:.2f}s"
        )

        # Assert resilience requirements
        assert persistence_success_rate >= 0.7, f"Persistence success rate too low: {persistence_success_rate:.1%}"
        assert retrieval_success_rate >= 0.9, f"Retrieval success rate too low: {retrieval_success_rate:.1%}"

        return {
            "persistence_success_rate": persistence_success_rate,
            "retrieval_success_rate": retrieval_success_rate,
            "avg_retries": avg_retries,
        }

    async def test_cache_failures_and_fallback(self, resilient_data_provider, benchmark_timer):
        """Test cache failures and fallback behavior."""
        symbols = ["CACHE_TEST_1", "CACHE_TEST_2", "CACHE_TEST_3"]
        strategy = "macd"
        parameters = STRATEGY_TEMPLATES[strategy]["parameters"]

        engine = VectorBTEngine(data_provider=resilient_data_provider)

        # Test cache behavior under failures
        cache_scenarios = [
            {"name": "normal_cache", "inject_failure": False},
            {"name": "cache_failures", "inject_failure": True},
        ]

        scenario_results = {}

        for scenario in cache_scenarios:
            if scenario["inject_failure"]:
                # Mock cache to randomly fail
                def failing_cache_get(key):
                    if random.random() < 0.4:  # 40% cache failure rate
                        raise ConnectionError("Cache connection failed")
                    return None  # Cache miss

                def failing_cache_set(key, value, ttl=None):
                    if random.random() < 0.3:  # 30% cache set failure rate
                        raise ConnectionError("Cache set operation failed")
                    return True

                cache_patches = [
                    patch('maverick_mcp.core.cache.CacheManager.get', side_effect=failing_cache_get),
                    patch('maverick_mcp.core.cache.CacheManager.set', side_effect=failing_cache_set),
                ]
            else:
                cache_patches = []

            with benchmark_timer() as timer:
                results = []
                cache_errors = []

                # Apply cache patches if needed
                with ExitStack() as stack:
                    for patch_context in cache_patches:
                        stack.enter_context(patch_context)

                    # Run backtests - should fallback gracefully on cache failures
                    for symbol in symbols:
                        try:
                            result = await engine.run_backtest(
                                symbol=symbol,
                                strategy_type=strategy,
                                parameters=parameters,
                                start_date="2023-01-01",
                                end_date="2023-12-31",
                            )
                            results.append(result)

                        except Exception as e:
                            cache_errors.append({"symbol": symbol, "error": str(e)})
                            logger.error(f"Backtest failed for {symbol} under {scenario['name']}: {e}")

            execution_time = timer.elapsed
            success_rate = len(results) / len(symbols)

            scenario_results[scenario["name"]] = {
                "execution_time": execution_time,
                "success_rate": success_rate,
                "cache_errors": len(cache_errors),
            }

            logger.info(
                f"{scenario['name'].upper()} Cache Scenario:\n"
                f"  • Execution Time: {execution_time:.2f}s\n"
                f"  • Success Rate: {success_rate:.1%}\n"
                f"  • Cache Errors: {len(cache_errors)}"
            )

            # Cache failures should not prevent backtests from completing
            assert success_rate >= 0.8, f"Success rate too low with cache issues: {success_rate:.1%}"

        # Cache failures might slightly increase execution time but shouldn't break functionality
        time_ratio = scenario_results["cache_failures"]["execution_time"] / scenario_results["normal_cache"]["execution_time"]
        assert time_ratio < 3.0, f"Cache failure time penalty too high: {time_ratio:.1f}x"

        return scenario_results

    async def test_circuit_breaker_behavior(self, resilient_data_provider, benchmark_timer):
        """Test circuit breaker behavior under load and failures."""
        symbols = ["CB_TEST_1", "CB_TEST_2", "CB_TEST_3", "CB_TEST_4", "CB_TEST_5"]
        strategy = "sma_cross"
        parameters = STRATEGY_TEMPLATES[strategy]["parameters"]

        # Mock circuit breaker states
        circuit_breaker_state = {"failures": 0, "state": "CLOSED", "last_failure": 0}
        failure_threshold = 3
        recovery_timeout = 2.0

        def circuit_breaker_wrapper(func):
            """Simple circuit breaker implementation."""
            async def wrapper(*args, **kwargs):
                current_time = time.time()

                # Check if circuit should reset
                if (circuit_breaker_state["state"] == "OPEN" and
                    current_time - circuit_breaker_state["last_failure"] > recovery_timeout):
                    circuit_breaker_state["state"] = "HALF_OPEN"
                    logger.info("Circuit breaker moved to HALF_OPEN state")

                # Circuit is open, reject immediately
                if circuit_breaker_state["state"] == "OPEN":
                    raise Exception("Circuit breaker is OPEN")

                try:
                    # Inject failures for testing
                    if random.random() < 0.4:  # 40% failure rate
                        raise ConnectionError("Simulated service failure")

                    result = await func(*args, **kwargs)

                    # Success - reset failure count if in HALF_OPEN state
                    if circuit_breaker_state["state"] == "HALF_OPEN":
                        circuit_breaker_state["state"] = "CLOSED"
                        circuit_breaker_state["failures"] = 0
                        logger.info("Circuit breaker CLOSED after successful recovery")

                    return result

                except Exception as e:
                    circuit_breaker_state["failures"] += 1
                    circuit_breaker_state["last_failure"] = current_time

                    if circuit_breaker_state["failures"] >= failure_threshold:
                        circuit_breaker_state["state"] = "OPEN"
                        logger.warning(f"Circuit breaker OPENED after {circuit_breaker_state['failures']} failures")

                    raise e

            return wrapper

        # Apply circuit breaker to engine operations
        engine = VectorBTEngine(data_provider=resilient_data_provider)

        with benchmark_timer() as timer:
            results = []
            circuit_breaker_trips = 0
            recovery_attempts = 0

            for _i, symbol in enumerate(symbols):
                try:
                    # Simulate circuit breaker behavior
                    current_symbol = symbol
                    @circuit_breaker_wrapper
                    async def protected_backtest(symbol_to_use=current_symbol):
                        return await engine.run_backtest(
                            symbol=symbol_to_use,
                            strategy_type=strategy,
                            parameters=parameters,
                            start_date="2023-01-01",
                            end_date="2023-12-31",
                        )

                    result = await protected_backtest()
                    results.append(result)
                    logger.info(f"✓ {symbol} succeeded (CB state: {circuit_breaker_state['state']})")

                except Exception as e:
                    if "Circuit breaker is OPEN" in str(e):
                        circuit_breaker_trips += 1
                        logger.warning(f"⚡ {symbol} blocked by circuit breaker")

                        # Wait for potential recovery
                        await asyncio.sleep(recovery_timeout + 0.1)
                        recovery_attempts += 1

                        # Try once more after recovery timeout
                        try:
                            recovery_symbol = symbol
                            @circuit_breaker_wrapper
                            async def recovery_backtest(symbol_to_use=recovery_symbol):
                                return await engine.run_backtest(
                                    symbol=symbol_to_use,
                                    strategy_type=strategy,
                                    parameters=parameters,
                                    start_date="2023-01-01",
                                    end_date="2023-12-31",
                                )

                            result = await recovery_backtest()
                            results.append(result)
                            logger.info(f"✓ {symbol} succeeded after circuit breaker recovery")

                        except Exception as recovery_error:
                            logger.error(f"✗ {symbol} failed even after recovery: {recovery_error}")
                    else:
                        logger.error(f"✗ {symbol} failed: {e}")

        execution_time = timer.elapsed
        success_rate = len(results) / len(symbols)
        circuit_breaker_effectiveness = circuit_breaker_trips > 0  # Circuit breaker activated

        logger.info(
            f"Circuit Breaker Behavior Test Results:\n"
            f"  • Symbols Tested: {len(symbols)}\n"
            f"  • Successful Results: {len(results)}\n"
            f"  • Success Rate: {success_rate:.1%}\n"
            f"  • Circuit Breaker Trips: {circuit_breaker_trips}\n"
            f"  • Recovery Attempts: {recovery_attempts}\n"
            f"  • Circuit Breaker Effectiveness: {circuit_breaker_effectiveness}\n"
            f"  • Final CB State: {circuit_breaker_state['state']}\n"
            f"  • Execution Time: {execution_time:.2f}s"
        )

        # Circuit breaker should provide some protection
        assert circuit_breaker_effectiveness, "Circuit breaker should have activated"
        assert success_rate >= 0.4, f"Success rate too low even with circuit breaker: {success_rate:.1%}"

        return {
            "success_rate": success_rate,
            "circuit_breaker_trips": circuit_breaker_trips,
            "recovery_attempts": recovery_attempts,
            "final_state": circuit_breaker_state["state"],
        }

    async def test_memory_pressure_resilience(self, resilient_data_provider, benchmark_timer):
        """Test system resilience under memory pressure."""
        symbols = ["MEM_TEST_1", "MEM_TEST_2", "MEM_TEST_3"]
        strategy = "bollinger"
        parameters = STRATEGY_TEMPLATES[strategy]["parameters"]

        # Test under different memory pressure levels
        pressure_levels = [0, 500, 1000]  # MB of memory pressure
        pressure_results = {}

        for pressure_mb in pressure_levels:
            with ChaosInjector.memory_pressure_injection(pressure_mb=pressure_mb):
                with benchmark_timer() as timer:
                    results = []
                    memory_errors = []

                    engine = VectorBTEngine(data_provider=resilient_data_provider)

                    for symbol in symbols:
                        try:
                            result = await engine.run_backtest(
                                symbol=symbol,
                                strategy_type=strategy,
                                parameters=parameters,
                                start_date="2023-01-01",
                                end_date="2023-12-31",
                            )
                            results.append(result)

                        except (MemoryError, Exception) as e:
                            memory_errors.append({"symbol": symbol, "error": str(e)})
                            logger.error(f"Memory pressure caused failure for {symbol}: {e}")

                execution_time = timer.elapsed
                success_rate = len(results) / len(symbols)

                pressure_results[f"{pressure_mb}mb"] = {
                    "pressure_mb": pressure_mb,
                    "execution_time": execution_time,
                    "success_rate": success_rate,
                    "memory_errors": len(memory_errors),
                }

                logger.info(
                    f"Memory Pressure {pressure_mb}MB Results:\n"
                    f"  • Execution Time: {execution_time:.2f}s\n"
                    f"  • Success Rate: {success_rate:.1%}\n"
                    f"  • Memory Errors: {len(memory_errors)}"
                )

        # System should be resilient to moderate memory pressure
        moderate_pressure_result = pressure_results["500mb"]
        high_pressure_result = pressure_results["1000mb"]

        assert moderate_pressure_result["success_rate"] >= 0.8, "Should handle moderate memory pressure"
        assert high_pressure_result["success_rate"] >= 0.5, "Should partially handle high memory pressure"

        return pressure_results

    async def test_cpu_overload_resilience(self, resilient_data_provider, benchmark_timer):
        """Test system resilience under CPU overload."""
        symbols = ["CPU_TEST_1", "CPU_TEST_2"]
        strategy = "momentum"
        parameters = STRATEGY_TEMPLATES[strategy]["parameters"]

        # Test under CPU load
        with ChaosInjector.cpu_load_injection(load_intensity=0.8, duration=10.0):
            with benchmark_timer() as timer:
                results = []
                timeout_errors = []

                engine = VectorBTEngine(data_provider=resilient_data_provider)

                for symbol in symbols:
                    try:
                        # Add timeout to prevent hanging under CPU load
                        result = await asyncio.wait_for(
                            engine.run_backtest(
                                symbol=symbol,
                                strategy_type=strategy,
                                parameters=parameters,
                                start_date="2023-01-01",
                                end_date="2023-12-31",
                            ),
                            timeout=30.0  # 30 second timeout
                        )
                        results.append(result)

                    except TimeoutError:
                        timeout_errors.append({"symbol": symbol, "error": "CPU overload timeout"})
                        logger.error(f"CPU overload caused timeout for {symbol}")

                    except Exception as e:
                        timeout_errors.append({"symbol": symbol, "error": str(e)})
                        logger.error(f"CPU overload caused failure for {symbol}: {e}")

            execution_time = timer.elapsed

        success_rate = len(results) / len(symbols)
        timeout_rate = len([e for e in timeout_errors if "timeout" in e["error"]]) / len(symbols)

        logger.info(
            f"CPU Overload Resilience Results:\n"
            f"  • Symbols Tested: {len(symbols)}\n"
            f"  • Successful Results: {len(results)}\n"
            f"  • Success Rate: {success_rate:.1%}\n"
            f"  • Timeout Rate: {timeout_rate:.1%}\n"
            f"  • Execution Time: {execution_time:.2f}s"
        )

        # System should handle some CPU pressure, though performance may degrade
        assert success_rate >= 0.5, f"Success rate too low under CPU load: {success_rate:.1%}"
        assert execution_time < 60.0, f"Execution time too long under CPU load: {execution_time:.1f}s"

        return {
            "success_rate": success_rate,
            "timeout_rate": timeout_rate,
            "execution_time": execution_time,
        }

    async def test_cascading_failure_recovery(self, resilient_data_provider, benchmark_timer):
        """Test recovery from cascading failures across multiple components."""
        symbols = ["CASCADE_1", "CASCADE_2", "CASCADE_3"]
        strategy = "rsi"
        parameters = STRATEGY_TEMPLATES[strategy]["parameters"]

        # Simulate cascading failures: API -> Cache -> Database
        with ChaosInjector.api_failure_injection(failure_rate=0.5):
            with ChaosInjector.memory_pressure_injection(pressure_mb=300):
                with benchmark_timer() as timer:
                    results = []
                    cascading_failures = []

                    engine = VectorBTEngine(data_provider=resilient_data_provider)

                    for symbol in symbols:
                        failure_chain = []
                        final_result = None

                        # Multiple recovery attempts with different strategies
                        for attempt in range(3):
                            try:
                                result = await engine.run_backtest(
                                    symbol=symbol,
                                    strategy_type=strategy,
                                    parameters=parameters,
                                    start_date="2023-01-01",
                                    end_date="2023-12-31",
                                )
                                final_result = result
                                break  # Success, exit retry loop

                            except Exception as e:
                                failure_chain.append(f"Attempt {attempt + 1}: {str(e)[:50]}")
                                if attempt < 2:
                                    # Progressive backoff and different strategies
                                    await asyncio.sleep(0.5 * (attempt + 1))

                        if final_result:
                            results.append(final_result)
                            logger.info(f"✓ {symbol} recovered after {len(failure_chain)} failures")
                        else:
                            cascading_failures.append({
                                "symbol": symbol,
                                "failure_chain": failure_chain
                            })
                            logger.error(f"✗ {symbol} failed completely: {failure_chain}")

                execution_time = timer.elapsed

        recovery_rate = len(results) / len(symbols)
        avg_failures_before_recovery = np.mean([
            len(cf["failure_chain"]) for cf in cascading_failures
        ]) if cascading_failures else 0

        logger.info(
            f"Cascading Failure Recovery Results:\n"
            f"  • Symbols Tested: {len(symbols)}\n"
            f"  • Successfully Recovered: {len(results)}\n"
            f"  • Complete Failures: {len(cascading_failures)}\n"
            f"  • Recovery Rate: {recovery_rate:.1%}\n"
            f"  • Avg Failures Before Recovery: {avg_failures_before_recovery:.1f}\n"
            f"  • Execution Time: {execution_time:.2f}s"
        )

        # System should show some recovery capability even under cascading failures
        assert recovery_rate >= 0.3, f"Recovery rate too low for cascading failures: {recovery_rate:.1%}"

        return {
            "recovery_rate": recovery_rate,
            "cascading_failures": len(cascading_failures),
            "avg_failures_before_recovery": avg_failures_before_recovery,
        }

if __name__ == "__main__":
    # Run chaos engineering tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto",
        "--timeout=900",  # 15 minute timeout for chaos tests
        "--durations=15",  # Show 15 slowest tests
    ])
