"""
High-Volume Integration Tests for Production Scenarios.

This test suite covers:
- Testing with 100+ symbols
- Testing with years of historical data
- Memory management under load
- Concurrent user scenarios
- Database performance under high load
- Cache efficiency with large datasets
- API rate limiting and throttling
"""

import asyncio
import gc
import logging
import os
import random
import time
from datetime import datetime, timedelta
from unittest.mock import Mock

import numpy as np
import pandas as pd
import psutil
import pytest

from maverick_mcp.backtesting import VectorBTEngine
from maverick_mcp.backtesting.persistence import BacktestPersistenceManager
from maverick_mcp.backtesting.strategies import STRATEGY_TEMPLATES

logger = logging.getLogger(__name__)

# High volume test parameters
LARGE_SYMBOL_SET = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "ADBE", "CRM", "ORCL",
    "NFLX", "INTC", "AMD", "QCOM", "AVGO", "TXN", "MU", "AMAT", "LRCX", "KLAC",
    # Finance
    "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "BRK-B", "BLK", "SPGI",
    "CME", "ICE", "MCO", "COF", "USB", "TFC", "PNC", "SCHW", "CB", "AIG",
    # Healthcare
    "JNJ", "PFE", "ABT", "MRK", "TMO", "DHR", "BMY", "ABBV", "AMGN", "GILD",
    "BIIB", "REGN", "VRTX", "ISRG", "SYK", "BSX", "MDT", "EW", "HOLX", "RMD",
    # Consumer
    "WMT", "PG", "KO", "PEP", "COST", "HD", "MCD", "NKE", "SBUX", "TGT",
    "LOW", "DIS", "CMCSA", "VZ", "T", "TMUS", "CVX", "XOM", "UNH", "CVS",
    # Industrials
    "BA", "CAT", "DE", "GE", "HON", "MMM", "LMT", "RTX", "UNP", "UPS",
    "FDX", "WM", "EMR", "ETN", "PH", "CMI", "PCAR", "ROK", "DOV", "ITW",
    # Extended set for 100+ symbols
    "F", "GM", "FORD", "RIVN", "LCID", "PLTR", "SNOW", "ZM", "DOCU", "OKTA",
]

STRATEGIES_FOR_VOLUME_TEST = ["sma_cross", "rsi", "macd", "bollinger", "momentum"]


class TestHighVolumeIntegration:
    """High-volume integration tests for production scenarios."""

    @pytest.fixture
    async def high_volume_data_provider(self):
        """Create data provider with large dataset simulation."""
        provider = Mock()

        def generate_multi_year_data(symbol: str, years: int = 3) -> pd.DataFrame:
            """Generate multi-year realistic data for a symbol."""
            # Generate deterministic but varied data based on symbol hash
            symbol_seed = hash(symbol) % 10000
            np.random.seed(symbol_seed)

            # Create 3 years of daily data
            start_date = datetime.now() - timedelta(days=years * 365)
            dates = pd.date_range(start=start_date, periods=years * 252, freq="B")  # Business days

            # Generate realistic price movements
            base_price = 50 + (symbol_seed % 200)  # Base price $50-$250
            returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns

            # Add some trend and volatility clustering
            trend = np.sin(np.arange(len(dates)) / 252 * 2 * np.pi) * 0.001  # Annual cycle
            returns += trend

            # Generate prices
            prices = base_price * np.cumprod(1 + returns)

            # Create OHLCV data
            high_mult = np.random.uniform(1.005, 1.03, len(dates))
            low_mult = np.random.uniform(0.97, 0.995, len(dates))
            open_mult = np.random.uniform(0.995, 1.005, len(dates))

            volumes = np.random.randint(100000, 10000000, len(dates))

            data = pd.DataFrame({
                "Open": prices * open_mult,
                "High": prices * high_mult,
                "Low": prices * low_mult,
                "Close": prices,
                "Volume": volumes,
                "Adj Close": prices,
            }, index=dates)

            # Ensure OHLC constraints
            data["High"] = np.maximum(data["High"], np.maximum(data["Open"], data["Close"]))
            data["Low"] = np.minimum(data["Low"], np.minimum(data["Open"], data["Close"]))

            return data

        provider.get_stock_data.side_effect = generate_multi_year_data
        return provider

    async def test_large_symbol_set_backtesting(self, high_volume_data_provider, benchmark_timer):
        """Test backtesting with 100+ symbols."""
        symbols = LARGE_SYMBOL_SET[:100]  # Use first 100 symbols
        strategy = "sma_cross"

        engine = VectorBTEngine(data_provider=high_volume_data_provider)
        parameters = STRATEGY_TEMPLATES[strategy]["parameters"]

        results = []
        failed_symbols = []

        # Track memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        with benchmark_timer() as timer:
            # Process symbols in batches to manage memory
            batch_size = 20
            for i in range(0, len(symbols), batch_size):
                batch_symbols = symbols[i:i + batch_size]

                # Process batch
                batch_tasks = []
                for symbol in batch_symbols:
                    task = engine.run_backtest(
                        symbol=symbol,
                        strategy_type=strategy,
                        parameters=parameters,
                        start_date="2022-01-01",
                        end_date="2023-12-31",
                    )
                    batch_tasks.append((symbol, task))

                # Execute batch concurrently
                batch_results = await asyncio.gather(
                    *[task for _, task in batch_tasks],
                    return_exceptions=True
                )

                # Process results
                for _j, (symbol, result) in enumerate(zip(batch_symbols, batch_results, strict=False)):
                    if isinstance(result, Exception):
                        failed_symbols.append(symbol)
                        logger.error(f"✗ {symbol} failed: {result}")
                    else:
                        results.append(result)
                        if len(results) % 10 == 0:
                            logger.info(f"Processed {len(results)} symbols...")

                # Force garbage collection after each batch
                gc.collect()

                # Check memory usage
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_growth = current_memory - initial_memory

                if memory_growth > 2000:  # More than 2GB growth
                    logger.warning(f"High memory usage detected: {memory_growth:.1f}MB")

        execution_time = timer.elapsed
        final_memory = process.memory_info().rss / 1024 / 1024
        total_memory_growth = final_memory - initial_memory

        # Performance assertions
        success_rate = len(results) / len(symbols)
        assert success_rate >= 0.85, f"Success rate too low: {success_rate:.1%}"
        assert execution_time < 1800, f"Execution time too long: {execution_time:.1f}s"  # 30 minutes max
        assert total_memory_growth < 3000, f"Memory growth too high: {total_memory_growth:.1f}MB"  # Max 3GB growth

        # Calculate performance metrics
        avg_execution_time = execution_time / len(symbols)
        throughput = len(results) / execution_time  # Backtests per second

        logger.info(
            f"Large Symbol Set Test Results:\n"
            f"  • Total Symbols: {len(symbols)}\n"
            f"  • Successful: {len(results)}\n"
            f"  • Failed: {len(failed_symbols)}\n"
            f"  • Success Rate: {success_rate:.1%}\n"
            f"  • Total Execution Time: {execution_time:.1f}s\n"
            f"  • Avg Time per Symbol: {avg_execution_time:.2f}s\n"
            f"  • Throughput: {throughput:.2f} backtests/second\n"
            f"  • Memory Growth: {total_memory_growth:.1f}MB\n"
            f"  • Failed Symbols: {failed_symbols[:10]}{'...' if len(failed_symbols) > 10 else ''}"
        )

        return {
            "symbols_processed": len(results),
            "execution_time": execution_time,
            "throughput": throughput,
            "memory_growth": total_memory_growth,
            "success_rate": success_rate,
        }

    async def test_multi_year_historical_data(self, high_volume_data_provider, benchmark_timer):
        """Test with years of historical data (high data volume)."""
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        strategy = "sma_cross"

        engine = VectorBTEngine(data_provider=high_volume_data_provider)
        parameters = STRATEGY_TEMPLATES[strategy]["parameters"]

        # Test with different time periods
        time_periods = [
            ("1_year", "2023-01-01", "2023-12-31"),
            ("2_years", "2022-01-01", "2023-12-31"),
            ("3_years", "2021-01-01", "2023-12-31"),
            ("5_years", "2019-01-01", "2023-12-31"),
        ]

        period_results = {}

        for period_name, start_date, end_date in time_periods:
            with benchmark_timer() as timer:
                period_data = []

                for symbol in symbols:
                    try:
                        result = await engine.run_backtest(
                            symbol=symbol,
                            strategy_type=strategy,
                            parameters=parameters,
                            start_date=start_date,
                            end_date=end_date,
                        )
                        period_data.append(result)

                    except Exception as e:
                        logger.error(f"Failed {symbol} for {period_name}: {e}")

                execution_time = timer.elapsed

                # Calculate average data points processed
                avg_data_points = np.mean([len(r.get("equity_curve", [])) for r in period_data])
                data_throughput = avg_data_points * len(period_data) / execution_time

                period_results[period_name] = {
                    "execution_time": execution_time,
                    "symbols_processed": len(period_data),
                    "avg_data_points": avg_data_points,
                    "data_throughput": data_throughput,
                }

                logger.info(
                    f"{period_name.upper()} Period Results:\n"
                    f"  • Execution Time: {execution_time:.1f}s\n"
                    f"  • Avg Data Points: {avg_data_points:.0f}\n"
                    f"  • Data Throughput: {data_throughput:.0f} points/second"
                )

        # Validate performance scales reasonably with data size
        one_year_time = period_results["1_year"]["execution_time"]
        three_year_time = period_results["3_years"]["execution_time"]

        # 3 years should not take more than 5x the time of 1 year (allow for overhead)
        time_scaling = three_year_time / one_year_time
        assert time_scaling < 5.0, f"Time scaling too poor: {time_scaling:.1f}x"

        return period_results

    async def test_concurrent_user_scenarios(self, high_volume_data_provider, benchmark_timer):
        """Test concurrent user scenarios with multiple simultaneous backtests."""
        symbols = LARGE_SYMBOL_SET[:50]
        strategies = STRATEGIES_FOR_VOLUME_TEST

        # Simulate different user scenarios
        user_scenarios = [
            {
                "user_id": f"user_{i}",
                "symbols": random.sample(symbols, 5),
                "strategy": random.choice(strategies),
                "start_date": "2022-01-01",
                "end_date": "2023-12-31",
            }
            for i in range(20)  # Simulate 20 concurrent users
        ]

        async def simulate_user_session(scenario):
            """Simulate a single user session."""
            engine = VectorBTEngine(data_provider=high_volume_data_provider)
            parameters = STRATEGY_TEMPLATES[scenario["strategy"]]["parameters"]

            user_results = []
            session_start = time.time()

            for symbol in scenario["symbols"]:
                try:
                    result = await engine.run_backtest(
                        symbol=symbol,
                        strategy_type=scenario["strategy"],
                        parameters=parameters,
                        start_date=scenario["start_date"],
                        end_date=scenario["end_date"],
                    )
                    user_results.append(result)

                except Exception as e:
                    logger.error(f"User {scenario['user_id']} failed on {symbol}: {e}")

            session_time = time.time() - session_start

            return {
                "user_id": scenario["user_id"],
                "results": user_results,
                "session_time": session_time,
                "symbols_processed": len(user_results),
                "success_rate": len(user_results) / len(scenario["symbols"]),
            }

        # Execute all user sessions concurrently
        with benchmark_timer() as timer:
            # Use semaphore to control concurrency
            semaphore = asyncio.Semaphore(10)  # Max 10 concurrent sessions

            async def run_with_semaphore(scenario):
                async with semaphore:
                    return await simulate_user_session(scenario)

            session_results = await asyncio.gather(
                *[run_with_semaphore(scenario) for scenario in user_scenarios],
                return_exceptions=True
            )

        total_execution_time = timer.elapsed

        # Analyze results
        successful_sessions = [r for r in session_results if isinstance(r, dict)]
        failed_sessions = len(session_results) - len(successful_sessions)

        total_backtests = sum(r["symbols_processed"] for r in successful_sessions)
        avg_session_time = np.mean([r["session_time"] for r in successful_sessions])
        avg_success_rate = np.mean([r["success_rate"] for r in successful_sessions])

        # Performance assertions
        session_success_rate = len(successful_sessions) / len(session_results)
        assert session_success_rate >= 0.8, f"Session success rate too low: {session_success_rate:.1%}"
        assert avg_success_rate >= 0.8, f"Average backtest success rate too low: {avg_success_rate:.1%}"
        assert total_execution_time < 600, f"Total execution time too long: {total_execution_time:.1f}s"  # 10 minutes max

        concurrent_throughput = total_backtests / total_execution_time

        logger.info(
            f"Concurrent User Scenarios Results:\n"
            f"  • Total Users: {len(user_scenarios)}\n"
            f"  • Successful Sessions: {len(successful_sessions)}\n"
            f"  • Failed Sessions: {failed_sessions}\n"
            f"  • Session Success Rate: {session_success_rate:.1%}\n"
            f"  • Total Backtests: {total_backtests}\n"
            f"  • Avg Session Time: {avg_session_time:.1f}s\n"
            f"  • Avg Backtest Success Rate: {avg_success_rate:.1%}\n"
            f"  • Total Execution Time: {total_execution_time:.1f}s\n"
            f"  • Concurrent Throughput: {concurrent_throughput:.2f} backtests/second"
        )

        return {
            "session_success_rate": session_success_rate,
            "avg_success_rate": avg_success_rate,
            "concurrent_throughput": concurrent_throughput,
            "total_execution_time": total_execution_time,
        }

    async def test_database_performance_under_load(self, high_volume_data_provider, db_session, benchmark_timer):
        """Test database performance under high load."""
        symbols = LARGE_SYMBOL_SET[:30]  # 30 symbols for DB test
        strategy = "sma_cross"

        engine = VectorBTEngine(data_provider=high_volume_data_provider)
        parameters = STRATEGY_TEMPLATES[strategy]["parameters"]

        # Run backtests and save to database
        backtest_results = []

        with benchmark_timer() as timer:
            # Generate backtest results
            for symbol in symbols:
                try:
                    result = await engine.run_backtest(
                        symbol=symbol,
                        strategy_type=strategy,
                        parameters=parameters,
                        start_date="2023-01-01",
                        end_date="2023-12-31",
                    )
                    backtest_results.append(result)
                except Exception as e:
                    logger.error(f"Backtest failed for {symbol}: {e}")

        backtest_generation_time = timer.elapsed

        # Test database operations under load
        with benchmark_timer() as db_timer:
            with BacktestPersistenceManager(session=db_session) as persistence:
                saved_ids = []

                # Batch save results
                for result in backtest_results:
                    try:
                        backtest_id = persistence.save_backtest_result(
                            vectorbt_results=result,
                            execution_time=2.0,
                            notes=f"High volume test - {result['symbol']}",
                        )
                        saved_ids.append(backtest_id)
                    except Exception as e:
                        logger.error(f"Save failed for {result['symbol']}: {e}")

                # Test batch retrieval
                retrieved_results = []
                for backtest_id in saved_ids:
                    try:
                        retrieved = persistence.get_backtest_by_id(backtest_id)
                        if retrieved:
                            retrieved_results.append(retrieved)
                    except Exception as e:
                        logger.error(f"Retrieval failed for {backtest_id}: {e}")

                # Test queries under load
                strategy_results = persistence.get_backtests_by_strategy(strategy)

        db_operation_time = db_timer.elapsed

        # Performance assertions
        save_success_rate = len(saved_ids) / len(backtest_results)
        retrieval_success_rate = len(retrieved_results) / len(saved_ids) if saved_ids else 0

        assert save_success_rate >= 0.95, f"Database save success rate too low: {save_success_rate:.1%}"
        assert retrieval_success_rate >= 0.95, f"Database retrieval success rate too low: {retrieval_success_rate:.1%}"
        assert db_operation_time < 300, f"Database operations too slow: {db_operation_time:.1f}s"  # 5 minutes max

        # Calculate database performance metrics
        save_throughput = len(saved_ids) / db_operation_time
        logger.info(
            f"Database Performance Under Load Results:\n"
            f"  • Backtest Generation: {backtest_generation_time:.1f}s\n"
            f"  • Database Operations: {db_operation_time:.1f}s\n"
            f"  • Backtests Generated: {len(backtest_results)}\n"
            f"  • Records Saved: {len(saved_ids)}\n"
            f"  • Records Retrieved: {len(retrieved_results)}\n"
            f"  • Save Success Rate: {save_success_rate:.1%}\n"
            f"  • Retrieval Success Rate: {retrieval_success_rate:.1%}\n"
            f"  • Save Throughput: {save_throughput:.2f} saves/second\n"
            f"  • Query Results: {len(strategy_results)} records"
        )

        return {
            "save_success_rate": save_success_rate,
            "retrieval_success_rate": retrieval_success_rate,
            "save_throughput": save_throughput,
            "db_operation_time": db_operation_time,
        }

    async def test_memory_management_large_datasets(self, high_volume_data_provider, benchmark_timer):
        """Test memory management with large datasets."""
        symbols = LARGE_SYMBOL_SET[:25]  # 25 symbols for memory test
        strategies = STRATEGIES_FOR_VOLUME_TEST

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_snapshots = []

        engine = VectorBTEngine(data_provider=high_volume_data_provider)

        with benchmark_timer() as timer:
            for i, symbol in enumerate(symbols):
                for strategy in strategies:
                    try:
                        parameters = STRATEGY_TEMPLATES[strategy]["parameters"]

                        # Run backtest
                        await engine.run_backtest(
                            symbol=symbol,
                            strategy_type=strategy,
                            parameters=parameters,
                            start_date="2021-01-01",  # 3 years of data
                            end_date="2023-12-31",
                        )

                        # Take memory snapshot
                        current_memory = process.memory_info().rss / 1024 / 1024
                        memory_snapshots.append({
                            "iteration": i * len(strategies) + strategies.index(strategy),
                            "symbol": symbol,
                            "strategy": strategy,
                            "memory_mb": current_memory,
                            "memory_growth": current_memory - initial_memory,
                        })

                        # Force periodic garbage collection
                        if (i * len(strategies) + strategies.index(strategy)) % 10 == 0:
                            gc.collect()

                    except Exception as e:
                        logger.error(f"Failed {symbol} with {strategy}: {e}")

        execution_time = timer.elapsed
        final_memory = process.memory_info().rss / 1024 / 1024
        total_memory_growth = final_memory - initial_memory
        peak_memory = max(snapshot["memory_mb"] for snapshot in memory_snapshots)

        # Analyze memory patterns
        memory_growths = [s["memory_growth"] for s in memory_snapshots]
        avg_memory_growth = np.mean(memory_growths)
        max_memory_growth = max(memory_growths)

        # Check for memory leaks (memory should not grow linearly with iterations)
        if len(memory_snapshots) > 10:
            # Linear regression to detect memory leaks
            iterations = [s["iteration"] for s in memory_snapshots]
            memory_values = [s["memory_growth"] for s in memory_snapshots]

            # Simple linear regression
            n = len(iterations)
            sum_x = sum(iterations)
            sum_y = sum(memory_values)
            sum_xy = sum(x * y for x, y in zip(iterations, memory_values, strict=False))
            sum_xx = sum(x * x for x in iterations)

            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)

            # Memory leak detection (slope should be small)
            memory_leak_rate = slope  # MB per iteration
        else:
            memory_leak_rate = 0

        # Performance assertions
        assert total_memory_growth < 2000, f"Total memory growth too high: {total_memory_growth:.1f}MB"
        assert peak_memory < initial_memory + 2500, f"Peak memory too high: {peak_memory:.1f}MB"
        assert abs(memory_leak_rate) < 5, f"Potential memory leak detected: {memory_leak_rate:.2f}MB/iteration"

        logger.info(
            f"Memory Management Large Datasets Results:\n"
            f"  • Initial Memory: {initial_memory:.1f}MB\n"
            f"  • Final Memory: {final_memory:.1f}MB\n"
            f"  • Total Growth: {total_memory_growth:.1f}MB\n"
            f"  • Peak Memory: {peak_memory:.1f}MB\n"
            f"  • Avg Growth: {avg_memory_growth:.1f}MB\n"
            f"  • Max Growth: {max_memory_growth:.1f}MB\n"
            f"  • Memory Leak Rate: {memory_leak_rate:.2f}MB/iteration\n"
            f"  • Execution Time: {execution_time:.1f}s\n"
            f"  • Iterations: {len(memory_snapshots)}"
        )

        return {
            "total_memory_growth": total_memory_growth,
            "peak_memory": peak_memory,
            "memory_leak_rate": memory_leak_rate,
            "execution_time": execution_time,
            "memory_snapshots": memory_snapshots,
        }

    async def test_cache_efficiency_large_dataset(self, high_volume_data_provider, benchmark_timer):
        """Test cache efficiency with large datasets."""
        # Test cache with repeated access patterns
        symbols = LARGE_SYMBOL_SET[:20]
        strategy = "sma_cross"

        engine = VectorBTEngine(data_provider=high_volume_data_provider)
        parameters = STRATEGY_TEMPLATES[strategy]["parameters"]

        # First pass - populate cache
        with benchmark_timer() as timer:
            first_pass_results = []
            for symbol in symbols:
                try:
                    result = await engine.run_backtest(
                        symbol=symbol,
                        strategy_type=strategy,
                        parameters=parameters,
                        start_date="2023-01-01",
                        end_date="2023-12-31",
                    )
                    first_pass_results.append(result)
                except Exception as e:
                    logger.error(f"First pass failed for {symbol}: {e}")

        first_pass_time = timer.elapsed

        # Second pass - should benefit from cache
        with benchmark_timer() as timer:
            second_pass_results = []
            for symbol in symbols:
                try:
                    result = await engine.run_backtest(
                        symbol=symbol,
                        strategy_type=strategy,
                        parameters=parameters,
                        start_date="2023-01-01",
                        end_date="2023-12-31",
                    )
                    second_pass_results.append(result)
                except Exception as e:
                    logger.error(f"Second pass failed for {symbol}: {e}")

        second_pass_time = timer.elapsed

        # Third pass - different parameters (no cache benefit)
        modified_parameters = {**parameters, "fast_period": parameters.get("fast_period", 10) + 5}
        with benchmark_timer() as timer:
            third_pass_results = []
            for symbol in symbols:
                try:
                    result = await engine.run_backtest(
                        symbol=symbol,
                        strategy_type=strategy,
                        parameters=modified_parameters,
                        start_date="2023-01-01",
                        end_date="2023-12-31",
                    )
                    third_pass_results.append(result)
                except Exception as e:
                    logger.error(f"Third pass failed for {symbol}: {e}")

        third_pass_time = timer.elapsed

        # Calculate cache efficiency metrics
        cache_speedup = first_pass_time / second_pass_time if second_pass_time > 0 else 1.0
        no_cache_comparison = first_pass_time / third_pass_time if third_pass_time > 0 else 1.0

        # Cache hit rate estimation (if second pass is significantly faster)
        estimated_cache_hit_rate = max(0, min(1, (first_pass_time - second_pass_time) / first_pass_time))

        logger.info(
            f"Cache Efficiency Large Dataset Results:\n"
            f"  • First Pass (populate): {first_pass_time:.2f}s ({len(first_pass_results)} symbols)\n"
            f"  • Second Pass (cached): {second_pass_time:.2f}s ({len(second_pass_results)} symbols)\n"
            f"  • Third Pass (no cache): {third_pass_time:.2f}s ({len(third_pass_results)} symbols)\n"
            f"  • Cache Speedup: {cache_speedup:.2f}x\n"
            f"  • No Cache Comparison: {no_cache_comparison:.2f}x\n"
            f"  • Estimated Cache Hit Rate: {estimated_cache_hit_rate:.1%}"
        )

        return {
            "first_pass_time": first_pass_time,
            "second_pass_time": second_pass_time,
            "third_pass_time": third_pass_time,
            "cache_speedup": cache_speedup,
            "estimated_cache_hit_rate": estimated_cache_hit_rate,
        }


if __name__ == "__main__":
    # Run high-volume integration tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto",
        "--timeout=3600",  # 1 hour timeout for high-volume tests
        "--durations=20",  # Show 20 slowest tests
        "-x",  # Stop on first failure
    ])
