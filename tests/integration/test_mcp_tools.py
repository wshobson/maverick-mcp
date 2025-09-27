"""
MCP Tools Integration Tests for Claude Desktop Interaction.

This test suite covers:
- All MCP tool registrations and functionality
- Tool parameter validation and error handling
- Tool response formats and data integrity
- Claude Desktop simulation and interaction patterns
- Real-world usage scenarios
- Performance and timeout handling
"""

import asyncio
import logging
from unittest.mock import Mock, patch

import numpy as np
import pytest
from fastmcp import Context

from maverick_mcp.api.routers.backtesting import setup_backtesting_tools

logger = logging.getLogger(__name__)


class MockFastMCP:
    """Mock FastMCP instance for testing tool registration."""

    def __init__(self):
        self.tools = {}
        self.tool_functions = {}

    def tool(self, name: str = None):
        """Mock tool decorator."""

        def decorator(func):
            tool_name = name or func.__name__
            self.tools[tool_name] = {
                "function": func,
                "name": tool_name,
                "signature": self._get_function_signature(func),
            }
            self.tool_functions[tool_name] = func
            return func

        return decorator

    def _get_function_signature(self, func):
        """Extract function signature for validation."""
        import inspect

        sig = inspect.signature(func)
        return {
            "parameters": list(sig.parameters.keys()),
            "annotations": {k: str(v.annotation) for k, v in sig.parameters.items()},
        }


class TestMCPToolsIntegration:
    """Integration tests for MCP tools and Claude Desktop interaction."""

    @pytest.fixture
    def mock_mcp(self):
        """Create mock FastMCP instance."""
        return MockFastMCP()

    @pytest.fixture
    def mock_context(self):
        """Create mock MCP context."""
        context = Mock(spec=Context)
        context.session = {}
        return context

    @pytest.fixture
    async def setup_tools(self, mock_mcp):
        """Set up all backtesting tools for testing."""
        setup_backtesting_tools(mock_mcp)
        return mock_mcp

    async def test_all_mcp_tools_registration(self, setup_tools):
        """Test that all MCP tools are properly registered."""
        mcp = setup_tools

        # Expected tools from backtesting router
        expected_tools = [
            "run_backtest",
            "optimize_strategy",
            "walk_forward_analysis",
            "monte_carlo_simulation",
            "compare_strategies",
            "list_strategies",
            "parse_strategy",
            "backtest_portfolio",
            "generate_backtest_charts",
            "generate_optimization_charts",
            "run_ml_strategy_backtest",
            "train_ml_predictor",
            "analyze_market_regimes",
            "create_strategy_ensemble",
        ]

        # Check all tools are registered
        registered_tools = set(mcp.tools.keys())
        expected_set = set(expected_tools)

        missing_tools = expected_set - registered_tools
        extra_tools = registered_tools - expected_set

        assert len(missing_tools) == 0, f"Missing tools: {missing_tools}"

        logger.info(f"✓ All {len(registered_tools)} MCP tools registered successfully")
        if extra_tools:
            logger.info(f"Additional tools found: {extra_tools}")

        # Validate each tool has proper signature
        for tool_name, tool_info in mcp.tools.items():
            assert callable(tool_info["function"]), f"Tool {tool_name} is not callable"
            assert "signature" in tool_info, f"Tool {tool_name} missing signature"

        return {
            "registered_tools": list(registered_tools),
            "tool_count": len(registered_tools),
        }

    async def test_run_backtest_tool_comprehensive(self, setup_tools, mock_context):
        """Test run_backtest tool with comprehensive parameter validation."""
        mcp = setup_tools
        tool_func = mcp.tool_functions["run_backtest"]

        # Test cases with different parameter combinations
        test_cases = [
            {
                "name": "basic_sma_cross",
                "params": {
                    "symbol": "AAPL",
                    "strategy": "sma_cross",
                    "fast_period": "10",
                    "slow_period": "20",
                },
                "should_succeed": True,
            },
            {
                "name": "rsi_strategy",
                "params": {
                    "symbol": "GOOGL",
                    "strategy": "rsi",
                    "period": "14",
                    "oversold": "30",
                    "overbought": "70",
                },
                "should_succeed": True,
            },
            {
                "name": "invalid_symbol",
                "params": {
                    "symbol": "",  # Empty symbol
                    "strategy": "sma_cross",
                },
                "should_succeed": False,
            },
            {
                "name": "invalid_strategy",
                "params": {
                    "symbol": "AAPL",
                    "strategy": "nonexistent_strategy",
                },
                "should_succeed": False,
            },
            {
                "name": "invalid_numeric_params",
                "params": {
                    "symbol": "AAPL",
                    "strategy": "sma_cross",
                    "fast_period": "invalid_number",
                },
                "should_succeed": False,
            },
        ]

        results = {}

        for test_case in test_cases:
            try:
                # Mock the VectorBT engine to avoid actual data fetching
                with patch("maverick_mcp.backtesting.VectorBTEngine") as mock_engine:
                    mock_instance = Mock()
                    mock_engine.return_value = mock_instance

                    # Mock successful backtest result
                    mock_result = {
                        "symbol": test_case["params"]["symbol"],
                        "strategy_type": test_case["params"]["strategy"],
                        "metrics": {
                            "total_return": 0.15,
                            "sharpe_ratio": 1.2,
                            "max_drawdown": -0.12,
                            "total_trades": 25,
                        },
                        "trades": [],
                        "equity_curve": [10000, 10100, 10200, 10300],
                        "drawdown_series": [0, -0.01, -0.02, 0],
                    }
                    mock_instance.run_backtest.return_value = mock_result

                    # Execute tool
                    result = await tool_func(mock_context, **test_case["params"])

                    if test_case["should_succeed"]:
                        assert isinstance(result, dict), (
                            f"Result should be dict for {test_case['name']}"
                        )
                        assert "symbol" in result, (
                            f"Missing symbol in result for {test_case['name']}"
                        )
                        assert "metrics" in result, (
                            f"Missing metrics in result for {test_case['name']}"
                        )
                        results[test_case["name"]] = {"success": True, "result": result}
                        logger.info(f"✓ {test_case['name']} succeeded as expected")
                    else:
                        # If we got here, it didn't fail as expected
                        results[test_case["name"]] = {
                            "success": False,
                            "unexpected_success": True,
                        }
                        logger.warning(
                            f"⚠ {test_case['name']} succeeded but was expected to fail"
                        )

            except Exception as e:
                if test_case["should_succeed"]:
                    results[test_case["name"]] = {"success": False, "error": str(e)}
                    logger.error(f"✗ {test_case['name']} failed unexpectedly: {e}")
                else:
                    results[test_case["name"]] = {
                        "success": True,
                        "expected_error": str(e),
                    }
                    logger.info(f"✓ {test_case['name']} failed as expected: {e}")

        # Calculate success rate
        total_tests = len(test_cases)
        successful_tests = sum(1 for r in results.values() if r.get("success", False))
        success_rate = successful_tests / total_tests

        assert success_rate >= 0.8, f"Success rate too low: {success_rate:.1%}"

        return {"test_results": results, "success_rate": success_rate}

    async def test_strategy_tools_integration(self, setup_tools, mock_context):
        """Test strategy-related tools integration."""
        mcp = setup_tools

        # Test list_strategies tool
        list_func = mcp.tool_functions["list_strategies"]
        strategies_result = await list_func(mock_context)

        assert isinstance(strategies_result, dict), "list_strategies should return dict"
        assert "available_strategies" in strategies_result, (
            "Missing available_strategies"
        )
        assert "total_count" in strategies_result, "Missing total_count"
        assert strategies_result["total_count"] > 0, "Should have strategies available"

        logger.info(f"✓ Found {strategies_result['total_count']} available strategies")

        # Test parse_strategy tool
        parse_func = mcp.tool_functions["parse_strategy"]

        parse_test_cases = [
            "Buy when RSI is below 30 and sell when above 70",
            "Use 10-day and 20-day moving average crossover",
            "MACD strategy with standard parameters",
            "Invalid strategy description that makes no sense",
        ]

        parse_results = {}
        for description in parse_test_cases:
            try:
                result = await parse_func(mock_context, description=description)
                assert isinstance(result, dict), "parse_strategy should return dict"
                assert "success" in result, "Missing success field"
                assert "strategy" in result, "Missing strategy field"

                parse_results[description] = result
                status = "✓" if result["success"] else "⚠"
                logger.info(
                    f"{status} Parsed: '{description}' -> {result['strategy'].get('strategy_type', 'unknown')}"
                )

            except Exception as e:
                parse_results[description] = {"error": str(e)}
                logger.error(f"✗ Parse failed for: '{description}' - {e}")

        return {
            "strategies_list": strategies_result,
            "parse_results": parse_results,
        }

    async def test_optimization_tools_integration(self, setup_tools, mock_context):
        """Test optimization-related tools integration."""
        mcp = setup_tools

        # Mock VectorBT engine for optimization tests
        with patch("maverick_mcp.backtesting.VectorBTEngine") as mock_engine_class:
            mock_engine = Mock()
            mock_engine_class.return_value = mock_engine

            # Mock optimization results
            mock_optimization_result = {
                "best_parameters": {"fast_period": 12, "slow_period": 26},
                "best_performance": {
                    "total_return": 0.25,
                    "sharpe_ratio": 1.8,
                    "max_drawdown": -0.08,
                },
                "optimization_results": [
                    {
                        "parameters": {"fast_period": 10, "slow_period": 20},
                        "metrics": {"sharpe_ratio": 1.2},
                    },
                    {
                        "parameters": {"fast_period": 12, "slow_period": 26},
                        "metrics": {"sharpe_ratio": 1.8},
                    },
                ],
            }
            mock_engine.optimize_parameters.return_value = mock_optimization_result

            # Test optimize_strategy tool
            optimize_func = mcp.tool_functions["optimize_strategy"]

            result = await optimize_func(
                mock_context,
                symbol="AAPL",
                strategy="sma_cross",
                optimization_level="medium",
                top_n=5,
            )

            assert isinstance(result, dict), "optimize_strategy should return dict"
            logger.info("✓ optimize_strategy tool executed successfully")

            # Test walk_forward_analysis tool
            walk_forward_func = mcp.tool_functions["walk_forward_analysis"]

            # Mock walk-forward analysis
            with patch(
                "maverick_mcp.backtesting.StrategyOptimizer"
            ) as mock_optimizer_class:
                mock_optimizer = Mock()
                mock_optimizer_class.return_value = mock_optimizer

                mock_walk_forward_result = {
                    "out_of_sample_performance": {
                        "total_return": 0.18,
                        "sharpe_ratio": 1.5,
                        "win_rate": 0.65,
                    },
                    "windows_tested": 4,
                    "average_window_performance": 0.15,
                }
                mock_optimizer.walk_forward_analysis.return_value = (
                    mock_walk_forward_result
                )

                result = await walk_forward_func(
                    mock_context,
                    symbol="AAPL",
                    strategy="sma_cross",
                    window_size=252,
                    step_size=63,
                )

                assert isinstance(result, dict), (
                    "walk_forward_analysis should return dict"
                )
                logger.info("✓ walk_forward_analysis tool executed successfully")

        return {"optimization_tests": "completed"}

    async def test_ml_tools_integration(self, setup_tools, mock_context):
        """Test ML-enhanced tools integration."""
        mcp = setup_tools

        # Test ML strategy tools
        ml_tools = [
            "run_ml_strategy_backtest",
            "train_ml_predictor",
            "analyze_market_regimes",
            "create_strategy_ensemble",
        ]

        ml_results = {}

        for tool_name in ml_tools:
            if tool_name in mcp.tool_functions:
                try:
                    tool_func = mcp.tool_functions[tool_name]

                    # Mock ML dependencies
                    with patch(
                        "maverick_mcp.backtesting.VectorBTEngine"
                    ) as mock_engine:
                        mock_instance = Mock()
                        mock_engine.return_value = mock_instance

                        # Mock historical data
                        import numpy as np
                        import pandas as pd

                        dates = pd.date_range(
                            start="2022-01-01", end="2023-12-31", freq="D"
                        )
                        mock_data = pd.DataFrame(
                            {
                                "open": np.random.uniform(100, 200, len(dates)),
                                "high": np.random.uniform(100, 200, len(dates)),
                                "low": np.random.uniform(100, 200, len(dates)),
                                "close": np.random.uniform(100, 200, len(dates)),
                                "volume": np.random.randint(
                                    1000000, 10000000, len(dates)
                                ),
                            },
                            index=dates,
                        )
                        mock_instance.get_historical_data.return_value = mock_data

                        # Test specific ML tools
                        if tool_name == "run_ml_strategy_backtest":
                            result = await tool_func(
                                mock_context,
                                symbol="AAPL",
                                strategy_type="ml_predictor",
                                model_type="random_forest",
                            )
                        elif tool_name == "train_ml_predictor":
                            result = await tool_func(
                                mock_context,
                                symbol="AAPL",
                                model_type="random_forest",
                                n_estimators=100,
                            )
                        elif tool_name == "analyze_market_regimes":
                            result = await tool_func(
                                mock_context,
                                symbol="AAPL",
                                method="hmm",
                                n_regimes=3,
                            )
                        elif tool_name == "create_strategy_ensemble":
                            result = await tool_func(
                                mock_context,
                                symbols=["AAPL", "GOOGL"],
                                base_strategies=["sma_cross", "rsi"],
                            )

                        ml_results[tool_name] = {
                            "success": True,
                            "type": type(result).__name__,
                        }
                        logger.info(f"✓ {tool_name} executed successfully")

                except Exception as e:
                    ml_results[tool_name] = {"success": False, "error": str(e)}
                    logger.error(f"✗ {tool_name} failed: {e}")
            else:
                ml_results[tool_name] = {"success": False, "error": "Tool not found"}

        return ml_results

    async def test_visualization_tools_integration(self, setup_tools, mock_context):
        """Test visualization tools integration."""
        mcp = setup_tools

        visualization_tools = [
            "generate_backtest_charts",
            "generate_optimization_charts",
        ]

        viz_results = {}

        for tool_name in visualization_tools:
            if tool_name in mcp.tool_functions:
                try:
                    tool_func = mcp.tool_functions[tool_name]

                    # Mock VectorBT engine and visualization dependencies
                    with patch(
                        "maverick_mcp.backtesting.VectorBTEngine"
                    ) as mock_engine:
                        mock_instance = Mock()
                        mock_engine.return_value = mock_instance

                        # Mock backtest result for charts
                        mock_result = {
                            "symbol": "AAPL",
                            "equity_curve": [10000, 10100, 10200, 10300, 10250],
                            "drawdown_series": [0, -0.01, -0.02, 0, -0.005],
                            "trades": [
                                {
                                    "entry_time": "2023-01-01",
                                    "exit_time": "2023-02-01",
                                    "pnl": 100,
                                },
                                {
                                    "entry_time": "2023-03-01",
                                    "exit_time": "2023-04-01",
                                    "pnl": -50,
                                },
                            ],
                            "metrics": {
                                "total_return": 0.15,
                                "sharpe_ratio": 1.2,
                                "max_drawdown": -0.08,
                                "total_trades": 10,
                            },
                        }
                        mock_instance.run_backtest.return_value = mock_result

                        # Mock visualization functions
                        with patch(
                            "maverick_mcp.backtesting.visualization.generate_equity_curve"
                        ) as mock_equity:
                            with patch(
                                "maverick_mcp.backtesting.visualization.generate_performance_dashboard"
                            ) as mock_dashboard:
                                with patch(
                                    "maverick_mcp.backtesting.visualization.generate_trade_scatter"
                                ) as mock_scatter:
                                    with patch(
                                        "maverick_mcp.backtesting.visualization.generate_optimization_heatmap"
                                    ) as mock_heatmap:
                                        # Mock chart returns (base64 strings)
                                        mock_chart_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                                        mock_equity.return_value = mock_chart_data
                                        mock_dashboard.return_value = mock_chart_data
                                        mock_scatter.return_value = mock_chart_data
                                        mock_heatmap.return_value = mock_chart_data

                                        # Execute visualization tool
                                        result = await tool_func(
                                            mock_context,
                                            symbol="AAPL",
                                            strategy="sma_cross",
                                            theme="light",
                                        )

                                        assert isinstance(result, dict), (
                                            f"{tool_name} should return dict"
                                        )

                                        # Validate chart data
                                        for chart_name, chart_data in result.items():
                                            assert isinstance(chart_data, str), (
                                                f"Chart {chart_name} should be string"
                                            )
                                            assert len(chart_data) > 0, (
                                                f"Chart {chart_name} should have data"
                                            )

                                        viz_results[tool_name] = {
                                            "success": True,
                                            "charts_generated": list(result.keys()),
                                            "chart_count": len(result),
                                        }
                                        logger.info(
                                            f"✓ {tool_name} generated {len(result)} charts successfully"
                                        )

                except Exception as e:
                    viz_results[tool_name] = {"success": False, "error": str(e)}
                    logger.error(f"✗ {tool_name} failed: {e}")
            else:
                viz_results[tool_name] = {"success": False, "error": "Tool not found"}

        return viz_results

    async def test_claude_desktop_simulation(self, setup_tools, mock_context):
        """Simulate realistic Claude Desktop usage patterns."""
        mcp = setup_tools

        # Simulate a typical Claude Desktop session
        session_commands = [
            {
                "command": "List available strategies",
                "tool": "list_strategies",
                "params": {},
            },
            {
                "command": "Run backtest for AAPL with SMA crossover",
                "tool": "run_backtest",
                "params": {
                    "symbol": "AAPL",
                    "strategy": "sma_cross",
                    "fast_period": "10",
                    "slow_period": "20",
                },
            },
            {
                "command": "Compare multiple strategies",
                "tool": "compare_strategies",
                "params": {
                    "symbol": "AAPL",
                    "strategies": ["sma_cross", "rsi", "macd"],
                },
            },
            {
                "command": "Generate charts for backtest",
                "tool": "generate_backtest_charts",
                "params": {
                    "symbol": "AAPL",
                    "strategy": "sma_cross",
                },
            },
        ]

        session_results = []

        # Mock all necessary dependencies for simulation
        with patch("maverick_mcp.backtesting.VectorBTEngine") as mock_engine_class:
            mock_engine = Mock()
            mock_engine_class.return_value = mock_engine

            # Mock results for different tools
            mock_backtest_result = {
                "symbol": "AAPL",
                "strategy_type": "sma_cross",
                "metrics": {"total_return": 0.15, "sharpe_ratio": 1.2},
                "trades": [],
                "equity_curve": [10000, 10150],
                "drawdown_series": [0, -0.02],
            }
            mock_engine.run_backtest.return_value = mock_backtest_result

            # Mock comparison results
            with patch(
                "maverick_mcp.backtesting.BacktestAnalyzer"
            ) as mock_analyzer_class:
                mock_analyzer = Mock()
                mock_analyzer_class.return_value = mock_analyzer

                mock_comparison = {
                    "strategy_rankings": [
                        {"strategy": "sma_cross", "rank": 1, "sharpe_ratio": 1.2},
                        {"strategy": "rsi", "rank": 2, "sharpe_ratio": 1.1},
                        {"strategy": "macd", "rank": 3, "sharpe_ratio": 0.9},
                    ],
                    "best_strategy": "sma_cross",
                }
                mock_analyzer.compare_strategies.return_value = mock_comparison

                # Mock visualization
                with patch(
                    "maverick_mcp.backtesting.visualization.generate_equity_curve"
                ) as mock_viz:
                    mock_viz.return_value = "mock_chart_data"

                    # Execute session commands
                    for command_info in session_commands:
                        try:
                            start_time = asyncio.get_event_loop().time()

                            tool_func = mcp.tool_functions[command_info["tool"]]
                            result = await tool_func(
                                mock_context, **command_info["params"]
                            )

                            execution_time = (
                                asyncio.get_event_loop().time() - start_time
                            )

                            session_results.append(
                                {
                                    "command": command_info["command"],
                                    "tool": command_info["tool"],
                                    "success": True,
                                    "execution_time": execution_time,
                                    "result_type": type(result).__name__,
                                }
                            )

                            logger.info(
                                f"✓ '{command_info['command']}' completed in {execution_time:.3f}s"
                            )

                        except Exception as e:
                            session_results.append(
                                {
                                    "command": command_info["command"],
                                    "tool": command_info["tool"],
                                    "success": False,
                                    "error": str(e),
                                }
                            )
                            logger.error(f"✗ '{command_info['command']}' failed: {e}")

        # Analyze session results
        total_commands = len(session_commands)
        successful_commands = sum(1 for r in session_results if r.get("success", False))
        success_rate = successful_commands / total_commands
        avg_execution_time = np.mean(
            [r.get("execution_time", 0) for r in session_results if r.get("success")]
        )

        assert success_rate >= 0.75, f"Session success rate too low: {success_rate:.1%}"
        assert avg_execution_time < 5.0, (
            f"Average execution time too high: {avg_execution_time:.3f}s"
        )

        logger.info(
            f"Claude Desktop Simulation Results:\n"
            f"  • Commands Executed: {total_commands}\n"
            f"  • Successful: {successful_commands}\n"
            f"  • Success Rate: {success_rate:.1%}\n"
            f"  • Avg Execution Time: {avg_execution_time:.3f}s"
        )

        return {
            "session_results": session_results,
            "success_rate": success_rate,
            "avg_execution_time": avg_execution_time,
        }

    async def test_tool_parameter_validation_comprehensive(
        self, setup_tools, mock_context
    ):
        """Test comprehensive parameter validation across all tools."""
        mcp = setup_tools

        validation_tests = []

        # Test parameter validation for key tools
        test_cases = [
            {
                "tool": "run_backtest",
                "valid_params": {"symbol": "AAPL", "strategy": "sma_cross"},
                "invalid_params": [
                    {"symbol": "", "strategy": "sma_cross"},  # Empty symbol
                    {"symbol": "AAPL", "strategy": ""},  # Empty strategy
                    {
                        "symbol": "AAPL",
                        "strategy": "sma_cross",
                        "fast_period": "not_a_number",
                    },  # Invalid number
                ],
            },
            {
                "tool": "optimize_strategy",
                "valid_params": {"symbol": "AAPL", "strategy": "sma_cross"},
                "invalid_params": [
                    {
                        "symbol": "AAPL",
                        "strategy": "invalid_strategy",
                    },  # Invalid strategy
                    {
                        "symbol": "AAPL",
                        "strategy": "sma_cross",
                        "top_n": -1,
                    },  # Negative top_n
                ],
            },
        ]

        for test_case in test_cases:
            tool_name = test_case["tool"]
            if tool_name in mcp.tool_functions:
                tool_func = mcp.tool_functions[tool_name]

                # Test valid parameters
                try:
                    with patch("maverick_mcp.backtesting.VectorBTEngine"):
                        await tool_func(mock_context, **test_case["valid_params"])
                        validation_tests.append(
                            {
                                "tool": tool_name,
                                "test": "valid_params",
                                "success": True,
                            }
                        )
                except Exception as e:
                    validation_tests.append(
                        {
                            "tool": tool_name,
                            "test": "valid_params",
                            "success": False,
                            "error": str(e),
                        }
                    )

                # Test invalid parameters
                for invalid_params in test_case["invalid_params"]:
                    try:
                        with patch("maverick_mcp.backtesting.VectorBTEngine"):
                            await tool_func(mock_context, **invalid_params)
                        # If we got here, validation didn't catch the error
                        validation_tests.append(
                            {
                                "tool": tool_name,
                                "test": f"invalid_params_{invalid_params}",
                                "success": False,
                                "error": "Validation should have failed but didn't",
                            }
                        )
                    except Exception as e:
                        # Expected to fail
                        validation_tests.append(
                            {
                                "tool": tool_name,
                                "test": f"invalid_params_{invalid_params}",
                                "success": True,
                                "expected_error": str(e),
                            }
                        )

        # Calculate validation success rate
        total_validation_tests = len(validation_tests)
        successful_validations = sum(
            1 for t in validation_tests if t.get("success", False)
        )
        validation_success_rate = (
            successful_validations / total_validation_tests
            if total_validation_tests > 0
            else 0
        )

        logger.info(
            f"Parameter Validation Results:\n"
            f"  • Total Validation Tests: {total_validation_tests}\n"
            f"  • Successful Validations: {successful_validations}\n"
            f"  • Validation Success Rate: {validation_success_rate:.1%}"
        )

        return {
            "validation_tests": validation_tests,
            "validation_success_rate": validation_success_rate,
        }


if __name__ == "__main__":
    # Run MCP tools integration tests
    pytest.main(
        [
            __file__,
            "-v",
            "--tb=short",
            "--asyncio-mode=auto",
            "--timeout=300",  # 5 minute timeout
            "--durations=10",
        ]
    )
