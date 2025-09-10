"""
Comprehensive tests for backtesting visualization module.

Tests cover:
- Chart generation and base64 encoding with matplotlib
- Equity curve plotting with drawdown subplots
- Trade scatter plots on price charts
- Parameter optimization heatmaps
- Portfolio allocation pie charts
- Strategy comparison line charts
- Performance dashboard table generation
- Theme support (light/dark modes)
- Image resolution and size optimization
- Error handling for malformed data
"""

import base64
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from maverick_mcp.backtesting.visualization import (
    generate_equity_curve,
    generate_optimization_heatmap,
    generate_performance_dashboard,
    generate_portfolio_allocation,
    generate_strategy_comparison,
    generate_trade_scatter,
    image_to_base64,
    set_chart_style,
)


class TestVisualizationUtilities:
    """Test suite for visualization utility functions."""

    def test_set_chart_style_light_theme(self):
        """Test light theme styling configuration."""
        set_chart_style("light")

        # Test that matplotlib parameters are set correctly
        assert plt.rcParams["axes.facecolor"] == "white"
        assert plt.rcParams["figure.facecolor"] == "white"
        assert plt.rcParams["font.size"] == 10
        assert plt.rcParams["text.color"] == "black"
        assert plt.rcParams["axes.labelcolor"] == "black"
        assert plt.rcParams["xtick.color"] == "black"
        assert plt.rcParams["ytick.color"] == "black"

    def test_set_chart_style_dark_theme(self):
        """Test dark theme styling configuration."""
        set_chart_style("dark")

        # Test that matplotlib parameters are set correctly
        assert plt.rcParams["axes.facecolor"] == "#1E1E1E"
        assert plt.rcParams["figure.facecolor"] == "#121212"
        assert plt.rcParams["font.size"] == 10
        assert plt.rcParams["text.color"] == "white"
        assert plt.rcParams["axes.labelcolor"] == "white"
        assert plt.rcParams["xtick.color"] == "white"
        assert plt.rcParams["ytick.color"] == "white"

    def test_image_to_base64_conversion(self):
        """Test image to base64 conversion with proper formatting."""
        # Create a simple test figure
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
        ax.set_title("Test Chart")

        # Convert to base64
        base64_str = image_to_base64(fig, dpi=100)

        # Test base64 string properties
        assert isinstance(base64_str, str)
        assert len(base64_str) > 100  # Should contain substantial data

        # Test that it's valid base64
        try:
            decoded_bytes = base64.b64decode(base64_str)
            assert len(decoded_bytes) > 0
        except Exception as e:
            pytest.fail(f"Invalid base64 encoding: {e}")

    def test_image_to_base64_size_optimization(self):
        """Test image size optimization and aspect ratio preservation."""
        # Create large figure
        fig, ax = plt.subplots(figsize=(20, 15))  # Large size
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])

        original_width, original_height = fig.get_size_inches()
        original_aspect = original_height / original_width

        # Convert with size constraint
        base64_str = image_to_base64(fig, dpi=100, max_width=800)

        # Test that resizing occurred
        final_width, final_height = fig.get_size_inches()
        final_aspect = final_height / final_width

        assert final_width <= 8.0  # 800px / 100dpi = 8 inches
        assert abs(final_aspect - original_aspect) < 0.01  # Aspect ratio preserved
        assert len(base64_str) > 0

    def test_image_to_base64_error_handling(self):
        """Test error handling in base64 conversion."""
        with patch(
            "matplotlib.figure.Figure.savefig", side_effect=Exception("Save error")
        ):
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3])

            result = image_to_base64(fig)
            assert result == ""  # Should return empty string on error


class TestEquityCurveGeneration:
    """Test suite for equity curve chart generation."""

    @pytest.fixture
    def sample_returns_data(self):
        """Create sample returns data for testing."""
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        returns = np.random.normal(0.001, 0.02, len(dates))
        cumulative_returns = np.cumprod(1 + returns)

        # Create drawdown series
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max * 100

        return pd.Series(cumulative_returns, index=dates), pd.Series(
            drawdown, index=dates
        )

    def test_generate_equity_curve_basic(self, sample_returns_data):
        """Test basic equity curve generation."""
        returns, drawdown = sample_returns_data

        base64_str = generate_equity_curve(returns, title="Test Equity Curve")

        assert isinstance(base64_str, str)
        assert len(base64_str) > 100

        # Test that it's valid base64 image
        try:
            decoded_bytes = base64.b64decode(base64_str)
            assert decoded_bytes.startswith(b"\x89PNG")  # PNG header
        except Exception as e:
            pytest.fail(f"Invalid PNG image: {e}")

    def test_generate_equity_curve_with_drawdown(self, sample_returns_data):
        """Test equity curve generation with drawdown subplot."""
        returns, drawdown = sample_returns_data

        base64_str = generate_equity_curve(
            returns, drawdown=drawdown, title="Equity Curve with Drawdown", theme="dark"
        )

        assert isinstance(base64_str, str)
        assert len(base64_str) > 100

        # Should be larger image due to subplot
        base64_no_dd = generate_equity_curve(returns, title="No Drawdown")
        assert len(base64_str) >= len(base64_no_dd)

    def test_generate_equity_curve_themes(self, sample_returns_data):
        """Test equity curve generation with different themes."""
        returns, _ = sample_returns_data

        light_chart = generate_equity_curve(returns, theme="light")
        dark_chart = generate_equity_curve(returns, theme="dark")

        assert len(light_chart) > 100
        assert len(dark_chart) > 100
        # Different themes should produce different images
        assert light_chart != dark_chart

    def test_generate_equity_curve_error_handling(self):
        """Test error handling in equity curve generation."""
        # Test with invalid data
        invalid_returns = pd.Series([])  # Empty series

        result = generate_equity_curve(invalid_returns)
        assert result == ""

        # Test with NaN data
        nan_returns = pd.Series([np.nan, np.nan, np.nan])
        result = generate_equity_curve(nan_returns)
        assert result == ""


class TestTradeScatterGeneration:
    """Test suite for trade scatter plot generation."""

    @pytest.fixture
    def sample_trade_data(self):
        """Create sample trade data for testing."""
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        prices = pd.Series(100 + np.random.walk(len(dates)), index=dates)

        # Create sample trades
        trade_dates = dates[::30]  # Every 30 days
        trades = []

        for i, trade_date in enumerate(trade_dates):
            if i % 2 == 0:  # Entry
                trades.append(
                    {
                        "date": trade_date,
                        "price": prices.loc[trade_date],
                        "type": "entry",
                    }
                )
            else:  # Exit
                trades.append(
                    {
                        "date": trade_date,
                        "price": prices.loc[trade_date],
                        "type": "exit",
                    }
                )

        trades_df = pd.DataFrame(trades).set_index("date")
        return prices, trades_df

    def test_generate_trade_scatter_basic(self, sample_trade_data):
        """Test basic trade scatter plot generation."""
        prices, trades = sample_trade_data

        base64_str = generate_trade_scatter(prices, trades, title="Trade Scatter Plot")

        assert isinstance(base64_str, str)
        assert len(base64_str) > 100

        # Verify valid PNG
        try:
            decoded_bytes = base64.b64decode(base64_str)
            assert decoded_bytes.startswith(b"\x89PNG")
        except Exception as e:
            pytest.fail(f"Invalid PNG image: {e}")

    def test_generate_trade_scatter_themes(self, sample_trade_data):
        """Test trade scatter plots with different themes."""
        prices, trades = sample_trade_data

        light_chart = generate_trade_scatter(prices, trades, theme="light")
        dark_chart = generate_trade_scatter(prices, trades, theme="dark")

        assert len(light_chart) > 100
        assert len(dark_chart) > 100
        assert light_chart != dark_chart

    def test_generate_trade_scatter_empty_trades(self, sample_trade_data):
        """Test trade scatter plot with empty trade data."""
        prices, _ = sample_trade_data
        empty_trades = pd.DataFrame(columns=["price", "type"])

        result = generate_trade_scatter(prices, empty_trades)
        assert result == ""

    def test_generate_trade_scatter_error_handling(self):
        """Test error handling in trade scatter generation."""
        # Test with mismatched data
        prices = pd.Series([1, 2, 3])
        trades = pd.DataFrame({"price": [10, 20], "type": ["entry", "exit"]})

        # Should handle gracefully
        result = generate_trade_scatter(prices, trades)
        # Might return empty string or valid chart depending on implementation
        assert isinstance(result, str)


class TestOptimizationHeatmapGeneration:
    """Test suite for parameter optimization heatmap generation."""

    @pytest.fixture
    def sample_optimization_data(self):
        """Create sample optimization results for testing."""
        parameters = ["param1", "param2", "param3"]
        results = {}

        for p1 in parameters:
            results[p1] = {}
            for p2 in parameters:
                # Create some performance metric
                results[p1][p2] = np.random.uniform(0.5, 2.0)

        return results

    def test_generate_optimization_heatmap_basic(self, sample_optimization_data):
        """Test basic optimization heatmap generation."""
        base64_str = generate_optimization_heatmap(
            sample_optimization_data, title="Parameter Optimization Heatmap"
        )

        assert isinstance(base64_str, str)
        assert len(base64_str) > 100

        # Verify valid PNG
        try:
            decoded_bytes = base64.b64decode(base64_str)
            assert decoded_bytes.startswith(b"\x89PNG")
        except Exception as e:
            pytest.fail(f"Invalid PNG image: {e}")

    def test_generate_optimization_heatmap_themes(self, sample_optimization_data):
        """Test optimization heatmap with different themes."""
        light_chart = generate_optimization_heatmap(
            sample_optimization_data, theme="light"
        )
        dark_chart = generate_optimization_heatmap(
            sample_optimization_data, theme="dark"
        )

        assert len(light_chart) > 100
        assert len(dark_chart) > 100
        assert light_chart != dark_chart

    def test_generate_optimization_heatmap_empty_data(self):
        """Test heatmap generation with empty data."""
        empty_data = {}

        result = generate_optimization_heatmap(empty_data)
        assert result == ""

    def test_generate_optimization_heatmap_error_handling(self):
        """Test error handling in heatmap generation."""
        # Test with malformed data
        malformed_data = {"param1": "not_a_dict"}

        result = generate_optimization_heatmap(malformed_data)
        assert result == ""


class TestPortfolioAllocationGeneration:
    """Test suite for portfolio allocation chart generation."""

    @pytest.fixture
    def sample_allocation_data(self):
        """Create sample allocation data for testing."""
        return {
            "AAPL": 0.25,
            "GOOGL": 0.20,
            "MSFT": 0.15,
            "TSLA": 0.15,
            "AMZN": 0.10,
            "Cash": 0.15,
        }

    def test_generate_portfolio_allocation_basic(self, sample_allocation_data):
        """Test basic portfolio allocation chart generation."""
        base64_str = generate_portfolio_allocation(
            sample_allocation_data, title="Portfolio Allocation"
        )

        assert isinstance(base64_str, str)
        assert len(base64_str) > 100

        # Verify valid PNG
        try:
            decoded_bytes = base64.b64decode(base64_str)
            assert decoded_bytes.startswith(b"\x89PNG")
        except Exception as e:
            pytest.fail(f"Invalid PNG image: {e}")

    def test_generate_portfolio_allocation_themes(self, sample_allocation_data):
        """Test portfolio allocation with different themes."""
        light_chart = generate_portfolio_allocation(
            sample_allocation_data, theme="light"
        )
        dark_chart = generate_portfolio_allocation(sample_allocation_data, theme="dark")

        assert len(light_chart) > 100
        assert len(dark_chart) > 100
        assert light_chart != dark_chart

    def test_generate_portfolio_allocation_empty_data(self):
        """Test allocation chart with empty data."""
        empty_data = {}

        result = generate_portfolio_allocation(empty_data)
        assert result == ""

    def test_generate_portfolio_allocation_single_asset(self):
        """Test allocation chart with single asset."""
        single_asset = {"AAPL": 1.0}

        result = generate_portfolio_allocation(single_asset)
        assert isinstance(result, str)
        assert len(result) > 100  # Should still generate valid chart

    def test_generate_portfolio_allocation_error_handling(self):
        """Test error handling in allocation chart generation."""
        # Test with invalid allocation values
        invalid_data = {"AAPL": "invalid_value"}

        result = generate_portfolio_allocation(invalid_data)
        assert result == ""


class TestStrategyComparisonGeneration:
    """Test suite for strategy comparison chart generation."""

    @pytest.fixture
    def sample_strategy_data(self):
        """Create sample strategy comparison data."""
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")

        strategies = {
            "Momentum": pd.Series(
                np.cumprod(1 + np.random.normal(0.0008, 0.015, len(dates))), index=dates
            ),
            "Mean Reversion": pd.Series(
                np.cumprod(1 + np.random.normal(0.0005, 0.012, len(dates))), index=dates
            ),
            "Breakout": pd.Series(
                np.cumprod(1 + np.random.normal(0.0012, 0.020, len(dates))), index=dates
            ),
        }

        return strategies

    def test_generate_strategy_comparison_basic(self, sample_strategy_data):
        """Test basic strategy comparison chart generation."""
        base64_str = generate_strategy_comparison(
            sample_strategy_data, title="Strategy Performance Comparison"
        )

        assert isinstance(base64_str, str)
        assert len(base64_str) > 100

        # Verify valid PNG
        try:
            decoded_bytes = base64.b64decode(base64_str)
            assert decoded_bytes.startswith(b"\x89PNG")
        except Exception as e:
            pytest.fail(f"Invalid PNG image: {e}")

    def test_generate_strategy_comparison_themes(self, sample_strategy_data):
        """Test strategy comparison with different themes."""
        light_chart = generate_strategy_comparison(sample_strategy_data, theme="light")
        dark_chart = generate_strategy_comparison(sample_strategy_data, theme="dark")

        assert len(light_chart) > 100
        assert len(dark_chart) > 100
        assert light_chart != dark_chart

    def test_generate_strategy_comparison_single_strategy(self):
        """Test comparison chart with single strategy."""
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        single_strategy = {
            "Only Strategy": pd.Series(
                np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates))), index=dates
            )
        }

        result = generate_strategy_comparison(single_strategy)
        assert isinstance(result, str)
        assert len(result) > 100

    def test_generate_strategy_comparison_empty_data(self):
        """Test comparison chart with empty data."""
        empty_data = {}

        result = generate_strategy_comparison(empty_data)
        assert result == ""

    def test_generate_strategy_comparison_error_handling(self):
        """Test error handling in strategy comparison generation."""
        # Test with mismatched data lengths
        dates1 = pd.date_range(start="2023-01-01", end="2023-06-30", freq="D")
        dates2 = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")

        mismatched_data = {
            "Strategy1": pd.Series(np.random.normal(0, 1, len(dates1)), index=dates1),
            "Strategy2": pd.Series(np.random.normal(0, 1, len(dates2)), index=dates2),
        }

        # Should handle gracefully
        result = generate_strategy_comparison(mismatched_data)
        assert isinstance(result, str)  # Might be empty or valid


class TestPerformanceDashboardGeneration:
    """Test suite for performance dashboard generation."""

    @pytest.fixture
    def sample_metrics_data(self):
        """Create sample performance metrics for testing."""
        return {
            "Total Return": 0.156,
            "Sharpe Ratio": 1.25,
            "Max Drawdown": -0.082,
            "Win Rate": 0.583,
            "Profit Factor": 1.35,
            "Total Trades": 24,
            "Annualized Return": 0.18,
            "Volatility": 0.16,
            "Calmar Ratio": 1.10,
            "Best Trade": 0.12,
        }

    def test_generate_performance_dashboard_basic(self, sample_metrics_data):
        """Test basic performance dashboard generation."""
        base64_str = generate_performance_dashboard(
            sample_metrics_data, title="Performance Dashboard"
        )

        assert isinstance(base64_str, str)
        assert len(base64_str) > 100

        # Verify valid PNG
        try:
            decoded_bytes = base64.b64decode(base64_str)
            assert decoded_bytes.startswith(b"\x89PNG")
        except Exception as e:
            pytest.fail(f"Invalid PNG image: {e}")

    def test_generate_performance_dashboard_themes(self, sample_metrics_data):
        """Test performance dashboard with different themes."""
        light_chart = generate_performance_dashboard(sample_metrics_data, theme="light")
        dark_chart = generate_performance_dashboard(sample_metrics_data, theme="dark")

        assert len(light_chart) > 100
        assert len(dark_chart) > 100
        assert light_chart != dark_chart

    def test_generate_performance_dashboard_mixed_data_types(self):
        """Test dashboard with mixed data types."""
        mixed_metrics = {
            "Total Return": 0.156,
            "Strategy": "Momentum",
            "Symbol": "AAPL",
            "Duration": "365 days",
            "Sharpe Ratio": 1.25,
            "Status": "Completed",
        }

        result = generate_performance_dashboard(mixed_metrics)
        assert isinstance(result, str)
        assert len(result) > 100

    def test_generate_performance_dashboard_empty_data(self):
        """Test dashboard with empty metrics."""
        empty_metrics = {}

        result = generate_performance_dashboard(empty_metrics)
        assert result == ""

    def test_generate_performance_dashboard_large_dataset(self):
        """Test dashboard with large number of metrics."""
        large_metrics = {f"Metric_{i}": np.random.uniform(-1, 2) for i in range(50)}

        result = generate_performance_dashboard(large_metrics)
        assert isinstance(result, str)
        # Might be empty if table becomes too large, or valid if handled properly

    def test_generate_performance_dashboard_error_handling(self):
        """Test error handling in dashboard generation."""
        # Test with invalid data that might cause table generation to fail
        problematic_metrics = {
            "Valid Metric": 1.25,
            "Problematic": [1, 2, 3],  # List instead of scalar
            "Another Valid": 0.85,
        }

        result = generate_performance_dashboard(problematic_metrics)
        assert isinstance(result, str)


class TestVisualizationIntegration:
    """Integration tests for visualization functions working together."""

    def test_consistent_theming_across_charts(self):
        """Test that theming is consistent across different chart types."""
        # Create sample data for different chart types
        dates = pd.date_range(start="2023-01-01", end="2023-06-30", freq="D")
        returns = pd.Series(
            np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates))), index=dates
        )

        allocation = {"AAPL": 0.4, "GOOGL": 0.3, "MSFT": 0.3}
        metrics = {"Return": 0.15, "Sharpe": 1.2, "Drawdown": -0.08}

        # Generate charts with same theme
        equity_chart = generate_equity_curve(returns, theme="dark")
        allocation_chart = generate_portfolio_allocation(allocation, theme="dark")
        dashboard_chart = generate_performance_dashboard(metrics, theme="dark")

        # All should generate valid base64 strings
        charts = [equity_chart, allocation_chart, dashboard_chart]
        for chart in charts:
            assert isinstance(chart, str)
            assert len(chart) > 100

            # Verify valid PNG
            try:
                decoded_bytes = base64.b64decode(chart)
                assert decoded_bytes.startswith(b"\x89PNG")
            except Exception as e:
                pytest.fail(f"Invalid PNG in themed charts: {e}")

    def test_memory_cleanup_after_chart_generation(self):
        """Test that matplotlib figures are properly cleaned up."""
        import matplotlib.pyplot as plt

        initial_figure_count = len(plt.get_fignums())

        # Generate multiple charts
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        returns = pd.Series(
            np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates))), index=dates
        )

        for i in range(10):
            chart = generate_equity_curve(returns, title=f"Test Chart {i}")
            assert len(chart) > 0

        final_figure_count = len(plt.get_fignums())

        # Figure count should not have increased (figures should be closed)
        assert final_figure_count <= initial_figure_count + 1  # Allow for 1 open figure

    def test_chart_generation_performance_benchmark(self, benchmark_timer):
        """Test chart generation performance benchmarks."""
        # Create substantial dataset
        dates = pd.date_range(
            start="2023-01-01", end="2023-12-31", freq="H"
        )  # Hourly data
        returns = pd.Series(
            np.cumprod(1 + np.random.normal(0.0001, 0.005, len(dates))), index=dates
        )

        with benchmark_timer() as timer:
            chart = generate_equity_curve(returns, title="Performance Test")

        # Should complete within reasonable time even with large dataset
        assert timer.elapsed < 5.0  # < 5 seconds
        assert len(chart) > 100  # Valid chart generated

    def test_concurrent_chart_generation(self):
        """Test concurrent chart generation doesn't cause conflicts."""
        import queue
        import threading

        results_queue = queue.Queue()
        error_queue = queue.Queue()

        def generate_chart(thread_id):
            try:
                dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
                returns = pd.Series(
                    np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates))),
                    index=dates,
                )

                chart = generate_equity_curve(returns, title=f"Thread {thread_id}")
                results_queue.put((thread_id, len(chart)))
            except Exception as e:
                error_queue.put(f"Thread {thread_id}: {e}")

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=generate_chart, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)

        # Check results
        assert error_queue.empty(), f"Errors: {list(error_queue.queue)}"
        assert results_queue.qsize() == 5

        # All should have generated valid charts
        while not results_queue.empty():
            thread_id, chart_length = results_queue.get()
            assert chart_length > 100


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main([__file__, "-v", "--tb=short", "--asyncio-mode=auto"])
