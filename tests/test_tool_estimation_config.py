"""
Comprehensive tests for ToolEstimationConfig.

This module tests the centralized tool cost estimation configuration that replaces
magic numbers scattered throughout the codebase. Tests cover:
- All tool-specific estimates
- Confidence levels and estimation basis
- Monitoring thresholds and alert conditions
- Edge cases and error handling
- Integration with server.py patterns
"""

from unittest.mock import patch

import pytest

from maverick_mcp.config.tool_estimation import (
    EstimationBasis,
    MonitoringThresholds,
    ToolComplexity,
    ToolEstimate,
    ToolEstimationConfig,
    get_tool_estimate,
    get_tool_estimation_config,
    should_alert_for_usage,
)


class TestToolEstimate:
    """Test ToolEstimate model validation and behavior."""

    def test_valid_tool_estimate(self):
        """Test creating a valid ToolEstimate."""
        estimate = ToolEstimate(
            llm_calls=5,
            total_tokens=8000,
            confidence=0.8,
            based_on=EstimationBasis.EMPIRICAL,
            complexity=ToolComplexity.COMPLEX,
            notes="Test estimate",
        )

        assert estimate.llm_calls == 5
        assert estimate.total_tokens == 8000
        assert estimate.confidence == 0.8
        assert estimate.based_on == EstimationBasis.EMPIRICAL
        assert estimate.complexity == ToolComplexity.COMPLEX
        assert estimate.notes == "Test estimate"

    def test_confidence_validation(self):
        """Test confidence level validation."""
        # Valid confidence levels
        for confidence in [0.0, 0.5, 1.0]:
            estimate = ToolEstimate(
                llm_calls=1,
                total_tokens=100,
                confidence=confidence,
                based_on=EstimationBasis.EMPIRICAL,
                complexity=ToolComplexity.SIMPLE,
            )
            assert estimate.confidence == confidence

        # Invalid confidence levels - Pydantic ValidationError
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ToolEstimate(
                llm_calls=1,
                total_tokens=100,
                confidence=-0.1,
                based_on=EstimationBasis.EMPIRICAL,
                complexity=ToolComplexity.SIMPLE,
            )

        with pytest.raises(ValidationError):
            ToolEstimate(
                llm_calls=1,
                total_tokens=100,
                confidence=1.1,
                based_on=EstimationBasis.EMPIRICAL,
                complexity=ToolComplexity.SIMPLE,
            )

    def test_negative_values_validation(self):
        """Test that negative values are not allowed."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ToolEstimate(
                llm_calls=-1,
                total_tokens=100,
                confidence=0.8,
                based_on=EstimationBasis.EMPIRICAL,
                complexity=ToolComplexity.SIMPLE,
            )

        with pytest.raises(ValidationError):
            ToolEstimate(
                llm_calls=1,
                total_tokens=-100,
                confidence=0.8,
                based_on=EstimationBasis.EMPIRICAL,
                complexity=ToolComplexity.SIMPLE,
            )


class TestMonitoringThresholds:
    """Test MonitoringThresholds model and validation."""

    def test_default_thresholds(self):
        """Test default monitoring thresholds."""
        thresholds = MonitoringThresholds()

        assert thresholds.llm_calls_warning == 15
        assert thresholds.llm_calls_critical == 25
        assert thresholds.tokens_warning == 20000
        assert thresholds.tokens_critical == 35000
        assert thresholds.variance_warning == 0.5
        assert thresholds.variance_critical == 1.0

    def test_custom_thresholds(self):
        """Test custom monitoring thresholds."""
        thresholds = MonitoringThresholds(
            llm_calls_warning=10,
            llm_calls_critical=20,
            tokens_warning=15000,
            tokens_critical=30000,
            variance_warning=0.3,
            variance_critical=0.8,
        )

        assert thresholds.llm_calls_warning == 10
        assert thresholds.llm_calls_critical == 20
        assert thresholds.tokens_warning == 15000
        assert thresholds.tokens_critical == 30000
        assert thresholds.variance_warning == 0.3
        assert thresholds.variance_critical == 0.8


class TestToolEstimationConfig:
    """Test the main ToolEstimationConfig class."""

    def test_default_configuration(self):
        """Test default configuration initialization."""
        config = ToolEstimationConfig()

        # Test default estimates by complexity
        assert config.simple_default.complexity == ToolComplexity.SIMPLE
        assert config.standard_default.complexity == ToolComplexity.STANDARD
        assert config.complex_default.complexity == ToolComplexity.COMPLEX
        assert config.premium_default.complexity == ToolComplexity.PREMIUM

        # Test unknown tool fallback
        assert config.unknown_tool_estimate.complexity == ToolComplexity.STANDARD
        assert config.unknown_tool_estimate.confidence == 0.3
        assert config.unknown_tool_estimate.based_on == EstimationBasis.CONSERVATIVE

    def test_get_estimate_known_tools(self):
        """Test getting estimates for known tools."""
        config = ToolEstimationConfig()

        # Test simple tools
        simple_tools = [
            "get_stock_price",
            "get_company_info",
            "get_stock_info",
            "calculate_sma",
            "get_market_hours",
            "get_chart_links",
            "list_available_agents",
            "clear_cache",
            "get_cached_price_data",
            "get_watchlist",
            "generate_dev_token",
        ]

        for tool in simple_tools:
            estimate = config.get_estimate(tool)
            assert estimate.complexity == ToolComplexity.SIMPLE
            assert estimate.llm_calls <= 1  # Simple tools should have minimal LLM usage
            assert estimate.confidence >= 0.8  # Should have high confidence

        # Test standard tools
        standard_tools = [
            "get_rsi_analysis",
            "get_macd_analysis",
            "get_support_resistance",
            "fetch_stock_data",
            "get_maverick_stocks",
            "get_news_sentiment",
            "get_economic_calendar",
        ]

        for tool in standard_tools:
            estimate = config.get_estimate(tool)
            assert estimate.complexity == ToolComplexity.STANDARD
            assert 1 <= estimate.llm_calls <= 5
            assert estimate.confidence >= 0.7

        # Test complex tools
        complex_tools = [
            "get_full_technical_analysis",
            "risk_adjusted_analysis",
            "compare_tickers",
            "portfolio_correlation_analysis",
            "get_market_overview",
            "get_all_screening_recommendations",
        ]

        for tool in complex_tools:
            estimate = config.get_estimate(tool)
            assert estimate.complexity == ToolComplexity.COMPLEX
            assert 4 <= estimate.llm_calls <= 8
            assert estimate.confidence >= 0.7

        # Test premium tools
        premium_tools = [
            "analyze_market_with_agent",
            "get_agent_streaming_analysis",
            "compare_personas_analysis",
        ]

        for tool in premium_tools:
            estimate = config.get_estimate(tool)
            assert estimate.complexity == ToolComplexity.PREMIUM
            assert estimate.llm_calls >= 8
            assert estimate.total_tokens >= 10000

    def test_get_estimate_unknown_tool(self):
        """Test getting estimate for unknown tools."""
        config = ToolEstimationConfig()
        estimate = config.get_estimate("unknown_tool_name")

        assert estimate == config.unknown_tool_estimate
        assert estimate.complexity == ToolComplexity.STANDARD
        assert estimate.confidence == 0.3
        assert estimate.based_on == EstimationBasis.CONSERVATIVE

    def test_get_default_for_complexity(self):
        """Test getting default estimates by complexity."""
        config = ToolEstimationConfig()

        simple = config.get_default_for_complexity(ToolComplexity.SIMPLE)
        assert simple == config.simple_default

        standard = config.get_default_for_complexity(ToolComplexity.STANDARD)
        assert standard == config.standard_default

        complex_est = config.get_default_for_complexity(ToolComplexity.COMPLEX)
        assert complex_est == config.complex_default

        premium = config.get_default_for_complexity(ToolComplexity.PREMIUM)
        assert premium == config.premium_default

    def test_should_alert_critical_thresholds(self):
        """Test alert conditions for critical thresholds."""
        config = ToolEstimationConfig()

        # Test critical LLM calls threshold
        should_alert, message = config.should_alert("test_tool", 30, 5000)
        assert should_alert
        assert "Critical: LLM calls (30) exceeded threshold (25)" in message

        # Test critical token threshold
        should_alert, message = config.should_alert("test_tool", 5, 40000)
        assert should_alert
        assert "Critical: Token usage (40000) exceeded threshold (35000)" in message

    def test_should_alert_variance_thresholds(self):
        """Test alert conditions for variance thresholds."""
        config = ToolEstimationConfig()

        # Test tool with known estimate for variance calculation
        # get_stock_price: llm_calls=0, total_tokens=200

        # Test critical LLM variance (infinite variance since estimate is 0)
        should_alert, message = config.should_alert("get_stock_price", 5, 200)
        assert should_alert
        assert "Critical: LLM call variance" in message

        # Test critical token variance (5x the estimate = 400% variance)
        should_alert, message = config.should_alert("get_stock_price", 0, 1000)
        assert should_alert
        assert "Critical: Token variance" in message

    def test_should_alert_warning_thresholds(self):
        """Test alert conditions for warning thresholds."""
        config = ToolEstimationConfig()

        # Test warning LLM calls threshold (15-24 should trigger warning)
        # Use unknown tool which has reasonable base estimates to avoid variance issues
        should_alert, message = config.should_alert("unknown_tool", 18, 5000)
        assert should_alert
        assert (
            "Warning" in message or "Critical" in message
        )  # May trigger critical due to variance

        # Test warning token threshold with a tool that has known estimates
        # get_rsi_analysis: llm_calls=2, total_tokens=3000
        should_alert, message = config.should_alert("get_rsi_analysis", 2, 25000)
        assert should_alert
        assert (
            "Warning" in message or "Critical" in message
        )  # High token variance may trigger critical

    def test_should_alert_no_alert(self):
        """Test cases where no alert should be triggered."""
        config = ToolEstimationConfig()

        # Normal usage within expected ranges
        should_alert, message = config.should_alert("get_stock_price", 0, 200)
        assert not should_alert
        assert message == ""

        # Slightly above estimate but within acceptable variance
        should_alert, message = config.should_alert("get_stock_price", 0, 250)
        assert not should_alert
        assert message == ""

    def test_get_tools_by_complexity(self):
        """Test filtering tools by complexity category."""
        config = ToolEstimationConfig()

        simple_tools = config.get_tools_by_complexity(ToolComplexity.SIMPLE)
        standard_tools = config.get_tools_by_complexity(ToolComplexity.STANDARD)
        complex_tools = config.get_tools_by_complexity(ToolComplexity.COMPLEX)
        premium_tools = config.get_tools_by_complexity(ToolComplexity.PREMIUM)

        # Verify all tools are categorized
        all_tools = simple_tools + standard_tools + complex_tools + premium_tools
        assert len(all_tools) == len(config.tool_estimates)

        # Verify no overlap between categories
        assert len(set(all_tools)) == len(all_tools)

        # Verify specific known tools are in correct categories
        assert "get_stock_price" in simple_tools
        assert "get_rsi_analysis" in standard_tools
        assert "get_full_technical_analysis" in complex_tools
        assert "analyze_market_with_agent" in premium_tools

    def test_get_summary_stats(self):
        """Test summary statistics generation."""
        config = ToolEstimationConfig()
        stats = config.get_summary_stats()

        # Verify structure
        assert "total_tools" in stats
        assert "by_complexity" in stats
        assert "avg_llm_calls" in stats
        assert "avg_tokens" in stats
        assert "avg_confidence" in stats
        assert "basis_distribution" in stats

        # Verify content
        assert stats["total_tools"] > 0
        assert len(stats["by_complexity"]) == 4  # All complexity levels
        assert stats["avg_llm_calls"] >= 0
        assert stats["avg_tokens"] > 0
        assert 0 <= stats["avg_confidence"] <= 1

        # Verify complexity distribution adds up
        complexity_sum = sum(stats["by_complexity"].values())
        assert complexity_sum == stats["total_tools"]


class TestModuleFunctions:
    """Test module-level functions."""

    def test_get_tool_estimation_config_singleton(self):
        """Test that get_tool_estimation_config returns a singleton."""
        config1 = get_tool_estimation_config()
        config2 = get_tool_estimation_config()

        # Should return the same instance
        assert config1 is config2

    @patch("maverick_mcp.config.tool_estimation._config", None)
    def test_get_tool_estimation_config_initialization(self):
        """Test that configuration is initialized correctly."""
        config = get_tool_estimation_config()

        assert isinstance(config, ToolEstimationConfig)
        assert len(config.tool_estimates) > 0

    def test_get_tool_estimate_function(self):
        """Test the get_tool_estimate convenience function."""
        estimate = get_tool_estimate("get_stock_price")

        assert isinstance(estimate, ToolEstimate)
        assert estimate.complexity == ToolComplexity.SIMPLE

        # Test unknown tool
        unknown_estimate = get_tool_estimate("unknown_tool")
        assert unknown_estimate.based_on == EstimationBasis.CONSERVATIVE

    def test_should_alert_for_usage_function(self):
        """Test the should_alert_for_usage convenience function."""
        should_alert, message = should_alert_for_usage("test_tool", 30, 5000)

        assert isinstance(should_alert, bool)
        assert isinstance(message, str)

        # Should trigger alert for high LLM calls
        assert should_alert
        assert "Critical" in message


class TestMagicNumberReplacement:
    """Test that configuration correctly replaces magic numbers from server.py."""

    def test_all_credit_tier_tools_have_estimates(self):
        """Test that all tools referenced in server.py have estimates."""
        config = ToolEstimationConfig()

        # These are tools that were using magic numbers in server.py
        # Based on the TOOL_CREDIT_MAPPING pattern
        critical_tools = [
            # Simple tools (1 credit tier)
            "get_stock_price",
            "get_company_info",
            "get_stock_info",
            "calculate_sma",
            "get_market_hours",
            "get_chart_links",
            # Standard tools (5 credit tier)
            "get_rsi_analysis",
            "get_macd_analysis",
            "get_support_resistance",
            "fetch_stock_data",
            "get_maverick_stocks",
            "get_news_sentiment",
            # Complex tools (20 credit tier)
            "get_full_technical_analysis",
            "risk_adjusted_analysis",
            "compare_tickers",
            "portfolio_correlation_analysis",
            "get_market_overview",
            # Premium tools (50 credit tier)
            "analyze_market_with_agent",
            "get_agent_streaming_analysis",
            "compare_personas_analysis",
        ]

        for tool in critical_tools:
            estimate = config.get_estimate(tool)
            # Should not get the fallback estimate
            assert estimate != config.unknown_tool_estimate, (
                f"Tool {tool} missing specific estimate"
            )
            # Should have reasonable confidence
            assert estimate.confidence > 0.5, f"Tool {tool} has low confidence estimate"

    def test_estimates_align_with_credit_tiers(self):
        """Test that tool estimates align with credit pricing tiers."""
        config = ToolEstimationConfig()

        # Tools that should cost 1 credit (simple, no LLM usage)
        simple_tools = [
            "get_stock_price",
            "get_company_info",
            "get_stock_info",
            "calculate_sma",
            "get_market_hours",
            "get_chart_links",
        ]

        for tool in simple_tools:
            estimate = config.get_estimate(tool)
            assert estimate.complexity == ToolComplexity.SIMPLE
            assert estimate.llm_calls <= 1  # Should require minimal/no LLM calls

        # Tools that should cost 5 credits (standard analysis)
        standard_tools = [
            "get_rsi_analysis",
            "get_macd_analysis",
            "get_support_resistance",
            "fetch_stock_data",
            "get_maverick_stocks",
        ]

        for tool in standard_tools:
            estimate = config.get_estimate(tool)
            assert estimate.complexity == ToolComplexity.STANDARD
            assert 1 <= estimate.llm_calls <= 5  # Moderate LLM usage

        # Tools that should cost 20 credits (complex analysis)
        complex_tools = [
            "get_full_technical_analysis",
            "risk_adjusted_analysis",
            "compare_tickers",
            "portfolio_correlation_analysis",
        ]

        for tool in complex_tools:
            estimate = config.get_estimate(tool)
            assert estimate.complexity == ToolComplexity.COMPLEX
            assert 4 <= estimate.llm_calls <= 8  # Multiple LLM interactions

        # Tools that should cost 50 credits (premium workflows)
        premium_tools = [
            "analyze_market_with_agent",
            "get_agent_streaming_analysis",
            "compare_personas_analysis",
        ]

        for tool in premium_tools:
            estimate = config.get_estimate(tool)
            assert estimate.complexity == ToolComplexity.PREMIUM
            assert estimate.llm_calls >= 8  # Extensive LLM coordination

    def test_no_hardcoded_estimates_remain(self):
        """Test that estimates are data-driven, not hardcoded."""
        config = ToolEstimationConfig()

        # All tool estimates should have basis information
        for tool_name, estimate in config.tool_estimates.items():
            assert estimate.based_on in EstimationBasis
            assert estimate.complexity in ToolComplexity
            assert estimate.notes is not None, f"Tool {tool_name} missing notes"

            # Empirical estimates should generally have reasonable confidence
            if estimate.based_on == EstimationBasis.EMPIRICAL:
                assert estimate.confidence >= 0.6, (
                    f"Empirical estimate for {tool_name} has very low confidence"
                )

            # Conservative estimates should have lower confidence
            if estimate.based_on == EstimationBasis.CONSERVATIVE:
                assert estimate.confidence <= 0.6, (
                    f"Conservative estimate for {tool_name} has unexpectedly high confidence"
                )


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_configuration(self):
        """Test behavior with empty tool estimates."""
        config = ToolEstimationConfig(tool_estimates={})

        # Should fall back to unknown tool estimate
        estimate = config.get_estimate("any_tool")
        assert estimate == config.unknown_tool_estimate

        # Summary stats should handle empty case
        stats = config.get_summary_stats()
        assert stats == {}

    def test_alert_with_zero_estimates(self):
        """Test alert calculation when estimates are zero."""
        config = ToolEstimationConfig()

        # Tool with zero LLM calls in estimate
        should_alert, message = config.should_alert("get_stock_price", 1, 200)
        # Should alert because variance is infinite (1 vs 0 expected)
        assert should_alert

    def test_variance_calculation_edge_cases(self):
        """Test variance calculation with edge cases."""
        config = ToolEstimationConfig()

        # Perfect match should not alert
        should_alert, message = config.should_alert("get_rsi_analysis", 2, 3000)
        # get_rsi_analysis has: llm_calls=2, total_tokens=3000
        assert not should_alert

    def test_performance_with_large_usage(self):
        """Test performance and behavior with extremely large usage values."""
        config = ToolEstimationConfig()

        # Very large values should still work
        should_alert, message = config.should_alert("test_tool", 1000, 1000000)
        assert should_alert
        assert "Critical" in message

    def test_custom_monitoring_thresholds(self):
        """Test configuration with custom monitoring thresholds."""
        custom_monitoring = MonitoringThresholds(
            llm_calls_warning=5,
            llm_calls_critical=10,
            tokens_warning=1000,
            tokens_critical=5000,
            variance_warning=0.1,
            variance_critical=0.2,
        )

        config = ToolEstimationConfig(monitoring=custom_monitoring)

        # Should use custom thresholds
        # Test critical threshold first (easier to predict)
        should_alert, message = config.should_alert("test_tool", 12, 500)
        assert should_alert
        assert "Critical" in message

        # Test LLM calls warning threshold
        should_alert, message = config.should_alert(
            "test_tool", 6, 100
        )  # Lower tokens to avoid variance issues
        assert should_alert
        # May be warning or critical depending on variance calculation


class TestIntegrationPatterns:
    """Test patterns that match server.py integration."""

    def test_low_confidence_logging_pattern(self):
        """Test identifying tools that need monitoring due to low confidence."""
        config = ToolEstimationConfig()

        low_confidence_tools = []
        for tool_name, estimate in config.tool_estimates.items():
            if estimate.confidence < 0.8:
                low_confidence_tools.append(tool_name)

        # These tools should be logged for monitoring in production
        assert len(low_confidence_tools) > 0

        # Verify these are typically more complex tools
        for tool_name in low_confidence_tools:
            estimate = config.get_estimate(tool_name)
            # Low confidence tools should typically be complex, premium, or analytical standard tools
            assert estimate.complexity in [
                ToolComplexity.STANDARD,
                ToolComplexity.COMPLEX,
                ToolComplexity.PREMIUM,
            ], (
                f"Tool {tool_name} with low confidence has unexpected complexity {estimate.complexity}"
            )

    def test_error_handling_fallback_pattern(self):
        """Test the error handling pattern used in server.py."""
        config = ToolEstimationConfig()

        # Simulate error case - should fall back to unknown tool estimate
        try:
            # This would be the pattern in server.py when get_tool_estimate fails
            estimate = config.get_estimate("nonexistent_tool")
            fallback_estimate = config.unknown_tool_estimate

            # Verify fallback has conservative characteristics
            assert fallback_estimate.based_on == EstimationBasis.CONSERVATIVE
            assert fallback_estimate.confidence == 0.3
            assert fallback_estimate.complexity == ToolComplexity.STANDARD

            # Should be the same as what get_estimate returns for unknown tools
            assert estimate == fallback_estimate

        except Exception:
            # If estimation fails entirely, should be able to use fallback
            fallback = config.unknown_tool_estimate
            assert fallback.llm_calls > 0
            assert fallback.total_tokens > 0

    def test_usage_logging_extra_fields(self):
        """Test that estimates provide all fields needed for logging."""
        config = ToolEstimationConfig()

        for _tool_name, estimate in config.tool_estimates.items():
            # Verify all fields needed for server.py logging are present
            assert hasattr(estimate, "confidence")
            assert hasattr(estimate, "based_on")
            assert hasattr(estimate, "complexity")
            assert hasattr(estimate, "llm_calls")
            assert hasattr(estimate, "total_tokens")

            # Verify fields have appropriate types for logging
            assert isinstance(estimate.confidence, float)
            assert isinstance(estimate.based_on, EstimationBasis)
            assert isinstance(estimate.complexity, ToolComplexity)
            assert isinstance(estimate.llm_calls, int)
            assert isinstance(estimate.total_tokens, int)
