"""
Tool cost estimation configuration.

Provides cost estimates for various tool operations to help with credit management
and budget allocation.
"""

from typing import Any


class ToolCostEstimator:
    """Estimates costs for tool operations based on complexity and resource usage."""

    # Base cost estimates for different tool categories (in credits)
    BASE_COSTS = {
        "search": {
            "simple": 1,  # Basic stock lookup
            "moderate": 3,  # Market screening
            "complex": 5,  # Technical analysis
            "very_complex": 8,  # Deep research
        },
        "analysis": {
            "simple": 2,  # Simple calculations
            "moderate": 4,  # Technical indicators
            "complex": 7,  # Portfolio analysis
            "very_complex": 12,  # AI-powered insights
        },
        "data": {
            "simple": 1,  # Single stock data
            "moderate": 2,  # Multiple stocks
            "complex": 4,  # Historical data
            "very_complex": 6,  # Real-time streaming
        },
        "research": {
            "simple": 3,  # Basic company info
            "moderate": 6,  # Market research
            "complex": 10,  # Deep analysis
            "very_complex": 15,  # Comprehensive research
        },
    }

    # Multipliers for additional parameters
    MULTIPLIERS = {
        "batch_size": {
            "small": 1.0,  # 1-10 items
            "medium": 1.5,  # 11-50 items
            "large": 2.0,  # 51+ items
        },
        "time_sensitivity": {
            "normal": 1.0,
            "urgent": 1.3,
            "real_time": 1.5,
        },
    }

    @classmethod
    def estimate_tool_cost(
        self,
        tool_name: str,
        category: str,
        complexity: str = "moderate",
        additional_params: dict[str, Any] = None,
    ) -> int:
        """
        Estimate the cost of running a tool operation.

        Args:
            tool_name: Name of the tool
            category: Category of operation (search, analysis, data, research)
            complexity: Complexity level (simple, moderate, complex, very_complex)
            additional_params: Additional parameters that may affect cost

        Returns:
            Estimated cost in credits
        """
        if additional_params is None:
            additional_params = {}

        # Get base cost
        base_cost = self.BASE_COSTS.get(category, {}).get(complexity, 3)

        # Apply multipliers
        total_cost = base_cost

        # Batch size multiplier
        batch_size = additional_params.get("batch_size", 1)
        if batch_size <= 10:
            batch_multiplier = self.MULTIPLIERS["batch_size"]["small"]
        elif batch_size <= 50:
            batch_multiplier = self.MULTIPLIERS["batch_size"]["medium"]
        else:
            batch_multiplier = self.MULTIPLIERS["batch_size"]["large"]

        total_cost *= batch_multiplier

        # Time sensitivity multiplier
        time_sensitivity = additional_params.get("time_sensitivity", "normal")
        time_multiplier = self.MULTIPLIERS["time_sensitivity"].get(
            time_sensitivity, 1.0
        )
        total_cost *= time_multiplier

        # Tool-specific adjustments
        if "portfolio" in tool_name.lower():
            total_cost *= 1.2  # Portfolio operations are more complex
        elif "screening" in tool_name.lower():
            total_cost *= 1.1  # Screening operations process more data
        elif "real_time" in tool_name.lower():
            total_cost *= 1.3  # Real-time operations are resource intensive

        return max(1, int(total_cost))  # Minimum 1 credit

    def get_category_costs(self, category: str) -> dict[str, int]:
        """Get all complexity costs for a category."""
        return self.BASE_COSTS.get(category, {})

    def estimate_batch_cost(
        self,
        tool_name: str,
        category: str,
        complexity: str,
        batch_size: int,
        time_sensitivity: str = "normal",
    ) -> int:
        """
        Estimate cost for batch operations.

        Args:
            tool_name: Name of the tool
            category: Category of operation
            complexity: Complexity level
            batch_size: Number of items in batch
            time_sensitivity: Time sensitivity level

        Returns:
            Estimated total cost in credits
        """
        params = {
            "batch_size": batch_size,
            "time_sensitivity": time_sensitivity,
        }
        return self.estimate_tool_cost(tool_name, category, complexity, params)


# Global instance for easy access
tool_cost_estimator = ToolCostEstimator()


def estimate_tool_cost(
    tool_name: str,
    category: str = "analysis",
    complexity: str = "moderate",
    **kwargs,
) -> int:
    """
    Convenience function to estimate tool cost.

    Args:
        tool_name: Name of the tool
        category: Category of operation
        complexity: Complexity level
        **kwargs: Additional parameters

    Returns:
        Estimated cost in credits
    """
    return tool_cost_estimator.estimate_tool_cost(
        tool_name, category, complexity, kwargs
    )
