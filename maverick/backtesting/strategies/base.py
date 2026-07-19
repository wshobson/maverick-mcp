"""Strategy ABC. Third-layer sibling: imports config and types.

Ported whole from `maverick_mcp/backtesting/strategies/base.py` (71 lines) --
it is the minimal seam the ML strategy classes in `ml/` subclass. Task 5
builds `templates.py` (the rule-based strategy templates and the strategy
parser) on top of this same ABC; that work is out of scope here.
"""

from abc import ABC, abstractmethod
from typing import Any

from pandas import DataFrame, Series


class Strategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, parameters: dict[str, Any] | None = None):
        """Initialize strategy with parameters.

        Args:
            parameters: Strategy parameters
        """
        self.parameters = parameters or {}

    @abstractmethod
    def generate_signals(self, data: DataFrame) -> tuple[Series, Series]:
        """Generate entry and exit signals.

        Args:
            data: Price data with OHLCV columns

        Returns:
            Tuple of (entry_signals, exit_signals) as boolean Series
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get strategy name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Get strategy description."""
        pass

    def validate_parameters(self) -> bool:
        """Validate strategy parameters.

        Returns:
            True if parameters are valid
        """
        return True

    def get_default_parameters(self) -> dict[str, Any]:
        """Get default parameters for the strategy.

        Returns:
            Dictionary of default parameters
        """
        return {}

    def to_dict(self) -> dict[str, Any]:
        """Convert strategy to dictionary representation.

        Returns:
            Dictionary with strategy details
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "default_parameters": self.get_default_parameters(),
        }
