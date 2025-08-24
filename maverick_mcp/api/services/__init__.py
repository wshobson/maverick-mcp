"""
Services package for MaverickMCP API.

This package contains service classes extracted from the large server.py file
to improve code organization and maintainability following SOLID principles.
"""

from .base_service import BaseService
from .market_service import MarketService
from .portfolio_service import PortfolioService
from .prompt_service import PromptService
from .resource_service import ResourceService

__all__ = [
    "BaseService",
    "MarketService",
    "PortfolioService",
    "PromptService",
    "ResourceService",
]
