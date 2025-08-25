"""
Router modules for organizing Maverick-MCP endpoints by domain.

This module contains domain-specific routers that organize
the MCP tools into logical groups for better maintainability.
Personal-use stock analysis MCP server.
"""

from .data import data_router
from .performance import get_performance_router
from .portfolio import portfolio_router
from .screening import screening_router
from .technical import technical_router

# Initialize performance router
performance_router = get_performance_router()

# Optional: LangGraph agents router
try:
    from .agents import agents_router

    has_agents = True
except ImportError:
    agents_router = None  # type: ignore[assignment]
    has_agents = False

__all__ = [
    "data_router",
    "performance_router",
    "portfolio_router",
    "screening_router",
    "technical_router",
]

if has_agents:
    __all__.append("agents_router")
