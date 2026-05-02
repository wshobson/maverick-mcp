"""Service layer for Maverick MCP — business logic and orchestration."""

from maverick_mcp.services.event_bus import EventBus
from maverick_mcp.services.registry import ServiceRegistry
from maverick_mcp.services.scheduler import Scheduler as MaverickScheduler

event_bus = EventBus()
registry = ServiceRegistry()
scheduler = MaverickScheduler()

__all__ = [
    "EventBus",
    "MaverickScheduler",
    "ServiceRegistry",
    "event_bus",
    "registry",
    "scheduler",
]
