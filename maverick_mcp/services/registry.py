"""Lightweight service registry for dependency lookup."""

from __future__ import annotations

from typing import Any


class ServiceRegistry:
    """A simple name-to-instance service registry.

    Provides a central place to register and retrieve service objects,
    enabling loose coupling between components without a heavy DI framework.

    Example:
        registry = ServiceRegistry()
        registry.register("cache", my_cache_service)
        cache = registry.get("cache")
    """

    def __init__(self) -> None:
        self._services: dict[str, Any] = {}

    def register(self, name: str, service: Any, replace: bool = False) -> None:
        """Register a service under the given name.

        Args:
            name: Unique identifier for the service.
            service: The service instance to register.
            replace: If True, allow replacing an existing registration.
                     If False (default), raise ValueError on duplicate names.

        Raises:
            ValueError: If *name* is already registered and *replace* is False.
        """
        if name in self._services and not replace:
            raise ValueError(
                f"Service {name!r} is already registered. Use replace=True to override."
            )
        self._services[name] = service

    def get(self, name: str) -> Any:
        """Retrieve a registered service by name.

        Args:
            name: The name the service was registered under.

        Returns:
            The registered service instance.

        Raises:
            KeyError: If no service with *name* has been registered.
        """
        if name not in self._services:
            raise KeyError(f"Service {name!r} is not registered.")
        return self._services[name]

    def get_optional(self, name: str) -> Any | None:
        """Retrieve a service by name, returning None if not found.

        Args:
            name: The name the service was registered under.

        Returns:
            The registered service instance, or None if not registered.
        """
        return self._services.get(name)

    def has(self, name: str) -> bool:
        """Check whether a service is registered under *name*.

        Args:
            name: The name to check.

        Returns:
            True if a service with that name is registered, False otherwise.
        """
        return name in self._services

    def list_services(self) -> list[str]:
        """Return the names of all registered services.

        Returns:
            List of registered service names in insertion order.
        """
        return list(self._services.keys())
