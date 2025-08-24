"""
Mock cache manager implementation for testing.
"""

import time
from typing import Any


class MockCacheManager:
    """
    Mock implementation of ICacheManager for testing.

    This implementation uses in-memory storage and provides predictable
    behavior for testing cache-dependent functionality.
    """

    def __init__(self):
        """Initialize the mock cache manager."""
        self._data: dict[str, dict[str, Any]] = {}
        self._call_log: list[dict[str, Any]] = []

    async def get(self, key: str) -> Any:
        """Get data from mock cache."""
        self._log_call("get", {"key": key})

        if key not in self._data:
            return None

        entry = self._data[key]

        # Check if expired
        if "expires_at" in entry and entry["expires_at"] < time.time():
            del self._data[key]
            return None

        return entry["value"]

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Store data in mock cache."""
        self._log_call("set", {"key": key, "value": value, "ttl": ttl})

        entry = {"value": value}

        if ttl is not None:
            entry["expires_at"] = time.time() + ttl

        self._data[key] = entry
        return True

    async def delete(self, key: str) -> bool:
        """Delete a key from mock cache."""
        self._log_call("delete", {"key": key})

        if key in self._data:
            del self._data[key]
            return True

        return False

    async def exists(self, key: str) -> bool:
        """Check if a key exists in mock cache."""
        self._log_call("exists", {"key": key})

        if key not in self._data:
            return False

        entry = self._data[key]

        # Check if expired
        if "expires_at" in entry and entry["expires_at"] < time.time():
            del self._data[key]
            return False

        return True

    async def clear(self, pattern: str | None = None) -> int:
        """Clear cache entries."""
        self._log_call("clear", {"pattern": pattern})

        if pattern is None:
            count = len(self._data)
            self._data.clear()
            return count

        # Simple pattern matching (only supports prefix*)
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            keys_to_delete = [k for k in self._data.keys() if k.startswith(prefix)]
        else:
            keys_to_delete = [k for k in self._data.keys() if k == pattern]

        for key in keys_to_delete:
            del self._data[key]

        return len(keys_to_delete)

    async def get_many(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values at once."""
        self._log_call("get_many", {"keys": keys})

        results = {}
        for key in keys:
            value = await self.get(key)
            if value is not None:
                results[key] = value

        return results

    async def set_many(self, items: list[tuple[str, Any, int | None]]) -> int:
        """Set multiple values at once."""
        self._log_call("set_many", {"items_count": len(items)})

        success_count = 0
        for key, value, ttl in items:
            if await self.set(key, value, ttl):
                success_count += 1

        return success_count

    async def delete_many(self, keys: list[str]) -> int:
        """Delete multiple keys."""
        self._log_call("delete_many", {"keys": keys})

        deleted_count = 0
        for key in keys:
            if await self.delete(key):
                deleted_count += 1

        return deleted_count

    async def exists_many(self, keys: list[str]) -> dict[str, bool]:
        """Check existence of multiple keys."""
        self._log_call("exists_many", {"keys": keys})

        results = {}
        for key in keys:
            results[key] = await self.exists(key)

        return results

    async def count_keys(self, pattern: str) -> int:
        """Count keys matching a pattern."""
        self._log_call("count_keys", {"pattern": pattern})

        if pattern.endswith("*"):
            prefix = pattern[:-1]
            return len([k for k in self._data.keys() if k.startswith(prefix)])
        else:
            return 1 if pattern in self._data else 0

    async def get_or_set(
        self, key: str, default_value: Any, ttl: int | None = None
    ) -> Any:
        """Get value from cache, setting it if it doesn't exist."""
        self._log_call(
            "get_or_set", {"key": key, "default_value": default_value, "ttl": ttl}
        )

        value = await self.get(key)
        if value is not None:
            return value

        await self.set(key, default_value, ttl)
        return default_value

    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment a numeric value in cache."""
        self._log_call("increment", {"key": key, "amount": amount})

        current = await self.get(key)

        if current is None:
            new_value = amount
        else:
            try:
                current_int = int(current)
                new_value = current_int + amount
            except (ValueError, TypeError):
                raise ValueError(f"Key {key} contains non-numeric value: {current}")

        await self.set(key, new_value)
        return new_value

    async def set_if_not_exists(
        self, key: str, value: Any, ttl: int | None = None
    ) -> bool:
        """Set a value only if the key doesn't already exist."""
        self._log_call("set_if_not_exists", {"key": key, "value": value, "ttl": ttl})

        if await self.exists(key):
            return False

        return await self.set(key, value, ttl)

    async def get_ttl(self, key: str) -> int | None:
        """Get the remaining time-to-live for a key."""
        self._log_call("get_ttl", {"key": key})

        if key not in self._data:
            return None

        entry = self._data[key]

        if "expires_at" not in entry:
            return None

        remaining = int(entry["expires_at"] - time.time())
        return max(0, remaining)

    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration time for an existing key."""
        self._log_call("expire", {"key": key, "ttl": ttl})

        if key not in self._data:
            return False

        self._data[key]["expires_at"] = time.time() + ttl
        return True

    # Testing utilities

    def _log_call(self, method: str, args: dict[str, Any]) -> None:
        """Log method calls for testing verification."""
        self._call_log.append(
            {
                "method": method,
                "args": args,
                "timestamp": time.time(),
            }
        )

    def get_call_log(self) -> list[dict[str, Any]]:
        """Get the log of method calls for testing verification."""
        return self._call_log.copy()

    def clear_call_log(self) -> None:
        """Clear the method call log."""
        self._call_log.clear()

    def get_cache_contents(self) -> dict[str, Any]:
        """Get all cache contents for testing verification."""
        return {k: v["value"] for k, v in self._data.items()}

    def set_cache_contents(self, contents: dict[str, Any]) -> None:
        """Set cache contents directly for testing setup."""
        self._data.clear()
        for key, value in contents.items():
            self._data[key] = {"value": value}

    def simulate_cache_expiry(self, key: str) -> None:
        """Simulate cache expiry for testing."""
        if key in self._data:
            self._data[key]["expires_at"] = time.time() - 1
