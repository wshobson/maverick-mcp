"""
Constants for the Maverick-MCP package.
"""

import os
from typing import Any


def clean_env_var(var_name, default):
    """Clean environment variable value to handle comments"""
    value = os.getenv(var_name, default)
    if value and isinstance(value, str):
        # Remove any trailing comments (anything after # that's not inside quotes)
        return value.split("#", 1)[0].strip()
    return value


# Configuration with defaults
CONFIG: dict[str, Any] = {
    "redis": {
        "host": clean_env_var("REDIS_HOST", "localhost"),
        "port": int(clean_env_var("REDIS_PORT", "6379")),
        "db": int(clean_env_var("REDIS_DB", "0")),
        "username": clean_env_var("REDIS_USERNAME", None),
        "password": clean_env_var("REDIS_PASSWORD", None),
        "ssl": clean_env_var("REDIS_SSL", "False").lower() == "true",
    },
    "cache": {
        "ttl": int(clean_env_var("CACHE_TTL_SECONDS", "604800")),  # 7 days default
        "enabled": clean_env_var("CACHE_ENABLED", "True").lower() == "true",
    },
    "yfinance": {
        "timeout": int(clean_env_var("YFINANCE_TIMEOUT_SECONDS", "30")),
    },
}

# Cache TTL in seconds
CACHE_TTL = CONFIG["cache"]["ttl"]
