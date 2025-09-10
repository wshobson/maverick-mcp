"""
Plotly configuration module for Maverick MCP.

This module configures Plotly defaults using the modern plotly.io.defaults API
to avoid deprecation warnings from the legacy kaleido.scope API.
"""

import logging
import warnings
from typing import Any

try:
    import plotly.io as pio

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logger = logging.getLogger(__name__)


def configure_plotly_defaults() -> None:
    """
    Configure Plotly defaults using the modern plotly.io.defaults API.

    This replaces the deprecated plotly.io.kaleido.scope configuration
    and helps reduce deprecation warnings.
    """
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly not available, skipping configuration")
        return

    try:
        # Configure modern Plotly defaults (replaces kaleido.scope configuration)
        pio.defaults.default_format = "png"
        pio.defaults.default_width = 800
        pio.defaults.default_height = 600
        pio.defaults.default_scale = 1.0

        # Configure additional defaults that don't trigger deprecation warnings
        if hasattr(pio.defaults, "mathjax"):
            pio.defaults.mathjax = None
        if hasattr(pio.defaults, "plotlyjs"):
            pio.defaults.plotlyjs = "auto"

        # Note: We avoid setting kaleido.scope properties directly to prevent warnings
        # The modern pio.defaults API should be used instead

        logger.info("✓ Plotly defaults configured successfully")

    except Exception as e:
        logger.error(f"Error configuring Plotly defaults: {e}")


def suppress_plotly_warnings() -> None:
    """
    Suppress specific Plotly/Kaleido deprecation warnings.

    These warnings come from the library internals and can't be fixed
    at the user code level until the libraries are updated.
    """
    try:
        # Comprehensive suppression of all kaleido-related deprecation warnings
        deprecation_patterns = [
            r".*plotly\.io\.kaleido\.scope\..*is deprecated.*",
            r".*Use of plotly\.io\.kaleido\.scope\..*is deprecated.*",
            r".*default_format.*deprecated.*",
            r".*default_width.*deprecated.*",
            r".*default_height.*deprecated.*",
            r".*default_scale.*deprecated.*",
            r".*mathjax.*deprecated.*",
            r".*plotlyjs.*deprecated.*",
        ]

        for pattern in deprecation_patterns:
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                message=pattern,
            )

        # Also suppress by module
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            module=r".*kaleido.*",
        )

        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            module=r"plotly\.io\._kaleido",
        )

        logger.debug("✓ Plotly deprecation warnings suppressed")

    except Exception as e:
        logger.error(f"Error suppressing Plotly warnings: {e}")


def setup_plotly() -> None:
    """
    Complete Plotly setup with modern configuration and warning suppression.

    This function should be called once during application initialization.
    """
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly not available, skipping setup")
        return

    # First suppress warnings to avoid noise during configuration
    suppress_plotly_warnings()

    # Then configure with modern API
    configure_plotly_defaults()

    logger.info("✓ Plotly setup completed")


def get_plotly_config() -> dict[str, Any]:
    """
    Get current Plotly configuration for debugging.

    Returns:
        Dictionary with current Plotly configuration settings
    """
    if not PLOTLY_AVAILABLE:
        return {"error": "Plotly not available"}

    config = {}

    try:
        # Modern defaults
        config["defaults"] = {
            "default_format": getattr(pio.defaults, "default_format", "unknown"),
            "default_width": getattr(pio.defaults, "default_width", "unknown"),
            "default_height": getattr(pio.defaults, "default_height", "unknown"),
            "default_scale": getattr(pio.defaults, "default_scale", "unknown"),
        }

        # Kaleido scope (if available)
        if hasattr(pio, "kaleido") and hasattr(pio.kaleido, "scope"):
            scope = pio.kaleido.scope
            config["kaleido_scope"] = {
                "mathjax": getattr(scope, "mathjax", "unknown"),
                "plotlyjs": getattr(scope, "plotlyjs", "unknown"),
                "configured": getattr(scope, "_configured", False),
            }

    except Exception as e:
        config["error"] = f"Error getting config: {e}"

    return config
