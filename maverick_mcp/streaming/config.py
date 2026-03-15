"""Configuration for the real-time price streaming subsystem."""

from __future__ import annotations

from pydantic import BaseModel, Field


class StreamingConfig(BaseModel):
    """Settings for the WebSocket price streaming service.

    All timing values are in seconds.
    """

    poll_interval: float = Field(
        default=15.0,
        ge=5.0,
        le=60.0,
        description="Seconds between price polling cycles (5-60)",
    )
    max_connections: int = Field(
        default=5,
        ge=1,
        description="Maximum concurrent WebSocket connections",
    )
    max_subscriptions: int = Field(
        default=50,
        ge=1,
        description="Maximum tickers that can be subscribed simultaneously",
    )
    alert_eval_every_n_polls: int = Field(
        default=3,
        ge=1,
        description="Evaluate watchlist alerts every Nth poll cycle",
    )
    max_messages_per_second: float = Field(
        default=2.0,
        ge=0.1,
        description="Per-connection outbound message rate limit",
    )
    heartbeat_interval: float = Field(
        default=30.0,
        ge=5.0,
        description="Seconds between heartbeat messages",
    )
    price_change_threshold: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum price change (%) to trigger an update. 0 = every update.",
    )
