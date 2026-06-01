"""Local OAuth helpers for the Schwab Trader API."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

AUTH_BASE_URL = "https://api.schwabapi.com/v1/oauth"


@dataclass(frozen=True)
class SchwabAuthConfig:
    """Schwab OAuth settings loaded from environment variables."""

    client_id: str
    client_secret: str
    redirect_uri: str
    token_file: Path

    @classmethod
    def from_env(cls) -> SchwabAuthConfig:
        """Build config from the local environment."""
        client_id = os.getenv("SCHWAB_CLIENT_ID", "").strip()
        client_secret = os.getenv("SCHWAB_CLIENT_SECRET", "").strip()
        redirect_uri = os.getenv(
            "SCHWAB_REDIRECT_URI", "https://127.0.0.1:8765/callback"
        ).strip()
        token_file = Path(
            os.getenv("SCHWAB_TOKEN_FILE", ".local/schwab-token.json").strip()
        )

        missing = [
            name
            for name, value in (
                ("SCHWAB_CLIENT_ID", client_id),
                ("SCHWAB_CLIENT_SECRET", client_secret),
                ("SCHWAB_REDIRECT_URI", redirect_uri),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"Missing Schwab configuration: {', '.join(missing)}")

        return cls(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            token_file=token_file,
        )

    def authorization_url(self) -> str:
        """Return the browser URL for Schwab account authorization."""
        query = urlencode(
            {
                "response_type": "code",
                "client_id": self.client_id,
                "redirect_uri": self.redirect_uri,
            }
        )
        return f"{AUTH_BASE_URL}/authorize?{query}"


class SchwabTokenStore:
    """Small JSON token store for local personal use."""

    def __init__(self, token_file: Path) -> None:
        self.token_file = token_file

    def exists(self) -> bool:
        """Return whether a token file exists."""
        return self.token_file.exists()

    def read(self) -> dict[str, Any] | None:
        """Read token data if present."""
        if not self.token_file.exists():
            return None
        with self.token_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    def write(self, token: dict[str, Any]) -> dict[str, Any]:
        """Persist token data with derived timestamps."""
        now = datetime.now(UTC)
        token = dict(token)
        token["saved_at"] = now.isoformat()
        if "expires_in" in token:
            token["expires_at"] = (
                now + timedelta(seconds=int(token["expires_in"]))
            ).isoformat()

        self.token_file.parent.mkdir(parents=True, exist_ok=True)
        with self.token_file.open("w", encoding="utf-8") as f:
            json.dump(token, f, indent=2)
            f.write("\n")
        self.token_file.chmod(0o600)
        return token

    def status(self) -> dict[str, Any]:
        """Return safe token status without secrets."""
        token = self.read()
        if not token:
            return {"configured": False, "token_file": str(self.token_file)}

        expires_at = token.get("expires_at")
        expired = None
        if expires_at:
            try:
                expired = datetime.fromisoformat(expires_at) <= datetime.now(UTC)
            except ValueError:
                expired = None

        return {
            "configured": True,
            "token_file": str(self.token_file),
            "has_access_token": bool(token.get("access_token")),
            "has_refresh_token": bool(token.get("refresh_token")),
            "expires_at": expires_at,
            "expired": expired,
        }


def exchange_authorization_code(
    config: SchwabAuthConfig,
    code: str,
    *,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Exchange an authorization code for access and refresh tokens."""
    response = httpx.post(
        f"{AUTH_BASE_URL}/token",
        auth=(config.client_id, config.client_secret),
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": config.redirect_uri,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def refresh_access_token(
    config: SchwabAuthConfig,
    refresh_token: str,
    *,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Refresh the Schwab access token."""
    response = httpx.post(
        f"{AUTH_BASE_URL}/token",
        auth=(config.client_id, config.client_secret),
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()
