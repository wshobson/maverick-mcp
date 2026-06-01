"""Thin read-only Schwab Trader API client."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx

from .auth import SchwabAuthConfig, SchwabTokenStore, refresh_access_token

TRADER_BASE_URL = "https://api.schwabapi.com/trader/v1"


class SchwabClient:
    """HTTP client for Schwab Trader API read-only calls."""

    def __init__(
        self,
        config: SchwabAuthConfig,
        token_store: SchwabTokenStore,
        *,
        timeout: float = 30.0,
    ) -> None:
        self.config = config
        self.token_store = token_store
        self.timeout = timeout

    def _access_token(self) -> str:
        token = self.token_store.read()
        if not token or not token.get("access_token"):
            raise ValueError("No Schwab access token found. Authorize Schwab first.")

        if self._is_expired(token) and token.get("refresh_token"):
            refreshed = refresh_access_token(
                self.config,
                token["refresh_token"],
                timeout=self.timeout,
            )
            if "refresh_token" not in refreshed:
                refreshed["refresh_token"] = token["refresh_token"]
            token = self.token_store.write(refreshed)

        return str(token["access_token"])

    @staticmethod
    def _is_expired(token: dict[str, Any]) -> bool:
        expires_at = token.get("expires_at")
        if not expires_at:
            return False
        try:
            # Refresh a little early so requests do not race token expiry.
            return (
                datetime.fromisoformat(expires_at).timestamp()
                <= datetime.now(UTC).timestamp() + 60
            )
        except ValueError:
            return False

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Run an authenticated GET request against the Trader API."""
        url = f"{TRADER_BASE_URL}{path}"
        response = httpx.get(
            url,
            headers={"Authorization": f"Bearer {self._access_token()}"},
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        if not response.content:
            return None
        return response.json()

    def account_numbers(self) -> Any:
        """Return account numbers and Schwab account hash values."""
        return self.get("/accounts/accountNumbers")

    def accounts(self, *, fields: str | None = "positions") -> Any:
        """Return linked accounts, optionally including positions."""
        params = {"fields": fields} if fields else None
        return self.get("/accounts", params=params)

    def transactions(
        self,
        account_hash: str,
        *,
        start_date: str,
        end_date: str,
        types: str = "TRADE",
    ) -> Any:
        """Return transactions for an account hash."""
        return self.get(
            f"/accounts/{account_hash}/transactions",
            params={
                "startDate": start_date,
                "endDate": end_date,
                "types": types,
            },
        )
