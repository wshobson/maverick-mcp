"""Minimal Stripe webhook handler used in the test-suite."""
from __future__ import annotations

import json
from typing import Any, Dict

from stripe import SignatureVerificationError

from maverick_mcp.utils.webhook_security import WebhookSecurity


class StripeCreditService:
    """Service responsible for validating Stripe webhook notifications."""

    def __init__(self, webhook_secret: str | None = None, *, max_timestamp_age: int = 300) -> None:
        self.webhook_secret = webhook_secret
        self.max_timestamp_age = max_timestamp_age

    async def handle_webhook(
        self,
        payload: bytes,
        signature_header: str,
        db_session: Any,
    ) -> Dict[str, Any]:
        """Validate the webhook signature and return a structured response."""

        if not self.webhook_secret:
            return {"status": "rejected", "reason": "missing_webhook_secret"}

        is_valid, error, parsed = WebhookSecurity.verify_webhook_signature(
            payload,
            signature_header,
            self.webhook_secret,
            max_timestamp_age=self.max_timestamp_age,
        )

        if not is_valid:
            if error and "Invalid signature" in error:
                raise SignatureVerificationError(error)

            reason = "invalid_signature"
            if error and "Timestamp" in error:
                reason = "timestamp_validation_failed"

            return {"status": "rejected", "reason": reason, "details": error}

        event: Dict[str, Any] = {}
        if payload:
            try:
                event = json.loads(payload.decode("utf-8"))
            except json.JSONDecodeError:
                event = {}

        return {"status": "accepted", "event": event, "signature": parsed}
