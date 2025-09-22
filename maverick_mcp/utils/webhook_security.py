"""Utility helpers for validating incoming webhook requests."""
from __future__ import annotations

import hmac
import hashlib
import time
from typing import Any, Dict, List, Tuple, TypedDict


class ParsedSignature(TypedDict):
    """Typed dictionary describing the parsed Stripe signature header."""

    t: str
    v1: List[str]


class WebhookSecurity:
    """Collection of helpers for validating webhook payloads."""

    @staticmethod
    def parse_stripe_signature(signature_header: str) -> ParsedSignature:
        """Parse the raw ``Stripe-Signature`` header into structured components."""

        if not signature_header:
            raise ValueError("Empty signature header")

        timestamp: str | None = None
        signatures: list[str] = []

        for component in signature_header.split(","):
            if "=" not in component:
                continue
            key, value = component.split("=", 1)
            key = key.strip()
            value = value.strip()

            if key == "t":
                timestamp = value
            elif key == "v1":
                signatures.append(value)

        if not timestamp or not signatures:
            raise ValueError("Missing required signature components")

        return ParsedSignature(t=timestamp, v1=signatures)

    @staticmethod
    def validate_timestamp(timestamp: str, *, max_age: int = 300) -> Tuple[bool, str]:
        """Validate a UNIX timestamp used in a webhook signature."""

        try:
            ts_value = int(timestamp)
        except (TypeError, ValueError):
            return False, "Invalid timestamp format"

        now = int(time.time())
        if ts_value > now + 60:
            return False, "Timestamp in future"
        if ts_value < now - max_age:
            return False, "Timestamp too old"
        return True, ""

    @staticmethod
    def constant_time_compare(expected: str, actual: str) -> bool:
        """Compare two strings using ``hmac.compare_digest`` to avoid timing attacks."""

        if len(expected) != len(actual):
            return False
        return hmac.compare_digest(expected, actual)

    @staticmethod
    def compute_signature(payload: bytes, secret: str, timestamp: str) -> str:
        """Compute the HMAC SHA256 signature for a Stripe webhook payload."""

        message = timestamp.encode("utf-8") + b"." + payload
        digest = hmac.new(secret.encode("utf-8"), message, hashlib.sha256)
        return digest.hexdigest()

    @staticmethod
    def verify_webhook_signature(
        payload: bytes,
        signature_header: str,
        secret: str,
        *,
        max_timestamp_age: int = 300,
    ) -> Tuple[bool, str | None, ParsedSignature]:
        """Verify the signature for an incoming webhook payload."""

        parsed = WebhookSecurity.parse_stripe_signature(signature_header)
        is_valid_ts, ts_error = WebhookSecurity.validate_timestamp(
            parsed["t"], max_age=max_timestamp_age
        )
        if not is_valid_ts:
            return False, f"Timestamp validation failed: {ts_error}", parsed

        expected_signature = WebhookSecurity.compute_signature(
            payload, secret, parsed["t"]
        )
        for candidate in parsed["v1"]:
            if WebhookSecurity.constant_time_compare(expected_signature, candidate):
                return True, None, parsed

        return False, "Invalid signature", parsed

    @staticmethod
    def extract_idempotency_key(event: Dict[str, Any]) -> str | None:
        """Extract an idempotency key from Stripe event structures."""

        metadata = event.get("metadata") or {}
        if isinstance(metadata, dict) and metadata.get("idempotency_key"):
            return str(metadata["idempotency_key"])

        data_object = event.get("data", {}).get("object", {})
        if isinstance(data_object, dict):
            object_metadata = data_object.get("metadata") or {}
            if isinstance(object_metadata, dict) and object_metadata.get("idempotency_key"):
                return str(object_metadata["idempotency_key"])

        request_info = event.get("request") or {}
        if isinstance(request_info, dict) and request_info.get("idempotency_key"):
            return str(request_info["idempotency_key"])

        event_id = event.get("id")
        return str(event_id) if event_id is not None else None
