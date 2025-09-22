"""Lightweight Stripe SDK stub used for testing.

Only the minimal interfaces required by the test-suite are implemented.
"""
from __future__ import annotations


class SignatureVerificationError(Exception):
    """Exception raised when webhook signature verification fails."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or "Invalid Stripe signature")
