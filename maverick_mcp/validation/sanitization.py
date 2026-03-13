"""
Input sanitization utilities for Maverick-MCP.

Provides functions for validating and sanitizing user-supplied inputs
(ticker symbols, free-text fields) to prevent injection attacks and
ensure data integrity.
"""

import re

# Ticker symbol pattern: 1-5 uppercase letters, with optional dot for class shares (BRK.B)
VALID_TICKER_PATTERN = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")

# Extended ticker pattern that also allows digits and hyphens (e.g., X, BF-B)
VALID_TICKER_EXTENDED_PATTERN = re.compile(r"^[A-Z0-9]{1,5}([.\-][A-Z0-9]{1,2})?$")


def sanitize_ticker(ticker: str, *, strict: bool = False) -> str:
    """Sanitize and validate a stock ticker symbol.

    Args:
        ticker: Raw ticker string from user input.
        strict: If True, only allow pure alpha tickers (no digits, dots, or
            hyphens). Defaults to False which permits common patterns like
            ``BRK.B`` or ``BF-B``.

    Returns:
        Uppercased, stripped ticker symbol.

    Raises:
        ValueError: If the ticker does not match the expected pattern.
    """
    ticker = ticker.strip().upper()

    # Remove null bytes
    ticker = ticker.replace("\x00", "")

    pattern = VALID_TICKER_PATTERN if strict else VALID_TICKER_EXTENDED_PATTERN
    if not pattern.match(ticker):
        raise ValueError(
            f"Invalid ticker symbol: {ticker!r}. "
            "Must be 1-5 uppercase alphanumeric characters, "
            "optionally followed by a dot or hyphen and 1-2 characters."
        )

    return ticker


def sanitize_tickers(tickers: list[str], *, strict: bool = False) -> list[str]:
    """Sanitize and validate a list of ticker symbols.

    Removes duplicates while preserving order.

    Args:
        tickers: List of raw ticker strings.
        strict: Passed through to :func:`sanitize_ticker`.

    Returns:
        De-duplicated list of sanitized ticker symbols.

    Raises:
        ValueError: If the list is empty or any ticker is invalid.
    """
    if not tickers:
        raise ValueError("At least one ticker symbol is required")

    seen: set[str] = set()
    result: list[str] = []
    for raw in tickers:
        clean = sanitize_ticker(raw, strict=strict)
        if clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result


def sanitize_text_input(text: str, max_length: int = 10000) -> str:
    """Sanitize free-text input to prevent injection and enforce size limits.

    Args:
        text: Raw text from user input.
        max_length: Maximum allowed length. Text exceeding this is truncated.

    Returns:
        Sanitized text string.
    """
    # Remove null bytes
    text = text.replace("\x00", "")

    # Truncate to max length
    if len(text) > max_length:
        text = text[:max_length]

    return text


def sanitize_portfolio_name(name: str, max_length: int = 50) -> str:
    """Sanitize a portfolio name.

    Args:
        name: Raw portfolio name.
        max_length: Maximum allowed length.

    Returns:
        Stripped, sanitized portfolio name.

    Raises:
        ValueError: If the name is empty or contains only whitespace after
            stripping.
    """
    name = name.strip()
    name = name.replace("\x00", "")

    if not name:
        raise ValueError("Portfolio name cannot be empty")

    if len(name) > max_length:
        name = name[:max_length]

    # Only allow printable characters (no control characters)
    if re.search(r"[\x00-\x1f\x7f]", name):
        raise ValueError("Portfolio name contains invalid control characters")

    return name
