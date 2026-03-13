"""
Logging configuration for Maverick-MCP.

Provides a SecretsFilter that masks sensitive values (API keys, tokens,
passwords) in log output to prevent accidental credential leakage.
"""

import logging
import re


class SecretsFilter(logging.Filter):
    """Logging filter that masks sensitive values in log records.

    Attaches to log handlers (or the root logger) and rewrites any message
    or argument that matches known secret patterns before the record is
    emitted.  The filter always returns ``True`` so that it never
    suppresses log records -- it only sanitizes them.
    """

    # Patterns to match API keys, tokens, passwords
    SENSITIVE_PATTERNS = [
        # Key-value patterns: api_key="...", token: "...", etc.
        re.compile(
            r"(?i)(api[_-]?key|token|secret|password|credential|authorization)"
            r'["\s:=]+["\']?([a-zA-Z0-9_\-./+=]{8,})["\']?'
        ),
        # OpenAI / OpenRouter style keys (sk-...)
        re.compile(r"(sk-[a-zA-Z0-9]{20,})"),
        # Bearer tokens
        re.compile(r"(Bearer\s+[a-zA-Z0-9_\-./+=]{20,})"),
    ]

    MASK = "****REDACTED****"

    def filter(self, record: logging.LogRecord) -> bool:
        """Mask sensitive values in the log message and arguments.

        Always returns ``True`` so the record is never dropped.
        """
        try:
            if isinstance(record.msg, str):
                record.msg = self._mask_secrets(record.msg)
            if record.args:
                if isinstance(record.args, dict):
                    record.args = {
                        k: self._mask_secrets(str(v)) if isinstance(v, str) else v
                        for k, v in record.args.items()
                    }
                elif isinstance(record.args, tuple):
                    record.args = tuple(
                        self._mask_secrets(str(a)) if isinstance(a, str) else a
                        for a in record.args
                    )
        except Exception:
            # Never let masking errors suppress log records
            pass
        return True

    def _mask_secrets(self, text: str) -> str:
        """Replace sensitive substrings in *text* with a redaction marker."""
        for pattern in self.SENSITIVE_PATTERNS:
            text = pattern.sub(self._replacement, text)
        return text

    @classmethod
    def _replacement(cls, match: re.Match) -> str:
        """Build a replacement string preserving the label when possible."""
        # For key-value patterns (group 1 = label, group 2 = secret value)
        if match.lastindex and match.lastindex >= 2:
            return f"{match.group(1)}={cls.MASK}"
        # For standalone secret patterns (Bearer tokens, sk-... keys)
        return cls.MASK


def install_secrets_filter() -> SecretsFilter:
    """Install the SecretsFilter on the root logger and all its handlers.

    Returns the filter instance so callers can hold a reference if needed.
    """
    secrets_filter = SecretsFilter()
    logging.root.addFilter(secrets_filter)
    for handler in logging.root.handlers:
        handler.addFilter(secrets_filter)
    return secrets_filter
