"""Capture a Schwab callback URL from the macOS clipboard and exchange it.

Usage:
    uv run python scripts/schwab_auth_clipboard.py

Open the printed URL, approve Schwab access, then copy the final callback URL
from the browser address bar. The script exchanges the code immediately.
"""

from __future__ import annotations

import subprocess
import time

from dotenv import load_dotenv

from maverick_mcp.api.routers.schwab import _extract_code, _safe_account_numbers
from maverick_mcp.providers.schwab import SchwabAuthConfig, SchwabClient
from maverick_mcp.providers.schwab.auth import (
    SchwabTokenStore,
    exchange_authorization_code,
)


def _clipboard() -> str:
    return subprocess.check_output(["pbpaste"], text=True).strip()


def main() -> None:
    load_dotenv()
    config = SchwabAuthConfig.from_env()
    store = SchwabTokenStore(config.token_file)

    print("Open this Schwab authorization URL:")
    print(config.authorization_url())
    print()
    print("After approval, copy the full callback URL from the browser address bar.")
    print("Waiting for a clipboard value that contains '?code=' ...")

    seen = ""
    while True:
        value = _clipboard()
        if value != seen:
            seen = value
            if value.startswith(config.redirect_uri) and "code=" in value:
                code = _extract_code(value)
                token = exchange_authorization_code(config, code)
                store.write(token)
                print("Token saved.")
                print(store.status())

                client = SchwabClient(config, store)
                accounts = client.account_numbers()
                print("Schwab account smoke test succeeded.")
                print(_safe_account_numbers(accounts))
                return
        time.sleep(0.5)


if __name__ == "__main__":
    main()
