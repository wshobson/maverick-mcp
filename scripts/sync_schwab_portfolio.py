"""Refresh Maverick's local Schwab portfolio snapshot."""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from maverick_mcp.providers.schwab import (
    SchwabAuthConfig,
    SchwabClient,
    SchwabTokenStore,
)
from maverick_mcp.providers.schwab.sync import sync_schwab_portfolio


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync live Schwab holdings into a local Maverick portfolio."
    )
    parser.add_argument("--portfolio-name", default="Schwab")
    parser.add_argument("--user-id", default="default")
    args = parser.parse_args()

    load_dotenv()
    config = SchwabAuthConfig.from_env()
    store = SchwabTokenStore(config.token_file)
    client = SchwabClient(config, store)
    result = sync_schwab_portfolio(
        client,
        portfolio_name=args.portfolio_name,
        user_id=args.user_id,
    )

    print(
        f"Synced {result['positions_synced']} Schwab positions into "
        f"{result['portfolio_name']} at {result['as_of']}."
    )
    print("Tickers:", ", ".join(result["tickers"]))


if __name__ == "__main__":
    main()
