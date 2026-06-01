# Schwab Trader API Setup

MaverickMCP can connect to the Schwab Trader API for local, read-only account
visibility and portfolio sync. The first supported flow fetches balances and
positions, then snapshots current holdings into Maverick's local portfolio
tables under a portfolio named `Schwab` by default.

This integration is educational tooling only. It is not investment, tax, or
trading advice, and it does not place orders.

## Schwab Developer App

Create a Schwab Developer Portal app with the `Accounts and Trading Production`
API product.

Recommended local settings:

```text
Callback URL: https://127.0.0.1:8765/callback
Order Limit: 0
```

Store the app credentials in `.env`:

```text
SCHWAB_CLIENT_ID=
SCHWAB_CLIENT_SECRET=
SCHWAB_REDIRECT_URI=https://127.0.0.1:8765/callback
SCHWAB_TOKEN_FILE=.local/schwab-token.json
SCHWAB_READ_ONLY=true
SCHWAB_ENABLE_TRADING=false
```

Do not commit `.env` or `.local/schwab-token.json`.

## Authorize Locally

Run:

```bash
uv run python scripts/schwab_auth_clipboard.py
```

Open the printed Schwab authorization URL in a browser and approve account
access. Schwab redirects to the local callback URL, which may show a browser
connection error because Maverick does not run a callback web server. Copy the
full callback URL from the browser address bar. The script watches the macOS
clipboard, exchanges the authorization code immediately, stores tokens in
`SCHWAB_TOKEN_FILE`, and smoke-tests account access.

Authorization codes are short-lived and one-use. If Schwab reports that a code
is expired or revoked, rerun the script and repeat the browser flow.

## Sync Portfolio

Run:

```bash
uv run python scripts/sync_schwab_portfolio.py
```

By default this creates or refreshes the local portfolio named `Schwab`.
The sync is a snapshot: existing positions in that portfolio are replaced with
the current Schwab positions. This avoids double-counting repeated imports.

To use a different local portfolio name:

```bash
uv run python scripts/sync_schwab_portfolio.py --portfolio-name "Schwab IRA"
```

After syncing, existing portfolio-aware Maverick tools can use the data:

```text
portfolio_get_my_portfolio(portfolio_name="Schwab")
portfolio_compare_tickers(portfolio_name="Schwab")
portfolio_portfolio_correlation_analysis(portfolio_name="Schwab")
```

## MCP Tools

The Schwab router registers these read-only tools:

- `schwab_get_auth_url`
- `schwab_exchange_code`
- `schwab_auth_status`
- `schwab_get_account_numbers`
- `schwab_get_account_summary`
- `schwab_get_positions`
- `schwab_sync_portfolio`
- `schwab_refresh_and_analyze_portfolio`

The MCP prompt registry also includes Schwab workflow prompts for refresh and
review, risk checks, single-position review, and non-executing trade-plan
drafting.

The tools intentionally avoid order placement, cancellation, replacement, and
other trading actions. Any future trading support should be implemented as a
separate safety-gated change behind `SCHWAB_ENABLE_TRADING=true`.

## Safety Notes

- Never log or commit access tokens, refresh tokens, client secrets, full
  account numbers, account hashes, portfolio exports, or database dumps.
- Keep token storage local and ignored by git.
- Treat Schwab as the source of truth for synced holdings.
- Do not invent cost basis when Schwab does not provide an average price; the
  mapper skips positions that cannot be safely represented.
