<!--
UPDATE 2026-09-26: https://www.pulsemcp.com/submit says PulseMCP is not
accepting new MCP server submissions and points maintainers to the official
MCP Registry, and PulseMCP already lists this repo from its own crawl (see
Status below). There is nothing to submit here. The July 2026 note follows.

HOW TO SUBMIT: mechanism UNCONFIRMED. https://www.pulsemcp.com/use-cases/submit
was the only submission-shaped page found and it states PulseMCP is "no
longer accepting new use case submissions" -- that specific page appears to
be for a different submission type ("use cases"), not necessarily server
listings, but no separate "submit a server" page was found. VERIFY the
current submission process at https://www.pulsemcp.com before doing
anything with this file -- it may require emailing the PulseMCP team, a
different form, or may not accept direct submissions at all (their
directory may be crawl-populated).
-->

# PulseMCP submission draft

**Status: mechanism not confirmed.** Web search and fetch (July 2026) found
PulseMCP's server directory (`pulsemcp.com/servers`, 20,000+ servers,
described as "daily-updated") but no working, fetchable "submit a server"
form distinct from the use-cases page above. PulseMCP may populate its
directory primarily by crawling GitHub/npm/PyPI rather than accepting
manual submissions — this is a plausible explanation for the missing form,
not a confirmed fact. **Do not treat any URL below as authoritative; none
was verified as a live submission endpoint.**

**Status (2026-09-26):** PulseMCP lists the repo at
<https://www.pulsemcp.com/servers/wshobson-maverick-financial-analysis>
("Maverick Financial Analysis"), with a pre-v1.0 description that still
mentions Tiingo. The page says PulseMCP manages a `server.json` for the repo
until the maintainer publishes one to the official MCP Registry, so the fix
for the stale listing is Step 2 of `docs/runbooks/releasing.md`, which waits
on the PyPI publish.

## Metadata to use if/when a submission path is found

- **Name**: Maverick MCP
- **Canonical MCP name**: `io.github.wshobson/maverick-mcp`
- **Description**: Personal-use, educational MCP server for stock analysis —
  market data, screening, technical indicators, portfolio tracking with
  cost-basis P&L, plus optional backtesting and deep-research extras. Not
  financial advice.
- **Repo URL**: https://github.com/wshobson/maverick-mcp
- **Install command**: `uvx --from "git+https://github.com/wshobson/maverick-mcp@v1.1.0" maverick-mcp --transport stdio`
  or `pip install "maverick-mcp-server[backtesting,research] @ git+https://github.com/wshobson/maverick-mcp@v1.1.0"`.
  Not the bare PyPI name: it belongs to an unrelated project until the name
  transfer completes (see `docs/runbooks/releasing.md`).
- **Transports**: stdio, streamable HTTP
- **Categories/tags**: finance, stocks, market-data, technical-analysis,
  portfolio, backtesting, research
- **License**: MIT

## Recommended next step at submit time

Check https://www.pulsemcp.com/submit in case submissions have reopened.
Otherwise nothing is needed: the crawler already picked up the GitHub repo
without a PyPI release, and publishing to the official MCP Registry is the
path PulseMCP points to.
