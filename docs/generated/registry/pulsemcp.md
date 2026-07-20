<!--
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

## Metadata to use if/when a submission path is found

- **Name**: Maverick MCP
- **Canonical MCP name**: `io.github.wshobson/maverick-mcp`
- **Description**: Personal-use, educational MCP server for stock analysis —
  market data, screening, technical indicators, portfolio tracking with
  cost-basis P&L, plus optional backtesting and deep-research extras. Not
  financial advice.
- **Repo URL**: https://github.com/wshobson/maverick-mcp
- **Install command**: `uvx maverick-mcp-server` (stdio; once published to
  PyPI) or `pip install "maverick-mcp-server[backtesting,research]"`
- **Transports**: stdio, streamable HTTP
- **Categories/tags**: finance, stocks, market-data, technical-analysis,
  portfolio, backtesting, research
- **License**: MIT

## Recommended next step at submit time

Check https://www.pulsemcp.com directly for a current "Submit a server" or
"Add a server" link (site navigation, footer), and/or check whether
PyPI/GitHub publication alone (Phase 9 Task 4) is sufficient for PulseMCP's
crawler to pick up the listing without manual submission.
