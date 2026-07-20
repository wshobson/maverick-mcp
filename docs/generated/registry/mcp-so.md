<!--
HOW TO SUBMIT: https://mcp.so/submit?type=server (free review or paid
"Premium" immediate publish, per the site) or by opening a GitHub issue
against mcp.so's repo per community reports. Submissions reportedly only
support public GitHub-hosted MCP servers, which maverick-mcp is. VERIFY the
current form fields at https://mcp.so/submit at submit time -- the form
itself returned 403 to automated fetch and was not directly inspectable.
-->

# mcp.so submission draft

Sourced from web search results describing `mcp.so`'s submission flow
(fetched July 2026); the submission page itself
(https://mcp.so/submit?type=server) returned HTTP 403 to automated fetch, so
**the exact current form field names were not directly verifiable and are
not fabricated here** — only the metadata values to paste in are given.

## Submission path

- Web form at https://mcp.so/submit?type=server ("Submit your project"),
  reportedly requiring choice of free review vs. paid immediate publish.
- Community reports also mention a GitHub-issue-based submission path
  against mcp.so's own repository as an alternative; unconfirmed.
- Reportedly only public GitHub-hosted MCP servers are currently supported,
  which matches maverick-mcp.

## Metadata to paste in

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

## Open questions (verify at submit time)

- Exact field names/required fields on the submission form (could not be
  fetched — form returned 403 to automated access).
- Whether "free review" has a stated turnaround time or acceptance
  criteria beyond "public GitHub MCP server."
