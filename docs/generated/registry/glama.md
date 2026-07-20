<!--
HOW TO SUBMIT: install the Glama GitHub App on the wshobson/maverick-mcp
repo (or connect it via https://glama.ai's submission UI) so Glama indexes
the repo directly — no separate form fields to fill in for a GitHub-hosted
server. See https://glama.ai/mcp/faq for the current flow.
-->

# Glama submission draft

Sourced from Glama's own site (`glama.ai`, `glama.ai/mcp/faq`) via web
search, fetched July 2026; the exact submission UI copy was not directly
fetchable (JS-rendered pages), so **verify the current flow at
https://glama.ai at submit time**.

## Submission path

Glama documents two ways to list a server:

1. **GitHub-hosted (this project's case)**: install the Glama GitHub App
   and connect `wshobson/maverick-mcp`. Glama then indexes the repo's tools,
   schemas, and README directly — no manual metadata form. This is the
   right path for maverick-mcp since it is a public GitHub repo with a
   standard MCP server layout.
2. **Remote connector**: for servers already running at a public HTTP(S)
   endpoint. Not applicable to maverick-mcp today (stdio-first; streamable
   HTTP requires the operator to run `make dev` locally, it is not a public
   hosted endpoint).

## Metadata Glama will likely surface from the repo (for reference/QA after indexing)

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

- Whether the GitHub App requires PyPI publication first, or will index a
  source-only repo (maverick-mcp-server is not yet on PyPI as of this
  draft — Phase 9 Task 4).
- Whether Glama's indexer needs `server.json` at the repo root (it already
  exists, Phase 9 Task 0) or has its own manifest expectations.
