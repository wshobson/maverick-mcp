<!--
HOW TO SUBMIT: Glama has already indexed the repo (see Status below), so
claim the listing at https://glama.ai/mcp/servers/@wshobson/maverick-mcp.
The documented path for a repo Glama has not indexed is to install the Glama
GitHub App on it (or connect it via https://glama.ai's submission UI) — no
separate form fields to fill in for a GitHub-hosted server. See
https://glama.ai/mcp/faq for the current flow.
-->

# Glama submission draft

Sourced from Glama's own site (`glama.ai`, `glama.ai/mcp/faq`) via web
search, fetched July 2026; the exact submission UI copy was not directly
fetchable (JS-rendered pages), so **verify the current flow at
https://glama.ai at submit time**.

**Status (2026-09-26):** Glama already lists the repo, unclaimed, at
<https://glama.ai/mcp/servers/@wshobson/maverick-mcp>, with the git-tag `uvx`
install command and a stale "100+ tools" description. The owner step is to
claim that listing rather than submit a new one.

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
- **Install command**: `uvx --from "git+https://github.com/wshobson/maverick-mcp@v1.1.0" maverick-mcp --transport stdio`
  or `pip install "maverick-mcp-server[backtesting,research] @ git+https://github.com/wshobson/maverick-mcp@v1.1.0"`.
  Not the bare PyPI name: it belongs to an unrelated project until the name
  transfer completes (see `docs/runbooks/releasing.md`).
- **Transports**: stdio, streamable HTTP
- **Categories/tags**: finance, stocks, market-data, technical-analysis,
  portfolio, backtesting, research
- **License**: MIT

## Open questions (verify at submit time)

- Whether the GitHub App requires PyPI publication first, or will index a
  source-only repo. Answered: the existing listing was indexed without a
  PyPI release.
- Whether Glama's indexer needs `server.json` at the repo root (it already
  exists, Phase 9 Task 0) or has its own manifest expectations.
