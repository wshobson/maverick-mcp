<!--
HOW TO SUBMIT: fork docker/mcp-registry, add servers/maverick-mcp/server.yaml
(content below) under a new directory, run `task validate -- --name maverick-mcp`
and `task build -- --tools maverick-mcp` per their CONTRIBUTING.md, then open
a PR with the title/body below. See https://github.com/docker/mcp-registry/blob/main/CONTRIBUTING.md
-->

# Docker MCP Catalog submission draft

Sourced from `docker/mcp-registry`'s `CONTRIBUTING.md` and `add_mcp_server.md`
(fetched July 2026), plus the field names observed in the repo's own
`servers/fetch/server.yaml` example. **Verify against the current
`docker/mcp-registry` CONTRIBUTING.md at submit time** — registry
requirements and the `task wizard` flow can change without notice, and
several fields below are best-effort inferences flagged individually.

## PR title

```
Add maverick-mcp server
```

## PR body

```markdown
## Summary

Adds `maverick-mcp` to the Docker MCP Catalog: a personal-use, educational
MCP server for stock analysis (market data, screening, technical indicators,
portfolio tracking with cost-basis P&L, plus optional backtesting and deep
research extras). Not financial advice.

- Canonical MCP name: `io.github.wshobson/maverick-mcp`
- Repository: https://github.com/wshobson/maverick-mcp
- License: MIT
- Transports: stdio (default), streamable HTTP

## Checklist

- [ ] `task validate -- --name maverick-mcp` passes
- [ ] `task build -- --tools maverick-mcp` passes
- [ ] `server.yaml` follows the current schema in CONTRIBUTING.md
```

## `servers/maverick-mcp/server.yaml` draft

```yaml
name: maverick-mcp
image: mcp/maverick-mcp
type: server
meta:
  category: finance # UNCONFIRMED: no published category enum found; verify
                     # the allowed value list against other servers in
                     # docker/mcp-registry at submit time.
  tags:
    - finance
    - stocks
    - market-data
    - technical-analysis
    - portfolio
    - backtesting
    - research
about:
  title: Maverick MCP
  description: >-
    Personal-use, educational MCP server for stock analysis: market data,
    screening, technical indicators, and portfolio tracking with cost-basis
    P&L, plus optional backtesting and deep-research extras. Not financial
    advice.
  icon: https://avatars.githubusercontent.com/wshobson # UNCONFIRMED: no
    # dedicated project icon exists yet; this is the GitHub owner avatar as
    # a placeholder. Replace with a real icon URL or omit if the schema
    # requires a project-specific asset.
source:
  project: https://github.com/wshobson/maverick-mcp
  branch: main
  commit: a2b33203ebde0c608dd94b6da03dcc680f2487d9 # v1.0.0 tag commit
  directory: "." # Dockerfile lives at the repo root
```

## Assumptions and open questions (verify at submit time)

- **`image: mcp/maverick-mcp`**: the observed convention in the repo's own
  `servers/fetch/server.yaml` (`image: mcp/fetch` for server `fetch`) is
  `mcp/<name>`, which implies Docker builds and publishes the image into
  their own `mcp/` namespace from the `source` block rather than the
  submitter pre-publishing to Docker Hub/GHCR. This ownership/build model is
  **inferred from the naming convention, not confirmed in the fetched docs**
  — confirm before filing.
- **`type: server` (local, containerized)** was chosen over `type: remote`
  because maverick-mcp's primary distribution is a local stdio process (via
  `uvx`), matching the `mcp/fetch` reference server's shape, not a hosted
  HTTP endpoint. Docker's local-type build uses the repo's existing
  `Dockerfile` at the repo root.
- **`meta.category` allowed values**: not found in the fetched
  documentation. `finance` is a reasonable guess given the project's domain;
  confirm the actual enum (if one exists) by checking `meta.category` values
  used by other `servers/*/server.yaml` entries in the registry before
  filing.
- The Dockerfile does not yet carry the
  `io.modelcontextprotocol.server.name=io.github.wshobson/maverick-mcp`
  LABEL (Phase 9 Task 2, not yet landed as of this draft). Docker's catalog
  submission may or may not require that label; check `add_mcp_server.md`
  again once Task 2 lands.
- The registry's own `task wizard` (for local servers) is described as "the
  easiest way to create your `server.yaml`" — running it against the real
  repo at submit time may produce a materially different file than this
  hand-drafted one. Prefer the wizard's output over this draft where they
  disagree.
