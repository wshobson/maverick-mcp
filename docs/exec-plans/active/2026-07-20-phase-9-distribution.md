# Phase 9: Distribution and Registry Rollout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement the prep tasks. **The publish tasks (P, marked below) each require explicit owner authorization AND credentials before execution — they perform irreversible, public, third-party actions under the maintainer's identity. Do NOT run a P-task autonomously.**

**Goal:** Publish maverick-mcp-server v1.0.0 to PyPI, GHCR, and every MCP registry that accepts maintainer submissions, and attach a one-click Claude Desktop bundle to the v1.0.0 GitHub release — so the server is installable via `uvx maverick-mcp-server` and discoverable on the official MCP Registry, Docker MCP Catalog, Smithery, Glama, PulseMCP, and mcp.so.

**Architecture:** Phase 9 of the modernization (the final phase). v1.0.0 is already tagged and GitHub-released (Phase 8). This phase is packaging-artifact publication + registry submissions. The design doc's "Packaging and distribution" section is the spec. The 2026-07-20 server recon's packaging section (`.superpowers/sdd/p8-recon-server.md` §5) is the current-state map.

## The authorization boundary

Prep tasks (files in the repo, no external side effects) run autonomously under the standing goal. Publish tasks send artifacts to external services under the owner's accounts and are mostly irreversible (a PyPI version number can never be reused; a registry listing is public). Each publish task lists the exact credential/secret it needs. The controller stops and asks the owner before any P-task.

## Global Constraints

Same gates as Phase 8 for any code/config/docs change. No secret is ever written to a tracked file — tokens live in GitHub Actions secrets or the owner's shell only. Every registry submission uses the canonical server name `io.github.wshobson/maverick-mcp`. server.json validates against the published MCP registry schema.

---

### Task 0 (prep): mcp-name provenance + server.json finalization

Add the `<!-- mcp-name: io.github.wshobson/maverick-mcp -->` comment to README (proves PyPI/registry ownership to the official registry). Finalize `server.json` against the current registry schema: the PyPI package entry (`registryType: pypi`, `identifier: maverick-mcp-server`, `runtimeHint: uvx`, version 1.0.0, stdio transport, the env-var declarations from Task 6), and a placeholder Docker package entry (filled in after GHCR push). Validate with the mcp-publisher schema (fetch it; do not invent fields). Add `docs/runbooks/releasing.md` documenting the whole publish sequence (so the owner can run the P-tasks by hand if preferred). Full docs gate. Commit `docs: add mcp-name provenance and finalize server.json for registry`.

### Task 1 (prep): mcp-publisher CI workflow

Add `.github/workflows/publish.yml` that, on a `v*` tag push, builds the wheel/sdist and publishes to PyPI via trusted publishing (OIDC, no token in repo) OR `PYPI_API_TOKEN` secret, then runs `mcp-publisher` to push server.json to the official registry. The workflow is INERT until (a) it's merged and (b) a tag triggers it with the secrets/trusted-publisher configured — so committing it has no side effect. Document the exact GitHub settings the owner must enable (PyPI trusted publisher for the repo, or the secret). Include a manual `workflow_dispatch` guard. Full gate (yaml lint via the existing CI). Commit `ci: add tag-triggered PyPI + MCP registry publish workflow`.

### Task 2 (prep): .mcpb bundle build + Dockerfile/GHCR label

Add a `make bundle` target (or script) that builds the `.mcpb` Claude Desktop bundle per the mcpb spec (manifest referencing the stdio entrypoint). Confirm the Dockerfile carries the `io.modelcontextprotocol.server.name=io.github.wshobson/maverick-mcp` LABEL and builds a coherent image (build not executed if no docker in env — verify coherence). Add a `docker/` GHCR publish step to publish.yml (inert until tag+secrets). Build the .mcpb locally and confirm it's well-formed (unzip/inspect the manifest). Full gate. Commit `feat(dist): add .mcpb bundle build and GHCR image label`.

### Task 3 (prep): registry submission drafts

Prepare, as tracked files under `docs/generated/registry/` (or similar), the submission content for each pull/submit-based registry so the owner can file them: the Docker MCP Catalog PR body + the server entry YAML/JSON it wants; the Smithery `smithery.yaml` (or CLI config) the `smithery` push needs; the Glama, PulseMCP, and mcp.so submission text/metadata (name, description, categories, install command, repo URL). Each as a ready-to-paste artifact with a one-line "how to submit" note. Note that the GitHub MCP Registry is curated (cannot be pushed) — no artifact, just a note. Full docs gate. Commit `docs: prepare registry submission drafts`.

### Task 4 (P — PyPI publish): REQUIRES owner PyPI trusted-publisher or PYPI_API_TOKEN

The actual PyPI release. Either the owner enables trusted publishing + re-pushes the v1.0.0 tag (re-running publish.yml), or `uv publish`/`twine upload dist/*` with the owner's token. IRREVERSIBLE: version 1.0.0 on PyPI is permanent. After publish: verify `uvx maverick-mcp-server --help` works from the real index in a clean environment; verify `pip install "maverick-mcp-server[backtesting,research]"` resolves. Then update the v1.0.0 GitHub release notes to state real PyPI availability (removing the "install from source until published" phrasing). **STOP — owner authorization required before this task.**

### Task 5 (P — official MCP Registry): REQUIRES PyPI published first (provenance) + registry auth

Run `mcp-publisher` (via the workflow on tag, or by hand) to push server.json to the official MCP Registry. Depends on Task 4 (the registry verifies PyPI ownership via the mcp-name comment). Verify the listing appears. **STOP — owner authorization required.**

### Task 6 (P — GHCR image): REQUIRES GHCR push permission

Publish the Docker image to GHCR with the registry label (via publish.yml on tag, or `docker push` by hand). Verify the image runs the server. Fill the Docker package entry in server.json (a follow-up prep commit). **STOP — owner authorization required.**

### Task 7 (P — third-party registries): REQUIRES per-registry submission under owner identity

File the Docker MCP Catalog PR, run the Smithery push (owner's Smithery auth), and submit to Glama, PulseMCP, and mcp.so using the Task 3 drafts. Each is a public action under the owner's identity. Attach the .mcpb bundle to the v1.0.0 GitHub release. **STOP — owner authorization required (and several need owner accounts/logins the agent cannot access).**

### Task 8 (close-out): after the publishes the owner authorizes

Reconcile which registries went live; update README badges + docs with the real listing URLs; decision-log addenda; move this plan to completed/; final verification; commit `docs: complete phase 9 (distribution and registry rollout)`; push; CI watch. The modernization is then fully complete.
