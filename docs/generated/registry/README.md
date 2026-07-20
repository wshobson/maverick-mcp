# Registry Submission Drafts

Ready-to-paste submission content for MCP registries that accept a
maintainer-filed PR/CLI-push/form, prepared ahead of time so Phase 9 Task 7
(publish) is "paste and submit," not "write from scratch." Nothing in this
directory has been submitted anywhere — see
[`docs/runbooks/releasing.md`](../../runbooks/releasing.md) for the full
publish sequence and its authorization gates.

Every draft uses the canonical server identity `io.github.wshobson/maverick-mcp`
(repo: <https://github.com/wshobson/maverick-mcp>, owner: `wshobson`).

## Drafts in this directory

| File | Registry | Mechanism |
| --- | --- | --- |
| `docker-mcp-catalog.md` | Docker MCP Catalog | GitHub PR against `docker/mcp-registry` adding `servers/maverick-mcp/server.yaml` |
| `smithery.yaml` | Smithery | `smithery` CLI push/deploy of the config at repo root |
| `glama.md` | Glama | GitHub App repo connection (or web form) |
| `pulsemcp.md` | PulseMCP | Web submission (mechanism unconfirmed — see file) |
| `mcp-so.md` | mcp.so | Web submission form or GitHub issue |

Each `.md` file opens with a one-line "how to submit" note. `smithery.yaml`
carries the same note as a YAML header comment, since it is meant to be
copied to the repo root (not consumed from this directory) when the owner
runs the Smithery push.

## Registries NOT drafted here

- **Official MCP Registry** (`registry.modelcontextprotocol.io`): not a
  pull/submit-based registry. It is pushed via `mcp-publisher` against
  `server.json` at the repo root, documented as Step 2 of
  [`docs/runbooks/releasing.md`](../../runbooks/releasing.md). No draft
  artifact belongs here.
- **GitHub-hosted MCP Registry / `modelcontextprotocol/registry` "curated"
  listings**: curated by the registry maintainers from crawled data, not a
  maintainer-filed submission. There is nothing to draft or push.

## Accuracy notes

These drafts were prepared by reading each registry's current public
documentation (fetched July 2026) where it was reachable, not by inventing
schemas. Where a registry's exact current field names, category enum, or
submission URL could not be confirmed (submission forms behind JS/auth,
undocumented enums, or ambiguous/contradictory CLI docs), the corresponding
file says so explicitly and flags "verify at submit time" rather than
presenting a guess as fact. See `.superpowers/sdd/p9-task-3-report.md`
(gitignored scratch, agent-local) for the full per-registry
sourced-vs-flagged breakdown.
