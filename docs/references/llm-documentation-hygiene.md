# LLM Documentation Hygiene

MaverickMCP follows an agent-legible documentation pattern inspired by OpenAI's
"Harness engineering: leveraging Codex in an agent-first world" article.

## Rules

- `AGENTS.md` is a table of contents, not an encyclopedia.
- `docs/` is the repository-local, versioned source of truth.
- Durable docs need a catalog entry.
- Historical/tool-owned docs must be marked as historical.
- Stale docs should be deleted after current facts are preserved.
- Prefer links and indexes over copying the same guidance into many files.
- If a rule must never drift, add a checker, test, or CI job.

## Current Enforcement

- `make docs-check` validates tracked Markdown/text docs against the catalog and
  checks relative Markdown links.
- `docs/CATALOG.md` records deleted and consolidated artifacts.
- Root agent files are intentionally short.
