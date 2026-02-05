# Specification: Modernization Implementation (Safe Wins)

**Track ID:** modernization-implementation_20260205
**Type:** Chore
**Created:** 2026-02-05
**Status:** Active

## Summary

Implement the P0 “safe wins” identified in `project-audit-modernization_20260205`, focusing on a single golden path and removing high-risk global side effects.

## Scope

- Implement minimal, safe changes (no major refactors) that improve maintainability and correctness.
- Follow strict TDD for behavior changes.
- No FastMCP major version upgrades in this track.

## Acceptance Criteria

- [ ] STDIO “golden path” is safe: importing/starting the server does not write non-protocol output to stdout
- [ ] SSE compatibility patch is applied only when SSE transport is used (no import-time monkey-patching)
- [ ] Module-level `load_dotenv()` and `logging.basicConfig()` side effects are removed from library modules and centralized in the server bootstrap path
- [ ] `make dev` / `scripts/dev.sh` defaults align with the golden path (streamable HTTP for HTTP dev, stdio for Claude Desktop)
- [ ] Tests cover the above behaviors
