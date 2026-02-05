# Workflow - MaverickMCP

## Test-Driven Development

### Policy: Strict TDD (for behavior changes)

Tests are **required before implementation** when changing behavior or adding new features. Follow Red-Green-Refactor:

1. **Red** - Write a failing test that defines expected behavior
2. **Green** - Implement the minimum change to pass
3. **Refactor** - Improve clarity while keeping tests green

### Testing Commands

```bash
make test
make test-all
make test-specific TEST=your_test_name
make test-cov
```

### When Tests Can Be Deferred

- Exploratory spikes (must be clearly marked in the track plan)
- Pure documentation changes
- Low-risk config-only changes (with verification notes)

## LLM Judge

**Current status:** No dedicated judge harness is configured yet.

- For now, record **“LLM Judge: N/A”** in task summaries unless a track defines a specific judge run.
- For end-to-end changes, prefer adding a reproducible smoke check (script or documented steps) as the track’s “LLM Judge” substitute.

## Commit Strategy

**Squash per track** (one clean commit per feature/fix on `main`).

- Keep granular commits on feature branches for review/debugging
- Use clear commit messages describing what changed and why

## Code Review

**Recommended for non-trivial changes**

Review checklist:

- [ ] Tests cover behavior changes
- [ ] Tool outputs remain stable and backwards compatible where expected
- [ ] Error handling is actionable (missing API keys, provider outages)
- [ ] Performance impact considered (caching, DB queries, network calls)

## Verification Checkpoints

**Verify at track completion**

Final verification should include:

- Running the affected test suite
- Exercising the impacted MCP tools (manually or via a script)
- Checking tool output structure and performance

## Task Lifecycle

```
pending → in_progress → completed
                ↓
            blocked → pending (after resolution)
```

## Branch Naming

```
<type>/<short-description>
```

Examples:

- `feat/portfolio-cost-basis`
- `fix/cache-ttl-regression`
- `chore/deps-uv-sync`

