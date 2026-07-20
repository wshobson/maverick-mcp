# Releasing

MaverickMCP has no automated release pipeline yet -- `.github/workflows/publish.yml`
is planned (Phase 9, Task 1) but not merged as of this writing. Every step
below is **owner-run by hand** using local credentials. None of this runs in
CI or as part of an agent's standing goal; the Phase 9 exec plan
(`docs/exec-plans/active/2026-07-20-phase-9-distribution.md`) marks each of
these as a "P-task" that requires explicit owner authorization because the
actions are public and mostly irreversible.

The canonical server identity across every step is `io.github.wshobson/maverick-mcp`.
`README.md` carries the `<!-- mcp-name: io.github.wshobson/maverick-mcp -->`
provenance comment the official registry uses to verify PyPI/repo ownership,
and `server.json` declares the package/transport surface registries read.

v1.0.0 is already tagged and released on GitHub
(`gh release list` shows `v1.0.0`). The steps below take that release the
rest of the way to installable-by-everyone.

## Sequence overview

1. PyPI publish (the wheel/sdist people actually install).
2. Official MCP Registry publish via `mcp-publisher` (depends on step 1 for
   provenance).
3. GHCR image push (independent of steps 1-2; can happen any time).
4. Third-party registry submissions (Docker MCP Catalog, Smithery, Glama,
   PulseMCP, mcp.so).
5. Attach the `.mcpb` bundle to the v1.0.0 GitHub release.

Steps 1 and 3 are independent of each other. Step 2 needs step 1 done first.
Step 4 can happen any time after step 1 (most third-party catalogs just want
a working `pip install`/`uvx` command). Step 5 is independent of everything
else once the bundle is built (Phase 9, Task 2).

## Step 1: PyPI publish

**Credential needed:** either a PyPI trusted-publisher binding configured on
the `wshobson/maverick-mcp` GitHub repo (no token, OIDC-based), or a
`PYPI_API_TOKEN` (a PyPI API token scoped to the `maverick-mcp-server`
project, or an account-wide token for the first publish since the project
doesn't exist on PyPI yet).

**Reversible?** No. A version number published to PyPI can never be reused,
even if yanked. Publishing 1.0.0 is a one-time, permanent action.

### Option A: trusted publishing + tag push (once `publish.yml` exists)

1. In the PyPI project settings (or, for a first-time publish, at
   <https://pypi.org/manage/account/publishing/>), add a trusted publisher:
   - Owner: `wshobson`
   - Repository: `maverick-mcp`
   - Workflow: `publish.yml`
   - Environment: leave blank unless the workflow defines one.
2. Push (or re-push) the `v1.0.0` tag so the workflow's tag trigger fires:
   ```bash
   git push origin v1.0.0
   ```
3. Watch the run: `gh run watch --repo wshobson/maverick-mcp`.

### Option B: manual build and upload (works today, no workflow needed)

```bash
uv build
uv publish  # prompts for a PyPI token, or reads UV_PUBLISH_TOKEN
```

Or with `twine`:

```bash
uv build
uvx twine upload dist/*
```

### After publish (either option)

Verify from a clean environment (no local editable install shadowing the
real package):

```bash
uvx maverick-mcp-server --help
pip install "maverick-mcp-server[backtesting,research]"
```

Then edit the v1.0.0 GitHub release notes to remove any "install from source
until published" phrasing and state real PyPI availability.

## Step 2: official MCP Registry publish

**Depends on:** Step 1 (the registry verifies PyPI ownership via the
`mcp-name` comment in the published package's README, which must match the
`server.json` `name` field).

**Credential needed:** `mcp-publisher` registry auth -- typically a GitHub
OAuth login tied to the `wshobson` account (the registry uses GitHub identity
to authorize `io.github.wshobson/*` server names). See the `mcp-publisher`
CLI's own `login` command for the current auth flow.

**Reversible?** Mostly. The registry supports updating an existing
`server.json` version or marking a server `deprecated`, but the published
history remains visible.

```bash
mcp-publisher login github
mcp-publisher publish
```

Run this from the repo root so `mcp-publisher` picks up `server.json`. If
`.github/workflows/publish.yml` exists and includes an `mcp-publisher` step,
this also runs automatically on the same tag push as Step 1 -- pick one path,
not both.

Verify the listing appears in the registry's search/detail page for
`io.github.wshobson/maverick-mcp`.

## Step 3: GHCR image push

**Credential needed:** a GitHub Personal Access Token (or `GITHUB_TOKEN` in
Actions) with `write:packages` scope, or `docker login ghcr.io` with the
owner's GitHub credentials.

**Reversible?** Partially. Individual image tags/digests can be deleted from
GHCR, but once pulled by others the image content is out in the world.

```bash
docker login ghcr.io -u wshobson
docker build -t ghcr.io/wshobson/maverick-mcp:1.0.0 -t ghcr.io/wshobson/maverick-mcp:latest .
docker push ghcr.io/wshobson/maverick-mcp:1.0.0
docker push ghcr.io/wshobson/maverick-mcp:latest
```

The Dockerfile must carry the
`io.modelcontextprotocol.server.name=io.github.wshobson/maverick-mcp` LABEL
before this step (Phase 9, Task 2). Verify the image runs:

```bash
docker run --rm ghcr.io/wshobson/maverick-mcp:1.0.0 --help
```

### Follow-up: add the Docker package entry to server.json

`server.json` currently ships **without** a Docker/`oci` package entry.
Task 0 considered adding one as a placeholder but decided against it: a
`registry_type: oci` entry has no meaningful placeholder value for
`identifier`/`version` before the image actually exists at that reference,
and an entry that fails registry-side validation (e.g. the registry checking
that the referenced image or PyPI package is pullable) is worse than simply
omitting it until it is real. Once this step's image push lands, add a
package entry like:

```json
{
  "registry_type": "oci",
  "registry_base_url": "https://ghcr.io",
  "identifier": "wshobson/maverick-mcp",
  "version": "1.0.0",
  "transport": { "type": "stdio" }
}
```

to the `packages` array in `server.json`, re-validate against the schema
(see below), and re-run Step 2 to push the updated `server.json`.

## Step 4: third-party registry submissions

Each of these is a public action taken under the owner's identity/account.
Drafts of the submission content live under `docs/generated/registry/`
(Phase 9, Task 3) so this step is "paste and submit," not "write from
scratch."

| Registry | Mechanism | Account needed |
| --- | --- | --- |
| Docker MCP Catalog | GitHub PR against the catalog repo | `wshobson` GitHub account |
| Smithery | `smithery` CLI push | Smithery account linked to the repo |
| Glama | Web submission form | Glama account (or none, if form is anonymous) |
| PulseMCP | Web submission form | PulseMCP account (or none) |
| mcp.so | Web submission form | mcp.so account (or none) |

The official GitHub-hosted MCP Registry (step 2) is curated differently from
these -- it is the canonical registry and is pushed via `mcp-publisher`, not
submitted as a PR/form. These are additional, independent listings.

**Reversible?** PRs can be closed/reverted; form submissions typically allow
delisting on request but there is no self-service undo for most of these.

## Step 5: attach the `.mcpb` bundle to the release

**Credential needed:** `gh` CLI authenticated as a user with push access to
`wshobson/maverick-mcp` (release-asset upload permission).

**Reversible?** Yes -- release assets can be deleted and re-uploaded.

```bash
make bundle   # builds the .mcpb (Phase 9, Task 2)
gh release upload v1.0.0 dist/maverick-mcp-server-1.0.0.mcpb
```

Verify the asset shows up on the release page and that Claude Desktop can
install it as a one-click bundle.

## Validating `server.json` after any edit

```bash
python3 -c "import json; json.load(open('server.json'))"
```

For real schema validation (not just JSON syntax), fetch the schema and
validate against it -- the `$schema` field in `server.json` names the exact
URL to use:

```bash
uv run --with jsonschema python3 - <<'EOF'
import json, urllib.request, jsonschema

data = json.load(open("server.json"))
schema = json.loads(urllib.request.urlopen(data["$schema"]).read())
jsonschema.validate(data, schema)
print("server.json is valid")
EOF
```
