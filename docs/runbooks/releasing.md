# Releasing

`.github/workflows/publish.yml` automates steps 1-3: a `v*` tag push, or a
manual `workflow_dispatch` run with `confirm=publish`, runs `build`, then
`publish-pypi` -> `publish-mcp-registry` and, in parallel, `publish-ghcr`.
Steps 4 and 5 are by hand. Every step is **owner-run**: none of this runs as
part of an agent's standing goal; the Phase 9 exec plan
(`docs/exec-plans/active/2026-07-20-phase-9-distribution.md`) marks each of
these as a "P-task" that requires explicit owner authorization because the
actions are public and mostly irreversible.

The canonical server identity across every step is `io.github.wshobson/maverick-mcp`.
`README.md` carries the `<!-- mcp-name: io.github.wshobson/maverick-mcp -->`
provenance comment the official registry uses to verify PyPI/repo ownership,
and `server.json` declares the package/transport surface registries read.

v1.0.0 was released on GitHub only and never reached PyPI. Pushing the
v1.1.0 tag on 2026-09-05 ran the publish workflow: `build` and `publish-ghcr`
succeeded, `publish-pypi` failed with `invalid-publisher` (see Step 1), and
`publish-mcp-registry` was skipped. The steps below take that release the
rest of the way to installable-by-everyone.

## Sequence overview

1. PyPI publish (the wheel/sdist people actually install).
2. Official MCP Registry publish via `mcp-publisher` (depends on step 1 for
   provenance).
3. GHCR image push (independent of steps 1-2; can happen any time).
4. Third-party registry submissions (Docker MCP Catalog, Smithery, Glama,
   PulseMCP, mcp.so).
5. Attach the `.mcpb` bundle to the v1.1.0 GitHub release.

Steps 1 and 3 are independent of each other. Step 2 needs step 1 done first.
Step 4 can happen any time after step 1 (most third-party catalogs just want
a working `pip install`/`uvx` command). Step 5 needs step 1: the bundle
(Phase 9, Task 2) launches the PyPI package.

## Step 1: PyPI publish

**Credential needed:** either a PyPI trusted-publisher binding configured on
the `wshobson/maverick-mcp` GitHub repo (no token, OIDC-based), or a
`PYPI_API_TOKEN` (a PyPI API token scoped to the `maverick-mcp-server`
project, which can only exist once the name transfer below completes).

**Reversible?** No. A version number published to PyPI can never be reused,
even if yanked. Publishing 1.1.0 is a one-time, permanent action.

**Status (2026-09-26):** the PyPI name `maverick-mcp-server` is held by an
unrelated, dormant project (all releases removed 2026-06-10, source
repositories gone), so the trusted-publishing exchange fails with
`invalid-publisher` no matter how the publisher is configured. A PEP 541
transfer request is on file with PyPI support (https://github.com/pypi/support/issues/12150);
it has had no action since it was filed on 2026-09-05. After the transfer: add the
publisher on the project's own Publishing settings (owner `wshobson`,
repository `maverick-mcp`, workflow `publish.yml`, environment `pypi`), then
run `gh workflow run publish.yml --ref v1.1.0 -f confirm=publish`, which
rebuilds the tag and publishes PyPI, the MCP Registry, and GHCR (re-pushing
the `1.1.0` and `latest` image tags). Without `--ref v1.1.0` the run builds
`main`, which carries post-release changes under the same version number,
and tags the image `main`. Until then the README must not point installs at
the PyPI name.

### Option A: trusted publishing + the publish workflow

1. In the PyPI project's own Publishing settings, add a trusted publisher.
   (The pending-publisher form at
   <https://pypi.org/manage/account/publishing/> rejects this name because
   the project already exists.)
   - Owner: `wshobson`
   - Repository: `maverick-mcp`
   - Workflow: `publish.yml`
   - Environment: `pypi`. The workflow's `publish-pypi` job runs inside the
     GitHub environment named `pypi`, so the publisher must name it; a
     blank environment does not match and PyPI rejects the run as
     `invalid-publisher`.
2. Start the workflow. For a new version, pushing its tag fires the tag
   trigger (`git push origin vX.Y.Z`). The `v1.1.0` tag is already on
   GitHub, and pushing an existing tag again fires nothing, so dispatch the
   run against the tag instead:
   ```bash
   gh workflow run publish.yml --repo wshobson/maverick-mcp --ref v1.1.0 -f confirm=publish
   ```
3. Watch the run: `gh run watch --repo wshobson/maverick-mcp`.

### Option B: manual build and upload (no workflow needed)

Also blocked until the name transfer. Build from a clean checkout of the
release tag, not `main`: `main` already carries post-release changes under
the same version number.

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

Then edit the v1.1.0 GitHub release notes to remove any "install from source
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
# Command syntax matches the registry's quickstart as of 2026-07-20; confirm
# against `mcp-publisher --help` on the installed CLI before running.
mcp-publisher login github
mcp-publisher publish
```

Run this from the repo root so `mcp-publisher` picks up `server.json`. The
`publish-mcp-registry` job in `.github/workflows/publish.yml` already runs
`mcp-publisher` (with `login github-oidc`) once `publish-pypi` succeeds in
the same run -- pick one path, not both.

Verify the listing appears in the registry's search/detail page for
`io.github.wshobson/maverick-mcp`.

## Step 3: GHCR image push

**Credential needed:** a GitHub Personal Access Token (or `GITHUB_TOKEN` in
Actions) with `write:packages` scope, or `docker login ghcr.io` with the
owner's GitHub credentials.

**Reversible?** Partially. Individual image tags/digests can be deleted from
GHCR, but once pulled by others the image content is out in the world.

**Status:** done for v1.1.0. The workflow's `publish-ghcr` job pushed
`ghcr.io/wshobson/maverick-mcp:1.1.0` and `:latest` on 2026-09-05. The manual
commands below are for a later version; build from a checkout of that
version's tag.

```bash
docker login ghcr.io -u wshobson
docker build -t ghcr.io/wshobson/maverick-mcp:X.Y.Z -t ghcr.io/wshobson/maverick-mcp:latest .
docker push ghcr.io/wshobson/maverick-mcp:X.Y.Z
docker push ghcr.io/wshobson/maverick-mcp:latest
```

The Dockerfile carries the
`io.modelcontextprotocol.server.name=io.github.wshobson/maverick-mcp` LABEL
(Phase 9, Task 2). Verify the image runs:

```bash
docker run --rm ghcr.io/wshobson/maverick-mcp:X.Y.Z python -m maverick.server --help
```

That command fits images built from the current two-stage Dockerfile, which
puts the venv on `PATH`. The published `1.1.0` image predates it: its venv
is not on `PATH`, so use `uv run python -m maverick.server --help` there.

### The Docker package entry in server.json

`server.json` has carried an `oci` package entry for the GHCR image since
2026-07-20, in the registry's current camelCase schema:

```json
{
  "registryType": "oci",
  "identifier": "ghcr.io/wshobson/maverick-mcp:1.1.0",
  "version": "1.1.0",
  "transport": { "type": "stdio" }
}
```

For a new release, bump its `identifier` tag and `version` with the rest of
`server.json`, re-validate against the schema (see below), and let Step 2
push the updated `server.json`.

## Step 4: third-party registry submissions

Each of these is a public action taken under the owner's identity/account.
Drafts of the submission content live under `docs/generated/registry/`
(Phase 9, Task 3) so this step is "paste and submit," not "write from
scratch."

| Registry | Mechanism | Account needed |
| --- | --- | --- |
| Docker MCP Catalog | GitHub PR against the catalog repo | `wshobson` GitHub account |
| Smithery | `smithery` CLI push | Smithery account linked to the repo |
| Glama | Indexes the repo from GitHub; claim the existing listing | Glama account |
| PulseMCP | Crawled; its submit page is paused and points to the official registry | none |
| mcp.so | Web submission form | mcp.so account (or none) |

**Status (2026-09-26):** the Docker MCP Catalog PR,
[docker/mcp-registry#4490](https://github.com/docker/mcp-registry/pull/4490),
was filed on 2026-07-20 against a pre-v1.1.0 commit and has had no activity
since. Glama already lists the repo, unclaimed, at
<https://glama.ai/mcp/servers/@wshobson/maverick-mcp>. PulseMCP lists it from
its own crawl at
<https://www.pulsemcp.com/servers/wshobson-maverick-financial-analysis>, with
a stale pre-v1.0 description. Nothing has been submitted to Smithery, Glama,
PulseMCP, or mcp.so.

The official GitHub-hosted MCP Registry (step 2) is curated differently from
these -- it is the canonical registry and is pushed via `mcp-publisher`, not
submitted as a PR/form. These are additional, independent listings.

**Reversible?** PRs can be closed/reverted; form submissions typically allow
delisting on request but there is no self-service undo for most of these.

## Step 5: attach the `.mcpb` bundle to the release

**Credential needed:** `gh` CLI authenticated as a user with push access to
`wshobson/maverick-mcp` (release-asset upload permission).

**Reversible?** Yes -- release assets can be deleted and re-uploaded.

**Status:** no release carries a bundle. The v1.0.0 asset was removed on
2026-09-05 because it launched the PyPI name.

```bash
make bundle   # builds dist/maverick-mcp.mcpb (thin bundle: launches the
              # PyPI package via uvx; requires uv + the PyPI publish first)
npx @anthropic-ai/mcpb validate dist/manifest.json  # official validator
gh release upload v1.1.0 dist/maverick-mcp.mcpb
```

The bundle vendors no code -- its manifest launches
`uvx --from maverick-mcp-server==<version> maverick-mcp --transport stdio`,
so it only works after Step 1 (PyPI) and on machines with uv installed.
Run the validator before uploading; the mcpb CLI is the authority on
manifest correctness. Verify the asset shows up on the release page and
that Claude Desktop can install it as a one-click bundle.

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
