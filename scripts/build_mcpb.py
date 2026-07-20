"""Build the Claude Desktop .mcpb bundle for maverick-mcp.

Produces dist/maverick-mcp.mcpb: a zip containing a manifest.json (mcpb
manifest spec 0.3, per anthropics/mcpb MANIFEST.md) whose mcp_config
launches the PyPI-published package via uvx. The bundle deliberately does
NOT vendor the Python runtime or dependencies -- uvx resolves the published
`maverick-mcp-server` wheel at first launch, which keeps the bundle tiny
and always consistent with the released package. Consequence: the bundle
only works once the package is on PyPI (Phase 9 Task 4) and requires uv on
the user's machine.

Validate before attaching to a release (the official CLI is the authority
on manifest correctness): `npx @anthropic-ai/mcpb validate dist/manifest.json`
-- see docs/runbooks/releasing.md.

Usage: uv run python scripts/build_mcpb.py  (or `make bundle`)
"""

from __future__ import annotations

import json
import pathlib
import tomllib
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent


def build() -> pathlib.Path:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    version = pyproject["project"]["version"]

    manifest = {
        "manifest_version": "0.3",
        "name": "maverick-mcp",
        "display_name": "Maverick MCP",
        "version": version,
        "description": (
            "Personal-use stock analysis MCP server: market data, screening, "
            "technical indicators, and portfolio tracking. Educational use "
            "only; not financial advice."
        ),
        "author": {
            "name": "Seth Hobson",
            "url": "https://github.com/wshobson",
        },
        "repository": {
            "type": "git",
            "url": "https://github.com/wshobson/maverick-mcp.git",
        },
        "license": "MIT",
        "server": {
            "type": "binary",
            "entry_point": "server/launch.txt",
            "mcp_config": {
                "command": "uvx",
                "args": [
                    "--from",
                    f"maverick-mcp-server=={version}",
                    "maverick-mcp",
                    "--transport",
                    "stdio",
                ],
            },
        },
        "compatibility": {
            "platforms": ["darwin", "win32", "linux"],
        },
    }

    launch_note = (
        "This bundle launches the PyPI-published maverick-mcp-server package "
        "via uvx (see manifest.json mcp_config). It requires uv installed "
        "and the package published to PyPI. No server code is vendored here."
    )

    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    out = dist / "maverick-mcp.mcpb"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
        zf.writestr("server/launch.txt", launch_note + "\n")
    # A sibling manifest copy makes `mcpb validate` easy without unzipping.
    (dist / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


if __name__ == "__main__":
    path = build()
    print(f"built {path} ({path.stat().st_size} bytes)")
