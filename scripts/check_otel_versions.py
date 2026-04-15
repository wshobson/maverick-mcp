#!/usr/bin/env python
"""Fail CI if `uv.lock` pins the OpenTelemetry package set at misaligned versions.

Why: chromadb depends on `opentelemetry-exporter-otlp-proto-grpc>=1.2.0` with no
coupling to `opentelemetry-api` / `opentelemetry-sdk`. uv's resolver is
occasionally free to pick a split set where api/sdk are modern (e.g. `1.41.0`)
but the exporter/proto packages are ancient (e.g. `1.11.1`). That combination
crashes at import time because the old `_pb2.py` files in
`opentelemetry-proto 1.11.x` were generated against `protoc < 3.19` and are
rejected by the modern `protobuf` runtime with:

    TypeError: Descriptors cannot be created directly.

See `docs/runbooks/otel-protobuf-crash.md` for the full incident.

This script asserts that every `opentelemetry-*` stable package in `uv.lock`
(api, sdk, exporter-otlp-proto-grpc, exporter-otlp-proto-common, proto) shares
the same `1.X.Y`. `opentelemetry-semantic-conventions` uses a different
versioning scheme (`0.X{b,rc}N`) and is exempt.

Exit 0: aligned. Exit 1: drift detected (print the mismatch table).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Stable packages that must track the same `1.X.Y`.
STABLE_PKGS = (
    "opentelemetry-api",
    "opentelemetry-sdk",
    "opentelemetry-exporter-otlp-proto-grpc",
    "opentelemetry-exporter-otlp-proto-common",
    "opentelemetry-proto",
)

LOCK_PATH = Path(__file__).resolve().parent.parent / "uv.lock"

_BLOCK_RE = re.compile(
    r'^name = "(?P<name>[^"]+)"\nversion = "(?P<version>[^"]+)"', re.MULTILINE
)


def parse_lockfile(lock_text: str) -> dict[str, str]:
    """Return {package_name: version} for every [[package]] block in uv.lock."""
    return {m["name"]: m["version"] for m in _BLOCK_RE.finditer(lock_text)}


def main() -> int:
    if not LOCK_PATH.exists():
        print(f"ERROR: {LOCK_PATH} not found", file=sys.stderr)
        return 1

    pins = parse_lockfile(LOCK_PATH.read_text())
    found = {p: pins.get(p) for p in STABLE_PKGS}

    # proto-common only exists on modern otel (>=1.12). Absence is only a
    # problem if the exporter is modern — old exporters vendored everything.
    required = {p: v for p, v in found.items() if v is not None}
    if "opentelemetry-exporter-otlp-proto-grpc" not in required:
        # chromadb / vectors extra not installed in this lock — nothing to check.
        return 0

    exporter_version = required["opentelemetry-exporter-otlp-proto-grpc"]
    exporter_major_minor = tuple(int(x) for x in exporter_version.split(".")[:2])

    # proto-common only required once exporter split it off (otel >= 1.12).
    if (
        exporter_major_minor >= (1, 12)
        and "opentelemetry-exporter-otlp-proto-common" not in required
    ):
        print(
            "ERROR: opentelemetry-exporter-otlp-proto-grpc is "
            f"{exporter_version} but opentelemetry-exporter-otlp-proto-common "
            "is missing from uv.lock. Modern exporter versions (>=1.12) split "
            "this out — both must be pinned.",
            file=sys.stderr,
        )
        return 1

    versions = set(required.values())
    if len(versions) > 1:
        print(
            "ERROR: OpenTelemetry stable packages pinned at mismatched versions "
            "in uv.lock. This reproduces the protobuf '_pb2.py' crash. See "
            "docs/runbooks/otel-protobuf-crash.md.",
            file=sys.stderr,
        )
        print("", file=sys.stderr)
        width = max(len(p) for p in required)
        for pkg, ver in sorted(required.items()):
            print(f"  {pkg:<{width}}  {ver}", file=sys.stderr)
        print("", file=sys.stderr)
        print(
            "Fix: pin the exporter explicitly in `pyproject.toml::[project.optional-dependencies].vectors`, "
            "e.g. `opentelemetry-exporter-otlp-proto-grpc>=1.30.0`, then "
            "`uv lock`.",
            file=sys.stderr,
        )
        return 1

    print(f"OpenTelemetry packages aligned at {versions.pop()} ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
