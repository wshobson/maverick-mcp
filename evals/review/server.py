"""Local review UI for one tool-surface trace run.

Serves `app.html` and a small JSON API over the run folder, bound to
127.0.0.1 only. Nothing here calls a model or any other network service.

    uv run python evals/review/server.py --run evals/tool_surface/runs/<run>

Annotations live in `<run>/annotations.json` and hold only the reviewer's own
words: the browser app is the only writer. `<run>/patterns.json` holds an
agent's DRAFT failure-mode taxonomy, which the app only displays.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

HOST = "127.0.0.1"
DEFAULT_PORT = 8765
APP_HTML = Path(__file__).with_name("app.html")
ANNOTATIONS_FILE = "annotations.json"
PATTERNS_FILE = "patterns.json"
MAX_BODY_BYTES = 5_000_000

VERDICTS = frozenset({"pass", "fail", "defer", "unset"})
ANCHOR_FIELDS = frozenset({"query", "text", "tool_args", "tool_result", "final_answer"})
ANNOTATION_KEYS = frozenset({"verdict", "trace_note", "spans", "updated"})
SPAN_KEYS = frozenset(
    {"id", "anchor", "start", "end", "quote", "note", "created", "updated"}
)

# Per-trace features for the map view, in column order.
FEATURE_NAMES = (
    "tool_calls",
    "tool_errors",
    "turns",
    "tool_result_chars",
    "final_answer_chars",
    "cost_usd",
)
CLUSTERS = 4


# --- files -----------------------------------------------------------------


def read_json(path: Path, default: Any) -> Any:
    """Parse `path`, or return `default` when it does not exist.

    A file that exists but does not parse raises, so a later write can never
    replace a damaged annotations file with an empty one.
    """
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _replace_with(path: Path, payload: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        os.fchmod(fd, 0o644)  # mkstemp makes 0600; match the other run files
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_json_atomic(path: Path, data: Any) -> None:
    """Write `data` as JSON through a temp file and a rename.

    The previous version, if any, is kept as `<name>.bak`.
    """
    payload = (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode()
    if path.exists():
        _replace_with(path.with_name(path.name + ".bak"), path.read_bytes())
    _replace_with(path, payload)


def load_traces(run_dir: Path) -> list[dict[str, Any]]:
    """Every trace in `<run>/traces`, ordered by case id (q01, q02, ...)."""
    traces = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (run_dir / "traces").glob("*.json")
    ]
    return sorted(traces, key=lambda trace: str(trace["case"]["id"]))


# --- validation --------------------------------------------------------------


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_span(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("a span must be an object")
    _require(set(value) <= SPAN_KEYS, f"unknown span keys: {set(value) - SPAN_KEYS}")
    anchor = value.get("anchor")
    if not isinstance(anchor, dict):
        raise ValueError("span.anchor must be an object")
    _require(anchor.get("field") in ANCHOR_FIELDS, "span.anchor.field is unknown")
    msg = anchor.get("msg")
    _require(msg is None or _is_int(msg), "span.anchor.msg must be an int or null")
    start, end = value.get("start"), value.get("end")
    if not (_is_int(start) and _is_int(end)):
        raise ValueError("span.start and span.end must be ints")
    _require(0 <= start < end, "span offsets must satisfy 0 <= start < end")
    for key in ("id", "quote", "note", "created", "updated"):
        _require(isinstance(value.get(key), str), f"span.{key} must be a string")
    _require(bool(value["id"]), "span.id must not be empty")
    return {
        "id": value["id"],
        "anchor": {"msg": msg, "field": anchor["field"]},
        "start": start,
        "end": end,
        "quote": value["quote"],
        "note": value["note"],
        "created": value["created"],
        "updated": value["updated"],
    }


def validate_annotation(value: object) -> dict[str, Any]:
    """One trace's annotation, exactly as the app sent it, or ValueError."""
    if not isinstance(value, dict):
        raise ValueError("an annotation must be an object")
    extra = set(value) - ANNOTATION_KEYS
    _require(not extra, f"unknown annotation keys: {extra}")
    verdict = value.get("verdict", "unset")
    _require(verdict in VERDICTS, f"verdict must be one of {sorted(VERDICTS)}")
    trace_note = value.get("trace_note", "")
    _require(isinstance(trace_note, str), "trace_note must be a string")
    spans = value.get("spans", [])
    if not isinstance(spans, list):
        raise ValueError("spans must be a list")
    updated = value.get("updated", "")
    _require(isinstance(updated, str), "updated must be a string")
    return {
        "verdict": verdict,
        "trace_note": trace_note,
        "spans": [validate_span(span) for span in spans],
        "updated": updated,
    }


def apply_annotation(
    doc: Mapping[str, Any], body: object, known_ids: Sequence[str]
) -> dict[str, Any]:
    """The annotations document with one trace's annotation replaced.

    `body` is `{"trace_id": ..., "annotation": {...}}` from the app.
    """
    if not isinstance(body, dict):
        raise ValueError("the body must be an object")
    trace_id = body.get("trace_id")
    _require(trace_id in known_ids, f"unknown trace_id: {trace_id!r}")
    traces = dict(doc.get("traces", {}))
    traces[str(trace_id)] = validate_annotation(body.get("annotation"))
    return {"version": 1, "traces": {key: traces[key] for key in sorted(traces)}}


def validate_patterns(value: object) -> dict[str, Any]:
    """A draft failure-mode taxonomy whose notes point at the reviewer's notes."""
    if not isinstance(value, dict):
        raise ValueError("patterns must be an object")
    modes = value.get("failure_modes")
    if not isinstance(modes, list):
        raise ValueError("patterns.failure_modes must be a list")
    for mode in modes:
        if not isinstance(mode, dict):
            raise ValueError("a failure mode must be an object")
        name = mode.get("name")
        _require(isinstance(name, str) and bool(name), "a failure mode needs a name")
        _require(
            isinstance(mode.get("description", ""), str),
            "a failure mode description must be a string",
        )
        notes = mode.get("notes")
        if not isinstance(notes, list):
            raise ValueError("failure mode notes must be a list")
        for ref in notes:
            _require(
                isinstance(ref, dict) and isinstance(ref.get("trace_id"), str),
                "each note reference needs a trace_id",
            )
            span_id = ref.get("span_id")
            _require(
                span_id is None or isinstance(span_id, str),
                "a note reference span_id must be a string or null",
            )
    return value


# --- features and projection -------------------------------------------------


def _as_text(value: object) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value)


def _tool_calls(trace: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        message
        for message in trace.get("messages", [])
        if (message.get("type") or message.get("kind")) == "tool_call"
    ]


def result_status(result: object) -> str | None:
    """The `status` field of a JSON tool result, if there is one."""
    try:
        parsed = json.loads(_as_text(result))
    except ValueError:
        return None
    status = parsed.get("status") if isinstance(parsed, dict) else None
    return status if isinstance(status, str) else None


def trace_features(trace: Mapping[str, Any]) -> dict[str, float]:
    """The map-view features for one trace, keyed by FEATURE_NAMES.

    A tool error is a call flagged `is_error` or a JSON result whose `status`
    is "error" (the server reports most failures the second way).
    """
    calls = _tool_calls(trace)
    return {
        "tool_calls": float(len(calls)),
        "tool_errors": float(
            sum(
                1
                for call in calls
                if call.get("is_error") or result_status(call.get("result")) == "error"
            )
        ),
        "turns": float(trace.get("num_turns") or 0),
        "tool_result_chars": float(
            sum(len(_as_text(call.get("result"))) for call in calls)
        ),
        "final_answer_chars": float(len(trace.get("final_answer") or "")),
        "cost_usd": float(trace.get("notional_cost_usd") or 0.0),
    }


def trace_stats(trace: Mapping[str, Any]) -> dict[str, float]:
    """The features plus the header numbers the app checks for outliers."""
    calls = _tool_calls(trace)
    return {
        **trace_features(trace),
        "max_tool_result_chars": float(
            max((len(_as_text(call.get("result"))) for call in calls), default=0)
        ),
        "duration_ms": float(trace.get("duration_ms") or 0),
    }


def project_2d(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA of non-negative features to two dimensions.

    Features go through log1p (sizes and costs are skewed) and a z-score
    (a constant column stays zero). Returns the n x 2 coordinates, the share
    of variance each component explains, and the 2 x f loadings. Each
    component's sign is fixed so its largest loading (the first, on a
    tie) is positive.
    """
    data = np.log1p(np.asarray(matrix, dtype=float))
    rows, cols = data.shape
    if rows < 2:
        return np.zeros((rows, 2)), np.zeros(2), np.zeros((2, cols))
    spread = data.std(axis=0)
    spread[spread == 0] = 1.0
    scaled = (data - data.mean(axis=0)) / spread
    _, singular, components = np.linalg.svd(scaled, full_matrices=False)
    loadings = np.zeros((2, cols))
    loadings[: min(2, len(components))] = components[:2]
    for row in loadings:
        # Take the first loading within float noise of the largest, so a tie
        # (for example two loadings at +/-0.7071) resolves the same way on
        # every BLAS build.
        magnitudes = np.abs(row)
        lead = int(np.flatnonzero(magnitudes >= magnitudes.max() - 1e-9)[0])
        if row[lead] < 0:
            row *= -1
    variance = singular**2
    explained = np.zeros(2)
    if variance.sum() > 0:
        top = variance[:2] / variance.sum()
        explained[: len(top)] = top
    return scaled @ loadings.T, explained, loadings


def kmeans(
    points: np.ndarray, k: int = CLUSTERS, seed: int = 0, max_iter: int = 100
) -> np.ndarray:
    """Cluster labels from k-means with k-means++ seeding.

    Deterministic for a given seed. Labels are renumbered so cluster 0 has
    the smallest mean x, which keeps colors steady across runs.
    """
    count = len(points)
    k = min(k, count)
    if count == 0:
        return np.zeros(0, dtype=int)
    rng = np.random.default_rng(seed)
    centers = [points[rng.integers(count)]]
    for _ in range(1, k):
        gaps = ((points[:, None, :] - np.array(centers)[None]) ** 2).sum(-1).min(1)
        total = gaps.sum()
        pick = rng.integers(count) if total == 0 else rng.choice(count, p=gaps / total)
        centers.append(points[pick])
    means = np.array(centers, dtype=float)
    labels = np.full(count, -1)
    for _ in range(max_iter):
        dist = ((points[:, None, :] - means[None]) ** 2).sum(-1)
        new = dist.argmin(1)
        if np.array_equal(new, labels):
            break
        labels = new
        for cluster in range(k):
            members = points[labels == cluster]
            means[cluster] = (
                members.mean(0) if len(members) else points[dist.min(1).argmax()]
            )
    order = np.argsort(means[:, 0], kind="stable")
    rank = np.empty(k, dtype=int)
    rank[order] = np.arange(k)
    return rank[labels]


def build_graph(traces: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The map view: 2-D PCA coordinates and a k-means cluster per trace."""
    features = [trace_features(trace) for trace in traces]
    matrix = np.array(
        [[row[name] for name in FEATURE_NAMES] for row in features], dtype=float
    ).reshape(len(features), len(FEATURE_NAMES))
    coords, explained, loadings = project_2d(matrix)
    labels = kmeans(coords)
    points = [
        {
            "id": trace["case"]["id"],
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "cluster": int(labels[i]),
            "task": trace["case"].get("task"),
            "request_type": trace["case"].get("request_type"),
            "data_state": trace["case"].get("data_state"),
            "query": trace["case"].get("query"),
            "features": features[i],
        }
        for i, trace in enumerate(traces)
    ]
    return {
        "features": list(FEATURE_NAMES),
        "explained_variance": [float(value) for value in explained],
        "loadings": [
            {
                name: float(weight)
                for name, weight in zip(FEATURE_NAMES, row, strict=True)
            }
            for row in loadings
        ],
        "clusters": int(labels.max()) + 1 if len(labels) else 0,
        "points": points,
    }


# --- the app -----------------------------------------------------------------


class ReviewApp:
    """The run folder's data behind the HTTP API."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.traces = load_traces(run_dir)
        self.ids = [str(trace["case"]["id"]) for trace in self.traces]
        self.lock = threading.Lock()

    @property
    def annotations_path(self) -> Path:
        return self.run_dir / ANNOTATIONS_FILE

    @property
    def patterns_path(self) -> Path:
        return self.run_dir / PATTERNS_FILE

    def samples(self) -> dict[str, Any]:
        return {
            "run": self.run_dir.name,
            "samples": [
                {"id": trace_id, "trace": trace, "stats": trace_stats(trace)}
                for trace_id, trace in zip(self.ids, self.traces, strict=True)
            ],
        }

    def graph(self) -> dict[str, Any]:
        return build_graph(self.traces)

    def annotations(self) -> dict[str, Any]:
        return read_json(self.annotations_path, {"version": 1, "traces": {}})

    def save_annotation(self, body: object) -> dict[str, Any]:
        with self.lock:
            doc = apply_annotation(self.annotations(), body, self.ids)
            write_json_atomic(self.annotations_path, doc)
        return {"ok": True, "saved_at": datetime.now(UTC).isoformat()}

    def patterns(self) -> Any:
        return read_json(self.patterns_path, None)

    def save_patterns(self, body: object) -> dict[str, Any]:
        doc = validate_patterns(body)
        with self.lock:
            write_json_atomic(self.patterns_path, doc)
        return {"ok": True, "saved_at": datetime.now(UTC).isoformat()}


def make_handler(app: ReviewApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "MaverickTraceReview/1"

        def _local_names(self) -> set[str]:
            port = self.server.server_address[1]
            return {f"127.0.0.1:{port}", f"localhost:{port}"}

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, data: Any) -> None:
            body = json.dumps(data, ensure_ascii=False).encode()
            self._send(status, body, "application/json; charset=utf-8")

        def _host_ok(self) -> bool:
            # Refuse other Host names so a DNS-rebinding page cannot reach us.
            if self.headers.get("Host") in self._local_names():
                return True
            self._json(403, {"error": "unexpected Host header"})
            return False

        def do_GET(self) -> None:
            if not self._host_ok():
                return
            path = self.path.split("?", 1)[0]
            try:
                if path in ("/", "/index.html"):
                    self._send(200, APP_HTML.read_bytes(), "text/html; charset=utf-8")
                elif path == "/api/samples":
                    self._json(200, app.samples())
                elif path == "/api/annotations":
                    self._json(200, app.annotations())
                elif path == "/api/graph":
                    self._json(200, app.graph())
                elif path == "/api/patterns":
                    self._json(200, app.patterns())
                else:
                    self._json(404, {"error": f"no route for {path}"})
            except Exception as exc:  # report, keep serving
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

        def do_POST(self) -> None:
            if not self._host_ok():
                return
            # Only this page may write: other origins are refused, and the
            # JSON content type forces a CORS preflight that we never grant.
            origin = self.headers.get("Origin")
            if origin is not None and origin not in {
                f"http://{name}" for name in self._local_names()
            }:
                self._json(403, {"error": "cross-origin writes are refused"})
                return
            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("application/json"):
                self._json(415, {"error": "send application/json"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                self._json(413, {"error": "body too large"})
                return
            path = self.path.split("?", 1)[0]
            try:
                body = json.loads(self.rfile.read(length) or b"null")
                if path == "/api/annotations":
                    self._json(200, app.save_annotation(body))
                elif path == "/api/patterns":
                    self._json(200, app.save_patterns(body))
                else:
                    self._json(404, {"error": f"no route for {path}"})
            except ValueError as exc:  # includes JSON decode errors
                self._json(400, {"error": str(exc)})
            except Exception as exc:
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
            # The app polls every 10 s; log only failures.
            if isinstance(code, int) and code >= 400:
                super().log_request(code, size)

    return Handler


def make_server(app: ReviewApp, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((HOST, port), make_handler(app))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Local review UI for one trace run.")
    parser.add_argument(
        "--run", required=True, type=Path, help="run folder holding traces/"
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    run_dir = args.run.resolve()
    if not (run_dir / "traces").is_dir():
        parser.error(f"{run_dir} has no traces/ folder")
    app = ReviewApp(run_dir)
    server = make_server(app, args.port)
    print(f"Reviewing {len(app.ids)} traces from {run_dir}")
    print(f"Open http://{HOST}:{server.server_address[1]}/  (Ctrl+C stops it)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
