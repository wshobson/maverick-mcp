"""evals/review/server.py: the annotation file handling and the map-view
projection. Pure functions and a ReviewApp over a temp run folder; no server
is started and nothing touches the network."""

import http.client
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from evals.review import server

STAMP = "2026-09-26T15:00:00Z"


def _span(**overrides: Any) -> dict[str, Any]:
    span: dict[str, Any] = {
        "id": "s-1",
        "anchor": {"msg": 2, "field": "tool_result"},
        "start": 4,
        "end": 11,
        "quote": "success",
        "note": "status says success on empty data",
        "created": STAMP,
        "updated": STAMP,
    }
    return span | overrides


def _annotation(**overrides: Any) -> dict[str, Any]:
    annotation: dict[str, Any] = {
        "verdict": "fail",
        "trace_note": "my words",
        "spans": [_span()],
        "updated": STAMP,
    }
    return annotation | overrides


def _trace(case_id: str, **overrides: Any) -> dict[str, Any]:
    trace: dict[str, Any] = {
        "case": {
            "id": case_id,
            "task": "lookup",
            "request_type": "specified",
            "data_state": "seeded",
            "query": f"query {case_id}",
        },
        "messages": [
            {
                "type": "tool_call",
                "name": "mcp__maverick__market_data_get_quote",
                "arguments": {"ticker": "X"},
                "result": '{"status": "success"}',
                "is_error": False,
            },
            {"type": "text", "text": "done"},
        ],
        "final_answer": "done",
        "num_turns": 2,
        "notional_cost_usd": 0.02,
        "duration_ms": 5000,
    }
    return trace | overrides


def _run_dir(tmp_path: Path, traces: list[dict[str, Any]]) -> Path:
    run = tmp_path / "run"
    (run / "traces").mkdir(parents=True)
    for trace in traces:
        path = run / "traces" / f"{trace['case']['id']}.json"
        path.write_text(json.dumps(trace))
    return run


class TestAtomicWrite:
    def test_first_write_has_no_backup(self, tmp_path: Path) -> None:
        path = tmp_path / "annotations.json"
        server.write_json_atomic(path, {"a": 1})
        assert json.loads(path.read_text()) == {"a": 1}
        assert not (tmp_path / "annotations.json.bak").exists()

    def test_second_write_keeps_the_previous_version(self, tmp_path: Path) -> None:
        path = tmp_path / "annotations.json"
        server.write_json_atomic(path, {"v": 1})
        server.write_json_atomic(path, {"v": 2})
        assert json.loads(path.read_text()) == {"v": 2}
        assert json.loads((tmp_path / "annotations.json.bak").read_text()) == {"v": 1}
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "annotations.json",
            "annotations.json.bak",
        ]

    def test_write_replaces_the_file_by_rename(self, tmp_path: Path) -> None:
        path = tmp_path / "annotations.json"
        server.write_json_atomic(path, {"v": 1})
        before = path.stat().st_ino
        server.write_json_atomic(path, {"v": 2})
        assert path.stat().st_ino != before

    def test_failed_serialization_leaves_the_file_alone(self, tmp_path: Path) -> None:
        path = tmp_path / "annotations.json"
        server.write_json_atomic(path, {"v": 1})
        with pytest.raises(TypeError):
            server.write_json_atomic(path, {"v": object()})
        assert json.loads(path.read_text()) == {"v": 1}
        assert not list(tmp_path.glob("*.tmp"))

    def test_damaged_file_raises_instead_of_defaulting(self, tmp_path: Path) -> None:
        path = tmp_path / "annotations.json"
        path.write_text("{not json")
        with pytest.raises(json.JSONDecodeError):
            server.read_json(path, {})
        assert server.read_json(tmp_path / "missing.json", {"d": 1}) == {"d": 1}


class TestValidation:
    def test_valid_annotation_round_trips_unchanged(self) -> None:
        annotation = _annotation()
        assert server.validate_annotation(annotation) == annotation

    def test_defaults_fill_missing_fields(self) -> None:
        assert server.validate_annotation({}) == {
            "verdict": "unset",
            "trace_note": "",
            "spans": [],
            "updated": "",
        }

    @pytest.mark.parametrize(
        "bad",
        [
            {"verdict": "great"},
            {"trace_note": 3},
            {"spans": "no"},
            {"suggestion": "an agent's words"},
        ],
    )
    def test_bad_annotations_are_refused(self, bad: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            server.validate_annotation(bad)

    @pytest.mark.parametrize(
        "bad",
        [
            {"start": 5, "end": 5},
            {"start": -1},
            {"start": True},
            {"anchor": {"msg": 1, "field": "thinking"}},
            {"anchor": {"msg": "1", "field": "text"}},
            {"note": None},
            {"id": ""},
            {"extra": 1},
        ],
    )
    def test_bad_spans_are_refused(self, bad: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            server.validate_span(_span(**bad))

    def test_final_answer_anchor_has_no_message_index(self) -> None:
        span = _span(anchor={"msg": None, "field": "final_answer"})
        assert server.validate_span(span)["anchor"] == {
            "msg": None,
            "field": "final_answer",
        }

    def test_patterns_need_named_modes_and_note_refs(self) -> None:
        good = {
            "status": "draft",
            "failure_modes": [
                {
                    "name": "Trusts a success status",
                    "notes": [
                        {"trace_id": "q02", "span_id": "s-1"},
                        {"trace_id": "q02", "span_id": None},
                    ],
                }
            ],
        }
        assert server.validate_patterns(good) == good
        with pytest.raises(ValueError):
            server.validate_patterns({"failure_modes": [{"name": "", "notes": []}]})
        with pytest.raises(ValueError):
            server.validate_patterns(
                {"failure_modes": [{"name": "x", "notes": [{"span_id": "s"}]}]}
            )


class TestApplyAnnotation:
    def test_replaces_one_trace_and_keeps_the_rest(self) -> None:
        doc = {"version": 1, "traces": {"q02": _annotation(trace_note="old")}}
        body = {"trace_id": "q01", "annotation": _annotation(verdict="pass")}
        out = server.apply_annotation(doc, body, ["q01", "q02"])
        assert list(out["traces"]) == ["q01", "q02"]
        assert out["traces"]["q01"]["verdict"] == "pass"
        assert out["traces"]["q02"]["trace_note"] == "old"

    def test_unknown_trace_is_refused(self) -> None:
        with pytest.raises(ValueError, match="q99"):
            server.apply_annotation({}, {"trace_id": "q99", "annotation": {}}, ["q01"])


class TestReviewApp:
    def test_save_then_read_round_trip(self, tmp_path: Path) -> None:
        run = _run_dir(tmp_path, [_trace("q02"), _trace("q01")])
        app = server.ReviewApp(run)
        assert app.ids == ["q01", "q02"]
        assert app.annotations() == {"version": 1, "traces": {}}

        app.save_annotation({"trace_id": "q01", "annotation": _annotation()})
        app.save_annotation(
            {"trace_id": "q01", "annotation": _annotation(trace_note="second")}
        )

        assert app.annotations()["traces"]["q01"]["trace_note"] == "second"
        backup = json.loads((run / "annotations.json.bak").read_text())
        assert backup["traces"]["q01"]["trace_note"] == "my words"

    def test_refused_write_leaves_the_file_alone(self, tmp_path: Path) -> None:
        run = _run_dir(tmp_path, [_trace("q01")])
        app = server.ReviewApp(run)
        app.save_annotation({"trace_id": "q01", "annotation": _annotation()})
        before = (run / "annotations.json").read_text()
        with pytest.raises(ValueError):
            app.save_annotation({"trace_id": "q01", "annotation": {"verdict": "great"}})
        assert (run / "annotations.json").read_text() == before

    def test_patterns_absent_then_saved(self, tmp_path: Path) -> None:
        app = server.ReviewApp(_run_dir(tmp_path, [_trace("q01")]))
        assert app.patterns() is None
        draft = {"failure_modes": [{"name": "m", "notes": []}]}
        app.save_patterns(draft)
        assert app.patterns() == draft

    def test_samples_are_ordered_and_carry_stats(self, tmp_path: Path) -> None:
        app = server.ReviewApp(_run_dir(tmp_path, [_trace("q02"), _trace("q01")]))
        samples = app.samples()["samples"]
        assert [s["id"] for s in samples] == ["q01", "q02"]
        assert samples[0]["stats"]["max_tool_result_chars"] == len(
            '{"status": "success"}'
        )


class TestFeatures:
    def test_trace_features(self) -> None:
        trace = _trace(
            "q01",
            messages=[
                {"type": "tool_call", "result": "abc", "is_error": True},
                {"type": "tool_call", "result": '{"status": "error"}'},
                {"type": "tool_call", "result": None},
                {"type": "text", "text": "hi"},
            ],
            final_answer="12345",
            num_turns=4,
            notional_cost_usd=0.5,
        )
        assert server.trace_features(trace) == {
            "tool_calls": 3.0,
            "tool_errors": 2.0,
            "turns": 4.0,
            "tool_result_chars": float(3 + len('{"status": "error"}')),
            "final_answer_chars": 5.0,
            "cost_usd": 0.5,
        }

    def test_result_status(self) -> None:
        assert server.result_status('{"status": "success"}') == "success"
        assert server.result_status("plain text") is None
        assert server.result_status("[1, 2]") is None


class TestProjection:
    def test_shapes_and_variance(self) -> None:
        rng = np.random.default_rng(1)
        matrix = rng.uniform(0, 10, size=(20, 6))
        coords, explained, loadings = server.project_2d(matrix)
        assert coords.shape == (20, 2)
        assert loadings.shape == (2, 6)
        assert explained[0] >= explained[1] >= 0
        assert explained.sum() <= 1 + 1e-9
        assert np.allclose(np.linalg.norm(loadings, axis=1), 1)

    def test_constant_column_gives_finite_output(self) -> None:
        matrix = np.array(
            [[1.0, 5.0, 9.0], [2.0, 5.0, 1.0], [3.0, 5.0, 4.0], [8.0, 5.0, 2.0]]
        )
        coords, explained, loadings = server.project_2d(matrix)
        assert np.isfinite(coords).all() and np.isfinite(explained).all()
        assert np.allclose(loadings[:, 1], 0)

    def test_deterministic_and_sign_fixed(self) -> None:
        matrix = np.array([[0.0, 1.0], [1.0, 3.0], [4.0, 2.0], [9.0, 8.0]])
        first = server.project_2d(matrix)
        second = server.project_2d(matrix[::-1].copy())
        assert np.allclose(first[2], second[2])
        for row in first[2]:
            magnitudes = np.abs(row)
            lead = int(np.flatnonzero(magnitudes >= magnitudes.max() - 1e-9)[0])
            assert row[lead] > 0

    def test_single_row(self) -> None:
        coords, explained, _ = server.project_2d(np.ones((1, 6)))
        assert coords.shape == (1, 2)
        assert not explained.any()


class TestKmeans:
    def test_separates_obvious_groups(self) -> None:
        points = np.array(
            [[0, 0], [0.1, 0], [10, 10], [10.1, 10], [0, 10], [0.1, 10], [10, 0]],
            dtype=float,
        )
        labels = server.kmeans(points, k=4)
        assert len(set(labels.tolist())) == 4
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[4] == labels[5]

    def test_deterministic_and_ordered_by_x(self) -> None:
        points = np.random.default_rng(3).normal(size=(20, 2))
        labels = server.kmeans(points, k=4, seed=0)
        assert np.array_equal(labels, server.kmeans(points, k=4, seed=0))
        means = [points[labels == c, 0].mean() for c in range(4)]
        assert means == sorted(means)

    def test_fewer_points_than_clusters(self) -> None:
        assert server.kmeans(np.zeros((0, 2))).size == 0
        assert server.kmeans(np.array([[1.0, 2.0], [3.0, 4.0]]), k=4).tolist() == [
            0,
            1,
        ]

    def test_build_graph(self) -> None:
        traces = [
            _trace(f"q{i:02d}", num_turns=i, notional_cost_usd=0.01 * i)
            for i in range(1, 9)
        ]
        graph = server.build_graph(traces)
        assert graph["features"] == list(server.FEATURE_NAMES)
        assert [p["id"] for p in graph["points"]] == [t["case"]["id"] for t in traces]
        assert graph["clusters"] == 4
        assert {p["cluster"] for p in graph["points"]} == {0, 1, 2, 3}
        assert len(graph["loadings"]) == 2


class TestContentLength:
    @pytest.mark.parametrize("value", ["abc", "-1"])
    def test_bad_content_length_gets_a_400(self, tmp_path: Path, value: str) -> None:
        httpd = server.make_server(
            server.ReviewApp(_run_dir(tmp_path, [_trace("q01")])), port=0
        )
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            port = httpd.server_address[1]
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.putrequest("POST", "/api/annotations", skip_host=True)
            conn.putheader("Host", f"127.0.0.1:{port}")
            conn.putheader("Content-Type", "application/json")
            conn.putheader("Content-Length", value)
            conn.endheaders()
            response = conn.getresponse()
            assert response.status == 400
            assert json.loads(response.read()) == {"error": "bad Content-Length"}
            conn.close()
        finally:
            httpd.shutdown()
            httpd.server_close()
