"""The request `evals/tool_surface/jev.py` sends to Jev, without the network."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from evals.tool_surface import jev


def test_ask_sends_one_noul_question_about_the_state() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.content)
        answer = {"acts_on_a_guess": {"noul": 0.8}}
        return httpx.Response(200, json={"model": "jev-x", "answers": answer})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        body = jev.ask(client, "k", "USER REQUEST: hi")
    assert body["answers"]["acts_on_a_guess"]["noul"] == 0.8
    assert seen["auth"] == "Bearer k"
    sent = seen["body"]
    assert isinstance(sent, dict)
    assert sent["model"] == jev.MODEL
    assert sent["state"] == "USER REQUEST: hi"
    assert sent["questions"]["acts_on_a_guess"]["type"] == "noul"


def _inputs(tmp_path: Path, texts: dict[str, str]) -> Path:
    folder = tmp_path / "inputs"
    folder.mkdir()
    for name, text in texts.items():
        (folder / f"{name}.txt").write_text(text)
    return folder


def _fake_ask(noul: float, tokens: int, fail_on: str | None = None) -> Any:
    def ask(client: httpx.Client, key: str, state: str) -> dict[str, Any]:
        if fail_on and fail_on in state:
            raise httpx.HTTPError("boom")
        if "not json" in state:
            raise ValueError("Expecting value")
        answer = {"acts_on_a_guess": {"noul": noul}}
        return {"model": "jev-x", "answers": answer, "usage": {"input_tokens": tokens}}

    return ask


def _run(monkeypatch: pytest.MonkeyPatch, inputs: Path, out: Path, ask: Any) -> None:
    monkeypatch.setattr(jev, "dotenv_values", lambda _path: {"TYPESAFE_API_KEY": "k"})
    monkeypatch.setattr(jev, "ask", ask)
    assert jev.main(["--inputs", str(inputs), "--out", str(out)]) == 0


def test_a_failed_request_is_recorded_and_the_rest_are_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(
        tmp_path, {"a": "fine", "b": "bad one", "c": "fine", "d": "not json"}
    )
    out = tmp_path / "out.json"
    _run(monkeypatch, inputs, out, _fake_ask(0.9, 10, fail_on="bad"))
    doc = json.loads(out.read_text())
    assert sorted(doc["judgments"]) == ["a", "c"]
    assert doc["judgments"]["a"]["result"] == "Fail"
    assert sorted(doc["errors"]) == ["b", "d"]


def test_a_rerun_skips_traces_already_judged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path, {"a": "x", "b": "y"})
    out = tmp_path / "out.json"
    _run(monkeypatch, inputs, out, _fake_ask(0.1, 10, fail_on="y"))
    _run(monkeypatch, inputs, out, _fake_ask(0.9, 10))
    doc = json.loads(out.read_text())
    assert doc["judgments"]["a"]["result"] == "Pass"  # kept from the first run
    assert doc["judgments"]["b"]["result"] == "Fail"
    assert doc["errors"] == {}
    assert doc["input_tokens"] == 20


def test_no_request_is_sent_that_could_pass_the_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(jev, "TOKEN_CAP", 1_000)
    inputs = _inputs(tmp_path, {"a": "x" * 100, "b": "x" * 10_000})
    out = tmp_path / "out.json"
    _run(monkeypatch, inputs, out, _fake_ask(0.1, 50))
    doc = json.loads(out.read_text())
    assert sorted(doc["judgments"]) == ["a"]
