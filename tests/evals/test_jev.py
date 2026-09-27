"""The request `evals/tool_surface/jev.py` sends to Jev, without the network."""

import json

import httpx

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
