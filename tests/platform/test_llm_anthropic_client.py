"""Real Anthropic SDK request serialization against a loopback-only server."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import cast

import pytest

from maverick.platform.llm import get_llm, reset_llm_settings

pytest.importorskip("langchain_anthropic")


class _RecordingServer(HTTPServer):
    requests: list[tuple[str, dict]]


class _MessagesHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        cast(_RecordingServer, self.server).requests.append((self.path, body))
        payload = json.dumps(
            {
                "id": "msg-test",
                "type": "message",
                "role": "assistant",
                "model": body["model"],
                "content": [{"type": "text", "text": "pong"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


@pytest.fixture
def messages_server(monkeypatch):
    for name in (
        "LLM_PROVIDER",
        "LLM_API_KEY",
        "LLM_MODEL",
        "LLM_BASE_URL",
        "LLM_TEMPERATURE",
        "ANTHROPIC_BASE_URL",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    reset_llm_settings()
    server = _RecordingServer(("127.0.0.1", 0), _MessagesHandler)
    server.requests = []
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    reset_llm_settings()


@pytest.mark.parametrize(
    "model", ["claude-sonnet-4-6", "claude-sonnet-5", "claude-opus-4-7"]
)
@pytest.mark.parametrize("temperature", [None, "1.0"])
async def test_anthropic_serialized_temperature(
    monkeypatch, messages_server, model, temperature
):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_API_KEY", "offline-test-key")
    monkeypatch.setenv("LLM_MODEL", model)
    monkeypatch.setenv(
        "LLM_BASE_URL", f"http://127.0.0.1:{messages_server.server_port}"
    )
    if temperature is not None:
        monkeypatch.setenv("LLM_TEMPERATURE", temperature)
    reply = await get_llm().ainvoke("ping")
    assert reply.content == "pong"
    path, body = messages_server.requests[0]
    assert len(messages_server.requests) == 1
    assert path == "/v1/messages"
    assert body["model"] == model
    if temperature is None:
        assert "temperature" not in body
    else:
        assert body["temperature"] == 1.0
