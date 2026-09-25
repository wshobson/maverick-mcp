"""Tests for maverick.platform.llm against the real `langchain_openai` classes.

`test_llm.py` stubs `langchain_openai`, so it cannot catch a breaking
`openai` or `langchain-openai` release. These tests build the real
`ChatOpenAI` through `get_llm()` and send one request to an in-process
loopback server, so the default HTTP client path runs end to end without
any external network. Skipped when the `research` extra is not installed.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from maverick.platform.llm import get_llm, reset_llm_settings

langchain_openai = pytest.importorskip("langchain_openai")

_ENV_VARS = (
    "LLM_PROVIDER",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_TEMPERATURE",
    # ChatOpenAI falls back to these when base_url is None; a developer's
    # shell must not redirect the tests.
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    # The loopback request must not be routed through a proxy.
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


@pytest.fixture(autouse=True)
def _fresh_settings(monkeypatch):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    reset_llm_settings()
    yield
    reset_llm_settings()


class _ChatCompletionsHandler(BaseHTTPRequestHandler):
    """Answer every POST with a minimal chat completion and record the call."""

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((self.path, body["model"]))
        payload = json.dumps(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "pong"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
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
def chat_server():
    server = HTTPServer(("127.0.0.1", 0), _ChatCompletionsHandler)
    server.requests = []
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


@pytest.mark.parametrize(
    ("provider", "base_url", "expected_base_url"),
    [
        ("openai", None, None),
        ("openrouter", None, "https://openrouter.ai/api/v1"),
        ("openai_compatible", "http://localhost:8080/v1", "http://localhost:8080/v1"),
    ],
)
def test_get_llm_builds_the_real_chat_openai(
    monkeypatch, provider, base_url, expected_base_url
):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("LLM_API_KEY", "key-123")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    if base_url is not None:
        monkeypatch.setenv("LLM_BASE_URL", base_url)

    llm = get_llm()

    assert isinstance(llm, langchain_openai.ChatOpenAI)
    assert llm.model_name == "test-model"
    assert llm.openai_api_base == expected_base_url


async def test_get_llm_chat_openai_round_trips_one_request(monkeypatch, chat_server):
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_API_KEY", "key-123")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_BASE_URL", f"http://127.0.0.1:{chat_server.server_port}/v1")

    reply = await get_llm().ainvoke("ping")

    assert reply.content == "pong"
    assert chat_server.requests == [("/v1/chat/completions", "test-model")]
