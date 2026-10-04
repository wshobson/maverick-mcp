"""Loopback SearXNG and OpenAI wire fixtures, never a paid provider.

The production CLI, research graph, HTTP stack, and LLM SDK are unmodified.
Only the configured remote endpoints return synthetic responses. URLs identify
synthetic documents and are never fetched. Request captures omit HTTP headers.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


class Provider(ThreadingHTTPServer):
    mode = "success"
    requests: list[dict]
    output: Path

    def capture(self, data):
        self.requests.append(data)
        with self.output.open("a") as stream:
            stream.write(json.dumps(data) + "\n")


class Handler(BaseHTTPRequestHandler):
    server: Provider

    def reply(self, payload, status=200):
        encoded = json.dumps(payload, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        self.server.capture({"method": "GET", "path": parsed.path, "query": query})
        if self.server.mode == "search_failure":
            self.reply({"error": "Synthetic search provider denied JSON"}, 403)
            return
        results = (
            []
            if self.server.mode == "empty"
            else [
                {
                    "url": "https://example.org/synthetic-financial-report",
                    "title": "Synthetic financial report for process testing",
                    "content": "Synthetic financial research fixture. Revenue and earnings growth with cash flow and risk disclosures. "
                    * 12,
                    "publishedDate": datetime.now(UTC).date().isoformat(),
                    "score": 0.9,
                }
            ]
        )
        self.reply({"results": results})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.capture({"method": "POST", "path": self.path, "body": body})
        if self.server.mode == "model_failure":
            self.reply(
                {
                    "error": {
                        "message": "Synthetic invalid model credentials",
                        "type": "authentication_error",
                    }
                },
                401,
            )
            return
        messages = str(body.get("messages", []))
        if "Convert this trading strategy" in messages:
            content = json.dumps(
                {
                    "strategy_type": "sma_cross",
                    "parameters": {"fast_period": 10, "slow_period": 20},
                }
            )
        elif "KEY_INSIGHTS" in messages:
            score = (
                {"score": 0.9}
                if self.server.mode == "object_scores"
                else {"invalid": []}
                if self.server.mode == "malformed_scores"
                else 0.9
            )
            content = json.dumps(
                {
                    "KEY_INSIGHTS": ["Synthetic earnings observation"],
                    "SENTIMENT": {"direction": "neutral", "confidence": score},
                    "RISK_FACTORS": ["Synthetic risk"],
                    "OPPORTUNITIES": ["Synthetic opportunity"],
                    "CREDIBILITY": score,
                    "RELEVANCE": score,
                    "SUMMARY": "Synthetic source analysis for a deterministic transport test.",
                }
            )
        else:
            content = "Synthetic research synthesis preserves the cited fixture and uncertainty."
        self.reply(
            {
                "id": "chatcmpl-synthetic",
                "object": "chat.completion",
                "created": 0,
                "model": body.get("model", "fixture-model"),
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        )

    def log_message(self, format, *args):
        pass


@contextmanager
def provider_server(output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    server = Provider(("127.0.0.1", 0), Handler)
    server.requests = []
    server.output = output
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
