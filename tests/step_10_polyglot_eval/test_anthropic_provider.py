"""`AnthropicProvider` against the real installed SDK and a local mock server.

The first version of this test drove the provider with a fake client, and the
fake accepted any keyword — so it passed while the real call failed. Run through
the server, the installed `anthropic` 1.2.0 raised before any network I/O:

    TypeError: Messages.stream() got an unexpected keyword argument 'temperature'

The 1.x SDK removed `temperature` from its signatures (not from the API), and
the provider's fallback turned that into "Synthesis unavailable" on every
question. So this module uses the real client, pointed at a local HTTP server
that speaks the Messages streaming protocol: an SDK signature change fails here,
and no request leaves the machine.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from index.providers import AnthropicProvider

EVENTS = [
    ("message_start", {"type": "message_start", "message": {
        "id": "msg_test", "type": "message", "role": "assistant",
        "model": "claude-haiku-4-5", "content": [], "stop_reason": None,
        "stop_sequence": None, "usage": {"input_tokens": 321, "output_tokens": 1}}}),
    ("content_block_start", {"type": "content_block_start", "index": 0,
                             "content_block": {"type": "text", "text": ""}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 0,
                             "delta": {"type": "text_delta", "text": "Hello"}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 0,
                             "delta": {"type": "text_delta", "text": " there."}}),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    ("message_delta", {"type": "message_delta",
                       "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                       "usage": {"output_tokens": 12}}),
    ("message_stop", {"type": "message_stop"}),
]


class MessagesServer:
    def __init__(self):
        self.requests: list[dict] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                length = int(self.headers.get("content-length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                server.requests.append({"path": self.path, "body": body})
                if self.path.endswith("/count_tokens"):
                    payload = json.dumps({"input_tokens": 42}).encode()
                    self.send_response(200)
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                for name, data in EVENTS:
                    self.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())
                self.wfile.flush()

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def server():
    srv = MessagesServer()
    yield srv
    srv.close()


@pytest.fixture
def provider(server):
    import anthropic

    client = anthropic.Anthropic(api_key="test-key", base_url=server.url, max_retries=0)
    return AnthropicProvider(client=client)


def test_streams_text_through_the_real_sdk(provider, server):
    assert "".join(provider.stream("system text", "the prompt")) == "Hello there."
    assert server.requests[0]["path"].endswith("/v1/messages")


def test_the_request_runs_at_temperature_zero(provider, server):
    """§8.2: "Eval runs at temperature 0." Carried in the request body."""
    list(provider.stream("system text", "the prompt"))
    body = server.requests[0]["body"]
    assert body["temperature"] == 0.0
    assert body["model"] == "claude-haiku-4-5"
    assert body["system"] == "system text"
    assert body["messages"] == [{"role": "user", "content": "the prompt"}]
    assert body["stream"] is True


def test_usage_is_reported_from_the_final_message(provider):
    usage: dict = {}
    list(provider.stream("s", "p", usage=usage))
    assert usage == {"input_tokens": 321, "output_tokens": 12}


def test_no_temperature_is_sent_when_it_is_switched_off(provider, server, monkeypatch):
    """Newer models reject sampling values; `LLM_TEMPERATURE = None` omits it."""
    monkeypatch.setattr("index.providers.LLM_TEMPERATURE", None)
    list(provider.stream("s", "p"))
    assert "temperature" not in server.requests[0]["body"]


def test_count_uses_the_token_counting_endpoint(provider, server):
    assert provider.count("some text") == 42
    assert server.requests[0]["path"].endswith("/v1/messages/count_tokens")
