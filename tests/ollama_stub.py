"""A local stand-in for an Ollama server, for tests.

The suite must not depend on a model being installed, on a GPU, or on what a
model happens to say. This speaks the three endpoints `index.ollama` uses —
`/api/tags`, `/api/embed`, `/api/chat` — records every request, and answers
from a script. It is a real HTTP server on a real socket, so the provider's
actual wire handling (streaming NDJSON, batching, timeouts) is what runs.
"""

from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def bag_of_words(text: str, dims: int) -> list[float]:
    """A deterministic embedding with *some* meaning: shared words, closer vectors.

    Enough structure for a test to tell a vector arm that works from one that
    returns noise — which `HashEmbeddingProvider`, by design, cannot.
    """
    vec = [0.0] * dims
    for word in "".join(c.lower() if c.isalnum() else " " for c in text).split():
        vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % dims] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


class OllamaStub:
    def __init__(
        self,
        *,
        models: tuple[str, ...] = ("qwen2.5-coder:3b", "nomic-embed-text:latest"),
        dims: int = 8,
        answer: str = "The `charge_card` function bills the card [C1].",
        chunk: int = 6,
        prompt_tokens: int = 123,
        error: str | None = None,
        embed_payload: dict | None = None,
    ) -> None:
        self.models = models
        self.dims = dims
        self.answer = answer
        self.chunk = chunk
        self.prompt_tokens = prompt_tokens
        self.error = error
        self.embed_payload = embed_payload
        self.requests: list[dict] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _json(self, payload: dict, status: int = 200) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                stub.requests.append({"path": self.path, "body": None})
                if self.path == "/api/tags":
                    self._json({"models": [{"name": name} for name in stub.models]})
                else:
                    self._json({"error": "not found"}, 404)

            def do_POST(self):
                length = int(self.headers.get("content-length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                stub.requests.append({"path": self.path, "body": body})
                if self.path == "/api/embed":
                    if stub.embed_payload is not None:
                        self._json(stub.embed_payload)
                        return
                    self._json({"embeddings": [bag_of_words(t, stub.dims) for t in body["input"]]})
                elif self.path == "/api/chat":
                    self.send_response(200)
                    self.send_header("content-type", "application/x-ndjson")
                    self.end_headers()
                    if stub.error:
                        self.wfile.write((json.dumps({"error": stub.error}) + "\n").encode())
                        return
                    text = stub.answer if body.get("messages") else ""
                    for start in range(0, len(text), stub.chunk):
                        piece = {"message": {"role": "assistant",
                                             "content": text[start : start + stub.chunk]},
                                 "done": False}
                        self.wfile.write((json.dumps(piece) + "\n").encode())
                        self.wfile.flush()
                    done = {"message": {"role": "assistant", "content": ""}, "done": True,
                            "done_reason": "stop", "prompt_eval_count": stub.prompt_tokens,
                            "eval_count": max(1, len(text) // 4)}
                    self.wfile.write((json.dumps(done) + "\n").encode())
                else:
                    self._json({"error": "not found"}, 404)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    def posts(self, path: str) -> list[dict]:
        return [r["body"] for r in self.requests if r["path"] == path and r["body"] is not None]

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
