"""S2's production providers, against stub transports.

The real APIs are never called from the suite — that would send source code to
a third party (decision 0001). What is tested is everything around the call:
batching, ordering, retry policy, typed failure, and that a configuration error
is not mistaken for an outage.
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from index.cache import EmbeddingUnavailable
from index.providers import AnthropicProvider, VoyageEmbeddingProvider


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class StubOpener:
    """Plays back a script of responses/exceptions and records each request."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    def __call__(self, request, timeout):
        body = json.loads(request.data.decode())
        self.requests.append({"body": body, "headers": dict(request.header_items()),
                              "timeout": timeout})
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        if callable(step):
            step = step(body)
        return Response(json.dumps(step).encode())


def ok(dims=4):
    def reply(body):
        n = len(body["input"])
        # Deliberately out of order: the provider must sort by `index`.
        return {"data": [{"index": i, "embedding": [float(i)] * dims}
                         for i in reversed(range(n))]}
    return reply


def http_error(code):
    return urllib.error.HTTPError("https://api.voyageai.com", code, "err", {}, None)


def provider(opener, **kw):
    kw.setdefault("dimensions", 4)
    return VoyageEmbeddingProvider(api_key="test-key", opener=opener, retries=2, **kw)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("index.providers.time.sleep", lambda s: None)


def test_batches_and_preserves_input_order():
    opener = StubOpener([ok(), ok(), ok()])
    vectors = provider(opener, batch_size=2).embed(["a", "b", "c", "d", "e"])

    assert [len(r["body"]["input"]) for r in opener.requests] == [2, 2, 1]
    assert vectors == [[0.0] * 4, [1.0] * 4, [0.0] * 4, [1.0] * 4, [0.0] * 4]


def test_request_shape():
    opener = StubOpener([ok()])
    provider(opener).embed(["def f(): pass"])
    request = opener.requests[0]
    assert request["body"]["model"] == "voyage-code-2"
    assert request["headers"]["Authorization"] == "Bearer test-key"
    assert request["timeout"] > 0


def test_an_empty_chunk_is_sent_as_a_space():
    """The API rejects empty strings; an empty chunk embeds as 'nothing here'."""
    opener = StubOpener([ok()])
    provider(opener).embed(["", "x"])
    assert opener.requests[0]["body"]["input"] == [" ", "x"]


def test_no_texts_no_request():
    opener = StubOpener([])
    assert provider(opener).embed([]) == []
    assert opener.requests == []


@pytest.mark.parametrize("transient", [http_error(429), http_error(503),
                                       urllib.error.URLError("dns"), TimeoutError()])
def test_transient_failures_are_retried(transient):
    opener = StubOpener([transient, ok()])
    assert len(provider(opener).embed(["x"])) == 1
    assert len(opener.requests) == 2


def test_a_bad_request_is_not_retried():
    opener = StubOpener([http_error(400), ok()])
    with pytest.raises(EmbeddingUnavailable):
        provider(opener).embed(["x"])
    assert len(opener.requests) == 1


def test_exhausted_retries_raise_the_typed_error():
    opener = StubOpener([http_error(500)] * 3)
    with pytest.raises(EmbeddingUnavailable):
        provider(opener).embed(["x"])
    assert len(opener.requests) == 3


@pytest.mark.parametrize("payload", [{}, {"data": None}, {"data": [{"index": 0}]},
                                     {"data": []}])
def test_a_malformed_reply_is_an_outage_not_a_crash(payload):
    opener = StubOpener([payload] * 3)
    with pytest.raises(EmbeddingUnavailable):
        provider(opener).embed(["x"])


def test_a_dimension_mismatch_is_a_configuration_error():
    """§12.11: a model change is a full re-index. Degrading would index every
    symbol without a vector, forever, under a "degraded" banner."""
    opener = StubOpener([ok(dims=8)])
    with pytest.raises(ValueError, match="dims"):
        provider(opener, dimensions=4).embed(["x"])


def test_a_missing_key_fails_at_construction(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
        VoyageEmbeddingProvider()
