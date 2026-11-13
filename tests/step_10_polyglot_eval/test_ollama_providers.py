"""S2's local pair — `index.ollama` — against a stub server on a real socket.

Decision 0003. The real models are never called from the suite: it must not
need a GPU, and a test that passes or fails on what a model felt like saying is
not a test. What is pinned is everything around the call — and the two rules
that guard silent failures: zero-padding to the index's dimensions, and the
prefixes the embedder was trained with.
"""

from __future__ import annotations

import math
import socket

import pytest

import metrics
from adapters.base import Symbol
from api.app import InvestigationService
from api.prompt import ANSWER_FORMAT_REMINDER
from config import EMBED_DIM, LOCAL_LLM_CONTEXT_TOKENS
from index.cache import EmbeddingUnavailable
from index.ollama import OllamaEmbeddingProvider, OllamaLLM, has_model, installed_models
from index.providers import ApproxCodeTokenizer
from pack.render import Block
from tests.ollama_stub import OllamaStub, bag_of_words


@pytest.fixture(autouse=True)
def reset():
    metrics.reset()
    yield
    metrics.reset()


@pytest.fixture
def stub():
    server = OllamaStub()
    yield server
    server.close()


def closed_port_url() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


# -- discovery ------------------------------------------------------------------


def test_installed_models_lists_what_the_server_holds(stub):
    assert installed_models(stub.url) == {"qwen2.5-coder:3b", "nomic-embed-text:latest"}


def test_no_server_is_none_not_an_exception():
    """Start-up asks this to decide whether a local path exists at all."""
    assert installed_models(closed_port_url(), timeout=0.5) is None


@pytest.mark.parametrize(
    "name, expected",
    [("qwen2.5-coder:3b", True), ("nomic-embed-text", True),
     ("nomic-embed-text:latest", True), ("qwen2.5-coder:7b", False), ("llama3", False)],
)
def test_has_model_matches_with_and_without_a_tag(stub, name, expected):
    assert has_model(installed_models(stub.url), name) is expected
    assert has_model(None, name) is False


# -- embeddings -----------------------------------------------------------------


def test_vectors_are_padded_to_the_index_dimensions(stub):
    """Neo4j does not reject a vector of the wrong length: it leaves the node
    out of the index without a word. Padding is what keeps it in."""
    provider = OllamaEmbeddingProvider(base_url=stub.url)
    vectors = provider.embed(["def charge_card(): pass", "class LoginView: pass"])

    assert [len(v) for v in vectors] == [EMBED_DIM, EMBED_DIM]
    assert provider.native_dimensions == 8
    assert provider.dimensions == EMBED_DIM
    assert all(v == 0.0 for v in vectors[0][8:])


def test_zero_padding_leaves_cosine_similarity_unchanged(stub):
    provider = OllamaEmbeddingProvider(base_url=stub.url)
    texts = ["charge the card for the user", "charge a card", "parse the date string"]
    padded = provider.embed(texts)
    prefix = "search_document: "
    native = [bag_of_words(prefix + t, 8) for t in texts]

    for i in range(3):
        for j in range(3):
            assert cosine(padded[i], padded[j]) == pytest.approx(cosine(native[i], native[j]))


def test_documents_and_queries_get_their_task_prefixes(stub):
    provider = OllamaEmbeddingProvider(base_url=stub.url)
    provider.embed(["def f(): pass"])
    provider.embed_query("what does f do")

    documents, query = stub.posts("/api/embed")
    assert documents["input"] == ["search_document: def f(): pass"]
    assert query["input"] == ["search_query: what does f do"]
    assert documents["model"] == "nomic-embed-text"


def test_an_unknown_embedder_gets_no_prefix(stub):
    provider = OllamaEmbeddingProvider("some-other-embedder", base_url=stub.url)
    provider.embed(["x"])
    assert stub.posts("/api/embed")[0]["input"] == ["x"]
    assert provider.name == "ollama:some-other-embedder"


def test_embedding_is_batched_and_order_is_kept(stub):
    provider = OllamaEmbeddingProvider(base_url=stub.url, batch_size=2)
    texts = [f"symbol number {i}" for i in range(5)]
    vectors = provider.embed(texts)

    assert [len(body["input"]) for body in stub.posts("/api/embed")] == [2, 2, 1]
    expected = [bag_of_words("search_document: " + t, 8) for t in texts]
    assert [v[:8] for v in vectors] == expected


def test_an_empty_chunk_is_sent_as_a_space(stub):
    OllamaEmbeddingProvider(base_url=stub.url).embed(["", "x"])
    assert stub.posts("/api/embed")[0]["input"] == ["search_document:  ", "search_document: x"]


def test_no_texts_no_request(stub):
    assert OllamaEmbeddingProvider(base_url=stub.url).embed([]) == []
    assert stub.posts("/api/embed") == []


def test_a_server_that_is_down_degrades_rather_than_crashes():
    provider = OllamaEmbeddingProvider(base_url=closed_port_url(), timeout=1.0)
    with pytest.raises(EmbeddingUnavailable):
        provider.embed(["x"])


@pytest.mark.parametrize("payload", [{}, {"embeddings": None}, {"embeddings": [["a"]]},
                                     {"embeddings": []}, {"error": "model not found"}])
def test_a_malformed_reply_is_an_outage(payload):
    server = OllamaStub(embed_payload=payload)
    try:
        with pytest.raises(EmbeddingUnavailable):
            OllamaEmbeddingProvider(base_url=server.url).embed(["x"])
    finally:
        server.close()


def test_a_model_wider_than_the_index_is_a_configuration_error():
    """Truncating would change every similarity; §12.11 makes this a decision."""
    server = OllamaStub(dims=EMBED_DIM + 1)
    try:
        with pytest.raises(ValueError, match="dims"):
            OllamaEmbeddingProvider(base_url=server.url).embed(["x"])
    finally:
        server.close()


def test_the_local_embedder_declares_what_it_is(stub):
    provider = OllamaEmbeddingProvider(base_url=stub.url)
    assert provider.semantic is True and provider.local is True


# -- synthesis ------------------------------------------------------------------


def test_the_answer_streams_in_order_and_usage_is_the_models_own(stub):
    llm = OllamaLLM(base_url=stub.url)
    usage: dict = {}
    chunks = list(llm.stream("system text", "the prompt", max_tokens=200, usage=usage))

    assert len(chunks) > 1, "the reply must arrive as a stream, not one piece"
    assert "".join(chunks) == stub.answer
    assert usage == {"input_tokens": 123, "output_tokens": max(1, len(stub.answer) // 4)}


def test_the_request_carries_temperature_zero_and_the_window(stub):
    list(OllamaLLM(base_url=stub.url).stream("system text", "the prompt", max_tokens=200))
    body = stub.posts("/api/chat")[0]

    assert body["model"] == "qwen2.5-coder:3b"
    assert body["stream"] is True
    assert body["options"] == {"num_ctx": LOCAL_LLM_CONTEXT_TOKENS, "num_predict": 200,
                               "temperature": 0.0}
    assert body["messages"] == [{"role": "system", "content": "system text"},
                                {"role": "user", "content": "the prompt"}]
    assert body["keep_alive"]


def test_temperature_is_omitted_when_switched_off(stub, monkeypatch):
    monkeypatch.setattr("index.ollama.LLM_TEMPERATURE", None)
    list(OllamaLLM(base_url=stub.url).stream("s", "p", max_tokens=10))
    assert "temperature" not in stub.posts("/api/chat")[0]["options"]


def test_an_error_line_in_the_stream_raises():
    server = OllamaStub(error="model requires more system memory")
    try:
        with pytest.raises(RuntimeError, match="more system memory"):
            list(OllamaLLM(base_url=server.url).stream("s", "p", max_tokens=10))
    finally:
        server.close()


def test_a_prompt_too_large_for_the_window_is_counted():
    """Past `num_ctx` the server drops the *start* of the prompt and answers
    anyway. The model's own count is the only witness, so it is recorded."""
    server = OllamaStub(prompt_tokens=LOCAL_LLM_CONTEXT_TOKENS - 100)
    try:
        list(OllamaLLM(base_url=server.url).stream("s", "p", max_tokens=2000))
        assert metrics.get("llm.local.context_tight") == 1
    finally:
        server.close()


def test_a_prompt_that_fits_is_not_counted(stub):
    list(OllamaLLM(base_url=stub.url).stream("s", "p", max_tokens=2000))
    assert metrics.get("llm.local.context_tight") == 0


def test_warm_loads_the_model_without_generating(stub):
    OllamaLLM(base_url=stub.url).warm()
    body = stub.posts("/api/chat")[0]
    assert body["messages"] == [] and body["model"] == "qwen2.5-coder:3b"


def test_count_is_the_stand_in_tokenizer(stub):
    text = "def charge_card(user, amount):\n    return gateway.charge(user, amount)\n"
    assert OllamaLLM(base_url=stub.url).count(text) == ApproxCodeTokenizer().count(text)


# -- through the answer pipeline ------------------------------------------------


def _symbol(name: str) -> Symbol:
    return Symbol(
        uid=f"uid_{name}", repo_id="r", rel_path=f"billing/{name}.py",
        qualified_name=f"billing.{name}", name=name, arity=1, ordinal=0, kind="function",
        signature=f"def {name}(user):", docstring=None,
        source_code=f"def {name}(user):\n    return bill(user)\n", enclosing_signature=None,
    )


def _service(llm) -> InvestigationService:
    return InvestigationService(
        retriever=lambda q, r: ([Block(_symbol("charge_card"), 1.0)], []),
        llm=llm, tokenizer=ApproxCodeTokenizer(), repo_id="r",
    )


def test_a_local_answer_is_cited_graded_and_free(stub):
    result = _service(OllamaLLM(base_url=stub.url)).answer("what bills the card")

    assert result.text == stub.answer
    assert result.report.coverage == 1.0
    assert result.cost.usd == 0.0
    assert result.cost.model == "ollama:qwen2.5-coder:3b"
    assert result.cost.input_tokens == 123, "the model's own count, not the stand-in's"


def test_the_answer_format_is_restated_after_the_context(stub):
    """F-026: a 3B model cited 22% of its claims from the system prompt alone,
    and 91% with the format restated last. The example must be foreign: one
    drawn from the codebase was copied into answers about something else."""
    _service(OllamaLLM(base_url=stub.url)).answer("what bills the card")
    prompt = stub.posts("/api/chat")[0]["messages"][1]["content"]

    assert prompt.endswith(ANSWER_FORMAT_REMINDER)
    assert prompt.index("# Context") < prompt.index("# How to answer")
    assert "resize" in ANSWER_FORMAT_REMINDER and "charge_card" not in ANSWER_FORMAT_REMINDER


def test_a_dead_model_server_degrades_to_retrieval_only():
    llm = OllamaLLM(base_url=closed_port_url(), timeout=1.0)
    events = [e for e, _r in _service(llm).stream("what bills the card")]
    kinds = [e["type"] for e in events if isinstance(e, dict)]

    assert "synthesis_unavailable" in kinds and kinds[-1] == "done"
    assert metrics.get("synthesis.unavailable") == 1
