"""S2 on this machine: an Ollama server for synthesis and for embeddings.

Decision 0003. The hosted pair decision 0002 names sends full function bodies
to a third party (§12.0), costs money per question, and needs two API keys. A
local model does none of that, and is weaker — which is a trade the owner of the
code gets to make, so both pairs sit behind the same seam and `api/cli.py`
picks by what is configured.

Spoken to over Ollama's HTTP API with the standard library, as the Voyage
provider is: the local path adds no dependency either.

Two things here are not obvious and both guard a silent failure:

* **Embeddings are zero-padded to `EMBED_DIM`.** §11.3's vector index is
  declared at 1,536 dimensions and Neo4j does not *reject* a vector of another
  length — it leaves the node out of the index without a word, which is T5's
  failure by a new route. Padding with zeros changes neither a dot product nor
  a norm, so cosine similarity between padded vectors is exactly what it was
  between the originals, and the frozen DDL stays as written (FINDINGS F-025).
* **Documents and queries get the prefixes the model was trained with.** A
  model asked to embed a query as if it were a document still returns a vector.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Iterator, Sequence

import metrics
from config import (
    EMBED_DIM,
    LLM_TEMPERATURE,
    LOCAL_EMBED_BATCH_SIZE,
    LOCAL_EMBED_MODEL,
    LOCAL_EMBED_PREFIXES,
    LOCAL_KEEP_ALIVE,
    LOCAL_LLM_CONTEXT_TOKENS,
    LOCAL_LLM_MODEL,
    LOCAL_TIMEOUT_SECONDS,
    OLLAMA_URL,
    RESERVED_OUT,
)
from api.prompt import ANSWER_FORMAT_REMINDER
from index.cache import EmbeddingUnavailable
from index.providers import ApproxCodeTokenizer

Vector = Sequence[float]

_NETWORK_ERRORS = (urllib.error.URLError, OSError, TimeoutError)


def _request(url: str, payload: dict | None) -> urllib.request.Request:
    data = None if payload is None else json.dumps(payload).encode()
    return urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )


def installed_models(base_url: str = OLLAMA_URL, *, timeout: float = 2.0, opener=None) -> set[str] | None:
    """Names of the models the server holds, or None when it does not answer.

    The short timeout is the point: this decides at start-up whether the local
    path exists at all, and a server that is not there must not cost the user
    the hosted default's ten-second connect timeout to find out.
    """
    opener = opener or urllib.request.urlopen
    try:
        with opener(_request(f"{base_url}/api/tags", None), timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return {entry["name"] for entry in payload.get("models", [])}
    except (*_NETWORK_ERRORS, ValueError, KeyError, TypeError):
        return None


def has_model(models: set[str] | None, name: str) -> bool:
    """`qwen2.5-coder:3b` matches itself; `nomic-embed-text` matches `…:latest`."""
    if not models:
        return False
    wanted = name if ":" in name else f"{name}:latest"
    return wanted in models or name in models


def _family(model: str) -> str:
    return model.split(":", 1)[0].rsplit("/", 1)[-1]


class OllamaEmbeddingProvider:
    """`nomic-embed-text` (or another local embedder), padded to `EMBED_DIM`.

    `name` carries the model, so the embedding cache — keyed on it — never hands
    one model's vector to another, and `api/cli.py` can tell when a graph was
    indexed with a different embedder than the one now answering queries.
    """

    semantic = True
    #: Runs on this machine: re-embedding a repository costs time, not money.
    local = True

    def __init__(
        self,
        model: str = LOCAL_EMBED_MODEL,
        *,
        base_url: str = OLLAMA_URL,
        dimensions: int = EMBED_DIM,
        batch_size: int = LOCAL_EMBED_BATCH_SIZE,
        timeout: float = LOCAL_TIMEOUT_SECONDS,
        opener=None,
    ) -> None:
        self.model = model
        self.name = f"ollama:{model}"
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.timeout = timeout
        self._url = f"{base_url}/api/embed"
        self._open = opener or urllib.request.urlopen
        self._document_prefix, self._query_prefix = LOCAL_EMBED_PREFIXES.get(
            _family(model), ("", "")
        )
        #: What the model itself emits, learned from its first reply.
        self.native_dimensions: int | None = None

    def embed(self, texts: list[str]) -> list[Vector]:
        """Document vectors — what the index stores."""
        return self._embed([self._document_prefix + (t if t.strip() else " ") for t in texts])

    def embed_query(self, text: str) -> Vector:
        """A query vector — a different task for the model, same space."""
        return self._embed([self._query_prefix + (text if text.strip() else " ")])[0]

    def _embed(self, texts: list[str]) -> list[Vector]:
        out: list[Vector] = []
        for start in range(0, len(texts), self.batch_size):
            out.extend(self._embed_batch(texts[start : start + self.batch_size]))
        return out

    def _embed_batch(self, texts: list[str]) -> list[Vector]:
        if not texts:
            return []
        payload = {"model": self.model, "input": texts, "truncate": True,
                   "keep_alive": LOCAL_KEEP_ALIVE}
        try:
            with self._open(_request(self._url, payload), timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            vectors = [list(map(float, vec)) for vec in body["embeddings"]]
        except (*_NETWORK_ERRORS, ValueError, KeyError, TypeError) as exc:
            # The server is down, the model is not pulled, or the reply is not
            # one. §9.3 gives all three the same answer: index and search on
            # fulltext plus expansion, and say search is degraded.
            raise EmbeddingUnavailable(f"{self.name}: {exc}") from exc
        if len(vectors) != len(texts):
            raise EmbeddingUnavailable(
                f"{self.name}: {len(vectors)} vectors for {len(texts)} inputs"
            )
        return [self._pad(vec) for vec in vectors]

    def _pad(self, vec: list[float]) -> list[float]:
        self.native_dimensions = len(vec)
        if len(vec) > self.dimensions:
            # Truncating would change every similarity. This is configuration,
            # not an outage (§12.11), so it stops rather than degrades.
            raise ValueError(
                f"{self.name} emits {len(vec)} dims; the vector index holds "
                f"{self.dimensions}. Choose a model of at most {self.dimensions}."
            )
        return vec + [0.0] * (self.dimensions - len(vec))


class OllamaLLM:
    """A local chat model behind S2's `.stream()`.

    `context_tokens` is a capacity, not a preference: past `num_ctx` the server
    drops the *start* of the prompt — the instructions — and answers anyway.
    The caller budgets the packed context to fit (`api/cli.py`), and this class
    counts the cases where the model's own tokenizer says it did not.
    """

    #: Dollars per million tokens, in and out. It runs on the user's hardware.
    rates: tuple[float, float] = (0.0, 0.0)
    local = True
    #: Restated after the context on every request; see `api.prompt`.
    answer_format: str = ANSWER_FORMAT_REMINDER

    def __init__(
        self,
        model: str = LOCAL_LLM_MODEL,
        *,
        base_url: str = OLLAMA_URL,
        context_tokens: int = LOCAL_LLM_CONTEXT_TOKENS,
        timeout: float = LOCAL_TIMEOUT_SECONDS,
        opener=None,
    ) -> None:
        self.model = model
        self.name = f"ollama:{model}"
        self.context_tokens = context_tokens
        self.timeout = timeout
        self._url = f"{base_url}/api/chat"
        self._open = opener or urllib.request.urlopen
        self._tokenizer = ApproxCodeTokenizer()

    def stream(
        self,
        system: str,
        prompt: str,
        *,
        max_tokens: int = RESERVED_OUT,
        usage: dict | None = None,
    ) -> Iterator[str]:
        options: dict = {"num_ctx": self.context_tokens, "num_predict": max_tokens}
        if LLM_TEMPERATURE is not None:
            options["temperature"] = LLM_TEMPERATURE      # §8.2: temperature 0
        payload = {
            "model": self.model,
            "stream": True,
            "keep_alive": LOCAL_KEEP_ALIVE,
            "options": options,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        with self._open(_request(self._url, payload), timeout=self.timeout) as response:
            for line in response:
                if not line.strip():
                    continue
                event = json.loads(line.decode("utf-8"))
                if event.get("error"):
                    raise RuntimeError(f"{self.name}: {event['error']}")
                text = (event.get("message") or {}).get("content") or ""
                if text:
                    yield text
                if event.get("done"):
                    self._finish(event, max_tokens, usage)
                    return

    def _finish(self, event: dict, max_tokens: int, usage: dict | None) -> None:
        prompt_tokens = event.get("prompt_eval_count") or 0
        if usage is not None:
            usage["input_tokens"] = prompt_tokens
            usage["output_tokens"] = event.get("eval_count") or 0
        if prompt_tokens and prompt_tokens + max_tokens > self.context_tokens:
            # The model's own count says the reply could not have been given
            # its full room; nearer the limit still, the prompt was cut.
            metrics.incr("llm.local.context_tight")

    def warm(self) -> None:
        """Load the model now, so the first question does not pay for it.

        A cold load was measured at 90 s on this class of machine (the weights
        come off disk and into the GPU); warm, the first token takes 0.1 s.
        An empty message list loads the model and generates nothing.
        """
        payload = {"model": self.model, "messages": [], "keep_alive": LOCAL_KEEP_ALIVE,
                   "options": {"num_ctx": self.context_tokens}}
        with self._open(_request(self._url, payload), timeout=self.timeout) as response:
            response.read()

    def count(self, text: str) -> int:
        """Ollama exposes no tokenizer, so this is the stand-in (F-006).

        The *reported* usage above is the model's own count; only planning uses
        this one, with `LOCAL_CONTEXT_HEADROOM` covering the difference.
        """
        return self._tokenizer.count(text)
