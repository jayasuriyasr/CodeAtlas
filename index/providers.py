"""S2 implementations.

Two live here today, and neither is the production pair:

* `ApproxCodeTokenizer` — a real lexical tokenizer, but **not the provider's**.
* `HashEmbeddingProvider` — deterministic, offline, for tests.

Why that is the current state, stated plainly rather than buried: §5.4 requires
the provider's real tokenizer, and both candidates need something this
environment does not have. Anthropic's count is an API call
(`messages.count_tokens`), and Voyage's tokenizer is a download. Either would
also send source code to a third party, which decision 0001 permits in
production but which a unit test must not do.

So the seam is real and every call site goes through it; the concrete
production implementations are wired at step 9, alongside the API keys.
`ApproxCodeTokenizer` is a stand-in whose error against the real tokenizer is
**unmeasured**, which is why §5.4's ceiling arithmetic cannot be checked until
that wiring lands. See F-006 in DESIGN.md.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import time
import urllib.error
import urllib.request
from typing import Iterator, Protocol, Sequence

from config import (
    EMBED_BATCH_SIZE,
    EMBED_DIM,
    EMBED_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    PROVIDER_TIMEOUT_SECONDS,
    RESERVED_OUT,
)
from index.cache import EmbeddingDisabled, EmbeddingUnavailable

Vector = Sequence[float]


# --------------------------------------------------------------------------
# Tokenization
# --------------------------------------------------------------------------

#: Lexical classes a code tokenizer separates. Ordered: the first alternative
#: that matches wins, so strings are consumed before their contents are.
_TOKEN = re.compile(
    r"""
      "{3}(?:.|\n)*?"{3}      # triple-quoted string
    | '{3}(?:.|\n)*?'{3}
    | "(?:\\.|[^"\\])*"       # single-line string
    | '(?:\\.|[^'\\])*'
    | \#[^\n]*                # comment
    | [A-Za-z_][A-Za-z_0-9]*  # identifier
    | \d+\.?\d*               # number
    | \s+                     # whitespace run
    | .                       # any single operator or punctuation char
    """,
    re.VERBOSE,
)

#: Sub-token split inside an identifier or string, approximating what a BPE
#: vocabulary does to a long word. Chosen to be deterministic, not accurate.
_SUBTOKEN_CHARS = 4


class ApproxCodeTokenizer:
    """A lexical tokenizer used where the provider's real one is not yet wired.

    It is not `len // 4`. It lexes the text into strings, comments, identifiers,
    numbers, whitespace and operators, then splits long lexemes into
    fixed-width pieces the way a BPE vocabulary would split a rare identifier.

    It is still an approximation, and the direction of its error is unknown. Do
    not report a token count from this class as a token count in any metric that
    §8.2 gates — `name` is deliberately conspicuous so such a report is
    traceable.
    """

    name = "approx-code-tokenizer(NOT-A-PROVIDER-TOKENIZER)"

    def count(self, text: str) -> int:
        if not text:
            return 0
        total = 0
        for match in _TOKEN.finditer(text):
            lexeme = match.group()
            if lexeme.isspace():
                # Runs of whitespace collapse the way real tokenizers merge
                # them: one token for a run, not one per character.
                total += 1 + lexeme.count("\n")
                continue
            total += max(1, math.ceil(len(lexeme) / _SUBTOKEN_CHARS))
        return total


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------


class HashEmbeddingProvider:
    """A deterministic embedding provider for tests.

    Same text in, same vector out, with no network and no source egress. It has
    no semantic structure whatsoever, so it can exercise cache behaviour,
    dimensionality and the T5 write path — and it can say nothing about recall.
    Any test that claims a retrieval quality number needs the real provider.
    """

    #: Its vectors mean nothing, so nothing may search with them (F-009).
    semantic = False

    def __init__(self, dimensions: int = EMBED_DIM, name: str = "hash-test-provider"):
        self.dimensions = dimensions
        self.name = name
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[Vector]:
        self.calls.append(list(texts))
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> Vector:
        out: list[float] = []
        counter = 0
        while len(out) < self.dimensions:
            digest = hashlib.sha256(f"{counter}\x00{text}".encode()).digest()
            for offset in range(0, len(digest), 4):
                if len(out) >= self.dimensions:
                    break
                (raw,) = struct.unpack(">I", digest[offset : offset + 4])
                out.append(raw / 0xFFFFFFFF * 2.0 - 1.0)
            counter += 1

        norm = math.sqrt(sum(v * v for v in out)) or 1.0
        return [v / norm for v in out]      # cosine similarity wants unit length


class DisabledEmbeddingProvider:
    """No embedder at all: `--offline`, or nothing configured and nothing local.

    The CLI used to fall back to `HashEmbeddingProvider` here, which *writes*
    its meaningless vectors — over real ones, if the graph had them. Raising
    instead takes the indexer's degraded path (§9.3): symbols are written
    without a vector, any vector they already had stays, and search runs on
    fulltext and expansion.
    """

    name = "none"
    semantic = False

    def __init__(self, dimensions: int = EMBED_DIM) -> None:
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[Vector]:
        raise EmbeddingDisabled("no embedding provider is configured")


class VoyageEmbeddingProvider:
    """The embedding provider decision 0002 names: `voyage-code-2`, 1536 dims.

    Spoken to over Voyage's REST endpoint with the standard library rather than
    an SDK, so the production provider adds no dependency. Unexercised against
    the live API by the suite for the same reason as `AnthropicProvider` — it
    sends source code to a third party — and tested against a stub transport.

    Failure is typed: anything that stops a batch embedding — a network error,
    a timeout, a 4xx/5xx after retries, a malformed reply — raises
    `EmbeddingUnavailable`, which §9.3 turns into a degraded index rather than a
    failed one. 429 and 5xx are retried with backoff; other 4xx are not, since
    repeating a bad request only repeats the refusal.

    `input_type` is left unset for documents and queries alike. Voyage accepts
    that, and it keeps document and query vectors in one space without the
    seam needing a second method.
    """

    URL = "https://api.voyageai.com/v1/embeddings"
    semantic = True

    def __init__(
        self,
        model: str = EMBED_MODEL,
        *,
        api_key: str | None = None,
        dimensions: int = EMBED_DIM,
        batch_size: int = EMBED_BATCH_SIZE,
        timeout: float = PROVIDER_TIMEOUT_SECONDS,
        retries: int = 3,
        opener=None,
    ) -> None:
        self.name = model
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.timeout = timeout
        self.retries = retries
        self._api_key = api_key if api_key is not None else os.environ.get("VOYAGE_API_KEY", "")
        self._open = opener or urllib.request.urlopen
        if not self._api_key:
            raise ValueError("VOYAGE_API_KEY is not set")

    def embed(self, texts: list[str]) -> list[Vector]:
        out: list[Vector] = []
        for start in range(0, len(texts), self.batch_size):
            out.extend(self._embed_batch(texts[start : start + self.batch_size]))
        return out

    def _embed_batch(self, texts: list[str]) -> list[Vector]:
        if not texts:
            return []
        # An empty string is rejected by the API; a single space embeds as
        # "nothing here", which is what an empty chunk is.
        body = json.dumps(
            {"model": self.model, "input": [t if t.strip() else " " for t in texts]}
        ).encode()
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(
                self.URL,
                data=body,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with self._open(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                vectors = _parse_embeddings(payload, len(texts))
            except urllib.error.HTTPError as exc:
                last = exc
                if exc.code != 429 and exc.code < 500:
                    break                          # a bad request stays bad
            except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
                last = exc                         # network, timeout, malformed reply
            else:
                # A wrong dimension is configuration, not an outage: §12.11
                # makes a model change a full re-index. Degrading would index
                # every symbol without a vector, forever, and say "degraded".
                for vec in vectors:
                    if len(vec) != self.dimensions:
                        raise ValueError(
                            f"{self.model} returned {len(vec)} dims, expected "
                            f"{self.dimensions}. §12.11: a model change is a full re-index."
                        )
                return vectors
            if attempt < self.retries:
                time.sleep(min(2 ** attempt, 8))
        raise EmbeddingUnavailable(f"{self.model}: {last}") from last


def _parse_embeddings(payload: dict, expected: int) -> list[Vector]:
    """Voyage's `{"data": [{"index", "embedding"}]}`, in input order."""
    try:
        rows = sorted(payload["data"], key=lambda row: row["index"])
        vectors = [list(map(float, row["embedding"])) for row in rows]
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"malformed embedding response: {exc!r}") from exc
    if len(vectors) != expected:
        raise ValueError(f"{len(vectors)} vectors for {expected} inputs")
    return vectors


# --------------------------------------------------------------------------
# S2 — the LLM half
# --------------------------------------------------------------------------


class LLMProvider(Protocol):
    """S2's `.stream()` and `.tokenize()`.

    §2 rates the LLM side of this seam as "one file" to swap, and that is
    accurate — unlike the embedding side, which is a full re-index (§12.11).

    `usage`, when passed, is filled with the provider's own `input_tokens` and
    `output_tokens` once the stream ends. It is a per-call out-parameter rather
    than provider state because one provider serves concurrent requests.
    """

    name: str

    def stream(
        self, system: str, prompt: str, *, max_tokens: int, usage: dict | None = None
    ) -> Iterator[str]: ...

    def count(self, text: str) -> int: ...


class AnthropicProvider:
    """The provider decision 0002 names: `claude-haiku-4-5`.

    Unexercised by the test suite, and deliberately so. Calling it sends full
    function bodies to a third party — which decision 0001 permits in
    production and a unit test must not do. Its first real run is the step 9
    gate, against a configured key.

    `count` uses the API's own token counter rather than a local approximation,
    which is what §5.4 means by "the provider's real tokenizer". That makes it a
    network call, so the packer holds a locally-computed count until this is
    wired — see FINDINGS F-006.
    """

    def __init__(self, model: str = LLM_MODEL, client=None) -> None:
        self.name = model
        self.model = model
        self._client = client

    def _get_client(self):
        if self._client is None:
            import anthropic                       # imported late: optional at test time

            self._client = anthropic.Anthropic()
        return self._client

    def stream(
        self,
        system: str,
        prompt: str,
        *,
        max_tokens: int = RESERVED_OUT,
        usage: dict | None = None,
    ):
        # anthropic 1.x removed `temperature` from the method signatures (a
        # TypeError if passed) but not from the API, and claude-haiku-4-5
        # still honours it. §8.2's "eval runs at temperature 0" depends on it,
        # so it travels in `extra_body`. Newer models reject non-default
        # sampling values: set LLM_TEMPERATURE to None for those.
        sampling = {"temperature": LLM_TEMPERATURE} if LLM_TEMPERATURE is not None else {}
        with self._get_client().messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            **({"extra_body": sampling} if sampling else {}),
        ) as stream:
            yield from stream.text_stream
            if usage is not None:
                final = stream.get_final_message().usage
                usage["input_tokens"] = final.input_tokens
                usage["output_tokens"] = final.output_tokens

    def count(self, text: str) -> int:
        response = self._get_client().messages.count_tokens(
            model=self.model, messages=[{"role": "user", "content": text}]
        )
        return response.input_tokens


class UnavailableLLM:
    """No model to answer with — and the reason, for whoever is waiting.

    Raising from `stream` puts the request on §9.3's "LLM provider" row: the
    reader gets the ranked, cited code blocks and is told synthesis is
    unavailable, rather than an error page or an answer from nowhere.
    """

    name = "none"

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def stream(
        self,
        system: str,
        prompt: str,
        *,
        max_tokens: int = RESERVED_OUT,
        usage: dict | None = None,
    ):
        raise ConnectionError(self.reason)
        yield ""                                   # pragma: no cover - a generator

    def count(self, text: str) -> int:
        return ApproxCodeTokenizer().count(text)


class ScriptedLLM:
    """A provider that streams a fixed answer, in chunks.

    Chunked rather than returned whole because §6.4's validator is incremental:
    it "buffers to the next sentence terminator, then evaluates T1/T2 on the
    closed span". A double that yielded the whole answer at once would never
    exercise a partial sentence, which is the only state that code runs in.
    """

    name = "scripted-test-llm"

    def __init__(
        self, *answers: str, chunk: int = 7, fail_after: int | None = None,
        usage: dict | None = None,
    ) -> None:
        self._answers = list(answers) or [""]
        self._chunk = chunk
        #: Raise `ConnectionError` after this many chunks of a call — the
        #: provider dropping mid-answer, for §9.3's "LLM provider" row.
        self._fail_after = fail_after
        #: Usage to report, as a real provider would; None reports nothing.
        self._usage = usage
        self.calls: list[tuple[str, str]] = []

    def stream(
        self,
        system: str,
        prompt: str,
        *,
        max_tokens: int = RESERVED_OUT,
        usage: dict | None = None,
    ):
        self.calls.append((system, prompt))
        answer = self._answers[min(len(self.calls) - 1, len(self._answers) - 1)]
        for index, start in enumerate(range(0, len(answer), self._chunk)):
            if self._fail_after is not None and index >= self._fail_after:
                raise ConnectionError("scripted provider failure")
            yield answer[start : start + self._chunk]
        if self._fail_after is not None and self._fail_after == 0:
            raise ConnectionError("scripted provider failure")
        if usage is not None and self._usage is not None:
            usage.update(self._usage)

    def count(self, text: str) -> int:
        return ApproxCodeTokenizer().count(text)
