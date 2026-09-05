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
that wiring lands. See FINDINGS.md F-006.
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
from typing import Iterator, Protocol, Sequence

from config import EMBED_DIM, LLM_MODEL, RESERVED_OUT

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


# --------------------------------------------------------------------------
# S2 — the LLM half
# --------------------------------------------------------------------------


class LLMProvider(Protocol):
    """S2's `.stream()` and `.tokenize()`.

    §2 rates the LLM side of this seam as "one file" to swap, and that is
    accurate — unlike the embedding side, which is a full re-index (§12.11).
    """

    name: str

    def stream(self, system: str, prompt: str, *, max_tokens: int) -> Iterator[str]: ...

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

    def stream(self, system: str, prompt: str, *, max_tokens: int = RESERVED_OUT):
        with self._get_client().messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            yield from stream.text_stream

    def count(self, text: str) -> int:
        response = self._get_client().messages.count_tokens(
            model=self.model, messages=[{"role": "user", "content": text}]
        )
        return response.input_tokens


class ScriptedLLM:
    """A provider that streams a fixed answer, in chunks.

    Chunked rather than returned whole because §6.4's validator is incremental:
    it "buffers to the next sentence terminator, then evaluates T1/T2 on the
    closed span". A double that yielded the whole answer at once would never
    exercise a partial sentence, which is the only state that code runs in.
    """

    name = "scripted-test-llm"

    def __init__(self, *answers: str, chunk: int = 7) -> None:
        self._answers = list(answers) or [""]
        self._chunk = chunk
        self.calls: list[tuple[str, str]] = []

    def stream(self, system: str, prompt: str, *, max_tokens: int = RESERVED_OUT):
        self.calls.append((system, prompt))
        answer = self._answers[min(len(self.calls) - 1, len(self._answers) - 1)]
        for start in range(0, len(answer), self._chunk):
            yield answer[start : start + self._chunk]

    def count(self, text: str) -> int:
        return ApproxCodeTokenizer().count(text)
