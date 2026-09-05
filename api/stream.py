"""§6.4 — the incremental citation parser.

> The client buffers to the next sentence terminator, then evaluates T1/T2 on
> the closed span: text renders immediately, ✓/⚠ land at sentence close, T3
> upgrades to ✓✓ after the stream. Perceived latency stays at TTFT.

Two things follow that are easy to get backwards. Text is emitted the moment it
arrives — buffering the *display* would spend the TTFT budget §8.2 gates at
1.2s p50. Only the *validation* waits for the sentence to close, because a
half-written sentence has no citation yet and flagging it would put a ⚠ on
every sentence in the answer, mid-stream, for a second.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator

from ground.classify import (
    HANDLE_RE,
    Sentence,
    classify_sentence,
    fenced_spans,
    handles_outside_fences,
)


@dataclass(frozen=True)
class TextEvent:
    """A chunk of answer text, for immediate display."""

    text: str
    type: str = "text"


@dataclass(frozen=True)
class SentenceEvent:
    """A closed sentence with its T1/T2 verdict."""

    text: str
    is_claim: bool
    cited: bool
    handles: tuple[int, ...]
    valid_handles: bool
    reason: str
    type: str = "sentence"

    @property
    def badge(self) -> str:
        """What the UI renders beside the sentence.

        A non-claim gets nothing rather than a tick: §12.4 warns the trust model
        rests on these markers, and ticking prose that asserted nothing about
        the code teaches the reader that a tick means very little.
        """
        if not self.is_claim:
            return ""
        if not self.valid_handles:
            return "⚠ unknown citation"
        return "✓" if self.cited else "⚠ uncited"


@dataclass
class StreamingValidator:
    """Buffers to a sentence terminator, then evaluates T1/T2 on the closed span."""

    handle_map: dict[str, str]
    known_names: frozenset[str] = frozenset()
    _buffer: str = ""
    _full: str = ""
    sentences: list[Sentence] = field(default_factory=list)

    @property
    def text(self) -> str:
        return self._full

    def feed(self, chunk: str) -> Iterator[TextEvent | SentenceEvent]:
        """Consume one streamed chunk, emitting text immediately."""
        self._full += chunk
        self._buffer += chunk
        yield TextEvent(text=chunk)

        for closed in self._drain():
            yield closed

    def close(self) -> Iterator[SentenceEvent]:
        """Flush whatever the model left unterminated.

        Models routinely end without a final full stop. Dropping the tail would
        silently remove a claim from T2's denominator — which raises Coverage
        by discarding evidence, the most flattering possible bug.
        """
        remainder = self._buffer.strip()
        self._buffer = ""
        if remainder:
            yield self._emit(remainder)

    # ------------------------------------------------------------------

    def _drain(self) -> Iterable[SentenceEvent]:
        while True:
            index = self._terminator_index(self._buffer)
            if index is None:
                return
            span = self._buffer[: index + 1]
            self._buffer = self._buffer[index + 1 :]
            if span.strip():
                yield self._emit(span.strip())

    def _terminator_index(self, buffer: str) -> int | None:
        """The first sentence terminator not inside a fenced block.

        A trailing handle is waited for: `... here. [C3]` cites the sentence it
        follows, and closing at the full stop would evaluate the sentence as
        uncited a moment before its citation arrives.
        """
        fences = fenced_spans(buffer)
        for i, char in enumerate(buffer):
            if char not in ".!?":
                continue
            if any(f.start <= i < f.end for f in fences):
                continue
            rest = buffer[i + 1 :]
            if rest[:1] not in ("", " ", "\n", "\t"):
                continue                       # "3.14" or "views.py"
            trailing = HANDLE_RE.match(rest.lstrip(" "))
            if trailing is not None:
                offset = len(rest) - len(rest.lstrip(" "))
                return i + offset + trailing.end()
            if rest == "":
                return None                    # may still be followed by a handle
            return i
        return None

    def _emit(self, text: str) -> SentenceEvent:
        sentence = Sentence(text=text, start=0, end=len(text))
        # Fence-aware, via the same helper the batch classifier uses (v9.1 #9).
        sentence.handles = handles_outside_fences(text)
        classify_sentence(sentence, self.known_names)
        self.sentences.append(sentence)

        valid = all(f"C{h}" in self.handle_map for h in sentence.handles)
        return SentenceEvent(
            text=text,
            is_claim=sentence.is_claim,
            cited=sentence.cited,
            handles=sentence.handles,
            valid_handles=valid,
            reason=sentence.reason,
        )
