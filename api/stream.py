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

import re

from ground.classify import (
    TRAILING_HANDLES_RE,
    Sentence,
    classify_sentence,
    handles_outside_fences,
    split_sentences,
)

#: What the start of a trailing handle looks like before the rest of it has
#: arrived: `[`, `[C`, `[C1`, with the spaces that may precede it. A buffer in
#: this state is waited on rather than evaluated.
_PARTIAL_HANDLE_RE = re.compile(r"[ \t]*(?:\[C\d+\][ \t]*)*\[(?:C\d*)?")

#: A sentence that closed on a terminator, possibly with its handles already
#: attached — the only kind trailing handles can still be added to.
_ENDS_AT_TERMINATOR_RE = re.compile(r"(?:[.!?]|\[C\d+\])$")


@dataclass(frozen=True)
class TextEvent:
    """A chunk of answer text, for immediate display."""

    text: str
    type: str = "text"


@dataclass(frozen=True)
class SentenceEvent:
    """A closed sentence with its T1/T2 verdict.

    `amends` is True when this event re-issues the previous sentence's verdict
    because citations arrived after it closed — see `StreamingValidator`. The
    client replaces the earlier badge rather than adding a second one.
    """

    text: str
    is_claim: bool
    cited: bool
    handles: tuple[int, ...]
    valid_handles: bool
    reason: str
    type: str = "sentence"
    amends: bool = False

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
    """Buffers to a sentence terminator, then evaluates T1/T2 on the closed span.

    **Segmentation is the batch classifier's, not a second implementation.**
    Every drain runs `split_sentences` over the buffer — the function T2's
    final report uses — so fences, abbreviations, decimals and trailing handles
    are decided once. Two segmenters disagreed before: the live badge said
    ⚠ uncited for a sentence the final report counted as cited, whenever the
    stream split `. [C1]` across chunks, which a real token stream does
    routinely (`.` · ` [` · `C1` · `]`).

    A sentence is closed when text follows it, or when its terminator is
    followed by whitespace. Two cases wait instead: a terminator at the very end
    of the buffer, and a buffer ending in a handle that has not finished
    arriving (`[`, `[C`, `[C1`). And when a sentence closed on `". "` and the
    next chunk opens with handles, those handles belong to it — §6.1's "end
    with one or more context handles" — so its verdict is re-issued as an
    amendment rather than the handles crediting the following sentence.
    """

    handle_map: dict[str, str]
    known_names: frozenset[str] = frozenset()
    _buffer: str = ""
    _full: str = ""
    #: True while the last emitted sentence closed on a terminator and nothing
    #: but spaces has followed it — the window in which handles still attach.
    _amendable: bool = False
    sentences: list[Sentence] = field(default_factory=list)

    @property
    def text(self) -> str:
        return self._full

    def feed(self, chunk: str) -> Iterator[TextEvent | SentenceEvent]:
        """Consume one streamed chunk, emitting text immediately."""
        self._full += chunk
        self._buffer += chunk
        yield TextEvent(text=chunk)

        for closed in self._drain(final=False):
            yield closed

    def close(self) -> Iterator[SentenceEvent]:
        """Flush whatever the model left unterminated.

        Models routinely end without a final full stop. Dropping the tail would
        silently remove a claim from T2's denominator — which raises Coverage
        by discarding evidence, the most flattering possible bug.
        """
        yield from self._drain(final=True)
        self._buffer = ""

    # ------------------------------------------------------------------

    def _drain(self, *, final: bool) -> Iterable[SentenceEvent]:
        if self._amendable:
            amended = self._take_trailing_handles(final=final)
            if amended is None:
                return                          # a handle is still arriving
            if amended:
                yield amended

        sentences = split_sentences(self._buffer)
        if not sentences:
            return

        if final:
            closing = sentences
        else:
            closing = sentences[:-1]
            last = sentences[-1]
            if _PARTIAL_HANDLE_RE.fullmatch(last.text):
                closing = sentences[:-2]        # its owner's handle is mid-flight
            elif last.end < len(self._buffer):
                closing = sentences             # terminator, then whitespace

        if not closing:
            return
        for sentence in closing:
            yield self._emit(sentence.text)
        cut = closing[-1].end
        rest = self._buffer[cut:]
        self._buffer = rest
        # Handles can still attach to the sentence just emitted while only
        # spaces follow its terminator — the batch rule, TRAILING_HANDLES_RE.
        self._amendable = bool(
            _ENDS_AT_TERMINATOR_RE.search(closing[-1].text) and rest.strip(" \t") == ""
        )

    def _take_trailing_handles(self, *, final: bool) -> SentenceEvent | None | bool:
        """Attach handles that arrived after the last sentence closed.

        Returns the amendment event, False when there is nothing to attach (the
        window has closed), or None while the buffer could still become one.
        """
        buffer = self._buffer
        if not final and (buffer.strip(" \t") == "" or _PARTIAL_HANDLE_RE.fullmatch(buffer)):
            return None                         # spaces so far, or `[C` mid-flight
        match = TRAILING_HANDLES_RE.match(buffer)
        if match is None:
            self._amendable = False
            return False
        if not final and match.end() == len(buffer):
            return None                         # another handle may follow

        self._buffer = buffer[match.end() :]
        # Still only spaces after the handles: a further handle may yet arrive.
        self._amendable = self._buffer.strip(" \t") == ""
        previous = self.sentences.pop()
        return self._emit(previous.text + buffer[: match.end()], amends=True)

    def _emit(self, text: str, *, amends: bool = False) -> SentenceEvent:
        text = text.strip()
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
            amends=amends,
        )
