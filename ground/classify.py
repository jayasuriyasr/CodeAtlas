"""§6.1 — deciding which sentences require a citation.

This module owns T2's denominator, and §6.1 is blunt about the cost of getting
it wrong:

> Deciding which sentences require a citation is classification, not pattern
> matching, and an over-inclusive denominator manufactures low Coverage and
> fires retries on answers that were never wrong.

So the rule is conservative by construction: **when ambiguous, exclude**. A
missed claim understates a problem; a false claim invents one and burns a retry.
§12.9 is honest that this predicate's errors propagate into Coverage, and §8.2
turns that from an assumption into a measured precision/recall pair.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: §6.4. `[C<n>]`, and nothing else is a handle.
HANDLE_RE = re.compile(r"\[C(\d+)\]")

#: Fenced code blocks, ``` or ~~~, with or without a language tag. Also inline
#: spans, which are handled separately because a backticked identifier is
#: *evidence of* a claim while a fenced block is an exclusion from one.
_FENCE_RE = re.compile(r"(?P<fence>```|~~~)[^\n]*\n.*?(?:(?P=fence)|\Z)", re.S)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")

#: §6.1's code-entity nouns, verbatim.
_ENTITY_NOUNS = (
    "function", "class", "method", "module", "route", "hook", "component",
    "endpoint",
)
_ENTITY_NOUN_RE = re.compile(
    r"\b(?:" + "|".join(_ENTITY_NOUNS) + r")s?\b", re.I
)

#: Hedges. §6.1 excludes "statements the model explicitly hedged, since
#: penalizing declared uncertainty trains the overconfidence this system exists
#: to prevent."
_HEDGE_RE = re.compile(
    r"\b(?:"
    r"might|may|maybe|probably|possibly|perhaps|likely|unclear|uncertain|"
    r"appears? to|seems? to|seems like|looks like|suggests?|"
    r"i (?:think|believe|suspect|am not sure)|not (?:sure|certain)|"
    r"cannot (?:tell|determine|confirm)|can't (?:tell|determine|confirm)|"
    r"i (?:do not|don't) (?:know|have)|no (?:evidence|information)|"
    r"presumably|apparently"
    r")\b",
    re.I,
)

#: Imperatives that open a sentence. An instruction asserts nothing about the
#: codebase, so requiring a citation for it manufactures a coverage failure.
_IMPERATIVE_RE = re.compile(
    r"^(?:please\s+)?(?:"
    r"see|check|note|consider|try|run|use|add|remove|open|look|review|"
    r"call|read|refer|start|stop|install|update|set|make|let|avoid|ensure"
    r")\b",
    re.I,
)


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    def contains(self, index: int) -> bool:
        return self.start <= index < self.end


@dataclass
class Sentence:
    text: str
    start: int
    end: int
    handles: tuple[int, ...] = ()
    is_claim: bool = False
    #: Why the classifier decided as it did. Kept because §12.9 says these
    #: errors propagate into Coverage, and a disagreement is much cheaper to
    #: settle when the reason is recorded than when it has to be re-derived.
    reason: str = ""

    @property
    def cited(self) -> bool:
        return bool(self.handles)


# --------------------------------------------------------------------------
# Fenced blocks
# --------------------------------------------------------------------------


def fenced_spans(text: str) -> list[Span]:
    """Byte ranges of fenced code blocks, including the fences themselves."""
    return [Span(m.start(), m.end()) for m in _FENCE_RE.finditer(text)]


def in_any(spans: list[Span], index: int) -> bool:
    return any(span.contains(index) for span in spans)


def handles_outside_fences(text: str) -> tuple[int, ...]:
    """Handle numbers in `text`, excluding any inside a fenced block.

    The single place both paths ask the question. The batch classifier and the
    streaming validator (§6.4) must agree exactly — they did not at first, and
    the streaming path counted `v = a[C2]` inside a snippet as a citation while
    the batch path correctly ignored it. Two implementations of v9.1 #9 is one
    too many.
    """
    fences = fenced_spans(text)
    return tuple(
        int(m.group(1))
        for m in HANDLE_RE.finditer(text)
        if not in_any(fences, m.start())
    )


def find_handles(text: str) -> list[tuple[int, int]]:
    """`(handle_number, position)` for every handle **outside** a fenced block.

    v9.1 #9. §6.4: "Recognized outside fenced code blocks only — answers about
    code routinely contain bracket-index expressions." `arr[C1]` inside a
    snippet is an array index, and treating it as a citation both credits a
    sentence that cited nothing and, when the block does not exist, reports the
    model for hallucinating its own example code.
    """
    fences = fenced_spans(text)
    return [
        (int(m.group(1)), m.start())
        for m in HANDLE_RE.finditer(text)
        if not in_any(fences, m.start())
    ]


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------

_TERMINATOR_RE = re.compile(r"[.!?](?=\s|$)")

#: Abbreviations whose full stop does not end a sentence. Short list on purpose:
#: over-splitting turns one claim into two, which changes T2's denominator.
_ABBREVIATIONS = ("e.g.", "i.e.", "etc.", "vs.", "cf.", "Fig.", "No.")


def split_sentences(text: str) -> list[Sentence]:
    """Split into sentences, skipping fenced blocks entirely.

    §6.4 describes the client buffering "to the next sentence terminator, then
    evaluat[ing] T1/T2 on the closed span", so this is also the boundary the
    streaming validator uses.
    """
    fences = fenced_spans(text)
    boundaries: list[int] = []

    for match in _TERMINATOR_RE.finditer(text):
        end = match.end()
        if in_any(fences, match.start()):
            continue
        window = text[max(0, end - 6) : end]
        if any(window.endswith(abbr) for abbr in _ABBREVIATIONS):
            continue
        # A handle may follow the terminator: "... is called here. [C3]"
        trailing = HANDLE_RE.match(text, end + 1 if text[end : end + 1] == " " else end)
        if trailing is not None:
            end = trailing.end()
        boundaries.append(end)

    # A closed fence ends a sentence too. Without this, a code block and the
    # prose after it land in one span, and that span is then classified once —
    # so a claim following a snippet inherits the snippet's exclusion.
    boundaries.extend(span.end for span in fences)

    out: list[Sentence] = []
    start = 0
    for end in sorted(set(boundaries)):
        if end <= start:
            continue
        chunk = text[start:end]
        if chunk.strip():
            out.append(Sentence(text=chunk.strip(), start=start, end=end))
        start = end

    tail = text[start:]
    if tail.strip():
        out.append(Sentence(text=tail.strip(), start=start, end=len(text)))

    return out


# --------------------------------------------------------------------------
# The predicate
# --------------------------------------------------------------------------


def _strip_fences(sentence_text: str) -> str:
    return _FENCE_RE.sub(" ", sentence_text)


def classify_sentence(
    sentence: Sentence, known_names: frozenset[str] = frozenset()
) -> Sentence:
    """§6.1's predicate, applied to one sentence.

    A sentence is a claim when it is **declarative** *and* **references a code
    entity** — a handle, a backticked identifier, a `qualified_name` present in
    packed context, or one of §6.1's code-entity nouns.

    `known_names` is the set of qualified names actually packed. Passing it lets
    an unbackticked mention of a real symbol count, which is the one form the
    other three tests miss.
    """
    body = _strip_fences(sentence.text).strip()

    if not body:
        return _decide(sentence, False, "empty outside fenced code")

    if body.rstrip().endswith("?"):
        return _decide(sentence, False, "interrogative")

    if _IMPERATIVE_RE.match(body):
        return _decide(sentence, False, "imperative")

    if _HEDGE_RE.search(body):
        return _decide(sentence, False, "hedged")

    has_handle = bool(HANDLE_RE.search(body))
    has_backticked = bool(_INLINE_CODE_RE.search(body))
    has_entity_noun = bool(_ENTITY_NOUN_RE.search(body))
    has_known_name = any(name and name in body for name in known_names)

    if has_handle:
        return _decide(sentence, True, "handle")
    if has_backticked:
        return _decide(sentence, True, "backticked identifier")
    if has_known_name:
        return _decide(sentence, True, "packed qualified_name")
    if has_entity_noun:
        return _decide(sentence, True, "code-entity noun")

    return _decide(sentence, False, "no code reference")


def _decide(sentence: Sentence, is_claim: bool, reason: str) -> Sentence:
    sentence.is_claim = is_claim
    sentence.reason = reason
    return sentence


@dataclass
class ClassifiedAnswer:
    text: str
    sentences: list[Sentence] = field(default_factory=list)

    @property
    def claims(self) -> list[Sentence]:
        return [s for s in self.sentences if s.is_claim]

    @property
    def cited_claims(self) -> list[Sentence]:
        return [s for s in self.claims if s.cited]

    @property
    def uncited_claims(self) -> list[Sentence]:
        return [s for s in self.claims if not s.cited]

    @property
    def all_handles(self) -> list[int]:
        return sorted({h for s in self.sentences for h in s.handles})


def classify(text: str, known_names: frozenset[str] = frozenset()) -> ClassifiedAnswer:
    """Segment, attach handles, and decide which sentences are claims."""
    fences = fenced_spans(text)
    sentences = split_sentences(text)

    for sentence in sentences:
        handles = tuple(
            int(m.group(1))
            for m in HANDLE_RE.finditer(sentence.text)
            if not in_any(
                [Span(f.start - sentence.start, f.end - sentence.start) for f in fences],
                m.start(),
            )
        )
        sentence.handles = handles
        classify_sentence(sentence, known_names)

    return ClassifiedAnswer(text=text, sentences=sentences)
