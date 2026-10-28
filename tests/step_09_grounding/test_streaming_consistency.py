"""§6.4: the live badge and the final report must agree.

The streaming validator decides T1/T2 as sentences close; `ground()` decides
them again over the whole answer. They used to segment differently, and the
difference showed exactly where it hurts: a real token stream splits
`. [C1]` into `.` · ` [` · `C1` · `]`, the live badge read ⚠ uncited, and the
final report counted the sentence as cited. The property tested here is the
fix — for any chunking of any answer, the sentences the stream closed are the
sentences the batch classifier finds.
"""

from __future__ import annotations

import random
import zlib

import pytest

from api.stream import SentenceEvent, StreamingValidator
from ground.classify import classify, split_sentences

HANDLES = {f"C{i}": f"uid_{i}" for i in range(1, 13)}

ANSWERS = [
    "The function `login` issues tokens. [C1] The class `User` stores them. [C2]",
    "The function `login` issues tokens. [C1][C2] Then `User` saves it [C3].",
    "The function `login` issues tokens.  [C1] [C2]\n[C3] next line `x` here.",
    "It handles auth, e.g. login, in `views` [C1]. Rate is 3.14 per call. Done",
    "Here [C1].\n\n```py\nv = a[C2]. x = 1.\n```\nAfter the fence `f` runs [C2]. ",
    "Which function bills it? The `charge` function bills it. [C9] Check `x`.",
    "No terminator at all `x` [C1]",
    "Ends with handle. [C1]",
    "Ends with a space. [C1] ",
    "A. [C1] B. [C2]. C! [C3]? D",
    "Trailing tab.\t[C1]\t[C2] x.",
    "The `a` calls `b`. [C1]\n\nThe `b` calls `c`. [C2] [C3] [C4]",
    "~~~\nfence. [C1]\n~~~\nThen `z`. [C2]",
    "[C1] leading handle `q`. ",
    "Brackets [x] and [C] and [C12x]. ok [C12].",
    "",
    "   ",
    ". . .",
    "Abbr vs. cf. Fig. No. 3 `x` [C1].",
    "Unclosed fence follows.\n```ts\nconst a = b[C1]. c",
]


def stream(text: str, sizes) -> StreamingValidator:
    validator = StreamingValidator(handle_map=HANDLES)
    i = 0
    for size in sizes:
        if i >= len(text):
            break
        list(validator.feed(text[i : i + size]))
        i += size
    if i < len(text):
        list(validator.feed(text[i:]))
    list(validator.close())
    return validator


def verdicts(sentences):
    return [(s.text, s.handles, s.is_claim) for s in sentences]


@pytest.mark.parametrize("text", ANSWERS)
def test_every_chunking_agrees_with_the_batch_classifier(text):
    expected = verdicts(classify(text).sentences)
    rng = random.Random(zlib.crc32(text.encode()))      # same chunkings every run
    for _trial in range(300):
        sizes = [rng.randint(1, 7) for _ in range(len(text) + 1)]
        got = verdicts(stream(text, sizes).sentences)
        assert got == expected, f"chunking {sizes[:20]}..."


@pytest.mark.parametrize("text", ANSWERS)
def test_one_character_at_a_time_agrees(text):
    assert verdicts(stream(text, [1] * (len(text) + 1)).sentences) == verdicts(
        classify(text).sentences
    )


def test_the_token_stream_that_used_to_fail():
    """`.` · ` [` · `C1` · `]` — the real tokenizer's split."""
    validator = StreamingValidator(handle_map=HANDLES)
    events = []
    for chunk in ["The function `login` issues tokens", ".", " [", "C1", "]", " Next."]:
        events += [e for e in validator.feed(chunk) if isinstance(e, SentenceEvent)]
    events += list(validator.close())

    first = events[0]
    assert first.cited and first.handles == (1,) and first.badge == "✓"
    assert not any(e.badge == "⚠ uncited" for e in events if "login" in e.text)


def test_a_late_handle_amends_the_closed_sentence():
    """`". "` closes a sentence immediately (the existing contract); a handle in
    the next chunk then re-issues its verdict instead of crediting the next
    sentence."""
    validator = StreamingValidator(handle_map=HANDLES)
    closed = [e for e in validator.feed("The `charge` function bills it. ") if isinstance(e, SentenceEvent)]
    assert closed and closed[0].badge == "⚠ uncited" and not closed[0].amends

    amended = [e for e in validator.feed("[C2] The next one.") if isinstance(e, SentenceEvent)]
    assert amended[0].amends and amended[0].handles == (2,) and amended[0].badge == "✓"
    assert len(validator.sentences) == 1, "the amendment replaces, it does not append"

    list(validator.close())
    assert verdicts(validator.sentences) == verdicts(
        classify("The `charge` function bills it. [C2] The next one.").sentences
    )


def test_a_handle_after_a_newline_starts_the_next_sentence():
    """The batch rule: trailing handles attach across spaces, not lines."""
    validator = StreamingValidator(handle_map=HANDLES)
    list(validator.feed("The `a` function runs.\n"))
    events = [e for e in validator.feed("[C1] The `b` function too.") if isinstance(e, SentenceEvent)]
    assert not any(e.amends for e in events)


def test_split_sentences_attaches_a_run_of_trailing_handles():
    sentences = split_sentences("The view calls it. [C1][C2] [C3] Then it returns.")
    assert sentences[0].text == "The view calls it. [C1][C2] [C3]"
    assert sentences[1].text == "Then it returns."


def test_handles_inside_a_fence_never_cite_even_mid_stream():
    text = "Look:\n```py\nx = arr[C1]\ny = arr[C2].\n```\nThe `f` function returns `y`. [C3]"
    for size in range(1, 9):
        validator = stream(text, [size] * len(text))
        cited = [s.handles for s in validator.sentences if s.cited]
        assert cited == [(3,)], f"chunk size {size}"
