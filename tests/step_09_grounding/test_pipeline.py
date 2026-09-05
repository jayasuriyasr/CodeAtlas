"""Step 9 — the query path end to end, and the SSE surface.

Everything here runs against `ScriptedLLM`. Calling the real provider would send
source code to a third party, which decision 0001 permits in production and a
unit test must not do — so what is tested is the orchestration, the streaming
validator, and the retry gate. The one thing only a real provider can establish
is whether the model actually cites, and that is the step 9 gate.
"""

from __future__ import annotations

import json

import pytest

import metrics
from adapters.base import Symbol
from api.app import AnswerCost, InvestigationService, create_app, sse
from api.prompt import SYSTEM, build_prompt
from api.stream import SentenceEvent, StreamingValidator, TextEvent
from config import LLM_MODEL
from index.providers import ApproxCodeTokenizer, ScriptedLLM
from pack.packer import pack
from pack.render import Block
from retrieve.router import QueryClass


@pytest.fixture(autouse=True)
def reset_metrics():
    metrics.reset()
    yield
    metrics.reset()


@pytest.fixture(scope="module")
def tok() -> ApproxCodeTokenizer:
    return ApproxCodeTokenizer()


def symbol(name: str, *, body_lines: int = 3) -> Symbol:
    body = "\n".join(f"    step_{i} = compute({i})" for i in range(body_lines))
    return Symbol(
        uid=f"uid_{name}",
        repo_id="repo",
        rel_path=f"billing/{name}.py",
        qualified_name=f"billing.{name}.{name}",
        name=name,
        arity=1,
        ordinal=0,
        kind="function",
        signature=f"def {name}(user):",
        docstring=f"Docs for {name}.",
        source_code=f"def {name}(user):\n{body}\n",
        enclosing_signature=None,
        used_imports=["common.utils.audit_event"],
    )


def block(name: str, score: float, **kw) -> Block:
    return Block(symbol=symbol(name, **kw), score=score)


def service(answers, *, seeds=None, neighbors=None, runner=None, tok=None, **kw):
    seeds = seeds if seeds is not None else [block("charge_card", 1.0)]
    neighbors = neighbors if neighbors is not None else [block("retry_failed", 0.5)]
    return InvestigationService(
        retriever=lambda q, routed: (seeds, neighbors),
        llm=ScriptedLLM(*answers),
        tokenizer=tok or ApproxCodeTokenizer(),
        graph_runner=runner,
        repo_id="repo",
        **kw,
    )


GOOD = "The `charge_card` function bills the stored card [C1]. It is retried by `retry_failed` [C2]."
UNCITED = "The `charge_card` function bills the card. The `retry_failed` function calls it again. The `Subscription` model tracks a plan."


# --------------------------------------------------------------------------
# The streaming validator (§6.4)
# --------------------------------------------------------------------------


def test_text_is_emitted_immediately_validation_waits():
    """§6.4: "text renders immediately, ✓/⚠ land at sentence close".

    Buffering the display would spend the TTFT budget §8.2 gates at 1.2s p50 on
    a sentence boundary the reader never asked for.
    """
    validator = StreamingValidator(handle_map={"C1": "uid_1"})
    events = list(validator.feed("The `charge_card` function bil"))

    assert [type(e) for e in events] == [TextEvent]
    assert events[0].text == "The `charge_card` function bil"

    closing = list(validator.feed("ls the card [C1]. "))
    assert any(isinstance(e, SentenceEvent) for e in closing)


def test_sentence_closes_only_after_a_trailing_handle():
    """`... here. [C3]` cites the sentence it follows.

    Closing at the full stop would evaluate the sentence as uncited a moment
    before its citation arrives — a ⚠ that appears and then corrects itself,
    which is worse than a slightly later ✓.
    """
    validator = StreamingValidator(handle_map={"C3": "uid_3"})
    list(validator.feed("The `charge_card` function bills the card."))
    assert validator.sentences == [], "closed before the handle could arrive"

    events = [e for e in validator.feed(" [C3] Next.") if isinstance(e, SentenceEvent)]
    assert events[0].cited and events[0].handles == (3,)


def test_close_flushes_an_unterminated_tail():
    """Models routinely stop without a full stop.

    Dropping the tail removes a claim from T2's denominator, which raises
    Coverage by discarding evidence — the most flattering possible bug.
    """
    validator = StreamingValidator(handle_map={})
    list(validator.feed("The `charge_card` function bills the card"))
    tail = list(validator.close())

    assert len(tail) == 1
    assert tail[0].is_claim and not tail[0].cited


def test_terminator_inside_a_fence_does_not_close_a_sentence():
    validator = StreamingValidator(handle_map={"C1": "uid_1"})
    events = list(validator.feed("Here [C1].\n\n```py\nx = 1. + 2.\n"))
    closed = [e for e in events if isinstance(e, SentenceEvent)]
    assert [e.text for e in closed] == ["Here [C1]."]


def test_decimal_numbers_do_not_close_a_sentence():
    validator = StreamingValidator(handle_map={})
    events = [e for e in validator.feed("The rate is 3.14 per call. ") if isinstance(e, SentenceEvent)]
    assert [e.text for e in events] == ["The rate is 3.14 per call."]


@pytest.mark.parametrize(
    "text, handle_map, expected",
    [
        ("The `charge_card` function bills it [C1]. ", {"C1": "u"}, "✓"),
        ("The `charge_card` function bills it. ", {"C1": "u"}, "⚠ uncited"),
        ("The `charge_card` function bills it [C9]. ", {"C1": "u"}, "⚠ unknown citation"),
        ("Which function bills it? ", {"C1": "u"}, ""),
    ],
)
def test_badges(text, handle_map, expected):
    """§12.4: the trust model rests on these markers.

    A non-claim gets no badge rather than a tick — ticking prose that asserted
    nothing teaches the reader that a tick means very little.
    """
    validator = StreamingValidator(handle_map=handle_map)
    events = [e for e in validator.feed(text) if isinstance(e, SentenceEvent)]
    assert events[0].badge == expected


def test_validator_handles_chunk_boundaries_anywhere():
    """A chunk boundary can fall inside a handle, a fence marker, or a word."""
    text = "The `charge_card` function bills it [C1].\n\n```py\nv = a[C2]\n```\n"
    for size in (1, 2, 3, 5, 13):
        validator = StreamingValidator(handle_map={"C1": "u"})
        for start in range(0, len(text), size):
            list(validator.feed(text[start : start + size]))
        list(validator.close())
        cited = [s for s in validator.sentences if s.cited]
        assert [s.handles for s in cited] == [(1,)], f"chunk size {size}"


# --------------------------------------------------------------------------
# The prompt (§6.1)
# --------------------------------------------------------------------------


def test_prompt_carries_the_three_instructions():
    """Each maps to a validator; a missing one silently disables it."""
    assert "must end with one or more context handles" in SYSTEM
    assert "state the uncertainty and do not cite" in SYSTEM
    assert "⟨N lines elided⟩" in SYSTEM


def test_prompt_includes_the_packing_report(tok):
    """§5.4's report reaches the model, not only the UI.

    It is what makes "the context does not contain the answer" a conclusion the
    model can reach rather than a gap it fills.
    """
    context = pack([block("charge_card", 1.0)], [], tok, available=4_000)
    prompt = build_prompt("what charges a card", context)

    assert "what charges a card" in prompt
    assert "[C1]" in prompt
    assert context.report() in prompt


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def test_a_question_returns_a_cited_answer():
    result = service([GOOD]).answer("what charges the card")

    assert result.text == GOOD
    assert result.report.coverage == 1.0
    assert result.report.handle_errors == []
    assert result.routed.query_class is QueryClass.SEMANTIC
    assert result.ttft_seconds is not None and result.ttft_seconds >= 0
    assert result.total_seconds >= result.ttft_seconds


def test_handles_in_the_prompt_match_the_handles_the_answer_may_cite():
    """The map the model is shown and the map T1 checks are the same object."""
    svc = service([GOOD])
    result = svc.answer("what charges the card")

    _system, prompt = svc.llm.calls[0]
    for handle in result.context.handles:
        assert f"[{handle}]" in prompt


def test_structural_question_is_routed_before_retrieval():
    result = service([GOOD]).answer("what calls `charge_card`")
    assert result.routed.query_class is QueryClass.STRUCTURAL


# --------------------------------------------------------------------------
# The retry (§6.3)
# --------------------------------------------------------------------------


def test_low_coverage_triggers_exactly_one_retry():
    svc = service([UNCITED, GOOD])
    result = svc.answer("what charges the card")

    assert result.retried
    assert len(svc.llm.calls) == 2, "the retry did not run, or ran more than once"
    assert result.text == GOOD
    assert result.report.coverage == 1.0
    assert metrics.get("ground.retry") == 1


def test_retry_runs_once_even_when_it_does_not_help():
    """§6.3: "never a loop". A second failure surfaces honestly."""
    svc = service([UNCITED, UNCITED])
    result = svc.answer("what charges the card")

    assert len(svc.llm.calls) == 2
    assert result.report.coverage < 0.6
    assert result.report.retry_count == 1


def test_a_good_first_answer_is_not_retried():
    svc = service([GOOD])
    svc.answer("what charges the card")
    assert len(svc.llm.calls) == 1
    assert metrics.get("ground.retry") == 0


def test_retry_prompt_says_why():
    """§8.2 runs at temperature 0.

    An identical prompt would produce an identical answer, so the retry would
    cost a full generation to change nothing.
    """
    svc = service([UNCITED, GOOD])
    svc.answer("what charges the card")

    _system, retry_prompt = svc.llm.calls[1]
    assert "not sufficiently grounded" in retry_prompt
    assert "# Why" in retry_prompt


def test_retry_promotes_dropped_blocks_into_the_second_context(tok):
    """§6.3's ordering, at the pipeline level.

    The first pack drops blocks for budget; the retry promotes them so the
    second attempt sees evidence the first never did.
    """
    seeds = [block(f"s{i}", score=1.0 - i / 100, body_lines=120) for i in range(6)]
    svc = service([UNCITED, GOOD], seeds=seeds, neighbors=[], available=1_200)

    first_context_uids = None

    for _event, result in svc.stream("what charges the card"):
        if first_context_uids is None and result.context.dropped:
            first_context_uids = {p.uid for p in result.context.blocks}

    assert result.retried
    assert first_context_uids is not None, "nothing was dropped; the case is untested"
    assert {p.uid for p in result.context.blocks} != first_context_uids


# --------------------------------------------------------------------------
# Cost and latency (§8.2)
# --------------------------------------------------------------------------


def test_cost_is_recorded_per_answer():
    """§8.2 records `$/query` p50/p95 **and** p50 packed tokens.

    §5.4: "What determines actual cost is p50 packed tokens, not the ceiling."
    Both are on the result so the p50 is measured rather than estimated.
    """
    result = service([GOOD]).answer("what charges the card")

    assert result.cost is not None
    assert result.cost.model == LLM_MODEL
    assert result.cost.packed_tokens == result.context.packed_tokens
    assert result.cost.usd > 0


def test_cost_arithmetic_matches_the_rate_card():
    """Decision 0002: Haiku 4.5 at $1.00 in / $5.00 out per Mtok."""
    cost = AnswerCost(packed_tokens=1_000_000, output_tokens=0)
    assert cost.usd == pytest.approx(1.00)

    cost = AnswerCost(packed_tokens=0, output_tokens=1_000_000)
    assert cost.usd == pytest.approx(5.00)


def test_cost_at_the_ceiling_exceeds_the_gate():
    """F-003, as arithmetic rather than prose.

    §5.4's break-even ignores output tokens; §8.2's gate does not. At the full
    ceiling with `RESERVED_OUT` of output, Haiku 4.5 lands above $0.02 — which
    is why decision 0002's lever is "instrument p50, do not clamp yet".
    """
    from config import AVAILABLE, GATE_COST_P50_USD, RESERVED_OUT

    ceiling = AnswerCost(packed_tokens=AVAILABLE, output_tokens=RESERVED_OUT)
    assert ceiling.usd > GATE_COST_P50_USD

    typical = AnswerCost(packed_tokens=9_000, output_tokens=400)
    assert typical.usd < GATE_COST_P50_USD


# --------------------------------------------------------------------------
# SSE and HTTP
# --------------------------------------------------------------------------


def test_sse_frames_are_well_formed():
    frame = sse(TextEvent(text="hello"))
    assert frame.startswith("data: ") and frame.endswith("\n\n")
    assert json.loads(frame[len("data: ") :])["text"] == "hello"

    sentence = sse(
        SentenceEvent(
            text="The `charge_card` function bills it [C1].",
            is_claim=True, cited=True, handles=(1,), valid_handles=True, reason="handle",
        )
    )
    payload = json.loads(sentence[len("data: ") :])
    assert payload["badge"] == "✓" and payload["handles"] == [1]


def test_ask_endpoint_streams_events():
    from fastapi.testclient import TestClient

    app = create_app(service([GOOD]))
    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "what charges the card"})

    assert response.status_code == 200
    frames = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert any(f["type"] == "text" for f in frames)
    assert any(f["type"] == "sentence" and f["badge"] == "✓" for f in frames)
    assert frames[-1]["type"] == "done"
    assert "Coverage" in frames[-1]["report"]


def test_rename_endpoint_reaches_the_indexer():
    """§4.4 Signal A's HTTP binding — `POST /index/rename`.

    Step 5 built the event boundary; this is the transport §4.4 names, and the
    route the VS Code extension posts to.
    """
    from fastapi.testclient import TestClient

    class SpyIndexer:
        def __init__(self):
            self.renames = []

        def on_rename(self, repo, old, new):
            self.renames.append((repo, old, new))

    indexer = SpyIndexer()
    app = create_app(service([GOOD]), indexer=indexer)
    with TestClient(app) as client:
        response = client.post(
            "/index/rename",
            json={"repo": "r", "old_path": "a.py", "new_path": "b.py"},
        )

    assert response.json() == {"accepted": True}
    assert indexer.renames == [("r", "a.py", "b.py")]


def test_rename_endpoint_without_an_indexer_degrades():
    from fastapi.testclient import TestClient

    app = create_app(service([GOOD]))
    with TestClient(app) as client:
        response = client.post(
            "/index/rename", json={"repo": "r", "old_path": "a.py", "new_path": "b.py"}
        )
    assert response.status_code == 200
    assert response.json()["accepted"] is False


def test_healthz_names_the_model():
    """§8.2: without the model name and version, `$/query` cannot be checked."""
    from fastapi.testclient import TestClient

    app = create_app(service([GOOD]))
    with TestClient(app) as client:
        assert client.get("/healthz").json() == {"ok": True, "model": LLM_MODEL}
