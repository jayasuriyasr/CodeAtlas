"""The answer pipeline's accounting and degradation, as the audit found them.

* TTFT is the first token the reader sees — a §6.3 retry must not overwrite it.
* `$/query` is what was billed: both generations of a retried answer.
* §6.3's "bounded expansion around uncited/conflicting spans" actually expands,
  and the retry prompt does not claim new context when there is none.
* §9.3's "LLM provider" row: retrieval-only, not a crash.
* Retrieval notes (route, frame badges, degraded search) reach the stream.
"""

from __future__ import annotations

import time

import pytest

import metrics
from adapters.base import Symbol
from api.app import AnswerCost, InvestigationService
from api.prompt import RETRY_PREFIX, RETRY_PREFIX_SAME_CONTEXT
from index.providers import ApproxCodeTokenizer, ScriptedLLM
from pack.render import Block

GOOD = "The `charge_card` function bills the stored card [C1]. It is retried by `retry_failed` [C2]."
UNCITED = (
    "The `charge_card` function bills the card. The `retry_failed` function calls it "
    "again. The `Subscription` model tracks a plan."
)


@pytest.fixture(autouse=True)
def reset():
    metrics.reset()
    yield
    metrics.reset()


def symbol(name: str) -> Symbol:
    return Symbol(
        uid=f"uid_{name}", repo_id="repo", rel_path=f"billing/{name}.py",
        qualified_name=f"billing.{name}.{name}", name=name, arity=1, ordinal=0,
        kind="function", signature=f"def {name}(user):", docstring=None,
        source_code=f"def {name}(user):\n    return compute(user)\n",
        enclosing_signature=None,
    )


def block(name: str, score: float) -> Block:
    return Block(symbol=symbol(name), score=score)


class SlowLLM(ScriptedLLM):
    """Each chunk takes `delay` seconds, so TTFT is measurable."""

    def __init__(self, *answers, delay=0.02, **kw):
        super().__init__(*answers, **kw)
        self.delay = delay

    def stream(self, system, prompt, *, max_tokens=2000, usage=None):
        for chunk in super().stream(system, prompt, max_tokens=max_tokens, usage=usage):
            time.sleep(self.delay)
            yield chunk


def service(llm, *, seeds=None, neighbors=None, expander=None, notes=None, **kw):
    seeds = seeds if seeds is not None else [block("charge_card", 1.0)]
    neighbors = neighbors if neighbors is not None else [block("retry_failed", 0.5)]

    def retriever(question, routed):
        if notes is None:
            return seeds, neighbors
        return seeds, neighbors, notes

    return InvestigationService(
        retriever=retriever, llm=llm, tokenizer=ApproxCodeTokenizer(),
        repo_id="repo", expander=expander, **kw,
    )


# -- TTFT and cost ----------------------------------------------------------


def test_ttft_is_the_first_answers_first_token_not_the_retrys():
    llm = SlowLLM(UNCITED, GOOD, chunk=10, delay=0.02)
    result = service(llm).answer("what charges the card")

    assert result.retried
    first_answer_seconds = (len(UNCITED) // 10 + 1) * 0.02
    assert result.ttft_seconds < first_answer_seconds / 2, (
        f"TTFT {result.ttft_seconds:.3f}s includes the whole first answer "
        f"(~{first_answer_seconds:.2f}s): the retry overwrote it"
    )


def test_cost_counts_both_generations_of_a_retried_answer():
    tok = ApproxCodeTokenizer()
    result = service(ScriptedLLM(UNCITED, GOOD)).answer("what charges the card")

    assert result.retried and result.cost.generations == 2
    assert result.cost.output_tokens == tok.count(UNCITED) + tok.count(GOOD)
    assert result.cost.input_tokens > 2 * result.context.packed_tokens, (
        "two prompts were sent, each carrying context, system text and question"
    )


def test_a_single_answer_is_one_generation():
    result = service(ScriptedLLM(GOOD)).answer("what charges the card")
    assert result.cost.generations == 1
    assert result.cost.output_tokens == ApproxCodeTokenizer().count(GOOD)


def test_provider_reported_usage_wins_over_local_counting():
    llm = ScriptedLLM(GOOD, usage={"input_tokens": 1234, "output_tokens": 56})
    result = service(llm).answer("what charges the card")
    assert (result.cost.input_tokens, result.cost.output_tokens) == (1234, 56)
    assert result.cost.usd == pytest.approx((1234 * 1.0 + 56 * 5.0) / 1_000_000)


def test_answer_cost_without_input_tokens_falls_back_to_packed_tokens():
    """§5.4's approximation, kept for callers that only know packed tokens."""
    assert AnswerCost(packed_tokens=1_000_000, output_tokens=0).usd == pytest.approx(1.0)
    assert AnswerCost(
        packed_tokens=1_000_000, output_tokens=0, input_tokens=2_000_000
    ).usd == pytest.approx(2.0)


# -- §6.3 retry ---------------------------------------------------------------


def test_retry_expands_around_cited_spans_when_nothing_was_dropped():
    """No dropped blocks, so §6.3 calls for expansion. It used to be computed
    and thrown away; the retry re-sent identical context."""
    asked = []

    def expander(uids):
        asked.append(list(uids))
        return [block("refund_card", 0.9)]

    first_cites_one = "The `charge_card` function bills it [C1]. It calls `x`. It calls `y`. It calls `z`."
    llm = ScriptedLLM(first_cites_one, GOOD)
    result = service(llm, expander=expander).answer("what charges the card")

    assert result.retried
    assert asked == [["uid_charge_card"]], "expansion centres on the cited span"
    assert "uid_refund_card" in {p.uid for p in result.context.blocks}
    _system, retry_prompt = llm.calls[1]
    assert retry_prompt.startswith(RETRY_PREFIX)
    retry_events = [e for e in result.events if isinstance(e, dict) and e.get("type") == "retry"]
    assert retry_events and retry_events[0]["added_blocks"] == 1


def test_retry_with_nothing_new_says_so():
    """Telling the model "additional context has been retrieved" over unchanged
    context invites it to cite evidence that is not there."""
    llm = ScriptedLLM(UNCITED, GOOD)
    service(llm, expander=None, neighbors=[]).answer("what charges the card")
    _system, retry_prompt = llm.calls[1]
    assert retry_prompt.startswith(RETRY_PREFIX_SAME_CONTEXT)
    assert "Additional context has been retrieved" not in retry_prompt


def test_a_retry_that_still_fails_is_flagged_for_review():
    result_events = list(service(ScriptedLLM(UNCITED, UNCITED)).stream("q"))
    done = result_events[-1][0]
    assert done["type"] == "done" and done["retried"] and done["needs_review"]


def test_a_good_answer_does_not_need_review():
    done = list(service(ScriptedLLM(GOOD)).stream("q"))[-1][0]
    assert done["needs_review"] is False and done["retried"] is False


# -- §9.3 LLM provider --------------------------------------------------------


def test_provider_failure_degrades_to_retrieval_only():
    llm = ScriptedLLM(GOOD, fail_after=0)
    events = [e for e, _r in service(llm).stream("what charges the card")]

    unavailable = [e for e in events if isinstance(e, dict) and e.get("type") == "synthesis_unavailable"]
    assert unavailable, "§9.3: 'Synthesis unavailable' must be surfaced"
    assert [b["qualified_name"] for b in unavailable[0]["blocks"]] == [
        "billing.charge_card.charge_card", "billing.retry_failed.retry_failed",
    ]
    done = events[-1]
    assert done["type"] == "done" and done["report"] == "Synthesis unavailable"
    assert metrics.get("synthesis.unavailable") == 1


def test_provider_dropping_mid_answer_is_not_graded():
    llm = ScriptedLLM(GOOD, fail_after=3, chunk=7)
    result = service(llm).answer("what charges the card")
    assert result.report is None, "a half answer's Coverage measures where it broke"


def test_provider_failure_on_the_retry_keeps_the_first_answer():
    class FailSecond(ScriptedLLM):
        def stream(self, system, prompt, *, max_tokens=2000, usage=None):
            if self.calls:
                self.calls.append((system, prompt))
                raise ConnectionError("down on retry")
            yield from super().stream(system, prompt, max_tokens=max_tokens, usage=usage)

    result = service(FailSecond(UNCITED)).answer("what charges the card")
    assert result.text == UNCITED
    assert result.report is not None and result.report.coverage < 0.6
    assert any(isinstance(e, dict) and e.get("during") == "retry" for e in result.events)


def test_a_validator_defect_is_not_mistaken_for_a_provider_outage(monkeypatch):
    """Only the provider's own exceptions become §9.3's degradation."""
    from api import stream as stream_module

    def broken(self, chunk):
        raise ZeroDivisionError("validator bug")
        yield  # pragma: no cover

    monkeypatch.setattr(stream_module.StreamingValidator, "feed", broken)
    with pytest.raises(ZeroDivisionError):
        service(ScriptedLLM(GOOD)).answer("q")


# -- retrieval notes ----------------------------------------------------------


def test_retrieval_notes_reach_the_stream_first():
    notes = {"route": "TRACE", "frames": [{"frame": "File x", "badge": "✓ direct"}]}
    events = [e for e, _r in service(ScriptedLLM(GOOD), notes=notes).stream("q")]
    assert events[0] == {"type": "retrieval", **notes}


def test_a_two_tuple_retriever_still_works():
    """The original `(seeds, neighbors)` contract is kept."""
    result = service(ScriptedLLM(GOOD)).answer("q")
    assert result.notes == {}
    assert not any(isinstance(e, dict) and e.get("type") == "retrieval" for e in result.events)


def test_no_context_at_all_still_answers_honestly():
    result = service(ScriptedLLM("I could not find this in the repository."),
                     seeds=[], neighbors=[]).answer("q")
    assert result.context.blocks == []
    assert result.report.coverage == 1.0 and not result.retried
