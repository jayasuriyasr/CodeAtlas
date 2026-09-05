"""Step 9 — T1, T2, T3 and §6.2's three signals.

§6.2 keeps the three separate deliberately: "Coverage 0.9 / RelationPrec 0.4 is
'well-sourced but misreading the code' — a different bug from 0.4 / 0.95, 'right
but improvising.' One blended number erases the distinction." So each is tested
on its own, and one test checks that they genuinely move independently.
"""

from __future__ import annotations

import pytest

import metrics
from config import COVERAGE_RETRY_THRESHOLD, MAX_RETRIES
from ground.validate import (
    RelationClaim,
    extract_relation_claims,
    ground,
    plan_retry,
    should_retry,
    validate_handles,
    verify_relations,
)
from ground.classify import classify


@pytest.fixture(autouse=True)
def reset_metrics():
    metrics.reset()
    yield
    metrics.reset()


HANDLES = {f"C{i}": f"uid_{i}" for i in range(1, 10)}


class StubRunner:
    """Answers the T3 query from a set of known edges and symbols."""

    def __init__(self, symbols=(), edges=()):
        self.symbols = set(symbols)
        self.edges = set(edges)          # (subject, relation, object)
        self.queries: list[str] = []

    def run(self, query, **params):
        self.queries.append(query)
        rows = []
        for claim in params["claims"]:
            rows.append(
                {
                    "idx": claim["idx"],
                    "subjects": int(claim["subject"] in self.symbols),
                    "objects": int(claim["object"] in self.symbols),
                    "edges": int(
                        (claim["subject"], claim["relation"], claim["object"])
                        in self.edges
                    ),
                }
            )
        return rows


# --------------------------------------------------------------------------
# T1
# --------------------------------------------------------------------------


def test_hallucinated_handle_caught():
    """`[C12]` against nine blocks: T1 flags it.

    A dict lookup, which is why §6.1 can run it during the stream. §5.4's
    after-packing handle assignment is what makes the verdict trustworthy —
    `C1..Cn` has no gaps, so an out-of-range handle is genuinely invented.
    """
    answer = classify("The `charge_card` function bills the card [C12].")
    errors = validate_handles(answer, HANDLES)

    assert [e.handle for e in errors] == [12]
    assert metrics.get("ground.t1.hallucinated_handle") == 1


def test_valid_handles_pass_t1():
    answer = classify("The `charge_card` function bills the card [C3].")
    assert validate_handles(answer, HANDLES) == []


def test_handle_inside_a_fence_is_not_checked_by_t1():
    """v9.1 #9 again, from T1's side.

    `matrix[C99]` in a snippet is an array index. Flagging it would report the
    model for hallucinating a citation it never made.
    """
    answer = classify(
        "Here is the shape [C1].\n\n```python\nvalue = matrix[C99]\n```\n"
    )
    assert validate_handles(answer, HANDLES) == []


# --------------------------------------------------------------------------
# T2 / Coverage
# --------------------------------------------------------------------------


def test_uncited_claim_flagged():
    """T2. A declarative sentence about code with no handle drags Coverage."""
    report = ground(
        "The `charge_card` function bills the card [C1]. "
        "The `retry_failed` function calls it again.",
        HANDLES,
    )
    assert len(report.answer.claims) == 2
    assert len(report.uncited_claims) == 1
    assert report.coverage == pytest.approx(0.5)


def test_coverage_is_one_when_everything_is_cited():
    report = ground(
        "The `charge_card` function bills the card [C1]. "
        "The `Subscription` model tracks the plan [C2].",
        HANDLES,
    )
    assert report.coverage == 1.0


def test_an_answer_with_no_claims_scores_one_not_zero():
    """"I could not find anything relevant" is the answer that most deserves
    to stand.

    Scoring it 0.0 would fire §6.3's retry on it, spend a second generation,
    and most likely produce a worse, less hedged answer the second time.
    """
    report = ground("I could not find anything relevant in the retrieved context.", HANDLES)
    assert report.answer.claims == []
    assert report.coverage == 1.0
    assert not should_retry(report)


def test_hedged_sentences_stay_out_of_the_denominator():
    """§6.1's exclusion, measured where it matters.

    Both sentences mention code; one hedges. If the hedge counted, Coverage
    would read 0.5 and the retry would fire on an answer whose only flaw was
    declaring its own uncertainty.
    """
    report = ground(
        "The `charge_card` function bills the card [C1]. "
        "It might also retry on failure.",
        HANDLES,
    )
    assert len(report.answer.claims) == 1
    assert report.coverage == 1.0


# --------------------------------------------------------------------------
# T3 — relations
# --------------------------------------------------------------------------


def test_relation_claims_are_extracted():
    answer = classify(
        "`SubscriptionView.post` calls `charge_card` [C1]. "
        "`billing.tasks` imports `common.utils` [C2]. "
        "`charge_card` is defined in `billing.tasks` [C3]. "
        "The route `subscription` dispatches to `SubscriptionView` [C4]."
    )
    claims = extract_relation_claims(answer)
    triples = {(c.subject, c.relation, c.object) for c in claims}

    assert ("SubscriptionView.post", "CALLS", "charge_card") in triples
    assert ("billing.tasks", "IMPORTS", "common.utils") in triples
    assert ("billing.tasks", "DEFINES", "charge_card") in triples, "inverted form"
    assert ("subscription", "DISPATCHES_TO", "SubscriptionView") in triples


def test_inverted_phrasing_is_normalised():
    """"`A` is called by `B`" asserts B -> A, not A -> B.

    Storing it the wrong way round would make every passive sentence a
    ConflictFlag, and §6.3 would retry on correct answers.
    """
    answer = classify("`charge_card` is called by `SubscriptionView.post` [C1].")
    claims = extract_relation_claims(answer)
    assert (claims[0].subject, claims[0].object) == (
        "SubscriptionView.post",
        "charge_card",
    )


def test_relation_claims_only_come_from_claim_sentences():
    """§6.2's denominator is *asserted* relations.

    A hedged or interrogative sentence asserts nothing, so counting it would
    make RelationPrec measure the model's speculation rather than its claims.
    """
    answer = classify(
        "`A` might call `B`. Does `C` call `D`? `E` calls `F` [C1]."
    )
    triples = {(c.subject, c.object) for c in extract_relation_claims(answer)}
    assert triples == {("E", "F")}


def test_relation_verified_batched():
    """T3, one query for the whole answer (§6.1's tier table).

    Per-claim queries would turn a ~40ms post-stream check into N round trips,
    on the path §8.2 gates at 2.5s p95.
    """
    runner = StubRunner(
        symbols={"a", "b", "c"}, edges={("a", "CALLS", "b")}
    )
    claims = [
        RelationClaim("a", "CALLS", "b", 0),
        RelationClaim("b", "CALLS", "c", 1),
        RelationClaim("c", "IMPORTS", "a", 2),
    ]
    verdicts = verify_relations(runner, "repo", claims)

    assert len(runner.queries) == 1, f"{len(runner.queries)} queries for 3 claims"
    assert [v.verified for v in verdicts] == [True, False, False]
    assert metrics.get("ground.t3.queries") == 1


def test_relation_precision_is_verified_over_asserted():
    runner = StubRunner(symbols={"a", "b", "c"}, edges={("a", "CALLS", "b")})
    report = ground(
        "`a` calls `b` [C1]. `b` calls `c` [C2].",
        HANDLES,
        runner=runner,
        repo_id="repo",
    )
    assert report.relation_precision == pytest.approx(0.5)


def test_contradiction_requires_both_endpoints_to_exist():
    """§12.3: the graph deliberately cannot see string-based references.

    `ForeignKey('app.Model')`, `AUTH_USER_MODEL`, `include()`, Celery task
    names — none is an AST edge. An unresolvable endpoint means "the graph does
    not know", not "the model is wrong". Counting it as a conflict would fire
    §6.3's retry on answers that were right.
    """
    known_both = StubRunner(symbols={"a", "b"}, edges=set())
    contradicted = ground("`a` calls `b` [C1].", HANDLES, runner=known_both, repo_id="r")
    assert contradicted.conflict_flags == 1
    assert contradicted.relation_precision == 0.0

    unknown_target = StubRunner(symbols={"a"}, edges=set())
    unverified = ground("`a` calls `b` [C1].", HANDLES, runner=unknown_target, repo_id="r")
    assert unverified.conflict_flags == 0, "an unknown symbol is not a contradiction"
    assert unverified.relation_precision == 0.0, "still unverified, though"


def test_t3_is_skipped_without_a_graph_runner():
    """A report without T3 is still a useful T1/T2 report.

    Reporting relations as verified when nothing verified them would be worse
    than saying nothing about them.
    """
    report = ground("`a` calls `b` [C1].", HANDLES)
    assert report.relation_verdicts == []
    assert report.conflict_flags == 0
    assert metrics.get("ground.t3.skipped") == 1


def test_no_relation_claims_means_no_query():
    runner = StubRunner()
    ground("The `charge_card` function bills the card [C1].", HANDLES, runner=runner)
    assert runner.queries == []


def test_relation_type_must_be_in_the_allowlist():
    """The relation reaches Cypher as a *value*, never as query text.

    §11.2 3b sets the rule for writes; the same reasoning applies to a read
    whose input is model output.
    """
    with pytest.raises(ValueError, match="allowlist|EDGE_TYPE"):
        RelationClaim("a", "DROP_EVERYTHING", "b", 0)


# --------------------------------------------------------------------------
# §6.2 — the signals stay separate
# --------------------------------------------------------------------------


def test_coverage_and_relation_precision_move_independently():
    """§6.2's whole argument for keeping them apart.

    High Coverage with low RelationPrec is "well-sourced but misreading the
    code"; the reverse is "right but improvising". A blended score would show
    the same middling number for both.
    """
    runner = StubRunner(symbols={"a", "b"}, edges=set())
    well_sourced_wrong = ground("`a` calls `b` [C1].", HANDLES, runner=runner, repo_id="r")
    assert well_sourced_wrong.coverage == 1.0
    assert well_sourced_wrong.relation_precision == 0.0

    right_but_uncited = ground(
        "`a` calls `b` [C1]. The `retry_failed` function runs later.",
        HANDLES,
        runner=StubRunner(symbols={"a", "b"}, edges={("a", "CALLS", "b")}),
        repo_id="r",
    )
    assert right_but_uncited.coverage == pytest.approx(0.5)
    assert right_but_uncited.relation_precision == 1.0


def test_report_renders_all_three_signals():
    report = ground("The `charge_card` function bills the card [C1].", HANDLES)
    rendered = report.render()
    assert "Coverage" in rendered
    assert "RelationPrec" in rendered
    assert "ConflictFlags" in rendered


# --------------------------------------------------------------------------
# §6.3 — the retry gate
# --------------------------------------------------------------------------


def test_retry_fires_on_low_coverage():
    report = ground(
        "The `charge_card` function bills the card. "
        "The `retry_failed` function calls it again. "
        "The `Subscription` model tracks the plan.",
        HANDLES,
    )
    assert report.coverage < COVERAGE_RETRY_THRESHOLD
    assert should_retry(report)


def test_retry_fires_on_a_conflict_even_at_full_coverage():
    """§6.3's condition is `Coverage < 0.6 or ConflictFlags > 0`.

    A perfectly cited answer that contradicts the graph is the "well-sourced but
    misreading the code" case, and citations alone would never catch it.
    """
    runner = StubRunner(symbols={"a", "b"}, edges=set())
    report = ground("`a` calls `b` [C1].", HANDLES, runner=runner, repo_id="r")
    assert report.coverage == 1.0
    assert report.conflict_flags == 1
    assert should_retry(report)


def test_retry_runs_once_only():
    """§6.3: "One retry, bounded, never a loop."

    A second attempt doubles cost and latency on an answer the first retry
    already failed to improve — and §6.3's own fallback is to surface honestly,
    not to keep trying.
    """
    text = "The `charge_card` function bills the card. The `retry_failed` function runs."
    first = ground(text, HANDLES, retry_count=0)
    second = ground(text, HANDLES, retry_count=MAX_RETRIES)

    assert should_retry(first)
    assert not should_retry(second)


def test_retry_promotes_dropped_first():
    """§6.3's ordering, not a heuristic.

    "If the packing report shows dropped blocks, the retry promotes those before
    expanding — unseen evidence is a likelier explanation for low coverage than
    an insufficient neighborhood." Expanding first spends a graph traversal to
    find context that was already retrieved and thrown away.
    """
    report = ground("The `charge_card` function bills the card.", HANDLES)
    plan = plan_retry(report, ["uid_dropped_a", "uid_dropped_b"], HANDLES)

    assert plan.promote == ("uid_dropped_a", "uid_dropped_b")
    assert plan.expand_around == ()
    assert "promoted before expanding" in plan.reason


def test_retry_expands_when_nothing_was_dropped():
    """With no unseen evidence, the neighbourhood is the remaining explanation."""
    runner = StubRunner(symbols={"a", "b"}, edges=set())
    report = ground("`a` calls `b` [C1].", HANDLES, runner=runner, repo_id="r")
    plan = plan_retry(report, [], HANDLES)

    assert plan.promote == ()
    assert plan.expand_around == ("uid_1",), "expansion centres on the conflicting span"
    assert "expansion" in plan.reason
