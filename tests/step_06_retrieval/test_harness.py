"""Step 6 — the eval harness's arithmetic, and the golden file's shape.

S7 computes every number §8.2 gates on. A harness that miscounts does not fail;
it reports a different number, and the gate then passes or blocks for the wrong
reason. So the arithmetic is tested directly, against hand-checkable cases,
with no database and no retrieval involved.
"""

from __future__ import annotations

import pytest

from eval.harness import EvalHarness, Question, load_questions


def question(qid: str, gold, router_class: str = "SEMANTIC") -> Question:
    return Question(
        id=qid, query=f"query {qid}", router_class=router_class, gold=tuple(gold)
    )


def harness(ranked_by_qid: dict[str, list[str]], corpus_uids: dict[tuple[str, str], list[str]]):
    return EvalHarness(
        resolve_uids=lambda rel, qn: corpus_uids.get((rel, qn), []),
        retrieve=lambda q: ranked_by_qid.get(q.id, []),
        stage="unit",
    )


# --------------------------------------------------------------------------
# Recall and MRR
# --------------------------------------------------------------------------


def test_recall_counts_a_question_once_however_many_gold_it_has():
    """Recall is over questions, not over gold symbols.

    A question with three gold symbols is one question. Counting per symbol
    would let a question with a large gold set dominate the score.
    """
    corpus = {("a.py", "a.one"): ["u1"], ("a.py", "a.two"): ["u2"]}
    report = harness({"q1": ["u2"]}, corpus).run(
        [question("q1", [("a.py", "a.one"), ("a.py", "a.two")])]
    )
    assert report.recall_at(10) == 1.0
    assert len(report.scoreable) == 1


def test_recall_at_k_respects_the_cutoff():
    corpus = {("a.py", "a.one"): ["gold"]}
    ranked = ["x"] * 9 + ["gold"]          # gold at rank 10
    report = harness({"q1": ranked}, corpus).run([question("q1", [("a.py", "a.one")])])

    assert report.recall_at(10) == 1.0
    assert report.recall_at(5) == 0.0
    assert report.recall_at(1) == 0.0


def test_mrr_is_the_reciprocal_of_the_first_hit():
    corpus = {("a.py", "a.one"): ["gold"]}
    reports = {
        rank: harness({"q1": ["x"] * (rank - 1) + ["gold"]}, corpus).run(
            [question("q1", [("a.py", "a.one")])]
        )
        for rank in (1, 2, 4)
    }
    assert reports[1].mrr() == pytest.approx(1.0)
    assert reports[2].mrr() == pytest.approx(0.5)
    assert reports[4].mrr() == pytest.approx(0.25)


def test_a_complete_miss_scores_zero_not_an_error():
    corpus = {("a.py", "a.one"): ["gold"]}
    report = harness({"q1": ["x", "y"]}, corpus).run([question("q1", [("a.py", "a.one")])])
    assert report.recall_at(10) == 0.0
    assert report.mrr() == 0.0
    assert [r.question.id for r in report.misses()] == ["q1"]


def test_empty_result_list_is_a_miss_not_a_crash():
    corpus = {("a.py", "a.one"): ["gold"]}
    report = harness({"q1": []}, corpus).run([question("q1", [("a.py", "a.one")])])
    assert report.recall_at(10) == 0.0


# --------------------------------------------------------------------------
# Unresolvable gold
# --------------------------------------------------------------------------


def test_unresolvable_gold_is_excluded_not_counted_as_a_miss():
    """A typo in the golden file must not read as a retrieval regression.

    Counting it as a miss would make the golden set's own errors look like
    retrieval getting worse — and §8.2 says two questions moving is a block.
    """
    corpus = {("a.py", "a.one"): ["gold"]}
    report = harness({"q1": ["gold"], "q2": ["x"]}, corpus).run(
        [
            question("q1", [("a.py", "a.one")]),
            question("q2", [("a.py", "a.typo")]),
        ]
    )
    assert len(report.results) == 2
    assert len(report.scoreable) == 1
    assert report.recall_at(10) == 1.0, "the unresolvable question must not drag the score"
    assert report.results[1].unresolved_gold == ("a.py::a.typo",)


def test_unresolvable_gold_is_visible_in_the_report():
    """Excluded, but never silently."""
    report = harness({"q1": []}, {}).run([question("q1", [("a.py", "a.gone")])])
    assert "gold not present in the corpus" in report.render()
    assert "a.py::a.gone" in report.render()


# --------------------------------------------------------------------------
# Per-class reporting and resolution
# --------------------------------------------------------------------------


def test_per_class_recall_is_reported_separately():
    """§8.2 reports per-class scores.

    A blended number hides the case §8.1 is built to expose: SEMANTIC questions
    on undocumented code are where retrieval is weakest (§12.5), and averaging
    them with STRUCTURAL hits conceals it.
    """
    corpus = {("a.py", f"a.s{i}"): [f"u{i}"] for i in range(4)}
    ranked = {"q0": ["u0"], "q1": ["u1"], "q2": ["x"], "q3": ["x"]}
    report = harness(ranked, corpus).run(
        [
            question("q0", [("a.py", "a.s0")], "STRUCTURAL"),
            question("q1", [("a.py", "a.s1")], "STRUCTURAL"),
            question("q2", [("a.py", "a.s2")], "SEMANTIC"),
            question("q3", [("a.py", "a.s3")], "SEMANTIC"),
        ]
    )
    assert report.by_class(10) == {"STRUCTURAL": 1.0, "SEMANTIC": 0.0}
    assert report.recall_at(10) == 0.5, "the blend hides the SEMANTIC failure"


def test_resolution_is_reported_with_the_score():
    """§8.2: "per-class scores are reported with the resolution stated".

    At 15 questions one flip is ~6.7 points, which is comparable to the gaps
    between several of the thresholds. A score printed without that is a score
    that invites over-reading.
    """
    corpus = {("a.py", f"a.s{i}"): [f"u{i}"] for i in range(15)}
    report = harness({}, corpus).run(
        [question(f"q{i}", [("a.py", f"a.s{i}")]) for i in range(15)]
    )
    assert report.resolution_pp == pytest.approx(6.667, abs=0.01)
    assert "6.7pp" in report.render()


def test_empty_report_does_not_divide_by_zero():
    report = harness({}, {}).run([])
    assert report.recall_at(10) == 0.0
    assert report.mrr() == 0.0
    assert report.resolution_pp == 0.0


def test_render_records_the_embedding_model():
    """§8.2: without the model name and version, the gate cannot be checked."""
    report = EvalHarness(
        lambda r, q: ["u"], lambda q: ["u"], stage="unit", embedding_model="voyage-code-2"
    ).run([question("q1", [("a.py", "a.one")])])
    assert "voyage-code-2" in report.render()

    unnamed = EvalHarness(lambda r, q: ["u"], lambda q: ["u"], stage="unit").run(
        [question("q1", [("a.py", "a.one")])]
    )
    assert "UNRECORDED" in unnamed.render()


def test_stage_is_recorded_so_the_two_measurements_are_never_confused():
    """§8.1 measures recall against **packed** context; step 6 measures against
    retrieved candidates, which is strictly more lenient. Recording the stage is
    what stops the easier number being compared to the gate."""
    report = EvalHarness(
        lambda r, q: ["u"], lambda q: ["u"], stage="retrieved"
    ).run([question("q1", [("a.py", "a.one")])])
    assert "stage=retrieved" in report.render()


# --------------------------------------------------------------------------
# The golden file itself
# --------------------------------------------------------------------------


def test_seed_set_is_well_formed():
    questions, meta = load_questions()

    assert len(questions) >= 15, "plan §6 asks for ~15 seed questions"
    assert len({q.id for q in questions}) == len(questions), "duplicate ids"
    assert meta["corpus"] == "tests/fixtures/repos/django_min"
    assert meta["stage"] == "retrieved"

    for q in questions:
        assert q.query.strip(), q.id
        assert q.gold, f"{q.id} has no gold"
        for rel_path, qualified_name in q.gold:
            assert rel_path.endswith(".py"), f"{q.id}: {rel_path}"
            assert "." in qualified_name, f"{q.id}: {qualified_name}"


def test_seed_set_gold_is_named_not_hashed():
    """Gold is `(rel_path, qualified_name)`, resolved to UIDs at run time.

    A UID is a truncated sha1 over `repo_id`; hard-coding one would make the
    golden file unreviewable in a diff and would break on a repo rename — for a
    corpus whose whole purpose is to be stable across runs.
    """
    questions, _meta = load_questions()
    for q in questions:
        for _rel, qualified_name in q.gold:
            assert not (len(qualified_name) == 20 and qualified_name.isalnum()), (
                f"{q.id} looks like a raw UID"
            )


def test_seed_set_covers_the_undocumented_case():
    """§8.1: >=5 of 15 SEMANTIC questions must target undocumented or
    poorly-named symbols (§12.5) — the eval measures the weakness rather than
    avoiding it. The seed set is a third of the final size, so this checks the
    intent is present rather than the full quota."""
    questions, raw = load_questions()
    annotated = [q for q in raw["questions"] if q.get("note")]
    assert annotated, "no question records why it was chosen"
    assert any("undocumented" in q["note"].lower() or "no docstring" in q["note"].lower()
               for q in annotated)
