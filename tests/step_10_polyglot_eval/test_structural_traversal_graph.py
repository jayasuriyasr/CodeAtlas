"""The STRUCTURAL traversal scored against the answer it should give.

Before: 13 of the 15 STRUCTURAL golden questions had the named symbol itself as
their only gold, so "Recall 1.00" there meant the entity was found, not that
"what calls X" returned X's callers. `expected_related` now holds the full
answer, read from the fixture's source and cross-checked with Python's `ast`,
and `eval/structural.py` scores the traversal against it.

The numbers are written to `eval/STRUCTURAL_REPORT.md`. The assertions pin
what this change set guarantees and what it knowingly does not do.
"""

from __future__ import annotations

import json

import config
from eval.structural import measure_traversal

from .pipeline_support import PY_REPO, ask

GOLDEN = config.REPO_ROOT / "eval" / "golden" / "golden_45.json"
REPORT = config.REPO_ROOT / "eval" / "STRUCTURAL_REPORT.md"


def measured(indexed):
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    return measure_traversal(golden["questions"], lambda q: ask(indexed, PY_REPO, q)[3])


def test_every_structural_question_has_an_expected_answer():
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    structural = [q for q in golden["questions"] if q["class"] == "STRUCTURAL"]
    assert len(structural) == 15
    assert all(q.get("expected_related") for q in structural)


def test_the_traversal_is_measured_and_recorded(indexed):
    report = measured(indexed)
    REPORT.write_text(
        "<!-- Written by tests/step_10_polyglot_eval/test_structural_traversal_graph.py -->\n\n"
        + report.render() + "\n", encoding="utf-8")

    assert len(report.results) == 15
    assert report.precision is not None and report.recall is not None
    # Regression floors, below the values measured when this was written
    # (see eval/STRUCTURAL_REPORT.md for the actual figures).
    assert report.precision >= 0.8, report.render()
    assert report.recall >= 0.6, report.render()


def test_what_the_inheritance_work_guarantees(indexed):
    by_id = {r.id: r for r in measured(indexed).results}
    assert not by_id["S07"].missed, "subclasses of TimestampMixin, through INHERITS"
    assert not by_id["S14"].missed, "callers of soft_delete, through self."


def test_what_it_knowingly_does_not_do(indexed):
    """`token.revoke()` calls a method on a local variable. Resolving it needs
    the variable's type, which SEI does not infer — so it is missed, and the
    miss is counted rather than hidden."""
    by_id = {r.id: r for r in measured(indexed).results}
    assert by_id["S15"].missed == {
        "authx.services.revoke_all", "authx.views.TokenRevokeView.delete"}
