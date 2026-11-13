"""A gate row says what its number was measured on.

The T2 classifier rows read "measured", beside rows measured on the real
pipeline. They were measured on the golden set's reference answers — sentences
the author wrote — with the gold symbols handed to the classifier as known
names. That is a check that the classifier agrees with its author on clean
prose, not its accuracy on model output, and the row now says so.
"""

from __future__ import annotations

from eval.gates import CLASSIFIER_SOURCE, build_gates


def test_the_classifier_rows_name_their_source():
    gates = {g.name: g for g in build_gates(
        classifier_precision=0.98, classifier_recall=0.96, recall_at_10=0.9, source="measured")}
    for name in ("T2 classifier precision", "T2 classifier recall"):
        assert gates[name].source == CLASSIFIER_SOURCE
        assert CLASSIFIER_SOURCE in gates[name].render()
    assert gates["Recall@10 (packed context)"].source == "measured"


def test_the_label_says_what_the_sentences_were():
    assert "reference" in CLASSIFIER_SOURCE and "not model output" in CLASSIFIER_SOURCE


def test_a_caller_may_still_name_another_source():
    gates = {g.name: g for g in build_gates(
        classifier_precision=0.9, classifier_recall=0.9, source="measured",
        classifier_source="annotated model answers")}
    assert gates["T2 classifier precision"].source == "annotated model answers"
