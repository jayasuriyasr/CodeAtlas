"""Step 10 — the ten gates, the four decision metrics, and the golden set.

Plan §10's gate is that "all ten gates produce values" and that the four
decision metrics are recorded. The distinction that matters here is between a
gate that *can* be computed and one that *has* been: an unmeasured gate must
render NOT MEASURED, never PASS and never 0.0, because §4 of the working rules
says record actual output, not expected.
"""

from __future__ import annotations

import json

import pytest

import config
import metrics
from eval.gates import (
    GateReport,
    build_decision_metrics,
    build_gates,
    percentile,
)
from eval.harness import load_questions

GOLDEN_45 = config.REPO_ROOT / "eval" / "golden" / "golden_45.json"


@pytest.fixture(autouse=True)
def reset_metrics():
    metrics.reset()
    yield
    metrics.reset()


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GOLDEN_45.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# §8.1 — the golden set
# --------------------------------------------------------------------------


def test_golden_set_is_45_questions_15_per_class(golden):
    """§8.1: "45 questions, 15 per router class"."""
    questions = golden["questions"]
    assert len(questions) == 45

    by_class: dict[str, int] = {}
    for question in questions:
        by_class[question["class"]] = by_class.get(question["class"], 0) + 1
    assert by_class == {"STRUCTURAL": 15, "SEMANTIC": 15, "TRACE": 15}


def test_golden_set_ids_are_unique(golden):
    ids = [q["id"] for q in golden["questions"]]
    assert len(set(ids)) == len(ids)


def test_structural_composition(golden):
    """§8.1: ">=4 on hub-adjacent symbols, exercising the `degree` cutoff"."""
    structural = [q for q in golden["questions"] if q["class"] == "STRUCTURAL"]
    hub_adjacent = [q for q in structural if q.get("hub_adjacent")]
    assert len(hub_adjacent) >= 4, len(hub_adjacent)


def test_semantic_composition(golden):
    """§8.1: ">=5 on undocumented or poorly-named symbols (§12.5)".

    §12.5 is the reason: retrieval is weakest exactly where a debugging
    assistant is most valuable, so the eval "measures the weakness rather than
    avoiding it".
    """
    semantic = [q for q in golden["questions"] if q["class"] == "SEMANTIC"]
    undocumented = [q for q in semantic if q.get("undocumented")]
    assert len(undocumented) >= 5, len(undocumented)


def test_trace_composition(golden):
    """§8.1: "8 Python, 7 TS/TSX — >=3 TS on App Router GET/POST handlers,
    exercising Tier 2 refusal"."""
    trace = [q for q in golden["questions"] if q["class"] == "TRACE"]
    python = [q for q in trace if q.get("language") == "python"]
    typescript = [q for q in trace if q.get("language") == "typescript"]
    app_router = [q for q in typescript if q.get("app_router")]

    assert len(python) == 8, len(python)
    assert len(typescript) == 7, len(typescript)
    assert len(app_router) >= 3, len(app_router)


def test_app_router_trace_questions_expect_a_refusal(golden):
    """The App Router questions exist to exercise T10, so their gold is plural.

    A question whose gold is a single handler would reward a guess — which is
    the behaviour §5.1 refuses.
    """
    for question in golden["questions"]:
        if question.get("app_router"):
            assert len(question["gold"]) >= 2, question["id"]
            assert question.get("expect_tier2_ambiguous"), question["id"]


def test_every_question_carries_gold_and_an_annotated_reference(golden):
    """§8.1: gold nodes "plus a reference answer annotated for claim sentences".

    The annotation is what makes §8.2's classifier precision/recall a measured
    number rather than an assumption (§12.9).
    """
    for question in golden["questions"]:
        assert question["gold"], question["id"]
        assert question["reference"], question["id"]
        assert len(question["claims"]) == len(question["reference"]), question["id"]
        assert all(isinstance(flag, bool) for flag in question["claims"]), question["id"]
        for rel_path, qualified_name in question["gold"]:
            assert rel_path.endswith((".py", ".ts", ".tsx")), question["id"]
            assert "." in qualified_name, question["id"]


def test_annotations_include_non_claims(golden):
    """A set annotated as all-claims cannot measure classifier precision."""
    flags = [flag for q in golden["questions"] for flag in q["claims"]]
    assert any(flags) and not all(flags)


def test_recall_stage_is_packed_context(golden):
    """§8.1: recall is measured against packed context, not candidates.

    "A gold node retrieved and then dropped by the packer never reached the
    model." The seed set from step 6 measured the easier quantity and says so;
    this one must not be confused with it.
    """
    assert golden["stage"] == "packed"


def test_seed_set_is_a_subset_of_the_full_set_intent():
    """Step 6's seed set stays, and stays labelled as the weaker measurement."""
    _questions, meta = load_questions()
    assert meta["stage"] == "retrieved"


# --------------------------------------------------------------------------
# §8.2 — all ten gates computable
# --------------------------------------------------------------------------


def test_all_gates_computable():
    """Plan §10's gate: "Every §8.2 metric produces a number."

    Given inputs, every gate yields a value and a verdict — nothing in the
    table is aspirational or missing a formula.
    """
    gates = build_gates(
        recall_at_10=0.82, mrr=0.61, coverage=0.88,
        classifier_precision=0.93, classifier_recall=0.87,
        relation_precision=0.91, cache_hit_rate=0.97,
        context_drop_rate=0.06, edge_survival=1.0,
        ttft_p50=0.8, ttft_p95=1.9, cost_p50=0.009, cost_p95=0.021,
        source="synthetic",
    )

    assert len(gates) == 13, "§8.2's ten rows, with the paired ones split out"
    assert all(gate.measured for gate in gates)
    assert all(gate.status in ("PASS", "FAIL") for gate in gates)
    assert all(isinstance(gate.value, float) for gate in gates)


def test_every_8_2_row_is_present():
    """Named against §8.2's own wording, so a dropped row is visible."""
    names = {gate.name for gate in build_gates()}
    for expected in (
        "Recall@10 (packed context)",
        "MRR",
        "Citation Coverage",
        "T2 classifier precision",
        "T2 classifier recall",
        "Relation Precision",
        "Embedding cache hit rate",
        "Context drop rate",
        "Inbound-edge survival across move",
        "TTFT p50",
        "TTFT p95",
        "$/query p50 (uncached)",
        "$/query p95 (uncached)",
    ):
        assert expected in names, expected


def test_thresholds_come_from_config_not_literals():
    """Every †. §12.10 expects all of them to move; `config.py` is where."""
    gates = {gate.name: gate for gate in build_gates()}
    assert gates["Recall@10 (packed context)"].threshold == config.GATE_RECALL_AT_10
    assert gates["Context drop rate"].threshold == config.GATE_CONTEXT_DROP_RATE
    assert gates["$/query p50 (uncached)"].threshold == config.GATE_COST_P50_USD


def test_an_unmeasured_gate_is_not_a_passing_gate():
    """The rule this whole module exists for.

    A gate defaulting to 0.0 would report PASS for "context drop rate" and FAIL
    for "recall" without a single run behind either. Both are lies, and the
    second is the kind that gets investigated.
    """
    gates = build_gates()
    assert all(not gate.measured for gate in gates)
    assert all(gate.status == "NOT MEASURED" for gate in gates)
    assert all("—" in gate.render() for gate in gates)


@pytest.mark.parametrize(
    "name, value, expected",
    [
        ("Recall@10 (packed context)", 0.80, "PASS"),
        ("Recall@10 (packed context)", 0.79, "FAIL"),
        ("Context drop rate", 0.10, "PASS"),
        ("Context drop rate", 0.11, "FAIL"),
        ("TTFT p50", 1.19, "PASS"),
        ("TTFT p50", 1.21, "FAIL"),
    ],
)
def test_gate_direction_is_respected(name, value, expected):
    """Half of §8.2's rows are upper bounds; inverting one silently passes it."""
    kwargs = {
        "Recall@10 (packed context)": "recall_at_10",
        "Context drop rate": "context_drop_rate",
        "TTFT p50": "ttft_p50",
    }[name]
    gate = {g.name: g for g in build_gates(**{kwargs: value})}[name]
    assert gate.status == expected


# --------------------------------------------------------------------------
# §8.2's resolution
# --------------------------------------------------------------------------


def test_gate_resolution_documented():
    """§8.2: "one flipped question moves a class score ~6.7 points".

    "Comparable to the difference between many of these thresholds" — so the
    report has to state it, or a 0.78 next to a 0.80 gate reads as a real
    regression when it may be one question.
    """
    report = GateReport(gates=build_gates(), decisions=build_decision_metrics({}))
    assert report.resolution_pp == pytest.approx(6.667, abs=0.01)

    rendered = report.render()
    assert "6.7pp" in rendered
    assert "15 questions per class" in rendered


def test_report_records_the_model_and_packed_tokens():
    """§8.2: "Recorded alongside: p50 packed tokens and the model name and
    version. Without both, neither the gate nor §5.4's break-even can be
    checked."
    """
    report = GateReport(
        gates=build_gates(), decisions=build_decision_metrics({}), p50_packed_tokens=8_900
    )
    rendered = report.render()

    assert config.LLM_MODEL in rendered
    assert config.EMBED_MODEL in rendered
    assert "8900" in rendered


def test_report_counts_passing_failing_and_unmeasured():
    report = GateReport(
        gates=build_gates(recall_at_10=0.9, mrr=0.1), decisions=[]
    )
    assert len(report.measured) == 2
    assert len(report.passing) == 1
    assert len(report.failing) == 1
    assert len(report.unmeasured) == 11
    assert "1 passing · 1 failing · 11 not measured" in report.render()


# --------------------------------------------------------------------------
# §8.3 — the four decision metrics
# --------------------------------------------------------------------------


def test_four_decision_metrics_are_computable():
    """§8.3: each deferred capability has exactly one metric, named once."""
    counters = {
        "move.undetected": 2,
        "cache.miss.header_only": 3,
        "cache.miss.total": 10,
        "pack.seed_l1_overflow": 1,
        "pack.seeds_total": 40,
    }
    decisions = build_decision_metrics(
        counters, ts_frames_total=10, ts_frames_resolved=4, moves_total=20
    )

    assert len(decisions) == 4
    values = {d.name: d.value for d in decisions}
    assert values["TS/TSX TRACE resolution"] == pytest.approx(0.4)
    assert values["Undetected-move rate"] == pytest.approx(0.1)
    assert values["Header-only miss share"] == pytest.approx(0.3)
    assert values["Seed L1-overflow rate"] == pytest.approx(0.025)


def test_decision_metrics_report_whether_the_trigger_fires():
    """§10.2's triggers read off these. The direction differs per row."""
    counters = {
        "move.undetected": 3, "cache.miss.header_only": 3, "cache.miss.total": 10,
        "pack.seed_l1_overflow": 4, "pack.seeds_total": 40,
    }
    decisions = {
        d.name: d
        for d in build_decision_metrics(
            counters, ts_frames_total=10, ts_frames_resolved=4, moves_total=20
        )
    }

    assert decisions["TS/TSX TRACE resolution"].fires is True     # 0.40 < 0.70
    assert decisions["Undetected-move rate"].fires is True        # 0.15 > 0.10
    assert decisions["Header-only miss share"].fires is True      # 0.30 > 0.20
    assert decisions["Seed L1-overflow rate"].fires is True       # 0.10 > 0.05


def test_a_metric_with_no_observations_is_unknown_not_zero():
    """An empty denominator is unknown, not zero.

    Reporting 0.0 would make "we never ran this" indistinguishable from "it
    never happened" — and §10.2 reads these in exactly that direction, so a
    false zero silently suppresses a trigger.
    """
    decisions = build_decision_metrics({})
    assert all(d.value is None for d in decisions)
    assert all(d.fires is None for d in decisions)
    assert all("—" in d.render() for d in decisions)


def test_decision_metrics_read_live_counters():
    """§9.4's "~7 lines of counters" feed these directly."""
    metrics.incr("pack.seeds_total", 20)
    metrics.incr("pack.seed_l1_overflow", 3)

    overflow = {d.name: d for d in build_decision_metrics()}["Seed L1-overflow rate"]
    assert overflow.value == pytest.approx(0.15)
    assert overflow.fires is True


def test_trace_resolution_is_scoped_to_ts_tsx():
    """§8.3: "Python resolves near-100% at Tier 0, and a blended rate would mask
    the failure the sourcemap tier addresses."

    The signature takes TS/TSX frame counts only — there is no parameter through
    which Python frames could dilute it.
    """
    import inspect

    params = inspect.signature(build_decision_metrics).parameters
    assert "ts_frames_total" in params and "ts_frames_resolved" in params
    assert not any("python" in name for name in params)


# --------------------------------------------------------------------------
# Percentiles
# --------------------------------------------------------------------------


def test_percentile_is_nearest_rank():
    """§8.2 reads p50/p95 against thresholds.

    An interpolated value can land between two observations that both fell on
    the same side of a threshold, producing a verdict no measurement supports.
    """
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert percentile(values, 0.5) == 0.3
    assert percentile(values, 0.95) == 0.5
    assert percentile(values, 0.0) == 0.1
    assert percentile([], 0.5) is None
    assert percentile([0.7], 0.95) == 0.7
