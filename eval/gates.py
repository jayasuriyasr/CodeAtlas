"""§8.2's ten gates and §8.3's four decision metrics.

Principle 6: "A metric named twice is defined once." So every quantity §8 and
§10 both mention is computed here, once, and both read it from the same place.

Two rules shape the design:

**An unmeasured gate is not a passing gate.** `value=None` renders as
NOT MEASURED, never as 0.0 and never as PASS. §4 of the plan's working rules
says to record actual output, not expected — a gate with no run behind it has
no output to record.

**The resolution travels with the score.** §8.2: "At 15 questions per class,
one flipped question moves a class score ~6.7 points — comparable to the
difference between many of these thresholds." A score printed without that
invites over-reading, so `render` prints both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import config
import metrics

GREATER = ">="
LESS = "<="


@dataclass(frozen=True)
class Gate:
    """One §8.2 row: what it catches, what it measured, and against what."""

    name: str
    catches: str
    threshold: float
    direction: str
    value: float | None = None
    #: Where the number came from. A gate whose provenance is a stand-in
    #: provider is a different claim from one measured against the real thing.
    source: str = ""
    unit: str = ""

    @property
    def measured(self) -> bool:
        return self.value is not None

    @property
    def status(self) -> str:
        if not self.measured:
            return "NOT MEASURED"
        if self.direction is GREATER or self.direction == GREATER:
            return "PASS" if self.value >= self.threshold else "FAIL"
        return "PASS" if self.value <= self.threshold else "FAIL"

    def render(self) -> str:
        shown = "—" if not self.measured else f"{self.value:.3f}{self.unit}"
        return (
            f"{self.name:<34} {shown:>12}  {self.direction} "
            f"{self.threshold}{self.unit:<4}  {self.status:<12} {self.source}"
        )


@dataclass(frozen=True)
class DecisionMetric:
    """One §8.3 row. These choose Phase 1 work; they do not gate a release."""

    name: str
    decides: str
    threshold: float
    direction: str
    value: float | None = None
    source: str = ""

    @property
    def fires(self) -> bool | None:
        """Whether the trigger in §10.2 is met. `None` when unmeasured."""
        if self.value is None:
            return None
        return (
            self.value > self.threshold
            if self.direction == ">"
            else self.value < self.threshold
        )

    def render(self) -> str:
        shown = "—" if self.value is None else f"{self.value:.3f}"
        fires = "—" if self.fires is None else ("FIRES" if self.fires else "no")
        return (
            f"{self.name:<28} {shown:>8}  {self.direction} {self.threshold:<6} "
            f"{fires:<7} {self.decides}"
        )


def _ratio(numerator: int, denominator: int) -> float | None:
    """`None` rather than 0.0 when nothing was observed.

    A rate over an empty denominator is not zero; it is unknown. Reporting it
    as zero would make "we never ran this" indistinguishable from "it never
    happened", and §8.3's thresholds are read in exactly that direction.
    """
    return numerator / denominator if denominator else None


# --------------------------------------------------------------------------
# §8.2
# --------------------------------------------------------------------------


def build_gates(
    *,
    recall_at_10: float | None = None,
    mrr: float | None = None,
    coverage: float | None = None,
    classifier_precision: float | None = None,
    classifier_recall: float | None = None,
    relation_precision: float | None = None,
    cache_hit_rate: float | None = None,
    context_drop_rate: float | None = None,
    edge_survival: float | None = None,
    ttft_p50: float | None = None,
    ttft_p95: float | None = None,
    cost_p50: float | None = None,
    cost_p95: float | None = None,
    source: str = "",
) -> list[Gate]:
    """§8.2's table, in its order, with its stated thresholds."""
    return [
        Gate("Recall@10 (packed context)", "retrieval failure",
             config.GATE_RECALL_AT_10, GREATER, recall_at_10, source),
        Gate("MRR", "ranking quality",
             config.GATE_MRR, GREATER, mrr, source),
        Gate("Citation Coverage", "asserting without support",
             config.GATE_CITATION_COVERAGE, GREATER, coverage, source),
        Gate("T2 classifier precision", "whether Coverage means anything",
             config.GATE_CLASSIFIER_PRECISION, GREATER, classifier_precision, source),
        Gate("T2 classifier recall", "whether Coverage means anything",
             config.GATE_CLASSIFIER_RECALL, GREATER, classifier_recall, source),
        Gate("Relation Precision", "misreading correct context",
             config.GATE_RELATION_PRECISION, GREATER, relation_precision, source),
        Gate("Embedding cache hit rate", "silent cache-key regression",
             config.GATE_CACHE_HIT_RATE, GREATER, cache_hit_rate, source),
        Gate("Context drop rate", "chronic budget pressure",
             config.GATE_CONTEXT_DROP_RATE, LESS, context_drop_rate, source),
        Gate("Inbound-edge survival across move", "move-attrition regression",
             config.GATE_INBOUND_EDGE_SURVIVAL, GREATER, edge_survival, source),
        Gate("TTFT p50", "perceived latency",
             config.GATE_TTFT_P50_S, LESS, ttft_p50, source, unit="s"),
        Gate("TTFT p95", "perceived latency",
             config.GATE_TTFT_P95_S, LESS, ttft_p95, source, unit="s"),
        Gate("$/query p50 (uncached)", "unit economics",
             config.GATE_COST_P50_USD, LESS, cost_p50, source),
        Gate("$/query p95 (uncached)", "unit economics",
             config.GATE_COST_P95_USD, LESS, cost_p95, source),
    ]


# --------------------------------------------------------------------------
# §8.3
# --------------------------------------------------------------------------


def build_decision_metrics(
    counters: dict[str, int] | None = None,
    *,
    ts_frames_total: int = 0,
    ts_frames_resolved: int = 0,
    moves_total: int = 0,
) -> list[DecisionMetric]:
    """§8.3's four, computed from the counters §9.4 says to keep.

    Each deferred capability has exactly one metric, "named once, referenced
    identically here and in §10.2".
    """
    counters = counters if counters is not None else metrics.snapshot()

    trace = _ratio(ts_frames_resolved, ts_frames_total)
    undetected = _ratio(counters.get("move.undetected", 0), moves_total)
    header_only = _ratio(
        counters.get("cache.miss.header_only", 0), counters.get("cache.miss.total", 0)
    )
    overflow = _ratio(
        counters.get("pack.seed_l1_overflow", 0), counters.get("pack.seeds_total", 0)
    )

    return [
        DecisionMetric(
            "TS/TSX TRACE resolution", "sourcemap tier",
            config.DECIDE_TRACE_RESOLUTION_BELOW, "<", trace,
            "tier 0+2 resolved / TS-TSX frames; tier2.ambiguous counts unresolved",
        ),
        DecisionMetric(
            "Undetected-move rate", "import-path repair",
            config.DECIDE_UNDETECTED_MOVE_ABOVE, ">", undetected,
            "move.undetected / total moves (§4.5 diff)",
        ),
        DecisionMetric(
            "Header-only miss share", "warm cache reuse",
            config.DECIDE_HEADER_ONLY_MISS_ABOVE, ">", header_only,
            "misses with unchanged stored body_hash / total misses",
        ),
        DecisionMetric(
            "Seed L1-overflow rate", "L2 skeleton",
            config.DECIDE_SEED_L1_OVERFLOW_ABOVE, ">", overflow,
            "pack.seed_l1_overflow / pack.seeds_total",
        ),
    ]


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------


@dataclass
class GateReport:
    gates: list[Gate] = field(default_factory=list)
    decisions: list[DecisionMetric] = field(default_factory=list)
    #: §8.2: recorded alongside every score, or neither the gate nor §5.4's
    #: break-even can be checked.
    model: str = config.LLM_MODEL
    embedding_model: str = config.EMBED_MODEL
    p50_packed_tokens: int | None = None
    questions_per_class: int = 15
    notes: list[str] = field(default_factory=list)

    @property
    def resolution_pp(self) -> float:
        """§8.2: one flipped question, in percentage points of a class score."""
        return 100.0 / self.questions_per_class if self.questions_per_class else 0.0

    @property
    def measured(self) -> list[Gate]:
        return [g for g in self.gates if g.measured]

    @property
    def passing(self) -> list[Gate]:
        return [g for g in self.gates if g.status == "PASS"]

    @property
    def failing(self) -> list[Gate]:
        return [g for g in self.gates if g.status == "FAIL"]

    @property
    def unmeasured(self) -> list[Gate]:
        return [g for g in self.gates if not g.measured]

    def render(self) -> str:
        lines = [
            "# §8.2 gates",
            "",
            f"model: {self.model}   embedding model: {self.embedding_model}",
            f"p50 packed tokens: "
            f"{self.p50_packed_tokens if self.p50_packed_tokens is not None else '—'}",
            f"resolution: one flipped question = {self.resolution_pp:.1f}pp "
            f"of a class score ({self.questions_per_class} questions per class)",
            "",
        ]
        lines += [gate.render() for gate in self.gates]
        lines += [
            "",
            f"{len(self.passing)} passing · {len(self.failing)} failing · "
            f"{len(self.unmeasured)} not measured, of {len(self.gates)}",
            "",
            "# §8.3 decision metrics",
            "",
        ]
        lines += [metric.render() for metric in self.decisions]
        if self.notes:
            lines += ["", "# notes", ""]
            lines += [f"- {note}" for note in self.notes]
        return "\n".join(lines)


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Nearest-rank percentile. `None` on an empty sample.

    Nearest-rank rather than interpolated: §8.2's p50 and p95 are read against
    thresholds, and an interpolated value can sit between two observations that
    both fell on the same side of one.
    """
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(fraction * len(ordered) + 0.5) - 1))
    return ordered[index]
