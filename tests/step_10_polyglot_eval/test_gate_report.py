"""Step 10's gate — record which gates pass and which do not.

Plan §10: "All ten gates produce values; record which pass and which do not."
That is not the same as "all ten gates pass", and the difference is the point.
This module measures every gate that can be measured *here*, leaves the rest
NOT MEASURED with a stated reason, and writes the result to
`eval/GATE_REPORT.md`.

Three gates are measurable offline and are measured for real:

* **T2 classifier precision / recall** — against the 45 hand-annotated
  reference answers in the golden set. §12.9 says this number "will not be
  1.0"; §8.2 gates it because Coverage is meaningless without it.
* **Embedding cache hit rate** — a real index of the Django fixture, twice.
* **Context drop rate** and **seed L1-overflow** — the real packer over real
  parsed symbols.

The rest need Neo4j, a real embedding provider, or a real LLM. Each is recorded
with which.
"""

from __future__ import annotations

import json

import pytest

import config
import metrics
from adapters.python import PythonAdapter
from eval.gates import GateReport, build_decision_metrics, build_gates, percentile
from ground.classify import classify
from index.cache import EmbeddingCache, embed_batch
from index.chunker import chunk_file
from index.providers import ApproxCodeTokenizer, HashEmbeddingProvider
from pack.packer import pack
from pack.render import Block
from retrieve.tokenize import build_search_text

GOLDEN_45 = config.REPO_ROOT / "eval" / "golden" / "golden_45.json"
#: The database-free subset. `test_gate_report_graph.py` writes the full
#: table to GATE_REPORT.md — two tests writing one path meant whichever ran
#: last decided what the artifact said, and the partial one won.
REPORT_PATH = config.REPO_ROOT / "eval" / "GATE_REPORT_no_database.md"

REPO = "repo_gate_report"


@pytest.fixture(autouse=True)
def reset_metrics():
    metrics.reset()
    yield
    metrics.reset()


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GOLDEN_45.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# The three measurable gates
# --------------------------------------------------------------------------


def measure_classifier(golden: dict) -> tuple[float, float, int]:
    """§6.1's predicate against the golden set's hand annotations.

    Every reference answer is split into sentences with a boolean per sentence.
    §8.2 gates precision >= 0.90 and recall >= 0.85 because, as §12.9 puts it,
    Coverage otherwise reads as "coverage as this classifier sees it" with no
    way to know how far that is from the truth.
    """
    known = frozenset(
        qualified_name
        for question in golden["questions"]
        for _rel, qualified_name in question["gold"]
    )

    tp = fp = fn = 0
    total = 0
    for question in golden["questions"]:
        for sentence, expected in zip(question["reference"], question["claims"]):
            predicted = bool(classify(sentence, known).claims)
            total += 1
            if predicted and expected:
                tp += 1
            elif predicted and not expected:
                fp += 1
            elif not predicted and expected:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return precision, recall, total


def measure_cache_hit_rate(fixtures_dir, tmp_path) -> float:
    """A real index of the Django fixture, run twice."""
    repo = fixtures_dir / "repos" / "django_min"
    adapter, tokenizer = PythonAdapter(), ApproxCodeTokenizer()
    provider = HashEmbeddingProvider()
    cache = EmbeddingCache(tmp_path / "gate.sqlite", model=provider.name)

    sources = {
        path.relative_to(repo).as_posix(): path.read_bytes()
        for path in sorted(repo.rglob("*.py"))
    }

    def index() -> None:
        batch = {}
        for rel, data in sources.items():
            parsed = chunk_file(adapter.parse(REPO, rel, data), tokenizer)
            for sym in parsed.symbols:
                sym.search_text = build_search_text(sym)
            batch[rel] = parsed
        embed_batch(REPO, batch, cache=cache, provider=provider)

    index()
    metrics.reset()
    index()
    hit_rate = metrics.get("cache.hits") / metrics.get("cache.lookups")
    cache.close()
    return hit_rate


def measure_packer(fixtures_dir) -> tuple[float, float, int]:
    """Drop rate, seed L1-overflow, and p50 packed tokens over real symbols."""
    repo = fixtures_dir / "repos" / "django_min"
    adapter, tokenizer = PythonAdapter(), ApproxCodeTokenizer()

    symbols = []
    for path in sorted(repo.rglob("*.py")):
        rel = path.relative_to(repo).as_posix()
        parsed = chunk_file(adapter.parse(REPO, rel, path.read_bytes()), tokenizer)
        symbols.extend(s for s in parsed.symbols if s.kind == "function")

    metrics.reset()
    packed_totals = []
    drop_rates = []

    # One pack per notional query: eight seeds, the rest as neighbours.
    for start in range(0, max(1, len(symbols) - 8), 8):
        window = symbols[start : start + 8]
        if not window:
            continue
        seeds = [Block(symbol=s, score=1.0 - i / 100) for i, s in enumerate(window)]
        neighbors = [
            Block(symbol=s, score=0.4) for s in symbols[start + 8 : start + 24]
        ]
        context = pack(seeds, neighbors, tokenizer)
        packed_totals.append(context.packed_tokens)
        drop_rates.append(context.drop_rate)

    drop_rate = sum(drop_rates) / len(drop_rates)
    overflow = metrics.get("pack.seed_l1_overflow") / max(
        1, metrics.get("pack.seeds_total")
    )
    return drop_rate, overflow, int(percentile(packed_totals, 0.5) or 0)


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------


def test_all_ten_gates_are_recorded(golden, fixtures_dir, tmp_path):
    """The subset measurable with no database at all.

    The assertions are on the three gates actually measured here. The others
    are asserted to be *honestly unmeasured* — a gate reporting PASS without a
    run behind it is the failure this whole report exists to prevent.
    """
    precision, recall, annotated_sentences = measure_classifier(golden)
    hit_rate = measure_cache_hit_rate(fixtures_dir, tmp_path)
    drop_rate, overflow, p50_packed = measure_packer(fixtures_dir)

    gates = build_gates(
        classifier_precision=precision,
        classifier_recall=recall,
        cache_hit_rate=hit_rate,
        context_drop_rate=drop_rate,
        source="measured offline",
    )
    decisions = build_decision_metrics(
        {
            "pack.seed_l1_overflow": metrics.get("pack.seed_l1_overflow"),
            "pack.seeds_total": metrics.get("pack.seeds_total"),
        }
    )

    report = GateReport(
        gates=gates,
        decisions=decisions,
        p50_packed_tokens=p50_packed,
        notes=[
            f"T2 classifier measured over {annotated_sentences} hand-annotated "
            f"sentences across all 45 golden questions.",
            "Embedding cache hit rate measured over a real double index of the "
            "Django fixture (73 symbols, 18 files).",
            "Context drop rate and p50 packed tokens measured with the "
            "stand-in tokenizer (FINDINGS F-006) — the figures move when the "
            "provider's real tokenizer is wired.",
            "Recall@10, MRR, Coverage, Relation Precision and inbound-edge "
            "survival need a live Neo4j; Recall additionally needs a real "
            "embedding provider (F-009). TTFT and $/query need a real LLM.",
            "NOT MEASURED is not PASS. Nine of the thirteen rows have no run "
            "behind them and say so.",
        ],
    )
    REPORT_PATH.write_text(report.render() + "\n", encoding="utf-8")

    # --- the gates that were actually measured ---------------------------
    by_name = {gate.name: gate for gate in gates}

    assert by_name["T2 classifier precision"].measured
    assert precision >= config.GATE_CLASSIFIER_PRECISION, (
        f"classifier precision {precision:.3f} — an over-inclusive denominator "
        f"manufactures low Coverage (§6.1)"
    )
    assert recall >= config.GATE_CLASSIFIER_RECALL, f"classifier recall {recall:.3f}"

    assert hit_rate >= config.GATE_CACHE_HIT_RATE, f"cache hit rate {hit_rate:.3f}"
    assert drop_rate <= config.GATE_CONTEXT_DROP_RATE, f"drop rate {drop_rate:.3f}"

    # --- and the ones that were not --------------------------------------
    unmeasured = {gate.name for gate in report.unmeasured}
    assert "Recall@10 (packed context)" in unmeasured
    assert "Inbound-edge survival across move" in unmeasured
    assert "$/query p50 (uncached)" in unmeasured
    assert all(gate.status == "NOT MEASURED" for gate in report.unmeasured)

    assert len(report.measured) == 4
    assert len(report.unmeasured) == 9
    assert not report.failing, [g.name for g in report.failing]


def test_the_report_is_written_and_readable(golden, fixtures_dir, tmp_path):
    test_all_ten_gates_are_recorded(golden, fixtures_dir, tmp_path)

    text = REPORT_PATH.read_text(encoding="utf-8")
    assert "§8.2 gates" in text
    assert "§8.3 decision metrics" in text
    assert "NOT MEASURED" in text
    assert "6.7pp" in text
    assert config.LLM_MODEL in text


def test_classifier_meets_its_gate_on_the_full_golden_set(golden):
    """§8.2's row, on the corpus §8.1 specifies rather than a sample.

    §12.9: "the number will not be 1.0, and Coverage reads as 'coverage as this
    classifier sees it'". Which is fine — but only once the number exists.
    """
    precision, recall, total = measure_classifier(golden)

    assert total >= 45, f"only {total} annotated sentences"
    assert 0.0 <= precision <= 1.0 and 0.0 <= recall <= 1.0
    assert precision >= config.GATE_CLASSIFIER_PRECISION
    assert recall >= config.GATE_CLASSIFIER_RECALL
