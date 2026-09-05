"""Step 10's gate, with a live graph — the full §8.2 table.

`test_gate_report.py` measures what needs no database. This adds the three that
do, against a real index of the Django fixture:

* **Recall@10 and MRR** — the seed set through the real fulltext arm. Measured
  against *packed* context, which is what §8.1 specifies and what step 6's
  baseline could not do: "a gold node retrieved and then dropped by the packer
  never reached the model."
* **Inbound-edge survival across a move** — §8.2's own fixture: `git mv` a file
  with five known callers and count them again.

Coverage, Relation Precision, TTFT and `$/query` stay NOT MEASURED. Each needs a
real LLM, and scripting one would measure the script.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

import config
import metrics
from adapters.python import PythonAdapter
from eval.gates import GateReport, build_decision_metrics, build_gates, percentile
from eval.harness import EvalHarness, load_questions
from graph.reader import GraphReader
from graph.writer import GraphWriter
from index.cache import EmbeddingCache, embed_batch
from index.chunker import chunk_file
from index.indexer import Indexer
from index.providers import ApproxCodeTokenizer, HashEmbeddingProvider
from pack.packer import pack
from pack.render import Block
from retrieve.hybrid import HybridRetriever, Neo4jSearchBackend
from retrieve.tokenize import build_search_text

from .test_gate_report import (
    GOLDEN_45,
    measure_cache_hit_rate,
    measure_classifier,
    measure_packer,
)

#: The authoritative artifact. Plan §10's gate reads off this one.
REPORT_PATH = config.REPO_ROOT / "eval" / "GATE_REPORT.md"

REPO = "repo_gate_graph"
MOVE_REPO = "repo_gate_move"


@pytest.fixture(autouse=True)
def reset_metrics():
    metrics.reset()
    yield
    metrics.reset()


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GOLDEN_45.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def corpus(graph_db, fixtures_dir, tmp_path_factory):
    """The Django fixture indexed through the real write path."""
    graph_db.run("MATCH (n) DETACH DELETE n")

    repo = fixtures_dir / "repos" / "django_min"
    adapter, tokenizer = PythonAdapter(), ApproxCodeTokenizer()
    provider = HashEmbeddingProvider()
    cache = EmbeddingCache(tmp_path_factory.mktemp("g") / "c.sqlite", model=provider.name)

    batch = {}
    for path in sorted(repo.rglob("*.py")):
        rel = path.relative_to(repo).as_posix()
        parsed = chunk_file(adapter.parse(REPO, rel, path.read_bytes()), tokenizer)
        for sym in parsed.symbols:
            sym.search_text = build_search_text(sym)
        batch[rel] = parsed

    vectors = embed_batch(REPO, batch, cache=cache, provider=provider)
    GraphWriter(graph_db.driver, graph_db.name).apply(
        REPO, batch, batch.keys(), vectors, time.time_ns()
    )
    cache.close()
    graph_db.run("CALL db.awaitIndexes(180)")
    return batch


def measure_recall_against_packed_context(corpus, graph_db) -> tuple[float, float, int, str]:
    """§8.1's measurement: recall against **packed** context, not candidates.

    Retrieve, then pack, then score against what survived packing. The
    difference is the whole point — a gold node the packer dropped never
    reached the model, and counting it as a hit would report retrieval quality
    the answer never saw.

    The fulltext arm only. The vector arm is `HashEmbeddingProvider`, which has
    no semantic structure (F-009), so including it would fuse noise with signal
    and report the result as recall.
    """
    questions, meta = load_questions()
    backend = Neo4jSearchBackend(graph_db.driver, graph_db.name)
    retriever = HybridRetriever(backend=backend)
    reader = GraphReader(graph_db.driver, graph_db.name)
    tokenizer = ApproxCodeTokenizer()

    by_uid = {s.uid: s for parsed in corpus.values() for s in parsed.symbols}
    packed_totals: list[int] = []

    def resolve(rel_path: str, qualified_name: str) -> list[str]:
        parsed = corpus.get(rel_path)
        if parsed is None:
            return []
        return [s.uid for s in parsed.symbols if s.qualified_name == qualified_name]

    def retrieve_packed(question) -> list[str]:
        candidates = retriever.search(REPO, question.query)
        seeds = [
            Block(symbol=by_uid[c.uid], score=1.0 / (rank + 1))
            for rank, c in enumerate(candidates[:8])
            if c.uid in by_uid
        ]
        neighbors = [
            Block(symbol=by_uid[c.uid], score=0.3)
            for c in candidates[8:24]
            if c.uid in by_uid
        ]
        context = pack(seeds, neighbors, tokenizer)
        packed_totals.append(context.packed_tokens)
        return [p.uid for p in context.blocks]

    report = EvalHarness(
        resolve, retrieve_packed, stage="packed (fulltext arm only)",
        embedding_model="n/a — fulltext only (F-009)",
    ).run(questions, corpus=meta["corpus"])

    return (
        report.recall_at(10),
        report.mrr(),
        int(percentile(packed_totals, 0.5) or 0),
        report.render(),
    )


def measure_edge_survival(graph_db, tmp_path, fixtures_dir) -> float:
    """§8.2's move-survival fixture (a), run for its number.

    `git mv` a file with five known inbound callers, flush, count them again.
    §4.4 on the failure: "the system then reports zero callers for code with
    dozens, which is worse than an error because it looks like an answer."
    """
    import subprocess

    from tests.step_05_indexer.conftest import CALLEE, CALLERS

    root = tmp_path / "movework"
    root.mkdir()

    def git(*args):
        subprocess.run(["git", *args], cwd=str(root), capture_output=True, check=True)

    git("init", "-q", ".")
    git("config", "core.autocrlf", "false")
    git("config", "user.email", "t@example.invalid")
    git("config", "user.name", "t")
    (root / "pkg").mkdir()
    (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pkg" / "callee.py").write_text(CALLEE, encoding="utf-8", newline="\n")
    (root / "pkg" / "callers.py").write_text(CALLERS, encoding="utf-8", newline="\n")
    git("add", "-A")
    git("commit", "-qm", "seed")

    provider = HashEmbeddingProvider()
    cache = EmbeddingCache(tmp_path / "move.sqlite", model=provider.name)
    indexer = Indexer(
        writer=GraphWriter(graph_db.driver, graph_db.name),
        reader=GraphReader(graph_db.driver, graph_db.name),
        cache=cache, provider=provider, tokenizer=ApproxCodeTokenizer(),
        roots={MOVE_REPO: root}, debounce=0.05,
    )
    asyncio.run(indexer.full_reconcile(MOVE_REPO))
    asyncio.run(indexer.startup(MOVE_REPO))

    def callers() -> set[str]:
        rows = graph_db.run(
            """
            MATCH (c:Symbol)-[:CALLS]->(h:Symbol {repo_id: $r, name: 'helper'})
            RETURN DISTINCT c.name AS name
            """,
            r=MOVE_REPO,
        )
        return {row["name"] for row in rows}

    before = callers()
    assert len(before) >= 5, f"baseline is {len(before)} callers, need >=5"

    async def move():
        subprocess.run(["git", "mv", "pkg/callee.py", "pkg/relocated.py"],
                       cwd=str(root), capture_output=True, check=True)
        indexer.on_rename(MOVE_REPO, "pkg/callee.py", "pkg/relocated.py")
        await asyncio.sleep(0.25)
        await indexer.join()

    asyncio.run(move())
    after = callers()
    cache.close()

    graph_db.run("MATCH (n:Symbol {repo_id: $r}) DETACH DELETE n", r=MOVE_REPO)
    graph_db.run("MATCH (f:File {repo_id: $r}) DETACH DELETE f", r=MOVE_REPO)
    return len(before & after) / len(before)


# --------------------------------------------------------------------------


def test_the_full_gate_table_is_recorded(golden, corpus, graph_db, fixtures_dir, tmp_path):
    """Plan §10: "All ten gates produce values; record which pass and which do not."

    Seven of thirteen rows are now measured against real runs. The remaining six
    need a real LLM, and are recorded as NOT MEASURED with that reason — a gate
    reporting PASS on a scripted answer would be measuring the script.
    """
    precision, recall_c, annotated = measure_classifier(golden)
    hit_rate = measure_cache_hit_rate(fixtures_dir, tmp_path)
    drop_rate, overflow, _p50_offline = measure_packer(fixtures_dir)
    recall10, mrr, p50_packed, recall_detail = measure_recall_against_packed_context(
        corpus, graph_db
    )
    survival = measure_edge_survival(graph_db, tmp_path, fixtures_dir)

    gates = build_gates(
        recall_at_10=recall10,
        mrr=mrr,
        classifier_precision=precision,
        classifier_recall=recall_c,
        cache_hit_rate=hit_rate,
        context_drop_rate=drop_rate,
        edge_survival=survival,
        source="measured",
    )
    report = GateReport(
        gates=gates,
        decisions=build_decision_metrics(
            {
                "pack.seed_l1_overflow": metrics.get("pack.seed_l1_overflow"),
                "pack.seeds_total": metrics.get("pack.seeds_total"),
            }
        ),
        p50_packed_tokens=p50_packed,
        notes=[
            f"Neo4j: measured against a live server. T2 classifier over "
            f"{annotated} hand-annotated sentences from all 45 golden questions.",
            "Recall@10 and MRR are against **packed** context (§8.1), the "
            "fulltext arm only — the vector arm is HashEmbeddingProvider, which "
            "has no semantic structure (F-009). A real embedding provider can "
            "only improve these.",
            "Inbound-edge survival: §8.2's fixture (a) — `git mv` a file with "
            "five known callers, flush, recount.",
            "Token figures use the stand-in tokenizer (F-006) and move when the "
            "provider's real one is wired.",
            "Citation Coverage, Relation Precision, TTFT and $/query need a real "
            "LLM. Scripting one would measure the script, so they are NOT "
            "MEASURED rather than reported.",
            "",
            "Recall detail:",
            *[f"    {line}" for line in recall_detail.splitlines()],
        ],
    )
    REPORT_PATH.write_text(report.render() + "\n", encoding="utf-8")

    by_name = {g.name: g for g in gates}
    assert by_name["Recall@10 (packed context)"].measured
    assert recall10 >= config.GATE_RECALL_AT_10, (
        f"Recall@10 {recall10:.3f} against packed context — §8.2 gates at "
        f"{config.GATE_RECALL_AT_10}. Per §6's note this is data about the "
        f"constant as much as about the code."
    )
    assert mrr >= config.GATE_MRR, f"MRR {mrr:.3f}"
    assert survival >= config.GATE_INBOUND_EDGE_SURVIVAL, (
        f"inbound-edge survival {survival:.3f} — F-007 made this silently zero"
    )
    assert survival == 1.0, "an in-place UID rewrite should lose nothing at all"

    assert len(report.measured) == 7
    assert not report.failing, [g.name for g in report.failing]
    assert {g.name for g in report.unmeasured} == {
        "Citation Coverage",
        "Relation Precision",
        "TTFT p50",
        "TTFT p95",
        "$/query p50 (uncached)",
        "$/query p95 (uncached)",
    }
