"""`eval/local_gates.py` — the arithmetic, and the run with stand-in models.

The script exists to be run against real local models (decision 0003), which
no test may require. What a test *can* hold is that its numbers are computed
the way §8.2 defines them, that an unmeasured quantity is never rendered as a
passing one, and that the whole measurement runs end to end over both fixtures
— with a scripted model, so the figures below describe the script, not a model.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import metrics
from api.cli import build_runtime
from eval.local_gates import (
    GOLDEN,
    REPORT_PATH,
    LocalRun,
    Outcome,
    language_of,
    language_of_path,
    measure,
    render,
)
from index.providers import ScriptedLLM

from .test_local_runtime_graph import FakeEmbedder


@pytest.fixture(autouse=True)
def reset():
    metrics.reset()
    yield
    metrics.reset()


def outcome(**fields) -> Outcome:
    defaults = dict(id="Q", router_class="SEMANTIC", language="python", answered=True)
    return Outcome(**{**defaults, **fields})


# -- the arithmetic -----------------------------------------------------------


def test_coverage_is_pooled_over_sentences_not_averaged_over_answers():
    """§6.2 defines the ratio over claim sentences. One uncited one-liner and
    one fully cited nine-sentence answer are 9 of 10, not a mean of 0.5."""
    run = LocalRun(outcomes=[outcome(claims=1, cited=0), outcome(claims=9, cited=9)])
    assert run.coverage == pytest.approx(0.9)


def test_an_answer_with_no_claims_neither_helps_nor_hurts_coverage():
    run = LocalRun(outcomes=[outcome(claims=0, cited=0), outcome(claims=4, cited=3)])
    assert run.coverage == pytest.approx(0.75)
    assert LocalRun(outcomes=[outcome(claims=0)]).coverage is None, "nothing to measure"


def test_relation_precision_is_unmeasured_when_nothing_was_asserted():
    """`GroundingReport` scores an answer with no relation claims 1.0, which is
    right for one answer and wrong for a gate: a model that never writes
    "`a` calls `b`" would pass Relation Precision by saying nothing."""
    run = LocalRun(outcomes=[outcome(relations=0), outcome(relations=0)])
    assert run.relation_precision is None
    assert "Relation Precision" in render(run)
    row = next(line for line in render(run).splitlines() if line.startswith("Relation Precision"))
    assert "NOT MEASURED" in row

    run = LocalRun(outcomes=[outcome(relations=4, relations_verified=3)])
    assert run.relation_precision == pytest.approx(0.75)


def test_a_question_the_model_never_answered_is_left_out_of_the_model_gates():
    """§9.3's retrieval-only fallback produced no answer to score, and no TTFT."""
    run = LocalRun(outcomes=[
        outcome(claims=2, cited=2, ttft=0.5, usd=0.0),
        outcome(answered=False, claims=0, cited=0, ttft=None),
    ])
    assert run.coverage == 1.0
    assert run.ttft(0.5) == 0.5
    assert len(run.answered) == 1


def test_ts_frames_are_counted_over_typescript_questions_only():
    run = LocalRun(outcomes=[
        outcome(language="python", frames_total=2, frames_resolved=2),
        outcome(language="typescript", frames_total=2, frames_resolved=1),
        outcome(language="typescript", frames_total=1, frames_resolved=0),
    ])
    assert run.ts_frames == (1, 3)


def test_the_corpus_of_a_question_and_of_a_gold_path():
    assert language_of({"class": "SEMANTIC"}) == "python"
    assert language_of({"class": "TRACE", "language": "typescript"}) == "typescript"
    assert language_of_path("components/UserCard.tsx") == "typescript"
    assert language_of_path("app/api/users/route.ts") == "typescript"
    assert language_of_path("authx/views.py") == "python"


def test_the_report_names_the_models_it_is_about():
    run = LocalRun(
        outcomes=[outcome(id="M01", claims=2, cited=2, ttft=0.4, total=1.0,
                          packed_tokens=900, gold_rank=1, text="It works [C1].")],
        llm="ollama:some-model", embedder="ollama:some-embedder", context_budget=5392,
        queries={"M01": "how are API tokens issued"},
        environment={"gpu": "a test"},
    )
    text = render(run)
    assert "model: ollama:some-model   embedding model: ollama:some-embedder" in text
    assert "| M01 | SEMANTIC |" in text
    assert "**M01** — how are API tokens issued" in text and "> It works [C1]." in text
    assert "5392" in text and "- gpu: a test" in text


# -- the run, with stand-in models ----------------------------------------------


CITED = "The symbol in question is implemented here [C1]."


@pytest.fixture
def runtimes(graph_db, fixtures_dir, tmp_path):
    graph_db.run("MATCH (n) DETACH DELETE n")
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    built = {
        language: build_runtime(
            f"repo_local_gates_{language}", (fixtures_dir.parent.parent / root).resolve(),
            ai="off", driver=graph_db.driver, cache_path=tmp_path / f"{language}.sqlite",
            llm=ScriptedLLM(CITED), provider=FakeEmbedder(),
        )
        for language, root in golden["corpus"].items()
    }
    for runtime in built.values():
        asyncio.run(runtime.indexer.sync_project(runtime.repo))
    yield built, golden
    for runtime in built.values():
        runtime.indexer.cache.close()
    graph_db.run("MATCH (n) DETACH DELETE n")


def test_the_whole_measurement_runs_over_both_fixtures(runtimes):
    built, golden = runtimes
    before = REPORT_PATH.read_bytes() if REPORT_PATH.exists() else None

    run = measure(built, golden)

    assert len(run.outcomes) == 45 and len(run.answered) == 45
    assert {o.language for o in run.outcomes} == {"python", "typescript"}
    assert run.llm == "scripted-test-llm" and run.embedder == "fake-local-a"

    # Recall, both ways, over every question whose gold exists in the corpus.
    for recall in (run.recall_fulltext, run.recall_hybrid):
        assert len(recall.scoreable) == 45, [r.question.id for r in recall.results if not r.scoreable]
    assert run.recall_fulltext.embedding_model.startswith("none")
    assert run.recall_hybrid.embedding_model == "fake-local-a"
    # Where the traversal ran, it is the answer (§5.1) and no embedder can
    # change it. The class as a whole is not asserted: two of its fifteen
    # questions fall back to search, where this crude embedder decides.
    traversed = [o for o in run.outcomes if o.route_taken == "STRUCTURAL"]
    assert len(traversed) >= 10
    assert all(o.gold_rank is not None for o in traversed), [
        o.id for o in traversed if o.gold_rank is None
    ]

    # The traversal, scored against expected_related, is part of the run.
    assert run.traversal is not None and len(run.traversal.results) == 15
    assert "structural traversal against expected_related" in render(run)

    # §8.3, through the real resolver: F-027's three of nine.
    assert run.ts_frames == (3, 9)

    # A scripted sentence with a handle: every answer is one cited claim.
    assert run.coverage == 1.0
    assert run.cost(0.5) is not None and run.ttft(0.5) is not None

    text = render(run)
    assert "TS/TSX TRACE resolution" in text and "3 of 9 repository frames" in text
    assert REPORT_PATH.read_bytes() == before if before is not None else not REPORT_PATH.exists(), (
        "measuring must not write the recorded report; only `main` does"
    )
