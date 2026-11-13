"""§8.2's model-dependent gates, measured with the local models (decision 0003).

`eval/GATE_REPORT.md` leaves six rows NOT MEASURED: Citation Coverage, Relation
Precision, TTFT and `$/query` each need a real LLM, and "scripting one would
measure the script". Its Recall@10 is the fulltext arm alone, because the test
suite's embedder has no semantic structure (F-009). This is the run that fills
those in on one machine — the golden set through the real router, the real
hybrid retriever with real embeddings, the packer, a real model and the
validators:

    python -m eval.local_gates                 # writes eval/GATE_REPORT_local.md

It is a script and not a test on purpose. It needs an Ollama server holding both
models, takes minutes, and its numbers belong to the hardware and the model
build it ran on; a red or green from it would say nothing about the code. What
it writes is a record (plan rule 4: actual output, never expected output).

The numbers describe `qwen2.5-coder:3b` and `nomic-embed-text` — not the hosted
pair decision 0002 names. Nothing here moves a row of `GATE_REPORT.md`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import config
import metrics
from eval.gates import GateReport, build_decision_metrics, build_gates, percentile
from eval.harness import EvalHarness, EvalReport, Question
from eval.structural import TraversalReport, measure_traversal
from pack.packer import pack
from retrieve.frames import is_library_path

GOLDEN = config.REPO_ROOT / "eval" / "golden" / "golden_45.json"
REPORT_PATH = config.REPO_ROOT / "eval" / "GATE_REPORT_local.md"

#: One repository per corpus language, named so no real index can collide.
REPO_IDS = {"python": "gate_local_python", "typescript": "gate_local_typescript"}


def language_of(question: dict) -> str:
    """Which corpus a golden question is asked of.

    Only TRACE questions carry `language`; every STRUCTURAL and SEMANTIC one is
    about the Django fixture. The gold's file extension says the same thing, and
    is what `resolve` has to go on.
    """
    return question.get("language") or "python"


def language_of_path(rel_path: str) -> str:
    return "typescript" if rel_path.endswith((".ts", ".tsx")) else "python"


# --------------------------------------------------------------------------
# One question's outcome
# --------------------------------------------------------------------------


@dataclass
class Outcome:
    id: str
    router_class: str
    language: str
    route_taken: str = ""
    answered: bool = False
    claims: int = 0
    cited: int = 0
    relations: int = 0
    relations_verified: int = 0
    conflict_flags: int = 0
    unknown_citations: int = 0
    #: Cited claims naming code their cited blocks do not contain.
    unsupported: int = 0
    retried: bool = False
    needs_review: bool = False
    ttft: float | None = None
    total: float | None = None
    packed_tokens: int = 0
    usd: float = 0.0
    #: 1-based rank of the first gold block in the packed context the kept
    #: answer was written from; None when no gold block reached the model.
    gold_rank: int | None = None
    frames_total: int = 0
    frames_resolved: int = 0
    text: str = ""

    @property
    def coverage(self) -> float:
        return self.cited / self.claims if self.claims else 1.0


@dataclass
class LocalRun:
    outcomes: list[Outcome] = field(default_factory=list)
    #: Recall against packed context, retrieval only: the vector arm off and on.
    recall_fulltext: EvalReport | None = None
    recall_hybrid: EvalReport | None = None
    #: The STRUCTURAL traversal against `expected_related` (eval/structural.py).
    traversal: TraversalReport | None = None
    llm: str = ""
    embedder: str = ""
    context_budget: int = 0
    environment: dict = field(default_factory=dict)
    counters: dict = field(default_factory=dict)
    #: id -> the question as asked, for the report's record of the answers.
    queries: dict = field(default_factory=dict)
    index_seconds: float = 0.0
    wall_seconds: float = 0.0

    # -- §8.2's quantities, each computed once ------------------------------

    @property
    def answered(self) -> list[Outcome]:
        return [o for o in self.outcomes if o.answered]

    @property
    def coverage(self) -> float | None:
        """Cited claim sentences ÷ claim sentences, pooled over the set.

        Pooled rather than a mean of per-answer ratios: §6.2 defines the ratio
        over sentences, and a mean would let a one-sentence answer weigh as
        much as a five-sentence one.
        """
        claims = sum(o.claims for o in self.answered)
        return sum(o.cited for o in self.answered) / claims if claims else None

    @property
    def relation_precision(self) -> float | None:
        asserted = sum(o.relations for o in self.answered)
        if not asserted:
            return None                   # nothing asserted is not "all verified"
        return sum(o.relations_verified for o in self.answered) / asserted

    def ttft(self, fraction: float) -> float | None:
        return percentile([o.ttft for o in self.answered if o.ttft is not None], fraction)

    def cost(self, fraction: float) -> float | None:
        return percentile([o.usd for o in self.answered], fraction)

    @property
    def p50_packed(self) -> int | None:
        value = percentile([o.packed_tokens for o in self.outcomes], 0.5)
        return int(value) if value is not None else None

    @property
    def ts_frames(self) -> tuple[int, int]:
        rows = [o for o in self.outcomes if o.language == "typescript"]
        return sum(o.frames_resolved for o in rows), sum(o.frames_total for o in rows)


# --------------------------------------------------------------------------
# Measuring
# --------------------------------------------------------------------------


def _questions(golden: dict) -> list[Question]:
    return [
        Question(id=q["id"], query=q["query"], router_class=q["class"],
                 gold=tuple((g[0], g[1]) for g in q["gold"]))
        for q in golden["questions"]
    ]


def _packed_uids(runtime, query: str, *, vectors: bool) -> list[str]:
    """What the model would be shown for `query`, without asking it."""
    retriever = runtime.retriever
    saved = retriever.embed_query
    retriever.embed_query = runtime.embed_query if vectors else None
    try:
        routed = runtime.service.router.route(query)
        seeds, neighbors, _notes = retriever(query, routed)
    finally:
        retriever.embed_query = saved
    context = pack(seeds, neighbors, runtime.service.tokenizer,
                   available=runtime.service.available)
    return [p.uid for p in context.blocks]


def _notes(runtime, query: str) -> dict:
    """The retriever's notes for `query` — the traversal's answer, no model."""
    routed = runtime.service.router.route(query)
    return runtime.retriever(query, routed)[2]


def measure_recall(runtimes: dict, golden: dict, *, vectors: bool) -> EvalReport:
    """§8.1: recall against **packed** context, through the real router.

    Run twice — the vector arm off, then on — over the same questions, so the
    difference is the embedder's contribution and nothing else's.
    """
    by_id = {q["id"]: language_of(q) for q in golden["questions"]}

    def resolve(rel_path: str, qualified_name: str) -> list[str]:
        runtime = runtimes[language_of_path(rel_path)]
        return [
            row["uid"] for row in runtime.service.graph_runner.run(
                "MATCH (s:Symbol {repo_id: $repo, rel_path: $path, qualified_name: $qn}) "
                "RETURN s.uid AS uid",
                repo=runtime.repo, path=rel_path, qn=qualified_name,
            )
        ]

    def retrieve(question: Question) -> Sequence[str]:
        return _packed_uids(runtimes[by_id[question.id]], question.query, vectors=vectors)

    embedder = next(iter(runtimes.values())).provider.name
    return EvalHarness(
        resolve, retrieve,
        stage="packed, vector arm " + ("on" if vectors else "off"),
        embedding_model=embedder if vectors else "none (fulltext and graph only)",
    ).run(_questions(golden), corpus="golden_45: django_min + nextjs_min")


def measure_answer(runtime, question: dict, gold_uids: set[str]) -> Outcome:
    """One golden question through the whole query path."""
    outcome = Outcome(id=question["id"], router_class=question["class"],
                      language=language_of(question))
    result = runtime.service.answer(question["query"])

    outcome.route_taken = result.notes.get("route_taken") or result.notes.get("route", "")
    outcome.packed_tokens = result.context.packed_tokens
    outcome.total = result.total_seconds
    outcome.ttft = result.ttft_seconds
    outcome.retried = result.retried
    outcome.text = result.text
    packed = [p.uid for p in result.context.blocks]
    outcome.gold_rank = next(
        (rank for rank, uid in enumerate(packed, start=1) if uid in gold_uids), None)

    # §8.3: tier 0 + tier 2 resolved ÷ TS/TSX frames. Library frames were never
    # going to resolve and are not the repository's frames.
    frames = result.routed.frames or ()
    for frame, note in zip(frames, result.notes.get("frames", []), strict=False):
        if frame.language == "python" or is_library_path(frame.file_path):
            continue
        outcome.frames_total += 1
        outcome.frames_resolved += note["tier"] in ("direct", "name")

    report = result.report
    if report is None or result.cost is None:
        return outcome                       # §9.3 retrieval-only: no answer to score
    outcome.answered = True
    outcome.claims = len(report.answer.claims)
    outcome.cited = len(report.answer.cited_claims)
    outcome.relations = len(report.relation_verdicts)
    outcome.relations_verified = sum(1 for v in report.relation_verdicts if v.verified)
    outcome.conflict_flags = report.conflict_flags
    outcome.unknown_citations = len(report.handle_errors)
    outcome.unsupported = len(report.unsupported)
    outcome.usd = result.cost.usd
    outcome.needs_review = result.retried and (
        report.coverage < config.COVERAGE_RETRY_THRESHOLD or report.conflict_flags > 0)
    return outcome


def measure(runtimes: dict, golden: dict, *, progress=None) -> LocalRun:
    """The whole run, given one indexed runtime per corpus language."""
    started = time.perf_counter()
    any_runtime = next(iter(runtimes.values()))
    run = LocalRun(
        llm=getattr(any_runtime.llm, "name", "?"),
        embedder=any_runtime.provider.name,
        context_budget=any_runtime.service.available,
    )
    run.recall_fulltext = measure_recall(runtimes, golden, vectors=False)
    run.recall_hybrid = measure_recall(runtimes, golden, vectors=True)
    run.traversal = measure_traversal(golden["questions"], lambda query: _notes(
        runtimes["python"], query))
    gold = {r.question.id: r.gold_uids for r in run.recall_hybrid.results}
    run.queries = {q["id"]: q["query"] for q in golden["questions"]}

    metrics.reset()
    for question in golden["questions"]:
        outcome = measure_answer(
            runtimes[language_of(question)], question, gold.get(question["id"], set()))
        run.outcomes.append(outcome)
        if progress is not None:
            progress(outcome)
    run.counters = metrics.snapshot()
    run.wall_seconds = time.perf_counter() - started
    return run


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------


def _seconds(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}s"


def render(run: LocalRun) -> str:
    hybrid = run.recall_hybrid
    resolved, total = run.ts_frames
    gates = build_gates(
        recall_at_10=hybrid.recall_at(10) if hybrid else None,
        mrr=hybrid.mrr() if hybrid else None,
        coverage=run.coverage,
        relation_precision=run.relation_precision,
        ttft_p50=run.ttft(0.5),
        ttft_p95=run.ttft(0.95),
        cost_p50=run.cost(0.5),
        cost_p95=run.cost(0.95),
        source="local models",
    )
    answered = run.answered
    retried = [o for o in answered if o.retried]
    relations = sum(o.relations for o in answered)
    notes = [
        f"{len(run.outcomes)} golden questions; {len(answered)} answered by the model, "
        f"{len(retried)} of them on a second attempt (§6.3), "
        f"{sum(o.needs_review for o in answered)} still below the Coverage threshold "
        f"after it.",
        f"Citation Coverage is pooled: {sum(o.cited for o in answered)} cited of "
        f"{sum(o.claims for o in answered)} claim sentences, kept answers only. "
        f"Mean of per-answer Coverage: "
        f"{sum(o.coverage for o in answered) / len(answered) if answered else 0:.3f}.",
        f"Citations of a block that was never shown (T1): "
        f"{sum(o.unknown_citations for o in answered)}. Coverage counts those "
        f"sentences as cited (§6.2); the console flags each one.",
        f"Cited but unsupported — a cited claim naming code that none of its cited "
        f"blocks contains: {sum(o.unsupported for o in answered)} of "
        f"{sum(o.cited for o in answered)} cited claims. Coverage counts them as cited.",
        f"Relation Precision is over {relations} relation claims the answers made "
        f"({sum(o.conflict_flags for o in answered)} contradicted the graph)."
        + ("" if relations else
           " None were made in the form T3 extracts — two backticked names joined "
           "by a relation verb, e.g. `a` calls `b` (§6.2, deliberately narrow) — "
           "so it is NOT MEASURED, not 1.0."),
        "TTFT is from the question arriving to the first word, model already "
        "loaded: routing, the query embedding, retrieval and packing are inside it.",
        f"Context budget {run.context_budget} tokens — the local model's window "
        f"less the reserves (decision 0003) — against §5.4's {config.AVAILABLE}.",
        "$/query is 0 because nothing is billed. It says nothing about the hosted "
        "pair's cost, which stays NOT MEASURED in GATE_REPORT.md.",
        "Rows that do not depend on the model (T2 classifier, cache hit rate, drop "
        "rate, edge survival) are measured in GATE_REPORT.md and not repeated here.",
        f"TS/TSX TRACE resolution: {resolved} of {total} repository frames resolved "
        f"by tier 0 or tier 2.",
    ]
    report = GateReport(
        gates=gates,
        decisions=build_decision_metrics(
            run.counters, ts_frames_total=total, ts_frames_resolved=resolved),
        model=run.llm,
        embedding_model=run.embedder,
        p50_packed_tokens=run.p50_packed,
        notes=notes,
    )

    lines = [
        "<!-- Written by `python -m eval.local_gates`. A record of one run on one "
        "machine; do not edit by hand. -->",
        "",
        report.render(),
        "",
        "# environment",
        "",
    ]
    lines += [f"- {key}: {value}" for key, value in run.environment.items()]
    lines += [
        f"- index: both fixtures in {run.index_seconds:.1f}s; "
        f"whole run {run.wall_seconds:.0f}s",
        "",
        "# recall against packed context (§8.1), vector arm off and on",
        "",
    ]
    for label, recall in (("off", run.recall_fulltext), ("on", run.recall_hybrid)):
        if recall is None:
            continue
        per_class = ", ".join(f"{c}={v:.2f}" for c, v in recall.by_class().items())
        missed = ", ".join(r.question.id for r in recall.misses()) or "none"
        lines.append(
            f"- vector arm {label:<3}: Recall@1 {recall.recall_at(1):.3f} · "
            f"Recall@5 {recall.recall_at(5):.3f} · Recall@10 {recall.recall_at(10):.3f} · "
            f"MRR {recall.mrr():.3f} · per class @10: {per_class} · missed: {missed}"
        )

    if run.traversal is not None:
        lines += ["", "# structural traversal against expected_related", "",
                  run.traversal.render()]
    lines += ["", "# by router class", ""]
    for cls in sorted({o.router_class for o in run.outcomes}):
        rows = [o for o in answered if o.router_class == cls]
        claims = sum(o.claims for o in rows)
        coverage = f"{sum(o.cited for o in rows) / claims:.3f}" if claims else "—"
        lines.append(
            f"- {cls:<10} {len(rows):>2} answered · Coverage {coverage} · "
            f"retried {sum(o.retried for o in rows)} · "
            f"TTFT p50 {_seconds(percentile([o.ttft for o in rows if o.ttft is not None], 0.5))} · "
            f"total p50 {_seconds(percentile([o.total for o in rows if o.total is not None], 0.5))}"
        )

    lines += [
        "",
        "# per question",
        "",
        "| id | class | route taken | gold rank | claims | cited | retried | TTFT | total "
        "| packed |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for o in run.outcomes:
        lines.append(
            f"| {o.id} | {o.router_class} | {o.route_taken or '—'} | "
            f"{o.gold_rank if o.gold_rank is not None else 'miss'} | {o.claims} | {o.cited} | "
            f"{'yes' if o.retried else ''} | {_seconds(o.ttft)} | {_seconds(o.total)} | "
            f"{o.packed_tokens} |"
        )

    # The answers themselves. The numbers above are about these sentences, and
    # a Coverage figure nobody can read the answers behind is taken on trust.
    lines += ["", "# the answers, as kept", ""]
    for o in run.outcomes:
        question = run.queries.get(o.id, "").strip().splitlines()
        asked = question[0] + (" …" if len(question) > 1 else "") if question else ""
        lines += [f"**{o.id}** — {asked}", ""]
        answer = (o.text.strip() or "(no answer)").splitlines()
        lines += [f"> {line}" if line.strip() else ">" for line in answer]
        lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Running it for real
# --------------------------------------------------------------------------


def _environment(llm_model: str, embed_model: str) -> dict:
    """What produced the numbers. Each line is best-effort: a missing tool
    leaves the line out rather than failing the run."""
    env: dict[str, str] = {"date": time.strftime("%Y-%m-%d")}
    try:
        with urllib.request.urlopen(f"{config.OLLAMA_URL}/api/version", timeout=5) as r:
            env["ollama"] = json.loads(r.read().decode("utf-8")).get("version", "?")
        with urllib.request.urlopen(f"{config.OLLAMA_URL}/api/tags", timeout=5) as r:
            models = json.loads(r.read().decode("utf-8")).get("models", [])
        for wanted in (llm_model, embed_model):
            for entry in models:
                if entry["name"] in (wanted, f"{wanted}:latest"):
                    details = entry.get("details", {})
                    env[entry["name"]] = (
                        f"{details.get('parameter_size', '?')} "
                        f"{details.get('quantization_level', '?')}, "
                        f"digest {entry.get('digest', '?')[:12]}"
                    )
    except (OSError, ValueError, KeyError):
        pass
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        if gpu:
            env["gpu"] = gpu.splitlines()[0]
    except (OSError, subprocess.SubprocessError):
        pass
    env["temperature"] = str(config.LLM_TEMPERATURE)
    return env


def main(argv: list[str] | None = None) -> int:
    from api.cli import _driver, _utf8_streams, build_runtime
    from index.ollama import OllamaLLM

    _utf8_streams()
    parser = argparse.ArgumentParser(prog="python -m eval.local_gates", description=__doc__)
    parser.add_argument("--out", type=Path, default=REPORT_PATH, help="where to write the report")
    args = parser.parse_args(argv)

    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    driver = _driver()
    try:
        driver.verify_connectivity()
    except Exception as exc:                       # noqa: BLE001 - report, don't mask
        print(f"Neo4j unavailable at {config.NEO4J_URI}: {exc}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as scratch:
        # A cache of its own: the run must embed for itself, not be handed
        # vectors by whatever the developer indexed last.
        runtimes = {
            language: build_runtime(
                REPO_IDS[language], (config.REPO_ROOT / root).resolve(), ai="local",
                driver=driver, cache_path=Path(scratch) / f"{language}.sqlite",
            )
            for language, root in golden["corpus"].items()
        }
        any_runtime = next(iter(runtimes.values()))
        if not isinstance(any_runtime.llm, OllamaLLM) or not any_runtime.provider.semantic:
            for note in any_runtime.notes:
                print(f"# {note}", file=sys.stderr)
            print("This run needs both local models. Run: sei setup", file=sys.stderr)
            return 1

        try:
            started = time.perf_counter()
            for language, runtime in runtimes.items():
                asyncio.run(runtime.indexer.sync_project(runtime.repo))
                print(f"# indexed {language}: {runtime.status()['graph']}", file=sys.stderr)
            index_seconds = time.perf_counter() - started
            any_runtime.llm.warm()

            def progress(o: Outcome) -> None:
                print(f"# {o.id} {o.route_taken:<10} coverage {o.coverage:.2f} "
                      f"gold {o.gold_rank or 'miss'} ttft {_seconds(o.ttft)} "
                      f"total {_seconds(o.total)}{' (retried)' if o.retried else ''}",
                      file=sys.stderr)

            run = measure(runtimes, golden, progress=progress)
            run.index_seconds = index_seconds
            run.environment = _environment(config.LOCAL_LLM_MODEL, config.LOCAL_EMBED_MODEL)
            args.out.write_text(render(run), encoding="utf-8")
            print(f"wrote {args.out}", file=sys.stderr)
        finally:
            # Leave the database as it was found: only this run's two repositories.
            with driver.session(database=config.NEO4J_DEFAULT_DATABASE) as session:
                for repo in REPO_IDS.values():
                    session.run("MATCH (n {repo_id: $repo}) DETACH DELETE n", repo=repo)
            for runtime in runtimes.values():
                runtime.indexer.cache.close()
            driver.close()
    return 0


if __name__ == "__main__":                         # pragma: no cover
    raise SystemExit(main())
