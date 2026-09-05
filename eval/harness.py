"""S7 — the eval harness.

§10.1 puts the harness on the never-cut list, with the reason stated plainly:
"twenty questions with gold UIDs beats forty-five planned and zero written."

**What this measures at step 6, and what it does not.** §8.1 requires recall to
be measured against **packed** context — "a gold node retrieved and then dropped
by the packer never reached the model". The packer does not exist until step 8,
so the seed set measures recall against *retrieved candidates*, which is a
strictly more lenient number. `stage` records which was measured, so the two are
never silently compared. Recall against packed context can only fall.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from config import GATE_MRR, GATE_RECALL_AT_10

SEED_SET = Path(__file__).parent / "golden" / "retrieval_seed.json"


@dataclass(frozen=True)
class Question:
    id: str
    query: str
    router_class: str
    #: (rel_path, qualified_name) pairs. Names, not UIDs — a UID is a hash over
    #: `repo_id` and is neither readable nor reviewable in a diff.
    gold: tuple[tuple[str, str], ...]
    note: str = ""


@dataclass
class QuestionResult:
    question: Question
    gold_uids: set[str]
    ranked: list[str]
    #: 1-based rank of the first gold hit, or None.
    first_hit: int | None = None
    #: Gold symbols the corpus does not contain. A question whose gold cannot be
    #: resolved is excluded from the score rather than counted as a miss —
    #: otherwise a typo in the golden file reads as a retrieval regression.
    unresolved_gold: tuple[str, ...] = ()

    @property
    def scoreable(self) -> bool:
        return bool(self.gold_uids)

    def hit_at(self, k: int) -> bool:
        return self.first_hit is not None and self.first_hit <= k


@dataclass
class EvalReport:
    stage: str
    corpus: str
    results: list[QuestionResult] = field(default_factory=list)
    #: Recorded alongside every score. §8.2: without the model name and version,
    #: neither the gate nor §5.4's break-even can be checked.
    embedding_model: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def scoreable(self) -> list[QuestionResult]:
        return [r for r in self.results if r.scoreable]

    def recall_at(self, k: int) -> float:
        scoreable = self.scoreable
        if not scoreable:
            return 0.0
        return sum(1 for r in scoreable if r.hit_at(k)) / len(scoreable)

    def mrr(self) -> float:
        """Mean reciprocal rank over the first gold hit."""
        scoreable = self.scoreable
        if not scoreable:
            return 0.0
        return sum(
            (1.0 / r.first_hit) if r.first_hit else 0.0 for r in scoreable
        ) / len(scoreable)

    def by_class(self, k: int = 10) -> dict[str, float]:
        """Per-class recall. §8.2 requires per-class reporting with the
        resolution stated: at 15 questions per class one flip moves a class
        score ~6.7 points."""
        out: dict[str, float] = {}
        for cls in sorted({r.question.router_class for r in self.scoreable}):
            rows = [r for r in self.scoreable if r.question.router_class == cls]
            out[cls] = sum(1 for r in rows if r.hit_at(k)) / len(rows)
        return out

    @property
    def resolution_pp(self) -> float:
        """How many percentage points one flipped question is worth."""
        scoreable = self.scoreable
        return 100.0 / len(scoreable) if scoreable else 0.0

    def misses(self, k: int = 10) -> list[QuestionResult]:
        return [r for r in self.scoreable if not r.hit_at(k)]

    def render(self) -> str:
        lines = [
            f"stage={self.stage}  corpus={self.corpus}",
            f"embedding model: {self.embedding_model or 'UNRECORDED'}",
            f"questions: {len(self.results)} ({len(self.scoreable)} scoreable)",
            f"resolution: one flipped question = {self.resolution_pp:.1f}pp",
            "",
            f"Recall@1  {self.recall_at(1):.3f}",
            f"Recall@5  {self.recall_at(5):.3f}",
            f"Recall@10 {self.recall_at(10):.3f}   (§8.2 gate >= {GATE_RECALL_AT_10})",
            f"MRR       {self.mrr():.3f}   (§8.2 gate >= {GATE_MRR})",
            "",
            "per class @10: "
            + ", ".join(f"{c}={v:.2f}" for c, v in self.by_class().items()),
        ]
        missed = self.misses()
        if missed:
            lines += ["", "missed at 10:"]
            lines += [f"  {r.question.id}  {r.question.query!r}" for r in missed]
        unresolved = [r for r in self.results if r.unresolved_gold]
        if unresolved:
            lines += ["", "gold not present in the corpus (excluded from scoring):"]
            lines += [
                f"  {r.question.id}: {', '.join(r.unresolved_gold)}" for r in unresolved
            ]
        for note in self.notes:
            lines += ["", note]
        return "\n".join(lines)


def load_questions(path: Path = SEED_SET) -> tuple[list[Question], dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    questions = [
        Question(
            id=q["id"],
            query=q["query"],
            router_class=q.get("class", "SEMANTIC"),
            gold=tuple((g[0], g[1]) for g in q["gold"]),
            note=q.get("note", ""),
        )
        for q in raw["questions"]
    ]
    return questions, raw


class EvalHarness:
    """`EvalHarness.run()` — S7.

    `retrieve` is injected rather than constructed so the harness can score any
    stage: raw candidates at step 6, packed blocks at step 8, and the full
    answer path at step 10. The metric definition does not change, only what is
    handed to it — which is §1's "a metric named twice is defined once".
    """

    def __init__(
        self,
        resolve_uids: Callable[[str, str], list[str]],
        retrieve: Callable[[Question], Sequence[str]],
        *,
        stage: str = "retrieved",
        embedding_model: str = "",
    ) -> None:
        self._resolve_uids = resolve_uids
        self._retrieve = retrieve
        self._stage = stage
        self._embedding_model = embedding_model

    def run(
        self, questions: Iterable[Question], *, corpus: str = ""
    ) -> EvalReport:
        report = EvalReport(
            stage=self._stage, corpus=corpus, embedding_model=self._embedding_model
        )

        for question in questions:
            gold_uids: set[str] = set()
            unresolved: list[str] = []
            for rel_path, qualified_name in question.gold:
                uids = self._resolve_uids(rel_path, qualified_name)
                if uids:
                    gold_uids.update(uids)
                else:
                    unresolved.append(f"{rel_path}::{qualified_name}")

            ranked = list(self._retrieve(question))
            first_hit = next(
                (i for i, uid in enumerate(ranked, start=1) if uid in gold_uids), None
            )
            report.results.append(
                QuestionResult(
                    question=question,
                    gold_uids=gold_uids,
                    ranked=ranked,
                    first_hit=first_hit,
                    unresolved_gold=tuple(unresolved),
                )
            )

        return report
