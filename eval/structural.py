"""Precision and recall of the STRUCTURAL traversal, against `expected_related`.

The golden set's `gold` for a structural question is the entity it names, so
Recall@10 on that class measures whether the entity was found — not whether
"what calls X" returned X's callers. Every STRUCTURAL question now also carries
`expected_related`, the full answer read off the fixture's source, and this
scores what the traversal returned against it.

Counted over symbols, micro-averaged: a question with six expected callers
weighs six times one with a single caller, which is what "how many of the
callers did it find" means.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable


@dataclass
class TraversalResult:
    id: str
    query: str
    rule: str
    route_taken: str
    expected: set[str]
    returned: set[str]
    fallback: str = ""

    @property
    def hits(self) -> set[str]:
        return self.expected & self.returned

    @property
    def missed(self) -> set[str]:
        return self.expected - self.returned

    @property
    def extra(self) -> set[str]:
        return self.returned - self.expected


@dataclass
class TraversalReport:
    results: list[TraversalResult] = field(default_factory=list)

    @property
    def precision(self) -> float | None:
        returned = sum(len(r.returned) for r in self.results)
        return sum(len(r.hits) for r in self.results) / returned if returned else None

    @property
    def recall(self) -> float | None:
        expected = sum(len(r.expected) for r in self.results)
        return sum(len(r.hits) for r in self.results) / expected if expected else None

    def render(self) -> str:
        def fmt(value: float | None) -> str:
            return "—" if value is None else f"{value:.3f}"

        lines = [
            f"Structural traversal over {len(self.results)} questions: precision "
            f"{fmt(self.precision)}, recall {fmt(self.recall)} "
            f"({sum(len(r.hits) for r in self.results)} of "
            f"{sum(len(r.expected) for r in self.results)} expected symbols found, "
            f"{sum(len(r.extra) for r in self.results)} returned that were not expected)",
            "",
            "| id | rule | route taken | found | missed | not expected |",
            "|---|---|---|---|---|---|",
        ]
        for r in self.results:
            lines.append(
                f"| {r.id} | {r.rule} | {r.route_taken} | {len(r.hits)}/{len(r.expected)} | "
                f"{', '.join(sorted(r.missed)) or ''} | {', '.join(sorted(r.extra)) or ''} |"
            )
        return "\n".join(lines)


def traversal_result(question: dict, notes: dict) -> TraversalResult:
    """Score one question from the notes the retriever returned for it."""
    structural = notes.get("structural") or {}
    route = notes.get("route_taken") or notes.get("route", "")
    returned = set()
    if route == "STRUCTURAL":
        returned = {r["qualified_name"] for r in structural.get("related", [])}
    return TraversalResult(
        id=question["id"],
        query=question["query"],
        rule=structural.get("rule", ""),
        route_taken=route,
        expected={qn for _path, qn in question.get("expected_related", [])},
        returned=returned,
        fallback=structural.get("fallback", "") or "",
    )


def measure_traversal(
    questions: Iterable[dict], notes_for: Callable[[str], dict]
) -> TraversalReport:
    """`notes_for(query)` is the retriever's notes for one question."""
    report = TraversalReport()
    for question in questions:
        if question.get("class") != "STRUCTURAL" or "expected_related" not in question:
            continue
        report.results.append(traversal_result(question, notes_for(question["query"])))
    return report
