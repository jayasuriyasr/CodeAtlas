"""§5.1 STRUCTURAL — entity resolve, then a Cypher traversal *is* the answer.

"what calls `charge_card`" has an exact answer one traversal away. Sending it
to the semantic path answers a question nobody asked, with an approximate
search over a question whose answer is known. So this path does two things and
no ranking:

1. **Resolve the entity** the question names — a backticked identifier first,
   then anything identifier-shaped, then plain words — against the graph. An
   exact `qualified_name` wins; otherwise every symbol whose name or dotted
   suffix matches is a candidate. More than `RERANK_TOP_K` candidates is a
   refusal rather than a pick (§5.1's house rule), and the caller falls back to
   SEMANTIC.
2. **Traverse** the relationship the router's rule names, one hop, over the
   edge types that mean it. The relationship types come from the fixed table
   below and are compared as values, never substituted into the query text.

`subclasses` follows `INHERITS` edges into the entity: the classes that name
it as a base. One hop, like every other rule — direct subclasses only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

import metrics
from config import RERANK_TOP_K, STRUCTURAL_RESULT_LIMIT

#: Router rule -> (direction from the entity, relationship types). "in" means
#: edges pointing *at* the entity (its callers, importers, users).
TRAVERSALS: dict[str, tuple[str, tuple[str, ...]]] = {
    "callers": ("in", ("CALLS", "INVOKES_HOOK")),
    "callees": ("out", ("CALLS", "INVOKES_HOOK")),
    "usages": (
        "in",
        ("CALLS", "IMPORTS", "INVOKES_HOOK", "RENDERS", "DISPATCHES_TO",
         "USES_SERIALIZER", "HAS_FIELD", "INHERITS"),
    ),
    "definition": ("in", ("DEFINES",)),
    "imports": ("in", ("IMPORTS",)),
    "routes": ("both", ("DISPATCHES_TO", "RENDERS")),
    "subclasses": ("in", ("INHERITS",)),
}

_BACKTICKED = re.compile(r"`([^`\n]+)`")
#: Identifier-shaped: snake_case, camelCase, PascalCase, dotted paths, or a
#: name followed by `()`. Plain lower-case words are not, on purpose.
_CODE_SHAPED = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+"      # dotted
    r"|[A-Za-z]*_[A-Za-z0-9_]*"                                      # snake_case
    r"|[a-z]+[A-Z][A-Za-z0-9]*"                                      # camelCase
    r"|[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*"                              # PascalCase
    r"|[A-Za-z_][A-Za-z0-9_]*(?=\(\)))"                              # name()
)
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

#: Words that appear in structural questions and never name a symbol.
_STOP = frozenset(
    """
    a an the what who which where how is are was were does do did of to in on
    for by from with and or not it its this that these those there here call
    calls called caller callers calling callee callees use uses used usage usages
    reference references referenced define defined declared declare implemented
    implement implementation implementations implementors subclass subclasses
    import imports imported route routes view views handler handlers handle
    handles function functions method methods class classes module modules
    component components hook hooks endpoint endpoints code repo repository
    me show list find all any every
    """.split()
)


def entity_terms(question: str) -> list[str]:
    """Candidate entity names, most specific first. Deterministic."""
    terms: list[str] = []

    def add(term: str) -> None:
        term = term.strip().strip(".()").strip()
        if term and term not in terms and term.lower() not in _STOP:
            terms.append(term)

    for match in _BACKTICKED.finditer(question):
        add(match.group(1))
    if terms:
        return terms
    for match in _CODE_SHAPED.finditer(question):
        add(match.group(1))
    if terms:
        return terms
    # Plain words, last first: "who calls login" names the entity at the end.
    for word in reversed(_WORD.findall(question)):
        if len(word) >= 3:
            add(word)
    return terms


class Runner(Protocol):
    def run(self, query: str, **params) -> list: ...


RESOLVE_CYPHER = """
UNWIND $terms AS t
MATCH (s:Symbol {repo_id: $repo_id, name: t.name})
WHERE s.qualified_name = t.term
   OR s.name = t.term
   OR s.qualified_name ENDS WITH ('.' + t.term)
RETURN t.term AS term, s.uid AS uid, s.qualified_name AS qualified_name,
       s.qualified_name = t.term AS exact
ORDER BY s.qualified_name
"""

_TRAVERSE = {
    "in": """
        MATCH (t:Symbol {repo_id: $repo_id}) WHERE t.uid IN $uids
        MATCH (n:Symbol {repo_id: $repo_id})-[r]->(t)
        WHERE type(r) IN $types AND NOT n.uid IN $uids
        RETURN DISTINCT n.uid AS uid, n.rel_path AS rel_path,
               n.start_line AS start_line, type(r) AS rel
        ORDER BY rel_path, start_line, uid
    """,
    "out": """
        MATCH (t:Symbol {repo_id: $repo_id}) WHERE t.uid IN $uids
        MATCH (t)-[r]->(n:Symbol {repo_id: $repo_id})
        WHERE type(r) IN $types AND NOT n.uid IN $uids
        RETURN DISTINCT n.uid AS uid, n.rel_path AS rel_path,
               n.start_line AS start_line, type(r) AS rel
        ORDER BY rel_path, start_line, uid
    """,
    "both": """
        MATCH (t:Symbol {repo_id: $repo_id}) WHERE t.uid IN $uids
        MATCH (t)-[r]-(n:Symbol {repo_id: $repo_id})
        WHERE type(r) IN $types AND NOT n.uid IN $uids
        RETURN DISTINCT n.uid AS uid, n.rel_path AS rel_path,
               n.start_line AS start_line, type(r) AS rel
        ORDER BY rel_path, start_line, uid
    """,
}


@dataclass
class StructuralAnswer:
    """What the traversal found, and how much of it is being shown."""

    rule: str
    term: str | None = None
    entity_uids: list[str] = field(default_factory=list)
    entity_names: list[str] = field(default_factory=list)
    related_uids: list[str] = field(default_factory=list)
    total_related: int = 0
    #: Why the traversal could not answer, when it could not. None on success.
    fallback: str | None = None

    def note(self) -> dict:
        note = {
            "rule": self.rule,
            "entity": self.entity_names,
            "related_total": self.total_related,
            "related_shown": len(self.related_uids),
        }
        if self.fallback:
            note["fallback"] = self.fallback
        return note


@dataclass
class StructuralResolver:
    runner: Runner
    max_entities: int = RERANK_TOP_K
    result_limit: int = STRUCTURAL_RESULT_LIMIT

    def answer(self, repo_id: str, question: str, rule: str | None) -> StructuralAnswer:
        answer = StructuralAnswer(rule=rule or "unknown")
        terms = entity_terms(question)
        if not terms:
            metrics.incr("structural.no_entity")
            answer.fallback = "no entity named in the question"
            return answer

        rows = self.runner.run(
            RESOLVE_CYPHER,
            repo_id=repo_id,
            terms=[{"term": t, "name": t.rsplit(".", 1)[-1]} for t in terms],
        )
        by_term: dict[str, list[dict]] = {}
        for row in rows:
            by_term.setdefault(row["term"], []).append(row)

        for term in terms:                      # most specific term first
            found = by_term.get(term)
            if not found:
                continue
            exact = [r for r in found if r["exact"]]
            chosen = exact or found
            if len(chosen) > self.max_entities:
                metrics.incr("structural.ambiguous")
                answer.term = term
                answer.fallback = (
                    f"`{term}` names {len(chosen)} symbols; refusing to pick one"
                )
                return answer
            answer.term = term
            answer.entity_uids = [r["uid"] for r in chosen]
            answer.entity_names = [r["qualified_name"] for r in chosen]
            break
        else:
            metrics.incr("structural.unresolved")
            answer.fallback = "no symbol in the graph matches the question"
            return answer

        traversal = TRAVERSALS.get(rule or "")
        if traversal is None:
            # A rule with no edge type behind it. The entity is still useful
            # context for the semantic path.
            metrics.incr("structural.no_traversal")
            answer.fallback = f"the graph has no edge type for {rule!r}"
            return answer

        direction, types = traversal
        related = self.runner.run(
            _TRAVERSE[direction],
            repo_id=repo_id,
            uids=answer.entity_uids,
            types=list(types),
        )
        seen: list[str] = []
        for row in related:
            if row["uid"] not in seen:
                seen.append(row["uid"])
        answer.total_related = len(seen)
        answer.related_uids = seen[: self.result_limit]
        metrics.incr("structural.answered")
        return answer
