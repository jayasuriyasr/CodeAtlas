"""§5.1 — the query router.

Three classes, three paths:

| Class | Path |
|---|---|
| STRUCTURAL | entity resolve -> Cypher traversal *is* the answer |
| SEMANTIC   | hybrid RRF -> 1-hop expand -> synthesize |
| TRACE      | frame resolution -> UID lookup -> expand |

"Regex heuristics first, LLM fallback for ambiguity." The ordering matters more
than it looks: sending "what calls `charge_card`" to the semantic path answers
a question nobody asked, with an ANN search over a question whose answer is one
graph traversal away and exact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import metrics
from retrieve.frames import StackFrame, parse_stack_trace


class QueryClass(str, Enum):
    STRUCTURAL = "STRUCTURAL"
    SEMANTIC = "SEMANTIC"
    TRACE = "TRACE"


#: Questions the graph answers exactly. Each names a relationship, which is why
#: a traversal is the answer rather than an input to one.
_STRUCTURAL = (
    ("callers", re.compile(r"\b(what|who|which)\b.{0,20}\bcalls?\b", re.I)),
    ("callers", re.compile(r"\bcallers?\s+of\b", re.I)),
    ("callees", re.compile(r"\bwhat\s+does\b.{0,40}\bcall\b", re.I)),
    ("usages", re.compile(r"\b(where|what)\b.{0,20}\b(is|are)\b.{0,30}\bused\b", re.I)),
    ("usages", re.compile(r"\b(usages?|references?)\s+(of|to)\b", re.I)),
    ("definition", re.compile(r"\bwhere\s+is\b.{0,40}\b(defined|declared|implemented)\b", re.I)),
    ("subclasses", re.compile(r"\b(subclasses|implementations|implementors)\s+of\b", re.I)),
    ("imports", re.compile(r"\b(what|which)\b.{0,20}\bimports?\b", re.I)),
    ("routes", re.compile(r"\b(what|which)\s+(view|handler|route)\b.{0,30}\bhandles?\b", re.I)),
)

#: Explanatory questions. Matched only to keep them away from the LLM fallback —
#: they would be classified SEMANTIC anyway, and paying for a classification
#: whose answer is the default is the kind of cost that hides in a p50.
_SEMANTIC = (
    re.compile(r"\bhow\s+(does|do|is|are|can|should)\b", re.I),
    re.compile(r"\bwhy\b", re.I),
    re.compile(r"\b(explain|describe|walk me through|overview|summar)", re.I),
    re.compile(r"\bwhat\s+(is|are)\s+the\s+(purpose|point|idea)\b", re.I),
)


@dataclass(frozen=True)
class Routed:
    query_class: QueryClass
    #: How the decision was reached: "regex", "llm", or "default".
    decided_by: str
    #: The named regex that matched, when one did. Kept so a misroute can be
    #: traced to the pattern that caused it rather than guessed at.
    rule: str | None = None
    frames: tuple[StackFrame, ...] = field(default_factory=tuple)


#: An LLM classifier: query -> QueryClass or None. S2's `.stream()` behind a
#: narrower signature. Injected rather than constructed, so the router is
#: testable without a network call and without source egress.
LLMClassifier = Callable[[str], "QueryClass | None"]


@dataclass
class QueryRouter:
    llm_classify: LLMClassifier | None = None

    def route(self, query: str) -> Routed:
        frames = parse_stack_trace(query)
        if frames:
            metrics.incr("router.trace")
            return Routed(QueryClass.TRACE, "regex", "stack_trace", tuple(frames))

        for rule, pattern in _STRUCTURAL:
            if pattern.search(query):
                metrics.incr("router.structural")
                return Routed(QueryClass.STRUCTURAL, "regex", rule)

        for pattern in _SEMANTIC:
            if pattern.search(query):
                metrics.incr("router.semantic")
                return Routed(QueryClass.SEMANTIC, "regex", "explanatory")

        # Ambiguous: neither a trace, nor a named relationship, nor an
        # explanatory form. §5.1's LLM fallback covers exactly this gap.
        if self.llm_classify is not None:
            decided = self.llm_classify(query)
            if decided is not None:
                metrics.incr("router.llm_fallback")
                return Routed(decided, "llm")
            metrics.incr("router.llm_declined")

        metrics.incr("router.default_semantic")
        return Routed(QueryClass.SEMANTIC, "default")
