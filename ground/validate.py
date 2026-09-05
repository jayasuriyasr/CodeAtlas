"""§6 — the three validation tiers and the three signals they produce.

| Tier | Check | Cost | When |
|---|---|---|---|
| T1 | Handle exists in the map | dict lookup | during the stream |
| T2 | Claim sentence has >=1 handle | heuristic classifier | during the stream |
| T3 | Relational claims verified in graph | 1 batched Cypher | post-stream, ~40ms |

§6.2 keeps the signals separate rather than blending them, and gives the
reason: "Coverage 0.9 / RelationPrec 0.4 is 'well-sourced but misreading the
code' — a different bug from 0.4 / 0.95, 'right but improvising.' One blended
number erases the distinction."
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import metrics
from config import COVERAGE_RETRY_THRESHOLD, EDGE_TYPE_ALLOWLIST, MAX_RETRIES
from ground.classify import ClassifiedAnswer, Sentence, classify

# --------------------------------------------------------------------------
# T1 — the handle exists
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HandleError:
    handle: int
    sentence: str


def validate_handles(
    answer: ClassifiedAnswer, handle_map: dict[str, str]
) -> list[HandleError]:
    """T1. Every cited handle must name a block that was actually packed.

    A dict lookup, which is the entire point: §14 calls this "grounding [that]
    is deterministic where it can be", cheap enough to run during the stream.

    §5.4's handle assignment is what makes this meaningful — handles are
    assigned *after* packing, so a gap in `C1..Cn` is impossible and a
    `[C12]` against nine blocks is genuinely the model inventing one.
    """
    known = set(handle_map)
    errors = [
        HandleError(handle=handle, sentence=sentence.text)
        for sentence in answer.sentences
        for handle in sentence.handles
        if f"C{handle}" not in known
    ]
    if errors:
        metrics.incr("ground.t1.hallucinated_handle", len(errors))
    return errors


# --------------------------------------------------------------------------
# T3 — relational claims
# --------------------------------------------------------------------------

_BACKTICKED = r"`([A-Za-z_][\w.]*)`"

#: Relation phrasings, mapped to the §11.2 allowlist. Deliberately narrow: a
#: verb list that tried to cover every phrasing would extract triples from
#: sentences that assert nothing, and §6.2's RelationPrec denominator would
#: then be full of things the model never claimed.
_RELATION_PATTERNS: tuple[tuple[str, re.Pattern[str], bool], ...] = (
    ("CALLS", re.compile(_BACKTICKED + r"\s+(?:calls|invokes)\s+" + _BACKTICKED), False),
    ("CALLS", re.compile(_BACKTICKED + r"\s+is\s+called\s+by\s+" + _BACKTICKED), True),
    ("IMPORTS", re.compile(_BACKTICKED + r"\s+imports\s+" + _BACKTICKED), False),
    ("DEFINES", re.compile(_BACKTICKED + r"\s+defines\s+" + _BACKTICKED), False),
    (
        "DEFINES",
        re.compile(_BACKTICKED + r"\s+is\s+defined\s+(?:in|by)\s+" + _BACKTICKED),
        True,
    ),
    (
        "DISPATCHES_TO",
        re.compile(_BACKTICKED + r"\s+(?:dispatches to|routes to)\s+" + _BACKTICKED),
        False,
    ),
)


@dataclass(frozen=True)
class RelationClaim:
    subject: str
    relation: str
    object: str
    sentence_index: int

    def __post_init__(self) -> None:
        if self.relation not in EDGE_TYPE_ALLOWLIST:
            raise ValueError(
                f"relation {self.relation!r} is not in the §11.2 3b allowlist "
                f"{EDGE_TYPE_ALLOWLIST}"
            )


@dataclass
class RelationVerdict:
    claim: RelationClaim
    subject_found: bool
    object_found: bool
    edge_found: bool

    @property
    def verified(self) -> bool:
        return self.edge_found

    @property
    def contradicted(self) -> bool:
        """Both endpoints are in the graph, and the asserted edge is not.

        The distinction from "unverified" is load-bearing. §12.3 lists what the
        graph deliberately cannot see — `ForeignKey('app.Model')`,
        `AUTH_USER_MODEL`, `include()`, Celery task names — so an unresolvable
        endpoint means "the graph does not know", not "the model is wrong".
        Counting those as conflicts would fire §6.3's retry on answers that
        were right.
        """
        return self.subject_found and self.object_found and not self.edge_found


def extract_relation_claims(answer: ClassifiedAnswer) -> list[RelationClaim]:
    """Pull `(subject, relation, object)` triples out of claim sentences.

    Only claim sentences are scanned: a hedged or interrogative sentence
    asserts no relation, and §6.2's denominator is *asserted* relations.
    """
    claims: list[RelationClaim] = []
    for index, sentence in enumerate(answer.sentences):
        if not sentence.is_claim:
            continue
        for relation, pattern, inverted in _RELATION_PATTERNS:
            for match in pattern.finditer(sentence.text):
                left, right = match.group(1), match.group(2)
                subject, obj = (right, left) if inverted else (left, right)
                claims.append(
                    RelationClaim(
                        subject=subject,
                        relation=relation,
                        object=obj,
                        sentence_index=index,
                    )
                )
    return claims


#: One statement for every claim (§6.1's tier table: "1 batched Cypher").
#: `COUNT { … }` is probe_count_subquery's construct; the relation type is
#: compared as a value rather than substituted into the pattern, so nothing
#: from model output reaches the query text.
#: A model names a symbol the way a developer does: `charge_card`,
#: `SubscriptionView.post`, occasionally the full `billing.views.SubscriptionView.post`.
#: All three have to resolve, so the match is exact-or-dotted-suffix. Requiring
#: the full qualified name would leave nearly every real claim `subject_found =
#: false` — reported as unverified rather than contradicted, which is the safe
#: direction but makes RelationPrec measure nothing.
#:
#: `ENDS WITH` cannot use an index. That is affordable here and nowhere else:
#: T3 is one post-stream query over the handful of relations in one answer
#: (§6.1 prices it at ~40ms), not a hot path.
_MATCHES = (
    "(%(n)s.qualified_name = c.%(f)s OR %(n)s.name = c.%(f)s "
    "OR %(n)s.qualified_name ENDS WITH ('.' + c.%(f)s))"
)
_SUBJ = _MATCHES % {"n": "a", "f": "subject"}
_OBJ = _MATCHES % {"n": "b", "f": "object"}

VERIFY_CYPHER = f"""
UNWIND $claims AS c
RETURN c.idx AS idx,
  COUNT {{
    MATCH (a:Symbol {{repo_id: $repo_id}})
    WHERE {_SUBJ}
  }} AS subjects,
  COUNT {{
    MATCH (b:Symbol {{repo_id: $repo_id}})
    WHERE {_OBJ}
  }} AS objects,
  COUNT {{
    MATCH (a:Symbol {{repo_id: $repo_id}})-[r]->(b:Symbol {{repo_id: $repo_id}})
    WHERE {_SUBJ} AND {_OBJ} AND type(r) = c.relation
  }} AS edges
"""


class Runner(Protocol):
    def run(self, query: str, **params) -> list: ...


def verify_relations(
    runner: Runner, repo_id: str, claims: Sequence[RelationClaim]
) -> list[RelationVerdict]:
    """T3. One query for every claim in the answer."""
    if not claims:
        return []

    rows = runner.run(
        VERIFY_CYPHER,
        repo_id=repo_id,
        claims=[
            {
                "idx": i,
                "subject": c.subject,
                "relation": c.relation,
                "object": c.object,
            }
            for i, c in enumerate(claims)
        ],
    )
    metrics.incr("ground.t3.queries")

    by_idx = {row["idx"]: row for row in rows}
    verdicts = []
    for i, claim in enumerate(claims):
        row = by_idx.get(i)
        verdicts.append(
            RelationVerdict(
                claim=claim,
                subject_found=bool(row and row["subjects"]),
                object_found=bool(row and row["objects"]),
                edge_found=bool(row and row["edges"]),
            )
        )
    return verdicts


# --------------------------------------------------------------------------
# §6.2 — the three signals
# --------------------------------------------------------------------------


@dataclass
class GroundingReport:
    answer: ClassifiedAnswer
    handle_errors: list[HandleError] = field(default_factory=list)
    relation_verdicts: list[RelationVerdict] = field(default_factory=list)
    retry_count: int = 0

    @property
    def coverage(self) -> float:
        """cited claim-sentences / total claim-sentences.

        An answer with no claim sentences scores 1.0, not 0.0: there was
        nothing to cite, and §6.3 would otherwise fire a retry on "I could not
        find anything relevant", which is the one answer that most deserves to
        stand.
        """
        claims = self.answer.claims
        if not claims:
            return 1.0
        return len(self.answer.cited_claims) / len(claims)

    @property
    def relation_precision(self) -> float:
        """verified relations / asserted relations."""
        if not self.relation_verdicts:
            return 1.0
        verified = sum(1 for v in self.relation_verdicts if v.verified)
        return verified / len(self.relation_verdicts)

    @property
    def conflict_flags(self) -> int:
        """Claims contradicting a graph edge — not merely unverified ones."""
        return sum(1 for v in self.relation_verdicts if v.contradicted)

    @property
    def uncited_claims(self) -> list[Sentence]:
        return self.answer.uncited_claims

    def render(self) -> str:
        return (
            f"Coverage {self.coverage:.2f} · "
            f"RelationPrec {self.relation_precision:.2f} · "
            f"ConflictFlags {self.conflict_flags}"
        )


def should_retry(report: GroundingReport) -> bool:
    """§6.3's gate: `Coverage < 0.6† or ConflictFlags > 0`.

    Bounded to one attempt. §6.3: "One retry, bounded, never a loop." A second
    retry doubles cost and latency for an answer the first retry already failed
    to improve, and §6.3's own fallback is to "surface honestly, offer manual
    Deep Dive" rather than to keep trying.
    """
    if report.retry_count >= MAX_RETRIES:
        return False
    return (
        report.coverage < COVERAGE_RETRY_THRESHOLD or report.conflict_flags > 0
    )


def ground(
    text: str,
    handle_map: dict[str, str],
    *,
    known_names: frozenset[str] = frozenset(),
    runner: Runner | None = None,
    repo_id: str = "",
    retry_count: int = 0,
) -> GroundingReport:
    """Run T1, T2 and T3 over one answer.

    T3 is skipped when no graph runner is supplied — §9.3's degradation table
    has no row for "Neo4j unavailable" other than hard failure, but a *report*
    without T3 is still a useful T1/T2 report, and pretending relations were
    verified would be worse than saying nothing about them.
    """
    answer = classify(text, known_names)

    report = GroundingReport(answer=answer, retry_count=retry_count)
    report.handle_errors = validate_handles(answer, handle_map)

    claims = extract_relation_claims(answer)
    if claims and runner is not None:
        report.relation_verdicts = verify_relations(runner, repo_id, claims)
    elif claims:
        metrics.incr("ground.t3.skipped", len(claims))

    metrics.incr("ground.claims", len(answer.claims))
    metrics.incr("ground.claims_cited", len(answer.cited_claims))
    return report


# --------------------------------------------------------------------------
# §6.3 — what a retry does before it regenerates
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RetryPlan:
    """What the second attempt should change.

    §6.3: "If the packing report shows dropped blocks, the retry promotes those
    before expanding — unseen evidence is a likelier explanation for low
    coverage than an insufficient neighborhood."
    """

    promote: tuple[str, ...]
    expand_around: tuple[str, ...]
    reason: str


def plan_retry(
    report: GroundingReport,
    dropped_uids: Sequence[str],
    handle_map: dict[str, str],
) -> RetryPlan:
    """Promotion first, expansion second. §6.3's ordering, not a heuristic."""
    conflicting = tuple(
        handle_map[f"C{h}"]
        for v in report.relation_verdicts
        if v.contradicted
        for h in report.answer.sentences[v.claim.sentence_index].handles
        if f"C{h}" in handle_map
    )

    if dropped_uids:
        return RetryPlan(
            promote=tuple(dropped_uids),
            expand_around=(),
            reason="dropped blocks promoted before expanding (§6.3)",
        )

    around = conflicting or tuple(
        handle_map[f"C{h}"]
        for s in report.answer.cited_claims
        for h in s.handles
        if f"C{h}" in handle_map
    )
    return RetryPlan(
        promote=(),
        expand_around=around,
        reason="bounded expansion around uncited or conflicting spans (§6.3)",
    )
