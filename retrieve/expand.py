"""§5.2 step 3 and §11.1 — bounded 1-hop neighbourhood expansion.

§5.2: "Step 3 is the justification for running a graph database. Without it,
Neo4j is an expensive Postgres."

Two guards make it bounded rather than explosive. The hub cutoff drops
neighbours whose degree exceeds `$hub_cutoff` — without it a 400-caller `logger`
is one hop from everything and floods every expansion. The score then divides by
`log(2 + degree)`, so a generic neighbour that does survive the cutoff still
ranks below a specific one.

§12.8 is honest about what that is: a proxy, not centrality. In densely
interconnected modules it will misrank.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from config import (
    EDGE_WEIGHT_DEFAULT,
    EDGE_WEIGHTS,
    GLOBAL_EXPANSION_LIMIT,
    HUB_CUTOFF,
    PER_SEED_LIMIT,
)

#: §11.1 (v10.1). Multi-type patterns are legal in MATCH; MERGE does not
#: accept them (probe_merge_multitype), which is why §11.2 3b is split by type.
#: All eight allowlisted types are listed: a type the expansion does not name is
#: a type that can be written and never read (v10.1 §0.0 R2).
EXPANSION_CYPHER = """
UNWIND $seed_uids AS seed_uid
MATCH (seed:Symbol {uid: seed_uid, repo_id: $repo_id})
CALL {
    WITH seed
    MATCH (seed)-[r:CALLS|IMPORTS|DEFINES|DISPATCHES_TO
                  |INVOKES_HOOK|RENDERS|HAS_FIELD|USES_SERIALIZER]-(n:Symbol)
    WHERE n.uid <> seed.uid
      AND n.repo_id = $repo_id
      AND coalesce(n.degree, 0) <= $hub_cutoff
    RETURN n, type(r) AS rel
    ORDER BY coalesce(n.degree, 0) ASC
    LIMIT $per_seed_limit
}
WITH n, rel, count(DISTINCT seed) AS seed_support
WITH n,
     sum(seed_support * coalesce($edge_weights[rel], $edge_weight_default)) AS raw,
     coalesce(n.degree, 1) AS deg
RETURN n.uid AS uid, n.qualified_name AS qualified_name, n.rel_path AS rel_path,
       n.signature AS signature, n.docstring AS docstring,
       raw / log(2.0 + deg) AS score
ORDER BY score DESC
LIMIT $global_limit
"""


@dataclass(frozen=True)
class Neighbor:
    uid: str
    qualified_name: str
    rel_path: str
    signature: str | None
    docstring: str | None
    score: float


class Runner(Protocol):
    def run(self, query: str, **params) -> list: ...


@dataclass
class NeighborhoodExpander:
    """One statement, parameterised by the † constants in `config.py`."""

    runner: Runner
    hub_cutoff: int = HUB_CUTOFF
    per_seed_limit: int = PER_SEED_LIMIT
    global_limit: int = GLOBAL_EXPANSION_LIMIT

    def expand(self, repo_id: str, seed_uids: list[str]) -> list[Neighbor]:
        """Neighbours of the seeds, hub-penalised and globally capped.

        The aggregation is two-stage and the order matters:
        `count(DISTINCT seed)` groups by (n, rel), then `sum` groups by n,
        summing across relationship types. Collapsing that into one stage would
        count a neighbour reached by two edge types once instead of twice.
        """
        if not seed_uids:
            return []
        rows = self.runner.run(
            EXPANSION_CYPHER,
            seed_uids=list(dict.fromkeys(seed_uids)),   # dedupe, keep order
            repo_id=repo_id,
            hub_cutoff=self.hub_cutoff,
            per_seed_limit=self.per_seed_limit,
            global_limit=self.global_limit,
            edge_weights=EDGE_WEIGHTS,
            edge_weight_default=EDGE_WEIGHT_DEFAULT,
        )
        return [
            Neighbor(
                uid=row["uid"],
                qualified_name=row["qualified_name"],
                rel_path=row["rel_path"],
                signature=row["signature"],
                docstring=row["docstring"],
                score=row["score"],
            )
            for row in rows
        ]
