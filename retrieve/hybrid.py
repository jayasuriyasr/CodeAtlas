"""§5.2 step 1 — hybrid retrieval by Reciprocal Rank Fusion.

Two arms, fused by rank rather than by score. Fusing raw scores would be wrong:
cosine similarity and Lucene's BM25 are not on a common scale, and normalising
them requires knowing each arm's distribution, which changes per query.

§9.3 makes the degradation explicit — when the embedding API is unavailable the
system runs "fulltext + graph expansion" and tells the user "semantic search
degraded". So a missing vector arm is a supported state here, not an error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import metrics
from config import RRF_K, RRF_TOP_K
from retrieve.tokenize import build_fulltext_query

Vector = Sequence[float]


@dataclass(frozen=True)
class Candidate:
    """One retrieved symbol, with the evidence for why it is here."""

    uid: str
    score: float
    #: Per-arm rank, 1-based. Absent from an arm means that arm did not return it.
    vector_rank: int | None = None
    fulltext_rank: int | None = None

    @property
    def arms(self) -> tuple[str, ...]:
        found = []
        if self.vector_rank is not None:
            found.append("vector")
        if self.fulltext_rank is not None:
            found.append("fulltext")
        return tuple(found)


class SearchBackend(Protocol):
    """S1. `RetrievalBackend.search(vec, filters, k)`, split by arm.

    §2 rates S1 "cheap to call across; changes the recovery model (§12.1)" —
    the interface is narrow, the consequences of swapping it are not.
    """

    def vector_search(self, repo_id: str, vec: Vector, k: int) -> list[tuple[str, float]]: ...

    def fulltext_search(self, repo_id: str, query: str, k: int) -> list[tuple[str, float]]: ...


def reciprocal_rank_fusion(
    arms: dict[str, list[tuple[str, float]]],
    *,
    k: int = RRF_K,
    limit: int = RRF_TOP_K,
) -> list[Candidate]:
    """Fuse ranked lists: score(d) = sum over arms of 1 / (k + rank(d)).

    Ties are broken by uid so the output is deterministic. §8.2 runs eval at
    temperature 0 and treats a single flipped question as a warning, which only
    means something if retrieval itself does not flip between identical runs.
    """
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}

    for arm, results in arms.items():
        for position, (uid, _raw) in enumerate(results, start=1):
            scores[uid] = scores.get(uid, 0.0) + 1.0 / (k + position)
            ranks.setdefault(uid, {})[arm] = position

    fused = [
        Candidate(
            uid=uid,
            score=score,
            vector_rank=ranks[uid].get("vector"),
            fulltext_rank=ranks[uid].get("fulltext"),
        )
        for uid, score in scores.items()
    ]
    fused.sort(key=lambda c: (-c.score, c.uid))
    return fused[:limit]


@dataclass
class HybridRetriever:
    """§5.2 steps 1-2. Step 2's reranker is Phase 1; the MVP uses raw RRF."""

    backend: SearchBackend
    top_k: int = RRF_TOP_K
    #: Per-arm fetch depth. Wider than `top_k` on purpose: fusion can only
    #: promote a document some arm actually returned.
    arm_k: int = field(default=RRF_TOP_K * 2)

    def search(
        self, repo_id: str, query: str, query_vec: Vector | None = None
    ) -> list[Candidate]:
        arms: dict[str, list[tuple[str, float]]] = {}

        fulltext_query = build_fulltext_query(query)
        if fulltext_query:
            arms["fulltext"] = self.backend.fulltext_search(
                repo_id, fulltext_query, self.arm_k
            )

        if query_vec is not None:
            arms["vector"] = self.backend.vector_search(repo_id, query_vec, self.arm_k)
        else:
            # §9.3: "Embedding API -> Fulltext + graph expansion", surfaced to
            # the user as "Semantic search degraded". Counted so the banner has
            # something behind it.
            metrics.incr("retrieve.vector_arm_unavailable")

        if not arms:
            return []

        results = reciprocal_rank_fusion(arms, limit=self.top_k)
        metrics.incr("retrieve.candidates", len(results))
        return results


# --------------------------------------------------------------------------
# S1's one implementation: the Neo4j indexes from §11.3
# --------------------------------------------------------------------------


class Neo4jSearchBackend:
    """Both arms served by one process — §12.1's correlated failure domain.

    Graph traversal, vector search and full-text search are three indexes inside
    one Neo4j. §9.3 gives independent fallbacks for the embedding API and for a
    cold vector index; there is none for Neo4j itself, and splitting the vector
    store out is not free (§12.1 explains what it costs §4.3).
    """

    VECTOR_INDEX = "symbol_code_vec"
    FULLTEXT_INDEX = "symbol_search"

    def __init__(self, driver, database: str) -> None:
        self._driver = driver
        self._database = database

    def _run(self, cypher: str, **params):
        # Named `cypher`, not `query`: `fulltext_search` passes a Cypher
        # parameter called `$query`, and a positional arg of the same name
        # collides with it — TypeError, only on the fulltext path, only
        # against a live server.
        with self._driver.session(database=self._database) as session:
            return list(session.run(cypher, params))

    def vector_search(self, repo_id: str, vec: Vector, k: int) -> list[tuple[str, float]]:
        # Over-fetch before filtering by repo_id: the index is shared across
        # tenants (§9.1's MVP is one database with repo_id on every node), so a
        # top-k taken before the filter can be entirely another tenant's rows.
        rows = self._run(
            f"""
            CALL db.index.vector.queryNodes('{self.VECTOR_INDEX}', $fetch, $vec)
            YIELD node, score
            WHERE node.repo_id = $repo_id
            RETURN node.uid AS uid, score ORDER BY score DESC LIMIT $k
            """,
            fetch=k * 4,
            vec=list(vec),
            repo_id=repo_id,
            k=k,
        )
        return [(row["uid"], row["score"]) for row in rows]

    def fulltext_search(self, repo_id: str, query: str, k: int) -> list[tuple[str, float]]:
        rows = self._run(
            f"""
            CALL db.index.fulltext.queryNodes('{self.FULLTEXT_INDEX}', $query)
            YIELD node, score
            WHERE node.repo_id = $repo_id
            RETURN node.uid AS uid, score ORDER BY score DESC LIMIT $k
            """,
            query=query,
            repo_id=repo_id,
            k=k,
        )
        return [(row["uid"], row["score"]) for row in rows]
