"""Step 6 — §5.2 step 1, Reciprocal Rank Fusion over two arms.

Pure logic against a stub backend. What the fusion does with two ranked lists
is decidable without a database; whether Neo4j's two indexes return sensible
lists is not, and lives in `test_retrieval_graph.py`.
"""

from __future__ import annotations

import pytest

import metrics
from config import RRF_K, RRF_TOP_K
from retrieve.hybrid import Candidate, HybridRetriever, reciprocal_rank_fusion


@pytest.fixture(autouse=True)
def reset_metrics():
    metrics.reset()
    yield
    metrics.reset()


class StubBackend:
    """Returns whatever the test hands it, and records that it was asked."""

    def __init__(self, vector=None, fulltext=None):
        self._vector = vector or []
        self._fulltext = fulltext or []
        self.vector_calls: list[tuple] = []
        self.fulltext_calls: list[tuple] = []

    def vector_search(self, repo_id, vec, k):
        self.vector_calls.append((repo_id, k))
        return self._vector[:k]

    def fulltext_search(self, repo_id, query, k):
        self.fulltext_calls.append((repo_id, query, k))
        return self._fulltext[:k]


# --------------------------------------------------------------------------
# The fusion itself
# --------------------------------------------------------------------------


def test_rrf_merges_both_arms():
    """A hit from either arm alone must surface.

    The point of two arms is that they fail differently: vector search misses
    an exact identifier it has no semantic handle on, fulltext misses a
    paraphrase. Fusion is only worth its complexity if a single-arm hit
    survives it.
    """
    fused = reciprocal_rank_fusion(
        {
            "vector": [("only_vector", 0.9), ("both", 0.8)],
            "fulltext": [("only_fulltext", 12.0), ("both", 9.0)],
        }
    )
    uids = [c.uid for c in fused]

    assert set(uids) == {"only_vector", "only_fulltext", "both"}
    assert uids[0] == "both", "a document both arms found must outrank either alone"

    by_uid = {c.uid: c for c in fused}
    assert by_uid["both"].arms == ("vector", "fulltext")
    assert by_uid["only_vector"].arms == ("vector",)
    assert by_uid["only_fulltext"].arms == ("fulltext",)


def test_rrf_scores_by_rank_not_by_raw_score():
    """Cosine similarity and BM25 are not on a common scale.

    Fusing raw scores would let whichever arm happens to emit larger numbers
    dominate — here fulltext's 500 against vector's 0.99, which says nothing
    about relevance.
    """
    fused = reciprocal_rank_fusion(
        {
            "vector": [("a", 0.99)],
            "fulltext": [("b", 500.0)],
        }
    )
    assert {c.uid for c in fused} == {"a", "b"}
    assert fused[0].score == pytest.approx(fused[1].score), (
        "rank 1 in either arm must score identically"
    )


def test_rrf_uses_the_configured_k():
    fused = reciprocal_rank_fusion({"vector": [("a", 1.0)]})
    assert fused[0].score == pytest.approx(1.0 / (RRF_K + 1))


def test_rrf_is_deterministic_on_ties():
    """§8.2 runs eval at temperature 0 and calls one flipped question a warning.

    That only means anything if retrieval does not flip between identical runs.
    """
    arms = {"vector": [("z", 1.0), ("a", 1.0)], "fulltext": [("m", 1.0)]}
    first = [c.uid for c in reciprocal_rank_fusion(arms)]
    for _ in range(5):
        assert [c.uid for c in reciprocal_rank_fusion(arms)] == first


def test_rrf_respects_the_limit():
    arms = {"vector": [(f"u{i}", 1.0) for i in range(100)]}
    assert len(reciprocal_rank_fusion(arms, limit=7)) == 7


def test_rrf_of_nothing_is_nothing():
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({"vector": []}) == []


# --------------------------------------------------------------------------
# The retriever
# --------------------------------------------------------------------------


def test_search_queries_both_arms(monkeypatch):
    backend = StubBackend(
        vector=[("a", 0.9)], fulltext=[("b", 3.0)]
    )
    retriever = HybridRetriever(backend=backend)
    results = retriever.search("repo", "get user by id", query_vec=[0.1, 0.2])

    assert {c.uid for c in results} == {"a", "b"}
    assert backend.vector_calls and backend.fulltext_calls


def test_fulltext_arm_receives_the_tokenized_query():
    """§5.3's query side, checked at the boundary where it would be dropped."""
    backend = StubBackend(fulltext=[("a", 1.0)])
    HybridRetriever(backend=backend).search("repo", "getUserById")

    _repo, sent, _k = backend.fulltext_calls[0]
    assert "getuserbyid" in sent
    assert "user" in sent, "the query reached Lucene un-split"


def test_missing_vector_arm_degrades_to_fulltext():
    """§9.3: embedding API down -> "Fulltext + graph expansion", not an error.

    The user is told "Semantic search degraded"; the counter is what puts
    something behind that banner.
    """
    backend = StubBackend(fulltext=[("a", 1.0), ("b", 0.5)])
    results = HybridRetriever(backend=backend).search("repo", "auth", query_vec=None)

    assert [c.uid for c in results] == ["a", "b"]
    assert backend.vector_calls == []
    assert metrics.get("retrieve.vector_arm_unavailable") == 1


def test_empty_query_with_no_vector_returns_nothing():
    """No arms is no results — never an unfiltered dump of the index."""
    backend = StubBackend(fulltext=[("a", 1.0)])
    assert HybridRetriever(backend=backend).search("repo", "   ") == []


def test_arms_are_fetched_deeper_than_the_final_top_k():
    """Fusion can only promote what some arm returned.

    Fetching exactly `top_k` per arm would cap the union at `top_k` and make the
    fusion decorative.
    """
    retriever = HybridRetriever(backend=StubBackend())
    assert retriever.arm_k > retriever.top_k
    assert retriever.top_k == RRF_TOP_K


def test_top_k_matches_the_spec_constant():
    """† §5.2 step 1: "Hybrid RRF ... -> top-30†"."""
    assert RRF_TOP_K == 30


def test_candidate_records_which_arm_found_it():
    """The packing report and the degradation banner both need this.

    A candidate that only fulltext found, during a vector outage, is a
    different claim from one both arms agreed on.
    """
    c = Candidate(uid="a", score=0.1, fulltext_rank=1)
    assert c.arms == ("fulltext",)
    assert Candidate(uid="b", score=0.1).arms == ()
