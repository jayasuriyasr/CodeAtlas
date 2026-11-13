"""The vector arm with more than one repository in the database (F-028).

§9.1's MVP is one database with `repo_id` on every node, and the vector index
is the one index that cannot lead with it: `queryNodes` returns the nearest
rows across every tenant, and the filter comes after. Measured with real
embeddings, a 73-symbol repository beside an 1,836-symbol one got 2, 11, 9, 0
and 0 of the 60 rows it asked for. Nothing failed; the vector arm was simply
not there, for the smaller repository only, depending on the question.

These tests build that shape directly: a large tenant whose vectors sit right
on the query, and a small one whose vectors are all further away.
"""

from __future__ import annotations

import math

import pytest

import metrics
import retrieve.hybrid as hybrid
from config import EMBED_DIM, VECTOR_OVERFETCH_FACTOR
from retrieve.hybrid import Neo4jSearchBackend

BIG, SMALL = "repo_tenancy_big", "repo_tenancy_small"
K = 5


def unit(angle: float) -> list[float]:
    """A unit vector in the plane of the first two dimensions."""
    return [math.cos(angle), math.sin(angle)] + [0.0] * (EMBED_DIM - 2)


QUERY = unit(0.0)


@pytest.fixture(autouse=True)
def reset():
    metrics.reset()
    yield
    metrics.reset()


def write(graph_db, repo: str, vectors: list[list[float]]) -> None:
    graph_db.run(
        """
        UNWIND range(0, size($vectors) - 1) AS i
        CREATE (:Symbol {repo_id: $repo, uid: $repo + '-' + toString(i), code_vec: $vectors[i]})
        """,
        repo=repo, vectors=vectors,
    )


@pytest.fixture
def two_tenants(graph_db):
    """300 symbols almost on the query, and 8 that are 0.5 to 1.2 rad away."""
    graph_db.run("MATCH (n) DETACH DELETE n")
    write(graph_db, BIG, [unit(0.0005 * i) for i in range(300)])
    write(graph_db, SMALL, [unit(0.5 + 0.1 * i) for i in range(8)])
    graph_db.run("CALL db.awaitIndexes(180)")
    yield Neo4jSearchBackend(graph_db.driver, graph_db.name)
    graph_db.run("MATCH (n) DETACH DELETE n")


def test_a_small_repository_still_gets_its_k_beside_a_large_one(two_tenants):
    """The first fetch — 20 rows — is entirely the big tenant's."""
    rows = two_tenants.vector_search(SMALL, QUERY, K)

    assert [uid for uid, _score in rows] == [f"{SMALL}-{i}" for i in range(K)], (
        "its own nearest five, nearest first"
    )
    scores = [score for _uid, score in rows]
    assert scores == sorted(scores, reverse=True)
    assert metrics.get("retrieve.vector_refetch") >= 1
    assert metrics.get("retrieve.vector_starved") == 0


def test_no_row_of_another_tenant_is_ever_returned(two_tenants):
    for repo in (BIG, SMALL):
        rows = two_tenants.vector_search(repo, QUERY, K)
        assert rows and all(uid.startswith(repo + "-") for uid, _score in rows)


def test_the_nearer_tenant_is_answered_in_one_query(two_tenants):
    """Widening is for the starved case; the ordinary one costs what it did."""
    rows = two_tenants.vector_search(BIG, QUERY, K)
    assert len(rows) == K
    assert metrics.get("retrieve.vector_refetch") == 0


def test_asking_for_more_than_a_repository_holds_ends_when_the_index_does(two_tenants):
    rows = two_tenants.vector_search(SMALL, QUERY, 50)
    assert len(rows) == 8, "all of it, and the loop stopped"
    assert metrics.get("retrieve.vector_starved") == 0


def test_a_repository_with_no_vectors_gets_none_and_terminates(two_tenants):
    assert two_tenants.vector_search("repo_tenancy_absent", QUERY, K) == []


def test_the_fetch_is_bounded_and_says_so_when_it_gives_up(two_tenants, monkeypatch):
    """Past `VECTOR_MAX_FETCH` the arm returns what it found — and counts it,
    so a starved tenant is a number on a dashboard rather than a mystery."""
    monkeypatch.setattr(hybrid, "VECTOR_MAX_FETCH", K * VECTOR_OVERFETCH_FACTOR)
    rows = two_tenants.vector_search(SMALL, QUERY, K)
    assert rows == []
    assert metrics.get("retrieve.vector_starved") == 1
    assert metrics.get("retrieve.vector_refetch") == 0
