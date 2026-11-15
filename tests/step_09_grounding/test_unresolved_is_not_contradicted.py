"""A relation the graph cannot see is unverified, not contradicted.

Measured (README; eval runs of 2026-11-14): of the 3B model's relation claims,
"`revoke_all` calls `revoke`" was flagged *contradicted* — both symbols exist
and no CALLS edge joins them — although it is true: `revoke_all` calls
`token.revoke()` on a loop variable, a call SEI records but cannot resolve. A
contradiction triggers §6.3's retry, so a correct answer was retried.

The graph can only say "no" when it resolved every call by that name. Each file
records the calls it could not resolve (`File.waiting`, step 3a); if the
subject's file holds one ending in the object's name, the claim is unverified.
"""

from __future__ import annotations

import time

import pytest

from adapters.python import PythonAdapter
from ground.validate import RelationClaim, verify_relations
from graph.writer import GraphWriter

REPO = "repo_t3_unresolved"

MODELS = (b"class ApiToken:\n    def revoke(self):\n        return None\n\n\n"
          b"def issue_token():\n    return ApiToken()\n")
SERVICES = (b"from pkg.models import ApiToken\n\n\n"
            b"def revoke_all(tokens):\n    for token in tokens:\n        token.revoke()\n\n\n"
            b"def count_tokens(tokens):\n    return len(tokens)\n")


class Runner:
    def __init__(self, graph_db):
        self.db = graph_db

    def run(self, query, **params):
        return [dict(r) for r in self.db.run(query, **params)]


@pytest.fixture
def indexed(graph_db):
    graph_db.run("MATCH (n {repo_id: $r}) DETACH DELETE n", r=REPO)
    files = {"pkg/models.py": MODELS, "pkg/services.py": SERVICES}
    batch = {rel: PythonAdapter().parse(REPO, rel, src) for rel, src in files.items()}
    GraphWriter(graph_db.driver, graph_db.name).apply(REPO, batch, batch.keys(), {}, time.time_ns())
    yield Runner(graph_db)
    graph_db.run("MATCH (n {repo_id: $r}) DETACH DELETE n", r=REPO)


def verdict(runner, subject, obj):
    (v,) = verify_relations(runner, REPO, [RelationClaim(subject, "CALLS", obj, 0)])
    return v


def test_a_call_the_graph_could_not_resolve_is_not_a_contradiction(indexed):
    v = verdict(indexed, "revoke_all", "revoke")
    assert v.subject_found and v.object_found and not v.edge_found
    assert not v.verified and not v.contradicted


def test_a_function_with_no_such_call_is_still_contradicted(indexed):
    v = verdict(indexed, "issue_token", "revoke")
    assert v.contradicted


def test_another_function_in_the_file_is_judged_by_the_file(indexed):
    """The record is per file, so a neighbour's unresolved `.revoke()` also makes
    this unverified — the safe direction: never a false contradiction."""
    v = verdict(indexed, "count_tokens", "revoke")
    assert not v.contradicted and not v.verified
