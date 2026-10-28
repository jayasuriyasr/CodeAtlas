"""F-014, closed: §11.1's subquery in its non-deprecated form (v10.2 R7).

v10.1's `CALL { WITH seed … }` executes on 5.26 but raises
`FeatureDeprecationWarning` — "Use CALL (seed) { ... }" — and is removed in a
future major. The expansion is the statement §5.2 calls "the justification for
running a graph database", so it is the worst one to have break on an upgrade.
The rewrite must change nothing but the syntax: same rows, same scores.
"""

from __future__ import annotations

from config import EDGE_WEIGHT_DEFAULT, EDGE_WEIGHTS, GLOBAL_EXPANSION_LIMIT, HUB_CUTOFF, PER_SEED_LIMIT
from retrieve.expand import EXPANSION_CYPHER

REPO = "repo_scope_clause"

LEGACY = EXPANSION_CYPHER.replace("CALL (seed) {\n", "CALL {\n    WITH seed\n")


def _params(seeds):
    return dict(
        seed_uids=seeds, repo_id=REPO, hub_cutoff=HUB_CUTOFF,
        per_seed_limit=PER_SEED_LIMIT, global_limit=GLOBAL_EXPANSION_LIMIT,
        edge_weights=EDGE_WEIGHTS, edge_weight_default=EDGE_WEIGHT_DEFAULT,
    )


def _populate(graph_db):
    graph_db.run("MATCH (n) DETACH DELETE n")
    graph_db.run(
        """
        UNWIND range(0, 299) AS i
        CREATE (:Symbol {uid: 'sc' + toString(i), repo_id: $repo, name: 'n' + toString(i),
                         qualified_name: 'm.n' + toString(i), rel_path: 'f.py',
                         degree: i % 7})
        """,
        repo=REPO,
    )
    graph_db.run(
        """
        MATCH (a:Symbol {repo_id: $repo})
        WITH a, toInteger(substring(a.uid, 2)) AS ai
        UNWIND range(1, 4) AS k
        MATCH (b:Symbol {uid: 'sc' + toString((ai * 11 + k * 17) % 300)})
        CREATE (a)-[:CALLS {origin_path: 'f.py', epoch: 1}]->(b)
        CREATE (b)-[:IMPORTS {origin_path: 'f.py', epoch: 1}]->(a)
        """,
        repo=REPO,
    )


def test_the_statement_uses_the_scope_clause():
    assert "CALL (seed) {" in EXPANSION_CYPHER
    assert "WITH seed\n" not in EXPANSION_CYPHER.split("CALL (seed) {", 1)[1].split("}", 1)[0]


def test_legacy_and_scoped_forms_return_identical_rows(graph_db):
    _populate(graph_db)
    seeds = [f"sc{i}" for i in (0, 5, 17, 42, 99)]
    with graph_db.session() as session:
        legacy = [dict(r) for r in session.run(LEGACY, _params(seeds))]
        scoped = [dict(r) for r in session.run(EXPANSION_CYPHER, _params(seeds))]

    assert scoped, "the fixture produced no neighbours; the comparison proves nothing"
    assert scoped == legacy
    graph_db.run("MATCH (n) DETACH DELETE n")


def test_the_scoped_form_raises_no_deprecation(graph_db):
    _populate(graph_db)
    with graph_db.session() as session:
        summary = session.run(EXPANSION_CYPHER, _params(["sc1"])).consume()
    codes = [n.get("code") for n in (summary.notifications or [])]
    assert not any("Deprecation" in (code or "") for code in codes), codes
    graph_db.run("MATCH (n) DETACH DELETE n")
