"""Step 1 probes 1-5 — the Cypher constructs spec §0.2 lists as unverified.

Each probe runs the construct in the exact shape the spec uses it, records the
server's actual response verbatim, and only then asserts. §11's own header is
the reason these exist: `COUNT { … }`, `NOT EXISTS { MATCH … }` and vector DDL
have all shifted across 5.x minors, so reading the manual settles nothing.
"""

from __future__ import annotations

import pytest
from neo4j.exceptions import ClientError, CypherSyntaxError, Neo4jError

from config import EDGE_WEIGHT_DEFAULT, EDGE_WEIGHTS, EMBED_DIM


@pytest.fixture(scope="module", autouse=True)
def seed_graph(throwaway_db):
    """Two symbols and one edge — enough for degree and existence probes."""
    throwaway_db.run(
        """
        MERGE (a:Symbol {uid: 'probe_a', repo_id: 'probe', name: 'alpha'})
        MERGE (b:Symbol {uid: 'probe_b', repo_id: 'probe', name: 'beta'})
        MERGE (a)-[:CALLS {origin_path: 'probe.py'}]->(b)
        """
    )
    return throwaway_db


# --------------------------------------------------------------------------
# probe_dynamic_map_key
# --------------------------------------------------------------------------


def test_probe_dynamic_map_key(throwaway_db, probe_log):
    """§11.1 scores with `$edge_weights[rel]`, where `rel` is a bound variable.

    If a map parameter cannot be indexed by a runtime value, every edge in the
    expansion silently scores at the coalesce default and the weight table in
    §5.2 does nothing.
    """
    rec = probe_log.record(
        "probe_dynamic_map_key",
        "Does `$edge_weights[rel]` resolve a dynamic key on a map parameter?",
    )
    stmt = (
        "WITH $rel AS rel\n"
        "RETURN $edge_weights[rel] AS direct,\n"
        "       coalesce($edge_weights[rel], $default) AS with_default"
    )
    rec.statement(stmt)

    try:
        hit = throwaway_db.run(
            stmt, rel="CALLS", edge_weights=EDGE_WEIGHTS, default=EDGE_WEIGHT_DEFAULT
        )[0]
        miss = throwaway_db.run(
            stmt, rel="NO_SUCH_REL", edge_weights=EDGE_WEIGHTS,
            default=EDGE_WEIGHT_DEFAULT,
        )[0]
    except Neo4jError as exc:
        rec.observe("raised", f"{type(exc).__name__}: {exc}")
        rec.conclude("REJECTED — §11.1 scoring must be rewritten")
        raise

    rec.observe("rel='CALLS' -> direct", hit["direct"])
    rec.observe("rel='CALLS' -> with_default", hit["with_default"])
    rec.observe("rel='NO_SUCH_REL' -> direct", miss["direct"])
    rec.observe("rel='NO_SUCH_REL' -> with_default", miss["with_default"])
    rec.conclude(
        "SUPPORTED — dynamic key resolves; a missing key yields null and "
        "coalesce covers it"
    )

    assert hit["direct"] == EDGE_WEIGHTS["CALLS"]
    assert miss["direct"] is None
    assert miss["with_default"] == EDGE_WEIGHT_DEFAULT


# --------------------------------------------------------------------------
# probe_count_subquery
# --------------------------------------------------------------------------


def test_probe_count_subquery(throwaway_db, probe_log):
    """§11.2 steps 6a/6b write `degree` with `SET s.degree = COUNT { (s)--() }`.

    Both the RETURN form and the SET form are probed: an expression legal in a
    projection is not automatically legal on the right of a SET.
    """
    rec = probe_log.record(
        "probe_count_subquery",
        "Is `COUNT { (n)--() }` available on this minor?",
    )
    read_stmt = (
        "MATCH (n:Symbol {uid: 'probe_a'}) RETURN COUNT { (n)--() } AS degree"
    )
    write_stmt = (
        "MATCH (s:Symbol)\n"
        "WHERE s.repo_id = 'probe'\n"
        "SET s.degree = COUNT { (s)--() }\n"
        "RETURN s.uid AS uid, s.degree AS degree ORDER BY uid"
    )
    rec.statement(read_stmt)
    rec.statement(write_stmt)

    try:
        degree = throwaway_db.run(read_stmt)[0]["degree"]
        written = throwaway_db.run(write_stmt)
    except Neo4jError as exc:
        rec.observe("raised", f"{type(exc).__name__}: {exc}")
        rec.conclude("UNAVAILABLE — §11.2 steps 6a/6b need a rewrite")
        raise

    rec.observe("RETURN COUNT { (n)--() }", degree)
    for row in written:
        rec.observe(f"SET degree on {row['uid']}", row["degree"])
    rec.conclude("SUPPORTED — legal in a projection and on the right of SET")

    assert degree == 1
    assert [r["degree"] for r in written] == [1, 1]


# --------------------------------------------------------------------------
# probe_exists_subquery
# --------------------------------------------------------------------------


def test_probe_exists_subquery(throwaway_db, probe_log):
    """§11.2 step 2a guards the in-place UID rewrite with this predicate.

    If it is unavailable the remap has no collision guard, and a file moved
    onto an occupied UID corrupts identity rather than skipping.
    """
    rec = probe_log.record(
        "probe_exists_subquery",
        "Is `NOT EXISTS { MATCH (c:Symbol {uid: $u}) }` available?",
    )
    stmt = (
        "MATCH (s:Symbol {uid: 'probe_a'})\n"
        "WHERE NOT EXISTS { MATCH (c:Symbol {uid: $u}) }\n"
        "RETURN s.uid AS uid"
    )
    rec.statement(stmt)

    try:
        free = throwaway_db.run(stmt, u="uid_that_does_not_exist")
        taken = throwaway_db.run(stmt, u="probe_b")
    except Neo4jError as exc:
        rec.observe("raised", f"{type(exc).__name__}: {exc}")
        rec.conclude("UNAVAILABLE — §11.2 step 2a loses its collision guard")
        raise

    rec.observe("target uid free -> rows", [r["uid"] for r in free])
    rec.observe("target uid taken -> rows", [r["uid"] for r in taken])
    rec.conclude(
        "SUPPORTED — predicate admits the free case and rejects the collision"
    )

    assert [r["uid"] for r in free] == ["probe_a"]
    assert taken == []


# --------------------------------------------------------------------------
# probe_vector_ddl
# --------------------------------------------------------------------------

_VEC_OPTS = (
    "OPTIONS { indexConfig: { `vector.dimensions`: "
    + str(EMBED_DIM)
    + ", `vector.similarity_function`: 'cosine' } }"
)

_PARENS = (
    "CREATE VECTOR INDEX probe_vec_parens IF NOT EXISTS\n"
    "  FOR (s:Symbol) ON (s.code_vec)\n  " + _VEC_OPTS
)
_BARE = (
    "CREATE VECTOR INDEX probe_vec_bare IF NOT EXISTS\n"
    "  FOR (s:Symbol) ON s.code_vec\n  " + _VEC_OPTS
)


def test_probe_vector_ddl(throwaway_db, probe_log):
    """Spec §11.3 writes `ON (s.code_vec)`. Both forms are tried.

    Recording which one the server accepts is the whole point: §0.2 flags the
    parenthesisation as version-dependent, and a wrong guess here means no
    vector index at all — which §12.1 says degrades silently to fulltext.
    """
    rec = probe_log.record(
        "probe_vector_ddl",
        "Does the vector index take `ON (s.code_vec)` or `ON s.code_vec`?",
    )
    rec.statement(_PARENS)
    rec.statement(_BARE)

    outcomes: dict[str, str] = {}
    for label, stmt in (("ON (s.code_vec)", _PARENS), ("ON s.code_vec", _BARE)):
        try:
            throwaway_db.run(stmt)
            outcomes[label] = "accepted"
        except Neo4jError as exc:
            outcomes[label] = f"{type(exc).__name__}: {exc}"
        rec.observe(label, outcomes[label])

    # Whichever forms were accepted, show what the server actually created.
    shown = throwaway_db.run(
        "SHOW INDEXES YIELD name, type, labelsOrTypes, properties, options "
        "WHERE name STARTS WITH 'probe_vec' "
        "RETURN name, type, labelsOrTypes, properties, options"
    )
    for row in shown:
        cfg = (row["options"] or {}).get("indexConfig")
        rec.observe(
            "SHOW INDEXES " + row["name"],
            "type={} labels={} properties={} indexConfig={}".format(
                row["type"], row["labelsOrTypes"], row["properties"], cfg
            ),
        )

    accepted = [k for k, v in outcomes.items() if v == "accepted"]
    rec.conclude("accepted form(s): " + (", ".join(accepted) or "NONE — step 1 stops here"))

    # The gate: the spec's own form must work. If it does not, everything
    # downstream that assumes a vector index is built on sand.
    assert outcomes["ON (s.code_vec)"] == "accepted", (
        "spec §11.3 uses the parenthesised form; it was rejected: "
        + outcomes["ON (s.code_vec)"]
    )


def test_probe_vector_ddl_round_trip(throwaway_db, probe_log):
    """DDL parsing is not the question anyone actually cares about.

    A syntactically accepted index that never returns a row is the failure mode
    §12.1 calls "present in traversal and invisible to hybrid retrieval". So
    write a vector and query it back.
    """
    rec = probe_log.records["probe_vector_ddl"]
    vec = [0.0] * EMBED_DIM
    vec[0] = 1.0

    write_stmt = (
        "MATCH (s:Symbol {uid: 'probe_a'}) SET s.code_vec = $vec "
        "RETURN size(s.code_vec) AS n"
    )
    query_stmt = (
        "CALL db.index.vector.queryNodes('probe_vec_parens', 1, $vec) "
        "YIELD node, score RETURN node.uid AS uid, score"
    )
    rec.statement(write_stmt)
    rec.statement(query_stmt)

    try:
        n = throwaway_db.run(write_stmt, vec=vec)[0]["n"]
        rec.observe("SET s.code_vec -> size()", n)
    except Neo4jError as exc:
        rec.observe("SET s.code_vec raised", f"{type(exc).__name__}: {exc}")
        raise

    throwaway_db.run("CALL db.awaitIndexes(120)")

    try:
        rows = throwaway_db.run(query_stmt, vec=vec)
    except Neo4jError as exc:
        rec.observe("queryNodes raised", f"{type(exc).__name__}: {exc}")
        raise

    rec.observe(
        "queryNodes rows", [(r["uid"], round(r["score"], 6)) for r in rows]
    )
    assert rows and rows[0]["uid"] == "probe_a"


# --------------------------------------------------------------------------
# probe_merge_multitype
# --------------------------------------------------------------------------


def test_probe_merge_multitype(throwaway_db, probe_log):
    """§11.2 step 3b splits edge writes by type. Confirm that is *required*.

    If a multi-type MERGE were legal, the per-type split would be defensive
    clutter. This records the rejection verbatim so the split has a cited
    reason rather than folklore.
    """
    rec = probe_log.record(
        "probe_merge_multitype",
        "Confirm `MERGE (a)-[:A|B]->(b)` is rejected — the per-type split in "
        "§11.2 3b is required, not defensive.",
    )
    merge_stmt = (
        "MATCH (a:Symbol {uid: 'probe_a'}), (b:Symbol {uid: 'probe_b'})\n"
        "MERGE (a)-[r:CALLS|IMPORTS]->(b)\n"
        "RETURN type(r) AS t"
    )
    match_stmt = (
        "MATCH (a:Symbol {uid: 'probe_a'})-[r:CALLS|IMPORTS]->(b:Symbol)\n"
        "RETURN type(r) AS t"
    )
    rec.statement(merge_stmt)
    rec.statement(match_stmt)

    raised: Neo4jError | None = None
    try:
        throwaway_db.run(merge_stmt)
    except Neo4jError as exc:
        raised = exc
        rec.observe("MERGE multi-type", f"{type(exc).__name__}: {exc}")
    else:
        rec.observe("MERGE multi-type", "ACCEPTED — no error raised")

    # The contrast that makes the rule memorable: MATCH takes it happily.
    matched = throwaway_db.run(match_stmt)
    rec.observe("MATCH multi-type", [r["t"] for r in matched])

    rec.conclude(
        "REJECTED as expected — per-type MERGE is required"
        if raised is not None
        else "ACCEPTED — contradicts §11.2 3b; record in FINDINGS.md before proceeding"
    )

    assert raised is not None, (
        "MERGE accepted a multi-type pattern. Spec §11.2 3b and CLAUDE.md's "
        "trap list both assume rejection — append to FINDINGS.md."
    )
    assert isinstance(raised, (CypherSyntaxError, ClientError))
    assert [r["t"] for r in matched] == ["CALLS"]
