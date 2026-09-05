"""Step 3 — the in-place UID rewrite (§11.2 2a/2b) and the `EXPLAIN` gate.

The remap is the whole reason path-derived UIDs are survivable. §4.4 states the
alternative plainly: handled naively, a move deletes and recreates every symbol
in a file, severing every inbound `CALLS` edge from unchanged files, and the
system then reports zero callers for code with dozens — "worse than an error
because it looks like an answer."
"""

from __future__ import annotations

import pytest

from adapters.base import Edge, symbol_uid
from index.moves import Remap

from .conftest import REPO, epoch, make_file, make_symbol


# --------------------------------------------------------------------------
# 2a — the rewrite
# --------------------------------------------------------------------------


@pytest.fixture
def moved_file(writer):
    """`a.py` holds `helper`, called from five symbols in the untouched `b.py`."""
    helper = make_symbol(REPO, "a.py", "a.helper")
    callers = [make_symbol(REPO, "b.py", f"b.caller{i}") for i in range(5)]
    edges = [
        Edge(source_uid=c.uid, kind="CALLS", origin_path="b.py", target_uid=helper.uid)
        for c in callers
    ]
    batch = {
        "a.py": make_file("a.py", [helper], content_hash="ha"),
        "b.py": make_file("b.py", callers, edges, content_hash="hb"),
    }
    writer.apply(REPO, batch, batch.keys(), {}, epoch())
    return helper, callers


def test_remap_preserves_inbound_edges(writer, graph_db, moved_file):
    """The point of the whole mechanism.

    UID is the identity every edge references, so rewriting it in place moves
    the symbol without touching a single relationship.
    """
    helper, callers = moved_file
    new_uid = symbol_uid(REPO, "moved/a.py", "a.helper", 0, 0)

    moved = writer.remap_uids(
        REPO,
        [Remap(old_uid=helper.uid, new_uid=new_uid, old_path="a.py",
               new_path="moved/a.py")],
        epoch(),
    )
    assert moved == 1

    rows = graph_db.run(
        "MATCH (c:Symbol)-[:CALLS]->(t:Symbol {uid: $uid}) RETURN count(c) AS n",
        uid=new_uid,
    )
    assert rows[0]["n"] == 5, "every caller followed the rewritten uid"
    assert graph_db.run(
        "MATCH (s:Symbol {uid: $uid}) RETURN count(s) AS n", uid=helper.uid
    )[0]["n"] == 0, "the old uid is gone, not duplicated"

    moved_row = graph_db.run(
        "MATCH (s:Symbol {uid: $uid}) RETURN s.rel_path AS rel, s.origin_path AS origin",
        uid=new_uid,
    )[0]
    assert moved_row["rel"] == "moved/a.py"
    assert moved_row["origin"] == "moved/a.py"


def test_remap_collision_is_skipped_not_merged(writer, graph_db, moved_file):
    """§4.4: on collision the remap is skipped and the symbol takes the normal
    upsert path as new.

    Merging instead would silently fuse two distinct symbols — the same failure
    T1's ordinal exists to prevent, arriving from the move path.
    """
    helper, _callers = moved_file
    occupant = make_symbol(REPO, "moved/a.py", "a.helper")
    batch = {"moved/a.py": make_file("moved/a.py", [occupant], content_hash="hm")}
    writer.apply(REPO, batch, batch.keys(), {}, epoch())

    moved = writer.remap_uids(
        REPO,
        [Remap(old_uid=helper.uid, new_uid=occupant.uid, old_path="a.py",
               new_path="moved/a.py")],
        epoch(),
    )
    assert moved == 0, "the target uid was taken"

    assert graph_db.run(
        "MATCH (s:Symbol {uid: $uid}) RETURN count(s) AS n", uid=helper.uid
    )[0]["n"] == 1, "the retired node survives, reachable via batch_paths"
    assert graph_db.run(
        "MATCH (c:Symbol)-[:CALLS]->(:Symbol {uid: $uid}) RETURN count(c) AS n",
        uid=helper.uid,
    )[0]["n"] == 5


def test_remap_repoints_edge_origin_paths(writer, graph_db):
    """§11.2 step 2b.

    Step 4 scopes stale-edge deletion by `r.origin_path`. If a moved file's
    outbound edges kept the old path, they would fall outside every future
    batch's scope and never be collected — stale forever, and invisible.
    """
    src = make_symbol(REPO, "a.py", "a.src")
    dst = make_symbol(REPO, "z.py", "z.dst")
    edge = Edge(source_uid=src.uid, kind="CALLS", origin_path="a.py",
                target_uid=dst.uid)
    batch = {
        "a.py": make_file("a.py", [src], [edge], content_hash="ha"),
        "z.py": make_file("z.py", [dst], content_hash="hz"),
    }
    writer.apply(REPO, batch, batch.keys(), {}, epoch())

    new_uid = symbol_uid(REPO, "moved/a.py", "a.src", 0, 0)
    writer.remap_uids(
        REPO,
        [Remap(old_uid=src.uid, new_uid=new_uid, old_path="a.py",
               new_path="moved/a.py")],
        epoch(),
    )

    origins = graph_db.run(
        "MATCH (:Symbol {uid: $uid})-[r:CALLS]->() RETURN r.origin_path AS origin",
        uid=new_uid,
    )
    assert [row["origin"] for row in origins] == ["moved/a.py"]


def test_remap_moves_the_file_node(writer, reader, graph_db):
    """The `:File` row moves with the symbols.

    §11.2 does not mention `:File` at all (FINDINGS.md F-004), but §4.1 seeds
    `_indexed` from `known_paths`. A stale File row would make the indexer
    believe the old path is still present and skip Signal B for it.
    """
    sym = make_symbol(REPO, "a.py", "a.f")
    batch = {"a.py": make_file("a.py", [sym], content_hash="ha")}
    writer.apply(REPO, batch, batch.keys(), {}, epoch())
    assert reader.known_paths(REPO) == {"a.py"}

    new_uid = symbol_uid(REPO, "moved/a.py", "a.f", 0, 0)
    writer.remap_uids(
        REPO,
        [Remap(old_uid=sym.uid, new_uid=new_uid, old_path="a.py",
               new_path="moved/a.py")],
        epoch(),
    )
    assert reader.known_paths(REPO) == {"moved/a.py"}


def test_remap_is_scoped_to_the_repo(writer, graph_db):
    """§9.1 again: a remap in one tenant must not touch another's symbols."""
    mine = make_symbol(REPO, "a.py", "a.f")
    theirs = make_symbol("other_repo", "a.py", "a.f")
    writer.apply(REPO, {"a.py": make_file("a.py", [mine])}, ["a.py"], {}, epoch())
    writer.apply("other_repo", {"a.py": make_file("a.py", [theirs])}, ["a.py"], {},
                 epoch())

    # A remap naming the *other* repo's uid, issued against ours.
    moved = writer.remap_uids(
        REPO,
        [Remap(old_uid=theirs.uid, new_uid="deadbeef", old_path="a.py",
               new_path="moved.py")],
        epoch(),
    )
    assert moved == 0
    assert graph_db.run(
        "MATCH (s:Symbol {uid: $uid}) RETURN count(s) AS n", uid=theirs.uid
    )[0]["n"] == 1


def test_empty_remap_list_is_a_noop(writer):
    assert writer.remap_uids(REPO, [], epoch()) == 0


# --------------------------------------------------------------------------
# The EXPLAIN gate — B8
# --------------------------------------------------------------------------


@pytest.fixture
def populated(graph_db):
    """A graph with realistic statistics before any `EXPLAIN` is read.

    **`EXPLAIN` on an empty database tells you nothing.** With no statistics the
    planner's cost estimates are arbitrary, so a plan taken there is not
    evidence about the plan production will get. Measured: the same statement
    reads `DirectedAllRelationshipsScan` on an empty graph and
    `NodeByLabelScan` on a populated one — two different wrong answers, neither
    of them the one that matters.

    Edges outnumber nodes here, as they do in a real call graph.
    """
    graph_db.run("MATCH (n) DETACH DELETE n")
    graph_db.run(
        """
        UNWIND range(0, 1999) AS i
        CREATE (n:Symbol {uid: 'plan_u' + toString(i), repo_id: $repo_id,
                          origin_path: 'f' + toString(i % 200) + '.py',
                          rel_path: 'f' + toString(i % 200) + '.py',
                          name: 'sym' + toString(i % 50), epoch: 1})
        """,
        repo_id=REPO,
    )
    graph_db.run(
        """
        MATCH (a:Symbol {repo_id: $repo_id})
        WITH a, toInteger(substring(a.uid, 6)) AS ai
        UNWIND range(1, 6) AS k
        MATCH (b:Symbol {uid: 'plan_u' + toString((ai * 7 + k * 13) % 2000)})
        CREATE (a)-[:CALLS {origin_path: a.origin_path, epoch: 1}]->(b)
        """,
        repo_id=REPO,
    )
    yield graph_db
    graph_db.run("MATCH (n) DETACH DELETE n")


def _plan_operators(graph_db, query: str, **params) -> list[dict]:
    """Flatten an `EXPLAIN` plan into a list of {operator, index, identifiers}.

    The index is read from `Details`, not from an `Index` argument: Neo4j 5.26
    reports a seek as `RANGE INDEX s:Symbol(repo_id, origin_path, epoch) WHERE
    ...` and never names the index. So the assertion below is on the label and
    property list, which is what actually identifies the index anyway — a
    rename in §11.3 should not fail this test, a wrong property list should.
    """
    with graph_db.session() as session:
        plan = session.run(f"EXPLAIN {query}", **params).consume().plan

    out: list[dict] = []
    stack = [plan]
    while stack:
        node = stack.pop()
        args = node.get("args", {})
        out.append(
            {
                "operator": node.get("operatorType", ""),
                "index": args.get("Index") or args.get("Details", ""),
            }
        )
        stack.extend(node.get("children", []))
    return out


STEP_4 = """
MATCH (s:Symbol)
WHERE s.repo_id = $repo_id AND s.origin_path IN $batch_paths
  AND s.epoch <= $epoch
MATCH (s)-[r]->()
WHERE r.epoch < $epoch
DELETE r
"""

STEP_5 = """
MATCH (s:Symbol)
WHERE s.repo_id = $repo_id AND s.origin_path IN $batch_paths
  AND s.epoch < $epoch
DETACH DELETE s
"""

STEP_6A = """
MATCH (s:Symbol)
WHERE s.repo_id = $repo_id AND s.origin_path IN $batch_paths
  AND s.epoch <= $epoch
SET s.degree = COUNT { (s)--() }
"""

STEP_6B = """
MATCH (s:Symbol)
WHERE s.repo_id = $repo_id AND s.origin_path IN $batch_paths
  AND s.epoch <= $epoch
MATCH (s)-[]-(n:Symbol)
WITH DISTINCT n
SET n.degree = COUNT { (n)--() }
"""


@pytest.mark.parametrize(
    "label, query",
    [("step 4", STEP_4), ("step 5", STEP_5), ("step 6a", STEP_6A), ("step 6b", STEP_6B)],
)
def test_reconciliation_statements_use_the_symbol_origin_index(populated, label, query):
    """Plan §3 gate, guarding B8: "shows `symbol_origin` index usage, not
    `NodeByLabelScan`".

    §11.3 exists so these statements do not touch every symbol in the tenant.
    The failure is not an error — it is the system getting slower in proportion
    to how much someone has indexed, which is the hardest kind to notice.

    Two things this asserts, and one it deliberately does not:

    * the `symbol_origin` index is used — matched on its label and property
      list, because Neo4j 5.26 never names the index in a plan;
    * neither `NodeByLabelScan` nor an all-relationships scan appears.

    It does **not** require a `NodeIndexSeek` specifically. `NodeIndexScan` over
    the same index also satisfies the gate's wording, and which one the planner
    picks depends on index statistics that Neo4j samples asynchronously — the
    same statement was measured producing both. Asserting the seek would make
    this test flip on sampling timing rather than on a code change. Which
    operator each statement got is printed instead, since the difference is
    real: a scan reads the whole index, a seek reads the matching range.
    """
    operators = _plan_operators(
        populated, query, repo_id=REPO, batch_paths=["f1.py", "f2.py"], epoch=99
    )
    names = [op["operator"] for op in operators]

    assert not any("NodeByLabelScan" in n for n in names), (
        f"{label} falls back to a label scan: {names}"
    )
    assert not any("AllRelationshipsScan" in n for n in names), (
        f"{label} scans every relationship in the database: {names}"
    )
    assert any("symbol_origin" in str(op["index"])
               or "Symbol(repo_id, origin_path, epoch)" in str(op["index"])
               for op in operators), (
        f"{label} does not use symbol_origin: "
        f"{[(op['operator'], op['index']) for op in operators]}"
    )

    access = [n for n in names if "NodeIndex" in n]
    print(f"  {label}: {access[0] if access else 'no index access'}")


def test_symbols_named_uses_the_symbol_name_index(populated):
    """§5.1 Tier 2's index. Step 7 depends on it; step 3 is where it is built."""
    operators = _plan_operators(
        populated,
        "MATCH (s:Symbol {repo_id: $repo_id, name: $name}) RETURN s.uid",
        repo_id=REPO,
        name="sym1",
    )
    assert not any("NodeByLabelScan" in op["operator"] for op in operators)
    assert any("Symbol(repo_id, name)" in str(op["index"]) for op in operators), (
        f"{[(op['operator'], op['index']) for op in operators]}"
    )
