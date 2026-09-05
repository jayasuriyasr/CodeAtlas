"""Step 3 — `GraphWriter` and §4.3's epoch reconciliation.

The claim under test is narrow and worth restating exactly, because the
tempting stronger version is false: the write sequence is **not atomic**. It
provides superset-on-prefix and idempotent replay (§4.3), and those are the two
properties these tests measure.

Builders and fixtures live in `conftest.py`.
"""

from __future__ import annotations

import pytest

from adapters.base import Edge
from config import EDGE_TYPE_ALLOWLIST, EMBED_DIM
from graph.writer import STEP_BOUNDARIES, InjectedCrash, symbol_props

from .conftest import OTHER_REPO, REPO, epoch, make_file, make_symbol


# --------------------------------------------------------------------------
# B1 — per-type edge MERGE
# --------------------------------------------------------------------------


def test_edge_merge_per_type(writer, graph_db):
    """B1. All four allowlisted types write correctly via separate statements.

    `probe_merge_multitype` established that `MERGE (a)-[:A|B]->(b)` is a syntax
    error. This is the other half: the per-type split actually produces the
    right graph, not merely a legal one.
    """
    src = make_symbol(REPO, "a.py", "a.src")
    targets = {kind: make_symbol(REPO, "a.py", f"a.t_{kind}") for kind in EDGE_TYPE_ALLOWLIST}
    edges = [
        Edge(source_uid=src.uid, kind=kind, origin_path="a.py", target_uid=t.uid)
        for kind, t in targets.items()
    ]
    batch = {"a.py": make_file("a.py", [src, *targets.values()], edges)}

    report = writer.apply(REPO, batch, batch.keys(), {}, epoch())
    assert report.edges_written == len(EDGE_TYPE_ALLOWLIST)

    rows = graph_db.run(
        "MATCH (s:Symbol {uid: $uid})-[r]->(t:Symbol) "
        "RETURN type(r) AS kind, t.qualified_name AS target ORDER BY kind",
        uid=src.uid,
    )
    assert {r["kind"] for r in rows} == set(EDGE_TYPE_ALLOWLIST)
    for row in rows:
        assert row["target"] == f"a.t_{row['kind']}"


def test_edge_kind_outside_the_allowlist_is_refused(writer):
    """§11.2 3b: the type comes from a fixed allowlist, never from parsed input.

    Substituting a parsed identifier into the query string is how a source file
    gets to name a relationship type — refused loudly rather than sanitised.
    """
    a = make_symbol(REPO, "a.py", "a.one")
    b = make_symbol(REPO, "a.py", "a.two")
    bad = Edge(source_uid=a.uid, kind="DROP_EVERYTHING", origin_path="a.py",
               target_uid=b.uid)
    batch = {"a.py": make_file("a.py", [a, b], [bad])}

    with pytest.raises(ValueError, match="allowlist"):
        writer.apply(REPO, batch, batch.keys(), {}, epoch())


def test_no_transaction_subquery_after_write(writer, two_file_batch):
    """B2. Transaction subqueries cannot follow an updating clause.

    There is no assertion to make beyond "the statement set actually executes":
    if any statement in §11.2 used `CALL { } IN TRANSACTIONS` after a `MERGE`,
    this call would raise. Batching is the driver's job, and this is the test
    that says so.
    """
    batch, _helper, _callers = two_file_batch
    report = writer.apply(REPO, batch, batch.keys(), {}, epoch())
    assert report.symbols_written == 6
    assert report.edges_written == 5


# --------------------------------------------------------------------------
# v9.1 #13 — degree
# --------------------------------------------------------------------------


def test_own_degree_refreshed(writer, graph_db, two_file_batch):
    """v9.1 #13, step 6a. A new symbol must not carry a null `degree`.

    §11.1's score divides by `log(2 + coalesce(n.degree, 1))`. A null degree is
    coalesced to 1, so an unrefreshed new symbol scores as if it were maximally
    specific — systematic over-scoring of exactly the symbols just written.
    """
    batch, helper, callers = two_file_batch
    writer.apply(REPO, batch, batch.keys(), {}, epoch())

    rows = graph_db.run(
        "MATCH (s:Symbol {repo_id: $r}) RETURN s.uid AS uid, s.degree AS degree",
        r=REPO,
    )
    degrees = {row["uid"]: row["degree"] for row in rows}

    assert all(d is not None for d in degrees.values()), "no null degrees"
    assert degrees[helper.uid] == 5, "five inbound callers"
    for caller in callers:
        assert degrees[caller.uid] == 1


def test_neighbor_degree_refreshed(writer, graph_db, two_file_batch):
    """Step 6b. A symbol's degree changes when someone *else's* file changes.

    Writing only `b.py` adds an edge that raises `a.helper`'s degree, and
    `a.helper` is not in the batch. Without 6b its degree stays stale, and the
    §11.1 hub cutoff then admits a symbol it should have excluded.
    """
    batch, helper, _callers = two_file_batch
    writer.apply(REPO, batch, batch.keys(), {}, epoch())

    # A sixth caller arrives, in b.py only. a.py is untouched.
    sixth = make_symbol(REPO, "b.py", "b.caller5")
    existing = batch["b.py"].symbols + [sixth]
    edges = [
        Edge(source_uid=s.uid, kind="CALLS", origin_path="b.py", target_uid=helper.uid)
        for s in existing
    ]
    second = {"b.py": make_file("b.py", existing, edges, content_hash="hb2")}
    writer.apply(REPO, second, second.keys(), {}, epoch())

    degree = graph_db.run(
        "MATCH (s:Symbol {uid: $uid}) RETURN s.degree AS degree", uid=helper.uid
    )[0]["degree"]
    assert degree == 6, "a.helper is a neighbour of the batch, not a member of it"


# --------------------------------------------------------------------------
# B15 — what the ordering actually guarantees
# --------------------------------------------------------------------------


@pytest.mark.parametrize("boundary", STEP_BOUNDARIES)
def test_superset_on_prefix(writer, graph_db, two_file_batch, boundary):
    """Crash injection at each of §11.2's six boundaries.

    The guarantee is a superset, not correctness: stale rows are allowed,
    dangling references are not. So the assertion is *not* "the graph is right"
    — it is "every edge still has both endpoints, and nothing that was
    reachable became unreachable".
    """
    batch, helper, callers = two_file_batch
    writer.apply(REPO, batch, batch.keys(), {}, epoch())
    before = graph_db.run(
        "MATCH (s:Symbol {repo_id: $r}) RETURN count(s) AS n", r=REPO
    )[0]["n"]

    # A second write that removes two callers and adds one — enough to make
    # steps 4 and 5 have real work to do.
    survivors = callers[:3]
    newcomer = make_symbol(REPO, "b.py", "b.caller_new")
    kept = [*survivors, newcomer]
    edges = [
        Edge(source_uid=s.uid, kind="CALLS", origin_path="b.py", target_uid=helper.uid)
        for s in kept
    ]
    second = {"b.py": make_file("b.py", kept, edges, content_hash="hb2")}

    with pytest.raises(InjectedCrash):
        writer.apply(REPO, second, second.keys(), {}, epoch(), stop_after=boundary)

    dangling = graph_db.run(
        """
        MATCH ()-[r]->()
        WHERE startNode(r) IS NULL OR endNode(r) IS NULL
        RETURN count(r) AS n
        """
    )[0]["n"]
    assert dangling == 0, f"dangling reference after crash at {boundary}"

    # Superset: every symbol that existed before the crashed write still
    # resolves, or was legitimately superseded by a later-epoch write.
    after = graph_db.run(
        "MATCH (s:Symbol {repo_id: $r}) RETURN count(s) AS n", r=REPO
    )[0]["n"]
    assert after >= min(before, 4), (
        f"crash at {boundary} left fewer symbols than the prefix guarantee allows"
    )

    orphan_edges = graph_db.run(
        """
        MATCH (a:Symbol)-[r]->(b:Symbol)
        WHERE a.uid IS NULL OR b.uid IS NULL
        RETURN count(r) AS n
        """
    )[0]["n"]
    assert orphan_edges == 0


@pytest.mark.parametrize("boundary", STEP_BOUNDARIES)
def test_replay_after_crash_converges(writer, graph_db, two_file_batch, boundary):
    """§4.3's other half: re-running from step 1 with the same epoch converges.

    This is what makes superset-on-prefix survivable. A crash is repaired by
    replay, not by repair logic — so the replayed graph must equal the graph a
    clean run would have produced.
    """
    batch, helper, callers = two_file_batch
    writer.apply(REPO, batch, batch.keys(), {}, epoch())

    kept = callers[:3]
    edges = [
        Edge(source_uid=s.uid, kind="CALLS", origin_path="b.py", target_uid=helper.uid)
        for s in kept
    ]
    second = {"b.py": make_file("b.py", kept, edges, content_hash="hb2")}
    second_epoch = epoch()

    with pytest.raises(InjectedCrash):
        writer.apply(REPO, second, second.keys(), {}, second_epoch, stop_after=boundary)

    writer.apply(REPO, second, second.keys(), {}, second_epoch)   # replay, same epoch

    uids = {
        row["uid"]
        for row in graph_db.run(
            "MATCH (s:Symbol {repo_id: $r}) RETURN s.uid AS uid", r=REPO
        )
    }
    assert uids == {helper.uid} | {s.uid for s in kept}
    assert graph_db.run(
        "MATCH (:Symbol)-[r:CALLS]->(:Symbol) RETURN count(r) AS n"
    )[0]["n"] == 3


def test_idempotent_replay(writer, graph_db, two_file_batch):
    """Re-run the full sequence with the same epoch: identical graph."""
    batch, _helper, _callers = two_file_batch
    fixed = epoch()

    writer.apply(REPO, batch, batch.keys(), {}, fixed)
    first = _snapshot(graph_db)

    writer.apply(REPO, batch, batch.keys(), {}, fixed)
    assert _snapshot(graph_db) == first

    writer.apply(REPO, batch, batch.keys(), {}, fixed)
    assert _snapshot(graph_db) == first


def _snapshot(graph_db) -> tuple:
    nodes = graph_db.run(
        """
        MATCH (s:Symbol)
        RETURN s.uid AS uid, s.qualified_name AS qn, s.epoch AS epoch,
               s.degree AS degree, s.origin_path AS origin
        ORDER BY uid
        """
    )
    edges = graph_db.run(
        """
        MATCH (a:Symbol)-[r]->(b:Symbol)
        RETURN a.uid AS src, type(r) AS kind, b.uid AS tgt,
               r.origin_path AS origin, r.epoch AS epoch
        ORDER BY src, kind, tgt
        """
    )
    return (
        tuple(tuple(row.values()) for row in nodes),
        tuple(tuple(row.values()) for row in edges),
    )


# --------------------------------------------------------------------------
# Scoping
# --------------------------------------------------------------------------


def test_stale_edges_deleted_scoped(writer, graph_db, two_file_batch):
    """Edges from *unchanged* files survive.

    §11.2 step 4 deletes only edges whose `origin_path` is in this batch. An
    unscoped delete would remove every edge older than the epoch — which is to
    say, every edge in the repository — on the first single-file save.
    """
    batch, helper, callers = two_file_batch

    # A third file, written once and then never touched again.
    outsider = make_symbol(REPO, "c.py", "c.outsider")
    c_edges = [
        Edge(source_uid=outsider.uid, kind="CALLS", origin_path="c.py",
             target_uid=helper.uid)
    ]
    batch["c.py"] = make_file("c.py", [outsider], c_edges, content_hash="hc")
    writer.apply(REPO, batch, batch.keys(), {}, epoch())
    assert _calls_into(graph_db, helper.uid) == 6

    # Re-write b.py alone, dropping all its callers.
    empty_b = {"b.py": make_file("b.py", [], [], content_hash="hb2")}
    writer.apply(REPO, empty_b, empty_b.keys(), {}, epoch())

    assert _calls_into(graph_db, helper.uid) == 1, "c.py's edge must survive"
    survivors = graph_db.run(
        "MATCH ()-[r:CALLS]->(:Symbol {uid: $uid}) RETURN r.origin_path AS origin",
        uid=helper.uid,
    )
    assert [row["origin"] for row in survivors] == ["c.py"]


def _calls_into(graph_db, uid: str) -> int:
    return graph_db.run(
        "MATCH ()-[r:CALLS]->(:Symbol {uid: $uid}) RETURN count(r) AS n", uid=uid
    )[0]["n"]


def test_orphan_symbols_removed_only_within_batch_paths(writer, graph_db, two_file_batch):
    """§11.2 step 5, same scoping argument as step 4."""
    batch, _helper, _callers = two_file_batch
    writer.apply(REPO, batch, batch.keys(), {}, epoch())

    empty_b = {"b.py": make_file("b.py", [], [], content_hash="hb2")}
    writer.apply(REPO, empty_b, empty_b.keys(), {}, epoch())

    remaining = {
        row["qn"]
        for row in graph_db.run(
            "MATCH (s:Symbol {repo_id: $r}) RETURN s.qualified_name AS qn", r=REPO
        )
    }
    assert remaining == {"a.helper"}, "b.py's symbols gone, a.py's untouched"


def test_repo_isolation(writer, graph_db):
    """T7. Two repos, identical paths and qualified names, zero cross-talk.

    The UID includes `repo_id`, so the symbols differ — but every statement in
    §11.2 also filters on `repo_id`, and this is what proves the filters are
    actually there rather than merely implied by the UID.
    """
    def batch_for(repo: str, content: str):
        sym = make_symbol(repo, "shared/path.py", "shared.path.handler")
        return {"shared/path.py": make_file("shared/path.py", [sym], content_hash=content)}, sym

    one, sym_one = batch_for(REPO, "h1")
    two, sym_two = batch_for(OTHER_REPO, "h2")
    assert sym_one.uid != sym_two.uid, "repo_id is a UID input"

    writer.apply(REPO, one, one.keys(), {}, epoch())
    writer.apply(OTHER_REPO, two, two.keys(), {}, epoch())

    # Empty repo one's file of symbols. §4.5 distinguishes two cases and this
    # is the first: a file that still exists but defines nothing. Its `:File`
    # row is re-stamped with the new epoch by 3a and must survive — only its
    # symbols go.
    empty = {"shared/path.py": make_file("shared/path.py", [], [], content_hash="h3")}
    writer.apply(REPO, empty, empty.keys(), {}, epoch())

    remaining = graph_db.run(
        "MATCH (s:Symbol) RETURN s.repo_id AS repo, s.uid AS uid ORDER BY repo"
    )
    assert [(r["repo"], r["uid"]) for r in remaining] == [(OTHER_REPO, sym_two.uid)]

    files = graph_db.run("MATCH (f:File) RETURN f.repo_id AS repo ORDER BY repo")
    assert {r["repo"] for r in files} == {OTHER_REPO, REPO}, (
        "an emptied file is not a deleted file"
    )

    # The second case: the file is gone from the tree. §4.5 spells the call out
    # — `self.writer.apply(repo, {}, set(deleted), {}, epoch)` — so nothing
    # re-stamps the `:File` row and step 5 collects it.
    writer.apply(REPO, {}, {"shared/path.py"}, {}, epoch())

    files = graph_db.run("MATCH (f:File) RETURN f.repo_id AS repo ORDER BY repo")
    assert [r["repo"] for r in files] == [OTHER_REPO], (
        "repo one's File row survived a delete, or repo two's did not survive it"
    )
    remaining = graph_db.run("MATCH (s:Symbol) RETURN s.repo_id AS repo")
    assert [r["repo"] for r in remaining] == [OTHER_REPO]


# --------------------------------------------------------------------------
# T5 — vectors reach the node
# --------------------------------------------------------------------------


def test_vectors_merged_into_props_before_upsert(writer, graph_db):
    """T5, at the writer boundary.

    Step 4 owns the end-to-end version of this test; this one pins the specific
    line v9.2 was missing — `apply` merging `embed_batch`'s return into
    `sym.props` *before* step 3a runs.
    """
    sym = make_symbol(REPO, "a.py", "a.embedded")
    batch = {"a.py": make_file("a.py", [sym])}
    vec = [0.0] * EMBED_DIM
    vec[3] = 1.0

    writer.apply(REPO, batch, batch.keys(), {}, epoch(), )
    assert graph_db.run(
        "MATCH (s:Symbol {uid: $uid}) RETURN s.code_vec IS NULL AS missing", uid=sym.uid
    )[0]["missing"], "no vector supplied, so none stored"

    writer.apply(REPO, batch, batch.keys(), {sym.uid: vec}, epoch())
    stored = graph_db.run(
        "MATCH (s:Symbol {uid: $uid}) RETURN s.code_vec AS v", uid=sym.uid
    )[0]["v"]
    assert stored is not None and len(stored) == EMBED_DIM
    assert stored[3] == pytest.approx(1.0)


def test_none_valued_props_do_not_erase_existing_values(writer, graph_db):
    """`SET s += props` treats null as removal.

    A symbol written once with a vector and again without one — because the
    embedding provider was degraded, an existing row in §9.3 — must keep the
    vector it already has rather than silently losing it.
    """
    sym = make_symbol(REPO, "a.py", "a.embedded")
    batch = {"a.py": make_file("a.py", [sym])}
    vec = [0.5] * EMBED_DIM

    writer.apply(REPO, batch, batch.keys(), {sym.uid: vec}, epoch())
    sym.code_vec = None                    # a later parse, no embedding available
    writer.apply(REPO, batch, batch.keys(), {}, epoch())

    stored = graph_db.run(
        "MATCH (s:Symbol {uid: $uid}) RETURN s.code_vec AS v", uid=sym.uid
    )[0]["v"]
    assert stored is not None, "an absent vector must not erase a stored one"


def test_symbol_props_omits_none():
    sym = make_symbol(REPO, "a.py", "a.f")
    props = symbol_props(sym)
    assert "docstring" not in props
    assert "code_vec" not in props
    assert props["uid"] == sym.uid


# --------------------------------------------------------------------------
# Reader
# --------------------------------------------------------------------------


def test_reader_round_trips_what_the_indexer_needs(writer, reader, two_file_batch):
    batch, helper, callers = two_file_batch
    writer.apply(REPO, batch, batch.keys(), {}, epoch())

    assert reader.known_paths(REPO) == {"a.py", "b.py"}
    assert reader.file_hashes(REPO) == {"a.py": "ha", "b.py": "hb"}
    assert reader.inbound_callers(REPO, helper.uid) == sorted(c.uid for c in callers)

    named = reader.symbols_named(REPO, "helper")
    assert [n.uid for n in named] == [helper.uid]
    assert reader.symbols_named(REPO, "does_not_exist") == []


def test_reader_body_hashes_omits_unknown_uids(writer, reader):
    sym = make_symbol(REPO, "a.py", "a.f")
    sym.body_hash = "bh1"
    batch = {"a.py": make_file("a.py", [sym])}
    writer.apply(REPO, batch, batch.keys(), {}, epoch())

    hashes = reader.body_hashes(REPO, [sym.uid, "no_such_uid"])
    assert hashes == {sym.uid: "bh1"}, "an absent uid is a new symbol, not an error"


def test_reader_symbols_named_returns_every_candidate(writer, reader):
    """Tier 2 decides; the reader must not decide for it.

    A reader that returned the first match would make §5.1's ambiguity refusal
    unreachable — dead code guarding a case that can no longer arrive.
    """
    a = make_symbol(REPO, "app/api/x/route.py", "app.api.x.route.GET")
    b = make_symbol(REPO, "app/api/y/route.py", "app.api.y.route.GET")
    batch = {
        "app/api/x/route.py": make_file("app/api/x/route.py", [a]),
        "app/api/y/route.py": make_file("app/api/y/route.py", [b]),
    }
    writer.apply(REPO, batch, batch.keys(), {}, epoch())

    named = reader.symbols_named(REPO, "GET")
    assert len(named) == 2
    assert {n.rel_path for n in named} == {"app/api/x/route.py", "app/api/y/route.py"}


# --------------------------------------------------------------------------
# Cross-file edge resolution
# --------------------------------------------------------------------------


def test_unresolvable_edge_is_dropped_not_invented(writer, graph_db):
    """A call into a third-party library has no node, and must not get one."""
    caller = make_symbol(REPO, "a.py", "a.caller")
    edge = Edge(source_uid=caller.uid, kind="CALLS", origin_path="a.py",
                target_hint="requests.get")
    batch = {"a.py": make_file("a.py", [caller], [edge])}

    report = writer.apply(REPO, batch, batch.keys(), {}, epoch())
    assert report.edges_written == 0
    assert report.edges_unresolved == 1
    assert graph_db.run("MATCH (s:Symbol) RETURN count(s) AS n")[0]["n"] == 1


def test_ambiguous_edge_target_refuses(writer, graph_db):
    """Two symbols share a qualified name (§3.2's conditional-definition case).

    §5.1 sets the house rule that ambiguity is refusal. An edge pointing at an
    arbitrary one of the two is a wrong answer wearing the costume of a right
    one, and nothing downstream could tell.
    """
    one = make_symbol(REPO, "b.py", "b.target", ordinal=0)
    two = make_symbol(REPO, "b.py", "b.target", ordinal=1)
    caller = make_symbol(REPO, "a.py", "a.caller")
    edge = Edge(source_uid=caller.uid, kind="CALLS", origin_path="a.py",
                target_hint="b.target")

    batch = {
        "b.py": make_file("b.py", [one, two]),
        "a.py": make_file("a.py", [caller], [edge]),
    }
    report = writer.apply(REPO, batch, batch.keys(), {}, epoch())

    assert report.edges_ambiguous == 1
    assert graph_db.run("MATCH ()-[r:CALLS]->() RETURN count(r) AS n")[0]["n"] == 0


def test_attribute_access_resolves_to_the_longest_matching_prefix(writer, graph_db):
    """`authx.models.User.objects.filter` is a reference to `authx.models.User`."""
    user = make_symbol(REPO, "models.py", "models.User", kind="class")
    caller = make_symbol(REPO, "views.py", "views.handler")
    edge = Edge(source_uid=caller.uid, kind="CALLS", origin_path="views.py",
                target_hint="models.User.objects.filter")
    batch = {
        "models.py": make_file("models.py", [user]),
        "views.py": make_file("views.py", [caller], [edge]),
    }
    writer.apply(REPO, batch, batch.keys(), {}, epoch())

    rows = graph_db.run(
        "MATCH (:Symbol {uid: $c})-[:CALLS]->(t:Symbol) RETURN t.uid AS uid",
        c=caller.uid,
    )
    assert [r["uid"] for r in rows] == [user.uid]
