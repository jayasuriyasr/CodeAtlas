"""Cross-file edge resolution — the defects the final audit found (F-015).

`GraphWriter._resolve_edge_targets` turns an adapter's `target_hint` into a uid.
Three rules are pinned here:

* trimming a hint never stops on a *module* — `pkg.z_callee.helper` with
  `z_callee` not yet indexed used to resolve to the package `pkg` and write
  `run -CALLS-> pkg`, a confident edge to the wrong node;
* symbols the batch is about to orphan (step 5) are not candidates, so a
  signature change saved together with its caller is not "ambiguous";
* the lookup is served by `symbol_name`, not a label scan per name.
"""

from __future__ import annotations

from adapters.base import Edge
from graph.writer import GraphWriter, _hint_prefixes

from .conftest import REPO, epoch, make_file, make_symbol
from .test_remap_and_plans import _plan_operators


def _calls(graph_db, source_uid: str) -> list[str]:
    rows = graph_db.run(
        "MATCH (:Symbol {uid: $u})-[:CALLS]->(t:Symbol) RETURN t.qualified_name AS qn",
        u=source_uid,
    )
    return sorted(r["qn"] for r in rows)


def test_hint_prefixes_skip_empty_segments():
    assert _hint_prefixes("a.b.c") == ["a.b.c", "a.b", "a"]
    assert _hint_prefixes("..lib.x") == []
    assert _hint_prefixes("a..b") == ["a"]
    assert _hint_prefixes("") == []


def test_trimming_never_lands_on_a_module(writer, graph_db):
    """The audit's reproduction: the package must not stand in for `helper`."""
    package = make_symbol(REPO, "pkg/__init__.py", "pkg", kind="module")
    caller = make_symbol(REPO, "pkg/a_caller.py", "pkg.a_caller.run")
    edge = Edge(source_uid=caller.uid, kind="CALLS", origin_path="pkg/a_caller.py",
                target_hint="pkg.z_callee.helper")
    batch = {
        "pkg/__init__.py": make_file("pkg/__init__.py", [package]),
        "pkg/a_caller.py": make_file("pkg/a_caller.py", [caller], [edge]),
    }

    report = writer.apply(REPO, batch, batch.keys(), {}, epoch())

    assert _calls(graph_db, caller.uid) == []
    assert report.edges_unresolved == 1
    assert report.edges_written == 0


def test_a_module_named_exactly_is_still_a_valid_target(writer, graph_db):
    """`import authx.models` names the module itself — that edge is correct."""
    module = make_symbol(REPO, "authx/models.py", "authx.models", kind="module")
    importer = make_symbol(REPO, "authx/views.py", "authx.views", kind="module")
    edge = Edge(source_uid=importer.uid, kind="IMPORTS", origin_path="authx/views.py",
                target_hint="authx.models")
    batch = {
        "authx/models.py": make_file("authx/models.py", [module]),
        "authx/views.py": make_file("authx/views.py", [importer], [edge]),
    }
    writer.apply(REPO, batch, batch.keys(), {}, epoch())

    rows = graph_db.run(
        "MATCH (:Symbol {uid: $u})-[:IMPORTS]->(t) RETURN t.uid AS uid", u=importer.uid
    )
    assert [r["uid"] for r in rows] == [module.uid]


def test_trimming_onto_a_class_still_resolves(writer, graph_db):
    """Only modules are refused: attribute access on a class is a reference."""
    user = make_symbol(REPO, "models.py", "models.User", kind="class")
    caller = make_symbol(REPO, "views.py", "views.handler")
    edge = Edge(source_uid=caller.uid, kind="CALLS", origin_path="views.py",
                target_hint="models.User.objects.create")
    batch = {
        "models.py": make_file("models.py", [user]),
        "views.py": make_file("views.py", [caller], [edge]),
    }
    writer.apply(REPO, batch, batch.keys(), {}, epoch())
    assert _calls(graph_db, caller.uid) == ["models.User"]


def test_a_hint_with_empty_segments_is_unresolved_not_an_error(writer, graph_db):
    caller = make_symbol(REPO, "a.py", "a.run")
    edge = Edge(source_uid=caller.uid, kind="CALLS", origin_path="a.py",
                target_hint="..lib.users.getUser")
    batch = {"a.py": make_file("a.py", [caller], [edge])}
    report = writer.apply(REPO, batch, batch.keys(), {}, epoch())
    assert report.edges_unresolved == 1


def test_signature_change_saved_with_its_caller_is_not_ambiguous(writer, graph_db):
    """`helper(x)` becomes `helper(x, y)`; the caller is updated in the same save.

    The old `helper` (arity 1) is still in the graph while 3b runs — step 5
    deletes it two statements later. Counting it as a candidate made the edge
    "ambiguous" and refused it for a reason that no longer existed.
    """
    old_helper = make_symbol(REPO, "b.py", "b.helper", arity=1)
    old_caller = make_symbol(REPO, "a.py", "a.run")
    old_edge = Edge(source_uid=old_caller.uid, kind="CALLS", origin_path="a.py",
                    target_hint="b.helper")
    first = {
        "b.py": make_file("b.py", [old_helper]),
        "a.py": make_file("a.py", [old_caller], [old_edge]),
    }
    writer.apply(REPO, first, first.keys(), {}, epoch())
    assert _calls(graph_db, old_caller.uid) == ["b.helper"]

    new_helper = make_symbol(REPO, "b.py", "b.helper", arity=2)
    new_caller = make_symbol(REPO, "a.py", "a.run")
    new_edge = Edge(source_uid=new_caller.uid, kind="CALLS", origin_path="a.py",
                    target_hint="b.helper")
    second = {
        "b.py": make_file("b.py", [new_helper], content_hash="h2"),
        "a.py": make_file("a.py", [new_caller], [new_edge], content_hash="h2"),
    }
    report = writer.apply(REPO, second, second.keys(), {}, epoch())

    assert report.edges_ambiguous == 0
    rows = graph_db.run(
        "MATCH (:Symbol {uid: $u})-[:CALLS]->(t) RETURN t.uid AS uid", u=new_caller.uid
    )
    assert [r["uid"] for r in rows] == [new_helper.uid]
    assert graph_db.run(
        "MATCH (s:Symbol {uid: $u}) RETURN count(s) AS n", u=old_helper.uid
    )[0]["n"] == 0, "the retired arity-1 helper should be gone after step 5"


def test_two_current_candidates_are_still_refused(writer, graph_db):
    """The doomed-symbol rule must not weaken §5.1's refusal for live symbols."""
    one = make_symbol(REPO, "c.py", "c.target", ordinal=0)
    two = make_symbol(REPO, "c.py", "c.target", ordinal=1)
    caller = make_symbol(REPO, "d.py", "d.run")
    edge = Edge(source_uid=caller.uid, kind="CALLS", origin_path="d.py",
                target_hint="c.target")
    writer.apply(REPO, {"c.py": make_file("c.py", [one, two])}, ["c.py"], {}, epoch())
    report = writer.apply(
        REPO, {"d.py": make_file("d.py", [caller], [edge])}, ["d.py"], {}, epoch()
    )
    assert report.edges_ambiguous == 1
    assert _calls(graph_db, caller.uid) == []


def test_resolution_is_scoped_to_the_repo(writer, graph_db):
    """Another tenant's identically-named symbol is not a candidate (§9.1)."""
    theirs = make_symbol("other_tenant", "b.py", "b.helper")
    writer.apply("other_tenant", {"b.py": make_file("b.py", [theirs])}, ["b.py"], {}, epoch())

    caller = make_symbol(REPO, "a.py", "a.run")
    edge = Edge(source_uid=caller.uid, kind="CALLS", origin_path="a.py",
                target_hint="b.helper")
    report = writer.apply(REPO, {"a.py": make_file("a.py", [caller], [edge])}, ["a.py"], {}, epoch())
    assert report.edges_written == 0
    assert _calls(graph_db, caller.uid) == []


def test_the_resolution_lookup_uses_the_symbol_name_index(graph_db):
    """No label scan per hint: the lookup restates `qualified_name = n` as an
    index-served `(repo_id, name)` match plus an equality filter."""
    graph_db.run("MATCH (n) DETACH DELETE n")
    graph_db.run(
        """
        UNWIND range(0, 1999) AS i
        CREATE (:Symbol {uid: 'res_u' + toString(i), repo_id: $repo_id,
                         name: 'n' + toString(i % 300),
                         qualified_name: 'm' + toString(i % 40) + '.n' + toString(i % 300),
                         origin_path: 'f' + toString(i % 200) + '.py', epoch: 1,
                         kind: 'function'})
        """,
        repo_id=REPO,
    )
    query = """
        UNWIND $names AS n
        MATCH (s:Symbol {repo_id: $repo_id, name: n.name})
        WHERE s.qualified_name = n.qn
          AND NOT (s.origin_path IN $batch_paths AND s.epoch < $epoch)
        RETURN n.qn AS name, collect({uid: s.uid, kind: s.kind}) AS candidates
    """
    # The statement under test must be the one the writer runs.
    import inspect

    assert "MATCH (s:Symbol {repo_id: $repo_id, name: n.name})" in inspect.getsource(
        GraphWriter._resolve_edge_targets
    )
    operators = _plan_operators(
        graph_db, query, repo_id=REPO,
        names=[{"qn": "m1.n1", "name": "n1"}], batch_paths=["f1.py"], epoch=5,
    )
    assert not any("NodeByLabelScan" in op["operator"] for op in operators), operators
    assert any("Symbol(repo_id, name)" in str(op["index"]) for op in operators), operators
