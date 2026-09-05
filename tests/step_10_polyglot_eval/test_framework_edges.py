"""Step 10 — §3.5's framework layer, with the types v10.1 made writable.

v10.0's §11.2 3b allowlist held four types while §3.5 named eight, so
`INVOKES_HOOK`, `RENDERS`, `HAS_FIELD` and `USES_SERIALIZER` could not be
written — and §11.1 would not have traversed them if they had been. An edge
written and never read is worse than one not written: the rows exist, and the
question they answer returns nothing.

These tests would all pass under the old map-onto-`CALLS` workaround if they
only checked that *an* edge existed. They check the type.
"""

from __future__ import annotations

import time

import pytest

from adapters.typescript import TypeScriptAdapter, hooks_invoked
from config import EDGE_TYPE_ALLOWLIST, EDGE_WEIGHTS
from graph.writer import GraphWriter
from retrieve.expand import EXPANSION_CYPHER, NeighborhoodExpander

REPO = "repo_step10_fw"


@pytest.fixture(scope="module")
def adapter() -> TypeScriptAdapter:
    return TypeScriptAdapter()


@pytest.fixture(scope="module")
def nextjs(fixtures_dir):
    return fixtures_dir / "repos" / "nextjs_min"


def parse(adapter, nextjs, rel: str):
    return adapter.parse(REPO, rel, (nextjs / rel).read_bytes())


def target_names(parsed, kind: str) -> set[str]:
    by_uid = {s.uid: s for s in parsed.symbols}
    return {
        by_uid[e.target_uid].name
        for e in parsed.edges
        if e.kind == kind and e.target_uid in by_uid
    }


def hint_names(parsed, kind: str) -> set[str]:
    return {
        (e.target_hint or "").rsplit(".", 1)[-1]
        for e in parsed.edges
        if e.kind == kind and e.target_hint
    }


# --------------------------------------------------------------------------
# The allowlist itself
# --------------------------------------------------------------------------


def test_allowlist_covers_every_type_3_5_names():
    """§3.5 names eight relationships. All eight must be writable."""
    assert set(EDGE_TYPE_ALLOWLIST) == {
        "CALLS", "IMPORTS", "DEFINES", "DISPATCHES_TO",
        "INVOKES_HOOK", "RENDERS", "HAS_FIELD", "USES_SERIALIZER",
    }


def test_every_allowlisted_type_has_a_weight():
    """§5.2's table and §11.2's allowlist must agree.

    A type with no weight scores at `EDGE_WEIGHT_DEFAULT` while §5.2 claims to
    weight it — a silently mis-ranked expansion, not an error.
    """
    assert set(EDGE_WEIGHTS) == set(EDGE_TYPE_ALLOWLIST)


def test_expansion_traverses_every_allowlisted_type():
    """§11.1's `MATCH` must name each one.

    A type the expansion does not list is a type that can be written and never
    read — the v10.0 defect, from the other direction.
    """
    for kind in EDGE_TYPE_ALLOWLIST:
        assert kind in EXPANSION_CYPHER, f"§11.1 does not traverse {kind}"


# --------------------------------------------------------------------------
# INVOKES_HOOK
# --------------------------------------------------------------------------


def test_hook_calls_are_typed_invokes_hook(adapter, nextjs):
    """§3.5: `(Component)-[:INVOKES_HOOK]->(Hook)`.

    Typed, not merely present. Under the v10.0 workaround these were `CALLS`
    edges distinguishable only by re-applying React's naming convention at read
    time; the distinction §5.2 weights is now in the graph itself.
    """
    parsed = parse(adapter, nextjs, "components/UserCard.tsx")
    hooks = [e for e in parsed.edges if e.kind == "INVOKES_HOOK"]

    assert hooks, "no INVOKES_HOOK edges — the hook calls fell back to CALLS"
    names = hint_names(parsed, "INVOKES_HOOK")
    assert {"useState", "useEffect", "useCallback"} <= names, names

    assert not (names & hint_names(parsed, "CALLS")), "a hook was emitted as both"


def test_ordinary_calls_are_still_calls(adapter, nextjs):
    """`formatName` is not a hook. Over-typing would be as wrong as under-typing."""
    assert "formatName" in hint_names(
        parse(adapter, nextjs, "components/UserCard.tsx"), "CALLS"
    )


def test_hooks_invoked_reads_the_edges(adapter, nextjs):
    hooks = hooks_invoked(parse(adapter, nextjs, "components/UserCard.tsx"))
    assert set(hooks["components.UserCard.UserCard"]) >= {
        "useState", "useEffect", "useCallback"
    }


# --------------------------------------------------------------------------
# RENDERS vs DISPATCHES_TO
# --------------------------------------------------------------------------


def test_a_page_renders_its_component(adapter, nextjs):
    """§3.5: `(Route)-[:RENDERS]->(Component)` from App Router conventions."""
    parsed = parse(adapter, nextjs, "app/orders/page.tsx")

    assert "OrdersPage" in target_names(parsed, "RENDERS")
    assert not [e for e in parsed.edges if e.kind == "DISPATCHES_TO"]


def test_a_route_dispatches_to_its_handlers(adapter, nextjs):
    """A `route.ts` handles a request; it does not render.

    §3.5 gives both relationships, and this is the shape its Django row already
    calls `DISPATCHES_TO`: a route dispatching to a view.
    """
    parsed = parse(adapter, nextjs, "app/api/users/route.ts")

    assert {"GET", "POST"} <= target_names(parsed, "DISPATCHES_TO")
    assert not [e for e in parsed.edges if e.kind == "RENDERS"]


def test_a_plain_component_is_neither(adapter, nextjs):
    parsed = parse(adapter, nextjs, "components/UserCard.tsx")
    assert not [e for e in parsed.edges if e.kind in ("RENDERS", "DISPATCHES_TO")]


# --------------------------------------------------------------------------
# End to end: the types survive the write and the traversal
# --------------------------------------------------------------------------


@pytest.fixture
def written(graph_db, adapter, nextjs):
    graph_db.run("MATCH (n:Symbol {repo_id: $r}) DETACH DELETE n", r=REPO)
    batch = {
        rel: parse(adapter, nextjs, rel)
        for rel in (
            "components/UserCard.tsx",
            "app/orders/page.tsx",
            "app/api/users/route.ts",
        )
    }
    GraphWriter(graph_db.driver, graph_db.name).apply(
        REPO, batch, batch.keys(), {}, time.time_ns()
    )
    yield batch
    graph_db.run("MATCH (n:Symbol {repo_id: $r}) DETACH DELETE n", r=REPO)
    graph_db.run("MATCH (f:File {repo_id: $r}) DETACH DELETE f", r=REPO)


def test_framework_types_are_written_to_the_graph(written, graph_db):
    """§11.2 3b writes one statement per type, from the allowlist.

    Under v10.0 this raised — the writer refuses a kind outside the allowlist —
    so the adapter could not emit these at all.
    """
    kinds = {
        row["kind"]
        for row in graph_db.run(
            """
            MATCH (a:Symbol {repo_id: $r})-[e]->(b:Symbol)
            RETURN DISTINCT type(e) AS kind ORDER BY kind
            """,
            r=REPO,
        )
    }
    assert "RENDERS" in kinds, kinds
    assert "DISPATCHES_TO" in kinds, kinds


def test_expansion_returns_a_rendered_component(written, graph_db):
    """The whole point of R2: the edge is written *and* read.

    A `RENDERS` edge §11.1 would not traverse is a row nothing can reach —
    strictly worse than not writing it, because the graph looks correct.
    """
    parsed = written["app/orders/page.tsx"]
    page = next(s for s in parsed.symbols if s.name == "OrdersPage")
    module = parsed.symbols[0]

    neighbors = NeighborhoodExpander(runner=graph_db).expand(REPO, [module.uid])
    assert page.uid in {n.uid for n in neighbors}, (
        "the RENDERS edge was written but not traversed"
    )
