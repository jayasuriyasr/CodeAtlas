"""§4.5 at scale — the edges a chunked reconcile used to lose (F-015, F-016).

A reconcile writes its files in `BULK_THRESHOLD`-sized chunks, and each chunk
resolves cross-file edges against the graph as it stands then. Before the
replay, a call from an early chunk into a file a later chunk added had no target
and was dropped — or, worse, trimmed onto the package module. Every test here
uses `bulk_threshold=1`, one file per chunk, so the ordering problem is
guaranteed rather than probable.

These run against the real writer: whether an edge exists is a claim about the
graph, and only the graph can answer it.
"""

from __future__ import annotations

import asyncio

import pytest

import metrics
from graph.reader import GraphReader
from graph.writer import GraphWriter

from .conftest import REPO


@pytest.fixture(autouse=True)
def clean(graph_db):
    graph_db.run("MATCH (n) DETACH DELETE n")
    metrics.reset()
    yield
    graph_db.run("MATCH (n) DETACH DELETE n")


@pytest.fixture
def writer(graph_db):
    return GraphWriter(graph_db.driver, graph_db.name)


@pytest.fixture
def reader(graph_db):
    return GraphReader(graph_db.driver, graph_db.name)


CALLER = '''\
from pkg.z_callee import helper


def run(x):
    return helper(x)
'''

CALLEE = '''\
def helper(x):
    return x * 2
'''

CALLEE_TWO_ARGS = '''\
def helper(x, y):
    return x * y
'''


def calls(graph_db) -> list[tuple[str, str]]:
    rows = graph_db.run(
        """
        MATCH (a:Symbol {repo_id: $r})-[:CALLS]->(b:Symbol)
        RETURN a.qualified_name AS a, b.qualified_name AS b
        ORDER BY a, b
        """,
        r=REPO,
    )
    return [(r["a"], r["b"]) for r in rows]


def test_a_call_into_a_later_chunk_is_written(make_indexer, writer, reader, graph_db, git_repo):
    """The audit's reproduction, against the real graph."""
    git_repo.write("pkg/__init__.py", "")
    git_repo.write("pkg/a_caller.py", CALLER)
    git_repo.write("pkg/z_callee.py", CALLEE)
    git_repo.commit()

    indexer = make_indexer(writer, reader, {REPO: git_repo.root}, bulk_threshold=1)
    asyncio.run(indexer.full_reconcile(REPO))

    assert calls(graph_db) == [("pkg.a_caller.run", "pkg.z_callee.helper")]
    assert metrics.get("reconcile.chunks_replayed") == 2


def test_chunked_and_single_chunk_reconciles_build_the_same_edges(
    make_indexer, writer, reader, graph_db, git_repo
):
    """Chunk size is a cost knob. It must never change what the graph says."""
    git_repo.write("pkg/__init__.py", "")
    for i in range(4):
        git_repo.write(
            f"pkg/m{i}.py",
            f"from pkg.m{(i + 1) % 4} import f{(i + 1) % 4}\n\n\n"
            f"def f{i}(x):\n    return f{(i + 1) % 4}(x)\n",
        )
    git_repo.commit()

    asyncio.run(make_indexer(writer, reader, {REPO: git_repo.root}).full_reconcile(REPO))
    single = calls(graph_db)
    graph_db.run("MATCH (n) DETACH DELETE n")

    chunked_indexer = make_indexer(writer, reader, {REPO: git_repo.root}, bulk_threshold=1)
    asyncio.run(chunked_indexer.full_reconcile(REPO))

    assert calls(graph_db) == single
    assert len(single) == 4, "the cycle m0 -> m1 -> m2 -> m3 -> m0 has four calls"


def test_a_callee_whose_signature_changed_in_a_later_chunk(
    make_indexer, writer, reader, graph_db, git_repo
):
    """The caller's chunk resolves to the *old* helper; the callee's chunk then
    deletes it (a new arity is a new UID). Replay re-points the edge."""
    git_repo.write("pkg/__init__.py", "")
    git_repo.write("pkg/a_caller.py", CALLER)
    git_repo.write("pkg/z_callee.py", CALLEE)
    git_repo.commit()
    indexer = make_indexer(writer, reader, {REPO: git_repo.root}, bulk_threshold=1)
    asyncio.run(indexer.full_reconcile(REPO))

    git_repo.write("pkg/z_callee.py", CALLEE_TWO_ARGS)
    git_repo.write("pkg/a_caller.py", CALLER.replace("helper(x)\n", "helper(x, 3)\n"))
    git_repo.commit()
    asyncio.run(indexer.full_reconcile(REPO))

    rows = graph_db.run(
        """
        MATCH (:Symbol {qualified_name: 'pkg.a_caller.run'})-[:CALLS]->(t)
        RETURN t.arity AS arity
        """
    )
    assert [r["arity"] for r in rows] == [2]


def test_sync_project_links_an_unchanged_caller_to_a_new_callee(
    make_indexer, writer, reader, graph_db, git_repo
):
    """F-016. A cross-file edge is resolved only when its *source* is written.

    The caller is indexed before its callee exists, so its call is dropped. The
    callee then arrives; the caller is unchanged, so an ordinary reconcile never
    revisits it. Sync Project — the recovery button — re-resolves everything.
    """
    git_repo.write("pkg/__init__.py", "")
    git_repo.write("pkg/a_caller.py", CALLER)
    git_repo.commit()
    indexer = make_indexer(writer, reader, {REPO: git_repo.root})
    asyncio.run(indexer.full_reconcile(REPO))
    assert calls(graph_db) == []

    git_repo.write("pkg/z_callee.py", CALLEE)
    git_repo.commit()
    asyncio.run(indexer.full_reconcile(REPO))
    assert calls(graph_db) == [], "the documented limitation, still present"

    asyncio.run(indexer.sync_project(REPO))
    assert calls(graph_db) == [("pkg.a_caller.run", "pkg.z_callee.helper")]


def test_sync_project_is_idempotent(make_indexer, writer, reader, graph_db, git_repo):
    git_repo.write("pkg/__init__.py", "")
    git_repo.write("pkg/a_caller.py", CALLER)
    git_repo.write("pkg/z_callee.py", CALLEE)
    git_repo.commit()
    indexer = make_indexer(writer, reader, {REPO: git_repo.root}, bulk_threshold=1)
    asyncio.run(indexer.sync_project(REPO))
    first = calls(graph_db)
    nodes = graph_db.run("MATCH (s:Symbol {repo_id: $r}) RETURN count(s) AS n", r=REPO)[0]["n"]
    asyncio.run(indexer.sync_project(REPO))
    assert calls(graph_db) == first
    assert graph_db.run(
        "MATCH (s:Symbol {repo_id: $r}) RETURN count(s) AS n", r=REPO
    )[0]["n"] == nodes


def test_a_dot_directory_keeps_its_path(make_indexer, writer, reader, graph_db, git_repo):
    """`.github/scripts/release.py` was normalised to `github/...` on every
    symbol while its `:File` kept the real path, so deleting the file left its
    symbols behind for good."""
    git_repo.write(".github/scripts/release.py", "def release():\n    return 1\n")
    git_repo.commit()
    indexer = make_indexer(writer, reader, {REPO: git_repo.root})
    asyncio.run(indexer.full_reconcile(REPO))

    paths = {r["p"] for r in graph_db.run(
        "MATCH (s:Symbol {repo_id: $r}) RETURN DISTINCT s.rel_path AS p", r=REPO
    )}
    assert paths == {".github/scripts/release.py"}

    (git_repo.root / ".github/scripts/release.py").unlink()
    asyncio.run(indexer.full_reconcile(REPO))
    assert graph_db.run(
        "MATCH (s:Symbol {repo_id: $r}) RETURN count(s) AS n", r=REPO
    )[0]["n"] == 0
