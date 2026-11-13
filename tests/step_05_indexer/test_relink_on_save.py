"""A save that changes a symbol's UID keeps the callers in other files.

A function's UID includes its parameter count (`symbol_uid`), so adding a
parameter makes a new node. Step 5 then deletes the old one with `DETACH
DELETE` — taking every incoming CALLS edge with it — and nothing re-resolved
the callers, which live in files nobody saved. "What calls `helper`" answered
zero until the next Sync Project, for code with five callers.
"""

from __future__ import annotations

import asyncio

import pytest

import metrics
from graph.reader import GraphReader
from graph.writer import GraphWriter

from .conftest import CALLEE, REPO, seed_move_fixture


@pytest.fixture(autouse=True)
def clean(graph_db):
    graph_db.run("MATCH (n) DETACH DELETE n")
    metrics.reset()
    yield
    graph_db.run("MATCH (n) DETACH DELETE n")


@pytest.fixture
def indexer(make_indexer, graph_db, git_repo):
    seed_move_fixture(git_repo)
    return make_indexer(
        GraphWriter(graph_db.driver, graph_db.name),
        GraphReader(graph_db.driver, graph_db.name),
        {REPO: git_repo.root},
        debounce=0.05,
    )


def callers_of(graph_db, qualified_name: str) -> set[str]:
    rows = graph_db.run(
        """
        MATCH (c:Symbol {repo_id: $r})-[:CALLS]->(:Symbol {repo_id: $r, qualified_name: $qn})
        RETURN DISTINCT c.qualified_name AS qn
        """,
        r=REPO, qn=qualified_name,
    )
    return {r["qn"] for r in rows}


def save(indexer, git_repo, rel_path: str, text: str) -> None:
    git_repo.write(rel_path, text)
    indexer.on_save(REPO, rel_path, text.encode())


def test_changing_a_callees_arity_keeps_its_callers(indexer, graph_db, git_repo):
    async def scenario():
        await indexer.startup(REPO)
        await indexer.sync_project(REPO)
        assert len(callers_of(graph_db, "pkg.callee.helper")) == 5

        save(indexer, git_repo, "pkg/callee.py",
             CALLEE.replace("def helper(value):", "def helper(value, scale=2):"))
        await indexer.settle()
        await indexer.aclose()

    asyncio.run(scenario())
    assert callers_of(graph_db, "pkg.callee.helper") == {
        f"pkg.callers.caller{i}" for i in range(5)
    }
    assert metrics.get("index.relinked_files") == 1


def test_a_save_that_keeps_every_uid_relinks_nothing(indexer, graph_db, git_repo):
    """A body edit keeps the UID; there is nothing to re-resolve and no
    reason to re-read another file."""
    async def scenario():
        await indexer.startup(REPO)
        await indexer.sync_project(REPO)
        save(indexer, git_repo, "pkg/callee.py", CALLEE.replace("value * 2", "value * 3"))
        await indexer.settle()
        await indexer.aclose()

    asyncio.run(scenario())
    assert len(callers_of(graph_db, "pkg.callee.helper")) == 5
    assert metrics.get("index.relinked_files") == 0


def test_a_deleted_callee_leaves_no_edge_and_does_not_loop(indexer, graph_db, git_repo):
    """Removing the function outright: the callers are re-read, find nothing
    to point at, and the re-link stops there."""
    async def scenario():
        await indexer.startup(REPO)
        await indexer.sync_project(REPO)
        save(indexer, git_repo, "pkg/callee.py", '"""Emptied."""\n')
        await indexer.settle()
        await indexer.aclose()

    asyncio.run(scenario())
    assert callers_of(graph_db, "pkg.callee.helper") == set()
    assert metrics.get("index.relinked_files") == 1


def test_a_failed_relink_is_retried_on_the_next_flush(make_indexer, graph_db, git_repo):
    """If the re-link write fails, the batch is already in the graph and would
    orphan nothing if written again — so the callers' files are what must be
    retried."""
    seed_move_fixture(git_repo)

    class FailsOnce(GraphWriter):
        armed = False

        def apply(self, repo_id, batch, batch_paths, vectors, epoch):
            if self.armed and set(batch) == {"pkg/callers.py"}:
                self.armed = False
                raise ConnectionError("Neo4j went away mid re-link")
            return super().apply(repo_id, batch, batch_paths, vectors, epoch)

    writer = FailsOnce(graph_db.driver, graph_db.name)
    indexer = make_indexer(writer, GraphReader(graph_db.driver, graph_db.name),
                           {REPO: git_repo.root}, debounce=0.05)

    async def scenario():
        await indexer.startup(REPO)
        await indexer.sync_project(REPO)
        writer.armed = True
        save(indexer, git_repo, "pkg/callee.py",
             CALLEE.replace("def helper(value):", "def helper(value, scale=2):"))
        for _ in range(100):                     # the retry is on a timer
            await asyncio.sleep(0.05)
            await indexer.settle()
            if len(callers_of(graph_db, "pkg.callee.helper")) == 5:
                break
        await indexer.aclose()

    asyncio.run(scenario())
    assert writer.armed is False, "the failure was injected"
    assert len(callers_of(graph_db, "pkg.callee.helper")) == 5
