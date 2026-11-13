"""The watcher against a real graph — the claims only the graph can settle.

`test_watcher.py` pins what the watcher *decides*. Whether a deleted file's
symbols are actually gone, and whether a `git mv` keeps its callers, are facts
about Neo4j.
"""

from __future__ import annotations

import asyncio
import os

import pytest

import metrics
from graph.reader import GraphReader
from graph.writer import GraphWriter
from index.watcher import TreeWatcher

from .conftest import REPO, seed_move_fixture

EXTENSIONS = (".py", ".pyi")


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


def names(graph_db) -> set[str]:
    return {r["qn"] for r in graph_db.run(
        "MATCH (s:Symbol {repo_id: $r}) RETURN s.qualified_name AS qn", r=REPO)}


def callers_of(graph_db, qualified_name: str) -> int:
    return graph_db.run(
        """
        MATCH (c:Symbol {repo_id: $r})-[:CALLS]->(:Symbol {qualified_name: $qn})
        RETURN count(DISTINCT c) AS n
        """,
        r=REPO, qn=qualified_name,
    )[0]["n"]


def bump(path, text: str) -> None:
    before = path.stat().st_mtime_ns if path.exists() else 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    if path.stat().st_mtime_ns == before:
        os.utime(path, ns=(before + 2_000_000, before + 2_000_000))


async def started(indexer, root) -> TreeWatcher:
    watcher = TreeWatcher(indexer=indexer, repo=REPO, root=root, extensions=EXTENSIONS)
    await asyncio.to_thread(watcher.prime)
    await indexer.startup(REPO)
    await indexer.sync_project(REPO)
    return watcher


def test_a_saved_file_reaches_the_graph(indexer, graph_db, git_repo):
    async def scenario():
        watcher = await started(indexer, git_repo.root)
        bump(git_repo.root / "pkg" / "callee.py",
             'def helper(value):\n    return value * 2\n\n\ndef brand_new(x):\n    return helper(x)\n')
        await watcher.poll_once()
        await indexer.settle()
        await indexer.aclose()

    assert "pkg.callee.brand_new" not in names(graph_db)
    asyncio.run(scenario())
    assert "pkg.callee.brand_new" in names(graph_db)
    assert callers_of(graph_db, "pkg.callee.helper") == 6, "five callers, plus the new one"


def test_a_deleted_files_symbols_leave_the_graph(indexer, graph_db, git_repo):
    async def scenario():
        watcher = await started(indexer, git_repo.root)
        assert "pkg.callers.caller0" in names(graph_db)
        (git_repo.root / "pkg" / "callers.py").unlink()
        await watcher.poll_once()
        await indexer.aclose()

    asyncio.run(scenario())
    left = names(graph_db)
    assert not any(name.startswith("pkg.callers") for name in left)
    assert "pkg.callee.helper" in left, "only the deleted file's symbols went"
    files = {r["p"] for r in graph_db.run(
        "MATCH (f:File {repo_id: $r}) RETURN f.rel_path AS p", r=REPO)}
    assert "pkg/callers.py" not in files


def test_a_staged_git_mv_keeps_every_caller(indexer, graph_db, git_repo):
    """§8.2's move-survival fixture (a), through the watcher instead of an
    editor event. Signal B sees the rename because the save is flushed before
    the reconcile runs — the other order would sever all five edges."""

    async def scenario():
        watcher = await started(indexer, git_repo.root)
        assert callers_of(graph_db, "pkg.callee.helper") == 5
        git_repo.mv("pkg/callee.py", "pkg/moved.py")
        delta = await watcher.poll_once()
        await indexer.aclose()
        return delta

    delta = asyncio.run(scenario())
    assert delta.saved == ["pkg/moved.py"] and delta.removed == ["pkg/callee.py"]
    assert callers_of(graph_db, "pkg.moved.helper") == 5
    assert "pkg.callee.helper" not in names(graph_db)
    assert metrics.get("move.unresolvable") == 0


def test_an_unstaged_move_is_delete_and_create_with_no_leftovers(indexer, graph_db, git_repo):
    """§4.4's "Undetected" row: no remap, by design — but no ghosts either."""

    async def scenario():
        watcher = await started(indexer, git_repo.root)
        (git_repo.root / "pkg" / "callee.py").rename(git_repo.root / "pkg" / "elsewhere.py")
        await watcher.poll_once()
        await indexer.aclose()

    asyncio.run(scenario())
    left = names(graph_db)
    assert "pkg.elsewhere.helper" in left and "pkg.callee.helper" not in left
    assert graph_db.run(
        "MATCH (s:Symbol {repo_id: $r}) WHERE s.rel_path = 'pkg/callee.py' RETURN count(s) AS n",
        r=REPO,
    )[0]["n"] == 0
