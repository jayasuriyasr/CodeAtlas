"""Step 5 — §4.5's `full_reconcile`.

T3/T4: it was called three times and specified nowhere, and one §8.3 metric
depended on state nothing provided. §4.5 resolves both — the undetected-move
rate falls out of the reconcile diff, with no stored previous-reconcile state.

Resumability is the property that makes the whole thing safe to interrupt:
each chunk is independently replayable under the same epoch (§4.3), so an
interrupted reconcile resumes by re-running rather than by repair logic.
"""

from __future__ import annotations

import asyncio

import pytest

import metrics
from graph.reader import GraphReader
from graph.writer import GraphWriter
from index.indexer import walk_working_tree

from .conftest import CALLEE, CALLERS, REPO, seed_move_fixture


@pytest.fixture(autouse=True)
def reset_metrics():
    metrics.reset()
    yield
    metrics.reset()


@pytest.fixture(autouse=True)
def clean(graph_db):
    graph_db.run("MATCH (n) DETACH DELETE n")
    yield
    graph_db.run("MATCH (n) DETACH DELETE n")


@pytest.fixture
def writer(graph_db) -> GraphWriter:
    return GraphWriter(graph_db.driver, graph_db.name)


@pytest.fixture
def reader(graph_db) -> GraphReader:
    return GraphReader(graph_db.driver, graph_db.name)


@pytest.fixture
def indexer(make_indexer, writer, reader, git_repo):
    seed_move_fixture(git_repo)
    return make_indexer(writer, reader, {REPO: git_repo.root})


def qualified_names(graph_db) -> set[str]:
    return {
        row["qn"]
        for row in graph_db.run(
            "MATCH (s:Symbol {repo_id: $r}) RETURN s.qualified_name AS qn", r=REPO
        )
    }


def file_paths(graph_db) -> set[str]:
    return {
        row["p"]
        for row in graph_db.run(
            "MATCH (f:File {repo_id: $r}) RETURN f.rel_path AS p", r=REPO
        )
    }


# --------------------------------------------------------------------------
# The diff
# --------------------------------------------------------------------------


def test_reconcile_indexes_the_whole_tree(indexer, graph_db, git_repo):
    asyncio.run(indexer.full_reconcile(REPO))

    assert file_paths(graph_db) == {
        "pkg/__init__.py", "pkg/callee.py", "pkg/callers.py",
    }
    assert "pkg.callee.helper" in qualified_names(graph_db)


def test_reconcile_picks_up_added_modified_and_deleted(indexer, graph_db, git_repo):
    """§4.5's three-way diff, all three arms in one pass."""
    asyncio.run(indexer.full_reconcile(REPO))

    git_repo.write("pkg/added.py", "def added():\n    pass\n")
    git_repo.write("pkg/callee.py", CALLEE + "\n\ndef extra():\n    pass\n")
    (git_repo.root / "pkg/callers.py").unlink()

    asyncio.run(indexer.full_reconcile(REPO))

    names = qualified_names(graph_db)
    assert "pkg.added.added" in names, "added"
    assert "pkg.callee.extra" in names, "modified"
    assert not any(n.startswith("pkg.callers") for n in names), "deleted"
    assert "pkg/callers.py" not in file_paths(graph_db)


def test_reconcile_is_a_noop_when_nothing_changed(indexer, graph_db):
    """`content_hash` is the whole diff.

    If it moved for a reason other than content — line endings, a re-read —
    every reconcile would re-embed the entire tree, which §4.2 calls the
    structural cost control failing silently.
    """
    asyncio.run(indexer.full_reconcile(REPO))
    before = qualified_names(graph_db)

    metrics.reset()
    asyncio.run(indexer.full_reconcile(REPO))

    assert qualified_names(graph_db) == before
    assert metrics.get("cache.miss.total") == 0, "a no-op reconcile re-embedded"


def test_reconcile_respects_gitignore(indexer, graph_db, git_repo):
    """§4.5: "respects .gitignore".

    Indexing a `.venv` would multiply the symbol count by the size of the
    dependency tree, and every one of those symbols competes in retrieval.
    """
    git_repo.write(".gitignore", "ignored/\n")
    git_repo.write("ignored/secret.py", "def secret():\n    pass\n")
    git_repo.commit("add gitignore")

    asyncio.run(indexer.full_reconcile(REPO))

    assert "ignored/secret.py" not in file_paths(graph_db)
    assert "ignored.secret.secret" not in qualified_names(graph_db)


def test_walk_working_tree_outside_git_uses_the_fallback(tmp_path):
    """No git available is a degraded mode, not an error (Principle 3)."""
    root = tmp_path / "plain"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "a.py").write_text("def a():\n    pass\n", encoding="utf-8")
    (root / "pkg" / "__pycache__").mkdir()
    (root / "pkg" / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\x00")
    (root / "notes.md").write_text("# notes\n", encoding="utf-8")

    tree = walk_working_tree(root, (".py",))
    assert set(tree) == {"pkg/a.py"}


# --------------------------------------------------------------------------
# T4 — the undetected-move counter
# --------------------------------------------------------------------------


def test_undetected_move_counted_from_the_diff(indexer, graph_db, git_repo):
    """§8.3's decision metric, produced with no stored previous-reconcile state.

    The scenario is §4.4's last table row: a plain `mv` outside git and outside
    the editor. No signal fires, so the reconcile sees a delete and an add — and
    a symbol with the same `qualified_name` suffix and the same `body_hash` at
    both ends.

    §10.2 uses this ratio to decide whether an import-path repair pass is worth
    building. It is measurement only: §4.4 forbids acting on content similarity,
    because "a wrong merge is silent and corrupts the graph".
    """
    asyncio.run(indexer.full_reconcile(REPO))
    metrics.reset()

    # A move no signal can see: file content unchanged, path changed, no git mv.
    (git_repo.root / "pkg/callee.py").rename(git_repo.root / "pkg/unseen.py")
    asyncio.run(indexer.full_reconcile(REPO))

    assert metrics.get("move.undetected") == 1
    assert "pkg/unseen.py" in file_paths(graph_db)
    assert "pkg/callee.py" not in file_paths(graph_db)


def test_undetected_move_counter_does_not_fire_on_ordinary_edits(indexer, git_repo):
    """A counter that fires on every edit tells §10.2 nothing."""
    asyncio.run(indexer.full_reconcile(REPO))
    metrics.reset()

    git_repo.write("pkg/callee.py", CALLEE + "\n\ndef extra():\n    pass\n")
    git_repo.write("pkg/brandnew.py", "def brandnew():\n    pass\n")
    asyncio.run(indexer.full_reconcile(REPO))

    assert metrics.get("move.undetected") == 0


def test_undetected_move_is_counted_in_files_not_symbols(indexer, git_repo):
    """§8.3's denominator is total *moves*, so the numerator must be files.

    Counting symbols would make a five-symbol file read as five moves and put
    the ratio past its 0.10 threshold on a single unnoticed rename.
    """
    asyncio.run(indexer.full_reconcile(REPO))
    metrics.reset()

    (git_repo.root / "pkg/callers.py").rename(git_repo.root / "pkg/unseen.py")
    asyncio.run(indexer.full_reconcile(REPO))

    assert metrics.get("move.undetected") == 1, (
        "callers.py holds six symbols; the count must still be one move"
    )


def test_undetected_move_does_not_produce_a_remap(indexer, graph_db, git_repo):
    """The counter must never become an action.

    §4.4 rules out content-similarity move detection outright. If this ever
    started remapping, a file split into two sharing boilerplate would silently
    fuse — and nothing downstream could tell.
    """
    asyncio.run(indexer.full_reconcile(REPO))
    old_uid = graph_db.run(
        "MATCH (s:Symbol {repo_id: $r, name: 'helper'}) RETURN s.uid AS uid", r=REPO
    )[0]["uid"]

    (git_repo.root / "pkg/callee.py").rename(git_repo.root / "pkg/unseen.py")
    asyncio.run(indexer.full_reconcile(REPO))

    assert metrics.get("move.undetected") == 1
    # Delete-and-create, exactly as §9.3's degradation row says: callers blind
    # until Sync. The point is that it degraded honestly, not that it recovered.
    rows = graph_db.run(
        "MATCH (s:Symbol {uid: $uid}) RETURN count(s) AS n", uid=old_uid
    )
    assert rows[0]["n"] == 0


# --------------------------------------------------------------------------
# T3 — resumability
# --------------------------------------------------------------------------


class _FailingWriter:
    """Wraps the real writer and dies on the Nth `apply`. §4.3's crash, at scale."""

    def __init__(self, inner: GraphWriter, fail_on: int) -> None:
        self._inner = inner
        self._fail_on = fail_on
        self.calls = 0

    def apply(self, *args, **kw):
        self.calls += 1
        if self.calls == self._fail_on:
            raise RuntimeError("reconcile killed mid-flight")
        return self._inner.apply(*args, **kw)

    def remap_uids(self, *args, **kw):
        return self._inner.remap_uids(*args, **kw)

    def refresh_all_degrees(self, *args, **kw):
        return self._inner.refresh_all_degrees(*args, **kw)


def test_reconcile_resumable(make_indexer, writer, reader, graph_db, git_repo):
    """Kill mid-reconcile, re-run, converge.

    §4.5: "each chunk is independently replayable under the same epoch (§4.3),
    so an interrupted reconcile resumes by re-running". The assertion is that
    the recovered graph equals the graph a clean run produces — not merely that
    the re-run does not raise.
    """
    seed_move_fixture(git_repo)
    for i in range(6):
        git_repo.write(f"pkg/mod{i}.py", f"def f{i}(x):\n    return x + {i}\n")
    git_repo.commit("more files")

    # A clean run, for the reference state.
    clean_indexer = make_indexer(writer, reader, {REPO: git_repo.root}, bulk_threshold=2)
    asyncio.run(clean_indexer.full_reconcile(REPO))
    expected = qualified_names(graph_db)
    assert len(expected) > 10, "fixture too small to have multiple chunks"

    graph_db.run("MATCH (n) DETACH DELETE n")

    # Now the interrupted run: several chunks in, the writer dies.
    failing = _FailingWriter(writer, fail_on=3)
    indexer = make_indexer(failing, reader, {REPO: git_repo.root}, bulk_threshold=2)

    with pytest.raises(RuntimeError, match="killed mid-flight"):
        asyncio.run(indexer.full_reconcile(REPO))

    partial = qualified_names(graph_db)
    assert partial, "nothing was written before the crash; the test proves nothing"
    assert partial < expected, "the crash did not actually interrupt anything"

    # Partial state is queryable throughout (§4.5) — no dangling references.
    assert graph_db.run(
        """
        MATCH (a:Symbol)-[r]->(b:Symbol)
        WHERE a.uid IS NULL OR b.uid IS NULL
        RETURN count(r) AS n
        """
    )[0]["n"] == 0

    # Resume by re-running. No repair logic, no stored progress.
    recovered = make_indexer(writer, reader, {REPO: git_repo.root}, bulk_threshold=2)
    asyncio.run(recovered.full_reconcile(REPO))

    assert qualified_names(graph_db) == expected


def test_reconcile_converges_when_run_twice(indexer, graph_db):
    """Idempotent replay (§4.3), at reconcile scale."""
    asyncio.run(indexer.full_reconcile(REPO))
    first = qualified_names(graph_db)
    asyncio.run(indexer.full_reconcile(REPO))
    assert qualified_names(graph_db) == first


def test_reconcile_refreshes_every_degree(indexer, graph_db):
    """§4.5 ends with a repo-wide degree pass: "one pass beats N local ones".

    A null degree is coalesced to 1 by §11.1's denominator, which systematically
    over-scores whatever the reconcile just wrote (v9.1 #13).
    """
    asyncio.run(indexer.full_reconcile(REPO))
    nulls = graph_db.run(
        "MATCH (s:Symbol {repo_id: $r}) WHERE s.degree IS NULL RETURN count(s) AS n",
        r=REPO,
    )
    assert nulls[0]["n"] == 0


def test_reconcile_seeds_indexed_for_signal_b(indexer, git_repo):
    """§4.5's last line. `_indexed` gates the git subprocess in `collect_moves`."""
    asyncio.run(indexer.full_reconcile(REPO))
    assert indexer._indexed[REPO] == {
        "pkg/__init__.py", "pkg/callee.py", "pkg/callers.py",
    }


def test_sync_project_runs_a_reconcile(indexer, graph_db, git_repo):
    """§4.1's manual trigger and §9.3's recovery path for every undetected move."""
    asyncio.run(indexer.full_reconcile(REPO))
    (git_repo.root / "pkg/callee.py").rename(git_repo.root / "pkg/unseen.py")

    asyncio.run(indexer.sync_project(REPO))

    assert "pkg/unseen.py" in file_paths(graph_db)
    assert "pkg.unseen.helper" in qualified_names(graph_db)
