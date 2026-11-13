"""`index.watcher` — keeping the graph current with no editor plugin.

The watcher writes nothing itself. It turns what it sees in the working tree
into the two events the indexer already takes: `on_save` for a new or changed
file, and a `reconcile` for files that vanished — in that order, because a
`git mv` looks like both and Signal B must see it first (§4.4).

These tests use a spy indexer to pin *what the watcher decides*, and the real
`Indexer` over a recording writer to pin that the decisions reach §4.1's flush.
"""

from __future__ import annotations

import asyncio
import os

import pytest

import metrics
from index.watcher import TreeWatcher

from .conftest import REPO

EXTENSIONS = (".py", ".pyi")


@pytest.fixture(autouse=True)
def reset():
    metrics.reset()
    yield
    metrics.reset()


class SpyIndexer:
    """Records the watcher's calls, in order."""

    def __init__(self):
        self.calls: list[tuple] = []

    def on_save(self, repo, path, content):
        self.calls.append(("save", path, content))

    async def settle(self):
        self.calls.append(("settle",))

    async def reconcile(self, repo):
        self.calls.append(("reconcile", repo))

    @property
    def saved(self) -> list[str]:
        return [c[1] for c in self.calls if c[0] == "save"]


def bump(path, text: str) -> None:
    """Write `text` and make sure the mtime moves, whatever the clock's grain."""
    before = path.stat().st_mtime_ns if path.exists() else 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    if path.stat().st_mtime_ns == before:
        os.utime(path, ns=(before + 2_000_000, before + 2_000_000))


def watcher_for(root, indexer) -> TreeWatcher:
    watcher = TreeWatcher(indexer=indexer, repo=REPO, root=root, extensions=EXTENSIONS)
    watcher.prime()
    return watcher


def poll(watcher):
    return asyncio.run(watcher.poll_once())


@pytest.fixture
def tree(tmp_path):
    """A plain directory — no git — so the fixed ignore list applies."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    (tmp_path / "pkg" / "b.py").write_text("def b():\n    return 2\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("# notes\n", encoding="utf-8")
    return tmp_path


def test_priming_reports_nothing(tree):
    """What differs at start-up is the start-up reconcile's job, not an event."""
    spy = SpyIndexer()
    watcher = watcher_for(tree, spy)
    assert poll(watcher).saved == [] and spy.calls == []
    assert watcher.files_watched == 2


def test_a_modified_file_is_saved_and_only_that_file(tree):
    spy = SpyIndexer()
    watcher = watcher_for(tree, spy)
    bump(tree / "pkg" / "a.py", "def a():\n    return 100\n")

    delta = poll(watcher)
    assert delta.saved == ["pkg/a.py"] and delta.removed == []
    assert spy.calls == [("save", "pkg/a.py", b"def a():\n    return 100\n")]


def test_a_new_file_is_saved(tree):
    spy = SpyIndexer()
    watcher = watcher_for(tree, spy)
    bump(tree / "pkg" / "sub" / "c.py", "def c():\n    return 3\n")
    assert poll(watcher).saved == ["pkg/sub/c.py"]


def test_touching_a_file_without_changing_it_is_not_a_save(tree):
    spy = SpyIndexer()
    watcher = watcher_for(tree, spy)
    target = tree / "pkg" / "a.py"
    bump(target, "def a():\n    return 5\n")
    poll(watcher)

    stamp = target.stat().st_mtime_ns + 5_000_000
    os.utime(target, ns=(stamp, stamp))
    assert poll(watcher).saved == [], "same bytes: nothing to index"
    assert spy.saved == ["pkg/a.py"]


def test_files_that_are_not_source_are_ignored(tree):
    spy = SpyIndexer()
    watcher = watcher_for(tree, spy)
    bump(tree / "notes.md", "# changed\n")
    bump(tree / "data.json", "{}\n")
    assert not poll(watcher) and spy.calls == []


def test_dependency_and_cache_directories_are_not_walked(tree):
    spy = SpyIndexer()
    watcher = watcher_for(tree, spy)
    bump(tree / "node_modules" / "lib" / "x.py", "def x(): pass\n")
    bump(tree / ".venv" / "site.py", "def y(): pass\n")
    bump(tree / "pkg" / "__pycache__" / "z.py", "def z(): pass\n")
    assert not poll(watcher) and spy.calls == []


def test_a_deleted_file_reconciles_after_pending_flushes_settle(tree):
    """Settle first: a flush in flight may be about to remap this very path."""
    spy = SpyIndexer()
    watcher = watcher_for(tree, spy)
    (tree / "pkg" / "b.py").unlink()

    delta = poll(watcher)
    assert delta.removed == ["pkg/b.py"] and delta.saved == []
    assert spy.calls == [("settle",), ("reconcile", REPO)]
    assert watcher.files_watched == 1


def test_a_move_is_a_save_then_a_reconcile_in_that_order(tree):
    """`git mv` and plain `mv` look the same here: one path gone, one new.
    The new path is saved first so Signal B can turn the pair into a remap."""
    spy = SpyIndexer()
    watcher = watcher_for(tree, spy)
    (tree / "pkg" / "b.py").rename(tree / "pkg" / "moved.py")

    delta = poll(watcher)
    assert delta.saved == ["pkg/moved.py"] and delta.removed == ["pkg/b.py"]
    assert [c[0] for c in spy.calls] == ["save", "settle", "reconcile"]


def test_nothing_changed_costs_no_listing(tree, monkeypatch):
    """An idle poll is a stat walk and nothing else — no git, no file reads."""
    watcher = watcher_for(tree, SpyIndexer())
    monkeypatch.setattr("index.watcher.list_indexable",
                        lambda *a: pytest.fail("listed files on an idle poll"))
    assert not poll(watcher)


def test_gitignore_is_respected(git_repo):
    git_repo.write(".gitignore", "generated/\n")
    git_repo.write("pkg/a.py", "def a():\n    return 1\n")
    git_repo.commit()
    spy = SpyIndexer()
    watcher = watcher_for(git_repo.root, spy)

    bump(git_repo.root / "generated" / "out.py", "def gen(): pass\n")
    bump(git_repo.root / "pkg" / "new.py", "def new(): pass\n")
    assert poll(watcher).saved == ["pkg/new.py"]


def test_a_file_un_ignored_by_editing_gitignore_is_picked_up(git_repo):
    """It became indexable without changing — its own mtime never moved."""
    git_repo.write(".gitignore", "generated/\n")
    git_repo.write("generated/out.py", "def gen(): pass\n")
    git_repo.write("pkg/a.py", "def a(): pass\n")
    git_repo.commit()
    spy = SpyIndexer()
    watcher = watcher_for(git_repo.root, spy)
    assert watcher.files_watched == 1

    bump(git_repo.root / ".gitignore", "# nothing ignored now\n")
    bump(git_repo.root / "pkg" / "a.py", "def a(): return 1\n")
    assert sorted(poll(watcher).saved) == ["generated/out.py", "pkg/a.py"]


def test_a_tracked_file_deleted_from_disk_counts_as_removed(git_repo):
    """git still lists it until the deletion is staged; the disk is the truth."""
    git_repo.write("pkg/a.py", "def a(): pass\n")
    git_repo.write("pkg/b.py", "def b(): pass\n")
    git_repo.commit()
    spy = SpyIndexer()
    watcher = watcher_for(git_repo.root, spy)
    (git_repo.root / "pkg" / "b.py").unlink()
    assert poll(watcher).removed == ["pkg/b.py"]


# -- into the real indexer --------------------------------------------------------


def test_a_change_reaches_the_flush(make_indexer, recording_writer, stub_reader, tree):
    indexer = make_indexer(recording_writer, stub_reader, {REPO: tree}, debounce=0.02)

    async def scenario():
        watcher = watcher_for(tree, indexer)
        bump(tree / "pkg" / "a.py", "def a():\n    return 7\n")
        await watcher.poll_once()
        await indexer.settle()
        await indexer.aclose()

    asyncio.run(scenario())
    assert recording_writer.last.paths == {"pkg/a.py"}


def test_a_deletion_reaches_the_writer_as_a_scoped_delete(
    make_indexer, recording_writer, stub_reader, tree
):
    """§4.5: a known path that is gone is applied with an empty batch, so
    §11.2 steps 4-5 remove its symbols."""
    stub_reader.paths[REPO] = {"pkg/b.py": "old-hash"}
    indexer = make_indexer(recording_writer, stub_reader, {REPO: tree}, debounce=0.02)

    async def scenario():
        watcher = watcher_for(tree, indexer)
        (tree / "pkg" / "b.py").unlink()
        await watcher.poll_once()
        await indexer.aclose()

    asyncio.run(scenario())
    deletes = [a for a in recording_writer.applied if not a.paths]
    assert deletes and deletes[-1].batch_paths == {"pkg/b.py"}


def test_settle_waits_for_an_armed_debounce(make_indexer, recording_writer, stub_reader, tree):
    """`join` alone returns at once while a save is still inside its window."""
    indexer = make_indexer(recording_writer, stub_reader, {REPO: tree}, debounce=0.15)

    async def scenario():
        indexer.on_save(REPO, "pkg/a.py", b"def a():\n    return 1\n")
        await indexer.join()
        assert recording_writer.applied == [], "join does not wait for a timer"
        await indexer.settle()
        assert recording_writer.last.paths == {"pkg/a.py"}
        await indexer.aclose()

    asyncio.run(scenario())


# -- the loop ---------------------------------------------------------------------


def test_run_indexes_first_then_watches(tree):
    spy = SpyIndexer()
    watcher = TreeWatcher(indexer=spy, repo=REPO, root=tree, extensions=EXTENSIONS, interval=0.02)
    states: list[str] = []

    async def bootstrap():
        states.append(watcher.state)

    async def scenario():
        task = asyncio.create_task(watcher.run(bootstrap))
        for _ in range(200):
            await asyncio.sleep(0.01)
            if watcher.state == "watching":
                break
        bump(tree / "pkg" / "a.py", "def a():\n    return 9\n")
        for _ in range(200):
            await asyncio.sleep(0.01)
            if spy.saved:
                break
        watcher.stop()
        await task

    asyncio.run(scenario())
    assert states == ["indexing"]
    assert spy.saved == ["pkg/a.py"]
    assert watcher.state == "stopped"


def test_a_failing_start_up_index_is_retried(tree, monkeypatch):
    """The usual cause is a database still booting; giving up would leave the
    server answering from an index nothing maintains."""
    monkeypatch.setattr("index.watcher._BOOTSTRAP_RETRY_SECONDS", 0.01)
    watcher = TreeWatcher(indexer=SpyIndexer(), repo=REPO, root=tree,
                          extensions=EXTENSIONS, interval=0.01)
    attempts = []

    async def bootstrap():
        attempts.append(watcher.state)
        if len(attempts) < 3:
            raise ConnectionError("database not ready")

    async def scenario():
        task = asyncio.create_task(watcher.run(bootstrap))
        for _ in range(300):
            await asyncio.sleep(0.01)
            if watcher.state == "watching":
                break
        assert watcher.last_error is None
        watcher.stop()
        await task

    asyncio.run(scenario())
    assert len(attempts) == 3
    assert metrics.get("watch.errors") == 2


def test_a_failed_poll_is_retried_not_lost(tree):
    """The file's mtime moves once. If the snapshot advanced past a save that
    failed, the change would never be seen again."""

    class Flaky(SpyIndexer):
        failures = 2

        def on_save(self, repo, path, content):
            if self.failures:
                self.failures -= 1
                raise RuntimeError("transient")
            super().on_save(repo, path, content)

    spy = Flaky()
    watcher = watcher_for(tree, spy)
    bump(tree / "pkg" / "a.py", "def a():\n    return 11\n")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            poll(watcher)
    assert poll(watcher).saved == ["pkg/a.py"]
    assert poll(watcher).saved == [], "delivered once, not again"


def test_a_failed_reconcile_keeps_the_deletion_pending(tree):
    class Flaky(SpyIndexer):
        failures = 1

        async def reconcile(self, repo):
            if self.failures:
                self.failures -= 1
                raise ConnectionError("database away")
            await super().reconcile(repo)

    spy = Flaky()
    watcher = watcher_for(tree, spy)
    (tree / "pkg" / "b.py").unlink()

    with pytest.raises(ConnectionError):
        poll(watcher)
    assert poll(watcher).removed == ["pkg/b.py"]
    assert ("reconcile", REPO) in spy.calls


def test_one_bad_poll_does_not_stop_the_loop(tree):
    class Flaky(SpyIndexer):
        failures = 1

        def on_save(self, repo, path, content):
            if self.failures:
                self.failures -= 1
                raise RuntimeError("transient")
            super().on_save(repo, path, content)

    spy = Flaky()
    watcher = TreeWatcher(indexer=spy, repo=REPO, root=tree, extensions=EXTENSIONS, interval=0.01)

    async def scenario():
        task = asyncio.create_task(watcher.run())
        await asyncio.sleep(0.05)
        bump(tree / "pkg" / "a.py", "def a():\n    return 11\n")
        for _ in range(300):
            await asyncio.sleep(0.01)
            if spy.saved:
                break
        watcher.stop()
        await task

    asyncio.run(scenario())
    assert spy.saved == ["pkg/a.py"], "the loop survived the error and retried"
    assert metrics.get("watch.errors") == 1
