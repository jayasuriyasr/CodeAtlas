"""Step 5 — §4.1's debounce contract and the traps around `collect_moves`.

These run against `RecordingWriter`, so what they assert is what the indexer
*decided*: which paths were in the batch, what `batch_paths` was, which remaps
were issued, how many epochs there were. That is exactly the layer §4.1
specifies. Whether the resulting graph keeps its edges is a different question,
answered in `test_move_survival.py` against a real database.

Every test drives a real event loop via `asyncio.run`, because the debounce is
`loop.call_later` and a test that stubbed the timer would be testing the stub.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

import metrics
from adapters.base import symbol_uid
from config import DEBOUNCE_SECONDS
from index.indexer import Indexer, RenameEvent

from .conftest import CALLEE, CALLERS, OTHER_REPO, REPO, seed_move_fixture

SETTLE = 0.25          # comfortably past the 50ms test debounce


@pytest.fixture(autouse=True)
def reset_metrics():
    metrics.reset()
    yield
    metrics.reset()


@pytest.fixture
def indexer(make_indexer, recording_writer, stub_reader, git_repo):
    seed_move_fixture(git_repo)
    return make_indexer(recording_writer, stub_reader, {REPO: git_repo.root})


async def _settle(indexer: Indexer) -> None:
    await asyncio.sleep(SETTLE)
    await indexer.join()


# --------------------------------------------------------------------------
# Debounce
# --------------------------------------------------------------------------


def test_debounce_coalesces(indexer, recording_writer, git_repo):
    """Eight saves inside one window produce one flush and one epoch.

    §4.3 step 1: "epoch = now_ns() -- once, for the whole batch". Two epochs
    would mean two write sequences, and the second one's step 4 would delete
    edges the first had just written under the older epoch.
    """

    async def main():
        for i in range(8):
            git_repo.write(f"pkg/mod{i}.py", f"def f{i}():\n    return {i}\n")
            indexer.on_save(
                REPO, f"pkg/mod{i}.py", (git_repo.root / f"pkg/mod{i}.py").read_bytes()
            )
            await asyncio.sleep(0.005)          # 8 saves well inside the window
        await _settle(indexer)

    asyncio.run(main())

    assert len(recording_writer.applied) == 1, (
        f"expected one flush, got {len(recording_writer.applied)}"
    )
    assert len(set(recording_writer.epochs)) == 1
    assert recording_writer.last.paths == {f"pkg/mod{i}.py" for i in range(8)}


def test_a_save_after_the_window_starts_a_new_batch(indexer, recording_writer, git_repo):
    """The debounce is idle-based, not a fixed schedule."""

    async def main():
        git_repo.write("pkg/one.py", "def one():\n    pass\n")
        indexer.on_save(REPO, "pkg/one.py", (git_repo.root / "pkg/one.py").read_bytes())
        await _settle(indexer)

        git_repo.write("pkg/two.py", "def two():\n    pass\n")
        indexer.on_save(REPO, "pkg/two.py", (git_repo.root / "pkg/two.py").read_bytes())
        await _settle(indexer)

    asyncio.run(main())

    assert [a.paths for a in recording_writer.applied] == [{"pkg/one.py"}, {"pkg/two.py"}]
    assert len(set(recording_writer.epochs)) == 2


def test_debounce_matches_the_spec_constant():
    """The tests above use 50ms to stay fast; §4.1's contract is 2s idle.

    `DEBOUNCE_SECONDS` is a † constant (§12.10) and lives in `config.py`, so
    this checks the shipped default rather than a literal in the indexer.
    """
    assert DEBOUNCE_SECONDS == 2.0
    assert Indexer.__dataclass_fields__["debounce"].default == DEBOUNCE_SECONDS


def test_pending_state_is_cleared_by_the_flush(indexer, recording_writer, git_repo):
    """A batch left in `_pending` would be rewritten under a second epoch."""

    async def main():
        git_repo.write("pkg/one.py", "def one():\n    pass\n")
        indexer.on_save(REPO, "pkg/one.py", (git_repo.root / "pkg/one.py").read_bytes())
        await _settle(indexer)
        assert indexer._pending[REPO] == {}
        await _settle(indexer)

    asyncio.run(main())
    assert len(recording_writer.applied) == 1


def test_unindexable_extensions_are_ignored(indexer, recording_writer, git_repo):
    """A README save must not schedule a flush that writes nothing."""

    async def main():
        git_repo.write("README.md", "# hello\n")
        indexer.on_save(REPO, "README.md", b"# hello\n")
        await _settle(indexer)

    asyncio.run(main())
    assert recording_writer.applied == []


# --------------------------------------------------------------------------
# Per-repo lock
# --------------------------------------------------------------------------


def test_concurrent_repos_dont_block(make_indexer, recording_writer, stub_reader,
                                     git_repo, tmp_path):
    """§4.1: "per-repo, not global".

    Repo A is held inside `writer.apply`. Repo B must still get through. A
    global lock would make one large repository's flush stall every other
    tenant's saves — invisible in development, and someone's outage later.

    Waits on an event rather than a sleep. The sleeping version passed alone and
    failed under load, and chasing that flake is what surfaced F-008: §4.1 runs
    Signal B's git subprocess on the event loop, so repo A could stall repo B
    even with a perfectly correct per-repo mutex.
    """
    seed_move_fixture(git_repo)
    other_root = tmp_path / "other"
    other_root.mkdir()
    (other_root / "solo.py").write_text("def solo():\n    pass\n", encoding="utf-8")

    gate = threading.Event()
    reached_b = threading.Event()
    recording_writer.gate = gate
    recording_writer.gate_repo = REPO
    recording_writer.signal = reached_b
    recording_writer.signal_repo = OTHER_REPO

    indexer = make_indexer(
        recording_writer, stub_reader, {REPO: git_repo.root, OTHER_REPO: other_root}
    )

    async def main():
        git_repo.write("pkg/slow.py", "def slow():\n    pass\n")
        indexer.on_save(REPO, "pkg/slow.py", (git_repo.root / "pkg/slow.py").read_bytes())
        indexer.on_save(OTHER_REPO, "solo.py", (other_root / "solo.py").read_bytes())

        # Repo A is parked inside apply. Repo B must reach its own apply anyway.
        got_through = await asyncio.to_thread(reached_b.wait, 10.0)
        assert got_through, (
            "repo B never reached the writer while repo A was held — the lock is "
            "global, or the event loop is blocked"
        )
        gate.set()
        await indexer.join()

    asyncio.run(main())

    assert {a.repo_id for a in recording_writer.applied} == {REPO, OTHER_REPO}


def test_same_repo_flushes_serialise(make_indexer, recording_writer, stub_reader,
                                     git_repo):
    """The other half: within one repo the mutex must actually hold.

    Two concurrent write sequences against one repo would interleave §11.2's
    steps, and step 4's delete would run against the other batch's epoch.
    """
    seed_move_fixture(git_repo)
    indexer = make_indexer(recording_writer, stub_reader, {REPO: git_repo.root})
    order: list[str] = []

    original = recording_writer.apply

    def tracking_apply(*args, **kw):
        order.append("enter")
        original(*args, **kw)
        order.append("exit")

    recording_writer.apply = tracking_apply

    async def main():
        git_repo.write("pkg/x.py", "def x():\n    pass\n")
        indexer.on_save(REPO, "pkg/x.py", (git_repo.root / "pkg/x.py").read_bytes())
        indexer._fire(REPO)                     # force a second concurrent flush
        indexer._fire(REPO)
        await _settle(indexer)

    asyncio.run(main())
    assert order == ["enter", "exit"] * (len(order) // 2), (
        f"flushes interleaved: {order}"
    )


# --------------------------------------------------------------------------
# v9.1 #14 — the stale pending entry
# --------------------------------------------------------------------------


def test_save_then_rename_no_duplicate(indexer, recording_writer, git_repo):
    """v9.1 #14. A save followed by a rename must not recreate the retired UID.

    §4.1: `pending.pop(mv.old_path, None)` — "or the upsert recreates old_uid".
    The user edits `callee.py`, then renames it. Without the eviction the batch
    still holds the old path, step 3a MERGEs a node at the old UID, and the
    remap that just retired it is undone.
    """

    async def main():
        indexer.on_save(REPO, "pkg/callee.py", CALLEE.encode())
        git_repo.mv("pkg/callee.py", "pkg/moved.py")
        indexer.on_rename(REPO, "pkg/callee.py", "pkg/moved.py")
        await _settle(indexer)

    asyncio.run(main())

    applied = recording_writer.last
    assert "pkg/callee.py" not in applied.paths, (
        "the moved-from path is still in the batch; the upsert will recreate old_uid"
    )
    assert "pkg/moved.py" in applied.paths

    # ...but it *is* still in scope for steps 4-5, which is a different thing.
    assert "pkg/callee.py" in applied.batch_paths

    # The old UID is the one the graph actually stored: computed at the old
    # path, with the qualified name the old path produced (F-007).
    old_uid = symbol_uid(REPO, "pkg/callee.py", "pkg.callee.helper", 1, 0)
    new_uid = symbol_uid(REPO, "pkg/moved.py", "pkg.moved.helper", 1, 0)
    remaps = recording_writer.all_remaps
    assert old_uid in {r.old_uid for r in remaps}, (
        "the remap does not name the UID the graph holds, so it will rewrite nothing"
    )
    assert new_uid in {r.new_uid for r in remaps}
    assert old_uid not in {r.new_uid for r in remaps}
    assert metrics.get("move.unresolvable") == 0


# --------------------------------------------------------------------------
# v9.1 #16 — batch_paths includes old_path
# --------------------------------------------------------------------------


def test_remap_collision_cleans_orphan(indexer, recording_writer, git_repo):
    """v9.1 #16. `batch_paths` must be `batch.keys() | {move.old_path}`.

    §4.4: on collision the remap is skipped and the symbol takes the normal
    upsert path as new — and "the retired node is reachable by steps 4-5 only
    because `batch_paths` includes `old_path`". Drop it and the orphan is
    permanently uncollectable, with un-repointed edges pointing at it.
    """

    async def main():
        git_repo.mv("pkg/callee.py", "pkg/moved.py")
        indexer.on_rename(REPO, "pkg/callee.py", "pkg/moved.py")
        await _settle(indexer)

    asyncio.run(main())

    applied = recording_writer.last
    assert applied.batch_paths == applied.paths | {"pkg/callee.py"}
    assert applied.batch_paths - applied.paths == {"pkg/callee.py"}


def test_batch_paths_equals_the_spec_expression(indexer, recording_writer, git_repo):
    """§4.3 states it as an expression; this checks the expression, not a case.

    The plan's §11 calls `batch_paths` "the single most coupled interface in the
    system" and warns that step 5 is where step 3 most often breaks.
    """

    async def main():
        indexer.on_save(REPO, "pkg/callers.py", CALLERS.encode())
        git_repo.mv("pkg/callee.py", "pkg/relocated.py")
        indexer.on_rename(REPO, "pkg/callee.py", "pkg/relocated.py")
        await _settle(indexer)

    asyncio.run(main())

    applied = recording_writer.last
    assert applied.paths == {"pkg/callers.py", "pkg/relocated.py"}
    assert applied.batch_paths == {"pkg/callers.py", "pkg/relocated.py", "pkg/callee.py"}


# --------------------------------------------------------------------------
# B5 — the rename chain
# --------------------------------------------------------------------------


def test_rename_chain_single_remap(indexer, recording_writer, git_repo):
    """B5. A->B then B->C in one window: one net remap, zero unresolvable.

    Issuing both would try to rewrite B, which is not on disk and has no
    `ParsedFile` — `resolve_moves` would increment `move.unresolvable`, a
    counter §9.4 requires to be identically zero.
    """

    async def main():
        git_repo.mv("pkg/callee.py", "pkg/middle.py")
        indexer.on_rename(REPO, "pkg/callee.py", "pkg/middle.py")
        git_repo.mv("pkg/middle.py", "pkg/final.py")
        indexer.on_rename(REPO, "pkg/middle.py", "pkg/final.py")
        await _settle(indexer)

    asyncio.run(main())

    remaps = recording_writer.all_remaps
    assert remaps, "no remap was issued"
    assert {(r.old_path, r.new_path) for r in remaps} == {
        ("pkg/callee.py", "pkg/final.py")
    }, "the intermediate path leaked into a remap"
    assert metrics.get("move.unresolvable") == 0

    applied = recording_writer.last
    assert "pkg/middle.py" not in applied.paths
    assert applied.batch_paths == {"pkg/final.py", "pkg/callee.py"}


# --------------------------------------------------------------------------
# v9.1 #17 — Signal B's new path is unparsed
# --------------------------------------------------------------------------


def test_signal_b_move_gets_parsed(indexer, recording_writer, git_repo):
    """v9.1 #17. A `git mv` with no IDE event still resolves.

    Signal B reads git, not the editor, so nothing has parsed the new path.
    §4.1's `collect_moves` parses it itself; without that, `resolve_moves` finds
    no `ParsedFile` and increments `move.unresolvable`.

    The flush is triggered here by an unrelated new file, which is also what
    opens Signal B's gate: §4.1 runs git only when Signal A found nothing *and*
    the batch holds a path the graph has never seen.
    """
    git_repo.mv("pkg/callee.py", "pkg/moved.py")            # no on_rename call

    async def main():
        git_repo.write("pkg/unrelated.py", "def unrelated():\n    pass\n")
        indexer.on_save(
            REPO, "pkg/unrelated.py",
            (git_repo.root / "pkg/unrelated.py").read_bytes(),
        )
        await _settle(indexer)

    asyncio.run(main())

    remaps = recording_writer.all_remaps
    assert remaps, "Signal B found no move"
    assert {(r.old_path, r.new_path) for r in remaps} == {
        ("pkg/callee.py", "pkg/moved.py")
    }
    assert metrics.get("move.unresolvable") == 0

    applied = recording_writer.last
    assert "pkg/moved.py" in applied.paths, "collect_moves did not parse the new path"
    assert "pkg/callee.py" in applied.batch_paths


def test_signal_b_is_not_run_when_signal_a_answered(indexer, recording_writer, git_repo):
    """§4.1 gates the subprocess deliberately.

    "Signal B costs a subprocess spawn (~50-200ms on a large repo), so run it
    only when Signal A found nothing." A second, git-derived copy of the same
    move would also be harmless only because `collapse_chains` is idempotent —
    relying on that instead of the gate spends the 200ms anyway.
    """
    git_repo.mv("pkg/callee.py", "pkg/moved.py")

    async def main():
        indexer.on_rename(REPO, "pkg/callee.py", "pkg/moved.py")   # Signal A
        await _settle(indexer)

    asyncio.run(main())

    assert metrics.get("git.unavailable") == 0
    assert {(r.old_path, r.new_path) for r in recording_writer.all_remaps} == {
        ("pkg/callee.py", "pkg/moved.py")
    }


def test_signal_b_is_not_run_when_every_pending_path_is_known(
    indexer, recording_writer, stub_reader, git_repo
):
    """The second half of §4.1's gate: the batch must hold an *unseen* path.

    An ordinary save of a file the graph already knows is the common case, and
    it must not spawn git.
    """
    stub_reader.paths[REPO] = {"pkg/callers.py": "hash"}
    git_repo.mv("pkg/callee.py", "pkg/moved.py")

    async def main():
        await indexer.startup(REPO)
        indexer.on_save(REPO, "pkg/callers.py", CALLERS.encode())
        await _settle(indexer)

    asyncio.run(main())

    assert recording_writer.all_remaps == [], (
        "git ran even though every pending path was already indexed"
    )


# --------------------------------------------------------------------------
# B10 — the new path cannot be read
# --------------------------------------------------------------------------


def test_new_path_unreadable_degrades(indexer, recording_writer, git_repo):
    """B10. No crash, the counter increments, the move degrades to undetected.

    Real scenario: `git mv a b` then delete `b`. git reports it as `RD` — a
    rename in the index whose new path is gone from the worktree — so Signal B
    supplies a move that cannot be parsed.
    """
    git_repo.mv("pkg/callee.py", "pkg/moved.py")
    (git_repo.root / "pkg/moved.py").unlink()
    assert "RD" in git_repo.status()

    async def main():
        git_repo.write("pkg/unrelated.py", "def unrelated():\n    pass\n")
        indexer.on_save(
            REPO, "pkg/unrelated.py",
            (git_repo.root / "pkg/unrelated.py").read_bytes(),
        )
        await _settle(indexer)

    asyncio.run(main())

    assert metrics.get("move.new_path_unreadable") == 1
    assert metrics.get("move.unresolvable") == 0, (
        "an unreadable new path must be dropped before resolve_moves sees it"
    )
    assert recording_writer.all_remaps == []
    # The unrelated save still went through — degrade, don't fail (Principle 3).
    assert recording_writer.last.paths == {"pkg/unrelated.py"}


def test_missing_repo_root_does_not_crash_signal_b(make_indexer, recording_writer,
                                                   stub_reader, tmp_path):
    """A repo with no registered root simply gets no Signal B."""
    root = tmp_path / "norepo"
    root.mkdir()
    (root / "a.py").write_text("def a():\n    pass\n", encoding="utf-8")
    indexer = make_indexer(recording_writer, stub_reader, {REPO: root})

    async def main():
        indexer.on_save(REPO, "a.py", (root / "a.py").read_bytes())
        await _settle(indexer)

    asyncio.run(main())
    assert recording_writer.last.paths == {"a.py"}
    assert recording_writer.all_remaps == []


# --------------------------------------------------------------------------
# Startup and the rename ingress
# --------------------------------------------------------------------------


def test_startup_seeds_indexed_from_the_graph(indexer, stub_reader):
    """§4.1. `_indexed` gates Signal B; an empty seed makes the gate always true.

    Which is also why FINDINGS F-004 mattered: without a `:File` node,
    `known_paths` returns nothing and git runs on every single flush.
    """
    stub_reader.paths[REPO] = {"pkg/callee.py": "h1", "pkg/callers.py": "h2"}
    asyncio.run(indexer.startup(REPO))
    assert indexer._indexed[REPO] == {"pkg/callee.py", "pkg/callers.py"}


def test_rename_event_ingress(indexer, recording_writer, git_repo):
    """§4.4 Signal A arrives at `/index/rename`; §2 calls it an event boundary.

    The HTTP binding is step 9's; this is the transport-agnostic entry point it
    will call, so the boundary is testable before FastAPI exists.
    """

    async def main():
        git_repo.mv("pkg/callee.py", "pkg/moved.py")
        indexer.on_rename_event(
            RenameEvent(repo=REPO, old_path="pkg/callee.py", new_path="pkg/moved.py")
        )
        await _settle(indexer)

    asyncio.run(main())

    assert {(r.old_path, r.new_path) for r in recording_writer.all_remaps} == {
        ("pkg/callee.py", "pkg/moved.py")
    }


def test_vectors_are_supplied_for_every_symbol_in_the_batch(indexer, recording_writer,
                                                            git_repo):
    """T5's precondition, at the indexer boundary.

    `writer.apply` merges these into `sym.props`. A batch that reached the
    writer with no vectors would index every symbol with a null embedding and
    look entirely healthy.
    """

    async def main():
        indexer.on_save(REPO, "pkg/callee.py", CALLEE.encode())
        await _settle(indexer)

    asyncio.run(main())

    applied = recording_writer.last
    assert applied.vector_uids, "no vectors reached the writer"
    parsed = indexer._parse(REPO, "pkg/callee.py", CALLEE.encode())
    assert applied.vector_uids == {s.uid for s in parsed.symbols}
