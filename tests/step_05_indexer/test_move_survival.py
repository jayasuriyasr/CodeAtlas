"""Step 5 — §8.2's three move-survival fixtures, against a real graph.

This is the module that can actually answer the question §4.4 poses. The
recording double in `test_debounce_and_flush.py` shows which remaps were
*issued*; only a database can show whether the inbound edges *survived*.

§4.4, on what failure looks like if they do not:

> the system then reports zero callers for code with dozens, which is worse
> than an error because it looks like an answer.

§8.2 gates inbound-edge survival across a move at >=0.95. These fixtures are
where that number comes from.
"""

from __future__ import annotations

import asyncio

import pytest

import metrics
from config import BULK_THRESHOLD
from graph.reader import GraphReader
from graph.writer import GraphWriter

from .conftest import CALLEE, CALLERS, REPO, seed_move_fixture

SETTLE = 0.25


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
def indexed(make_indexer, writer, reader, git_repo):
    """A git repo with `pkg/callee.helper` and five callers, fully indexed."""
    seed_move_fixture(git_repo)
    indexer = make_indexer(writer, reader, {REPO: git_repo.root})
    asyncio.run(indexer.full_reconcile(REPO))
    asyncio.run(indexer.startup(REPO))
    return indexer


def helper_callers(reader: GraphReader, graph_db) -> list[str]:
    """Inbound CALLS on whichever node currently is `helper`.

    Found by name rather than by UID on purpose: the UID is the thing under
    test, so looking it up by UID would assume the answer.
    """
    rows = graph_db.run(
        """
        MATCH (h:Symbol {repo_id: $r, name: 'helper', kind: 'function'})
        OPTIONAL MATCH (c:Symbol)-[:CALLS]->(h)
        RETURN h.uid AS uid, h.rel_path AS rel, collect(c.qualified_name) AS callers
        """,
        r=REPO,
    )
    assert len(rows) == 1, f"expected exactly one helper, got {len(rows)}"
    return sorted(n for n in rows[0]["callers"] if n)


async def _settle(indexer) -> None:
    await asyncio.sleep(SETTLE)
    await indexer.join()


# --------------------------------------------------------------------------
# The baseline the fixtures depend on
# --------------------------------------------------------------------------


def test_fixture_has_five_inbound_callers(indexed, reader, graph_db):
    """§8.2 fixture (a) specifies ">=5 known inbound callers".

    If the baseline were zero, every survival assertion below would pass
    vacuously — which is the most likely way this whole module goes green while
    testing nothing.
    """
    callers = helper_callers(reader, graph_db)
    assert callers == [f"pkg.callers.caller{i}" for i in range(5)], callers


# --------------------------------------------------------------------------
# Fixture (a) — git mv, one file
# --------------------------------------------------------------------------


def test_move_preserves_inbound_edges(indexed, reader, graph_db, git_repo):
    """§8.2 fixture (a): `git mv` a file with >=5 callers, flush, count unchanged.

    The edges are never touched. §4.4: "Because UID is the identity every edge
    references, rewriting it preserves inbound edges without touching them."
    """
    before = helper_callers(reader, graph_db)

    async def main():
        git_repo.mv("pkg/callee.py", "pkg/relocated.py")
        indexed.on_rename(REPO, "pkg/callee.py", "pkg/relocated.py")
        await _settle(indexed)

    asyncio.run(main())

    assert helper_callers(reader, graph_db) == before
    assert metrics.get("move.unresolvable") == 0
    assert metrics.get("move.requalify_failed") == 0

    moved = graph_db.run(
        "MATCH (h:Symbol {repo_id: $r, name: 'helper', kind: 'function'}) "
        "RETURN h.rel_path AS rel, h.origin_path AS origin, h.qualified_name AS qn",
        r=REPO,
    )[0]
    assert moved["rel"] == "pkg/relocated.py"
    assert moved["origin"] == "pkg/relocated.py"
    assert moved["qn"] == "pkg.relocated.helper", (
        "the qualified name follows the module path — this is F-007's premise"
    )


def test_move_leaves_no_duplicate_symbol(indexed, graph_db, git_repo):
    """A rewrite, not a copy. Two nodes would split the callers between them."""

    async def main():
        git_repo.mv("pkg/callee.py", "pkg/relocated.py")
        indexed.on_rename(REPO, "pkg/callee.py", "pkg/relocated.py")
        await _settle(indexed)

    asyncio.run(main())

    rows = graph_db.run(
        "MATCH (s:Symbol {repo_id: $r}) WHERE s.rel_path STARTS WITH 'pkg/callee' "
        "RETURN count(s) AS n",
        r=REPO,
    )
    assert rows[0]["n"] == 0, "the old path still has symbols"


def test_save_then_rename_leaves_no_retired_uid(indexed, reader, graph_db, git_repo):
    """v9.1 #14 against the graph.

    The batch-level version is in `test_debounce_and_flush.py`. This is what it
    was protecting: without `pending.pop(mv.old_path)`, step 3a MERGEs a node at
    the retired UID and the graph ends up with two `helper`s.
    """
    before = helper_callers(reader, graph_db)

    async def main():
        indexed.on_save(REPO, "pkg/callee.py", CALLEE.encode())
        git_repo.mv("pkg/callee.py", "pkg/relocated.py")
        indexed.on_rename(REPO, "pkg/callee.py", "pkg/relocated.py")
        await _settle(indexed)

    asyncio.run(main())

    assert helper_callers(reader, graph_db) == before
    assert graph_db.run(
        "MATCH (s:Symbol {repo_id: $r, name: 'helper'}) RETURN count(s) AS n", r=REPO
    )[0]["n"] == 1


def test_remap_collision_cleans_orphan(indexed, graph_db, git_repo):
    """v9.1 #16 against the graph.

    A file moved onto an occupied path: §4.4 skips the remap and the symbol
    takes the normal upsert path as new. The retired node is reachable by steps
    4-5 **only because `batch_paths` includes `old_path`** — drop that and the
    orphan is permanently uncollectable.
    """

    async def main():
        # `occupied.py` already holds a symbol whose UID the move would collide with.
        git_repo.write("pkg/occupied.py", CALLEE)
        indexed.on_save(REPO, "pkg/occupied.py", CALLEE.encode())
        await _settle(indexed)

        # `git rm` refuses an untracked file, and `occupied.py` was written
        # but never committed. Remove it from the worktree and let `git mv`
        # take the path.
        (git_repo.root / "pkg/occupied.py").unlink()
        git_repo.mv("pkg/callee.py", "pkg/occupied.py")
        indexed.on_rename(REPO, "pkg/callee.py", "pkg/occupied.py")
        await _settle(indexed)

    asyncio.run(main())

    orphans = graph_db.run(
        "MATCH (s:Symbol {repo_id: $r, origin_path: 'pkg/callee.py'}) "
        "RETURN s.qualified_name AS qn",
        r=REPO,
    )
    assert orphans == [], f"orphaned symbols left at the old path: {orphans}"


# --------------------------------------------------------------------------
# Fixture (b) — the bulk branch
# --------------------------------------------------------------------------


def test_bulk_rename_preserves_edges(make_indexer, writer, reader, graph_db, git_repo):
    """§8.2 fixture (b), guarding the v8.2 flush defect.

    §4.1 puts the remap **before** the bulk branch for exactly this reason:

    > Remaps apply before the bulk branch: after the rewrite, graph and
    > filesystem agree, so reconcile finds less to do.

    Move it after, and a rename large enough to trip `BULK_THRESHOLD` loses
    every move to `full_reconcile`, which sees only deletes and adds.

    The threshold is lowered here rather than writing 200 files: the defect is
    in the *branch*, and a six-file batch takes it just as truly as a
    six-hundred-file one. `test_bulk_threshold_matches_the_spec_constant` pins
    the shipped value.
    """
    seed_move_fixture(git_repo)
    indexer = make_indexer(writer, reader, {REPO: git_repo.root}, bulk_threshold=3)
    asyncio.run(indexer.full_reconcile(REPO))
    asyncio.run(indexer.startup(REPO))

    before = helper_callers(reader, graph_db)
    assert len(before) == 5

    async def main():
        # Six renames in one window: over the (lowered) threshold.
        git_repo.mv("pkg/callee.py", "pkg/relocated.py")
        indexer.on_rename(REPO, "pkg/callee.py", "pkg/relocated.py")
        for i in range(5):
            git_repo.write(f"pkg/filler{i}.py", f"def filler{i}():\n    pass\n")
            indexer.on_save(
                REPO, f"pkg/filler{i}.py",
                (git_repo.root / f"pkg/filler{i}.py").read_bytes(),
            )
        await _settle(indexer)

    asyncio.run(main())

    assert helper_callers(reader, graph_db) == before, (
        "the bulk branch dropped the move and severed the callers"
    )
    assert metrics.get("move.unresolvable") == 0


def test_bulk_threshold_matches_the_spec_constant():
    """† §4.1. Lowered in the test above; pinned here."""
    assert BULK_THRESHOLD == 200


# --------------------------------------------------------------------------
# Fixture (c) — the rename chain
# --------------------------------------------------------------------------


def test_rename_chain_single_remap(indexed, reader, graph_db, git_repo):
    """§8.2 fixture (c): A->B then B->C in one window.

    One net remap, `move.unresolvable == 0`, caller count unchanged. Issuing
    both remaps would try to rewrite B — a path that is not on disk and has no
    `ParsedFile`.
    """
    before = helper_callers(reader, graph_db)

    async def main():
        git_repo.mv("pkg/callee.py", "pkg/middle.py")
        indexed.on_rename(REPO, "pkg/callee.py", "pkg/middle.py")
        git_repo.mv("pkg/middle.py", "pkg/final.py")
        indexed.on_rename(REPO, "pkg/middle.py", "pkg/final.py")
        await _settle(indexed)

    asyncio.run(main())

    assert helper_callers(reader, graph_db) == before
    assert metrics.get("move.unresolvable") == 0

    rows = graph_db.run(
        "MATCH (s:Symbol {repo_id: $r, name: 'helper'}) RETURN s.rel_path AS rel", r=REPO
    )
    assert [row["rel"] for row in rows] == ["pkg/final.py"]
    assert graph_db.run(
        "MATCH (s:Symbol {repo_id: $r}) WHERE s.rel_path CONTAINS 'middle' "
        "RETURN count(s) AS n",
        r=REPO,
    )[0]["n"] == 0, "the intermediate path left a node behind"


# --------------------------------------------------------------------------
# Signal B, end to end
# --------------------------------------------------------------------------


def test_signal_b_move_preserves_edges(indexed, reader, graph_db, git_repo):
    """v9.1 #17 against the graph: a `git mv` with no editor event.

    `collect_moves` parses the new path itself, so the remap resolves and the
    edges survive — the same outcome as Signal A, one flush later.
    """
    before = helper_callers(reader, graph_db)
    git_repo.mv("pkg/callee.py", "pkg/relocated.py")        # no on_rename

    async def main():
        git_repo.write("pkg/unrelated.py", "def unrelated():\n    pass\n")
        indexed.on_save(
            REPO, "pkg/unrelated.py",
            (git_repo.root / "pkg/unrelated.py").read_bytes(),
        )
        await _settle(indexed)

    asyncio.run(main())

    assert helper_callers(reader, graph_db) == before
    assert metrics.get("move.unresolvable") == 0


# --------------------------------------------------------------------------
# The §8.2 gate itself
# --------------------------------------------------------------------------


def test_inbound_edge_survival_rate(indexed, reader, graph_db, git_repo):
    """§8.2: inbound-edge survival across a move, gated at >=0.95.

    Reported as the ratio the gate is written in, so the number in PROGRESS.md
    is the number the gate names rather than a re-derivation of it.
    """
    before = set(helper_callers(reader, graph_db))
    assert before, "no baseline edges"

    async def main():
        git_repo.mv("pkg/callee.py", "pkg/relocated.py")
        indexed.on_rename(REPO, "pkg/callee.py", "pkg/relocated.py")
        await _settle(indexed)

    asyncio.run(main())

    after = set(helper_callers(reader, graph_db))
    survival = len(before & after) / len(before)
    assert survival >= 0.95, f"inbound-edge survival {survival:.2f} < 0.95"
    assert survival == 1.0, "an in-place UID rewrite should lose nothing at all"
