"""Write batches are ordered by an epoch that never goes backwards.

Steps 4 and 5 delete what is *older* than the batch: `r.epoch < $epoch`,
`s.epoch < $epoch`. The epoch was `time.time_ns()`, so after the system clock
stepped back (an NTP correction, a laptop resuming, a VM restored) every new
batch was older than the data it replaced, and stale edges and deleted
functions stayed in the graph for as long as the clock was behind.

The epoch is now `max(wall clock, last epoch + 1)`, and "last epoch" is read
back from the graph the first time a repository is written in a process, so
a restart does not reset it.
"""

from __future__ import annotations

import asyncio
import time

import pytest

import index.indexer as indexer_module
import metrics
from graph.reader import GraphReader
from graph.writer import GraphWriter

from .conftest import REPO

TWO = "def keep():\n    return 1\n\n\ndef drop():\n    return 2\n"
ONE = "def keep():\n    return 1\n"


@pytest.fixture(autouse=True)
def clean(graph_db):
    graph_db.run("MATCH (n) DETACH DELETE n")
    metrics.reset()
    yield
    graph_db.run("MATCH (n) DETACH DELETE n")


def names(graph_db) -> set[str]:
    rows = graph_db.run("MATCH (s:Symbol {repo_id: $r}) RETURN s.qualified_name AS qn", r=REPO)
    return {r["qn"] for r in rows}


def build(make_indexer, graph_db, root):
    return make_indexer(GraphWriter(graph_db.driver, graph_db.name),
                        GraphReader(graph_db.driver, graph_db.name),
                        {REPO: root}, debounce=0.01)


def write_and_flush(indexer, root, text: str) -> None:
    (root / "mod.py").write_text(text, encoding="utf-8", newline="\n")

    async def scenario():
        indexer.on_save(REPO, "mod.py", text.encode())
        await indexer.settle()
        await indexer.aclose()

    asyncio.run(scenario())


def test_a_clock_that_steps_back_still_deletes_what_was_removed(
    make_indexer, graph_db, tmp_path, monkeypatch
):
    indexer = build(make_indexer, graph_db, tmp_path)
    write_and_flush(indexer, tmp_path, TWO)
    assert "mod.drop" in names(graph_db)

    an_hour_ago = time.time_ns() - 3_600 * 10**9
    monkeypatch.setattr(indexer_module, "_wall_clock_ns", lambda: an_hour_ago)
    write_and_flush(indexer, tmp_path, ONE)

    assert names(graph_db) == {"mod", "mod.keep"}


def test_the_epoch_survives_a_restart(make_indexer, graph_db, tmp_path, monkeypatch):
    """A new process has no memory of the last epoch; the graph does."""
    write_and_flush(build(make_indexer, graph_db, tmp_path), tmp_path, TWO)

    an_hour_ago = time.time_ns() - 3_600 * 10**9
    monkeypatch.setattr(indexer_module, "_wall_clock_ns", lambda: an_hour_ago)
    write_and_flush(build(make_indexer, graph_db, tmp_path), tmp_path, ONE)

    assert names(graph_db) == {"mod", "mod.keep"}


def test_two_batches_in_the_same_nanosecond_still_order(make_indexer, graph_db, tmp_path, monkeypatch):
    """A frozen clock is the limiting case: each batch must still be newer."""
    frozen = time.time_ns()
    monkeypatch.setattr(indexer_module, "_wall_clock_ns", lambda: frozen)
    indexer = build(make_indexer, graph_db, tmp_path)
    write_and_flush(indexer, tmp_path, TWO)
    write_and_flush(indexer, tmp_path, ONE)
    assert names(graph_db) == {"mod", "mod.keep"}
