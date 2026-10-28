"""§9.3's indexing rows: a failed flush converges, a dead embedding API degrades.

"Flush crashes mid-sequence — superset-on-prefix; next flush converges." That
row was only half true: the batch had been drained from `_pending` before the
write, so a failed write lost it, and the "next flush" did not contain it. These
tests hold the other half — the batch goes back, and nothing is dropped.
"""

from __future__ import annotations

import asyncio

import pytest

import metrics
from config import FLUSH_MAX_RETRIES
from index.cache import EmbeddingUnavailable

from .conftest import REPO, RecordingWriter


@pytest.fixture(autouse=True)
def reset():
    metrics.reset()
    yield
    metrics.reset()


class FlakyWriter(RecordingWriter):
    """Fails the first `failures` applies, then records like its parent."""

    def __init__(self, failures: int):
        super().__init__()
        self.failures = failures
        self.attempts = 0
        #: Symbol names in each *successful* apply, to tell old content from new.
        self.names: list[set[str]] = []

    def apply(self, repo_id, batch, batch_paths, vectors, epoch):
        self.attempts += 1
        if self.attempts <= self.failures:
            raise ConnectionError("Neo4j went away")
        self.names.append({s.name for f in batch.values() for s in f.symbols})
        return super().apply(repo_id, batch, batch_paths, vectors, epoch)


class DownProvider:
    name = "down-provider"
    dimensions = 1536

    def embed(self, texts):
        raise EmbeddingUnavailable("embedding API unreachable")


def test_a_failed_flush_is_requeued_and_converges(make_indexer, stub_reader, git_repo):
    writer = FlakyWriter(failures=1)
    indexer = make_indexer(writer, stub_reader, {REPO: git_repo.root}, debounce=0.01)

    async def scenario():
        indexer.on_save(REPO, "a.py", b"def a():\n    return 1\n")
        for _ in range(200):
            await asyncio.sleep(0.01)
            if writer.applied:
                break
        await indexer.aclose()

    asyncio.run(scenario())

    assert writer.attempts == 2
    assert writer.last.paths == {"a.py"}
    assert metrics.get("index.flush_failed") == 1
    assert indexer._pending[REPO] == {}


def test_a_newer_save_wins_over_the_requeued_copy(make_indexer, stub_reader, git_repo):
    """Re-queueing must not resurrect stale content over a save made since."""
    writer = FlakyWriter(failures=1)
    indexer = make_indexer(writer, stub_reader, {REPO: git_repo.root}, debounce=0.01)

    async def scenario():
        indexer.on_save(REPO, "a.py", b"def old():\n    return 1\n")
        for _ in range(200):
            await asyncio.sleep(0.005)
            if writer.attempts >= 1:
                break
        indexer.on_save(REPO, "a.py", b"def new():\n    return 2\n")
        for _ in range(400):
            await asyncio.sleep(0.01)
            if writer.applied:
                break
        await indexer.aclose()

    asyncio.run(scenario())
    assert writer.applied, "the re-queued batch never flushed"
    assert "new" in writer.names[-1] and "old" not in writer.names[-1]
    assert metrics.get("index.flush_failed") == 1


def test_join_does_not_spin_on_a_finished_task(make_indexer, recording_writer, stub_reader, git_repo):
    """Python 3.12's `gather` completes eagerly over finished tasks, without
    yielding — so a finished task whose discard-callback had not yet run kept
    `join` looping forever. Run under a watchdog: a regression must fail this
    test, not hang the suite."""
    import threading

    indexer = make_indexer(recording_writer, stub_reader, {REPO: git_repo.root})

    async def scenario():
        task = asyncio.ensure_future(asyncio.sleep(0))
        await task
        indexer._tasks.add(task)              # finished, never discarded
        await indexer.join()

    worker = threading.Thread(target=asyncio.run, args=(scenario(),), daemon=True)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive(), "Indexer.join() spun on a finished task"
    assert indexer._tasks == set()


def test_retries_stop_but_nothing_is_dropped(make_indexer, stub_reader, git_repo):
    writer = FlakyWriter(failures=10_000)
    indexer = make_indexer(writer, stub_reader, {REPO: git_repo.root}, debounce=0.001)

    async def scenario():
        indexer.on_save(REPO, "a.py", b"def a():\n    return 1\n")
        for _ in range(400):
            await asyncio.sleep(0.01)
            if metrics.get("index.flush_retries_exhausted"):
                break
        await indexer.aclose()

    asyncio.run(scenario())

    assert writer.attempts == FLUSH_MAX_RETRIES + 1
    assert metrics.get("index.flush_retries_exhausted") == 1
    assert "a.py" in indexer._pending[REPO], "the batch must still be pending"
    assert REPO not in indexer._timers, "no timer left polling a dead database"


def test_a_requeued_rename_still_resolves(make_indexer, stub_reader, git_repo):
    """Moves go back with the batch, so the retry remaps rather than recreating."""
    git_repo.write("pkg/old.py", "def helper(x):\n    return x\n")
    git_repo.commit()
    (git_repo.root / "pkg/old.py").rename(git_repo.root / "pkg/new.py")

    writer = FlakyWriter(failures=1)
    indexer = make_indexer(writer, stub_reader, {REPO: git_repo.root}, debounce=0.01)

    async def scenario():
        indexer.on_rename(REPO, "pkg/old.py", "pkg/new.py")
        for _ in range(300):
            await asyncio.sleep(0.01)
            if writer.applied:
                break
        await indexer.aclose()

    asyncio.run(scenario())

    assert writer.last.batch_paths == {"pkg/old.py", "pkg/new.py"}
    assert writer.all_remaps, "the retried flush lost its move"


def test_an_unreachable_embedding_api_degrades_instead_of_failing(
    make_indexer, recording_writer, stub_reader, git_repo, cache
):
    """§9.3: "Embedding API — fulltext + graph expansion". The symbols are
    written without vectors (the fulltext arm still finds them); the flush does
    not fail, and nothing is re-queued for a provider outage."""
    indexer = make_indexer(recording_writer, stub_reader, {REPO: git_repo.root})
    indexer.provider = DownProvider()
    cache.dimensions = 1536

    async def scenario():
        indexer.on_save(REPO, "a.py", b"def a():\n    return 1\n")
        await asyncio.sleep(0.2)
        await indexer.aclose()

    asyncio.run(scenario())

    assert recording_writer.last.paths == {"a.py"}
    assert recording_writer.last.vector_uids == set()
    assert metrics.get("index.embedding_unavailable") == 1
    assert metrics.get("index.flush_failed") == 0


def test_cached_vectors_survive_an_embedding_outage(
    make_indexer, recording_writer, stub_reader, git_repo, cache
):
    """What the cache already holds is still used: an outage costs only the
    symbols that changed."""
    indexer = make_indexer(recording_writer, stub_reader, {REPO: git_repo.root})

    async def save(source: bytes):
        indexer.on_save(REPO, "a.py", source)
        await asyncio.sleep(0.2)
        await indexer.join()

    source = b"def a():\n    return 1\n\n\ndef b():\n    return 2\n"
    asyncio.run(save(source))
    embedded = set(recording_writer.last.vector_uids)
    assert embedded

    indexer.provider = DownProvider()
    asyncio.run(save(source.replace(b"return 2", b"return 3")))

    still = recording_writer.last.vector_uids
    assert still and still < embedded, "unchanged symbols keep their cached vectors"
