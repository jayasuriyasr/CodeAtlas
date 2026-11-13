"""One embedder per repository (FINDINGS F-025).

Nothing in the graph says which model wrote a `code_vec`. Two models' vectors
share a column and no space, so a query embedded by one and compared against
documents embedded by another returns confident noise. The indexer therefore
records the model each time it re-embeds a whole repository, and refuses to add
a second model's vectors one file at a time.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

import metrics
from index.cache import EmbeddingCache, EmbeddingDisabled, EmbeddingUnavailable, embedder_key
from index.providers import DisabledEmbeddingProvider, HashEmbeddingProvider

from .conftest import REPO


@pytest.fixture(autouse=True)
def reset():
    metrics.reset()
    yield
    metrics.reset()


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    (tmp_path / "pkg" / "b.py").write_text("def b():\n    return 2\n", encoding="utf-8")
    return tmp_path


def recorded(cache) -> str | None:
    return cache.get_meta(embedder_key(REPO))


# -- the record itself ------------------------------------------------------------


def test_meta_round_trips_and_survives_reopening(tmp_path):
    path = tmp_path / "cache.sqlite"
    first = EmbeddingCache(path, model="m")
    assert first.get_meta("k") is None
    first.set_meta("k", "one")
    first.set_meta("k", "two")
    first.close()

    reopened = EmbeddingCache(path, model="m")
    assert reopened.get_meta("k") == "two"
    reopened.close()


def test_the_cache_is_safe_to_share_between_threads(tmp_path):
    """The indexer embeds on a worker thread while a request thread reads."""
    cache = EmbeddingCache(tmp_path / "cache.sqlite", model="m", dimensions=4)
    errors: list[Exception] = []

    def work(n: int) -> None:
        try:
            for i in range(60):
                cache.put_many([(f"k{n}-{i}", [0.1, 0.2, 0.3, 0.4])])
                cache.get_many([f"k{n}-{i}", f"k{(n + 1) % 4}-{i}"])
                cache.set_meta(f"m{n}", str(i))
                cache.get_meta(f"m{(n + 1) % 4}")
        except Exception as exc:                    # noqa: BLE001 - collected, asserted
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(n,)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert len(cache) == 240
    cache.close()


# -- when the indexer records it --------------------------------------------------


def test_a_first_index_records_its_embedder(make_indexer, recording_writer, stub_reader, tree, cache):
    indexer = make_indexer(recording_writer, stub_reader, {REPO: tree})
    assert recorded(cache) is None
    asyncio.run(indexer.full_reconcile(REPO))
    assert recorded(cache) == "hash-test-provider"


def test_a_diff_reconcile_does_not_claim_the_whole_repository(
    make_indexer, recording_writer, stub_reader, tree, cache
):
    """Only one file was re-embedded; the rest are whatever they were."""
    stub_reader.paths[REPO] = {"pkg/a.py": "stale-hash", "pkg/b.py": "stale-hash"}
    indexer = make_indexer(recording_writer, stub_reader, {REPO: tree})
    asyncio.run(indexer.full_reconcile(REPO))
    assert recorded(cache) is None


def test_sync_project_records_the_embedder(make_indexer, recording_writer, stub_reader, tree, cache):
    stub_reader.paths[REPO] = {"pkg/a.py": "stale-hash"}
    indexer = make_indexer(recording_writer, stub_reader, {REPO: tree})
    asyncio.run(indexer.sync_project(REPO))
    assert recorded(cache) == "hash-test-provider"


def test_an_incremental_write_never_mixes_in_a_second_model(
    make_indexer, recording_writer, stub_reader, tree, cache
):
    """A handful of vectors from another model would sit in the first model's
    index and be compared as if they belonged."""
    cache.set_meta(embedder_key(REPO), "some-other-model")
    indexer = make_indexer(recording_writer, stub_reader, {REPO: tree}, debounce=0.01)

    async def scenario():
        indexer.on_save(REPO, "pkg/a.py", b"def a():\n    return 5\n")
        await indexer.settle()
        await indexer.aclose()

    asyncio.run(scenario())
    assert recording_writer.last.paths == {"pkg/a.py"}, "the file is still indexed"
    assert recording_writer.last.vector_uids == set(), "but with no vectors"
    assert metrics.get("index.embedder_mismatch") == 1
    assert recorded(cache) == "some-other-model"


def test_a_full_re_embed_is_what_changes_the_embedder(
    make_indexer, recording_writer, stub_reader, tree, cache
):
    cache.set_meta(embedder_key(REPO), "some-other-model")
    stub_reader.paths[REPO] = {"pkg/a.py": "h", "pkg/b.py": "h"}
    indexer = make_indexer(recording_writer, stub_reader, {REPO: tree})
    asyncio.run(indexer.sync_project(REPO))

    assert recorded(cache) == "hash-test-provider"
    assert all(batch.vector_uids for batch in recording_writer.applied if batch.paths)


def test_a_degraded_re_embed_is_not_recorded(make_indexer, recording_writer, stub_reader, tree, cache):
    """If the model was unreachable, the repository was *not* re-embedded."""

    class Down:
        name = "down-model"
        dimensions = 1536

        def embed(self, texts):
            raise EmbeddingUnavailable("unreachable")

    indexer = make_indexer(recording_writer, stub_reader, {REPO: tree})
    indexer.provider = Down()
    asyncio.run(indexer.full_reconcile(REPO))
    assert recorded(cache) is None
    assert metrics.get("index.embedding_unavailable") >= 1


# -- no embedder at all ------------------------------------------------------------


def test_the_disabled_provider_raises_its_own_kind_of_unavailable():
    provider = DisabledEmbeddingProvider()
    assert provider.semantic is False and provider.name == "none"
    with pytest.raises(EmbeddingDisabled):
        provider.embed(["x"])
    assert issubclass(EmbeddingDisabled, EmbeddingUnavailable)


def test_indexing_with_no_embedder_writes_no_vectors_and_raises_no_alarm(
    make_indexer, recording_writer, stub_reader, tree
):
    """`--offline` used to write the hash stand-in's vectors — over real ones,
    if the graph had them. Now it writes none, and does not log an outage for
    something the user chose."""
    indexer = make_indexer(recording_writer, stub_reader, {REPO: tree})
    indexer.provider = DisabledEmbeddingProvider()
    asyncio.run(indexer.full_reconcile(REPO))

    assert all(batch.vector_uids == set() for batch in recording_writer.applied)
    assert metrics.get("index.embedding_unavailable") == 0


def test_the_hash_stand_in_is_marked_as_meaningless():
    assert HashEmbeddingProvider().semantic is False
