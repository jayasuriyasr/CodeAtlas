"""The local runtime end to end, against a real graph (decision 0003).

`api.cli.build_runtime` wires one repository's indexer, retriever and answering
service; `Runtime.bootstrap` brings the graph level with the tree when the
server starts. These tests use an embedder with real — if crude — semantic
structure, which is what the hash stand-in cannot be: they can tell a vector
arm that works from one that returns noise, and they exercise the rule that a
repository has exactly one embedder at a time (F-025).
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time

import pytest

import metrics
from api.app import create_app
from api.cli import build_runtime
from config import EMBED_DIM
from index.cache import embedder_key
from index.providers import DisabledEmbeddingProvider, ScriptedLLM
from index.watcher import TreeWatcher
from retrieve.router import QueryRouter
from tests.ollama_stub import bag_of_words

REPO = "repo_local_runtime"
ANSWER = "The `issue_token` function creates the token [C1]."


class FakeEmbedder:
    """Bag-of-words vectors, padded like the real local provider's."""

    semantic = True
    dimensions = EMBED_DIM

    def __init__(self, name: str = "fake-local-a", *, local: bool = True) -> None:
        self.name = name
        self.local = local
        self.calls = 0

    def embed(self, texts):
        self.calls += len(texts)
        return [bag_of_words(text, 256) + [0.0] * (EMBED_DIM - 256) for text in texts]

    def embed_query(self, text):
        return bag_of_words(text, 256) + [0.0] * (EMBED_DIM - 256)


@pytest.fixture(autouse=True)
def clean(graph_db):
    graph_db.run("MATCH (n) DETACH DELETE n")
    metrics.reset()
    yield
    graph_db.run("MATCH (n) DETACH DELETE n")


@pytest.fixture
def project(fixtures_dir, tmp_path):
    """A private copy of the Django fixture, so tests may edit it."""
    root = tmp_path / "project"
    shutil.copytree(fixtures_dir / "repos" / "django_min", root)
    return root


@pytest.fixture
def cache_path(tmp_path):
    return tmp_path / "embeddings.sqlite"


def runtime_for(graph_db, project, cache_path, provider, llm=None):
    return build_runtime(
        REPO, project, ai="off", driver=graph_db.driver, cache_path=cache_path,
        llm=llm or ScriptedLLM(ANSWER), provider=provider,
    )


def recorded(runtime) -> str | None:
    return runtime.indexer.cache.get_meta(embedder_key(REPO))


def bump(path, text: str) -> None:
    before = path.stat().st_mtime_ns if path.exists() else 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    if path.stat().st_mtime_ns == before:
        os.utime(path, ns=(before + 2_000_000, before + 2_000_000))


# -- start-up ---------------------------------------------------------------------


def test_bootstrap_indexes_a_project_it_has_never_seen(graph_db, project, cache_path):
    runtime = runtime_for(graph_db, project, cache_path, FakeEmbedder())
    try:
        assert runtime.status()["graph"] == {"files": 0, "symbols": 0, "embedded": 0,
                                             "relationships": 0}
        assert runtime.vector_search()[0] is False

        asyncio.run(runtime.bootstrap())

        graph = runtime.status()["graph"]
        assert graph["files"] == 18 and graph["symbols"] == 73
        assert graph["embedded"] == graph["symbols"], "every symbol got a vector"
        assert graph["relationships"] > 0
        assert recorded(runtime) == "fake-local-a"
        assert runtime.vector_search() == (True, "")
    finally:
        runtime.indexer.cache.close()


def test_a_second_start_only_reconciles(graph_db, project, cache_path):
    first = runtime_for(graph_db, project, cache_path, FakeEmbedder())
    asyncio.run(first.bootstrap())
    first.indexer.cache.close()

    provider = FakeEmbedder()
    second = runtime_for(graph_db, project, cache_path, provider)
    try:
        asyncio.run(second.bootstrap())
        assert provider.calls == 0, "nothing changed, so nothing was embedded again"
        assert second.vector_search()[0] is True
    finally:
        second.indexer.cache.close()


def test_changes_made_while_nothing_was_watching_are_caught_at_start(
    graph_db, project, cache_path
):
    first = runtime_for(graph_db, project, cache_path, FakeEmbedder())
    asyncio.run(first.bootstrap())
    first.indexer.cache.close()

    bump(project / "billing" / "refunds.py", "def refund_card(user):\n    return True\n")
    (project / "common" / "mixins.py").unlink()

    second = runtime_for(graph_db, project, cache_path, FakeEmbedder())
    try:
        asyncio.run(second.bootstrap())
        names = {r["qn"] for r in graph_db.run(
            "MATCH (s:Symbol {repo_id: $r}) RETURN s.qualified_name AS qn", r=REPO)}
        assert "billing.refunds.refund_card" in names
        assert not any(n.startswith("common.mixins") for n in names)
    finally:
        second.indexer.cache.close()


# -- the vector arm ---------------------------------------------------------------


def test_the_vector_arm_contributes_real_candidates(graph_db, project, cache_path):
    runtime = runtime_for(graph_db, project, cache_path, FakeEmbedder())
    try:
        asyncio.run(runtime.bootstrap())
        question = "explain how api tokens are issued"
        routed = QueryRouter().route(question)
        _seeds, _neighbors, notes = runtime.retriever(question, routed)
        assert notes["vector_search"] == "on"

        candidates = runtime.retriever.hybrid.search(
            REPO, question, query_vec=runtime.embed_query(question))
        assert any(c.vector_rank is not None for c in candidates)
    finally:
        runtime.indexer.cache.close()


def test_a_different_embedder_may_not_search_this_index(graph_db, project, cache_path):
    """Two models' vectors share a column and no space. Comparing across them
    returns ranked results that mean nothing — so the arm is off, and says why."""
    first = runtime_for(graph_db, project, cache_path, FakeEmbedder("fake-local-a"))
    asyncio.run(first.bootstrap())
    first.indexer.cache.close()

    second = runtime_for(graph_db, project, cache_path, FakeEmbedder("fake-local-b"))
    try:
        usable, why = second.vector_search()
        assert usable is False
        assert "fake-local-a" in why and "fake-local-b" in why

        question = "explain how api tokens are issued"
        _s, _n, notes = second.retriever(question, QueryRouter().route(question))
        assert notes["vector_search"] == "degraded"
        assert "fake-local-a" in notes["vector_search_note"]
        assert second.status()["vector_search"] is False
    finally:
        second.indexer.cache.close()


def test_a_new_local_embedder_re_embeds_at_start(graph_db, project, cache_path):
    """Local models are free to run, so the switch is made for the user."""
    first = runtime_for(graph_db, project, cache_path, FakeEmbedder("fake-local-a"))
    asyncio.run(first.bootstrap())
    first.indexer.cache.close()

    provider = FakeEmbedder("fake-local-b")
    second = runtime_for(graph_db, project, cache_path, provider)
    try:
        asyncio.run(second.bootstrap())
        assert provider.calls == 73, "every symbol was re-embedded"
        assert recorded(second) == "fake-local-b"
        assert second.vector_search()[0] is True
    finally:
        second.indexer.cache.close()


def test_a_new_billed_embedder_is_never_run_without_being_asked(graph_db, project, cache_path):
    """Re-embedding a repository with a hosted model costs money."""
    first = runtime_for(graph_db, project, cache_path, FakeEmbedder("fake-local-a"))
    asyncio.run(first.bootstrap())
    first.indexer.cache.close()

    provider = FakeEmbedder("fake-hosted", local=False)
    second = runtime_for(graph_db, project, cache_path, provider)
    try:
        asyncio.run(second.bootstrap())
        assert provider.calls == 0
        assert recorded(second) == "fake-local-a"
        assert second.vector_search()[0] is False
    finally:
        second.indexer.cache.close()


def test_running_with_no_embedder_leaves_existing_vectors_alone(graph_db, project, cache_path):
    """`--offline` used to overwrite real vectors with the hash stand-in's."""
    first = runtime_for(graph_db, project, cache_path, FakeEmbedder())
    asyncio.run(first.bootstrap())
    before = first.status()["graph"]["embedded"]
    first.indexer.cache.close()

    second = runtime_for(graph_db, project, cache_path, DisabledEmbeddingProvider())
    try:
        asyncio.run(second.indexer.full_reconcile(REPO, reparse_all=True))
        assert second.status()["graph"]["embedded"] == before == 73
        assert recorded(second) == "fake-local-a", "the record still names who wrote them"
        assert second.vector_search()[0] is False
    finally:
        second.indexer.cache.close()


# -- the whole server ---------------------------------------------------------------


def test_the_server_indexes_at_start_and_follows_edits(graph_db, project, cache_path):
    """`--serve --watch`, in one process: start on an empty graph, answer a
    question, then see a file saved on disk show up in the graph by itself."""
    from fastapi.testclient import TestClient

    runtime = runtime_for(graph_db, project, cache_path, FakeEmbedder())
    watcher = TreeWatcher(
        indexer=runtime.indexer, repo=REPO, root=project,
        extensions=tuple(runtime.indexer.adapters), interval=0.05,
    )
    runtime.indexer.debounce = 0.05
    app = create_app(runtime.service, runtime.indexer, repos=[REPO],
                     watcher=watcher, bootstrap=runtime.bootstrap, status=runtime.status)

    def wait_for(predicate, seconds=30.0):
        deadline = time.time() + seconds
        while time.time() < deadline:
            payload = client.get("/status").json()
            if predicate(payload):
                return payload
            time.sleep(0.1)
        raise AssertionError(f"timed out; last status: {payload}")

    try:
        with TestClient(app) as client:
            ready = wait_for(lambda s: s["watcher"]["state"] == "watching")
            assert ready["graph"]["symbols"] == 73 and ready["vector_search"] is True
            assert ready["embeddings"] == "fake-local-a"

            body = client.post("/ask", json={"question": "how are API tokens issued"}).text
            assert '"type": "context"' in body and '"type": "done"' in body

            bump(project / "billing" / "refunds.py",
                 "def refund_card(user):\n    return True\n")
            grown = wait_for(lambda s: s["graph"]["symbols"] == 75)
            assert grown["graph"]["files"] == 19
            assert grown["watcher"]["last_change_at"] is not None
    finally:
        runtime.indexer.cache.close()
