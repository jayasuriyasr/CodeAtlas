"""The FastAPI surface against a *real* `Indexer`, not a spy.

`test_pipeline.py` posts renames to a spy object. With the real indexer the
route failed: `on_rename` arms §4.1's debounce timer on the running event loop,
and a plain `def` route runs in FastAPI's thread pool, where there is none —
`RuntimeError: no running event loop` on every rename from a real editor.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from api.app import InvestigationService, create_app, sse
from api.stream import SentenceEvent
from index.cache import EmbeddingCache
from index.indexer import Indexer
from index.providers import ApproxCodeTokenizer, HashEmbeddingProvider, ScriptedLLM

from .test_pipeline_hardening import GOOD, block

REPO = "repo_http"


class Writer:
    def __init__(self):
        self.applied = []

    def apply(self, repo_id, batch, batch_paths, vectors, epoch):
        self.applied.append((repo_id, set(batch), set(batch_paths)))

    def remap_uids(self, repo_id, remaps, epoch):
        return len(remaps)

    def refresh_all_degrees(self, repo_id):
        pass


class Reader:
    def __init__(self):
        self.startups = 0

    def known_paths(self, repo_id):
        self.startups += 1
        return set()

    def file_hashes(self, repo_id):
        return {}

    def body_hashes(self, repo_id, uids):
        return {}

    def symbol_identities(self, repo_id, paths):
        return {}


@pytest.fixture
def indexer(tmp_path):
    provider = HashEmbeddingProvider()
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "new.py").write_text("def moved():\n    return 1\n", encoding="utf-8")
    idx = Indexer(
        writer=Writer(), reader=Reader(),
        cache=EmbeddingCache(tmp_path / "c.sqlite", model=provider.name),
        provider=provider, tokenizer=ApproxCodeTokenizer(),
        roots={REPO: tmp_path}, debounce=0.05,
    )
    yield idx
    idx.cache.close()


def service():
    return InvestigationService(
        retriever=lambda q, r: ([block("charge_card", 1.0)], []),
        llm=ScriptedLLM(GOOD), tokenizer=ApproxCodeTokenizer(), repo_id=REPO,
    )


def test_rename_reaches_a_real_indexer_and_flushes(indexer):
    from fastapi.testclient import TestClient

    app = create_app(service(), indexer=indexer, repos=[REPO])
    with TestClient(app) as client:
        response = client.post(
            "/index/rename", json={"repo": REPO, "old_path": "pkg/old.py", "new_path": "pkg/new.py"}
        )
        assert response.status_code == 200 and response.json() == {"accepted": True}

        async def wait_for_flush():
            for _ in range(100):
                if indexer.writer.applied:
                    return
                await asyncio.sleep(0.02)

        client.portal.call(wait_for_flush)

    assert indexer.writer.applied, "the debounce timer never fired"
    _repo, paths, batch_paths = indexer.writer.applied[-1]
    assert paths == {"pkg/new.py"} and batch_paths == {"pkg/old.py", "pkg/new.py"}


def test_startup_seeds_indexed_paths(indexer):
    from fastapi.testclient import TestClient

    with TestClient(create_app(service(), indexer=indexer, repos=[REPO])):
        pass
    assert indexer.reader.startups == 1


def test_rename_for_an_unknown_repo_is_refused(indexer):
    from fastapi.testclient import TestClient

    with TestClient(create_app(service(), indexer=indexer, repos=[REPO])) as client:
        response = client.post(
            "/index/rename", json={"repo": "nope", "old_path": "a.py", "new_path": "b.py"}
        )
    assert response.json()["accepted"] is False
    assert "nope" not in indexer._renames


def test_sync_endpoint_runs_sync_project(indexer):
    from fastapi.testclient import TestClient

    with TestClient(create_app(service(), indexer=indexer, repos=[REPO])) as client:
        response = client.post("/index/sync", json={"repo": REPO})
        unknown = client.post("/index/sync", json={"repo": "nope"})
    assert response.json() == {"accepted": True}
    assert unknown.json()["accepted"] is False
    assert indexer.writer.applied, "sync wrote nothing"


def test_sync_without_an_indexer_degrades():
    from fastapi.testclient import TestClient

    with TestClient(create_app(service())) as client:
        assert client.post("/index/sync", json={"repo": REPO}).json()["accepted"] is False


def test_sse_frames_carry_retrieval_retry_and_amendments():
    amended = sse(SentenceEvent(text="x. [C1]", is_claim=True, cited=True, handles=(1,),
                                valid_handles=True, reason="handle", amends=True))
    assert json.loads(amended[len("data: "):])["amends"] is True

    notes = sse({"type": "retrieval", "route": "TRACE", "frames": []})
    assert json.loads(notes[len("data: "):])["route"] == "TRACE"


def test_ask_endpoint_streams_the_done_payload():
    from fastapi.testclient import TestClient

    with TestClient(create_app(service())) as client:
        body = client.post("/ask", json={"question": "what charges the card"}).text
    frames = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
    done = frames[-1]
    assert done["type"] == "done"
    assert {"report", "context", "retried", "needs_review"} <= set(done)


def test_an_empty_question_is_rejected_by_validation():
    from fastapi.testclient import TestClient

    with TestClient(create_app(service())) as client:
        assert client.post("/ask", json={}).status_code == 422

