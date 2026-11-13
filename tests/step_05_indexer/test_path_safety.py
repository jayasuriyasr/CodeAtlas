"""A path from outside the indexer never reaches a file outside the repository.

`POST /index/rename` handed `new_path` to `Indexer._read`, which did
`root / rel_path`. On Windows and POSIX alike, joining an absolute path
replaces the root, and `..` walks out of it — so one request could pull any
`.py` or `.ts` file on the machine into the index, and from there into a
prompt.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from index.indexer import UnsafePath, safe_rel_path

from .conftest import REPO


@pytest.fixture
def layout(tmp_path):
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "ok.py").write_text("def fine():\n    return 1\n", encoding="utf-8")
    secret = tmp_path / "outside" / "secret.py"
    secret.parent.mkdir()
    secret.write_text('def leaked():\n    return "do not index me"\n', encoding="utf-8")
    return root, secret


@pytest.mark.parametrize(
    "rel_path",
    [
        "../outside/secret.py",
        "pkg/../../outside/secret.py",
        "..\\outside\\secret.py",
        "/etc/passwd.py",
        "C:/Windows/win.py",
        "C:\\Windows\\win.py",
        "\\\\server\\share\\x.py",
        "",
    ],
)
def test_a_path_that_leaves_the_root_is_refused(layout, rel_path):
    root, _ = layout
    with pytest.raises(UnsafePath):
        safe_rel_path(root, rel_path)


@pytest.mark.parametrize("rel_path, expected", [
    ("pkg/ok.py", "pkg/ok.py"),
    ("pkg\\ok.py", "pkg/ok.py"),
    ("./pkg/ok.py", "pkg/ok.py"),
    ("pkg/new_file.py", "pkg/new_file.py"),
])
def test_a_path_inside_the_root_is_normalised(layout, rel_path, expected):
    root, _ = layout
    assert safe_rel_path(root, rel_path) == expected


def test_a_symlink_out_of_the_root_is_refused(layout):
    root, secret = layout
    link = root / "pkg" / "link.py"
    try:
        os.symlink(secret, link)
    except (OSError, NotImplementedError):
        pytest.skip("creating a symlink needs privileges this machine does not grant")
    with pytest.raises(UnsafePath):
        safe_rel_path(root, "pkg/link.py")


def test_a_rename_to_an_outside_path_is_refused_and_indexes_nothing(
    make_indexer, recording_writer, stub_reader, layout
):
    root, _ = layout
    indexer = make_indexer(recording_writer, stub_reader, {REPO: root}, debounce=0.01)

    async def scenario():
        with pytest.raises(UnsafePath):
            indexer.on_rename(REPO, "pkg/ok.py", "../outside/secret.py")
        await indexer.settle()
        await indexer.aclose()

    asyncio.run(scenario())
    written = [path for batch in recording_writer.applied for path in batch.paths]
    assert not any("secret" in path for path in written)


def test_the_http_route_rejects_it(make_indexer, recording_writer, stub_reader, layout):
    from fastapi.testclient import TestClient

    from api.app import InvestigationService, create_app
    from index.providers import ApproxCodeTokenizer, ScriptedLLM

    root, _ = layout
    indexer = make_indexer(recording_writer, stub_reader, {REPO: root}, debounce=0.01)
    service = InvestigationService(retriever=lambda q, r: ([], []), llm=ScriptedLLM("x"),
                                   tokenizer=ApproxCodeTokenizer())
    with TestClient(create_app(service, indexer, repos=[REPO])) as client:
        reply = client.post("/index/rename", json={
            "repo": REPO, "old_path": "pkg/ok.py", "new_path": "../outside/secret.py"})
    assert reply.status_code == 400
    assert reply.json()["accepted"] is False and "outside" in reply.json()["reason"]
