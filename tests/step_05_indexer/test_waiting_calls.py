"""A call written before its target exists is linked once the target appears.

Reproduced first: save `pkg/caller.py` (which calls `pkg.helpers.helper`) while
`pkg/helpers.py` does not exist yet, then save `pkg/helpers.py`. The first write
drops the edge — nothing to point at — and nothing ever re-resolves it, because
the re-link after a save only covers callers of symbols whose UID changed. So
"what calls `helper`" stays empty until Sync Project.
"""

from __future__ import annotations

import asyncio

import pytest

import metrics
from graph.reader import GraphReader
from graph.writer import GraphWriter

from .conftest import REPO

CALLER = "from pkg.helpers import helper\n\n\ndef run(x):\n    return helper(x)\n"
HELPERS = "def helper(value):\n    return value * 2\n"


@pytest.fixture(autouse=True)
def clean(graph_db):
    graph_db.run("MATCH (n) DETACH DELETE n")
    metrics.reset()
    yield
    graph_db.run("MATCH (n) DETACH DELETE n")


def callers_of(graph_db, qualified_name: str) -> set[str]:
    rows = graph_db.run(
        "MATCH (c:Symbol {repo_id: $r})-[:CALLS]->(:Symbol {repo_id: $r, qualified_name: $qn}) "
        "RETURN DISTINCT c.qualified_name AS qn", r=REPO, qn=qualified_name)
    return {r["qn"] for r in rows}


@pytest.fixture
def indexer(make_indexer, graph_db, tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    return make_indexer(GraphWriter(graph_db.driver, graph_db.name),
                        GraphReader(graph_db.driver, graph_db.name),
                        {REPO: tmp_path}, debounce=0.02)


def save(indexer, root, rel_path: str, text: str):
    (root / rel_path).write_text(text, encoding="utf-8", newline="\n")
    indexer.on_save(REPO, rel_path, text.encode())


def test_a_call_saved_before_its_target_is_linked_when_the_target_arrives(indexer, graph_db, tmp_path):
    async def scenario():
        save(indexer, tmp_path, "pkg/caller.py", CALLER)
        await indexer.settle()
        assert callers_of(graph_db, "pkg.helpers.helper") == set()   # nothing to point at yet
        save(indexer, tmp_path, "pkg/helpers.py", HELPERS)
        await indexer.settle()
        await indexer.aclose()

    asyncio.run(scenario())
    assert callers_of(graph_db, "pkg.helpers.helper") == {"pkg.caller.run"}


def test_a_waiting_call_to_a_library_triggers_nothing(indexer, graph_db, tmp_path):
    """`requests.get` never appears in the repository; recording it must not
    make every later save re-write the caller's file."""
    async def scenario():
        save(indexer, tmp_path, "pkg/net.py", "import requests\n\n\ndef fetch(u):\n    return requests.get(u)\n")
        await indexer.settle()
        save(indexer, tmp_path, "pkg/helpers.py", HELPERS)
        await indexer.settle()
        await indexer.aclose()

    asyncio.run(scenario())
    assert metrics.get("index.relinked_files") == 0


def test_the_waiting_record_is_replaced_when_the_caller_changes(indexer, graph_db, tmp_path):
    """Once `run` no longer calls `helper`, adding `helper` re-links nothing."""
    async def scenario():
        save(indexer, tmp_path, "pkg/caller.py", CALLER)
        await indexer.settle()
        save(indexer, tmp_path, "pkg/caller.py", "def run(x):\n    return x\n")
        await indexer.settle()
        save(indexer, tmp_path, "pkg/helpers.py", HELPERS)
        await indexer.settle()
        await indexer.aclose()

    asyncio.run(scenario())
    assert callers_of(graph_db, "pkg.helpers.helper") == set()
    assert metrics.get("index.relinked_files") == 0
