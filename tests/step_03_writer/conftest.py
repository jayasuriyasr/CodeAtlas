"""Shared scaffolding for the step 3 writer tests.

Builders live here rather than in one test module so the two modules do not
import fixtures from each other — a cross-module fixture import works until
someone runs a single file, and then fails for a reason unrelated to the code
under test.
"""

from __future__ import annotations

import time

import pytest

from adapters.base import Edge, ParsedFile, Symbol, symbol_uid
from graph.reader import GraphReader
from graph.writer import GraphWriter

REPO = "repo_step3"
OTHER_REPO = "repo_step3_other"


def epoch() -> int:
    """One epoch per batch (§4.3 step 1). Nanoseconds, so two calls differ."""
    return time.time_ns()


def make_symbol(
    repo_id: str,
    rel_path: str,
    qualified_name: str,
    *,
    arity: int = 0,
    ordinal: int = 0,
    **kw,
) -> Symbol:
    name = qualified_name.rsplit(".", 1)[-1]
    return Symbol(
        uid=symbol_uid(repo_id, rel_path, qualified_name, arity, ordinal),
        repo_id=repo_id,
        rel_path=rel_path,
        qualified_name=qualified_name,
        name=name,
        arity=arity,
        ordinal=ordinal,
        kind=kw.pop("kind", "function"),
        signature=kw.pop("signature", f"def {name}():"),
        docstring=kw.pop("docstring", None),
        source_code=kw.pop("source_code", f"def {name}():\n    pass\n"),
        enclosing_signature=kw.pop("enclosing_signature", None),
        start_line=kw.pop("start_line", 1),
        end_line=kw.pop("end_line", 2),
        **kw,
    )


def make_file(
    rel_path: str,
    symbols: list[Symbol],
    edges: list[Edge] | None = None,
    content_hash: str = "h0",
) -> ParsedFile:
    return ParsedFile(
        rel_path=rel_path,
        content_hash=content_hash,
        symbols=symbols,
        edges=edges or [],
        language="python",
    )


@pytest.fixture
def writer(graph_db) -> GraphWriter:
    return GraphWriter(graph_db.driver, graph_db.name)


@pytest.fixture
def reader(graph_db) -> GraphReader:
    return GraphReader(graph_db.driver, graph_db.name)


@pytest.fixture(autouse=True)
def clean(graph_db):
    """Each test owns the graph; the module owns the database (F-001)."""
    graph_db.run("MATCH (n) DETACH DELETE n")
    yield
    graph_db.run("MATCH (n) DETACH DELETE n")


@pytest.fixture
def two_file_batch(writer):
    """`a.py` defines `helper`; `b.py` holds five callers of it.

    Five is not arbitrary — §8.2's move-survival fixture (a) specifies "a file
    with >=5 known inbound callers", and step 5 reuses this exact shape.
    """
    helper = make_symbol(REPO, "a.py", "a.helper")
    callers = [make_symbol(REPO, "b.py", f"b.caller{i}") for i in range(5)]
    edges = [
        Edge(source_uid=c.uid, kind="CALLS", origin_path="b.py", target_uid=helper.uid)
        for c in callers
    ]
    batch = {
        "a.py": make_file("a.py", [helper], content_hash="ha"),
        "b.py": make_file("b.py", callers, edges, content_hash="hb"),
    }
    return batch, helper, callers
