"""Bounded multi-hop: "what eventually calls X", "what is affected if I change X".

Before: "what eventually calls X" was answered with X's direct callers (one
hop), and "what is affected if I change X" was not recognised at all. The
`impact` rule walks callers transitively — with dispatch through overrides at
every hop — to `MULTIHOP_MAX_DEPTH`, returns the call path for every result,
and stops at `MULTIHOP_MAX_RESULTS`, saying so instead of silently cutting.
"""

from __future__ import annotations

import pytest

import config
from adapters.python import PythonAdapter
from graph.writer import GraphWriter
from retrieve.router import QueryClass, QueryRouter
from retrieve.structural import StructuralResolver

REPO = "repo_multihop"


class Runner:
    def __init__(self, graph_db):
        self.db = graph_db

    def run(self, query, **params):
        return [dict(r) for r in self.db.run(query, **params)]


@pytest.fixture(autouse=True)
def clean(graph_db):
    graph_db.run("MATCH (n) DETACH DELETE n")
    yield
    graph_db.run("MATCH (n) DETACH DELETE n")


def index(graph_db, source: bytes, rel_path: str = "pkg/chain.py"):
    import time

    parsed = PythonAdapter().parse(REPO, rel_path, source)
    GraphWriter(graph_db.driver, graph_db.name).apply(
        REPO, {rel_path: parsed}, [rel_path], {}, time.time_ns())


def chain(n: int) -> bytes:
    """f0 <- f1 <- … <- f{n-1}: each calls the one before it."""
    lines = ["def f0():\n    return 0\n"]
    lines += [f"def f{i}():\n    return f{i - 1}()\n" for i in range(1, n)]
    return "\n\n".join(lines).encode()


def ask(graph_db, question: str, **limits):
    routed = QueryRouter().route(question)
    resolver = StructuralResolver(runner=Runner(graph_db), **limits)
    return routed, resolver.answer(REPO, question, routed.rule)


def names(graph_db, uids):
    rows = graph_db.run("MATCH (s:Symbol {repo_id: $r}) RETURN s.uid AS uid, s.qualified_name AS qn", r=REPO)
    by_uid = {r["uid"]: r["qn"] for r in rows}
    return [by_uid[u] for u in uids]


@pytest.mark.parametrize("question", [
    "what eventually calls f0", "what transitively calls `f0`",
    "what is affected if I change f0", "what breaks if I change `f0`",
])
def test_the_router_recognises_multi_hop_questions(question):
    routed = QueryRouter().route(question)
    assert routed.query_class is QueryClass.STRUCTURAL and routed.rule == "impact"


def test_a_plain_callers_question_is_still_one_hop():
    assert QueryRouter().route("what calls f0").rule == "callers"


def test_callers_are_found_transitively_with_their_paths(graph_db):
    index(graph_db, chain(4))
    _r, answer = ask(graph_db, "what eventually calls `f0`")
    assert names(graph_db, answer.related_uids) == ["pkg.chain.f1", "pkg.chain.f2", "pkg.chain.f3"]
    paths = {names(graph_db, [u])[0]: p for u, p in answer.paths.items()}
    assert paths["pkg.chain.f3"] == ["pkg.chain.f3", "pkg.chain.f2", "pkg.chain.f1", "pkg.chain.f0"]
    assert answer.truncated is False


def test_the_walk_stops_at_the_depth_limit(graph_db):
    index(graph_db, chain(config.MULTIHOP_MAX_DEPTH + 3))
    _r, answer = ask(graph_db, "what eventually calls `f0`")
    found = names(graph_db, answer.related_uids)
    assert len(found) == config.MULTIHOP_MAX_DEPTH
    assert f"pkg.chain.f{config.MULTIHOP_MAX_DEPTH + 1}" not in found
    assert answer.note()["depth_limit"] == config.MULTIHOP_MAX_DEPTH


def test_hitting_the_result_cap_is_said_not_hidden(graph_db):
    many = "def f0():\n    return 0\n\n\n" + "\n\n".join(
        f"def c{i}():\n    return f0()\n" for i in range(12))
    index(graph_db, many.encode())
    _r, answer = ask(graph_db, "what is affected if I change `f0`", max_related=5)
    assert len(answer.related_uids) == 5
    assert answer.truncated is True
    note = answer.note()
    assert note["truncated"] is True and note["related_shown"] == 5


def test_a_cycle_ends(graph_db):
    index(graph_db, (b"def ping():\n    return pong()\n\n\ndef pong():\n    return ping()\n\n\n"
                     b"def start():\n    return ping()\n"))
    _r, answer = ask(graph_db, "what eventually calls `ping`")
    assert sorted(names(graph_db, answer.related_uids)) == ["pkg.chain.pong", "pkg.chain.start"]


def test_dispatch_is_followed_at_every_hop(graph_db):
    index(graph_db, (
        b"class Base:\n    def run(self):\n        return self.hook()\n\n"
        b"    def hook(self):\n        return 0\n\n\n"
        b"class Sub(Base):\n    def hook(self):\n        return 1\n\n\n"
        b"def main():\n    worker = Base()\n    return worker.run()\n"))
    _r, answer = ask(graph_db, "what eventually calls `Sub.hook`")
    found = names(graph_db, answer.related_uids)
    assert found[:1] == ["pkg.chain.Base.run"] and "pkg.chain.main" in found
