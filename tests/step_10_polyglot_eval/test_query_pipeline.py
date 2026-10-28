"""§5.1's three paths, end to end over both fixtures in a real graph.

Until the final audit the router's decision was computed and ignored: every
question took the SEMANTIC path, the frame resolver and the STRUCTURAL
traversal were reachable only from their own unit tests, and the CLI indexed
Python alone. This module indexes the Django *and* Next.js fixtures through the
real `Indexer` with both adapters, then asks each kind of question through the
production `QueryRetriever` — and, at the end, through the CLI itself.
"""

from __future__ import annotations

import argparse
import asyncio
import io
from contextlib import redirect_stderr, redirect_stdout

import pytest

import metrics
from adapters.python import PythonAdapter
from adapters.typescript import TypeScriptAdapter
from api.app import InvestigationService
from graph.reader import GraphReader
from graph.writer import GraphWriter
from index.cache import EmbeddingCache
from index.indexer import Indexer
from index.providers import ApproxCodeTokenizer, HashEmbeddingProvider, ScriptedLLM
from retrieve.expand import NeighborhoodExpander
from retrieve.hybrid import HybridRetriever, Neo4jSearchBackend
from retrieve.pipeline import QueryRetriever
from retrieve.router import QueryRouter
from retrieve.structural import StructuralResolver, entity_terms

PY_REPO = "repo_pipeline_py"
TS_REPO = "repo_pipeline_ts"


class Runner:
    def __init__(self, graph_db):
        self._db = graph_db

    def run(self, query, **params):
        return [dict(r) for r in self._db.run(query, **params)]


def _adapters():
    py, ts = PythonAdapter(), TypeScriptAdapter()
    return {**{e: py for e in py.extensions}, **{e: ts for e in ts.extensions}}


@pytest.fixture(scope="module")
def indexed(graph_db, fixtures_dir, tmp_path_factory):
    graph_db.run("MATCH (n) DETACH DELETE n")
    provider = HashEmbeddingProvider()
    cache = EmbeddingCache(tmp_path_factory.mktemp("q") / "c.sqlite", model=provider.name)
    for repo, name in ((PY_REPO, "django_min"), (TS_REPO, "nextjs_min")):
        indexer = Indexer(
            writer=GraphWriter(graph_db.driver, graph_db.name),
            reader=GraphReader(graph_db.driver, graph_db.name),
            cache=cache, provider=provider, tokenizer=ApproxCodeTokenizer(),
            roots={repo: fixtures_dir / "repos" / name}, adapters=_adapters(),
        )
        asyncio.run(indexer.full_reconcile(repo))
    yield graph_db
    cache.close()
    graph_db.run("MATCH (n) DETACH DELETE n")


@pytest.fixture(autouse=True)
def reset():
    metrics.reset()
    yield
    metrics.reset()


def retriever(graph_db, repo, root=None) -> QueryRetriever:
    runner = Runner(graph_db)
    return QueryRetriever(
        repo_id=repo,
        reader=GraphReader(graph_db.driver, graph_db.name),
        hybrid=HybridRetriever(backend=Neo4jSearchBackend(graph_db.driver, graph_db.name)),
        expander=NeighborhoodExpander(runner=runner),
        structural=StructuralResolver(runner=runner),
        repo_root=root,
    )


def ask(graph_db, repo, question):
    routed = QueryRouter().route(question)
    seeds, neighbors, notes = retriever(graph_db, repo)(question, routed)
    return routed, seeds, neighbors, notes


def names(blocks):
    return [b.symbol.qualified_name for b in blocks]


# -- entity extraction ------------------------------------------------------


@pytest.mark.parametrize(
    "question, first",
    [
        ("what calls `charge_card`", "charge_card"),
        ("who calls billing.tasks.charge_card", "billing.tasks.charge_card"),
        ("what calls charge_card?", "charge_card"),
        ("callers of SubscriptionView.post", "SubscriptionView.post"),
        ("what calls login", "login"),
        ("where is getUserById used", "getUserById"),
    ],
)
def test_entity_terms(question, first):
    assert entity_terms(question)[0] == first


def test_entity_terms_ignore_question_words():
    assert entity_terms("what calls the function") == []


# -- STRUCTURAL ---------------------------------------------------------------


def test_what_calls_is_answered_by_traversal(indexed):
    routed, seeds, neighbors, notes = ask(indexed, PY_REPO, "what calls `charge_card`")

    assert notes["route_taken"] == "STRUCTURAL"
    assert names(seeds)[0] == "billing.tasks.charge_card"
    assert set(names(seeds)[1:] + names(neighbors)) == {
        "billing.tasks.run_monthly_billing",
        "billing.tasks.retry_failed",
        "billing.views.SubscriptionView.post",
    }
    assert notes["structural"]["related_total"] == 3


def test_what_does_x_call(indexed):
    _r, seeds, neighbors, notes = ask(indexed, PY_REPO, "what does `retry_failed` call")
    assert notes["structural"]["rule"] == "callees"
    assert "billing.tasks.charge_card" in names(seeds) + names(neighbors)


def test_usages_include_imports_as_well_as_calls(indexed):
    _r, seeds, neighbors, notes = ask(indexed, PY_REPO, "where is `charge_card` used")
    related = set(names(seeds) + names(neighbors))
    assert "billing.views" in related, "billing/views.py imports charge_card"
    assert "billing.tasks.retry_failed" in related


def test_an_unknown_entity_falls_back_to_semantic_and_says_so(indexed):
    _r, _seeds, _n, notes = ask(indexed, PY_REPO, "what calls `no_such_function_anywhere`")
    assert notes["route"] == "STRUCTURAL" and notes["route_taken"] == "SEMANTIC"
    assert "no symbol" in notes["structural"]["fallback"]


def test_subclasses_has_no_edge_and_falls_back_with_the_entity_first(indexed):
    _r, seeds, _n, notes = ask(indexed, PY_REPO, "subclasses of `TimestampMixin`")
    assert notes["route_taken"] == "SEMANTIC"
    assert "no edge type" in notes["structural"]["fallback"]
    assert names(seeds)[0] == "common.mixins.TimestampMixin"


def test_a_name_shared_by_many_symbols_is_refused(indexed, graph_db):
    resolver = StructuralResolver(runner=Runner(indexed), max_entities=1)
    answer = resolver.answer(TS_REPO, "what calls `GET`", "callers")
    assert answer.fallback and "refusing" in answer.fallback
    assert metrics.get("structural.ambiguous") == 1


def test_structural_is_scoped_to_the_repo(indexed):
    _r, seeds, neighbors, notes = ask(indexed, TS_REPO, "what calls `charge_card`")
    assert "billing.tasks.charge_card" not in names(seeds) + names(neighbors)


# -- TRACE --------------------------------------------------------------------


PY_TRACE = (
    "Traceback (most recent call last):\n"
    '  File "/srv/app/.venv/lib/python3.12/site-packages/celery/app/trace.py", line 451, in trace_task\n'
    "    R = retval = fun(*args, **kwargs)\n"
    '  File "/srv/app/billing/tasks.py", line 31, in retry_failed\n'
    "    return charge_card(user, amount_cents, idempotency_key=idempotency_key)\n"
    '  File "/srv/app/billing/tasks.py", line 14, in charge_card\n'
    '    audit_event("charge_card", user_id=user.pk, amount=amount_cents)\n'
    "AttributeError: 'NoneType' object has no attribute 'pk'\n"
)


def test_a_python_trace_from_a_container_resolves_directly(indexed):
    routed, seeds, neighbors, notes = ask(indexed, PY_REPO, PY_TRACE)

    assert routed.query_class.value == "TRACE"
    tiers = [f["tier"] for f in notes["frames"]]
    assert tiers == ["unresolved", "direct", "direct"], notes["frames"]
    assert names(seeds)[:2] == ["billing.tasks.charge_card", "billing.tasks.retry_failed"], (
        "innermost frame first"
    )
    assert "tier3" not in notes, "only a library frame failed; Tier 3 is not needed"
    assert neighbors, "§5.1 TRACE: resolve, then expand"


JS_TRACE = (
    "TypeError: users.map is not a function\n"
    "    at GET (/app/.next/server/chunks/9002.js:1:120)\n"
    "    at async handler (/app/node_modules/next/dist/server/route.js:12:9)"
)


def test_an_ambiguous_ts_frame_refuses_and_falls_to_tier3(indexed):
    _routed, seeds, _n, notes = ask(indexed, TS_REPO, JS_TRACE)

    get = next(f for f in notes["frames"] if "GET" in f["frame"])
    assert get["tier"] == "unresolved" and get["badge"] == "? unresolved"
    assert metrics.get("trace.tier2.ambiguous") == 1
    assert "users.map is not a function" in notes["tier3"]
    assert seeds, "Tier 3 still retrieves something to answer from"


# -- SEMANTIC -----------------------------------------------------------------


def test_semantic_retrieves_and_expands(indexed):
    _r, seeds, neighbors, notes = ask(indexed, PY_REPO, "how does monthly billing charge subscriptions")
    assert notes["route"] == "SEMANTIC" and notes["vector_search"] == "off"
    assert "billing.tasks.run_monthly_billing" in names(seeds)
    assert neighbors


def test_a_huge_pasted_question_does_not_break_fulltext(indexed):
    """Lucene's 1,024-clause limit used to fail the whole fulltext arm."""
    question = "explain " + " ".join(f"word{i}" for i in range(5000))
    _r, _seeds, _n, notes = ask(indexed, PY_REPO, question)
    assert notes["route"] == "SEMANTIC"


def test_typescript_symbols_are_retrievable(indexed):
    _r, seeds, _n, _notes = ask(indexed, TS_REPO, "explain the UserCard component")
    assert any("UserCard" in n for n in names(seeds))


# -- the whole answer path ----------------------------------------------------


def test_a_structural_question_answered_end_to_end(indexed):
    qr = retriever(indexed, PY_REPO)
    answer = (
        "`charge_card` is called by `run_monthly_billing` [C2]. "
        "It is also called by `retry_failed` [C3]."
    )
    service = InvestigationService(
        retriever=qr, llm=ScriptedLLM(answer), tokenizer=ApproxCodeTokenizer(),
        graph_runner=Runner(indexed), repo_id=PY_REPO, expander=qr.expand,
    )
    result = service.answer("what calls `charge_card`")

    assert result.notes["route_taken"] == "STRUCTURAL"
    assert result.report.handle_errors == []
    assert result.report.relation_verdicts, "T3 checked the asserted CALLS edges"
    assert all(v.verified for v in result.report.relation_verdicts)


def test_the_cli_answers_offline(indexed, fixtures_dir, tmp_path):
    from api.cli import _run, build_runtime

    root = fixtures_dir / "repos" / "django_min"
    runtime = build_runtime(
        PY_REPO, root, offline=True, driver=indexed.driver,
        cache_path=tmp_path / "cli.sqlite", llm=ScriptedLLM("unused"),
    )
    args = argparse.Namespace(
        question="what calls `charge_card`", repo=PY_REPO, root=root,
        reindex=False, sync=False, offline=True, serve=False,
    )
    out, err = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            code = _run(args, runtime)
    finally:
        runtime.indexer.cache.close()

    assert code == 0
    assert "billing/tasks.py::billing.tasks.charge_card" in out.getvalue()
    assert "# route: STRUCTURAL" in err.getvalue()
    assert "callers of billing.tasks.charge_card: 3 found" in err.getvalue()


def test_the_cli_streams_an_answer_with_badges(indexed, fixtures_dir, tmp_path):
    from api.cli import _run, build_runtime

    root = fixtures_dir / "repos" / "django_min"
    runtime = build_runtime(
        PY_REPO, root, offline=True, driver=indexed.driver,
        cache_path=tmp_path / "cli.sqlite",
        llm=ScriptedLLM("The `charge_card` function bills a user [C1]."),
    )
    args = argparse.Namespace(
        question="how does `charge_card` work", repo=PY_REPO, root=root,
        reindex=False, sync=False, offline=False, serve=False,
    )
    out = io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            code = _run(args, runtime)
    finally:
        runtime.indexer.cache.close()

    assert code == 0
    assert "✓" in out.getvalue()
    assert "Coverage 1.00" in out.getvalue()
