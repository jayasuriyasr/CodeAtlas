"""Indexing and asking, shared by the step-10 modules that query a real graph.

`indexed` itself is registered in this directory's `conftest.py`; importing a
fixture into a test module makes pytest see it twice.
"""

from __future__ import annotations

import asyncio


from adapters.python import PythonAdapter
from adapters.typescript import TypeScriptAdapter
from graph.reader import GraphReader
from graph.writer import GraphWriter
from index.cache import EmbeddingCache
from index.indexer import Indexer
from index.providers import ApproxCodeTokenizer, HashEmbeddingProvider
from retrieve.expand import NeighborhoodExpander
from retrieve.hybrid import HybridRetriever, Neo4jSearchBackend
from retrieve.pipeline import QueryRetriever
from retrieve.router import QueryRouter
from retrieve.structural import StructuralResolver

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


def index_fixtures(graph_db, fixtures_dir, tmp_path_factory):
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
