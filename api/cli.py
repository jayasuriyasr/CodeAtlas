"""The CLI. Cut-ladder step 4 drops the Next.js console; this is what remains.

Plan §9: "*(Cut-ladder)* Next.js console. CLI is sufficient to gate this step."
So this is the surface the step 9 gate is demonstrated through — a real question
about the Django fixture, answered with citations, streamed.

`python -m api.cli --repo <id> --root <path> "how are tokens issued"`

It wires the real components end to end: index the tree, retrieve, expand, pack,
synthesize, validate. Which is also why it needs a live Neo4j and a configured
provider — every substitute that makes a unit test safe makes this demo a lie.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

import metrics
from adapters.python import PythonAdapter
from config import (
    EMBED_MODEL,
    LLM_MODEL,
    NEO4J_DEFAULT_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
)
from graph.bootstrap import apply_schema
from graph.reader import GraphReader
from graph.writer import GraphWriter
from index.cache import EmbeddingCache
from index.indexer import Indexer
from index.providers import (
    AnthropicProvider,
    ApproxCodeTokenizer,
    HashEmbeddingProvider,
)
from pack.render import Block
from retrieve.expand import NeighborhoodExpander
from retrieve.hybrid import HybridRetriever, Neo4jSearchBackend
from retrieve.router import QueryClass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sei", description="Ask a question about a repo.")
    parser.add_argument("question", nargs="?", help="the question to answer")
    parser.add_argument("--repo", default="local", help="repo_id (§9.1 tenancy key)")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--reindex", action="store_true", help="run a full reconcile first")
    parser.add_argument(
        "--offline",
        action="store_true",
        help=(
            "use the stand-in embedding provider and skip synthesis. Retrieval "
            "only — no source leaves the machine."
        ),
    )
    return parser


def _driver():
    from neo4j import GraphDatabase

    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def _embedding_provider(offline: bool):
    """The real provider needs a key; the stand-in says so in its name.

    §12.0 is a scoping decision, not a caveat: running with `--offline` keeps
    source on the machine, and the retrieval it produces is fulltext-quality
    only (F-009).
    """
    if offline or not os.environ.get("VOYAGE_API_KEY"):
        if not offline:
            print(
                "warning: VOYAGE_API_KEY not set; falling back to the stand-in "
                "embedding provider. Vector search will return noise (F-009).",
                file=sys.stderr,
            )
        return HashEmbeddingProvider()

    raise SystemExit(
        f"A real {EMBED_MODEL} client is not wired yet. Run with --offline, or "
        f"implement the provider before relying on vector recall."
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.question and not args.reindex:
        build_parser().print_help()
        return 2

    driver = _driver()
    try:
        driver.verify_connectivity()
    except Exception as exc:                       # noqa: BLE001 - report, don't mask
        print(f"Neo4j unavailable at {NEO4J_URI}: {exc}", file=sys.stderr)
        print("Start it with: docker compose up -d", file=sys.stderr)
        return 1

    database = NEO4J_DEFAULT_DATABASE
    apply_schema(driver, database)

    writer = GraphWriter(driver, database)
    reader = GraphReader(driver, database)
    tokenizer = ApproxCodeTokenizer()
    provider = _embedding_provider(args.offline)
    cache = EmbeddingCache(model=provider.name)

    if args.reindex:
        indexer = Indexer(
            writer=writer, reader=reader, cache=cache, provider=provider,
            tokenizer=tokenizer, roots={args.repo: args.root},
            adapters={ext: PythonAdapter() for ext in PythonAdapter.extensions},
        )
        started = time.perf_counter()
        asyncio.run(indexer.full_reconcile(args.repo))
        print(f"indexed {args.root} in {time.perf_counter() - started:.1f}s", file=sys.stderr)

    if not args.question:
        return 0

    retriever = HybridRetriever(backend=Neo4jSearchBackend(driver, database))
    expander = NeighborhoodExpander(runner=_Runner(driver, database))

    def retrieve(question: str, routed):
        vec = provider.embed([question])[0] if not args.offline else None
        candidates = retriever.search(args.repo, question, query_vec=vec)
        seed_uids = [c.uid for c in candidates[:8]]

        seeds = [
            Block(symbol=sym, score=1.0 / (rank + 1))
            for rank, uid in enumerate(seed_uids)
            if (sym := _load(reader, args.repo, uid)) is not None
        ]
        neighbors = [
            Block(symbol=_neighbor_symbol(n), score=n.score)
            for n in expander.expand(args.repo, seed_uids)
        ]
        return seeds, neighbors

    if args.offline:
        seeds, neighbors = retrieve(args.question, None)
        print(f"# Retrieval only (--offline). Model {LLM_MODEL} not called.\n")
        for block in seeds:
            print(f"  {block.symbol.rel_path}::{block.symbol.qualified_name}")
        return 0

    from api.app import InvestigationService
    from api.stream import SentenceEvent, TextEvent

    service = InvestigationService(
        retriever=retrieve,
        llm=AnthropicProvider(),
        tokenizer=tokenizer,
        graph_runner=_Runner(driver, database),
        repo_id=args.repo,
    )

    result = None
    for event, result in service.stream(args.question):
        if isinstance(event, TextEvent):
            sys.stdout.write(event.text)
            sys.stdout.flush()
        elif isinstance(event, SentenceEvent) and event.badge:
            sys.stdout.write(f"  {event.badge}\n")

    print("\n\n" + result.context.report())
    print(result.report.render())
    print(
        f"TTFT {result.ttft_seconds:.2f}s · total {result.total_seconds:.2f}s · "
        f"{result.cost.packed_tokens} packed tokens · ${result.cost.usd:.4f} "
        f"({result.cost.model})"
    )
    if result.retried:
        print("(answer was refined once — §6.3)")
    print(f"counters: {metrics.snapshot()}", file=sys.stderr)
    return 0


class _Runner:
    """`run(cypher, **params) -> list[dict]`, the shape expansion and T3 expect."""

    def __init__(self, driver, database: str) -> None:
        self._driver = driver
        self._database = database

    def run(self, cypher: str, **params):
        # Parameters as a dict, not `**kwargs`: the driver's own signature is
        # `Session.run(query, parameters=None, **kwargs)`, so a Cypher parameter
        # named `$query` — which the fulltext search uses — collides with it.
        with self._driver.session(database=self._database) as session:
            return [dict(row) for row in session.run(cypher, params)]


def _load(reader: GraphReader, repo_id: str, uid: str):
    from adapters.base import Symbol

    props = reader.symbol(repo_id, uid)
    if props is None:
        return None
    return Symbol(
        uid=props["uid"], repo_id=repo_id, rel_path=props.get("rel_path", ""),
        qualified_name=props.get("qualified_name", ""), name=props.get("name", ""),
        arity=props.get("arity", 0), ordinal=props.get("ordinal", 0),
        kind=props.get("kind", "function"), signature=props.get("signature", ""),
        docstring=props.get("docstring"), source_code=props.get("source_code", ""),
        enclosing_signature=props.get("enclosing_signature"),
        used_imports=list(props.get("used_imports") or []),
        start_line=props.get("start_line", 0), end_line=props.get("end_line", 0),
    )


def _neighbor_symbol(neighbor):
    from adapters.base import Symbol

    return Symbol(
        uid=neighbor.uid, repo_id="", rel_path=neighbor.rel_path,
        qualified_name=neighbor.qualified_name,
        name=neighbor.qualified_name.rsplit(".", 1)[-1],
        arity=0, ordinal=0, kind="function",
        signature=neighbor.signature or neighbor.qualified_name,
        docstring=neighbor.docstring, source_code="", enclosing_signature=None,
    )


if __name__ == "__main__":                         # pragma: no cover
    raise SystemExit(main())
