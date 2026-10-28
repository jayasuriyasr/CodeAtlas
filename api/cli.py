"""The CLI, and the server entry point. Cut-ladder step 4 drops the Next.js
console; this is what remains.

Plan §9: "*(Cut-ladder)* Next.js console. CLI is sufficient to gate this step."
So this is the surface the step 9 gate is demonstrated through — a real question
about the Django fixture, answered with citations, streamed.

    python -m api.cli --repo <id> --root <path> --reindex "how are tokens issued"
    python -m api.cli --repo <id> --root <path> --offline "what calls `charge_card`"
    python -m api.cli --repo <id> --root <path> --sync
    python -m api.cli --repo <id> --root <path> --serve --port 8000

It wires the real components end to end: index the tree with both language
adapters, route the question (§5.1), retrieve by the route, expand, pack,
synthesize, validate. Which is also why it needs a live Neo4j and configured
providers — every substitute that makes a unit test safe makes this demo a lie,
and each one in use is announced on stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import metrics
from adapters.python import PythonAdapter
from adapters.typescript import TypeScriptAdapter
from config import (
    EMBED_MODEL,
    EMBEDDING_CACHE_PATH,
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
    VoyageEmbeddingProvider,
)
from retrieve.expand import NeighborhoodExpander
from retrieve.hybrid import HybridRetriever, Neo4jSearchBackend
from retrieve.pipeline import QueryRetriever
from retrieve.structural import StructuralResolver


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sei", description="Ask a question about a repo.")
    parser.add_argument("question", nargs="?", help="the question to answer")
    parser.add_argument("--repo", default="local", help="repo_id (§9.1 tenancy key)")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="re-index the whole tree first, re-resolving every edge",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Sync Project (§4.1): the recovery path for undetected moves",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help=(
            "use the stand-in embedding provider and skip synthesis. Retrieval "
            "only — no source leaves the machine."
        ),
    )
    parser.add_argument("--serve", action="store_true", help="run the HTTP API (FastAPI + SSE)")
    parser.add_argument("--host", default="127.0.0.1", help="--serve bind address")
    parser.add_argument("--port", type=int, default=8000, help="--serve port")
    return parser


def _utf8_streams() -> None:
    """Make stdout/stderr able to print ✓ and ⚠ wherever this runs.

    A Windows console or a redirected stream defaults to the locale code page
    (cp1252), which cannot encode the badges §6.4 prints: the first cited
    sentence raised `UnicodeEncodeError` and killed the answer mid-stream.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass                                  # not a reconfigurable text stream


def _driver():
    from neo4j import GraphDatabase

    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def _embedding_provider(offline: bool):
    """The real provider when a key is configured; the stand-in otherwise.

    Returns `(provider, vectors_meaningful)`. The stand-in's vectors carry no
    semantic structure (F-009), so with it the vector arm is switched off for
    queries rather than fused in as noise.
    """
    if offline:
        return HashEmbeddingProvider(), False
    if os.environ.get("VOYAGE_API_KEY"):
        return VoyageEmbeddingProvider(), True
    print(
        "warning: VOYAGE_API_KEY not set; using the stand-in embedding provider. "
        "Retrieval runs on fulltext and graph expansion only (F-009).",
        file=sys.stderr,
    )
    return HashEmbeddingProvider(), False


def _adapters() -> dict:
    python, typescript = PythonAdapter(), TypeScriptAdapter()
    adapters = {ext: python for ext in python.extensions}
    adapters.update({ext: typescript for ext in typescript.extensions})
    return adapters


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


@dataclass
class Runtime:
    """Everything one repository needs, built once and shared by CLI and server."""

    driver: object
    indexer: Indexer
    service: object
    retriever: QueryRetriever

    def close(self) -> None:
        self.indexer.cache.close()
        self.driver.close()


def build_runtime(
    repo: str,
    root: Path,
    *,
    offline: bool,
    driver=None,
    cache_path: Path | None = None,
    llm=None,
) -> Runtime:
    """Wire one repository's indexer, retriever and answering service.

    `cache_path` and `llm` exist for tests: a test must neither write into the
    developer's embedding cache nor send source to a provider.
    """
    from api.app import InvestigationService

    driver = driver or _driver()
    database = NEO4J_DEFAULT_DATABASE
    apply_schema(driver, database)

    writer = GraphWriter(driver, database)
    reader = GraphReader(driver, database)
    tokenizer = ApproxCodeTokenizer()
    provider, meaningful = _embedding_provider(offline)
    cache = EmbeddingCache(
        cache_path or EMBEDDING_CACHE_PATH,
        model=provider.name,
        dimensions=provider.dimensions,
    )
    indexer = Indexer(
        writer=writer, reader=reader, cache=cache, provider=provider,
        tokenizer=tokenizer, roots={repo: root}, adapters=_adapters(),
    )
    runner = _Runner(driver, database)
    retriever = QueryRetriever(
        repo_id=repo,
        reader=reader,
        hybrid=HybridRetriever(backend=Neo4jSearchBackend(driver, database)),
        expander=NeighborhoodExpander(runner=runner),
        structural=StructuralResolver(runner=runner),
        embed_query=(lambda text: provider.embed([text])[0]) if meaningful else None,
        repo_root=root,
    )
    service = InvestigationService(
        retriever=retriever,
        llm=llm or AnthropicProvider(),
        tokenizer=tokenizer,
        graph_runner=runner,
        repo_id=repo,
        expander=retriever.expand,
    )
    return Runtime(driver=driver, indexer=indexer, service=service, retriever=retriever)


def _print_notes(notes: dict) -> None:
    route = notes.get("route_taken") or notes.get("route")
    print(f"# route: {route}", file=sys.stderr)
    if notes.get("vector_search") == "degraded":
        print("# Semantic search degraded (embedding API unavailable)", file=sys.stderr)
    structural = notes.get("structural")
    if structural:
        line = (
            f"# {structural['rule']} of {', '.join(structural['entity']) or '?'}: "
            f"{structural['related_total']} found, {structural['related_shown']} shown"
        )
        if structural.get("fallback"):
            line += f" — {structural['fallback']}; answered semantically"
        print(line, file=sys.stderr)
    for frame in notes.get("frames", []):
        print(f"# {frame['badge']:<14} {frame['frame']}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    _utf8_streams()
    args = build_parser().parse_args(argv)
    if not (args.question or args.reindex or args.sync or args.serve):
        build_parser().print_help()
        return 2
    if not args.root.is_dir():
        print(f"--root {args.root} is not a directory", file=sys.stderr)
        return 2

    driver = _driver()
    try:
        driver.verify_connectivity()
    except Exception as exc:                       # noqa: BLE001 - report, don't mask
        print(f"Neo4j unavailable at {NEO4J_URI}: {exc}", file=sys.stderr)
        print("Start it with: docker compose up -d", file=sys.stderr)
        driver.close()
        return 1

    runtime = build_runtime(args.repo, args.root.resolve(), offline=args.offline, driver=driver)
    try:
        return _run(args, runtime)
    finally:
        runtime.close()


def _run(args, runtime: Runtime) -> int:
    if args.reindex or args.sync:
        started = time.perf_counter()
        if args.sync:
            asyncio.run(runtime.indexer.sync_project(args.repo))
        else:
            asyncio.run(runtime.indexer.full_reconcile(args.repo, reparse_all=True))
        print(f"indexed {args.root} in {time.perf_counter() - started:.1f}s", file=sys.stderr)

    if args.serve:
        return _serve(args, runtime)

    if not args.question:
        return 0

    if args.offline:
        routed = runtime.service.router.route(args.question)
        seeds, neighbors, notes = runtime.retriever(args.question, routed)
        _print_notes(notes)
        print(f"# Retrieval only (--offline). Model {LLM_MODEL} not called.\n")
        for block in seeds:
            print(f"  {block.symbol.rel_path}::{block.symbol.qualified_name}")
        if neighbors:
            print("\n# neighbours")
            for block in neighbors:
                print(f"  {block.symbol.rel_path}::{block.symbol.qualified_name}")
        return 0

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "warning: ANTHROPIC_API_KEY not set; synthesis will be unavailable and "
            "the answer is retrieval-only (§9.3).",
            file=sys.stderr,
        )

    from api.stream import SentenceEvent, TextEvent

    result = None
    for event, result in runtime.service.stream(args.question):
        if isinstance(event, TextEvent):
            sys.stdout.write(event.text)
            sys.stdout.flush()
        elif isinstance(event, SentenceEvent) and event.badge:
            prefix = "  (amended) " if event.amends else "  "
            sys.stdout.write(f"{prefix}{event.badge}\n")
        elif isinstance(event, dict) and event.get("type") == "retrieval":
            _print_notes(event)
        elif isinstance(event, dict) and event.get("type") == "retry":
            print(f"\n\n# Refining… ({event['reason']})\n", file=sys.stderr)
        elif isinstance(event, dict) and event.get("type") == "synthesis_unavailable":
            print(f"\n# Synthesis unavailable: {event['reason']}", file=sys.stderr)
            for block in event.get("blocks", []):
                print(f"  [{block['handle']}] {block['rel_path']}::{block['qualified_name']}")

    print("\n\n" + result.context.report())
    if result.report is not None:
        print(result.report.render())
    print(
        f"TTFT {result.ttft_seconds or 0:.2f}s · total {result.total_seconds or 0:.2f}s · "
        f"{result.cost.packed_tokens if result.cost else 0} packed tokens · "
        f"${result.cost.usd if result.cost else 0:.4f} ({LLM_MODEL}, {EMBED_MODEL})"
    )
    if result.retried:
        print("(answer was refined once — §6.3)")
    print(f"counters: {metrics.snapshot()}", file=sys.stderr)
    return 0


def _serve(args, runtime: Runtime) -> int:
    import uvicorn

    from api.app import create_app

    app = create_app(runtime.service, runtime.indexer, repos=[args.repo])
    print(f"serving {args.repo} ({args.root}) on http://{args.host}:{args.port}", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":                         # pragma: no cover
    raise SystemExit(main())
