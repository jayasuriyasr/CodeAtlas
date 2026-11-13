"""The CLI, and the server entry point. Cut-ladder step 4 drops the Next.js
console; this is what remains.

Plan §9: "*(Cut-ladder)* Next.js console. CLI is sufficient to gate this step."
So this is the surface the step 9 gate is demonstrated through — a real question
about the Django fixture, answered with citations, streamed.

    python -m api.cli --repo <id> --root <path> --reindex "how are tokens issued"
    python -m api.cli --repo <id> --root <path> --offline "what calls `charge_card`"
    python -m api.cli --repo <id> --root <path> --sync
    python -m api.cli --repo <id> --root <path> --serve --watch --port 8000

It wires the real components end to end: index the tree with both language
adapters, route the question (§5.1), retrieve by the route, expand, pack,
synthesize, validate. Which providers answer is decided here (`choose_providers`)
and announced on stderr — hosted when its key is set, the local Ollama models
otherwise (decision 0003), nothing at all under `--offline`.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import secrets
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import metrics
from adapters.python import PythonAdapter
from adapters.typescript import TypeScriptAdapter
from config import (
    AI_MODE,
    AVAILABLE,
    EMBEDDING_CACHE_PATH,
    LOCAL_CONTEXT_HEADROOM,
    LOCAL_EMBED_MODEL,
    LOCAL_LLM_MODEL,
    NEO4J_DEFAULT_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    OLLAMA_URL,
    RESERVED_OUT,
    RESERVED_SYS,
)
from graph.bootstrap import apply_schema
from graph.reader import GraphReader
from graph.writer import GraphWriter
from index.cache import EmbeddingCache, EmbeddingUnavailable, embedder_key
from index.indexer import Indexer
from index.ollama import OllamaEmbeddingProvider, OllamaLLM, has_model, installed_models
from index.providers import (
    AnthropicProvider,
    ApproxCodeTokenizer,
    DisabledEmbeddingProvider,
    UnavailableLLM,
    VoyageEmbeddingProvider,
)
from index.watcher import TreeWatcher
from retrieve.expand import NeighborhoodExpander
from retrieve.hybrid import HybridRetriever, Neo4jSearchBackend
from retrieve.pipeline import QueryRetriever
from retrieve.structural import StructuralResolver

AI_MODES = ("auto", "local", "cloud", "off")

#: The `Host` header values a server bound to this machine answers to.
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
#: Shortest `SEI_API_TOKEN` accepted from the environment.
MIN_TOKEN_LENGTH = 16


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
        "--ai",
        choices=AI_MODES,
        default=AI_MODE if AI_MODE in AI_MODES else "auto",
        help=(
            "which models answer: auto (hosted when its API key is set, else the "
            "local Ollama models), local, cloud, or off"
        ),
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help=(
            "no model at all, hosted or local: retrieval only, by fulltext and "
            "graph. Same as --ai off, and prints the retrieved code instead of "
            "an answer."
        ),
    )
    parser.add_argument("--serve", action="store_true", help="run the console and HTTP API")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="with --serve: index at start-up and re-index files as they change",
    )
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

    # The server notes, as warnings, every property and relationship type a
    # query names before the first write has created it. On a new database
    # that is every query, and the driver logs each one in full.
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# --------------------------------------------------------------------------
# Which models answer (decisions 0002 and 0003)
# --------------------------------------------------------------------------


def _hosted_llm_configured() -> bool:
    return any(os.environ.get(name) for name in
               ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"))


#: "Ask the server" — distinct from None, which is the answer "there is none".
_PROBE: set[str] = set()


def choose_providers(mode: str, *, local_models: set[str] | None = _PROBE):
    """`(llm, embedding_provider, notes)` for an AI mode.

    `local_models` is what the Ollama server holds: a set of names, or None when
    no server answers. Left at its default, the server is asked.

    `auto` prefers a hosted provider *the user has configured* — setting a key
    is a decision to send code to that service (§12.0) — and otherwise uses the
    local models if the Ollama server answers and holds them. The two halves
    are chosen independently: one key without the other is a normal state.

    Nothing here falls back silently. Each note says what was chosen and what
    it costs the answer, and the caller prints them.
    """
    notes: list[str] = []
    if mode == "off":
        return (UnavailableLLM("no model: running with --ai off"),
                DisabledEmbeddingProvider(), ["AI off: retrieval by fulltext and graph only."])

    if local_models is _PROBE:
        local_models = installed_models() if mode in ("auto", "local") else None
    local_llm = has_model(local_models, LOCAL_LLM_MODEL)
    local_embed = has_model(local_models, LOCAL_EMBED_MODEL)

    # -- synthesis ---------------------------------------------------------
    if mode == "cloud" or (mode == "auto" and _hosted_llm_configured()):
        llm = AnthropicProvider()
    elif local_llm:
        llm = OllamaLLM()
        notes.append(f"Answers: local model {LOCAL_LLM_MODEL} (nothing leaves this machine).")
    else:
        llm = UnavailableLLM(_why_no_local(local_models, LOCAL_LLM_MODEL))
        notes.append(f"No model for answers ({llm.reason}); answers are retrieval-only (§9.3).")

    # -- embeddings --------------------------------------------------------
    if mode in ("cloud", "auto") and os.environ.get("VOYAGE_API_KEY"):
        provider = VoyageEmbeddingProvider()
    elif mode != "cloud" and local_embed:
        provider = OllamaEmbeddingProvider()
        notes.append(f"Search: local embeddings {LOCAL_EMBED_MODEL}.")
    else:
        provider = DisabledEmbeddingProvider()
        why = "VOYAGE_API_KEY not set" if mode == "cloud" else _why_no_local(
            local_models, LOCAL_EMBED_MODEL)
        notes.append(f"No embedding model ({why}); search is fulltext and graph only (F-009).")
    return llm, provider, notes


def _why_no_local(models: set[str] | None, wanted: str) -> str:
    if models is None:
        return f"no Ollama server at {OLLAMA_URL}"
    return f"model {wanted} is not pulled — run: ollama pull {wanted}"


def context_budget(llm) -> int:
    """How many tokens of packed context this model is given.

    §5.4's `AVAILABLE` is a quality setting far below a hosted model's window.
    A local model's window is a hard capacity, smaller than that setting, so
    the budget becomes what fits: the window, less the answer and the system
    prompt, less headroom for the stand-in tokenizer's error (F-006, F-025).
    """
    window = getattr(llm, "context_tokens", None)
    if window is None:
        return AVAILABLE
    fits = int((window - RESERVED_OUT - RESERVED_SYS) * LOCAL_CONTEXT_HEADROOM)
    return max(0, min(AVAILABLE, fits))


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
    repo: str = ""
    root: Path | None = None
    reader: GraphReader | None = None
    notes: tuple[str, ...] = ()

    @property
    def provider(self):
        return self.indexer.provider

    @property
    def llm(self):
        return self.service.llm

    def close(self) -> None:
        self.indexer.cache.close()
        self.driver.close()

    # -- vectors -----------------------------------------------------------

    def vector_search(self) -> tuple[bool, str]:
        """Whether queries may use the vector arm, and why not when they may not."""
        provider = self.provider
        if not getattr(provider, "semantic", False):
            return False, "no embedding model is configured"
        recorded = self.indexer.cache.get_meta(embedder_key(self.repo))
        if recorded != provider.name:
            was = recorded or "no embedding model"
            return False, (f"the index was embedded with {was}, not {provider.name} — "
                           f"re-index to use vector search")
        return True, ""

    def embed_query(self, text: str):
        """Embed a query — only against an index the same model embedded.

        Otherwise the comparison is between two unrelated vector spaces: it
        returns results, ranked, and they mean nothing.
        """
        usable, why = self.vector_search()
        if not usable:
            raise EmbeddingUnavailable(why)
        provider = self.provider
        embed = getattr(provider, "embed_query", None)
        return embed(text) if embed is not None else provider.embed([text])[0]

    # -- keeping the index current ------------------------------------------

    async def bootstrap(self) -> None:
        """Bring the graph level with the working tree at server start.

        Three cases. Never indexed: index everything. Indexed with another
        embedder: re-embed everything — automatically when the new one is local
        and free, on request when it is hosted and billed. Otherwise: the
        ordinary diff, for whatever changed while nothing was watching.
        """
        indexer, repo, provider = self.indexer, self.repo, self.provider
        warm = getattr(self.llm, "warm", None)
        if warm is not None:
            # Load the answer model while the index catches up, not on the
            # first question. A failure here is not fatal: asking will say why.
            loading = asyncio.create_task(asyncio.to_thread(warm))
            loading.add_done_callback(lambda task: task.cancelled() or task.exception())
        known = await asyncio.to_thread(self.reader.known_paths, repo)
        recorded = indexer.cache.get_meta(embedder_key(repo))
        stale = getattr(provider, "semantic", False) and recorded != provider.name

        if not known:
            print(f"indexing {self.root} for the first time…", file=sys.stderr)
            await indexer.sync_project(repo)
        elif stale and getattr(provider, "local", False):
            print(f"re-embedding {repo} with {provider.name} "
                  f"(was {recorded or 'none'})…", file=sys.stderr)
            await indexer.sync_project(repo)
        else:
            if stale:
                print(f"note: {repo} was embedded with {recorded or 'no model'}; vector "
                      f"search stays off until you run --reindex ({provider.name} is "
                      f"billed, so that is not done for you).", file=sys.stderr)
            await indexer.reconcile(repo)
        print(f"index is current: {self.status().get('graph')}", file=sys.stderr)

    def status(self) -> dict:
        usable, why = self.vector_search()
        llm = self.llm
        payload = {
            "repo": self.repo,
            "root": str(self.root) if self.root else None,
            "llm": None if isinstance(llm, UnavailableLLM) else getattr(llm, "name", None),
            "llm_note": getattr(llm, "reason", None),
            "embeddings": self.provider.name,
            "vector_search": usable,
            "vector_search_note": why or None,
            "context_tokens": self.service.available,
        }
        try:
            payload["graph"] = self.reader.stats(self.repo)
        except Exception as exc:                   # noqa: BLE001 - status must answer
            payload["graph"] = None
            payload["neo4j_error"] = f"{type(exc).__name__}: {exc}"
        return payload


def build_runtime(
    repo: str,
    root: Path,
    *,
    offline: bool = False,
    ai: str | None = None,
    driver=None,
    cache_path: Path | None = None,
    llm=None,
    provider=None,
) -> Runtime:
    """Wire one repository's indexer, retriever and answering service.

    `cache_path`, `llm` and `provider` exist for tests: a test must neither
    write into the developer's embedding cache nor send source to a provider.
    """
    from api.app import InvestigationService

    driver = driver or _driver()
    database = NEO4J_DEFAULT_DATABASE
    apply_schema(driver, database)

    mode = "off" if offline else (ai or "auto")
    chosen_llm, chosen_provider, notes = choose_providers(mode)
    llm = llm or chosen_llm
    provider = provider or chosen_provider

    writer = GraphWriter(driver, database)
    reader = GraphReader(driver, database)
    tokenizer = ApproxCodeTokenizer()
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
        repo_root=root,
    )
    service = InvestigationService(
        retriever=retriever,
        llm=llm,
        tokenizer=tokenizer,
        graph_runner=runner,
        repo_id=repo,
        expander=retriever.expand,
        available=context_budget(llm),
    )
    runtime = Runtime(
        driver=driver, indexer=indexer, service=service, retriever=retriever,
        repo=repo, root=root, reader=reader, notes=tuple(notes),
    )
    if getattr(provider, "semantic", False):
        retriever.embed_query = runtime.embed_query
    return runtime


def _print_notes(notes: dict) -> None:
    route = notes.get("route_taken") or notes.get("route")
    print(f"# route: {route}", file=sys.stderr)
    if notes.get("vector_search") == "degraded":
        reason = notes.get("vector_search_note") or "embedding model unavailable"
        print(f"# Semantic search degraded ({reason})", file=sys.stderr)
    structural = notes.get("structural")
    if structural:
        line = (
            f"# {structural['rule']} of {', '.join(structural['entity']) or '?'}: "
            f"{structural['related_total']} found, {structural['related_shown']} shown"
        )
        if structural.get("fallback"):
            line += f" — {structural['fallback']}; answered semantically"
        print(line, file=sys.stderr)
        for item in structural.get("related", []):
            print(f"#   {item['rel_path']}:{item['start_line']}  {item['qualified_name']}",
                  file=sys.stderr)
    for frame in notes.get("frames", []):
        print(f"# {frame['badge']:<14} {frame['frame']}", file=sys.stderr)


#: The index the test suite leaves on its database (tests/db_guard.py).
TEST_DATABASE_MARKER = "sei_test_database_marker"


def refuse_test_database(driver, database: str = NEO4J_DEFAULT_DATABASE) -> None:
    """Stop if `database` is the test suite's: the next test run would wipe it."""
    with driver.session(database=database) as session:
        marked = session.run(
            "SHOW INDEXES YIELD name WHERE name = $name RETURN count(*) AS n",
            name=TEST_DATABASE_MARKER,
        ).single()["n"]
    if marked:
        raise SystemExit(
            f"this Neo4j is the test suite's database (it carries {TEST_DATABASE_MARKER}); "
            f"the next test run would delete anything indexed here. Point SEI_NEO4J_URI "
            f"at the app's own server."
        )


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
        print("Start it with: sei start   (or: docker compose up -d)", file=sys.stderr)
        driver.close()
        return 1
    try:
        refuse_test_database(driver)
    except SystemExit as stop:
        print(stop, file=sys.stderr)
        driver.close()
        return 1

    runtime = build_runtime(
        args.repo, args.root.resolve(), offline=args.offline, ai=args.ai, driver=driver
    )
    for note in runtime.notes:
        print(f"# {note}", file=sys.stderr)
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
        print("# Retrieval only (--offline). No model was called.\n")
        for block in seeds:
            print(f"  {block.symbol.rel_path}::{block.symbol.qualified_name}")
        if neighbors:
            print("\n# neighbours")
            for block in neighbors:
                print(f"  {block.symbol.rel_path}::{block.symbol.qualified_name}")
        return 0

    from api.stream import SentenceEvent, TextEvent

    # The answer streams as it is written; the verdicts follow it. A terminal
    # cannot go back and badge a sentence once the next one has started, and
    # badges printed mid-stream landed in the middle of the following sentence.
    result = None
    verdicts: list[SentenceEvent] = []
    for event, result in runtime.service.stream(args.question):
        if isinstance(event, TextEvent):
            sys.stdout.write(event.text)
            sys.stdout.flush()
        elif isinstance(event, SentenceEvent):
            if event.amends and verdicts:
                verdicts[-1] = event
            else:
                verdicts.append(event)
        elif isinstance(event, dict) and event.get("type") == "retrieval":
            _print_notes(event)
        elif isinstance(event, dict) and event.get("type") == "retry":
            verdicts.clear()
            # On stdout, with the answer: the two attempts are one stream, and
            # with the notice on stderr a redirected answer read as a single
            # paragraph that contradicted itself halfway through.
            print(f"\n\n# Refining… ({event['reason']})\n")
        elif isinstance(event, dict) and event.get("type") == "synthesis_unavailable":
            print(f"\n# Synthesis unavailable: {event['reason']}", file=sys.stderr)
            for block in event.get("blocks", []):
                print(f"  [{block['handle']}] {block['rel_path']}::{block['qualified_name']}")

    claims = [v for v in verdicts if v.badge]
    if claims:
        print("\n\n# Claims")
        for verdict in claims:
            line = " ".join(verdict.text.split())
            print(f"  {verdict.badge:<20} {line[:110]}{'…' if len(line) > 110 else ''}")
    cited = {h for v in verdicts for h in v.handles}
    sources = [p for p in result.context.blocks if p.handle and int(p.handle[1:]) in cited]
    if sources:
        # A handle is only worth citing if the reader can see what it names.
        print("\n# Sources")
        for packed in sources:
            symbol = packed.block.symbol
            print(f"  [{packed.handle}] {symbol.rel_path}:{symbol.start_line}  "
                  f"{symbol.qualified_name}")
    print("\n" + result.context.report())
    if result.report is not None:
        print(result.report.render())
        for sentence, missing in result.report.unsupported:
            line = " ".join(sentence.split())
            print(f"  ⚠ cited but unsupported ({', '.join(missing)} not in the cited code): "
                  f"{line[:90]}{'…' if len(line) > 90 else ''}")
    cost = result.cost
    print(
        f"TTFT {result.ttft_seconds or 0:.2f}s · total {result.total_seconds or 0:.2f}s · "
        f"{cost.packed_tokens if cost else 0} packed tokens · "
        f"${cost.usd if cost else 0:.4f} "
        f"({cost.model if cost else 'no model'}, {runtime.provider.name})"
    )
    if result.retried:
        print("(answer was refined once — §6.3)")
    print(f"counters: {metrics.snapshot()}", file=sys.stderr)
    return 0


def serve_security(host: str, *, env_token: str | None) -> tuple[str, frozenset[str] | None]:
    """The token and allowed `Host` names for a server bound to `host`.

    A local bind gets a fresh random token for this run (or `SEI_API_TOKEN`,
    so a script that started the server can call it) and answers only to
    localhost names. A bind anyone else can reach serves the source code to
    them, so it is refused unless the user chose a token deliberately — and
    then the Host check is dropped, because clients will use the machine's
    own name and the token is what protects it.
    """
    if env_token is not None and len(env_token) < MIN_TOKEN_LENGTH:
        raise SystemExit(f"SEI_API_TOKEN must be at least {MIN_TOKEN_LENGTH} characters")
    if host in LOCAL_HOSTS:
        return env_token or secrets.token_urlsafe(32), LOCAL_HOSTS
    if not env_token:
        raise SystemExit(
            f"refusing to serve on {host}: anyone who can reach it could read the "
            f"source code. Set SEI_API_TOKEN to a long random value first."
        )
    return env_token, None


def _serve(args, runtime: Runtime) -> int:
    import uvicorn

    from api.app import create_app

    watcher = None
    if getattr(args, "watch", False):
        watcher = TreeWatcher(
            indexer=runtime.indexer, repo=args.repo, root=runtime.root,
            extensions=tuple(runtime.indexer.adapters),
        )
    token, hosts = serve_security(args.host, env_token=os.environ.get("SEI_API_TOKEN"))
    app = create_app(
        runtime.service, runtime.indexer, repos=[args.repo],
        watcher=watcher,
        bootstrap=runtime.bootstrap if watcher is not None else None,
        status=runtime.status,
        token=token,
        allowed_hosts=hosts,
    )
    shown = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    print(f"serving {args.repo} ({args.root}) on http://{args.host}:{args.port}", file=sys.stderr)
    print(f"console: http://{shown}:{args.port}/?token={token}", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":                         # pragma: no cover
    raise SystemExit(main())
