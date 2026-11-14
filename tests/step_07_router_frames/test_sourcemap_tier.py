"""§8.3's sourcemap tier: a compiled frame resolved through the build's source map.

Reproduced first: two App Router route files both export `GET` and `POST`, so a
frame `at GET (/app/.next/server/chunks/routes.js:8:22)` is refused at Tier 2
(ambiguous), and a minified `at t (…)` is skipped — although the build's own
source map says exactly which file and line each position came from. §8.3's
decision metric for this tier is TS/TSX TRACE resolution < 0.70; the golden
set measured 3 of 9.

The fixture is real compiler output, not hand-written: `tests/fixtures/
sourcemaps/build/` was produced by TypeScript 5.6.3 from the two route files
beside it with

    tsc --outFile .next/server/chunks/routes.js --module amd --target es2017 \\
        --sourceMap --inlineSources --rootDir . app/api/users/route.ts \\
        app/api/orders/route.ts

so one chunk maps to two sources, as a Next.js server chunk does. The tests
copy it into a temporary repository at `.next/server/chunks/`.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

import metrics
from retrieve.frames import FrameResolver, parse_stack_trace
from retrieve.sourcemap import decode_vlq, original_position, source_path

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sourcemaps"
REPO = "repo_sourcemap"


@dataclass
class Sym:
    uid: str
    qualified_name: str
    start_line: int
    end_line: int
    kind: str = "function"


SYMBOLS = {
    "app/api/users/route.ts": [Sym("u-get", "app.api.users.route.GET", 3, 6),
                               Sym("u-post", "app.api.users.route.POST", 8, 13),
                               Sym("u-load", "app.api.users.route.loadUsers", 15, 17)],
    "app/api/orders/route.ts": [Sym("o-get", "app.api.orders.route.GET", 3, 6),
                                Sym("o-post", "app.api.orders.route.POST", 8, 13),
                                Sym("o-load", "app.api.orders.route.loadOrders", 15, 17)],
}


class Reader:
    def symbol_at_line(self, repo_id, rel_path, line):
        hits = [s for s in SYMBOLS.get(rel_path, []) if s.start_line <= line <= s.end_line]
        return hits[0] if hits else None

    def symbols_named(self, repo_id, name):
        return [s for syms in SYMBOLS.values() for s in syms
                if s.qualified_name.rsplit(".", 1)[-1] == name]


@pytest.fixture(autouse=True)
def reset_metrics():
    metrics.reset()
    yield
    metrics.reset()


@pytest.fixture
def repo(tmp_path):
    shutil.copytree(FIXTURE / "app", tmp_path / "app")
    chunks = tmp_path / ".next" / "server" / "chunks"
    chunks.mkdir(parents=True)
    for name in ("routes.js", "routes.js.map"):
        shutil.copy(FIXTURE / "build" / name, chunks / name)
    return tmp_path


def position_of(repo: Path, text: str) -> tuple[int, int]:
    """Where `text` starts in the compiled chunk, as V8 prints it (1-based)."""
    lines = (repo / ".next/server/chunks/routes.js").read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, 1):
        if text in line:
            return number, line.index(text) + 1
    raise AssertionError(text)


def resolve(repo: Path, trace: str, *, root: Path | None = None):
    frames = parse_stack_trace(trace)
    resolver = FrameResolver(reader=Reader(), repo_root=root or repo,
                             known_paths=set(SYMBOLS))
    return resolver.resolve_all(REPO, frames)


def test_vlq_decodes_the_specification_examples():
    assert decode_vlq("A") == [0] and decode_vlq("C") == [1] and decode_vlq("D") == [-1]
    assert decode_vlq("gB") == [16] and decode_vlq("AAgBC") == [0, 0, 16, 1]


def test_a_js_frame_keeps_its_column():
    (frame,) = parse_stack_trace("Error: x\n    at GET (/app/.next/server/chunks/routes.js:8:22)")
    assert (frame.line_no, frame.col_no) == (8, 22)


def test_a_compiled_position_maps_to_its_source_line(repo):
    source_map = json.loads((repo / ".next/server/chunks/routes.js.map").read_text("utf-8"))
    line, col = position_of(repo, "users.map")
    original = original_position(source_map, line, col)
    assert original is not None and original.line == 5
    rel = source_path(source_map, original.source, repo / ".next/server/chunks/routes.js.map", repo)
    assert rel == "app/api/users/route.ts"


def test_an_ambiguous_route_handler_resolves_through_the_map(repo):
    line, col = position_of(repo, "throw new Error(\"bad request\")")
    (frame,) = resolve(repo, f"Error: bad request\n    at POST (/app/.next/server/chunks/routes.js:{line}:{col})")
    assert (frame.uid, frame.tier, frame.badge) == ("u-post", "sourcemap", "✓ source map")
    assert metrics.snapshot().get("frames_resolved_by_tier.sourcemap") == 1


def test_a_minified_name_resolves_through_the_map(repo):
    line, col = position_of(repo, "orders.filter")
    (frame, anonymous) = resolve(
        repo, f"TypeError: t is not a function\n    at t (/app/.next/server/chunks/routes.js:{line}:{col})"
              "\n    at <anonymous>")
    assert (frame.uid, frame.tier) == ("o-get", "sourcemap")
    assert anonymous.uid is None                   # no position: nothing to map


def test_without_a_map_the_refusal_is_unchanged(repo, tmp_path_factory):
    elsewhere = tmp_path_factory.mktemp("no_build")
    line, col = position_of(repo, "throw new Error(\"bad request\")")
    (frame,) = resolve(repo, f"Error: bad request\n    at POST (/app/.next/server/chunks/routes.js:{line}:{col})",
                       root=elsewhere)
    assert frame.uid is None and frame.tier == "unresolved"
    assert metrics.snapshot().get("trace.tier2.ambiguous") == 1


def test_a_map_older_than_the_source_is_not_trusted(repo):
    """The map embeds the source it was built from; a different file on disk
    means its line numbers may point anywhere, so it is not used."""
    route = repo / "app/api/users/route.ts"
    route.write_text("// moved down a line\n" + route.read_text("utf-8"), encoding="utf-8")
    line, col = position_of(repo, "throw new Error(\"bad request\")")
    (frame,) = resolve(repo, f"Error: bad request\n    at POST (/app/.next/server/chunks/routes.js:{line}:{col})")
    assert frame.uid is None
    assert metrics.snapshot().get("trace.sourcemap.stale") == 1


def test_the_trace_route_resolves_through_the_map_on_a_real_index(repo, graph_db, tmp_path_factory):
    """End to end: the real TS adapter, the graph, and the pipeline's TRACE route."""
    import asyncio

    from adapters.typescript import TypeScriptAdapter
    from graph.reader import GraphReader
    from graph.writer import GraphWriter
    from index.cache import EmbeddingCache
    from index.indexer import Indexer
    from index.providers import ApproxCodeTokenizer, HashEmbeddingProvider
    from retrieve.router import QueryRouter
    from tests.step_10_polyglot_eval.pipeline_support import retriever

    graph_db.run("MATCH (n) DETACH DELETE n")
    provider = HashEmbeddingProvider()
    cache = EmbeddingCache(tmp_path_factory.mktemp("c") / "c.sqlite", model=provider.name)
    ts = TypeScriptAdapter()
    try:
        asyncio.run(Indexer(
            writer=GraphWriter(graph_db.driver, graph_db.name),
            reader=GraphReader(graph_db.driver, graph_db.name),
            cache=cache, provider=provider, tokenizer=ApproxCodeTokenizer(),
            roots={REPO: repo}, adapters={e: ts for e in ts.extensions},
        ).full_reconcile(REPO))
        line, col = position_of(repo, "throw new Error(\"missing total\")")
        question = f"Error: missing total\n    at POST (/app/.next/server/chunks/routes.js:{line}:{col})"
        _seeds, _neighbors, notes = retriever(graph_db, REPO, root=repo)(
            question, QueryRouter().route(question))
        (frame,) = notes["frames"]
        named = graph_db.run("MATCH (s:Symbol {repo_id: $r, uid: $u}) RETURN s.qualified_name AS qn",
                             r=REPO, u=frame["uid"])
    finally:
        cache.close()
        graph_db.run("MATCH (n) DETACH DELETE n")
    assert frame["tier"] == "sourcemap"
    assert [row["qn"] for row in named] == ["app.api.orders.route.POST"]
