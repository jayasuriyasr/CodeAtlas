"""Step 7's gate — the refusal path is reachable, not dead code.

Plan §7: "`trace.tier2.ambiguous` fires on the App Router fixture (proves the
refusal path is reachable, not dead code)."

That distinction is the whole point of the gate. A refusal branch that no real
corpus can reach is indistinguishable from one that works, right up until a
customer's repository reaches it.

**The TS adapter is step 10.** So this module reads the real fixture files to
establish that two route files genuinely export `GET`, then writes the
corresponding symbols through `GraphWriter` directly. That is the honest way to
make the path reachable today: the collision is the fixture's, the graph write
is real, and only the parse is stood in for.
"""

from __future__ import annotations

import re
import time

import pytest

import metrics
from adapters.base import ParsedFile, Symbol, symbol_uid
from graph.reader import GraphReader
from graph.writer import GraphWriter
from retrieve.frames import FrameResolver, parse_stack_trace

REPO = "repo_step7_graph"

_EXPORTED_FN = re.compile(r"export\s+(?:async\s+)?function\s+(\w+)\s*\(", re.M)


@pytest.fixture(autouse=True)
def reset_metrics():
    metrics.reset()
    yield
    metrics.reset()


@pytest.fixture(autouse=True)
def clean(graph_db):
    graph_db.run("MATCH (n) DETACH DELETE n")
    yield
    graph_db.run("MATCH (n) DETACH DELETE n")


@pytest.fixture
def reader(graph_db) -> GraphReader:
    return GraphReader(graph_db.driver, graph_db.name)


@pytest.fixture
def app_router(graph_db, fixtures_dir):
    """Index the App Router fixture's exported handlers.

    Symbols are built from what the files actually declare, so if someone
    removes a route the fixture stops carrying the collision and this module
    fails loudly rather than passing vacuously.
    """
    repo = fixtures_dir / "repos" / "nextjs_min"
    batch: dict[str, ParsedFile] = {}

    for path in sorted(repo.rglob("route.ts")):
        rel = path.relative_to(repo).as_posix()
        module = rel[: -len(".ts")].replace("/", ".")
        symbols = []
        for ordinal, name in enumerate(_EXPORTED_FN.findall(path.read_text(encoding="utf-8"))):
            qualified_name = f"{module}.{name}"
            symbols.append(
                Symbol(
                    uid=symbol_uid(REPO, rel, qualified_name, 1, 0),
                    repo_id=REPO, rel_path=rel, qualified_name=qualified_name,
                    name=name, arity=1, ordinal=0, kind="function",
                    signature=f"export async function {name}(request)",
                    docstring=None, source_code=f"function {name}(request) {{}}",
                    enclosing_signature=None, start_line=1 + ordinal * 10,
                    end_line=9 + ordinal * 10,
                )
            )
        batch[rel] = ParsedFile(rel, f"h_{rel}", symbols, [], language="typescript")

    GraphWriter(graph_db.driver, graph_db.name).apply(
        REPO, batch, batch.keys(), {}, time.time_ns()
    )
    return batch


def test_the_fixture_actually_carries_the_collision(app_router):
    """The gate is only evidence if the corpus contains the failure case."""
    gets = [
        s
        for parsed in app_router.values()
        for s in parsed.symbols
        if s.name == "GET"
    ]
    assert len(gets) >= 2, (
        f"the App Router fixture no longer has colliding GET handlers: "
        f"{[s.qualified_name for s in gets]}"
    )
    assert len({s.rel_path for s in gets}) >= 2, "the handlers are in one file"
    assert len({s.uid for s in gets}) == len(gets), (
        "the UIDs collapsed — rel_path is in the UID for exactly this reason (§3.2)"
    )


def test_tier2_ambiguous_fires_on_the_app_router_fixture(app_router, reader):
    """Plan §7's gate.

    The stack frame carries a compiled chunk path and the bare name `GET`.
    §5.1: "the same reason sourcemaps are needed makes the path useless for
    disambiguation" — so Tier 0 misses and Tier 2 finds several candidates.
    """
    trace = (
        "TypeError: Cannot read properties of undefined (reading 'id')\n"
        "    at GET (/app/.next/server/chunks/4821.js:1:5678)\n"
    )
    frames = parse_stack_trace(trace)
    assert frames, "the trace did not parse"

    resolved = FrameResolver(reader=reader).resolve_all(REPO, frames)

    assert [r.uid for r in resolved] == [None], "the resolver guessed"
    assert resolved[0].tier == "unresolved"
    assert metrics.get("trace.tier2.ambiguous") == 1, (
        "the refusal path did not fire on the App Router fixture"
    )


def test_reader_returns_every_colliding_candidate(app_router, reader):
    """The reader must not decide for the resolver.

    A reader that returned the first match would make the refusal unreachable —
    dead code guarding a case that can no longer arrive, which is precisely what
    this gate exists to disprove.
    """
    candidates = reader.symbols_named(REPO, "GET")
    assert len(candidates) >= 2
    assert len({c.rel_path for c in candidates}) >= 2


def test_a_uniquely_named_handler_still_resolves(app_router, graph_db, reader):
    """Refusal must not become paralysis.

    §10.1's cut-ladder step 1 is "cut Tier 2 name match" — worth roughly three
    days. Keeping it only pays if the unambiguous case actually resolves.
    """
    unique = Symbol(
        uid=symbol_uid(REPO, "app/api/health/route.ts", "app.api.health.route.healthz", 0, 0),
        repo_id=REPO, rel_path="app/api/health/route.ts",
        qualified_name="app.api.health.route.healthz", name="healthz",
        arity=0, ordinal=0, kind="function",
        signature="export function healthz()", docstring=None,
        source_code="function healthz() {}", enclosing_signature=None,
        start_line=1, end_line=3,
    )
    batch = {
        "app/api/health/route.ts": ParsedFile(
            "app/api/health/route.ts", "hh", [unique], [], language="typescript"
        )
    }
    GraphWriter(graph_db.driver, graph_db.name).apply(
        REPO, batch, batch.keys(), {}, time.time_ns()
    )

    trace = "Error: boom\n    at healthz (/app/.next/server/chunks/9.js:1:2)\n"
    resolved = FrameResolver(reader=reader).resolve_all(REPO, parse_stack_trace(trace))

    assert resolved[0].uid == unique.uid
    assert resolved[0].tier == "name"
    assert resolved[0].badge == "~ name match"


def test_tier0_resolves_against_the_real_index(graph_db, reader):
    """§5.1 Tier 0 through `symbol_lines`, including the innermost-symbol rule.

    A traceback line inside `LoginView.post` is inside `LoginView` too. The
    method is the answer; the class is 200 lines the developer did not ask about.
    """
    def sym(qualified_name, name, start, end):
        return Symbol(
            uid=symbol_uid(REPO, "authx/views.py", qualified_name, 1, 0),
            repo_id=REPO, rel_path="authx/views.py", qualified_name=qualified_name,
            name=name, arity=1, ordinal=0, kind="function",
            signature=f"def {name}(self)", docstring=None,
            source_code="...", enclosing_signature=None,
            start_line=start, end_line=end,
        )

    cls = sym("authx.views.LoginView", "LoginView", 10, 90)
    method = sym("authx.views.LoginView.post", "post", 30, 60)
    batch = {
        "authx/views.py": ParsedFile("authx/views.py", "hv", [cls, method], [])
    }
    GraphWriter(graph_db.driver, graph_db.name).apply(
        REPO, batch, batch.keys(), {}, time.time_ns()
    )

    trace = (
        'Traceback (most recent call last):\n'
        '  File "/srv/app/authx/views.py", line 42, in post\n'
        "    user = verify(email)\n"
        "AttributeError: boom\n"
    )
    resolved = FrameResolver(reader=reader, repo_root="/srv/app").resolve_all(
        REPO, parse_stack_trace(trace)
    )

    assert resolved[0].uid == method.uid, "resolved to the class, not the method"
    assert resolved[0].tier == "direct"
    assert metrics.get("trace.tier0.resolved") == 1


def test_trace_resolution_rate_is_computable(app_router, reader):
    """§8.3's decision metric: Tier 0+2 resolved / TS-TSX frames.

    Scoped to TS/TSX because Python resolves near-100% at Tier 0 and a blended
    rate would mask the failure the sourcemap tier addresses. Recorded, not
    gated — the threshold is < 0.70 and the corpus here is two route files.
    """
    trace = (
        "TypeError: boom\n"
        "    at GET (/app/.next/server/chunks/4821.js:1:5678)\n"
        "    at t (/app/.next/server/chunks/4821.js:1:9012)\n"
        "    at <anonymous>\n"
    )
    resolved = FrameResolver(reader=reader).resolve_all(REPO, parse_stack_trace(trace))

    ts_frames = [r for r in resolved if r.frame.language == "javascript"]
    assert ts_frames, "no TS/TSX frames parsed"
    rate = sum(1 for r in ts_frames if r.uid is not None) / len(ts_frames)

    assert rate == 0.0, "every frame here is ambiguous or mangled by construction"
    assert metrics.get("trace.tier2.ambiguous") == 1
    assert metrics.get("trace.mangled_skipped") == 2
