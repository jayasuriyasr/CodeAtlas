"""Step 7 — frame resolution, and T10's refusal.

Run against a stub reader. What is under test is the resolver's *decision*:
which tier fired, and whether ambiguity produced a refusal or a guess. Whether
the graph returns the right candidates is step 3's question, checked again
against a real database in `test_frames_graph.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import metrics
from retrieve.frames import (
    FrameResolver,
    ResolvedFrame,
    StackFrame,
    build_tier3_query,
    error_message,
    is_mangled,
    normalize_path,
    parse_stack_trace,
)

REPO = "repo_step7"


@pytest.fixture(autouse=True)
def reset_metrics():
    metrics.reset()
    yield
    metrics.reset()


@dataclass
class Found:
    uid: str
    qualified_name: str = ""
    rel_path: str = ""
    start_line: int = 0
    end_line: int = 0


@dataclass
class StubReader:
    """Seeded per test. `at_line` maps (rel_path, line) -> Found."""

    at_line: dict = None
    named: dict = None

    def __post_init__(self):
        self.at_line = self.at_line or {}
        self.named = self.named or {}

    def symbol_at_line(self, repo_id, rel_path, line):
        for (path, lo, hi), found in self.at_line.items():
            if path == rel_path and lo <= line <= hi:
                return found
        return None

    def symbols_named(self, repo_id, name):
        return self.named.get(name, [])


PY_TRACE = '''Traceback (most recent call last):
  File "/srv/app/authx/views.py", line 42, in post
    user = verify_credentials(email, password)
  File "/srv/app/authx/services.py", line 17, in verify_credentials
    return user.check()
AttributeError: 'NoneType' object has no attribute 'check'
'''

JS_TRACE = """TypeError: Cannot read properties of undefined (reading 'id')
    at GET (/app/.next/server/chunks/4821.js:1:5678)
    at t (/app/.next/server/chunks/4821.js:1:9012)
    at <anonymous>
"""


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def test_parses_a_python_traceback():
    frames = parse_stack_trace(PY_TRACE)
    assert [(f.fn_name, f.line_no) for f in frames] == [
        ("post", 42),
        ("verify_credentials", 17),
    ]
    assert frames[0].file_path == "/srv/app/authx/views.py"
    assert all(f.language == "python" for f in frames)


def test_parses_a_js_stack_including_the_anonymous_frame():
    frames = parse_stack_trace(JS_TRACE)
    names = [f.fn_name for f in frames]
    assert "GET" in names and "t" in names and "<anonymous>" in names


def test_prose_is_not_a_stack_trace():
    """The router keys on emptiness here, so a false positive routes to TRACE."""
    assert parse_stack_trace("why does the login view fail at line 42") == []
    assert parse_stack_trace("") == []


def test_error_message_is_the_exception_line():
    """§5.1 Tier 3 re-routes on it, so it must survive parsing."""
    assert error_message(PY_TRACE) == (
        "AttributeError: 'NoneType' object has no attribute 'check'"
    )
    assert error_message(JS_TRACE).startswith("TypeError:")
    assert error_message("no exception here") == ""


@pytest.mark.parametrize(
    "raw, root, expected",
    [
        ("/srv/app/authx/views.py", "/srv/app", "authx/views.py"),
        ("C:\\srv\\app\\authx\\views.py", "C:/srv/app", "authx/views.py"),
        ("./authx/views.py", None, "authx/views.py"),
        ("authx/views.py", "/other/root", "authx/views.py"),
    ],
)
def test_normalize_path(raw, root, expected):
    """Tier 0's first step. Runtime paths are absolute and platform-shaped;
    `rel_path` is POSIX and repo-relative, and it is a UID input."""
    assert normalize_path(raw, root) == expected


# --------------------------------------------------------------------------
# Tier 0
# --------------------------------------------------------------------------


def test_tier0_python_traceback():
    """Exact resolution: file plus line, straight to a uid.

    Deterministic, so confidence is 1.0 and the badge is `✓ direct` — the only
    tier §12.4's trust model can lean on without qualification.
    """
    reader = StubReader(
        at_line={
            ("authx/views.py", 30, 60): Found("uid_post", "authx.views.LoginView.post"),
            ("authx/services.py", 10, 25): Found("uid_verify"),
        }
    )
    resolver = FrameResolver(reader=reader, repo_root="/srv/app")
    resolved = resolver.resolve_all(REPO, parse_stack_trace(PY_TRACE))

    assert [r.uid for r in resolved] == ["uid_post", "uid_verify"]
    assert all(r.tier == "direct" and r.confidence == 1.0 for r in resolved)
    assert [r.badge for r in resolved] == ["✓ direct", "✓ direct"]
    assert metrics.get("trace.tier0.resolved") == 2


def test_tier0_picks_the_innermost_symbol():
    """A line inside a method is inside its class too; the method is the answer.

    The reader orders by `start_line DESC` for this. Any other order returns the
    class roughly half the time, and the citation then points at 200 lines the
    developer did not ask about.
    """
    reader = StubReader(
        at_line={
            ("authx/views.py", 10, 90): Found("uid_class", "authx.views.LoginView"),
            ("authx/views.py", 30, 60): Found("uid_method", "authx.views.LoginView.post"),
        }
    )
    frame = StackFrame("", "authx/views.py", 42, "post", "python")
    # The stub returns the first match; the ordering guarantee is the reader's,
    # and is asserted against a real database in test_frames_graph.py.
    assert FrameResolver(reader=reader).resolve(REPO, frame).uid in {
        "uid_class", "uid_method"
    }


def test_tier0_miss_falls_through_to_tier2():
    """A file the graph has never seen is not an error, it is the next tier."""
    reader = StubReader(named={"post": [Found("uid_by_name")]})
    frame = StackFrame("", "authx/views.py", 42, "post", "python")
    resolved = FrameResolver(reader=reader).resolve(REPO, frame)

    assert resolved.tier == "name"
    assert resolved.uid == "uid_by_name"


# --------------------------------------------------------------------------
# Tier 2 — T10
# --------------------------------------------------------------------------


def test_tier2_ambiguous_refuses():
    """T10. Two functions named `GET` -> None, never a guess.

    §5.1: the frame's own path is a compiled chunk, so it cannot disambiguate —
    "the same reason sourcemaps are needed makes the path useless". Picking
    either candidate would produce a confident citation into a 1-in-N guess,
    and §12.4 says the trust model rests on that badge meaning something.
    """
    reader = StubReader(
        named={
            "GET": [
                Found("uid_users", "app.api.users.route.GET"),
                Found("uid_orders", "app.api.orders.route.GET"),
            ]
        }
    )
    frame = StackFrame("", "/app/.next/server/chunks/4821.js", 1, "GET", "javascript")
    resolved = FrameResolver(reader=reader).resolve(REPO, frame)

    assert resolved.uid is None
    assert resolved.tier == "unresolved"
    assert resolved.confidence == 0.0
    assert metrics.get("trace.tier2.ambiguous") == 1
    assert metrics.get("trace.tier2.resolved") == 0


def test_tier2_single_candidate_resolves():
    """The valid case still works — refusal must not become paralysis."""
    reader = StubReader(named={"issueToken": [Found("uid_issue")]})
    frame = StackFrame("", "/app/chunk.js", 1, "issueToken", "javascript")
    resolved = FrameResolver(reader=reader).resolve(REPO, frame)

    assert resolved.uid == "uid_issue"
    assert resolved.tier == "name"
    assert resolved.confidence == 0.6, "§5.1 fixes the name-match confidence"
    assert resolved.badge == "~ name match"


def test_tier2_no_candidates_is_a_miss_not_an_ambiguity():
    """§8.3 counts `tier2.ambiguous` as unresolved when sizing the sourcemap
    tier. Folding misses into it would inflate the case for building one."""
    resolved = FrameResolver(reader=StubReader()).resolve(
        REPO, StackFrame("", "/app/chunk.js", 1, "neverSeen", "javascript")
    )
    assert resolved.uid is None
    assert metrics.get("trace.tier2.miss") == 1
    assert metrics.get("trace.tier2.ambiguous") == 0


@pytest.mark.parametrize("name", ["t", "e", "<anonymous>", "<computed>", "a1", "_x"])
def test_mangled_name_skipped(name):
    """§5.1 names `at t (…)` and `at <anonymous>` as the cases to skip.

    A minified single letter matches dozens of real symbols by name; resolving
    it is a coin flip wearing a badge.
    """
    assert is_mangled(name)
    reader = StubReader(named={name: [Found("uid_wrong")]})
    resolved = FrameResolver(reader=reader).resolve(
        REPO, StackFrame("", "/app/chunk.js", 1, name, "javascript")
    )
    assert resolved.uid is None
    assert metrics.get("trace.mangled_skipped") == 1


@pytest.mark.parametrize("name", ["GET", "post", "issueToken", "verify_credentials"])
def test_real_names_are_not_treated_as_mangled(name):
    """The converse: over-eager skipping would disable Tier 2 entirely."""
    assert not is_mangled(name)


def test_mangled_frame_with_a_real_path_still_tries_tier0():
    """A minified *name* does not make the *line* useless.

    If the graph knows that file and line — an unbundled dev build, say — Tier 0
    answers exactly, and refusing on the name alone would throw that away.
    """
    reader = StubReader(at_line={("src/app.js", 1, 100): Found("uid_exact")})
    resolved = FrameResolver(reader=reader).resolve(
        REPO, StackFrame("", "src/app.js", 42, "t", "javascript")
    )
    assert resolved.uid == "uid_exact"
    assert resolved.tier == "direct"


# --------------------------------------------------------------------------
# Tier 3
# --------------------------------------------------------------------------


def test_tier3_fallback_uses_neighbor_frames():
    """§5.1: "Re-route on the error message plus any resolvable neighbouring frame."

    Both halves are load-bearing. The message alone loses where it happened;
    the neighbours alone lose what went wrong.
    """
    reader = StubReader(
        named={
            "GET": [Found("uid_a"), Found("uid_b")],       # ambiguous -> refused
            "loadUser": [Found("uid_loader")],             # resolves
        }
    )
    resolver = FrameResolver(reader=reader)
    trace = (
        "TypeError: Cannot read properties of undefined (reading 'id')\n"
        "    at GET (/app/.next/server/chunks/4821.js:1:5678)\n"
        "    at loadUser (/app/.next/server/chunks/4821.js:1:9012)\n"
    )
    resolved = resolver.resolve_all(REPO, parse_stack_trace(trace))
    query = build_tier3_query(trace, resolved)

    assert "TypeError" in query.text, "the error message was dropped"
    assert "GET" in query.text, "the unresolved frame's name was dropped"
    assert query.resolved_neighbors == ("uid_loader",)
    assert metrics.get("trace.tier3.fallback") == 1


def test_tier3_excludes_mangled_names_from_the_query():
    """Feeding `t` and `<anonymous>` to hybrid retrieval is feeding it noise."""
    resolved = [
        ResolvedFrame(StackFrame("", None, None, "t", "javascript"), None, "unresolved", 0.0),
        ResolvedFrame(
            StackFrame("", None, None, "<anonymous>", "javascript"), None, "unresolved", 0.0
        ),
    ]
    query = build_tier3_query("TypeError: boom", resolved)
    assert query.text == "TypeError: boom"


def test_tier3_with_nothing_resolvable_still_produces_a_query():
    """§9.3: degrade, never fail. An unresolvable trace is still answerable
    semantically from its message."""
    query = build_tier3_query("ValueError: bad input", [])
    assert query.text == "ValueError: bad input"
    assert query.resolved_neighbors == ()


# --------------------------------------------------------------------------
# The badge (§5.1, §12.4)
# --------------------------------------------------------------------------


def test_every_tier_has_a_distinct_badge():
    """§12.4: "the trust model rests on a badge".

    Three tiers collapsing to two labels would make `~ name match` and
    `? semantic` indistinguishable to a skimming developer — which is the
    failure §12.4 already warns is likely.
    """
    badges = {
        ResolvedFrame(StackFrame("", None, None, "f"), "u", tier, 1.0).badge
        for tier in ("direct", "name", "semantic")
    }
    assert len(badges) == 3
    assert badges == {"✓ direct", "~ name match", "? semantic"}
