"""Frame shapes real runtimes print, and what a frame's name can mean (F-027).

Found by measuring §8.3's TS/TSX TRACE resolution through the real pipeline for
the first time: 0 of 9 repository frames resolved. Three were refusals that
§5.1 asks for (two route files both export `GET`). The rest were not:

* `at async GET (…)` was not parsed as a frame at all, nor was any frame with a
  Windows path — the colon after the drive letter ended the match.
* `UserCard` was "ambiguous" between the component and the module of the file
  it lives in. A stack frame never names a module.
* `UserCardError.describe` matched nothing, because a symbol's `name` is
  `describe`.

None of the fixes is a guess: each *removes* candidates, and more than one
survivor is still a refusal ("Tier 2 … never guess").
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import metrics
from retrieve.frames import FrameResolver, StackFrame, parse_stack_trace
from retrieve.router import QueryClass, QueryRouter


@pytest.fixture(autouse=True)
def reset():
    metrics.reset()
    yield
    metrics.reset()


# -- parsing --------------------------------------------------------------------


@pytest.mark.parametrize(
    "line, fn, path, line_no",
    [
        ("at GET (/app/.next/server/chunks/4821.js:1:5678)",
         "GET", "/app/.next/server/chunks/4821.js", 1),
        ("at async GET (/app/app/api/users/route.ts:14:20)",
         "GET", "/app/app/api/users/route.ts", 14),
        ("at async /app/server.js:10:3", None, "/app/server.js", 10),
        ("at new UserCardError (/app/components/UserCard.tsx:44:5)",
         "UserCardError", "/app/components/UserCard.tsx", 44),
        ("at Object.save [as persist] (/app/src/store.ts:7:9)",
         "Object.save", "/app/src/store.ts", 7),
        ("at render (C:\\work\\shop\\components\\UserCard.tsx:12:3)",
         "render", "C:\\work\\shop\\components\\UserCard.tsx", 12),
        ("at C:\\work\\shop\\server.js:3:1", None, "C:\\work\\shop\\server.js", 3),
        ("at describe (file:///C:/work/shop/components/UserCard.tsx:48:12)",
         "describe", "file:///C:/work/shop/components/UserCard.tsx", 48),
        ("at load (webpack-internal:///./src/store.ts:7:9)",
         "load", "webpack-internal:///./src/store.ts", 7),
        ("at processTicksAndRejections (node:internal/process/task_queues:95:5)",
         "processTicksAndRejections", "node:internal/process/task_queues", 95),
    ],
)
def test_the_shapes_v8_prints_are_all_frames(line, fn, path, line_no):
    frames = parse_stack_trace(f"TypeError: boom\n    {line}\n")
    assert len(frames) == 1, frames
    frame = frames[0]
    assert (frame.fn_name, frame.file_path, frame.line_no) == (fn, path, line_no)
    assert frame.language == "javascript"


def test_a_frame_with_no_location_is_still_not_a_frame():
    """`at async Promise.all (index 0)` has nowhere to resolve to."""
    assert parse_stack_trace("    at async Promise.all (index 0)\n") == []


@pytest.mark.parametrize(
    "prose",
    [
        "look at the store (see notes) before the release",
        "we met at noon and again at 12:30",
        "what happens at async boundaries in the route handlers",
        "is a new UserCard created at every render",
    ],
)
def test_prose_about_code_is_still_not_a_trace(prose):
    """The router keys on emptiness (§5.1): a looser pattern must not turn a
    question into a TRACE."""
    assert parse_stack_trace(prose) == []
    assert QueryRouter().route(prose).query_class is not QueryClass.TRACE


def test_an_awaited_route_handler_is_a_trace():
    routed = QueryRouter().route(
        "Error: bad request\n    at async POST (/app/app/api/users/route.ts:22:11)\n"
    )
    assert routed.query_class is QueryClass.TRACE
    assert [f.fn_name for f in routed.frames] == ["POST"]


# -- what a name can mean -------------------------------------------------------


@dataclass
class Named:
    uid: str
    qualified_name: str
    kind: str = "function"


class Reader:
    """Name lookup only: every frame here has a path Tier 0 cannot place."""

    def __init__(self, *symbols: Named) -> None:
        self.symbols = symbols
        self.asked: list[str] = []

    def symbol_at_line(self, repo_id, rel_path, line):
        return None

    def symbols_named(self, repo_id, name):
        self.asked.append(name)
        return [s for s in self.symbols if s.qualified_name.rsplit(".", 1)[-1] == name]


def frame(fn: str) -> StackFrame:
    return StackFrame(raw=f"at {fn} (/app/.next/server/chunks/1.js:1:1)",
                      file_path="/app/.next/server/chunks/1.js", line_no=1,
                      fn_name=fn, language="javascript")


COMPONENT = (
    Named("u-module", "components.UserCard", kind="module"),
    Named("u-fn", "components.UserCard.UserCard"),
    Named("u-class", "components.UserCard.UserCardError", kind="class"),
    Named("u-describe", "components.UserCard.UserCardError.describe"),
)


def resolve(fn: str, *symbols: Named):
    reader = Reader(*symbols)
    return FrameResolver(reader=reader).resolve("r", frame(fn)), reader


def test_a_component_is_not_ambiguous_with_its_own_file():
    """`UserCard.tsx` exports `UserCard`: the React convention, and until now
    a guaranteed refusal."""
    resolved, _ = resolve("UserCard", *COMPONENT)
    assert (resolved.uid, resolved.tier) == ("u-fn", "name")
    assert metrics.get("trace.tier2.resolved") == 1
    assert metrics.get("trace.tier2.ambiguous") == 0


def test_a_module_alone_is_never_what_a_frame_names():
    resolved, _ = resolve("UserCard", Named("u-module", "components.UserCard", kind="module"))
    assert resolved.uid is None and resolved.tier == "unresolved"
    assert metrics.get("trace.tier2.miss") == 1


def test_two_functions_of_one_name_are_still_refused():
    """§5.1, T10. Removing the module must not weaken this."""
    resolved, _ = resolve(
        "GET",
        Named("m1", "app.api.users.route", kind="module"),
        Named("g1", "app.api.users.route.GET"),
        Named("g2", "app.api.orders.route.GET"),
    )
    assert resolved.uid is None
    assert metrics.get("trace.tier2.ambiguous") == 1


def test_a_method_frame_matches_by_its_class():
    resolved, reader = resolve("UserCardError.describe", *COMPONENT)
    assert (resolved.uid, resolved.tier) == ("u-describe", "name")
    assert reader.asked == ["describe"], "looked up by the name the graph stores"


def test_the_class_in_a_method_frame_disambiguates():
    """Two classes with a `describe` each: the dotted name says which."""
    resolved, _ = resolve(
        "UserCardError.describe",
        Named("a", "components.UserCard.UserCardError.describe"),
        Named("b", "components.Other.OtherError.describe"),
    )
    assert resolved.uid == "a"


def test_a_method_of_a_class_the_repository_does_not_have_is_a_miss():
    resolved, _ = resolve(
        "Response.describe", Named("a", "components.UserCard.UserCardError.describe"))
    assert resolved.uid is None
    assert metrics.get("trace.tier2.miss") == 1


def test_a_partial_class_name_does_not_match():
    """`Error.describe` must not match `UserCardError.describe`: the suffix is
    compared at a dot, not anywhere in the string."""
    resolved, _ = resolve(
        "Error.describe", Named("a", "components.UserCard.UserCardError.describe"))
    assert resolved.uid is None


@pytest.mark.parametrize("fn", ["Object.save", "Function.save", "Module.save", "exports.save"])
def test_a_receiver_v8_made_up_is_not_part_of_the_name(fn):
    resolved, reader = resolve(fn, Named("s", "src.store.save"))
    assert resolved.uid == "s" and reader.asked == ["save"]


def test_a_constructor_frame_resolves_to_its_class():
    resolved, _ = resolve("UserCardError", *COMPONENT)
    assert resolved.uid == "u-class"


@pytest.mark.parametrize("fn", ["Object.t", "Object.<anonymous>", "e.t"])
def test_a_minified_name_behind_a_receiver_is_still_minified(fn):
    resolved, reader = resolve(fn, Named("t", "src.store.t"))
    assert resolved.uid is None
    assert reader.asked == [], "never looked up: a 1-in-N guess is worse than no answer"


def test_a_reader_that_reports_no_kind_keeps_every_candidate():
    """Older readers return objects without `kind`; nothing is dropped on a
    field that is not there."""

    @dataclass
    class Bare:
        uid: str
        qualified_name: str

    reader = Reader()
    reader.symbols_named = lambda repo, name: [Bare("x", "pkg.mod.helper")]
    resolved = FrameResolver(reader=reader).resolve("r", frame("helper"))
    assert resolved.uid == "x"
