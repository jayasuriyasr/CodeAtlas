"""§5.1 frame resolution on traces from somewhere else.

A trace pasted into SEI was almost never produced in the checkout SEI indexed:
it comes from a container (`/app/...`), CI, or a colleague's machine. Two
refinements make Tier 0 and Tier 2 honest about that:

* a path is matched to the graph's `:File` paths by its longest unique suffix —
  and refused when the suffix is shared, as Tier 2 refuses a shared name;
* a library or runtime frame (`site-packages`, `node_modules`, `<frozen …>`) is
  never name-matched: it is not repository code, and a unique name match on
  Django's own `dispatch` is a confident citation into the wrong file.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import metrics
from retrieve.frames import (
    FrameResolver,
    StackFrame,
    is_library_path,
    normalize_path,
    parse_stack_trace,
)

REPO = "repo"
KNOWN = {"authx/views.py", "billing/views.py", "billing/tasks.py", "common/utils.py"}


@pytest.fixture(autouse=True)
def reset():
    metrics.reset()
    yield
    metrics.reset()


@dataclass(frozen=True)
class Found:
    uid: str


class Reader:
    def __init__(self, lines=None, named=None):
        self.lines = lines or {}
        self.named = named or {}
        self.line_calls: list[tuple[str, int]] = []

    def symbol_at_line(self, repo_id, rel_path, line):
        self.line_calls.append((rel_path, line))
        return self.lines.get((rel_path, line))

    def symbols_named(self, repo_id, name):
        return self.named.get(name, [])


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("/app/authx/views.py", "authx/views.py"),
        ("/home/ci/build/repo/billing/tasks.py", "billing/tasks.py"),
        ("C:\\Users\\dev\\proj\\common\\utils.py", "common/utils.py"),
        ("authx/views.py", "authx/views.py"),
    ],
)
def test_a_foreign_absolute_path_matches_by_unique_suffix(raw, expected):
    assert normalize_path(raw, None, KNOWN) == expected


def test_a_shared_suffix_is_refused_not_guessed():
    """`views.py` alone could be either app's views — no match is the answer."""
    assert normalize_path("/somewhere/else/views.py", None, KNOWN) == "somewhere/else/views.py"


def test_the_local_root_still_wins_and_ignores_case(tmp_path):
    root = tmp_path / "Repo"
    raw = str(root).upper() + "\\authx\\views.py"
    assert normalize_path(raw, root, None) == "authx/views.py"


def test_tier0_resolves_a_container_trace_through_known_paths():
    reader = Reader(lines={("authx/views.py", 42): Found("uid_post")})
    resolver = FrameResolver(reader=reader, known_paths=KNOWN)
    frame = StackFrame("", "/app/authx/views.py", 42, "post", "python")

    resolved = resolver.resolve(REPO, frame)
    assert (resolved.uid, resolved.tier) == ("uid_post", "direct")


@pytest.mark.parametrize(
    "path",
    [
        "/usr/lib/python3.12/site-packages/django/views/generic/base.py",
        "/app/.venv/lib/python3.12/site-packages/rest_framework/views.py",
        "/app/node_modules/next/dist/server/route.js",
        "node:internal/process/task_queues",
        "<frozen importlib._bootstrap>",
        "C:\\Python312\\Lib\\site-packages\\django\\core\\handlers\\base.py",
    ],
)
def test_library_paths_are_recognised(path):
    assert is_library_path(path)


@pytest.mark.parametrize(
    "path",
    ["authx/views.py", "/app/authx/views.py", "/app/lib/python/helpers.py",
     "mylib/python_utils.py", "src/node_modules_shim.ts", None, ""],
)
def test_repository_paths_are_not_library_paths(path):
    assert not is_library_path(path)


def test_a_library_frame_is_never_name_matched():
    """The repo has exactly one `dispatch`; Django's frame must not claim it."""
    reader = Reader(named={"dispatch": [Found("uid_repo_dispatch")]})
    frame = StackFrame(
        "", "/app/.venv/lib/python3.12/site-packages/django/views/generic/base.py",
        143, "dispatch", "python",
    )
    resolved = FrameResolver(reader=reader).resolve(REPO, frame)

    assert resolved.uid is None and resolved.tier == "unresolved"
    assert metrics.get("trace.library_frame") == 1
    assert metrics.get("trace.tier2.resolved") == 0


def test_a_full_django_traceback_resolves_only_repository_frames():
    trace = (
        "Traceback (most recent call last):\n"
        '  File "/app/.venv/lib/python3.12/site-packages/django/core/handlers/base.py", line 197, in _get_response\n'
        "    response = wrapped_callback(request, *callback_args, **callback_kwargs)\n"
        '  File "/app/.venv/lib/python3.12/site-packages/django/views/generic/base.py", line 143, in dispatch\n'
        "    return handler(request, *args, **kwargs)\n"
        '  File "/app/authx/views.py", line 42, in post\n'
        "    token = issue_token(user)\n"
        "ValueError: inactive user\n"
    )
    reader = Reader(
        lines={("authx/views.py", 42): Found("uid_post")},
        named={"dispatch": [Found("uid_wrong")], "_get_response": [Found("uid_wrong2")]},
    )
    frames = parse_stack_trace(trace)
    resolved = FrameResolver(reader=reader, known_paths=KNOWN).resolve_all(REPO, frames)

    assert [r.uid for r in resolved] == [None, None, "uid_post"]
    assert [r.tier for r in resolved] == ["unresolved", "unresolved", "direct"]
