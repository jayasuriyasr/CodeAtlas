"""Step 5 — the pure move logic: chain collapse, git parsing, resolution.

No database and no event loop. These are the functions §4.1 and §4.4 specify
directly, and they are worth isolating because every failure downstream of them
looks like a graph problem.
"""

from __future__ import annotations

import pytest

import metrics
from adapters.base import symbol_uid
from adapters.python import PythonAdapter
from index.moves import (
    Move,
    collapse_chains,
    parse_git_renames,
    resolve_moves,
)

REPO = "repo_step5"


@pytest.fixture(autouse=True)
def reset_metrics():
    metrics.reset()
    yield
    metrics.reset()


# --------------------------------------------------------------------------
# collapse_chains (§4.1)
# --------------------------------------------------------------------------


def test_chain_collapses_to_one_net_move():
    """B5. A->B then B->C inside one debounce window is one move: A->C.

    §4.1 gives the reason: processing B->C evicts `pending[B]`, so the A->B
    remap would find nothing in the batch. Collapsing first means the only
    remap issued is the one whose new path is actually on disk.
    """
    collapsed = collapse_chains([Move("a.py", "b.py"), Move("b.py", "c.py")])
    assert collapsed == [Move("a.py", "c.py")]


def test_long_chain_collapses():
    moves = [Move("a.py", "b.py"), Move("b.py", "c.py"), Move("c.py", "d.py")]
    assert collapse_chains(moves) == [Move("a.py", "d.py")]


def test_round_trip_collapses_to_nothing():
    """A->B->A leaves the file where it started, so no UID needs rewriting."""
    assert collapse_chains([Move("a.py", "b.py"), Move("b.py", "a.py")]) == []


def test_duplicate_move_records_are_idempotent():
    """§4.1: "idempotent over duplicate Move records".

    Both editor signals can fire for one rename, and a doubled remap would
    attempt to rewrite a UID that no longer exists.
    """
    assert collapse_chains([Move("a.py", "b.py"), Move("a.py", "b.py")]) == [
        Move("a.py", "b.py")
    ]


def _key(moves):
    return sorted((m.old_path, m.new_path) for m in moves)


def test_independent_moves_are_preserved():
    moves = [Move("a.py", "x.py"), Move("b.py", "y.py")]
    assert _key(collapse_chains(moves)) == _key(moves)


def test_collapse_does_not_depend_on_input_order_for_independent_moves():
    """§4.1: collapsing "removes any dependence on Cypher row ordering"."""
    moves = [Move("a.py", "x.py"), Move("b.py", "y.py"), Move("c.py", "z.py")]
    assert _key(collapse_chains(moves)) == _key(collapse_chains(list(reversed(moves))))


def test_empty_input():
    assert collapse_chains([]) == []


# --------------------------------------------------------------------------
# parse_git_renames (§4.4 Signal B)
# --------------------------------------------------------------------------


def test_parses_a_staged_rename():
    """Format confirmed against git 2.52 on 2026-08-27: `R  ORIG -> NEW`."""
    assert parse_git_renames("R  pkg/alpha.py -> pkg/beta.py\n") == [
        Move("pkg/alpha.py", "pkg/beta.py")
    ]


@pytest.mark.parametrize("status", ["R ", "RM", "RD"])
def test_parses_every_rename_status_pair(status):
    """`RM` is renamed-then-edited; `RD` is renamed-then-deleted from the worktree.

    Matching only the exact string `R ` would drop both, and `RD` in particular
    is the real-world source of the `move.new_path_unreadable` case.
    """
    assert parse_git_renames(f"{status} a.py -> b.py\n") == [Move("a.py", "b.py")]


def test_parses_quoted_paths_with_spaces():
    """git quotes any path with a space.

    Left unhandled, the parsed path keeps its quotes, matches nothing in
    `pending`, and increments `move.unresolvable` — which §9.4 requires to be
    identically zero.
    """
    line = 'R  "pkg/has space.py" -> "pkg/moved space.py"\n'
    assert parse_git_renames(line) == [Move("pkg/has space.py", "pkg/moved space.py")]


def test_ignores_non_rename_lines():
    """§4.4: an unstaged move appears as `D old` plus `?? new` and is not detected.

    Inferring a move from that pair is exactly the similarity heuristic §4.4
    refuses — "a wrong merge is silent and corrupts the graph".
    """
    porcelain = " M pkg/edited.py\n?? pkg/new.py\n D pkg/gone.py\nA  pkg/added.py\n"
    assert parse_git_renames(porcelain) == []


def test_mixed_output():
    porcelain = (
        " M pkg/edited.py\n"
        "R  pkg/a.py -> pkg/b.py\n"
        "?? pkg/new.py\n"
        "RM pkg/c.py -> pkg/d.py\n"
    )
    assert parse_git_renames(porcelain) == [
        Move("pkg/a.py", "pkg/b.py"),
        Move("pkg/c.py", "pkg/d.py"),
    ]


def test_empty_output():
    assert parse_git_renames("") == []


def test_rename_to_the_same_path_is_not_a_move():
    assert parse_git_renames("R  a.py -> a.py\n") == []


# --------------------------------------------------------------------------
# resolve_moves (§4.4)
# --------------------------------------------------------------------------


SOURCE = b'''\
def helper(value):
    return value


class Thing:
    def method(self, a, b):
        return a + b
'''


@pytest.fixture
def parsed_at_new_path():
    return PythonAdapter().parse(REPO, "pkg/new.py", SOURCE)


def test_resolve_moves_computes_the_uid_the_graph_actually_stored(parsed_at_new_path):
    """F-007. The old UID must be what was written when the file lived at the old path.

    §4.4 computes both UIDs from the new file's symbols, "since qualified_name,
    arity, and ordinal survive a move". Arity and ordinal do. `qualified_name`
    does not — §3.2 derives it from the module path, so `pkg.new.helper` at
    `pkg/new.py` was `pkg.old.helper` at `pkg/old.py`.

    Taken literally the remap names a UID that was never written, §11.2 2a
    matches nothing, and every inbound edge is severed silently.
    """
    adapter = PythonAdapter()
    batch = {"pkg/new.py": parsed_at_new_path}
    remaps = resolve_moves(
        REPO, [Move("pkg/old.py", "pkg/new.py")], batch, adapter_for=lambda _p: adapter
    )

    assert len(remaps) == len(parsed_at_new_path.symbols)
    assert metrics.get("move.unresolvable") == 0
    assert metrics.get("move.requalify_failed") == 0

    # The UIDs the graph would be holding: the same bytes, parsed at the old path.
    stored = adapter.parse(REPO, "pkg/old.py", parsed_at_new_path.source)
    assert {r.old_uid for r in remaps} == {s.uid for s in stored.symbols}, (
        "the remap names UIDs that were never written to the graph"
    )
    assert {r.new_uid for r in remaps} == {s.uid for s in parsed_at_new_path.symbols}
    assert all(r.old_uid != r.new_uid for r in remaps)
    assert all(
        (r.old_path, r.new_path) == ("pkg/old.py", "pkg/new.py") for r in remaps
    )


def test_the_spec_formula_would_have_matched_nothing(parsed_at_new_path):
    """The failure F-007 describes, demonstrated rather than asserted about.

    Kept because the fix is easy to un-fix: anyone "simplifying" `resolve_moves`
    back to §4.4's one-liner reintroduces a defect with no symptom.
    """
    adapter = PythonAdapter()
    stored = {s.uid for s in adapter.parse(REPO, "pkg/old.py", parsed_at_new_path.source).symbols}

    literal = {
        symbol_uid(REPO, "pkg/old.py", s.qualified_name, s.arity, s.ordinal)
        for s in parsed_at_new_path.symbols
    }
    assert literal.isdisjoint(stored), (
        "§4.4's literal formula happens to work here; F-007 needs re-checking"
    )

    corrected = {
        r.old_uid
        for r in resolve_moves(
            REPO, [Move("pkg/old.py", "pkg/new.py")],
            {"pkg/new.py": parsed_at_new_path}, adapter_for=lambda _p: adapter,
        )
    }
    assert corrected == stored


def test_resolve_moves_covers_every_symbol_in_the_file(parsed_at_new_path):
    """A partial remap severs the edges of whichever symbols it skipped."""
    remaps = resolve_moves(
        REPO, [Move("pkg/old.py", "pkg/new.py")], {"pkg/new.py": parsed_at_new_path},
        adapter_for=lambda _p: PythonAdapter(),
    )
    assert {r.new_uid for r in remaps} == {s.uid for s in parsed_at_new_path.symbols}


def test_module_symbol_is_remapped_too(parsed_at_new_path):
    """The `:Module` symbol's own name changes with the filename.

    A guard that required `name` to match across the move would drop it, and
    §3.4.5's file chunk would be orphaned at the old path on every move.
    """
    remaps = resolve_moves(
        REPO, [Move("pkg/old.py", "pkg/new.py")], {"pkg/new.py": parsed_at_new_path},
        adapter_for=lambda _p: PythonAdapter(),
    )
    module = next(s for s in parsed_at_new_path.symbols if s.kind == "module")
    assert module.uid in {r.new_uid for r in remaps}
    assert metrics.get("move.requalify_failed") == 0


def test_unknown_language_at_the_old_path_refuses(parsed_at_new_path):
    """No adapter means no way to recover the old qualified names.

    Refusing beats emitting remaps that would rewrite the wrong nodes — the
    same reasoning §4.4 gives for having no similarity heuristic.
    """
    remaps = resolve_moves(
        REPO, [Move("pkg/old.py", "pkg/new.py")], {"pkg/new.py": parsed_at_new_path},
        adapter_for=lambda _p: None,
    )
    assert remaps == []
    assert metrics.get("move.requalify_failed") == 1
    assert metrics.get("move.unresolvable") == 0


@pytest.mark.expects_unresolvable
def test_unresolvable_move_is_counted_not_guessed():
    """§9.4 alarms on `move.unresolvable` — it must be identically zero.

    So the counter has to exist and fire, or the alarm is watching a number
    nothing can ever change. `collect_moves` is what keeps it at zero in
    practice, by parsing the new path itself.
    """
    remaps = resolve_moves(REPO, [Move("pkg/old.py", "pkg/missing.py")], {},
                           adapter_for=lambda _p: PythonAdapter())
    assert remaps == []
    assert metrics.get("move.unresolvable") == 1


def test_no_moves_no_remaps():
    assert resolve_moves(REPO, [], {}, adapter_for=lambda _p: PythonAdapter()) == []
    assert metrics.get("move.unresolvable") == 0
