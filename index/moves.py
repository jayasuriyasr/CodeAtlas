"""Move bookkeeping — the types, the chain collapse, and the two signals.

§4.4 is blunt about why any of this exists. Path-derived UIDs mean a move
changes every UID in a file, and handled naively the symbols are deleted and
recreated, severing every inbound `CALLS` edge from unchanged files:

> the system then reports zero callers for code with dozens, which is worse
> than an error because it looks like an answer.

The fix is an in-place UID rewrite, and everything here exists to work out
*which* rewrites to perform. Note what is deliberately absent: there is no
similarity heuristic. §4.4 rejects one outright — a missed move is recoverable
by one button and visible in the badge, a wrong merge is silent and corrupts
the graph.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

import metrics
from adapters.base import ParsedFile


@dataclass(frozen=True)
class Move:
    """One file relocation, from either signal in §4.4."""

    old_path: str
    new_path: str


@dataclass(frozen=True)
class Remap:
    """One symbol's in-place UID rewrite (§4.4).

    Rewriting the UID moves the symbol without touching a single inbound edge,
    because the UID *is* the identity every edge references.

    `arity` and `ordinal` survive a move; `qualified_name` does not, since §3.2
    derives it from the module path. §4.4 accounts for that as of v10.1 — see
    `resolve_moves`.
    """

    old_uid: str
    new_uid: str
    old_path: str
    new_path: str


# --------------------------------------------------------------------------
# Chain collapse (§4.1)
# --------------------------------------------------------------------------


def collapse_chains(moves: list[Move]) -> list[Move]:
    """A->B then B->C in one window becomes A->C. §4.1, verbatim.

    Without this, processing B->C evicts `pending[B]`, so the A->B remap finds
    nothing in the batch. Also idempotent over duplicate `Move` records, and
    removes any dependence on Cypher row ordering in §11.2 step 2a.

    A round trip (A->B then B->A) collapses to nothing, which is correct: the
    file is where it started and no UID needs rewriting.
    """
    origin: dict[str, str] = {}                     # current_path -> original_path
    for mv in moves:
        origin[mv.new_path] = origin.pop(mv.old_path, mv.old_path)
    return [Move(old, new) for new, old in origin.items() if old != new]


#: §4.1 names this `_collapse_chains`. Kept as an alias so a reader following
#: the spec finds the name it expects.
_collapse_chains = collapse_chains


# --------------------------------------------------------------------------
# Signal B — git (§4.4)
# --------------------------------------------------------------------------

#: `R  old -> new`, confirmed against git 2.52 on 2026-08-27. The status pair
#: may be `R `, `RM` (renamed then edited) or `RD` (renamed then deleted from
#: the worktree), so only the first column is matched.
_RENAME_LINE = re.compile(r"^R.\s+(.*?) -> (.*)$")

_ESCAPES = {
    "a": "\a", "b": "\b", "f": "\f", "n": "\n",
    "r": "\r", "t": "\t", "v": "\v", "\\": "\\", '"': '"',
}


def _unquote(path: str) -> str:
    """Undo git's C-style path quoting.

    git quotes any path with a space or a non-ASCII byte, so the naive version
    of this parser silently produces paths that begin with a `"` — which then
    match nothing in `pending`, resolve to no `ParsedFile`, and increment
    `move.unresolvable`, a counter §9.4 requires to be identically zero.
    """
    if not (path.startswith('"') and path.endswith('"')):
        return path

    body = path[1:-1]
    out: list[str] = []
    raw = bytearray()
    i = 0
    while i < len(body):
        ch = body[i]
        if ch != "\\":
            raw.extend(ch.encode("utf-8"))
            i += 1
            continue
        nxt = body[i + 1] if i + 1 < len(body) else ""
        if nxt in _ESCAPES:
            raw.extend(_ESCAPES[nxt].encode("utf-8"))
            i += 2
        elif nxt.isdigit():                          # \NNN octal byte
            raw.append(int(body[i + 1 : i + 4], 8))
            i += 4
        else:
            raw.extend(ch.encode("utf-8"))
            i += 1
    out.append(raw.decode("utf-8", errors="replace"))
    return "".join(out)


def parse_git_renames(porcelain: str) -> list[Move]:
    """Read `git status --porcelain -M` output into moves.

    Detects renames against index-vs-HEAD. §4.4's table is the honest scope: an
    unstaged working-tree move appears as `D old` plus `?? new` and is **not**
    detected — those lines are ignored here rather than guessed at.
    """
    moves: list[Move] = []
    for line in porcelain.splitlines():
        match = _RENAME_LINE.match(line)
        if match is None:
            continue
        old, new = _unquote(match.group(1)), _unquote(match.group(2))
        if old != new:
            moves.append(Move(old_path=old, new_path=new))
    return moves


# --------------------------------------------------------------------------
# Resolution (§4.4)
# --------------------------------------------------------------------------


def resolve_moves(
    repo: str,
    moves: list[Move],
    batch: dict[str, ParsedFile],
    *,
    adapter_for: Callable[[str], Any],
) -> list[Remap]:
    """§4.4 (v10.1).

    The *new* UID comes from the new file's parsed symbols. The old one cannot:
    `arity` and `ordinal` survive a move, but `qualified_name` does not, since
    §3.2 derives it from the module path — `pkg/callee.py::helper` is
    `pkg.callee.helper` and becomes `pkg.moved.helper` the instant the file
    moves.

    v10.0 computed both from the new symbols. Every `old_uid` was then derived
    from the *new* qualified name against the *old* path — a UID that was never
    written. §11.2 step 2a's `MATCH` found nothing, every remap was a silent
    no-op, and the move severed exactly the inbound edges the mechanism exists
    to preserve, while `move.unresolvable` stayed zero because a `ParsedFile`
    *was* found. (v10.1 §0.0 R1; FINDINGS F-007 carries the run.)

    So the same bytes are re-parsed at the old path, reproducing the qualified
    names the graph actually holds. Pairing is positional and then verified:
    identical bytes through one adapter give identical symbol order
    (`test_parse_is_deterministic`), and the check catches it if that stops
    being true.

    `move.unresolvable` should be identically zero (§9.4, alarmed). It fires
    only when a move's new path never reached the batch — which `collect_moves`
    exists to prevent by parsing it itself.
    """
    out: list[Remap] = []
    for mv in moves:
        parsed = batch.get(mv.new_path)
        if parsed is None:
            metrics.incr("move.unresolvable")
            continue

        adapter = adapter_for(mv.old_path)
        as_old = (
            adapter.parse(repo, mv.old_path, parsed.source) if adapter is not None else None
        )
        if as_old is None or len(as_old.symbols) != len(parsed.symbols):
            # The old path is a different language, or the two parses disagree.
            # Refusing beats emitting remaps that would rewrite the wrong nodes.
            metrics.incr("move.requalify_failed")
            continue

        for old_sym, new_sym in zip(as_old.symbols, parsed.symbols):
            # Defence in depth: identical bytes through one adapter give
            # identical symbol order, so a mismatch here means that stopped
            # being true. `name` is exempt for `:Module` symbols — a module's
            # name *is* the filename, so it is expected to change across a move.
            same = (old_sym.kind, old_sym.arity, old_sym.ordinal) == (
                new_sym.kind, new_sym.arity, new_sym.ordinal
            ) and (old_sym.kind == "module" or old_sym.name == new_sym.name)
            if not same:
                metrics.incr("move.requalify_failed")
                continue
            out.append(
                Remap(
                    old_uid=old_sym.uid,
                    new_uid=new_sym.uid,
                    old_path=mv.old_path,
                    new_path=mv.new_path,
                )
            )
    return out
