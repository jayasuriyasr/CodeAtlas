"""§5.1 — stack frame resolution, and the refusal that makes the badge mean something.

Three tiers, and the middle one is the interesting one:

* **Tier 0 — Direct (Python).** Normalize to repo-relative, then
  `(repo_id, rel_path, start_line <= N <= end_line)`. Deterministic.
* **Tier 2 — Name match (TS/JS/TSX).** Exactly one candidate resolves.
  **More than one is a refusal**, not a choice.
* **Tier 3 — SEMANTIC fallback.** Re-route on the error message plus any
  neighbouring frame that did resolve.

§5.1 on why Tier 2 refuses rather than picks:

> Next.js App Router route files each export a function named `GET`, `POST`,
> `PUT`, or `DELETE`, so a repo carries dozens of identical bare names, and the
> frame's own path is a compiled chunk — the same reason sourcemaps are needed
> makes the path useless for disambiguation.

And the consequence, which is a trust argument rather than an accuracy one:
"A `~ name match` badge on a 1-in-40 guess is worse than no answer, because
§12.4 shows the trust model depends on that badge meaning something."
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import metrics

# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StackFrame:
    """One line of a pasted stack trace, before resolution."""

    raw: str
    file_path: str | None
    line_no: int | None
    fn_name: str | None
    language: str = "unknown"


@dataclass(frozen=True)
class ResolvedFrame:
    """A frame, resolved or explicitly not.

    `tier` drives the per-frame badge §5.1 requires: `✓ direct`, `~ name match`,
    `? semantic`. `uid` is None when the frame was refused.
    """

    frame: StackFrame
    uid: str | None
    tier: str                       # "direct" | "name" | "semantic" | "unresolved"
    confidence: float

    BADGES = {
        "direct": "✓ direct",
        "name": "~ name match",
        "semantic": "? semantic",
        "unresolved": "? unresolved",
    }

    @property
    def badge(self) -> str:
        return self.BADGES[self.tier]


_PY_HEADER = re.compile(r"^\s*Traceback \(most recent call last\)", re.M)
_PY_FRAME = re.compile(
    r'^\s*File "(?P<path>[^"]+)", line (?P<line>\d+), in (?P<fn>[^\s]+)', re.M
)
#: `at fn (path:line:col)` and the bare `at path:line:col` form.
_JS_FRAME = re.compile(
    r"^\s*at\s+(?:(?P<fn>[^\s(]+)\s+\()?(?P<path>[^\s():]+):(?P<line>\d+):(?P<col>\d+)\)?",
    re.M,
)
_JS_ANON = re.compile(r"^\s*at\s+(?P<fn><[^>]+>)\s*$", re.M)

#: Names a minifier produced, or the runtime invented. §5.1 lists `at t (…)` and
#: `at <anonymous>` as the cases to skip.
#:
#: One or two identifier characters is the minifier's output shape. This does
#: skip a genuine two-character function, and that is the intended trade: a
#: skipped frame degrades to Tier 3, while a resolved wrong one produces a
#: confident citation into unrelated code.
_MANGLED = re.compile(r"^(?:<[^>]*>|[A-Za-z_$][A-Za-z0-9_$]?)$")


def is_mangled(fn_name: str | None) -> bool:
    return fn_name is None or bool(_MANGLED.match(fn_name))


def parse_stack_trace(text: str) -> list[StackFrame]:
    """Extract frames from pasted text. Empty when the text is not a trace.

    Emptiness is what the router keys on, so this must not match prose. The
    patterns are anchored on the shapes runtimes actually emit — a `File "...",
    line N, in fn` line, or an `at fn (path:line:col)` line.
    """
    frames: list[StackFrame] = []

    for match in _PY_FRAME.finditer(text):
        frames.append(
            StackFrame(
                raw=match.group(0).strip(),
                file_path=match.group("path"),
                line_no=int(match.group("line")),
                fn_name=match.group("fn"),
                language="python",
            )
        )

    for match in _JS_FRAME.finditer(text):
        frames.append(
            StackFrame(
                raw=match.group(0).strip(),
                file_path=match.group("path"),
                line_no=int(match.group("line")),
                fn_name=match.group("fn"),
                language="javascript",
            )
        )

    for match in _JS_ANON.finditer(text):
        frames.append(
            StackFrame(
                raw=match.group(0).strip(),
                file_path=None,
                line_no=None,
                fn_name=match.group("fn"),
                language="javascript",
            )
        )

    # A bare Python header with no frame lines is still a trace being pasted.
    if not frames and _PY_HEADER.search(text):
        frames.append(
            StackFrame(raw=text.strip().splitlines()[0], file_path=None,
                       line_no=None, fn_name=None, language="python")
        )

    frames.sort(key=lambda f: text.find(f.raw))
    return frames


def error_message(text: str) -> str:
    """The exception line, for Tier 3's re-route.

    §5.1: "Re-route on the error message plus any resolvable neighbouring
    frame." The message is usually the most retrievable thing in a trace, since
    it often contains an identifier from the code that raised it.
    """
    candidates = [
        line.strip()
        for line in text.splitlines()
        if re.match(r"^\s*[A-Z][A-Za-z0-9_.]*(Error|Exception|Warning)\b", line)
    ]
    return candidates[-1] if candidates else ""


def normalize_path(file_path: str, repo_root: Path | str | None = None) -> str:
    """A runtime path made repo-relative (§5.1 Tier 0's first step).

    Runtime paths are absolute and platform-shaped; `rel_path` is POSIX and
    repo-relative, and it is a UID input. Getting this wrong makes Tier 0 miss
    every frame while looking like a graph problem.
    """
    path = file_path.replace("\\", "/")
    if repo_root is not None:
        root = str(Path(repo_root)).replace("\\", "/").rstrip("/")
        if path.startswith(root + "/"):
            return path[len(root) + 1 :]
    return path.lstrip("./")


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------


class FrameReader(Protocol):
    """The two lookups §5.1 needs, and no more."""

    def symbol_at_line(self, repo_id: str, rel_path: str, line: int): ...

    def symbols_named(self, repo_id: str, name: str) -> Sequence: ...


@dataclass
class FrameResolver:
    reader: FrameReader
    repo_root: Path | str | None = None

    def resolve(self, repo_id: str, frame: StackFrame) -> ResolvedFrame:
        """One frame, through the tiers in order."""
        if is_mangled(frame.fn_name) and frame.file_path is None:
            metrics.incr("trace.mangled_skipped")
            return ResolvedFrame(frame, None, "unresolved", 0.0)

        direct = self._tier0(repo_id, frame)
        if direct is not None:
            return direct

        return self._tier2(repo_id, frame)

    def resolve_all(self, repo_id: str, frames: Sequence[StackFrame]) -> list[ResolvedFrame]:
        return [self.resolve(repo_id, f) for f in frames]

    # -- Tier 0 ------------------------------------------------------------

    def _tier0(self, repo_id: str, frame: StackFrame) -> ResolvedFrame | None:
        """Direct: an exact file and line. Deterministic, so confidence is 1.0.

        §8.3 scopes the TRACE-resolution decision metric to TS/TSX precisely
        because Python resolves near-100% here, and a blended rate would mask
        the failure the sourcemap tier is meant to address.
        """
        if frame.file_path is None or frame.line_no is None:
            return None

        rel_path = normalize_path(frame.file_path, self.repo_root)
        found = self.reader.symbol_at_line(repo_id, rel_path, frame.line_no)
        if found is None:
            return None

        metrics.incr("trace.tier0.resolved")
        metrics.incr("frames_resolved_by_tier.direct")
        return ResolvedFrame(frame, found.uid, "direct", 1.0)

    # -- Tier 2 ------------------------------------------------------------

    def _tier2(self, repo_id: str, frame: StackFrame) -> ResolvedFrame:
        """Name match, with ambiguity as refusal. §5.1, T10.

        The counters are the point as much as the return value: §8.3 counts
        `tier2.ambiguous` as *unresolved* when deciding whether to build the
        sourcemap tier, so a refusal that quietly returned a guess would also
        hide the evidence for fixing it properly.
        """
        if is_mangled(frame.fn_name):
            metrics.incr("trace.mangled_skipped")
            return ResolvedFrame(frame, None, "unresolved", 0.0)

        cands = list(self.reader.symbols_named(repo_id, frame.fn_name))
        if len(cands) == 1:
            metrics.incr("trace.tier2.resolved")
            metrics.incr("frames_resolved_by_tier.name")
            return ResolvedFrame(frame, cands[0].uid, "name", 0.6)

        metrics.incr("trace.tier2.ambiguous" if cands else "trace.tier2.miss")
        return ResolvedFrame(frame, None, "unresolved", 0.0)


# --------------------------------------------------------------------------
# Tier 3
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Tier3Query:
    """What an unresolved trace falls back to (§5.1 Tier 3).

    "Re-route on the error message plus any resolvable neighbouring frame."
    Both halves matter: the message alone loses the location, and the
    neighbours alone lose what went wrong.
    """

    text: str
    resolved_neighbors: tuple[str, ...]


def build_tier3_query(
    trace_text: str, resolved: Sequence[ResolvedFrame]
) -> Tier3Query:
    message = error_message(trace_text)
    neighbors = tuple(r.uid for r in resolved if r.uid is not None)
    names = [
        r.frame.fn_name
        for r in resolved
        if r.uid is None and r.frame.fn_name and not is_mangled(r.frame.fn_name)
    ]
    text = " ".join(filter(None, [message, *names]))
    metrics.incr("trace.tier3.fallback")
    return Tier3Query(text=text, resolved_neighbors=neighbors)
