"""The S3 seam: what every language adapter produces, and the UID contract.

`symbol_uid` is reproduced verbatim from spec §3.2 and must stay that way — it
is the identity every edge in the graph references, so a change to it is a
change to the meaning of every stored relationship, not a refactor.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


# --------------------------------------------------------------------------
# Identity (spec §3.2)
# --------------------------------------------------------------------------


def symbol_uid(
    repo_id: str,
    rel_path: str,
    qualified_name: str,
    arity: int,
    ordinal: int = 0,
) -> str:
    """Spec §3.2, verbatim.

    Excluded from the UID by design: line numbers, because they churn on every
    edit above the symbol, and body content, because a symbol whose body changes
    is the same symbol.
    """
    return hashlib.sha1(
        f"{repo_id}\x00{rel_path}\x00{qualified_name}\x00{arity}\x00{ordinal}".encode()
    ).hexdigest()[:20]


def normalize_rel_path(rel_path: str) -> str:
    """Force a repo-relative path to POSIX form before it reaches a UID.

    `rel_path` is a UID input, so `auth\\views.py` and `auth/views.py` would
    otherwise hash to different symbols — the whole graph would fork the first
    time it was indexed from Windows and read from CI. Normalising at the
    boundary keeps `symbol_uid` itself verbatim to the spec.

    Leading `./` and `/` are removed as *prefixes*, never as a character set.
    `str.lstrip("./")` strips any run of dots and slashes, which turned
    `.github/scripts/x.py` into `github/scripts/x.py`: a path that does not
    exist, recorded on every symbol while the `:File` node kept the real one,
    so §11.2 steps 4-6 never scoped onto those symbols again.
    """
    path = rel_path.replace("\\", "/")
    while True:
        if path.startswith("./"):
            path = path[2:]
        elif path.startswith("/"):
            path = path[1:]
        else:
            return path


def content_hash(source: bytes) -> str:
    """Hash of a file's content, for `:File.content_hash` (§3.3, §4.5).

    Line endings are normalised first: a repo checked out with `core.autocrlf`
    would otherwise report every file modified against an index built on Linux,
    and §4.5's reconcile would re-embed the entire tree.
    """
    text = source.decode("utf-8", errors="replace").replace("\r\n", "\n")
    return hashlib.sha256(text.encode()).hexdigest()


# --------------------------------------------------------------------------
# The parse product (spec §3.3, §4.2 T6)
# --------------------------------------------------------------------------


@dataclass
class Symbol:
    """One indexable symbol. Fields mirror §3.3's property list.

    §4.2 T6: `ParsedFile.symbols` is the only collection — a symbol carries its
    own `chunk_text` rather than there being a parallel `chunks` list. The
    step-4 fields below are declared here and populated by the chunker, so the
    data model is defined once rather than grown.
    """

    # Identity — everything the UID is computed from, plus the UID itself.
    uid: str
    repo_id: str
    rel_path: str
    qualified_name: str
    name: str
    arity: int
    ordinal: int
    kind: str                                   # "function" | "class" | "module"

    # Content.
    signature: str
    docstring: str | None
    source_code: str
    enclosing_signature: str | None             # full, incl. bases (§3.3)
    used_imports: list[str] = field(default_factory=list)   # per-symbol (§3.4)
    decorators: list[str] = field(default_factory=list)
    #: A class's declared base classes, in declaration order, as absolute
    #: qualified names where the import table can say so (`models.Model` ->
    #: `django.db.models.Model`). Empty for anything that is not a class. The
    #: writer walks these to resolve `self.method()` to an inherited method,
    #: and declaration order is what Python's MRO is computed from.
    bases: list[str] = field(default_factory=list)

    # Position. Mutable properties, deliberately not UID inputs (§3.2).
    start_line: int = 0
    end_line: int = 0

    # Framework layer (§3.5). False for every Python symbol.
    is_client_component: bool = False

    # Populated by the step-4 chunker, not the adapter.
    chunk_text: str | None = None
    header_hash: str | None = None
    body_hash: str | None = None
    search_text: str | None = None
    code_vec: list[float] | None = None

    # origin_path is set by the writer (§11.2 3a), not the parser.

    @property
    def identity(self) -> tuple[str, int]:
        """The pair that ordinals disambiguate (§3.2 T1)."""
        return (self.qualified_name, self.arity)


@dataclass(frozen=True)
class Edge:
    """A relationship between two symbols.

    `target_uid` is populated only when the target resolves inside the same
    file. Cross-file targets carry `target_hint` — a qualified name — and are
    resolved against the graph later. §11.2 step 3b matches both endpoints by
    uid, so an edge that never resolves is simply never written, which is the
    correct outcome for a call into a third-party library.
    """

    source_uid: str
    kind: str                                   # from config.EDGE_TYPE_ALLOWLIST
    origin_path: str
    target_uid: str | None = None
    target_hint: str | None = None
    #: A member lookup: `target_hint` is `Class.method` for a call made through
    #: `self`, `cls` or `this`, and the method may be defined on the class or
    #: inherited from one of its bases. The writer resolves it by walking the
    #: class's method resolution order, and refuses when that walk is uncertain.
    through_bases: bool = False
    #: With `through_bases`: a `super()` call, so the class itself is skipped
    #: and the lookup starts at the next class in its resolution order.
    after_owner: bool = False


def edge_target_name(edge: Edge, by_qname: dict[str, Symbol]) -> str:
    """The qualified name an edge points at: the resolved symbol's, or its hint."""
    if edge.target_uid is not None:
        for sym in by_qname.values():
            if sym.uid == edge.target_uid:
                return sym.qualified_name
    return edge.target_hint or ""


def member_call_edge(
    sym: Symbol, method: str, by_qname: dict[str, Symbol], *, via_super: bool = False
) -> Edge | None:
    """The edge for `self.method()` / `this.method()` made inside `sym`, or None.

    Shared by both adapters; each one decides what counts as a receiver and
    hands over the method name. `sym` must be a method — its parent a class —
    or `self` is just a parameter name and this returns None.

    A method of the enclosing class defined in the same file resolves now. Any
    other name may be inherited from a base in another file, so it becomes a
    member lookup (`through_bases=True`) for the writer to resolve against the
    class's method resolution order.
    """
    owner_qname = sym.qualified_name.rsplit(".", 1)[0]
    owner = by_qname.get(owner_qname)
    if owner is None or owner.kind != "class":
        return None
    member = f"{owner_qname}.{method}"
    if via_super:
        # Never the class's own method: that is the point of `super()`.
        return Edge(source_uid=sym.uid, kind="CALLS", origin_path=sym.rel_path,
                    target_hint=member, through_bases=True, after_owner=True)
    if member in by_qname:
        return Edge(source_uid=sym.uid, kind="CALLS", origin_path=sym.rel_path,
                    target_uid=by_qname[member].uid)
    return Edge(source_uid=sym.uid, kind="CALLS", origin_path=sym.rel_path,
                target_hint=member, through_bases=True)


@dataclass
class ParsedFile:
    """What `LanguageAdapter.parse` returns (plan §2).

    `source` is the raw bytes this file was parsed from, required by §3.4 since
    v10.1. A move needs the *same bytes* re-parsed at the old path to recover the
    UIDs the graph actually stored: `qualified_name` includes the module path
    (§3.2), so it does not survive a move. v10.0 assumed it did, which made every
    remap a silent no-op (v10.1 §0.0 R1, FINDINGS F-007).

    Holding the bytes costs little next to `Symbol.source_code`, which already
    stores the text of every symbol in the file.
    """

    rel_path: str
    content_hash: str
    symbols: list[Symbol] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    language: str = ""
    source: bytes = b""

    @property
    def uids(self) -> list[str]:
        return [s.uid for s in self.symbols]


@dataclass(frozen=True)
class StackFrame:
    """One line of a pasted stack trace, before resolution (§5.1)."""

    raw: str
    file_path: str | None
    line_no: int | None
    fn_name: str | None


@dataclass(frozen=True)
class ResolvedFrame:
    uid: str
    tier: str                                   # "direct" | "name" | "semantic"
    confidence: float


@runtime_checkable
class LanguageAdapter(Protocol):
    """S3. One implementation per language; `parse` is the whole contract."""

    name: str
    extensions: tuple[str, ...]

    def parse(self, repo_id: str, rel_path: str, source: bytes) -> ParsedFile: ...

    def resolve_frame(self, frame: StackFrame) -> StackFrame: ...
