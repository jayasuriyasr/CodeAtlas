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
    """
    return rel_path.replace("\\", "/").lstrip("./")


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
