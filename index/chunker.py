"""Chunk construction (§3.4), secret scrubbing (§9.2), cache keys (§4.2).

One rule drives the whole module and is worth stating before the code:

> The `imports used` line is built by walking the symbol's own AST ... only what
> this symbol references. Populating it from the file's import block instead
> makes the header a function of the whole file, so one added import
> invalidates every cached embedding in it. — §3.4

The adapter already resolves imports per symbol (`Symbol.used_imports`). This
module's job is to keep that property intact all the way into `header_hash`,
and to make sure §9.2's redaction happens *before* the hash rather than after.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import metrics
from adapters.base import ParsedFile, Symbol
from config import CHUNK_OVERFLOW_TOKENS


class Tokenizer(Protocol):
    """S2's `.tokenize()` half, narrowed to what the chunker and packer need.

    §5.4 is explicit: use the provider's real tokenizer, because `len // 4` is
    off by up to 2x on dense code. Nothing in this repository divides a length
    by four; the count always comes through this interface.
    """

    name: str

    def count(self, text: str) -> int: ...


# --------------------------------------------------------------------------
# §9.2 — secret scrubbing, before anything is hashed, embedded, or prompted
# --------------------------------------------------------------------------

REDACTED = "«redacted»"

#: Known key shapes. Deliberately narrow: §12.0 is honest that this "will not
#: catch a credential shaped like ordinary text, one assembled from parts, or a
#: proprietary algorithm". Widening the patterns does not change that, and a
#: pattern that fires on ordinary code costs retrieval signal for nothing.
SCRUB_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("stripe_key", re.compile(r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{16,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    (
        "private_key",
        re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----.*?"
            r"-----END (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----",
            re.DOTALL,
        ),
    ),
    ("high_entropy_literal", re.compile(r"""(?<=['"])[A-Za-z0-9+/=_-]{33,}(?=['"])""")),
)

#: Bumped when SCRUB_RULES changes. §9.2: "Redaction precedes hashing so rule
#: changes invalidate the cache" — that only holds if a rule change actually
#: alters the hashed text, which it does not when a *new* rule matches nothing
#: in a given symbol. Folding the version into the header closes that gap.
SCRUB_RULES_VERSION = "1"


def scrub_secrets(text: str) -> str:
    """Redact known key formats and high-entropy literals (§9.2).

    Applied at chunk construction, before embedding and before prompting; the
    unredacted form is never stored on the node.
    """
    for name, pattern in SCRUB_RULES:
        text, count = pattern.subn(f"{REDACTED}:{name}", text)
        if count:
            metrics.incr("scrub.redactions", count)
    return text


# --------------------------------------------------------------------------
# §4.2 — the cache key
# --------------------------------------------------------------------------


def _normalize(src: str) -> str:
    """§4.2, verbatim.

    Comments are retained: they carry retrieval signal. Secret scrubbing (§9.2)
    runs before this, so scrub-rule changes invalidate correctly.
    """
    return "\n".join(
        line.rstrip() for line in src.replace("\r\n", "\n").split("\n")
    ).strip()


def body_hash(source_code: str) -> str:
    return hashlib.sha256(_normalize(source_code).encode()).hexdigest()[:20]


def header_hash(sym: Symbol) -> str:
    """§4.2's header component.

    Every element is something the context header actually shows (§3.4.2), which
    is the invariant that makes the split meaningful: if a field is in the
    header hash but not in the header text, a change to it invalidates a cache
    entry whose content did not change.
    """
    return hashlib.sha256(
        "\x00".join(
            [
                sym.rel_path,
                sym.enclosing_signature or "",
                sym.signature,
                *sorted(sym.used_imports),
                SCRUB_RULES_VERSION,
            ]
        ).encode()
    ).hexdigest()[:12]


def cache_key(sym: Symbol) -> str:
    """§4.2. `header_hash:body_hash` — the composite key that is the whole
    structural cost control (§9.5)."""
    return f"{header_hash(sym)}:{body_hash(sym.source_code)}"


# --------------------------------------------------------------------------
# §3.4 — the context header and the chunk
# --------------------------------------------------------------------------


def build_context_header(sym: Symbol) -> str:
    """§3.4.2's two lines.

        # repo/auth/views.py :: class LoginView(APIView) :: def post(self, request) -> Response
        # imports used: rest_framework.Response, .models.User, .serializers.LoginSerializer

    The second line is omitted entirely when the symbol references no imports —
    an empty `imports used:` would be a constant string in every such header,
    which is noise in the embedding for no signal.
    """
    parts = [sym.rel_path]
    if sym.enclosing_signature:
        parts.append(sym.enclosing_signature.rstrip(":"))
    parts.append(sym.signature.rstrip(":"))

    lines = ["# " + " :: ".join(parts)]
    if sym.used_imports:
        lines.append("# imports used: " + ", ".join(sorted(sym.used_imports)))
    return "\n".join(lines)


def class_chunk_body(sym: Symbol, members: list[Symbol]) -> str:
    """§3.4.4's class chunk: signature, docstring, method signatures."""
    lines = [sym.signature]
    if sym.docstring:
        lines.append(f'    """{sym.docstring}"""')
    for member in members:
        lines.append(f"    {member.signature}")
    return "\n".join(lines)


def module_chunk_body(sym: Symbol, top_level: list[Symbol]) -> str:
    """§3.4.5's file chunk: import block plus the top-level symbol list."""
    lines = []
    if sym.used_imports:
        lines.extend(f"import {name}" for name in sorted(sym.used_imports))
        lines.append("")
    lines.extend(member.signature for member in top_level)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# §3.4.3 — overflow split
# --------------------------------------------------------------------------

#: A line that starts a new top-level statement inside a body: no leading
#: whitespace beyond the body's own indent, and not a continuation.
_BLANK = re.compile(r"^\s*$")


def split_overflow(
    body: str,
    header: str,
    tokenizer: Tokenizer,
    cap: int = CHUNK_OVERFLOW_TOKENS,
) -> list[str]:
    """Split an oversized body on top-level statement boundaries, header repeated.

    "Top level" is relative to the body: the least-indented non-blank line is
    the statement indent, and a split may only happen immediately before a line
    at that indent. Splitting anywhere else lands mid-block, and a chunk that
    begins inside an `if` body reads as a different program than the one that is
    there.

    Returns one part when the body fits; the caller decides what to do with more
    than one — see FINDINGS.md F-005, because §3.3 gives a symbol exactly one
    `code_vec`.
    """
    whole = f"{header}\n{body}"
    if tokenizer.count(whole) <= cap:
        return [whole]

    lines = body.split("\n")

    # The statement indent is measured over the body *after* its first line,
    # because `sym.source_code` for a function starts with the signature at
    # indent 0. Including it would make 0 the minimum, leave the signature as
    # the only line at that indent, and — since a split before line 0 is
    # meaningless — produce no boundaries at all and no split.
    indents = [
        len(line) - len(line.lstrip())
        for line in lines[1:]
        if not _BLANK.match(line)
    ]
    if not indents:
        return [whole]                    # a one-line body cannot be split
    statement_indent = min(indents)

    boundaries = {
        i
        for i, line in enumerate(lines)
        if i > 0
        and not _BLANK.match(line)
        and (len(line) - len(line.lstrip())) == statement_indent
    }

    # Costs are accumulated per line rather than by re-counting the joined text
    # on every iteration, which would be quadratic in a long body. One token is
    # added per newline, matching what the tokenizer charges for the join.
    header_cost = tokenizer.count(header) + 1
    parts: list[str] = []
    current: list[str] = []
    running = header_cost

    for i, line in enumerate(lines):
        line_cost = tokenizer.count(line) + 1
        if running + line_cost > cap and current and i in boundaries:
            parts.append(f"{header}\n" + "\n".join(current))
            current = [line]
            running = header_cost + line_cost
        else:
            current.append(line)
            running += line_cost

    if current:
        parts.append(f"{header}\n" + "\n".join(current))

    if len(parts) > 1:
        metrics.incr("chunk.overflow_split")
        metrics.incr("chunk.overflow_parts", len(parts) - 1)
    return parts


# --------------------------------------------------------------------------
# Driving it over a parsed file
# --------------------------------------------------------------------------


def chunk_file(parsed: ParsedFile, tokenizer: Tokenizer) -> ParsedFile:
    """Populate `chunk_text`, `header_hash` and `body_hash` on every symbol.

    Ordering matters and is the point of §9.2's last sentence: scrub, then
    hash, then build the chunk. Hashing first would let a scrub-rule change slip
    past the cache, and the stale embedding would be of text that still
    contained the secret.
    """
    members_by_parent: dict[str, list[Symbol]] = {}
    for sym in parsed.symbols:
        parent = sym.qualified_name.rsplit(".", 1)[0]
        members_by_parent.setdefault(parent, []).append(sym)

    for sym in parsed.symbols:
        sym.source_code = scrub_secrets(sym.source_code)
        if sym.docstring:
            sym.docstring = scrub_secrets(sym.docstring)

        members = members_by_parent.get(sym.qualified_name, [])
        if sym.kind == "class":
            body = class_chunk_body(sym, members)
        elif sym.kind == "module":
            body = module_chunk_body(sym, members)
        else:
            body = sym.source_code

        sym.header_hash = header_hash(sym)
        sym.body_hash = body_hash(sym.source_code)

        header = build_context_header(sym)
        parts = split_overflow(body, header, tokenizer)
        sym.chunk_text = parts[0]

    return parsed
