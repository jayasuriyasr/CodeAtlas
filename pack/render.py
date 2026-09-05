"""§5.4's rendering ladder: L0 full, L1 elided, L3 stub.

| Level | Rendering | Cost |
|---|---|---|
| L0 full | header + complete body | 100% |
| L1 elided | header + body, nested function bodies -> `# ⟨N lines elided⟩` | 30-60% |
| L3 stub | header + signature + first docstring line | ~5% |

L2 (a control-flow spine) is deferred: it needs per-language statement
classification and sits between two levels that already work.

**Elision markers are never silent.** §5.4 is explicit about why: they tell the
model content is missing so it declines to cite rather than inferring. Silent
truncation manufactures exactly the confident-uncited claim T2 exists to catch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from adapters.base import Symbol
from adapters.grammars import python_parser, walk
from index.chunker import Tokenizer, build_context_header

#: §5.4's marker. The angle brackets are the spec's; keeping them literal means
#: the prompt's instruction ("Content marked ⟨N lines elided⟩ is not available
#: to you") matches what the model actually sees.
ELISION = "# ⟨{n} lines elided⟩"
TRUNCATION = "# ⟨truncated: {shown} of {total} lines shown⟩"

_DEF_NODES = ("function_definition", "class_definition")
_BLANK = re.compile(r"^\s*$")


@dataclass(frozen=True)
class Block:
    """A retrieved symbol plus the score that ordered it."""

    symbol: Symbol
    score: float

    @property
    def uid(self) -> str:
        return self.symbol.uid


# --------------------------------------------------------------------------
# L1 — nested-body elision
# --------------------------------------------------------------------------


def elide_nested_bodies(source: str) -> tuple[str, int]:
    """Replace nested definition bodies with a marker. Returns (text, count).

    §5.4 keeps L1 because "pure greedy-truncate spends the whole seed budget on
    one 4,000-token generated file", and because it is "a walk over tree-sitter
    nodes already in hand" rather than new machinery.

    Only the outermost nested definitions are elided — a definition inside an
    already-elided body is gone with it, and eliding it separately would corrupt
    the offsets.
    """
    if not source.strip():
        return source, 0

    data = source.encode("utf-8")
    root = python_parser().parse(data).root_node

    def def_ancestor_count(node) -> int:
        count, cur = 0, node.parent
        while cur is not None:
            if cur.type in _DEF_NODES:
                count += 1
            cur = cur.parent
        return count

    nested = [
        node
        for node in walk(root)
        if node.type in _DEF_NODES and def_ancestor_count(node) >= 1
    ]
    # Keep only the outermost: drop any node contained in another candidate.
    spans = [(n.start_byte, n.end_byte) for n in nested]
    outermost = [
        n
        for n in nested
        if not any(
            lo < n.start_byte and n.end_byte <= hi for lo, hi in spans
        )
    ]

    replacements = []
    for node in outermost:
        body = node.child_by_field_name("body")
        if body is None:
            continue
        body_text = data[body.start_byte : body.end_byte].decode("utf-8", errors="replace")
        line_count = len(body_text.splitlines())
        if line_count <= 1:
            continue                       # a one-liner costs more to mark than to keep
        replacements.append((body.start_byte, body.end_byte, line_count))

    if not replacements:
        return source, 0

    out = data
    for start, end, line_count in sorted(replacements, reverse=True):
        marker = ELISION.format(n=line_count).encode("utf-8")
        out = out[:start] + marker + out[end:]

    return out.decode("utf-8", errors="replace"), len(replacements)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def render(block: Block, level: str) -> str:
    """Render one block at one level. The header is always present.

    The header carries the path, the enclosing signature and the symbol's own
    imports (§3.4.2), which is what lets the model say *where* a claim came from
    even when the body has been elided down to nothing.
    """
    symbol = block.symbol
    header = build_context_header(symbol)

    if level == "full":
        return f"{header}\n{symbol.source_code}"

    if level == "elided":
        body, count = elide_nested_bodies(symbol.source_code)
        if count == 0:
            return f"{header}\n{body}"     # nothing nested to elide
        return f"{header}\n{body}"

    if level == "stub":
        lines = [header, symbol.signature]
        if symbol.docstring:
            first = symbol.docstring.strip().splitlines()[0].strip()
            if first:
                lines.append(f'    """{first}"""')
        return "\n".join(lines)

    raise ValueError(f"unknown render level {level!r}")


# --------------------------------------------------------------------------
# Truncation — the pathological top seed
# --------------------------------------------------------------------------


def truncate_marked(block: Block, budget: int, tok: Tokenizer) -> tuple[str, int]:
    """Cut at a statement boundary with a loud marker. Returns (text, cost).

    §5.4 reaches here only for a pathological *top* seed — a single block too
    large even at L1, with nothing else packed to fall back on. Principle 3 says
    degrade rather than error, so it is cut rather than dropped.

    The marker's own cost is reserved before any content is added, so the
    returned cost is <= budget even in the degenerate case where nothing fits.
    """
    header = build_context_header(block.symbol)
    body, _ = elide_nested_bodies(block.symbol.source_code)
    lines = body.split("\n")

    marker_cost = tok.count(TRUNCATION.format(shown=len(lines), total=len(lines))) + 1
    header_cost = tok.count(header) + 1
    room = budget - header_cost - marker_cost

    # Statement boundaries, measured over the body after its first line — the
    # same rule the chunker's overflow split uses, and for the same reason: a
    # cut inside an `if` body reads as a different program than the one there.
    indents = [
        len(line) - len(line.lstrip()) for line in lines[1:] if not _BLANK.match(line)
    ]
    statement_indent = min(indents) if indents else 0
    boundaries = {
        i
        for i, line in enumerate(lines)
        if i > 0
        and not _BLANK.match(line)
        and (len(line) - len(line.lstrip())) == statement_indent
    }

    kept: list[str] = []
    used = 0
    for i, line in enumerate(lines):
        cost = tok.count(line) + 1
        if used + cost > room:
            break
        kept.append(line)
        used += cost

    # Retreat to the last statement boundary so the cut is not mid-block.
    while kept and len(kept) not in boundaries and len(kept) < len(lines):
        kept.pop()
    if not kept:
        kept = lines[:1]                   # the signature line, at minimum

    text = "\n".join(
        [header, *kept, TRUNCATION.format(shown=len(kept), total=len(lines))]
    )
    return text, tok.count(text)
