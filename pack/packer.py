"""§5.4 — the context packer.

`CTX_BUDGET` sits far below the model's window deliberately: long contexts
degrade retrieval precision, inflate cost, and raise TTFT. It is a quality
parameter, not a capacity limit.

Two accounting rules carry the defects this module exists to prevent:

* **v9.1 #12** — every branch that appends to `packed` must also add its cost to
  `used`, the truncation branch included. Miss one and the budget is breached by
  exactly the size of the block nobody counted.
* **Handles are assigned after packing.** Pre-assignment leaves gaps in the
  `C1..Cn` sequence for blocks that were dropped, and the model cites into
  them — which T1 then reports as a hallucination, for a mistake the packer
  made.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import metrics
from config import AVAILABLE, MAX_SEED_FRAC, SEED_SHARE
from index.chunker import Tokenizer
from pack.render import Block, render, truncate_marked


@dataclass
class Packed:
    block: Block
    level: str                              # full | elided | stub | truncated
    cost: int
    text: str
    handle: str | None = None

    @property
    def uid(self) -> str:
        return self.block.uid


@dataclass
class PackedContext:
    blocks: list[Packed] = field(default_factory=list)
    dropped: list[Block] = field(default_factory=list)
    packed_tokens: int = 0
    seeds_total: int = 0
    budget: int = AVAILABLE

    @property
    def handles(self) -> dict[str, str]:
        """`{C1: uid, ...}` — the map §6.1 says the server holds.

        T1 is a dict lookup against this, which is the whole reason citation
        validation costs nothing during the stream.
        """
        return {p.handle: p.uid for p in self.blocks if p.handle}

    def render(self) -> str:
        return "\n\n".join(f"[{p.handle}]\n{p.text}" for p in self.blocks)

    def report(self) -> str:
        """§5.4's packing report, surfaced in the UI.

        "A developer who sees '2 omitted' knows to narrow the question; one who
        sees nothing assumes completeness."
        """
        full = sum(1 for p in self.blocks if p.level == "full")
        elided = sum(1 for p in self.blocks if p.level == "elided")
        stubs = sum(1 for p in self.blocks if p.level == "stub")
        truncated = sum(1 for p in self.blocks if p.level == "truncated")
        total = len(self.blocks) + len(self.dropped)

        parts = [f"{full} of {total} results in full"]
        if elided:
            parts.append(f"{elided} elided")
        if truncated:
            parts.append(f"{truncated} truncated")
        if stubs:
            parts.append(f"{stubs} as signatures")
        if self.dropped:
            parts.append(f"{len(self.dropped)} omitted")
        return (
            "Context: "
            + ", ".join(parts)
            + f" ({self.packed_tokens / 1000:.1f}k / {self.budget / 1000:.1f}k tokens)"
        )

    @property
    def drop_rate(self) -> float:
        """§8.2's context drop rate, gated at <= 0.10."""
        total = len(self.blocks) + len(self.dropped)
        return len(self.dropped) / total if total else 0.0


def pack(
    seeds: Sequence[Block],
    neighbors: Sequence[Block],
    tok: Tokenizer,
    *,
    available: int = AVAILABLE,
    seed_share: float = SEED_SHARE,
    max_seed_frac: float = MAX_SEED_FRAC,
) -> PackedContext:
    """§5.4's `pack`, with the accounting made explicit.

    Greedy by score. A seed is tried at L0 then L1; if neither fits it is
    dropped — unless nothing has been packed at all, in which case the top seed
    is truncated with a marker rather than leaving the model with no context
    (Principle 3).
    """
    seed_cap = int(available * seed_share)
    nbr_cap = available - seed_cap
    per_seed_cap = int(seed_cap * max_seed_frac)

    ctx = PackedContext(seeds_total=len(seeds), budget=available)
    used = 0

    for block in sorted(seeds, key=lambda b: (-b.score, b.uid)):
        for level in ("full", "elided"):
            text = render(block, level)
            cost = tok.count(text)
            if cost <= per_seed_cap and used + cost <= seed_cap:
                ctx.blocks.append(Packed(block, level, cost, text))
                used += cost
                break
        else:
            if not ctx.blocks:
                # Pathological top seed: truncate at a statement boundary with a
                # loud marker rather than erroring.
                text, cost = truncate_marked(block, per_seed_cap, tok)
                ctx.blocks.append(Packed(block, "truncated", cost, text))
                used += cost                # v9.1 #12: this line is the defect
                metrics.incr("pack.seed_truncated")
            else:
                ctx.dropped.append(block)
                metrics.incr("pack.seed_l1_overflow")

    metrics.incr("pack.seeds_total", len(seeds))

    used += _fill_stubs(neighbors, nbr_cap, tok, ctx)

    # Handles are assigned here, after packing, so C1..Cn name only blocks that
    # are actually present.
    for i, packed in enumerate(ctx.blocks, start=1):
        packed.handle = f"C{i}"

    ctx.packed_tokens = used
    return ctx


def _fill_stubs(
    neighbors: Sequence[Block], nbr_cap: int, tok: Tokenizer, ctx: PackedContext
) -> int:
    """Fill the neighbour budget with L3 stubs. Returns the tokens used.

    Neighbours are rendered at stub level only. §5.2 step 3 brings them in for
    structure — who calls this, what it dispatches to — and a signature carries
    that. Their bodies would spend the seed budget's worth of tokens on context
    the question did not ask for.

    §8.3 is explicit that a neighbour dropped here is *not* a seed L1-overflow:
    "drop rate includes neighbors dropped at stub level, which L2 cannot help."
    """
    used = 0
    seen = {p.uid for p in ctx.blocks}

    for block in sorted(neighbors, key=lambda b: (-b.score, b.uid)):
        if block.uid in seen:
            continue                        # already packed as a seed
        text = render(block, "stub")
        cost = tok.count(text)
        if used + cost > nbr_cap:
            ctx.dropped.append(block)
            metrics.incr("pack.neighbor_dropped")
            continue
        ctx.blocks.append(Packed(block, "stub", cost, text))
        seen.add(block.uid)
        used += cost

    return used
