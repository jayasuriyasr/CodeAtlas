"""Step 8 — the context packer.

The defect that names this step is v9.1 #12: a branch that appends to `packed`
without adding its cost to `used`. It does not raise. The budget is simply
breached by exactly the size of the block nobody counted, and the first symptom
is a provider error or a bill.

No database. Every number here is produced by the real tokenizer seam — §5.4
forbids `len // 4`, and `test_no_length_heuristic` (step 4) enforces that across
the whole source tree.
"""

from __future__ import annotations

import random

import pytest

import metrics
from adapters.base import Symbol
from config import AVAILABLE, MAX_SEED_FRAC, SEED_SHARE
from index.providers import ApproxCodeTokenizer
from pack.packer import PackedContext, pack
from pack.render import ELISION, TRUNCATION, Block, elide_nested_bodies, render


@pytest.fixture(autouse=True)
def reset_metrics():
    metrics.reset()
    yield
    metrics.reset()


@pytest.fixture(scope="module")
def tok() -> ApproxCodeTokenizer:
    return ApproxCodeTokenizer()


def make_symbol(
    name: str, body_lines: int = 3, *, docstring: str | None = None, nested: int = 0
) -> Symbol:
    """A synthetic symbol.

    `nested` adds inner function definitions, which is what gives L1 something
    to elide. A flat body has no nested definitions, so `render(..., "elided")`
    returns the same text as `"full"` — correct behaviour, and the reason the
    fuzz needs both shapes to reach all four levels.
    """
    lines: list[str] = []
    for i in range(nested):
        lines.append(f"    def helper_{i}(value):")
        lines.extend(f"        step_{i}_{j} = value + {j}" for j in range(8))
        lines.append(f"        return step_{i}_0")
    lines.extend(f"    value_{i} = compute_{i}(payload)" for i in range(body_lines))
    source = f"def {name}(payload):\n" + "\n".join(lines) + "\n"
    return Symbol(
        uid=f"uid_{name}",
        repo_id="r",
        rel_path=f"pkg/{name}.py",
        qualified_name=f"pkg.{name}.{name}",
        name=name,
        arity=1,
        ordinal=0,
        kind="function",
        signature=f"def {name}(payload):",
        docstring=docstring,
        source_code=source,
        enclosing_signature=None,
        used_imports=["pkg.helpers.compute"],
    )


def block(name: str, score: float, body_lines: int = 3, **kw) -> Block:
    return Block(symbol=make_symbol(name, body_lines, **kw), score=score)


def fuzz_blocks(rng, prefix: str, count: int, max_lines: int) -> list[Block]:
    """Random blocks, half of them with nested definitions.

    Both shapes are needed: L1 can only save budget on a body that has
    nested definitions to elide, so a fuzz built only from flat functions
    never reaches the elided branch at all.
    """
    return [
        block(
            f"{prefix}{i}",
            score=rng.random(),
            body_lines=rng.randint(1, max_lines),
            nested=rng.choice([0, 0, 1, 3, 8]),
        )
        for i in range(count)
    ]


NESTED = '''\
def outer(payload):
    """Outer docstring."""
    total = 0

    def inner(value):
        scaled = value * 2
        adjusted = scaled + 1
        return adjusted

    for item in payload:
        total += inner(item)
    return total
'''


# --------------------------------------------------------------------------
# The budget — v9.1 #12
# --------------------------------------------------------------------------


def test_budget_never_exceeded(tok):
    """v9.1 #12, across every branch including truncation.

    The seeds here are sized so all three paths fire in one call: some fit at
    L0, some only at L1, one is dropped, and the accounting must still close.
    """
    seeds = [block(f"s{i}", score=1.0 - i / 100, body_lines=40 * (i + 1)) for i in range(8)]
    neighbors = [block(f"n{i}", score=0.5 - i / 100) for i in range(20)]

    ctx = pack(seeds, neighbors, tok, available=4_000)

    assert ctx.packed_tokens <= 4_000, (
        f"budget breached: {ctx.packed_tokens} > 4000"
    )
    assert ctx.packed_tokens == sum(p.cost for p in ctx.blocks), (
        "the running total disagrees with the blocks actually packed"
    )


def test_truncation_branch_is_counted(tok):
    """The specific line v9.1 #12 names: `used += t.cost` in the truncate path.

    Reached only when the *top* seed is too large at every level and nothing has
    been packed yet — so a missing `+=` here is invisible except on exactly the
    pathological input this test builds.
    """
    huge = block("huge", score=1.0, body_lines=4_000)
    ctx = pack([huge], [], tok, available=4_000)

    assert [p.level for p in ctx.blocks] == ["truncated"]
    assert ctx.packed_tokens > 0, "the truncated block cost nothing — it was not counted"
    assert ctx.packed_tokens == ctx.blocks[0].cost
    assert metrics.get("pack.seed_truncated") == 1


def test_truncation_accounting(tok):
    """§5.4's stated case: a pathological top seed, then five more.

    The five that follow must be accounted against a budget the truncated block
    has already consumed from. If truncation were free, the packer would keep
    filling as though the first block were not there.
    """
    seeds = [block("huge", score=1.0, body_lines=4_000)] + [
        block(f"s{i}", score=0.5 - i / 100, body_lines=5) for i in range(5)
    ]
    ctx = pack(seeds, [], tok, available=4_000)

    assert ctx.blocks[0].level == "truncated"
    assert ctx.packed_tokens <= 4_000
    assert ctx.packed_tokens == sum(p.cost for p in ctx.blocks)


def test_no_single_seed_exceeds_max_seed_frac(tok):
    """† `MAX_SEED_FRAC` — no one seed takes more than 40% of the seed budget.

    Without the cap, one generated file consumes the whole seed share and every
    other result is dropped for it.
    """
    available = 4_000
    per_seed_cap = int(available * SEED_SHARE * MAX_SEED_FRAC)

    seeds = [block(f"s{i}", score=1.0 - i / 10, body_lines=30) for i in range(6)]
    ctx = pack(seeds, [], tok, available=available)

    for packed in ctx.blocks:
        assert packed.cost <= per_seed_cap, (
            f"{packed.uid} took {packed.cost} of a {per_seed_cap} per-seed cap"
        )


def test_budget_constants_match_the_spec():
    """§5.4's arithmetic, from `config.py` rather than a literal."""
    assert AVAILABLE == 21_200
    assert SEED_SHARE == 0.60
    assert MAX_SEED_FRAC == 0.40


# --------------------------------------------------------------------------
# Handles
# --------------------------------------------------------------------------


def test_handles_contiguous_after_drops(tok):
    """No gaps for the model to cite into.

    §5.4: handles are assigned after packing "so C1..Cn only name blocks
    actually present. Pre-assignment leaves gaps the model may cite into, which
    T1 then flags as hallucination" — blaming the model for the packer's error.
    """
    seeds = [block(f"s{i}", score=1.0 - i / 100, body_lines=60 * (i + 1)) for i in range(8)]
    ctx = pack(seeds, [], tok, available=4_000)

    assert ctx.dropped, "nothing was dropped; the test does not exercise the case"

    handles = [p.handle for p in ctx.blocks]
    assert handles == [f"C{i}" for i in range(1, len(ctx.blocks) + 1)]
    assert len(set(handles)) == len(handles)


def test_handle_map_covers_exactly_the_packed_blocks(tok):
    """§6.1: "The server holds `{C1: uid_a, …}`".

    T1 is a dict lookup against this map. A handle in the text with no entry
    here is reported as a hallucination; an entry with no text is a citation
    target the model never saw.
    """
    seeds = [block(f"s{i}", score=1.0 - i / 10) for i in range(4)]
    ctx = pack(seeds, [block("n1", 0.1)], tok, available=4_000)

    assert set(ctx.handles) == {p.handle for p in ctx.blocks}
    assert set(ctx.handles.values()) == {p.uid for p in ctx.blocks}
    for handle in ctx.handles:
        assert f"[{handle}]" in ctx.render()


def test_dropped_blocks_have_no_handle(tok):
    seeds = [block(f"s{i}", score=1.0 - i / 100, body_lines=60 * (i + 1)) for i in range(8)]
    ctx = pack(seeds, [], tok, available=4_000)

    packed_uids = {p.uid for p in ctx.blocks}
    assert all(b.uid not in packed_uids for b in ctx.dropped)


# --------------------------------------------------------------------------
# L1 elision
# --------------------------------------------------------------------------


def test_elision_marker_present(tok):
    """Never silent.

    §5.4: the marker tells the model content is missing "so it declines to cite
    rather than inferring. Silent truncation manufactures exactly the
    confident-uncited claim T2 exists to catch."
    """
    elided, count = elide_nested_bodies(NESTED)
    assert count == 1
    assert "⟨" in elided and "lines elided⟩" in elided
    assert "scaled = value * 2" not in elided, "the nested body survived"
    assert "def inner(value):" in elided, "the nested signature was lost too"
    assert "for item in payload:" in elided, "the outer body was elided as well"


def test_elision_preserves_indentation(tok):
    """A marker at the wrong indent turns valid code into a parse error.

    The model reads this as source; misaligned output invites it to infer
    structure that is not there.
    """
    elided, _ = elide_nested_bodies(NESTED)
    marker_line = next(line for line in elided.splitlines() if "elided⟩" in line)
    assert marker_line.startswith("        "), repr(marker_line)


def test_no_mid_line_split(tok):
    """Statement boundaries only — an elision never cuts inside a line."""
    elided, _ = elide_nested_bodies(NESTED)
    for line in elided.splitlines():
        if "elided⟩" in line:
            assert line.strip().startswith("#"), f"marker spliced into code: {line!r}"
            assert line.strip().endswith("⟩"), f"content trails the marker: {line!r}"


def test_l1_saves_budget(tok):
    """A large file elides to fit where L0 would not.

    §5.4 keeps L1 for exactly this: "pure greedy-truncate spends the whole seed
    budget on one 4,000-token generated file".
    """
    big = "def outer(payload):\n" + "\n".join(
        [
            "    def helper_%d(v):" % i
            + "\n"
            + "\n".join(f"        step_{i}_{j} = v + {j}" for j in range(20))
            for i in range(10)
        ]
    )
    sym = make_symbol("outer")
    sym.source_code = big
    blk = Block(symbol=sym, score=1.0)

    full_cost = tok.count(render(blk, "full"))
    elided_cost = tok.count(render(blk, "elided"))

    assert elided_cost < full_cost, "L1 saved nothing"
    assert elided_cost < full_cost * 0.7, (
        f"L1 saved only {1 - elided_cost / full_cost:.0%}; §5.4 expects 30-60% of L0"
    )


def test_elision_of_a_flat_function_changes_nothing(tok):
    """No nested definitions means no elision, not an empty body."""
    flat = "def f(a):\n    b = a + 1\n    return b\n"
    elided, count = elide_nested_bodies(flat)
    assert count == 0
    assert elided == flat


def test_elision_leaves_one_line_bodies_alone(tok):
    """A one-line body costs more to mark than to keep."""
    source = "def outer(a):\n    def inner(b):\n        return b\n    return inner(a)\n"
    _elided, count = elide_nested_bodies(source)
    assert count == 0


# --------------------------------------------------------------------------
# L3 stubs
# --------------------------------------------------------------------------


def test_stub_is_signature_plus_first_docstring_line(tok):
    blk = block("f", 1.0, docstring="Charge the card.\n\nLong explanation follows.")
    text = render(blk, "stub")

    assert "def f(payload):" in text
    assert "Charge the card." in text
    assert "Long explanation follows." not in text
    assert "value_0 = compute_0(payload)" not in text, "a body leaked into the stub"


def test_stub_is_far_cheaper_than_full(tok):
    """§5.4 prices L3 at ~5% of L0."""
    blk = block("f", 1.0, body_lines=60)
    assert tok.count(render(blk, "stub")) < tok.count(render(blk, "full")) * 0.4


def test_every_level_keeps_the_header(tok):
    """§3.4.2's header is what lets a claim be located even when the body is gone."""
    blk = block("f", 1.0, docstring="Doc.")
    for level in ("full", "elided", "stub"):
        text = render(blk, level)
        assert "pkg/f.py" in text, level
        assert "imports used" in text, level


def test_unknown_level_is_refused(tok):
    with pytest.raises(ValueError, match="unknown render level"):
        render(block("f", 1.0), "skeleton")


# --------------------------------------------------------------------------
# Counters — §8.3
# --------------------------------------------------------------------------


def test_counters_distinct(tok):
    """`seed_l1_overflow` and `seed_truncated` measure different decisions.

    §5.4: the first is a seed dropped after L1 was still too large, "which L2
    would fix"; the second is the rare pathological path, "which L2 would not".
    §8.3 divides only the first by `pack.seeds_total`, so conflating them would
    inflate the case for building L2.
    """
    seeds = [block("huge", score=1.0, body_lines=4_000)] + [
        block(f"big{i}", score=0.9 - i / 100, body_lines=400) for i in range(4)
    ]
    pack(seeds, [], tok, available=4_000)

    assert metrics.get("pack.seed_truncated") == 1
    assert metrics.get("pack.seed_l1_overflow") == 4
    assert metrics.get("pack.seeds_total") == 5


def test_neighbor_drops_are_not_seed_overflow(tok):
    """§8.3: "drop rate includes neighbors dropped at stub level, which L2
    cannot help." Counting them as seed overflow would point the trigger at the
    wrong fix."""
    neighbors = [block(f"n{i}", score=0.5, body_lines=200) for i in range(40)]
    pack([block("s", 1.0)], neighbors, tok, available=1_500)

    assert metrics.get("pack.neighbor_dropped") > 0
    assert metrics.get("pack.seed_l1_overflow") == 0


def test_seed_l1_overflow_rate_is_computable(tok):
    """§8.3's decision metric for the L2 skeleton, threshold > 0.05."""
    seeds = [block(f"s{i}", score=1.0 - i / 100, body_lines=300) for i in range(6)]
    pack(seeds, [], tok, available=4_000)

    total = metrics.get("pack.seeds_total")
    overflow = metrics.get("pack.seed_l1_overflow")
    assert total == 6
    assert 0.0 <= overflow / total <= 1.0


# --------------------------------------------------------------------------
# The packing report
# --------------------------------------------------------------------------


def test_packing_report_names_what_was_omitted(tok):
    """§5.4: "A developer who sees '2 omitted' knows to narrow the question;
    one who sees nothing assumes completeness"."""
    seeds = [block(f"s{i}", score=1.0 - i / 100, body_lines=60 * (i + 1)) for i in range(8)]
    ctx = pack(seeds, [], tok, available=4_000)

    report = ctx.report()
    assert "omitted" in report
    assert f"{len(ctx.dropped)} omitted" in report
    assert "tokens)" in report


def test_drop_rate_is_reported(tok):
    """§8.2 gates context drop rate at <= 0.10."""
    ctx = pack([block("s", 1.0)], [block("n", 0.5)], tok, available=4_000)
    assert ctx.drop_rate == 0.0

    seeds = [block(f"s{i}", score=1.0 - i / 100, body_lines=60 * (i + 1)) for i in range(8)]
    crowded = pack(seeds, [], tok, available=4_000)
    assert 0.0 < crowded.drop_rate <= 1.0


def test_empty_input_packs_to_nothing(tok):
    ctx = pack([], [], tok)
    assert ctx.blocks == [] and ctx.dropped == []
    assert ctx.packed_tokens == 0
    assert ctx.handles == {}


# --------------------------------------------------------------------------
# Ordering
# --------------------------------------------------------------------------


def test_seeds_are_packed_greedily_by_score(tok):
    seeds = [block("low", 0.1), block("high", 0.9), block("mid", 0.5)]
    ctx = pack(seeds, [], tok, available=4_000)
    assert [p.block.symbol.name for p in ctx.blocks] == ["high", "mid", "low"]


def test_packing_is_deterministic_on_equal_scores(tok):
    """§8.2 runs eval at temperature 0; the packed set must not vary."""
    seeds = [block(f"s{i}", score=0.5) for i in range(6)]
    first = [p.uid for p in pack(seeds, [], tok, available=4_000).blocks]
    for _ in range(5):
        assert [p.uid for p in pack(seeds, [], tok, available=4_000).blocks] == first


def test_a_neighbor_already_packed_as_a_seed_is_not_duplicated(tok):
    """Two copies of one symbol get two handles and waste budget on the second."""
    shared = block("shared", 1.0)
    ctx = pack([shared], [shared, block("other", 0.5)], tok, available=4_000)

    uids = [p.uid for p in ctx.blocks]
    assert len(uids) == len(set(uids))
    assert uids.count("uid_shared") == 1


# --------------------------------------------------------------------------
# The step 8 gate — fuzz
# --------------------------------------------------------------------------


def test_budget_never_breached_under_fuzz(tok):
    """Plan §8 gate: 500 random seed/neighbour size distributions.

    Randomised over the two dimensions that interact — how many blocks and how
    large each is — because the accounting bug this guards only appears when a
    particular branch happens to fire.
    """
    rng = random.Random(20260828)
    breaches: list[str] = []

    for case in range(500):
        available = rng.choice([800, 1_500, 4_000, 21_200])
        seeds = fuzz_blocks(rng, f"s{case}_", rng.randint(0, 12), 600)
        neighbors = fuzz_blocks(rng, f"n{case}_", rng.randint(0, 30), 60)

        ctx = pack(seeds, neighbors, tok, available=available)

        if ctx.packed_tokens > available:
            breaches.append(
                f"case {case}: {ctx.packed_tokens} > {available} "
                f"(seeds={len(seeds)}, neighbors={len(neighbors)}, "
                f"levels={[p.level for p in ctx.blocks]})"
            )
        if ctx.packed_tokens != sum(p.cost for p in ctx.blocks):
            breaches.append(f"case {case}: accounting drift")

    assert not breaches, "\n".join(breaches[:5])


def test_fuzz_actually_exercises_every_branch(tok):
    """A fuzz run that never truncates has not tested the truncation branch.

    Without this, the gate above could pass on 500 cases that all fit
    comfortably — which is the easiest way for a fuzz test to be decorative.
    """
    rng = random.Random(20260828)
    levels: set[str] = set()
    dropped = 0

    for case in range(500):
        available = rng.choice([800, 1_500, 4_000, 21_200])
        seeds = fuzz_blocks(rng, f"s{case}_", rng.randint(0, 12), 600)
        neighbors = fuzz_blocks(rng, f"n{case}_", rng.randint(0, 30), 60)
        ctx = pack(seeds, neighbors, tok, available=available)
        levels.update(p.level for p in ctx.blocks)
        dropped += len(ctx.dropped)

    assert {"full", "elided", "stub", "truncated"} <= levels, (
        f"the fuzz never reached: {{'full','elided','stub','truncated'}} - {levels}"
    )
    assert dropped > 0, "the fuzz never dropped a block"
