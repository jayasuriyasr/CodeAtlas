"""The spec and the implementation must not drift apart.

v10.1 corrected three defects the code had already worked around. That leaves a
new hazard: the spec and the code can now disagree in the *other* direction, and
nothing would notice — a revision nobody implements is fiction, and an
implementation that quietly diverges from its spec is worse, because the
document is what the next person reads.

These tests parse `SEI_Architecture_v10.1.md` and compare it to what the code
actually does. They are deliberately narrow: only the things v10.1 changed, plus
the claim R1 removed. A general "does the code match the spec" test is not
possible and pretending otherwise would be theatre.
"""

from __future__ import annotations

import inspect
import re

import pytest

import config
from config import EDGE_TYPE_ALLOWLIST, EDGE_WEIGHTS
from graph.writer import GraphWriter
from index.moves import resolve_moves
from retrieve.expand import EXPANSION_CYPHER

SPEC = config.REPO_ROOT / "SEI_Architecture_v10.1.md"
FROZEN = config.REPO_ROOT / "SEI_Architecture_v10.0.md"


@pytest.fixture(scope="module")
def spec() -> str:
    return SPEC.read_text(encoding="utf-8")


def test_v10_0_is_retained_unmodified():
    """The plan's working rule 2: revisions are versioned, never in place.

    v10.0 is what every finding in `FINDINGS.md` cites. Editing it would make
    those citations unverifiable — the record of what was wrong is as much the
    point as the correction.
    """
    assert FROZEN.exists(), "v10.0 was deleted rather than superseded"
    original = FROZEN.read_text(encoding="utf-8")

    assert "since qualified_name, arity, and ordinal survive a move" in original, (
        "v10.0 no longer carries the R1 defect — it has been edited in place"
    )
    assert "allowlist {CALLS, IMPORTS, DEFINES, DISPATCHES_TO}" in original, (
        "v10.0 no longer carries the R2 defect — it has been edited in place"
    )


def test_v10_1_records_its_own_revision(spec):
    """§0.0 must name the defects and cite the runs (CLAUDE.md rule 1)."""
    assert "### 0.0 Changes from v10.0" in spec
    for marker in ("**R1**", "**R2**", "**R3**"):
        assert marker in spec, marker
    assert "HintedIndexNotFound" in spec, "R3 does not cite its run"
    assert "MATCHES A STORED NODE: False" in spec, "R1 does not cite its run"


# --------------------------------------------------------------------------
# R1
# --------------------------------------------------------------------------


def test_r1_the_false_claim_is_gone(spec):
    """The sentence that made every remap a no-op."""
    assert "since qualified_name, arity, and ordinal survive a move" not in spec


def test_r1_resolve_moves_takes_an_adapter(spec):
    """The fix is re-parsing at the old path, which needs an adapter.

    Checked on the signature rather than the body: a `resolve_moves` that could
    not reach an adapter could not implement §4.4 as revised, whatever its
    docstring said.
    """
    assert "adapter_for" in spec, "§4.4 does not pass an adapter"
    assert "adapter_for" in inspect.signature(resolve_moves).parameters


def test_r1_parsed_file_retains_source(spec):
    """§3.4 requires it, because the re-parse needs the bytes."""
    from adapters.base import ParsedFile

    assert "retains `source`" in spec
    assert "source" in ParsedFile.__dataclass_fields__


# --------------------------------------------------------------------------
# R2
# --------------------------------------------------------------------------


def test_r2_allowlist_matches_the_spec(spec):
    """§11.2 3b's allowlist and `config.EDGE_TYPE_ALLOWLIST` are one list."""
    body = re.search(r"allowlist \{([^}]*)\}", spec, re.S).group(1)
    named = {t.strip().strip("-").strip() for t in body.replace("--", " ").split(",")}
    assert {t for t in named if t} == set(EDGE_TYPE_ALLOWLIST)


def test_r2_expansion_matches_the_spec(spec):
    """§11.1's `MATCH` and `EXPANSION_CYPHER`'s must name the same types.

    A type in one and not the other is an edge written and never read, or a
    traversal for something nothing writes.
    """

    def types(text: str) -> set[str]:
        pattern = re.search(r"MATCH \(seed\)-\[r:([^\]]*)\]", text, re.S).group(1)
        return {t for t in re.split(r"[|\s]+", pattern) if t}

    assert types(spec) == types(EXPANSION_CYPHER) == set(EDGE_TYPE_ALLOWLIST)


def test_r2_edge_weights_match_the_spec(spec):
    """§5.2's table and `config.EDGE_WEIGHTS`, value for value."""
    listed = {
        name: float(value)
        for name, value in re.findall(r"`([A-Z_]+) ([0-9.]+)`", spec)
    }
    assert listed == EDGE_WEIGHTS


# --------------------------------------------------------------------------
# R3
# --------------------------------------------------------------------------


def test_r3_epoch_predicate_is_on_three_statements(spec):
    """Steps 4, 6a and 6b — the three that constrained only two properties.

    Step 5 already had `s.epoch < $epoch`, which is why it was the one statement
    that used the index under v10.0.
    """
    assert spec.count("AND s.epoch <= $epoch") == 3
    assert inspect.getsource(GraphWriter).count("AND s.epoch <= $epoch") == 3


def test_r3_explains_why_it_is_not_a_filter(spec):
    """A reader who takes it for a filter will remove it as redundant."""
    assert "PREFIX of its properties" in spec
    assert "tautology" in spec
