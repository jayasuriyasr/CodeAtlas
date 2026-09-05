"""A repository-wide guard on §5.4's `never len // 4`.

`test_tokenizer_is_not_a_length_heuristic` proves the tokenizer we have is not
a character count. This proves nobody quietly added one somewhere else — which
is the realistic failure, because dividing a length by four is what you reach
for at 6pm when the real tokenizer needs an API key.

CLAUDE.md lists this among the traps that caused real defects. §5.4: `len // 4`
"is off by up to 2x on dense code", so a budget computed that way silently
overshoots or wastes 40% of the context window, and no test of the packer would
notice.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SOURCE_DIRS = (
    "adapters", "graph", "index", "retrieve", "pack", "ground", "api", "eval",
)

#: `len(x) // 4`, `len(x) / 4`, and the same with any small divisor — the shape,
#: not one spelling of it.
_HEURISTIC = re.compile(r"len\s*\([^)]*\)\s*(?://|/)\s*\d")


def _python_sources() -> list[Path]:
    out = [REPO_ROOT / "config.py", REPO_ROOT / "metrics.py"]
    for directory in SOURCE_DIRS:
        out.extend(sorted((REPO_ROOT / directory).rglob("*.py")))
    return [p for p in out if p.exists()]


def test_no_character_count_token_estimate():
    offenders = []
    for path in _python_sources():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _HEURISTIC.search(line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")

    assert not offenders, (
        "a length-based token estimate appeared; §5.4 requires the provider's "
        "real tokenizer via S2:\n  " + "\n  ".join(offenders)
    )


def test_the_guard_can_actually_fail():
    """A guard nobody has seen fail is a guard nobody knows works."""
    assert _HEURISTIC.search("tokens = len(text) // 4")
    assert _HEURISTIC.search("cost = len(chunk_text) / 4")
    assert not _HEURISTIC.search("total = len(parts) - 1")
