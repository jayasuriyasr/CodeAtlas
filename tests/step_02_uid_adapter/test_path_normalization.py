"""`rel_path` is a UID input (§3.2), so its normalisation is identity.

The audit found `normalize_rel_path` stripping a *set* of characters
(`str.lstrip("./")`) where it meant a prefix: `.github/scripts/release.py`
became `github/scripts/release.py`, a path that does not exist, while the
`:File` node kept the real one.
"""

from __future__ import annotations

import pytest

from adapters.base import normalize_rel_path, symbol_uid
from adapters.python import PythonAdapter, module_qualified_name


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("authx/views.py", "authx/views.py"),
        ("./authx/views.py", "authx/views.py"),
        ("authx\\views.py", "authx/views.py"),
        (".\\authx\\views.py", "authx/views.py"),
        ("/authx/views.py", "authx/views.py"),
        ("././authx/views.py", "authx/views.py"),
        (".github/scripts/release.py", ".github/scripts/release.py"),
        ("./.github/scripts/release.py", ".github/scripts/release.py"),
        (".storybook/main.ts", ".storybook/main.ts"),
        ("..hidden/x.py", "..hidden/x.py"),
        ("pkg/.private.py", "pkg/.private.py"),
        ("", ""),
    ],
)
def test_normalize_strips_prefixes_not_characters(raw, expected):
    assert normalize_rel_path(raw) == expected


def test_normalisation_is_idempotent():
    for raw in (".github/x.py", "./a/b.py", "a\\b\\c.py", "/x.py"):
        once = normalize_rel_path(raw)
        assert normalize_rel_path(once) == once


def test_a_dot_directory_and_its_undotted_twin_are_different_symbols():
    """Before the fix both normalised to `github/x.py` and shared every UID."""
    adapter = PythonAdapter()
    source = b"def run():\n    return 1\n"
    dotted = adapter.parse("r", ".github/x.py", source)
    plain = adapter.parse("r", "github/x.py", source)
    assert set(dotted.uids).isdisjoint(plain.uids)
    assert dotted.symbols[0].rel_path == ".github/x.py"


def test_windows_and_posix_spellings_share_uids():
    adapter = PythonAdapter()
    source = b"def run():\n    return 1\n"
    assert adapter.parse("r", "pkg\\mod.py", source).uids == adapter.parse(
        "r", "pkg/mod.py", source
    ).uids


def test_dot_directory_module_name_is_stable():
    assert module_qualified_name(".github/scripts/release.py") == ".github.scripts.release"
    assert symbol_uid("r", ".github/a.py", "x", 0) != symbol_uid("r", "github/a.py", "x", 0)
