"""Step 2 — the UID contract (§3.2) and the Python arity rule.

Every test here guards a defect that was found by reading, not by running:
the plan's §12 table names T1, T2 and T8. The common failure mode is silent —
a UID that changes when it should not severs every inbound edge, and the system
then reports zero callers for code with dozens, which §4.4 calls worse than an
error because it looks like an answer.
"""

from __future__ import annotations

import pytest

from adapters.base import normalize_rel_path, symbol_uid
from adapters.python import PythonAdapter, compute_arity, module_qualified_name
from adapters.grammars import python_parser

REPO = "repo_step2"


@pytest.fixture(scope="module")
def adapter() -> PythonAdapter:
    return PythonAdapter()


def parse(adapter: PythonAdapter, source: str, rel_path: str = "pkg/mod.py"):
    return adapter.parse(REPO, rel_path, source.encode())


def by_qname(parsed, qualified_name: str):
    return [s for s in parsed.symbols if s.qualified_name == qualified_name]


# --------------------------------------------------------------------------
# T1 — UID collision on same-name/same-arity definitions
# --------------------------------------------------------------------------


CONDITIONAL_DEFS = '''
import sys
from typing import TYPE_CHECKING

if sys.platform == "win32":
    def clock_skew(now):
        return 0
else:
    def clock_skew(now):
        return 1

if TYPE_CHECKING:
    def annotate(x):
        ...
else:
    def annotate(x):
        return x
'''


def test_overload_uids_distinct(adapter):
    """T1. Conditionally-defined same-name/same-arity functions get distinct UIDs.

    Without the source-order ordinal these four functions produce two UIDs, and
    `MERGE` collapses each pair into one node whose properties are whichever
    branch was written last (§3.2).
    """
    parsed = parse(adapter, CONDITIONAL_DEFS)

    skews = by_qname(parsed, "pkg.mod.clock_skew")
    assert len(skews) == 2, "both platform branches must be symbols"
    assert {s.arity for s in skews} == {1}, "same arity — the collision precondition"
    assert skews[0].uid != skews[1].uid
    assert [s.ordinal for s in skews] == [0, 1]

    annotates = by_qname(parsed, "pkg.mod.annotate")
    assert len(annotates) == 2
    assert annotates[0].uid != annotates[1].uid

    assert len(parsed.uids) == len(set(parsed.uids)), "no collisions anywhere in the file"


def test_ordinal_follows_source_order_not_traversal_order(adapter):
    """The ordinal is only stable if it is assigned in source order.

    A collector that assigned ordinals as it happened to walk the tree would
    give the same file different UIDs depending on traversal, which is the same
    defect as having no ordinal at all — just harder to see.
    """
    parsed = parse(adapter, CONDITIONAL_DEFS)
    skews = by_qname(parsed, "pkg.mod.clock_skew")
    assert skews[0].start_line < skews[1].start_line
    assert skews[0].ordinal == 0 and skews[1].ordinal == 1


def test_uid_is_pure_over_its_five_inputs():
    """§3.2's function must depend on exactly its arguments, and each of them."""
    base = dict(
        repo_id=REPO, rel_path="a/b.py", qualified_name="a.b.f", arity=2, ordinal=0
    )
    uid = symbol_uid(**base)
    assert uid == symbol_uid(**base), "same inputs, same UID"
    assert len(uid) == 20

    for field, changed in (
        ("repo_id", "other_repo"),
        ("rel_path", "a/c.py"),
        ("qualified_name", "a.b.g"),
        ("arity", 3),
        ("ordinal", 1),
    ):
        assert symbol_uid(**{**base, field: changed}) != uid, f"{field} must matter"


# --------------------------------------------------------------------------
# T2 — arity
# --------------------------------------------------------------------------


ARITY_CASES = '''
class Service:
    def method(self, a, b):
        pass

    @classmethod
    def factory(cls, a):
        pass

    @staticmethod
    def helper(a, b, c):
        pass

    def varargs(self, a, *args, **kwargs):
        pass

    def kwonly(self, a, *, b, c=1):
        pass

    def posonly(self, a, /, b):
        pass


def free(a, b=1, *args, c, **kwargs):
    pass


def receiver_named_self_at_module_level(self, a):
    pass
'''


@pytest.mark.parametrize(
    "qualified_name, expected_arity, why",
    [
        ("pkg.mod.Service.method", 2, "self excluded"),
        ("pkg.mod.Service.factory", 1, "cls excluded"),
        ("pkg.mod.Service.helper", 3, "staticmethod has no receiver to exclude"),
        ("pkg.mod.Service.varargs", 1, "*args and **kwargs excluded"),
        ("pkg.mod.Service.kwonly", 3, "keyword-only params counted, `*` is not one"),
        ("pkg.mod.Service.posonly", 2, "`/` is punctuation, not a parameter"),
        ("pkg.mod.free", 3, "a, b=1, c — varargs excluded"),
        ("pkg.mod.receiver_named_self_at_module_level", 2, "not a method: self counts"),
    ],
)
def test_arity_excludes_self_and_varargs(adapter, qualified_name, expected_arity, why):
    """T2. §3.2's Python row, one row of the table per case."""
    parsed = parse(adapter, ARITY_CASES)
    found = by_qname(parsed, qualified_name)
    assert len(found) == 1, f"{qualified_name} not found"
    assert found[0].arity == expected_arity, why


def test_default_value_change_preserves_uid(adapter):
    """T2. `timeout=30` -> `timeout=60` must not churn the UID.

    §3.2: "Defaults are counted but their *values* are not." If the value
    leaked into arity, every tuning change to a default would delete and
    recreate the symbol and sever its inbound CALLS edges.
    """
    before = parse(adapter, "def connect(host, timeout=30):\n    return host\n")
    after = parse(adapter, "def connect(host, timeout=60):\n    return host\n")

    assert by_qname(before, "pkg.mod.connect")[0].arity == 2
    assert (
        by_qname(before, "pkg.mod.connect")[0].uid
        == by_qname(after, "pkg.mod.connect")[0].uid
    )


def test_adding_a_parameter_does_change_the_uid(adapter):
    """The other half of T2: arity must actually be an input.

    A rule that never changes the UID is as broken as one that always does; it
    would merge two genuinely different functions onto one node.
    """
    before = parse(adapter, "def connect(host):\n    return host\n")
    after = parse(adapter, "def connect(host, port):\n    return host\n")
    assert (
        by_qname(before, "pkg.mod.connect")[0].uid
        != by_qname(after, "pkg.mod.connect")[0].uid
    )


def test_annotation_change_preserves_uid(adapter):
    """Adding a type annotation changes neither the name nor the count."""
    before = parse(adapter, "def f(a, b):\n    return a\n")
    after = parse(adapter, "def f(a: int, b: str) -> int:\n    return a\n")
    assert by_qname(before, "pkg.mod.f")[0].uid == by_qname(after, "pkg.mod.f")[0].uid


def test_compute_arity_is_callable_without_the_adapter():
    """The rule is the contract; keep it testable on its own."""
    tree = python_parser().parse(b"def f(self, a, *args, b=1, **kw):\n    pass\n")
    func = tree.root_node.named_children[0]
    assert compute_arity(func, b"def f(self, a, *args, b=1, **kw):\n    pass\n",
                         is_method=False) == 3
    assert compute_arity(func, b"def f(self, a, *args, b=1, **kw):\n    pass\n",
                         is_method=True) == 2


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_parse_is_deterministic(adapter, fixtures_dir):
    """Parse twice, byte-identical UID set — over a real file, not a snippet."""
    source = (fixtures_dir / "repos" / "django_min" / "authx" / "views.py").read_bytes()

    first = adapter.parse(REPO, "authx/views.py", source)
    second = adapter.parse(REPO, "authx/views.py", source)

    assert first.uids == second.uids
    assert first.content_hash == second.content_hash
    assert [(s.qualified_name, s.arity, s.ordinal) for s in first.symbols] == [
        (s.qualified_name, s.arity, s.ordinal) for s in second.symbols
    ]


def test_rel_path_separator_does_not_fork_identity(adapter):
    """A UID input that varies by platform forks the graph invisibly.

    `rel_path` is hashed into the UID (§3.2). Indexing from Windows and reading
    from CI would otherwise produce two disjoint symbol sets for one file.
    """
    assert normalize_rel_path("authx\\views.py") == "authx/views.py"
    assert normalize_rel_path("./authx/views.py") == "authx/views.py"

    source = "def f(a):\n    return a\n".encode()
    windows = adapter.parse(REPO, "authx\\views.py", source)
    posix = adapter.parse(REPO, "authx/views.py", source)
    assert windows.uids == posix.uids


@pytest.mark.parametrize(
    "rel_path, expected",
    [
        ("authx/views.py", "authx.views"),
        ("authx/__init__.py", "authx"),
        ("manage.py", "manage"),
        ("a/b/c/d.py", "a.b.c.d"),
        ("authx\\views.py", "authx.views"),
    ],
)
def test_module_qualified_name(rel_path, expected):
    assert module_qualified_name(rel_path) == expected


# --------------------------------------------------------------------------
# T8 — what is not a symbol
# --------------------------------------------------------------------------


LAMBDAS = '''
handler = lambda request: request.user

def outer(items):
    keyed = sorted(items, key=lambda i: i.name)
    fn = lambda: None
    return keyed, fn

callbacks = [lambda x: x + 1, lambda y: y * 2]
'''


def test_anonymous_lambda_not_a_symbol(adapter):
    """T8. §3.2: truly anonymous functions are not symbols and are not indexed.

    They appear only inside their enclosing symbol's body — which the last
    assertion checks, because dropping them from the *text* too would lose
    retrieval signal that is genuinely there.
    """
    parsed = parse(adapter, LAMBDAS)

    kinds = {s.kind for s in parsed.symbols}
    assert kinds <= {"module", "function", "class"}

    names = {s.qualified_name for s in parsed.symbols}
    assert names == {"pkg.mod", "pkg.mod.outer"}, (
        "only the module and the one named def are symbols"
    )
    assert not any("lambda" in n for n in names)

    outer = by_qname(parsed, "pkg.mod.outer")[0]
    assert "lambda i: i.name" in outer.source_code, "still present in the body text"


def test_nested_named_function_is_a_symbol(adapter):
    """The converse of T8: a *named* nested function is not anonymous.

    §3.2 derives qualified names from "the module path plus enclosing scopes",
    which only means something if nested scopes produce symbols.
    """
    parsed = parse(
        adapter,
        "def outer(a):\n"
        "    def inner(b):\n"
        "        return b\n"
        "    return inner(a)\n",
    )
    names = {s.qualified_name for s in parsed.symbols}
    assert "pkg.mod.outer.inner" in names


# --------------------------------------------------------------------------
# Line movement
# --------------------------------------------------------------------------


def test_line_move_preserves_uid(adapter):
    """Adding a line above a function must not change its UID.

    §3.2 excludes line numbers from the UID precisely because they churn on
    every edit above the symbol. They are stored as mutable properties instead,
    and this checks that they *are* stored — a UID that ignores position is only
    useful if something else still knows where the symbol is (§5.1 Tier 0).
    """
    before = parse(adapter, "def target(a):\n    return a\n")
    after = parse(adapter, "import os\n\n\ndef target(a):\n    return a\n")

    b = by_qname(before, "pkg.mod.target")[0]
    a = by_qname(after, "pkg.mod.target")[0]

    assert b.uid == a.uid
    assert b.start_line == 1 and a.start_line == 4, "position still tracked"


def test_body_change_preserves_uid(adapter):
    """§3.2: "a symbol whose body changes is the same symbol"."""
    before = parse(adapter, "def target(a):\n    return a\n")
    after = parse(adapter, "def target(a):\n    log(a)\n    return a + 1\n")
    assert (
        by_qname(before, "pkg.mod.target")[0].uid
        == by_qname(after, "pkg.mod.target")[0].uid
    )


def test_decorator_change_preserves_uid_but_changes_source(adapter):
    """A decorator is part of what the symbol *does*, not of what it *is*.

    So the UID holds — but `source_code` must include the decorator, or §4.2's
    body_hash would miss the change and serve a stale embedding.
    """
    before = parse(adapter, "@cached\ndef target(a):\n    return a\n")
    after = parse(adapter, "@retry(3)\ndef target(a):\n    return a\n")

    b = by_qname(before, "pkg.mod.target")[0]
    a = by_qname(after, "pkg.mod.target")[0]

    assert b.uid == a.uid
    assert b.decorators == ["@cached"] and a.decorators == ["@retry(3)"]
    assert "@cached" in b.source_code and "@retry(3)" in a.source_code
