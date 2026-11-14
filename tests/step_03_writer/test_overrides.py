"""Overrides are edges, and "what calls X" includes callers reached by dispatch.

Reproduced first: `Base.run` calls `self.hook()`, and `Sub(Base)` overrides
`hook`. The call resolves to `Base.hook` (the method `self` names in Base's
own body), so "what calls `Sub.hook`" answered nothing — although every
`Sub().run()` ends up in `Sub.hook`.

Now the writer links each method to the method it overrides (OVERRIDES, found
along the class's resolution order, refused when a library base comes first),
and the callers rule also lists the callers of the overridden method, marked
as reached through dispatch.
"""

from __future__ import annotations

from retrieve.structural import StructuralResolver

from .conftest import REPO
from .test_inherited_calls import index

BASE = (b"class Base:\n    def run(self):\n        return self.hook()\n\n"
        b"    def hook(self):\n        return 0\n")
SUB = b"from pkg.base import Base\n\n\nclass Sub(Base):\n    def hook(self):\n        return 1\n"


class Runner:
    def __init__(self, graph_db):
        self.db = graph_db

    def run(self, query, **params):
        return [dict(r) for r in self.db.run(query, **params)]


def overrides(graph_db) -> list[tuple[str, str]]:
    rows = graph_db.run(
        "MATCH (a:Symbol {repo_id: $r})-[:OVERRIDES]->(b:Symbol) "
        "RETURN a.qualified_name AS a, b.qualified_name AS b ORDER BY a", r=REPO)
    return [(r["a"], r["b"]) for r in rows]


def test_an_override_is_linked_to_the_method_it_overrides(writer, graph_db):
    index(writer, {"pkg/base.py": BASE, "pkg/sub.py": SUB})
    assert overrides(graph_db) == [("pkg.sub.Sub.hook", "pkg.base.Base.hook")]


def test_the_nearest_definition_up_the_order_is_the_one_overridden(writer, graph_db):
    index(writer, {
        "pkg/base.py": BASE,
        "pkg/mid.py": b"from pkg.base import Base\n\n\nclass Mid(Base):\n    pass\n",
        "pkg/leaf.py": b"from pkg.mid import Mid\n\n\nclass Leaf(Mid):\n    def hook(self):\n        return 2\n",
    })
    assert ("pkg.leaf.Leaf.hook", "pkg.base.Base.hook") in overrides(graph_db)


def test_no_override_behind_a_library_base(writer, graph_db):
    """`External` may define `hook` itself; which one is overridden is unknown."""
    index(writer, {
        "pkg/base.py": BASE,
        "pkg/sub.py": (b"from lib import External\nfrom pkg.base import Base\n\n\n"
                       b"class Sub(External, Base):\n    def hook(self):\n        return 1\n"),
    })
    assert overrides(graph_db) == []


def test_a_new_method_overrides_nothing(writer, graph_db):
    index(writer, {"pkg/base.py": BASE, "pkg/sub.py": (
        b"from pkg.base import Base\n\n\nclass Sub(Base):\n    def extra(self):\n        return 1\n")})
    assert overrides(graph_db) == []


def test_callers_of_an_override_include_callers_through_dispatch(writer, graph_db):
    index(writer, {"pkg/base.py": BASE, "pkg/sub.py": SUB})
    answer = StructuralResolver(runner=Runner(graph_db)).answer(
        REPO, "what calls `Sub.hook`", "callers")
    names = {row["qualified_name"]: row for row in graph_db.run(
        "MATCH (s:Symbol {repo_id: $r}) RETURN s.uid AS uid, s.qualified_name AS qualified_name", r=REPO)}
    by_uid = {row["uid"]: qn for qn, row in names.items()}
    assert [by_uid[u] for u in answer.related_uids] == ["pkg.base.Base.run"]
    assert answer.via[answer.related_uids[0]] == "pkg.base.Base.hook"


def test_callers_of_the_base_method_are_unchanged(writer, graph_db):
    index(writer, {"pkg/base.py": BASE, "pkg/sub.py": SUB})
    answer = StructuralResolver(runner=Runner(graph_db)).answer(
        REPO, "what calls `Base.hook`", "callers")
    assert len(answer.related_uids) == 1 and answer.via == {}
