"""A method called on a class named directly: `FilterSet.filter_for_field(…)`.

The largest single cause of missed callers in the benchmark: 26 of the 92 that
django-filter's run missed (eval/benchmark_hand_check.md). `FilterSet` inherits
`filter_for_field` — a classmethod — from `BaseFilterSet`, so the ordinary
hint `filterset.FilterSet.filter_for_field` names nothing, and the call was lost.

The adapter now also emits a member lookup for `Name.method()` when `Name` is
an import or a same-file definition. The writer walks the class's resolution
order as it does for `self.method()`. When `Name` turns out not to be one class
in the repository — a module (`curl.post`), a library class — the lookup is
dropped without counting, because the ordinary edge for the same call stands.
"""

from __future__ import annotations

from .conftest import REPO
from .test_inherited_calls import calls_of, index

BASE = (b"class BaseFilterSet:\n    @classmethod\n    def filter_for_field(cls, f, name):\n"
        b"        return name\n\n\nclass FilterSet(BaseFilterSet):\n    pass\n")


def test_an_inherited_classmethod_called_on_the_class_resolves(writer, graph_db):
    index(writer, {
        "pkg/filterset.py": BASE,
        "pkg/tests.py": (b"from pkg.filterset import FilterSet\n\n\n"
                         b"def test_it():\n    return FilterSet.filter_for_field(None, 'x')\n"),
    })
    assert "pkg.filterset.BaseFilterSet.filter_for_field" in calls_of(graph_db, "pkg.tests.test_it")


def test_a_same_file_class_resolves_too(writer, graph_db):
    index(writer, {"pkg/filterset.py": BASE + (
        b"\n\ndef build():\n    return FilterSet.filter_for_field(None, 'y')\n")})
    assert "pkg.filterset.BaseFilterSet.filter_for_field" in calls_of(graph_db, "pkg.filterset.build")


def test_a_method_defined_on_the_class_itself_is_one_edge(writer, graph_db):
    index(writer, {
        "pkg/filterset.py": BASE,
        "pkg/tests.py": (b"from pkg.filterset import BaseFilterSet\n\n\n"
                         b"def test_it():\n    return BaseFilterSet.filter_for_field(None, 'x')\n"),
    })
    rows = graph_db.run(
        "MATCH (:Symbol {repo_id: $r, qualified_name: 'pkg.tests.test_it'})-[c:CALLS]->"
        "(:Symbol {qualified_name: 'pkg.filterset.BaseFilterSet.filter_for_field'}) "
        "RETURN count(c) AS n", r=REPO)
    assert rows[0]["n"] == 1


def test_a_module_function_call_is_unchanged_and_not_counted(writer, graph_db):
    report = index(writer, {
        "pkg/curl.py": b"def post(url):\n    return url\n",
        "pkg/views.py": b"from pkg import curl\n\n\ndef add():\n    return curl.post('u')\n",
    })
    assert calls_of(graph_db, "pkg.views.add") == ["pkg.curl.post"]
    assert report.edges_unresolved == 0


def test_a_library_class_gets_no_edge(writer, graph_db):
    report = index(writer, {"pkg/views.py": (
        b"from django.utils import timezone\n\n\ndef now():\n    return timezone.now()\n")})
    assert calls_of(graph_db, "pkg.views.now") == []
    # The import and the ordinary call hint, as before the change — the
    # lookup on `timezone` adds nothing.
    assert report.edges_unresolved == 2
