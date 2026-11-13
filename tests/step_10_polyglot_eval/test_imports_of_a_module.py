""""What imports `common.utils`" lists the modules that import from it.

Found by scoring the traversal against `expected_related` (golden S08): the
answer was empty although four modules import from `common.utils`. An IMPORTS
edge points at what is imported — `from common.utils import audit_event` is an
edge to the function — so the edges into the module itself are only the rare
`import common.utils`. The traversal now also follows IMPORTS into the symbols
the module defines.
"""

from __future__ import annotations

from .pipeline_support import PY_REPO, ask


def related(notes) -> set[str]:
    return {r["qualified_name"] for r in notes["structural"]["related"]}


def test_from_imports_count_as_importing_the_module(indexed):
    _r, _seeds, _n, notes = ask(indexed, PY_REPO, "what imports `common.utils`")
    assert notes["route_taken"] == "STRUCTURAL"
    assert related(notes) == {"authx.services", "authx.views", "billing.tasks", "billing.views"}


def test_importers_of_a_function_are_unchanged(indexed):
    """The entity is a function: its importers are the modules naming it."""
    _r, _seeds, _n, notes = ask(indexed, PY_REPO, "what imports `chunked`")
    assert related(notes) == {"billing.tasks"}
