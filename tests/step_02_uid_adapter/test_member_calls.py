"""Calls through `self`, `cls` and `this`, and the class hierarchy behind them.

A method call such as `self.soft_delete()` used to become the hint
`self.soft_delete`. No qualified name starts with `self.`, so the writer could
never resolve it and the CALLS edge was silently dropped — for a method of the
same class as much as for an inherited one. `this.flush()` in TypeScript had the
same gap.

What the adapter can settle on its own it now settles: a method of the
enclosing class defined in the same file gets its uid at parse time. What it
cannot — a method that may live on a base class in another file — goes to the
writer as a *member lookup*, `Class.method` with `through_bases=True`, and the
class records its bases (in declaration order) so the writer can walk them.
"""

from __future__ import annotations

import pytest

from adapters.python import PythonAdapter
from adapters.typescript import TypeScriptAdapter

REPO = "repo_member_calls"


@pytest.fixture(scope="module")
def py() -> PythonAdapter:
    return PythonAdapter()


@pytest.fixture(scope="module")
def ts() -> TypeScriptAdapter:
    return TypeScriptAdapter()


def by_name(parsed) -> dict:
    return {s.qualified_name: s for s in parsed.symbols}


def calls_from(parsed, qualified_name: str):
    uid = by_name(parsed)[qualified_name].uid
    return [e for e in parsed.edges if e.kind == "CALLS" and e.source_uid == uid]


# -- Python ---------------------------------------------------------------------


SAME_CLASS = b'''
class Cart:
    def add(self, item):
        self.items.append(item)
        return self.total()

    def total(self):
        return 0

    @classmethod
    def empty(cls):
        return cls.build([])

    @classmethod
    def build(cls, items):
        return cls()
'''


def test_self_call_to_a_method_of_the_same_class_resolves_at_parse_time(py):
    parsed = py.parse(REPO, "shop/cart.py", SAME_CLASS)
    names = by_name(parsed)
    edges = calls_from(parsed, "shop.cart.Cart.add")
    targets = {e.target_uid for e in edges if e.target_uid}
    assert names["shop.cart.Cart.total"].uid in targets


def test_cls_call_resolves_the_same_way(py):
    parsed = py.parse(REPO, "shop/cart.py", SAME_CLASS)
    names = by_name(parsed)
    targets = {e.target_uid for e in calls_from(parsed, "shop.cart.Cart.empty")}
    assert names["shop.cart.Cart.build"].uid in targets


def test_attribute_of_self_is_not_mistaken_for_a_method(py):
    """`self.items.append(...)` calls a method of whatever `items` holds —
    unknowable without types. It must not become a member lookup on Cart."""
    parsed = py.parse(REPO, "shop/cart.py", SAME_CLASS)
    edges = calls_from(parsed, "shop.cart.Cart.add")
    assert not any(e.through_bases and e.target_hint.endswith(".append") for e in edges)


def test_inherited_method_becomes_a_member_lookup(py, fixtures_dir):
    """The fixture's own case: `Subscription.cancel` calls `self.soft_delete()`,
    which `SoftDeleteMixin` defines in another file."""
    path = fixtures_dir / "repos" / "django_min" / "billing" / "models.py"
    parsed = py.parse(REPO, "billing/models.py", path.read_bytes())
    edges = calls_from(parsed, "billing.models.Subscription.cancel")
    lookups = [e for e in edges if e.through_bases]
    assert [e.target_hint for e in lookups] == ["billing.models.Subscription.soft_delete"]
    assert all(e.target_uid is None for e in lookups)


def test_a_class_records_its_bases_in_declaration_order(py, fixtures_dir):
    path = fixtures_dir / "repos" / "django_min" / "billing" / "models.py"
    parsed = py.parse(REPO, "billing/models.py", path.read_bytes())
    subscription = by_name(parsed)["billing.models.Subscription"]
    assert subscription.bases == [
        "common.mixins.TimestampMixin",
        "common.mixins.SoftDeleteMixin",
        "django.db.models.Model",
    ]
    assert by_name(parsed)["billing.models.Subscription.cancel"].bases == []


def test_inherits_edges_point_at_each_base(py, fixtures_dir):
    path = fixtures_dir / "repos" / "django_min" / "billing" / "models.py"
    parsed = py.parse(REPO, "billing/models.py", path.read_bytes())
    uid = by_name(parsed)["billing.models.Subscription"].uid
    hints = [e.target_hint for e in parsed.edges if e.kind == "INHERITS" and e.source_uid == uid]
    assert hints == [
        "common.mixins.TimestampMixin",
        "common.mixins.SoftDeleteMixin",
        "django.db.models.Model",
    ]


def test_a_same_file_base_is_resolved_at_parse_time(py):
    parsed = py.parse(REPO, "pkg/shapes.py", b"class Base:\n    pass\n\n\nclass Square(Base, metaclass=Meta):\n    pass\n")
    names = by_name(parsed)
    square = names["pkg.shapes.Square"]
    assert square.bases == ["pkg.shapes.Base"], "metaclass= is not a base"
    inherits = [e for e in parsed.edges if e.kind == "INHERITS" and e.source_uid == square.uid]
    assert [e.target_uid for e in inherits] == [names["pkg.shapes.Base"].uid]


def test_a_call_on_another_object_is_left_as_it_was(py):
    parsed = py.parse(REPO, "pkg/svc.py", b"def run(order):\n    order.ship()\n")
    (edge,) = calls_from(parsed, "pkg.svc.run")
    assert edge.target_hint == "order.ship" and not edge.through_bases


def test_self_in_a_plain_function_is_not_a_member_lookup(py):
    """`self` outside a class body is just a parameter name."""
    parsed = py.parse(REPO, "pkg/util.py", b"def helper(self):\n    self.go()\n")
    (edge,) = calls_from(parsed, "pkg.util.helper")
    assert not edge.through_bases


# -- TypeScript -----------------------------------------------------------------


STORE = b"""import { Base } from "./base";

export class Store extends Base {
  save(): void {
    this.flush();
    this.persist();
  }
  flush(): void {}
}
"""


def test_this_call_to_a_method_of_the_same_class_resolves_at_parse_time(ts):
    parsed = ts.parse(REPO, "src/store.ts", STORE)
    names = by_name(parsed)
    targets = {e.target_uid for e in calls_from(parsed, "src.store.Store.save")}
    assert names["src.store.Store.flush"].uid in targets


def test_this_call_to_an_inherited_method_becomes_a_member_lookup(ts):
    parsed = ts.parse(REPO, "src/store.ts", STORE)
    lookups = [e for e in calls_from(parsed, "src.store.Store.save") if e.through_bases]
    assert [e.target_hint for e in lookups] == ["src.store.Store.persist"]


def test_extends_is_recorded_as_a_base_and_an_inherits_edge(ts):
    parsed = ts.parse(REPO, "src/store.ts", STORE)
    store = by_name(parsed)["src.store.Store"]
    assert store.bases == ["src.base.Base"]
    inherits = [e for e in parsed.edges if e.kind == "INHERITS" and e.source_uid == store.uid]
    assert [e.target_hint for e in inherits] == ["src.base.Base"]


def test_implements_is_not_inheritance(ts):
    """An interface carries no implementation for `this.x()` to land in."""
    parsed = ts.parse(REPO, "src/job.ts", b"export class Job implements Runnable {\n  run(): void {}\n}\n")
    assert by_name(parsed)["src.job.Job"].bases == []
