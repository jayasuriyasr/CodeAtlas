"""Calls on objects whose type is certain become member lookups on that type.

Measured before this change (eval/benchmark.py, every function with callers):
on django-filter 61% of the callers SEI missed were `x = ClassName(...)` then
`x.method()`. `x.method()` went through the ordinary call path and became the
hint `x.method`, which no qualified name can match.

The adapter now gives a receiver a type only when it is certain from the code
in front of it, and otherwise leaves the call alone:

* a local assigned exactly once, from a call (`form = SignupForm(data)`);
* a parameter or local with a plain annotation (`form: SignupForm`);
* `self.attr`, assigned exactly once in the class, in `__init__`, from a call;
* a class attribute assigned from a call in the class body (Django's
  `objects = TagManager()`), reached as `Model.objects.method()`.

Whether the callee is actually a class is not the adapter's to decide — a
factory function looks the same — so the edge is a member lookup that the
writer resolves only against a class it knows (`test_typed_receivers_graph.py`).
"""

from __future__ import annotations

import pytest

from adapters.python import PythonAdapter

REPO = "repo_typed"


@pytest.fixture(scope="module")
def py():
    return PythonAdapter()


def lookups(parsed, qualified_name: str) -> list[str]:
    uid = {s.qualified_name: s.uid for s in parsed.symbols}[qualified_name]
    return sorted(e.target_hint for e in parsed.edges
                  if e.kind == "CALLS" and e.source_uid == uid and e.through_bases)


def test_a_local_built_from_an_imported_class(py):
    parsed = py.parse(REPO, "app/views.py", (
        b"from app.forms import SignupForm\n\n\n"
        b"def signup(data):\n    form = SignupForm(data)\n    return form.clean()\n"))
    assert lookups(parsed, "app.views.signup") == ["app.forms.SignupForm.clean"]


def test_a_local_built_from_a_class_in_the_same_file(py):
    parsed = py.parse(REPO, "pkg/cart.py", (
        b"class Cart:\n    def total(self):\n        return 0\n\n\n"
        b"def checkout():\n    cart = Cart()\n    return cart.total()\n"))
    assert lookups(parsed, "pkg.cart.checkout") == ["pkg.cart.Cart.total"]


def test_a_local_assigned_twice_has_no_type(py):
    parsed = py.parse(REPO, "app/views.py", (
        b"from app.forms import A, B\n\n\n"
        b"def pick(flag):\n    form = A()\n    if flag:\n        form = B()\n    return form.clean()\n"))
    assert lookups(parsed, "app.views.pick") == []


@pytest.mark.parametrize("binding", [
    "for form in forms:\n        pass",
    "with open_form() as form:\n        pass",
    "form += 1",
    "global form",
])
def test_a_local_bound_any_other_way_has_no_type(py, binding):
    source = (f"from app.forms import A\n\n\ndef f(forms):\n    form = A()\n"
              f"    {binding}\n    return form.clean()\n").encode()
    assert lookups(py.parse(REPO, "app/views.py", source), "app.views.f") == []


def test_an_annotated_parameter(py):
    parsed = py.parse(REPO, "app/views.py", (
        b"from app.forms import SignupForm\n\n\n"
        b"def handle(form: SignupForm, other):\n    other.save()\n    return form.clean()\n"))
    assert lookups(parsed, "app.views.handle") == ["app.forms.SignupForm.clean"]


@pytest.mark.parametrize("annotation", ["Optional[SignupForm]", "'SignupForm'", "SignupForm | None"])
def test_an_annotation_that_is_not_a_plain_class_name_is_not_trusted(py, annotation):
    source = (f"from app.forms import SignupForm\n\n\n"
              f"def handle(form: {annotation}):\n    return form.clean()\n").encode()
    assert lookups(py.parse(REPO, "app/views.py", source), "app.views.handle") == []


def test_an_annotated_local(py):
    parsed = py.parse(REPO, "app/views.py", (
        b"from app.forms import SignupForm\n\n\n"
        b"def handle(raw):\n    form: SignupForm = build(raw)\n    return form.clean()\n"))
    assert lookups(parsed, "app.views.handle") == ["app.forms.SignupForm.clean"]


def test_an_attribute_set_once_in_init(py):
    parsed = py.parse(REPO, "app/service.py", (
        b"from app.repo import Repo\n\n\n"
        b"class Service:\n    def __init__(self):\n        self.repo = Repo()\n\n"
        b"    def run(self):\n        return self.repo.save()\n"))
    assert lookups(parsed, "app.service.Service.run") == ["app.repo.Repo.save"]


def test_an_attribute_assigned_outside_init_too_has_no_type(py):
    parsed = py.parse(REPO, "app/service.py", (
        b"from app.repo import Repo, Other\n\n\n"
        b"class Service:\n    def __init__(self):\n        self.repo = Repo()\n\n"
        b"    def swap(self):\n        self.repo = Other()\n\n"
        b"    def run(self):\n        return self.repo.save()\n"))
    assert lookups(parsed, "app.service.Service.run") == []


def test_a_class_attribute_reached_through_the_class(py):
    """Django's manager: `objects = TagManager()` on the model."""
    parsed = py.parse(REPO, "tags/models.py", (
        b"class TagManager:\n    def get_by_natural_key(self, name):\n        return name\n\n\n"
        b"class Tag:\n    objects = TagManager()\n\n\n"
        b"def lookup(name):\n    return Tag.objects.get_by_natural_key(name)\n"))
    assert lookups(parsed, "tags.models.lookup") == ["tags.models.TagManager.get_by_natural_key"]


def test_a_class_attribute_of_an_imported_class_goes_to_the_writer(py):
    parsed = py.parse(REPO, "tags/views.py", (
        b"from tags.models import Tag\n\n\n"
        b"def lookup(name):\n    return Tag.objects.get_by_natural_key(name)\n"))
    uid = {s.qualified_name: s.uid for s in parsed.symbols}["tags.views.lookup"]
    (edge,) = [e for e in parsed.edges if e.source_uid == uid and e.via_attribute]
    assert edge.target_hint == "tags.models.Tag.objects.get_by_natural_key" and edge.through_bases


def test_a_class_records_its_attribute_types(py):
    parsed = py.parse(REPO, "tags/models.py", (
        b"from tags.managers import TagManager\n\n\n"
        b"class Tag:\n    objects = TagManager()\n    name = 'x'\n\n"
        b"    def __init__(self):\n        self.cache = Cache()\n"))
    tag = {s.qualified_name: s for s in parsed.symbols}["tags.models.Tag"]
    assert tag.attributes == ["objects=tags.managers.TagManager", "cache=Cache"]


def test_a_call_on_an_untyped_parameter_is_left_as_it_was(py):
    parsed = py.parse(REPO, "app/views.py", b"def ship(order):\n    return order.ship()\n")
    uid = {s.qualified_name: s.uid for s in parsed.symbols}["app.views.ship"]
    (edge,) = [e for e in parsed.edges if e.source_uid == uid and e.kind == "CALLS"]
    assert edge.target_hint == "order.ship" and not edge.through_bases
