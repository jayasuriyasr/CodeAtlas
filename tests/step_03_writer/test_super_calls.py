"""Calls through `super()` resolve to the next class in the method resolution order.

Found by the open-source benchmark: on django-filter, most of the callers SEI
missed were `super().__init__(...)`, `super().clean(...)` and `super().filter(...)`
in subclasses — recall 0.28. `super().m()` inside class C is the same lookup as
`self.m()` with C itself skipped, so it rides on the same machinery.
"""

from __future__ import annotations

from adapters.python import PythonAdapter
from adapters.typescript import TypeScriptAdapter

from .conftest import REPO, epoch
from .test_inherited_calls import calls_of, index


def test_the_adapter_marks_a_super_call(fixtures_dir):
    parsed = PythonAdapter().parse(REPO, "pkg/forms.py", (
        b"class Base:\n    def clean(self):\n        return 1\n\n\n"
        b"class Child(Base):\n    def clean(self):\n        return super().clean() + 1\n"))
    uid = {s.qualified_name: s.uid for s in parsed.symbols}["pkg.forms.Child.clean"]
    # `super()` is itself a call (to a builtin, so it never resolves); the
    # member lookup is the other one.
    (edge,) = [e for e in parsed.edges
               if e.kind == "CALLS" and e.source_uid == uid and e.through_bases]
    assert edge.after_owner
    assert edge.target_hint == "pkg.forms.Child.clean" and edge.target_uid is None


def test_super_reaches_the_parent_not_the_method_itself(writer, graph_db):
    index(writer, {"pkg/forms.py": (
        b"class Base:\n    def clean(self):\n        return 1\n\n\n"
        b"class Child(Base):\n    def clean(self):\n        return super().clean() + 1\n")})
    assert calls_of(graph_db, "pkg.forms.Child.clean") == ["pkg.forms.Base.clean"]


def test_super_init_across_files(writer, graph_db):
    index(writer, {
        "pkg/base.py": b"class Field:\n    def __init__(self, required=True):\n        self.required = required\n",
        "pkg/fields.py": (b"from pkg.base import Field\n\n\nclass CharField(Field):\n"
                          b"    def __init__(self, max_length=None, **kwargs):\n"
                          b"        super().__init__(**kwargs)\n"),
    })
    assert calls_of(graph_db, "pkg.fields.CharField.__init__") == ["pkg.base.Field.__init__"]


def test_two_argument_super_is_the_same_call(writer, graph_db):
    index(writer, {"pkg/forms.py": (
        b"class Base:\n    def clean(self):\n        return 1\n\n\n"
        b"class Child(Base):\n    def clean(self):\n        return super(Child, self).clean()\n")})
    assert calls_of(graph_db, "pkg.forms.Child.clean") == ["pkg.forms.Base.clean"]


def test_super_into_a_library_base_is_left_unresolved(writer, graph_db):
    index(writer, {"pkg/forms.py": (
        b"from django import forms\n\n\nclass Child(forms.Form):\n"
        b"    def clean(self):\n        return super().clean()\n")})
    assert calls_of(graph_db, "pkg.forms.Child.clean") == []


def test_typescript_super_calls(writer, graph_db):
    batch = {
        "src/base.ts": b"export class Base {\n  constructor(x: number) {}\n  save(): void {}\n}\n",
        "src/store.ts": (b'import { Base } from "./base";\n\n'
                         b"export class Store extends Base {\n"
                         b"  constructor() {\n    super(1);\n  }\n"
                         b"  save(): void {\n    super.save();\n  }\n}\n"),
    }
    parsed = {p: TypeScriptAdapter().parse(REPO, p, s) for p, s in batch.items()}
    writer.apply(REPO, parsed, parsed.keys(), {}, epoch())
    assert calls_of(graph_db, "src.store.Store.save") == ["src.base.Base.save"]
    assert calls_of(graph_db, "src.store.Store.constructor") == ["src.base.Base.constructor"]
