"""Member lookups and INHERITS edges, against a real graph.

The adapter turns `self.method()` into a *member lookup* when the method is not
defined in the same file (`test_member_calls.py`). The writer resolves it by
walking the class's method resolution order — C3, the order Python itself
uses — over the classes the graph knows. Two rules keep that honest:

* an unknown base (a library class, or one defined twice) earlier in the order
  than the first repository definition is a refusal: the library class may
  well define the method itself;
* two definitions of the method on one class is a refusal, as everywhere else.
"""

from __future__ import annotations

from adapters.python import PythonAdapter
from adapters.typescript import TypeScriptAdapter

from .conftest import REPO, epoch

PY = PythonAdapter()
TS = TypeScriptAdapter()


def index(writer, files: dict[str, bytes]):
    batch = {}
    for rel_path, source in files.items():
        adapter = TS if rel_path.endswith((".ts", ".tsx")) else PY
        batch[rel_path] = adapter.parse(REPO, rel_path, source)
    return writer.apply(REPO, batch, batch.keys(), {}, epoch())


def calls_of(graph_db, qualified_name: str) -> list[str]:
    rows = graph_db.run(
        """
        MATCH (s:Symbol {repo_id: $r, qualified_name: $qn})-[:CALLS]->(t:Symbol)
        RETURN t.qualified_name AS qn ORDER BY qn
        """,
        r=REPO, qn=qualified_name,
    )
    return [r["qn"] for r in rows]


def fixture(fixtures_dir, *paths: str) -> dict[str, bytes]:
    root = fixtures_dir / "repos" / "django_min"
    return {p: (root / p).read_bytes() for p in paths}


# -- the fixture's own case -------------------------------------------------------


def test_an_inherited_method_call_reaches_the_base_class(writer, graph_db, fixtures_dir):
    """`Subscription.cancel` calls `self.soft_delete()`; golden question S14
    ("callers of soft_delete") depends on this edge existing."""
    index(writer, fixture(fixtures_dir, "billing/models.py", "common/mixins.py", "authx/models.py"))
    assert "common.mixins.SoftDeleteMixin.soft_delete" in calls_of(
        graph_db, "billing.models.Subscription.cancel")


def test_a_library_method_is_not_attributed_to_the_repository(writer, graph_db, fixtures_dir):
    """`User.display_name` calls `self.get_full_name()`, Django's
    `AbstractUser`'s. No repository class defines it, so there is no edge."""
    index(writer, fixture(fixtures_dir, "authx/models.py", "common/mixins.py"))
    assert calls_of(graph_db, "authx.models.User.display_name") == []


def test_inherits_edges_are_written_to_repository_classes_only(writer, graph_db, fixtures_dir):
    index(writer, fixture(fixtures_dir, "billing/models.py", "common/mixins.py", "authx/models.py"))
    rows = graph_db.run(
        """
        MATCH (c:Symbol {repo_id: $r})-[:INHERITS]->(b:Symbol)
        RETURN c.qualified_name AS child, b.qualified_name AS base ORDER BY child, base
        """,
        r=REPO,
    )
    assert [(r["child"], r["base"]) for r in rows] == [
        ("authx.models.ApiToken", "common.mixins.TimestampMixin"),
        ("authx.models.User", "common.mixins.TimestampMixin"),
        ("billing.models.Plan", "common.mixins.TimestampMixin"),
        ("billing.models.Subscription", "common.mixins.SoftDeleteMixin"),
        ("billing.models.Subscription", "common.mixins.TimestampMixin"),
    ]


# -- the order -------------------------------------------------------------------


DIAMOND = {
    "pkg/a.py": b"class A:\n    def m(self):\n        return 'A'\n",
    "pkg/b.py": b"from pkg.a import A\n\n\nclass B(A):\n    def m(self):\n        return 'B'\n",
    "pkg/c.py": b"from pkg.a import A\n\n\nclass C(A):\n    pass\n",
    "pkg/d.py": (b"from pkg.b import B\nfrom pkg.c import C\n\n\n"
                 b"class D(C, B):\n    def run(self):\n        return self.m()\n"),
}


def test_the_order_is_c3_not_depth_first(writer, graph_db):
    """D(C, B) with C(A), B(A): Python's MRO is D, C, B, A, so `self.m()` is
    B's. Walking depth-first would reach A through C first and pick A.m."""
    index(writer, DIAMOND)
    assert calls_of(graph_db, "pkg.d.D.run") == ["pkg.b.B.m"]


def test_a_library_base_earlier_in_the_order_is_a_refusal(writer, graph_db):
    """`Base` from a library precedes the repository's mixin, and may define
    `save` itself. Pointing at the mixin would be a guess."""
    report = index(writer, {
        "app/mixins.py": b"class Audited:\n    def save(self):\n        pass\n",
        "app/models.py": (b"from library import Base\nfrom app.mixins import Audited\n\n\n"
                          b"class Order(Base, Audited):\n    def close(self):\n        self.save()\n"),
    })
    assert calls_of(graph_db, "app.models.Order.close") == []
    assert report.edges_unresolved >= 1


def test_the_repository_base_wins_when_it_comes_first(writer, graph_db):
    index(writer, {
        "app/mixins.py": b"class Audited:\n    def save(self):\n        pass\n",
        "app/models.py": (b"from library import Base\nfrom app.mixins import Audited\n\n\n"
                          b"class Order(Audited, Base):\n    def close(self):\n        self.save()\n"),
    })
    assert calls_of(graph_db, "app.models.Order.close") == ["app.mixins.Audited.save"]


def test_a_base_defined_twice_is_a_refusal(writer, graph_db):
    """Two classes named `pkg.base.Base` (a conditional definition): which
    one `Child` inherits from depends on a branch taken at import time."""
    report = index(writer, {
        "pkg/base.py": (b"import sys\nif sys.platform == 'win32':\n"
                        b"    class Base:\n        def go(self):\n            pass\n"
                        b"else:\n    class Base:\n        def go(self):\n            pass\n"),
        "pkg/child.py": (b"from pkg.base import Base\n\n\n"
                         b"class Child(Base):\n    def run(self):\n        self.go()\n"),
    })
    assert calls_of(graph_db, "pkg.child.Child.run") == []
    assert report.edges_unresolved + report.edges_ambiguous >= 1


def test_an_inherits_hint_is_never_trimmed_onto_a_neighbour(writer, graph_db):
    """`Config.Nested` does not exist; trimming it to `Config` — what call
    hints do — would record a base class the code never named."""
    index(writer, {
        "pkg/config.py": b"class Config:\n    pass\n",
        "pkg/app.py": b"from pkg.config import Config\n\n\nclass App(Config.Nested):\n    pass\n",
    })
    rows = graph_db.run(
        "MATCH (:Symbol {repo_id: $r, qualified_name: 'pkg.app.App'})-[:INHERITS]->(b) RETURN b",
        r=REPO,
    )
    assert rows == []


def test_a_cycle_in_the_hierarchy_does_not_hang_or_guess(writer, graph_db):
    report = index(writer, {
        "pkg/x.py": b"from pkg.y import Y\n\n\nclass X(Y):\n    def run(self):\n        self.go()\n",
        "pkg/y.py": b"from pkg.x import X\n\n\nclass Y(X):\n    def go(self):\n        pass\n",
    })
    assert calls_of(graph_db, "pkg.x.X.run") == []
    assert report.edges_unresolved + report.edges_ambiguous >= 1


# -- TypeScript ------------------------------------------------------------------


def test_a_typescript_this_call_reaches_the_parent_class(writer, graph_db):
    index(writer, {
        "src/base.ts": b"export class Base {\n  persist(): void {}\n}\n",
        "src/store.ts": (b'import { Base } from "./base";\n\n'
                         b"export class Store extends Base {\n  save(): void {\n    this.persist();\n  }\n}\n"),
    })
    assert calls_of(graph_db, "src.store.Store.save") == ["src.base.Base.persist"]
