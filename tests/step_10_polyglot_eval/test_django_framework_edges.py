"""§3.5's Django row, which no adapter emitted until the final audit.

`DISPATCHES_TO`, `HAS_FIELD` and `USES_SERIALIZER` became writable in v10.1
(R2), and the TypeScript adapter used its half; the Python adapter emitted none
of the three. These pin what it now emits on the Django fixture, that string
references stay invisible (§12.3), and that the edges are written and traversed.
"""

from __future__ import annotations

import time

import pytest

from adapters.python import PythonAdapter
from graph.writer import GraphWriter
from retrieve.structural import StructuralResolver

REPO = "repo_django_edges"


@pytest.fixture(scope="module")
def root(fixtures_dir):
    return fixtures_dir / "repos" / "django_min"


def parse(root, rel: str, source: bytes | None = None):
    data = source if source is not None else (root / rel).read_bytes()
    return PythonAdapter().parse(REPO, rel, data)


def edges_of(parsed, kind: str) -> set[tuple[str, str]]:
    names = {s.uid: s.qualified_name for s in parsed.symbols}
    return {
        (names[e.source_uid], names.get(e.target_uid) or e.target_hint)
        for e in parsed.edges
        if e.kind == kind
    }


def test_urls_dispatch_to_their_views(root):
    assert edges_of(parse(root, "authx/urls.py"), "DISPATCHES_TO") == {
        ("authx.urls", "authx.views.LoginView"),
        ("authx.urls", "authx.views.ProfileView"),
        ("authx.urls", "authx.views.TokenRevokeView"),
    }


def test_include_is_a_string_reference_not_an_edge(root):
    """§12.3: `include("authx.urls")` names a module in a string."""
    assert edges_of(parse(root, "config/urls.py"), "DISPATCHES_TO") == {
        ("config.urls", "authx.views.HealthView"),
    }


def test_function_views_and_re_path_dispatch_too(root):
    src = (
        b"from django.urls import re_path, path\n"
        b"from . import views\n\n"
        b"urlpatterns = [\n"
        b"    re_path(r'^old/$', views.legacy),\n"
        b"    path('x', views.XView.as_view(template='a')),\n"
        b"    path('noview'),\n"
        b"]\n"
    )
    got = edges_of(parse(root, "shop/urls.py", src), "DISPATCHES_TO")
    assert got == {("shop.urls", "shop.views.legacy"), ("shop.urls", "shop.views.XView")}


def test_only_urls_modules_dispatch(root):
    """`path(...)` in any other file is not a route table."""
    src = b"from django.urls import path\nfrom .views import V\nx = path('a', V.as_view())\n"
    assert edges_of(parse(root, "shop/routes_helper.py", src), "DISPATCHES_TO") == set()


def test_views_use_the_serializers_they_instantiate(root):
    assert edges_of(parse(root, "authx/views.py"), "USES_SERIALIZER") == {
        ("authx.views.LoginView.post", "authx.serializers.LoginSerializer"),
        ("authx.views.ProfileView.get", "authx.serializers.UserSerializer"),
        ("authx.views.ProfileView.patch", "authx.serializers.UserSerializer"),
    }


def test_serializer_class_attribute_is_a_use(root):
    src = (
        b"from rest_framework import generics\n"
        b"from .serializers import OrderSerializer\n\n"
        b"class OrderList(generics.ListAPIView):\n"
        b"    serializer_class = OrderSerializer\n"
    )
    assert edges_of(parse(root, "shop/views.py", src), "USES_SERIALIZER") == {
        ("shop.views.OrderList", "shop.serializers.OrderSerializer"),
    }


def test_a_serializer_nesting_another_is_not_a_view_using_it(root):
    src = (
        b"from rest_framework import serializers\n\n"
        b"class LineSerializer(serializers.Serializer):\n"
        b"    pass\n\n"
        b"class OrderSerializer(serializers.Serializer):\n"
        b"    def to_representation(self, obj):\n"
        b"        return LineSerializer(obj.lines, many=True).data\n"
    )
    assert edges_of(parse(root, "shop/serializers.py", src), "USES_SERIALIZER") == set()


def test_relational_fields_link_models(root):
    assert edges_of(parse(root, "billing/models.py"), "HAS_FIELD") == {
        ("billing.models.Subscription", "authx.models.User"),
        ("billing.models.Subscription", "billing.models.Plan"),
    }
    assert edges_of(parse(root, "authx/models.py"), "HAS_FIELD") == {
        ("authx.models.ApiToken", "authx.models.User"),
    }


def test_string_model_references_stay_invisible(root):
    """`ForeignKey("auth.User")` and `settings.AUTH_USER_MODEL` are §12.3's cases."""
    src = (
        b"from django.conf import settings\nfrom django.db import models\n\n"
        b"class Note(models.Model):\n"
        b"    owner = models.ForeignKey('auth.User', on_delete=models.CASCADE)\n"
        b"    title = models.CharField(max_length=10)\n"
    )
    got = edges_of(parse(root, "shop/models.py", src), "HAS_FIELD")
    assert got == set(), "a string target and a scalar field produce no edge"


def test_django_edges_are_written_and_traversed(graph_db, root):
    graph_db.run("MATCH (n) WHERE n.repo_id = $r DETACH DELETE n", r=REPO)
    rels = [
        "authx/__init__.py", "authx/models.py", "authx/serializers.py",
        "authx/services.py", "authx/views.py", "authx/urls.py",
        "billing/__init__.py", "billing/models.py",
        "common/__init__.py", "common/mixins.py", "common/utils.py",
    ]
    batch = {rel: parse(root, rel) for rel in rels}
    GraphWriter(graph_db.driver, graph_db.name).apply(
        REPO, batch, batch.keys(), {}, time.time_ns()
    )

    kinds = {
        r["k"]: r["n"]
        for r in graph_db.run(
            "MATCH (:Symbol {repo_id: $r})-[e]->(:Symbol) RETURN type(e) AS k, count(e) AS n",
            r=REPO,
        )
    }
    assert kinds.get("DISPATCHES_TO") == 3
    assert kinds.get("USES_SERIALIZER") == 3
    assert kinds.get("HAS_FIELD") == 3        # ApiToken->User, Subscription->User, ->Plan

    class Runner:
        def run(self, query, **params):
            return [dict(row) for row in graph_db.run(query, **params)]

    usages = StructuralResolver(runner=Runner()).answer(REPO, "where is `User` used", "usages")
    related = {
        r["qn"] for r in graph_db.run(
            "MATCH (s:Symbol) WHERE s.uid IN $u RETURN s.qualified_name AS qn",
            u=usages.related_uids,
        )
    }
    assert {"authx.models.ApiToken", "billing.models.Subscription"} <= related, (
        "HAS_FIELD makes 'where is User used' include the models that point at it"
    )
    graph_db.run("MATCH (n) WHERE n.repo_id = $r DETACH DELETE n", r=REPO)
