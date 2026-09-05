"""Step 2 — what `PythonAdapter.parse` produces beyond the UID.

`used_imports` and `enclosing_signature` are checked here rather than at step 4
because they are parse-time facts. Step 4 hashes them into `header_hash`, and a
cache defect there is invisible; a wrong `used_imports` here is visible.
"""

from __future__ import annotations

import pytest

from adapters.python import PythonAdapter

REPO = "repo_step2"


@pytest.fixture(scope="module")
def adapter() -> PythonAdapter:
    return PythonAdapter()


@pytest.fixture(scope="module")
def django_repo(fixtures_dir):
    return fixtures_dir / "repos" / "django_min"


@pytest.fixture(scope="module")
def views(adapter, django_repo):
    path = django_repo / "authx" / "views.py"
    return adapter.parse(REPO, "authx/views.py", path.read_bytes())


def one(parsed, qualified_name):
    found = [s for s in parsed.symbols if s.qualified_name == qualified_name]
    assert len(found) == 1, f"expected exactly one {qualified_name}, got {len(found)}"
    return found[0]


# --------------------------------------------------------------------------
# Symbol content
# --------------------------------------------------------------------------


def test_enclosing_signature_includes_base_classes(views):
    """§3.3: `enclosing_signature` is "full, incl. bases".

    Step 4's `test_base_class_rename_invalidates` (v9.1 T11) depends entirely on
    this: if the bases were stripped, `class X(A)` -> `class X(B)` would produce
    an identical header and serve a stale embedding.
    """
    post = one(views, "authx.views.LoginView.post")
    assert post.enclosing_signature == "class LoginView(APIView):"

    login_view = one(views, "authx.views.LoginView")
    assert login_view.enclosing_signature is None, "a top-level class encloses nothing"


def test_used_imports_are_per_symbol_not_per_file(views):
    """§3.4's central rule, checked at the source.

    `mask_email` is imported by the file and referenced only by `LoginView.post`.
    If `used_imports` were built from the file's import block, every symbol in
    the file would list it — and §3.4 spells out the consequence: one added
    import invalidates every cached embedding in the file.
    """
    post = one(views, "authx.views.LoginView.post")
    health_get = one(views, "authx.views.HealthView.get")

    assert "common.utils.mask_email" in post.used_imports
    assert "common.utils.mask_email" not in health_get.used_imports

    # The file-level module symbol *does* see everything — §3.4.5's file chunk
    # is the import block plus the top-level symbol list.
    module = one(views, "authx.views")
    assert "common.utils.mask_email" in module.used_imports


def test_used_imports_resolve_relative_and_aliased_forms(adapter):
    """§3.4's worked example keeps the leading dot on relative imports."""
    parsed = adapter.parse(
        REPO,
        "authx/views.py",
        b"import os.path as osp\n"
        b"from .models import User\n"
        b"from ..common import utils as u\n"
        b"from rest_framework.response import Response\n"
        b"\n"
        b"def handler(request):\n"
        b"    return Response(osp.join(u.tmp(), User.name))\n",
    )
    handler = one(parsed, "authx.views.handler")
    assert handler.used_imports == [
        "..common.utils",
        ".models.User",
        "os.path",
        "rest_framework.response.Response",
    ]


def test_class_used_imports_exclude_method_bodies(adapter):
    """§3.4.4: a class chunk is signature + docstring + method signatures.

    An import used only inside a method body is not shown in the class chunk, so
    letting it into the class's header would make every method-body edit
    invalidate the class's cached embedding for no visible change.
    """
    parsed = adapter.parse(
        REPO,
        "authx/views.py",
        b"from .base import APIView\n"
        b"from .deep import only_in_a_body\n"
        b"\n"
        b"class V(APIView):\n"
        b"    def post(self, request):\n"
        b"        return only_in_a_body(request)\n",
    )
    cls = one(parsed, "authx.views.V")
    method = one(parsed, "authx.views.V.post")

    assert cls.used_imports == [".base.APIView"]
    assert ".deep.only_in_a_body" in method.used_imports


def test_signature_is_whitespace_normalised(adapter):
    """A signature wrapped by a formatter is the same signature.

    §4.2 hashes `signature` into `header_hash`, so if reformatting changed it,
    running `black` would invalidate every embedding in the repository.
    """
    flat = adapter.parse(REPO, "m.py", b"def f(a, b, c):\n    pass\n")
    wrapped = adapter.parse(
        REPO, "m.py", b"def f(\n    a,\n    b,\n    c,\n):\n    pass\n"
    )
    assert one(flat, "m.f").signature == "def f(a, b, c):"
    assert one(wrapped, "m.f").signature == "def f( a, b, c, ):"
    assert one(flat, "m.f").uid == one(wrapped, "m.f").uid


def test_docstrings_extracted(views):
    assert one(views, "authx.views.LoginView").docstring == (
        "Exchange credentials for an API token."
    )
    assert one(views, "authx.views.LoginView.post").docstring == (
        "Validate credentials and issue a token."
    )
    assert one(views, "authx.views.TokenRevokeView").docstring is None


def test_kinds_match_the_label_set(views):
    """§3.1's labels are Function, Class, Module. There is no Method label."""
    assert {s.kind for s in views.symbols} == {"module", "class", "function"}
    assert one(views, "authx.views.LoginView.post").kind == "function"


def test_content_hash_ignores_line_endings(adapter):
    """§4.5 diffs on `content_hash`.

    If CRLF changed it, a checkout with `core.autocrlf=true` would report every
    file modified and re-embed the entire tree on first reconcile.
    """
    lf = adapter.parse(REPO, "m.py", b"def f():\n    pass\n")
    crlf = adapter.parse(REPO, "m.py", b"def f():\r\n    pass\r\n")
    assert lf.content_hash == crlf.content_hash


# --------------------------------------------------------------------------
# Edges
# --------------------------------------------------------------------------


def test_defines_edges_link_parent_to_child(views):
    """DEFINES is always intra-file, so it always resolves to a UID now."""
    defines = [e for e in views.edges if e.kind == "DEFINES"]
    assert defines, "expected DEFINES edges"
    assert all(e.target_uid is not None for e in defines)

    login_view = one(views, "authx.views.LoginView")
    post = one(views, "authx.views.LoginView.post")
    assert any(
        e.source_uid == login_view.uid and e.target_uid == post.uid for e in defines
    )

    module = one(views, "authx.views")
    assert any(
        e.source_uid == module.uid and e.target_uid == login_view.uid for e in defines
    )


def test_calls_edges_are_attributed_to_the_innermost_symbol(adapter):
    """A call inside a nested function belongs to the nested function."""
    parsed = adapter.parse(
        REPO,
        "m.py",
        b"def outer():\n"
        b"    def inner():\n"
        b"        deep()\n"
        b"    shallow()\n"
        b"    return inner\n",
    )
    outer = one(parsed, "m.outer")
    inner = one(parsed, "m.outer.inner")
    calls = [e for e in parsed.edges if e.kind == "CALLS"]

    hints = {(e.source_uid, e.target_hint) for e in calls if e.target_hint}
    assert (inner.uid, "deep") in hints
    assert (outer.uid, "shallow") in hints
    assert (outer.uid, "deep") not in hints


def test_intra_file_calls_resolve_to_a_uid(adapter):
    """A call to a same-file function needs no graph lookup to resolve."""
    parsed = adapter.parse(
        REPO, "m.py", b"def helper():\n    pass\n\n\ndef caller():\n    helper()\n"
    )
    helper = one(parsed, "m.helper")
    caller = one(parsed, "m.caller")
    assert any(
        e.kind == "CALLS" and e.source_uid == caller.uid and e.target_uid == helper.uid
        for e in parsed.edges
    )


def test_cross_file_calls_carry_a_resolved_hint(views):
    """An unresolved target is a hint, never a fabricated UID.

    §11.2 3b matches both endpoints by uid, so an edge that never resolves is
    simply never written — the right answer for a call into a third-party
    library, and much better than inventing a node for it.

    The hint is *absolute*. `used_imports` keeps `.services.verify_credentials`
    because that is what §3.4's header shows; an edge target cannot, because a
    leading dot matches no qualified name in the graph and every intra-package
    call would be silently dropped.
    """
    calls = [e for e in views.edges if e.kind == "CALLS"]
    post = one(views, "authx.views.LoginView.post")
    hints = {e.target_hint for e in calls if e.source_uid == post.uid}

    assert "authx.services.verify_credentials" in hints
    assert "common.utils.audit_event" in hints
    assert not any(h and h.startswith(".") for h in hints), (
        "a relative hint can never match a symbol's qualified_name"
    )
    assert all(
        e.target_uid is None or e.target_hint is None for e in calls
    ), "an edge resolves or it hints, never both"


@pytest.mark.parametrize(
    "dotted, package, expected",
    [
        (".services.issue_token", "authx", "authx.services.issue_token"),
        (".models.User", "authx", "authx.models.User"),
        ("..common.utils", "authx.sub", "authx.common.utils"),
        ("rest_framework.response", "authx", "rest_framework.response"),
        (".", "authx", "authx"),
    ],
)
def test_absolutize(dotted, package, expected):
    from adapters.python import absolutize

    assert absolutize(dotted, package) == expected


@pytest.mark.parametrize(
    "rel_path, expected",
    [
        ("authx/views.py", "authx"),
        ("authx/__init__.py", "authx"),
        ("manage.py", ""),
        ("a/b/c.py", "a.b"),
    ],
)
def test_package_of(rel_path, expected):
    from adapters.python import package_of

    assert package_of(rel_path) == expected


def test_edge_kinds_stay_inside_the_allowlist(views):
    """§11.2 3b substitutes the relationship type from a fixed allowlist,
    never from parsed input. The adapter must not invent a fifth kind."""
    from config import EDGE_TYPE_ALLOWLIST

    assert {e.kind for e in views.edges} <= set(EDGE_TYPE_ALLOWLIST)


def test_imports_edges_come_from_the_module_symbol(views):
    module = one(views, "authx.views")
    imports = [e for e in views.edges if e.kind == "IMPORTS"]
    assert imports
    assert all(e.source_uid == module.uid for e in imports)
    assert "authx.services.issue_token" in {e.target_hint for e in imports}


# --------------------------------------------------------------------------
# The step 2 gate — a whole tree, zero collisions
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def whole_tree(adapter, django_repo):
    """Parse every .py in the Django fixture, as the indexer would."""
    parsed = {}
    for path in sorted(django_repo.rglob("*.py")):
        rel = path.relative_to(django_repo).as_posix()
        parsed[rel] = adapter.parse(REPO, rel, path.read_bytes())
    return parsed


def test_django_fixture_parses_without_collisions(whole_tree):
    """Plan §2 gate: zero UID collisions across the whole tree."""
    seen: dict[str, str] = {}
    collisions: list[tuple[str, str, str]] = []

    for rel, parsed in whole_tree.items():
        for sym in parsed.symbols:
            previous = seen.get(sym.uid)
            if previous is not None:
                collisions.append((sym.uid, previous, f"{rel}::{sym.qualified_name}"))
            seen[sym.uid] = f"{rel}::{sym.qualified_name}"

    assert not collisions, f"UID collisions: {collisions}"
    assert len(seen) > 60, f"fixture too small to be evidence: {len(seen)} symbols"


def test_django_fixture_has_the_t1_shape_in_it(whole_tree):
    """The gate is only evidence if the fixture contains the failure case.

    `authx/services.py` defines `clock_skew_seconds` twice under a platform
    guard. A fixture with no same-name/same-arity pair would pass the collision
    check without exercising the ordinal at all.
    """
    services = whole_tree["authx/services.py"]
    skews = [
        s for s in services.symbols
        if s.qualified_name == "authx.services.clock_skew_seconds"
    ]
    assert len(skews) == 2
    assert len({s.uid for s in skews}) == 2
    assert [s.ordinal for s in skews] == [0, 1]


def test_every_symbol_has_the_properties_3_3_requires(whole_tree):
    """A missing property surfaces at step 3 as a null column, not an error."""
    for rel, parsed in whole_tree.items():
        for sym in parsed.symbols:
            assert sym.uid and len(sym.uid) == 20, rel
            assert sym.repo_id == REPO
            assert sym.rel_path == rel
            assert sym.qualified_name
            assert sym.name
            assert sym.kind in ("module", "class", "function")
            assert sym.arity >= 0
            assert sym.ordinal >= 0
            assert sym.signature
            # An empty `__init__.py` is a legitimate package marker with no
            # source, so the module symbol's text may be empty — but never None,
            # which would break §4.2's body_hash.
            assert sym.source_code is not None
            if sym.kind != "module":
                assert sym.source_code
            assert 1 <= sym.start_line <= sym.end_line


def test_whole_tree_parse_is_deterministic(adapter, django_repo, whole_tree):
    """Re-parse the tree; every UID set must be identical."""
    for rel, parsed in whole_tree.items():
        again = adapter.parse(REPO, rel, (django_repo / rel).read_bytes())
        assert again.uids == parsed.uids, rel
