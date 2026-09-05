"""Step 10 — the TS/TSX adapter.

Plan §10 frames this as the architecture's own test: "if S3 is a real seam,
adding a language is additive. **If it is not, that is a finding about the
architecture**". The result is recorded in PROGRESS.md rather than asserted
here — what these tests check is that the second language obeys the same UID
contract as the first, on the cases §3.2 uses to motivate it.
"""

from __future__ import annotations

import pytest

from adapters.base import symbol_uid
from adapters.typescript import (
    TypeScriptAdapter,
    compute_arity,
    hooks_invoked,
    is_client_component,
    is_hook_call,
    module_qualified_name,
)
from adapters.grammars import tsx_parser

REPO = "repo_step10"


@pytest.fixture(scope="module")
def adapter() -> TypeScriptAdapter:
    return TypeScriptAdapter()


@pytest.fixture(scope="module")
def nextjs(fixtures_dir):
    return fixtures_dir / "repos" / "nextjs_min"


@pytest.fixture(scope="module")
def route(adapter, nextjs):
    path = nextjs / "app" / "api" / "users" / "route.ts"
    return adapter.parse(REPO, "app/api/users/route.ts", path.read_bytes())


@pytest.fixture(scope="module")
def component(adapter, nextjs):
    path = nextjs / "components" / "UserCard.tsx"
    return adapter.parse(REPO, "components/UserCard.tsx", path.read_bytes())


def parse(adapter, source: str, rel_path: str = "src/mod.ts"):
    return adapter.parse(REPO, rel_path, source.encode())


def by_qname(parsed, qualified_name: str):
    return [s for s in parsed.symbols if s.qualified_name == qualified_name]


# --------------------------------------------------------------------------
# T1, in its original habitat
# --------------------------------------------------------------------------


def test_ts_overload_uids_distinct(route):
    """T1. Three `parse()` overloads, one qualified_name, one arity.

    §3.2 uses exactly this example: without the source-order ordinal they
    "produce three identical UIDs, and `MERGE` collapses them into one node
    whose properties are whichever declaration was written last."

    F-002 is the trap underneath it: two of the three are `function_signature`
    nodes, not `function_declaration`. An adapter matching only the latter finds
    one, assigns ordinal 0, and the collision never becomes visible.
    """
    overloads = by_qname(route, "app.api.users.route.parse")

    assert len(overloads) == 3, (
        f"expected three declarations, found {len(overloads)} — the bodiless "
        f"`function_signature` overloads were probably skipped (F-002)"
    )
    assert {s.arity for s in overloads} == {1}, "same arity is the precondition"
    assert [s.ordinal for s in overloads] == [0, 1, 2]
    assert len({s.uid for s in overloads}) == 3

    # And the UIDs are exactly what §3.2's function produces.
    for sym in overloads:
        assert sym.uid == symbol_uid(
            REPO, "app/api/users/route.ts", sym.qualified_name, 1, sym.ordinal
        )


def test_overload_ordinals_follow_source_order(route):
    overloads = by_qname(route, "app.api.users.route.parse")
    assert [s.start_line for s in overloads] == sorted(s.start_line for s in overloads)


def test_no_uid_collisions_anywhere_in_the_fixture(route, component):
    for parsed in (route, component):
        assert len(parsed.uids) == len(set(parsed.uids)), parsed.rel_path


# --------------------------------------------------------------------------
# T2 — the TS arity rule
# --------------------------------------------------------------------------


ARITY_CASES = """
export function required(a: string, b: number): void {}
export function optional(a: string, b?: number): void {}
export function rest(a: string, ...others: number[]): void {}
export function onlyRest(...others: number[]): void {}
export function withThis(this: Window, a: number): void {}
export function defaulted(a: string, b: number = 5): void {}
export function none(): void {}
export class Widget {
  method(a: string, b?: number): void {}
}
"""


@pytest.mark.parametrize(
    "qualified_name, expected, why",
    [
        ("src.mod.required", 2, "declared params counted"),
        ("src.mod.optional", 2, "§3.2: optional (`b?`) is counted"),
        ("src.mod.rest", 1, "§3.2: rest (`...r`) is excluded"),
        ("src.mod.onlyRest", 0, "a rest-only signature has arity 0"),
        ("src.mod.withThis", 1, "§3.2: the `this` param is excluded"),
        ("src.mod.defaulted", 2, "a default is still a declared param"),
        ("src.mod.none", 0, "no params"),
        ("src.mod.Widget.method", 2, "methods follow the same rule"),
    ],
)
def test_ts_arity_optional_and_rest(adapter, qualified_name, expected, why):
    """§3.2's TS/JS row: optional counted, rest and `this` excluded.

    The rest case is the one with a trap: tree-sitter reports `...others` as a
    `required_parameter` whose *pattern* is a `rest_pattern`, so excluding by
    node type alone counts it.
    """
    parsed = parse(adapter, ARITY_CASES)
    found = by_qname(parsed, qualified_name)
    assert len(found) == 1, qualified_name
    assert found[0].arity == expected, why


def test_adding_a_rest_param_does_not_churn_the_uid(adapter):
    """The point of excluding rest params.

    A signature that gains `...args` has not become a different function, and
    churning its UID would sever every inbound CALLS edge (§4.4).
    """
    before = parse(adapter, "export function f(a: string): void {}")
    after = parse(adapter, "export function f(a: string, ...rest: number[]): void {}")
    assert by_qname(before, "src.mod.f")[0].uid == by_qname(after, "src.mod.f")[0].uid


def test_making_a_param_optional_does_change_the_uid(adapter):
    """The converse: optional params *are* counted, so arity really moves."""
    before = parse(adapter, "export function f(a: string): void {}")
    after = parse(adapter, "export function f(a: string, b?: number): void {}")
    assert by_qname(before, "src.mod.f")[0].uid != by_qname(after, "src.mod.f")[0].uid


def test_compute_arity_is_callable_on_its_own():
    source = b"function f(this: W, a: string, b?: number, ...r: any[]): void {}"
    node = tsx_parser().parse(source).root_node.named_children[0]
    assert compute_arity(node, source) == 2


# --------------------------------------------------------------------------
# T8 — default exports
# --------------------------------------------------------------------------


def test_default_export_qualified_name(adapter):
    """T8. "For anonymous default exports, `<module>.default`."

    Without a rule, an anonymous default export has no name to qualify — and a
    Next.js page is exactly that by convention.
    """
    parsed = parse(adapter, "export default () => <span />;\n", "app/page.tsx")
    names = {s.qualified_name for s in parsed.symbols if s.kind == "function"}
    assert names == {"app.page.default"}


def test_named_default_export_keeps_its_name(adapter):
    """`export default function UserCard()` is not anonymous.

    Collapsing it to `.default` would lose the name a developer searches for,
    and make every page in the tree share one qualified name.
    """
    parsed = parse(
        adapter, "export default function UserCard(props: P) { return null; }\n",
        "components/UserCard.tsx",
    )
    assert by_qname(parsed, "components.UserCard.UserCard")


def test_arrow_assigned_to_a_binding_takes_the_binding_name(component):
    """§3.2: "Arrow functions assigned to a binding take the binding's name"."""
    assert by_qname(component, "components.UserCard.AnonymousWrapper")


def test_inline_callbacks_are_not_symbols(component):
    """§3.2: "Truly anonymous functions — IIFEs, inline callbacks — are not
    symbols and are not indexed."

    `UserCard.tsx` passes several arrow callbacks to `useEffect`, `then` and
    `useCallback`. Indexing them would fill the graph with unnamed one-liners
    that compete with real symbols in retrieval.
    """
    names = {s.qualified_name for s in component.symbols}
    assert not any("<anonymous>" in n for n in names), names
    assert len(component.symbols) == 6, sorted(names)


@pytest.mark.parametrize(
    "rel_path, expected",
    [
        ("app/api/users/route.ts", "app.api.users.route"),
        ("components/UserCard.tsx", "components.UserCard"),
        ("src/lib/index.ts", "src.lib.index"),
        ("src/types.d.ts", "src.types"),
    ],
)
def test_module_qualified_name(rel_path, expected):
    """`index.ts` is deliberately not collapsed — see the docstring for why."""
    assert module_qualified_name(rel_path) == expected


# --------------------------------------------------------------------------
# The App Router collision — step 7's refusal is load-bearing
# --------------------------------------------------------------------------


def test_app_router_get_handlers_collide_by_name(adapter, nextjs):
    """Confirms §5.1's Tier 2 refusal guards a real case, not a hypothetical.

    Every `route.ts` in the tree exports a function named `GET`. §3.2 gives this
    as the reason `rel_path` stays in the UID; §5.1 gives it as the reason Tier 2
    refuses. Both are the same fact, and this is it.
    """
    routes = sorted(nextjs.rglob("route.ts"))
    assert len(routes) >= 2, "the fixture no longer carries the collision"

    handlers = []
    for path in routes:
        rel = path.relative_to(nextjs).as_posix()
        parsed = adapter.parse(REPO, rel, path.read_bytes())
        handlers.extend(s for s in parsed.symbols if s.name == "GET")

    assert len(handlers) >= 2
    assert len({s.name for s in handlers}) == 1, "the bare name is identical"
    assert len({s.qualified_name for s in handlers}) == len(handlers), (
        "qualified names differ — because the module path differs"
    )
    assert len({s.uid for s in handlers}) == len(handlers), (
        "UIDs collapsed; rel_path is in the UID precisely to stop this (§3.2)"
    )


# --------------------------------------------------------------------------
# The framework layer (§3.5)
# --------------------------------------------------------------------------


def test_use_client_directive_detected(component, route):
    """§3.5: `is_client_component` from `'use client'`.

    Applied to every symbol in the file, because the directive is file-scoped —
    it is what decides whether the code runs in the browser.
    """
    assert all(s.is_client_component for s in component.symbols)
    assert not any(s.is_client_component for s in route.symbols)


def test_use_client_must_be_the_first_statement(adapter):
    """A string further down the file is not a directive.

    Searching for the literal would mark a server component as client-side, and
    §3.5 uses this property to decide where code runs.
    """
    real = parse(adapter, "'use client';\nexport function f() {}\n")
    assert real.symbols[0].is_client_component

    fake = parse(adapter, "export function f() {\n  const s = 'use client';\n}\n")
    assert not fake.symbols[0].is_client_component


def test_use_client_survives_a_preceding_directive(adapter):
    source = "'use strict';\n'use client';\nexport function f() {}\n"
    assert parse(adapter, source).symbols[0].is_client_component


@pytest.mark.parametrize(
    "name, expected",
    [("useState", True), ("useEffect", True), ("use_thing", True),
     ("user", False), ("used", False), ("useless", False), ("compute", False)],
)
def test_hook_naming_rule(name, expected):
    """React's naming convention is the only syntactic marker a hook has."""
    assert is_hook_call(name) is expected


def test_hooks_are_traceable(component):
    """§3.5's `INVOKES_HOOK` intent, reachable despite F-010.

    The relationship type does not survive §11.2's allowlist, so hooks are
    emitted as `CALLS` and surfaced through this helper. The question stays
    answerable; only the edge label is lost.
    """
    hooks = hooks_invoked(component)
    assert "components.UserCard.UserCard" in hooks
    assert set(hooks["components.UserCard.UserCard"]) >= {
        "useState", "useEffect", "useCallback"
    }


def test_app_router_dispatches_to_its_handlers(route):
    """§3.5's `(Route)-[:RENDERS]->(Component)`, as `DISPATCHES_TO` (F-010)."""
    from config import EDGE_TYPE_ALLOWLIST

    dispatches = [e for e in route.edges if e.kind == "DISPATCHES_TO"]
    assert dispatches, "the route file produced no dispatch edges"

    handlers = {s.uid for s in route.symbols if s.name in ("GET", "POST")}
    assert handlers <= {e.target_uid for e in dispatches}
    assert {e.kind for e in route.edges} <= set(EDGE_TYPE_ALLOWLIST), (
        "an edge kind outside the §11.2 3b allowlist would be written and "
        "never traversed (F-010)"
    )


def test_a_non_route_file_produces_no_dispatch_edges(component):
    assert not [e for e in component.edges if e.kind == "DISPATCHES_TO"]


# --------------------------------------------------------------------------
# Determinism and the shared contract
# --------------------------------------------------------------------------


def test_parse_is_deterministic(adapter, nextjs):
    source = (nextjs / "components" / "UserCard.tsx").read_bytes()
    first = adapter.parse(REPO, "components/UserCard.tsx", source)
    second = adapter.parse(REPO, "components/UserCard.tsx", source)
    assert first.uids == second.uids
    assert first.content_hash == second.content_hash


def test_ts_and_python_adapters_share_the_uid_contract(adapter):
    """S3's actual claim: one identity scheme, two languages.

    If the two adapters disagreed about how a UID is built, a repo containing
    both would have two identity systems and no way to notice.
    """
    from adapters.python import PythonAdapter

    ts = parse(adapter, "export function handler(a: string): void {}", "src/mod.ts")
    py = PythonAdapter().parse(REPO, "src/mod.py", b"def handler(a):\n    pass\n")

    ts_sym = by_qname(ts, "src.mod.handler")[0]
    py_sym = [s for s in py.symbols if s.qualified_name == "src.mod.handler"][0]

    assert ts_sym.arity == py_sym.arity == 1
    assert ts_sym.uid == symbol_uid(REPO, "src/mod.ts", "src.mod.handler", 1, 0)
    assert py_sym.uid == symbol_uid(REPO, "src/mod.py", "src.mod.handler", 1, 0)
    assert ts_sym.uid != py_sym.uid, "different files, different symbols"


def test_adapter_satisfies_the_s3_protocol(adapter):
    from adapters.base import LanguageAdapter

    assert isinstance(adapter, LanguageAdapter)
    assert adapter.extensions and adapter.name


def test_every_symbol_has_the_properties_3_3_requires(route, component):
    for parsed in (route, component):
        for sym in parsed.symbols:
            assert sym.uid and len(sym.uid) == 20
            assert sym.qualified_name and sym.name
            assert sym.kind in ("module", "class", "function")
            assert sym.arity >= 0 and sym.ordinal >= 0
            assert sym.signature
            assert sym.source_code is not None
            assert 1 <= sym.start_line <= sym.end_line
