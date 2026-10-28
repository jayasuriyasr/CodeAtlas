"""TypeScript import specifiers as edge targets.

The adapter used the raw specifier as the edge hint: `../lib/users` from
`app/api/users/route.ts` became `../lib/users.getUserById`, a dotted name with
empty segments that can match no symbol. Every cross-file TS edge was dropped.
Relative specifiers and the `@/` alias now resolve to the module qualified name
the target file actually has; bare packages stay unresolvable, as they should.
"""

from __future__ import annotations

import pytest

from adapters.typescript import TypeScriptAdapter, resolve_specifier


@pytest.mark.parametrize(
    "spec, importer, expected",
    [
        ("../lib/users", "app/api/users/route.ts", "app.api.lib.users"),
        ("../../lib/users", "app/api/users/route.ts", "app.lib.users"),
        ("./helpers", "components/UserCard.tsx", "components.helpers"),
        ("./helpers.ts", "components/UserCard.tsx", "components.helpers"),
        ("@/lib/users", "app/api/users/route.ts", "lib.users"),
        ("@/components/OrderList", "app/orders/page.tsx", "components.OrderList"),
        ("react", "components/UserCard.tsx", "react"),
        ("next/server", "app/api/users/route.ts", "next/server"),
        ("../../../outside", "app/api/route.ts", "../../../outside"),
        ("./", "lib/index.ts", "./"),
    ],
)
def test_resolve_specifier(spec, importer, expected):
    assert resolve_specifier(spec, importer) == expected


def edges(source: str, rel_path: str):
    return TypeScriptAdapter().parse("r", rel_path, source.encode()).edges


def test_call_hints_use_resolved_module_names():
    src = (
        "import { getUserById } from '../../lib/users';\n"
        "export async function GET(req: Request) {\n"
        "  return getUserById('1');\n"
        "}\n"
    )
    hints = {e.target_hint for e in edges(src, "app/api/route.ts") if e.kind == "CALLS"}
    assert hints == {"lib.users.getUserById"}


def test_import_edges_use_resolved_module_names():
    src = (
        "import OrderList from '@/components/OrderList';\n"
        "import * as api from './api';\n"
        "import { useState } from 'react';\n"
        "export default function Page() { return OrderList; }\n"
    )
    hints = {e.target_hint for e in edges(src, "app/orders/page.tsx") if e.kind == "IMPORTS"}
    assert hints == {"components.OrderList.default", "app.orders.api", "react.useState"}


def test_the_context_header_keeps_what_the_developer_wrote():
    """Resolution changes edge targets, not `used_imports`: the header (§3.4)
    and its cache key stay as written, so a resolution rule change does not
    invalidate every cached TS embedding."""
    src = (
        "import { getUserById } from '../../lib/users';\n"
        "export function load() { return getUserById('1'); }\n"
    )
    parsed = TypeScriptAdapter().parse("r", "app/api/route.ts", src.encode())
    load = next(s for s in parsed.symbols if s.name == "load")
    assert load.used_imports == ["../../lib/users.getUserById"]


def test_member_calls_on_a_namespace_import_resolve_through_it():
    src = (
        "import * as users from '@/lib/users';\n"
        "export function load() { return users.getUserById('1'); }\n"
    )
    hints = {e.target_hint for e in edges(src, "app/x.ts") if e.kind == "CALLS"}
    assert hints == {"lib.users.getUserById"}
