"""Step 1 probe 6 — does the TSX grammar emit what §3.4's chunker assumes?

The chunker's unit is one symbol (§3.4.1), its header needs the enclosing
signature and the symbol's own free identifiers (§3.4.2), and §3.2 requires
telling three same-arity `parse` overloads apart by source order. All three
depend on tree-sitter-typescript emitting specific node types for real
Next.js code. §0.2 lists this as unverified and the plan says stop if it fails.

The probe records the node types the grammar *actually* emits over the two
fixture files, then checks the assumed set against it. Nothing here is
expectation: `ASSUMED` is what the chunker will be written against, and the
recorded list is what the parser produced.
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURE_REPO = Path(__file__).resolve().parents[1] / "fixtures" / "repos" / "nextjs_min"
ROUTE = FIXTURE_REPO / "app" / "api" / "users" / "route.ts"
COMPONENT = FIXTURE_REPO / "components" / "UserCard.tsx"


#: Node types the §3.4 chunker and the §3.2 UID rules are written against.
#: Grouped by which spec obligation needs them, so a miss says what breaks.
ASSUMED: dict[str, set[str]] = {
    "§3.4.1 symbol boundaries": {
        "function_declaration",
        "class_declaration",
        "method_definition",
        "arrow_function",
        "variable_declarator",
        "lexical_declaration",
        "statement_block",
    },
    "§3.2 arity rule (optional counted, rest excluded)": {
        "formal_parameters",
        "required_parameter",
        "optional_parameter",
        "rest_pattern",
    },
    "§3.2 qualified names / default export": {
        "export_statement",
    },
    "§3.4.2 header: resolved imports": {
        "import_statement",
        "identifier",
    },
    "§3.5 framework layer (hooks, JSX, 'use client')": {
        "call_expression",
        "jsx_element",
        "jsx_self_closing_element",
        "expression_statement",
        "string",
    },
    "TS surface the adapter must not choke on": {
        "interface_declaration",
        "type_annotation",
        "comment",
    },
}

ALL_ASSUMED = set().union(*ASSUMED.values())


def _load_tsx_parser():
    """Build a TSX parser, tolerating the tree-sitter 0.22 -> 0.23 API change."""
    tree_sitter = pytest.importorskip("tree_sitter", reason="tree-sitter not installed")
    ts_ts = pytest.importorskip(
        "tree_sitter_typescript", reason="tree-sitter-typescript not installed"
    )

    language = tree_sitter.Language(ts_ts.language_tsx())
    try:                                  # tree-sitter >= 0.23
        return tree_sitter.Parser(language), tree_sitter
    except TypeError:                     # tree-sitter 0.22 and earlier
        parser = tree_sitter.Parser()
        parser.set_language(language)
        return parser, tree_sitter


def _walk(node):
    stack = [node]
    while stack:
        cur = stack.pop()
        yield cur
        stack.extend(reversed(cur.children))


def _node_types(root) -> set[str]:
    """Named node types only — anonymous tokens like ';' are not grammar facts."""
    return {n.type for n in _walk(root) if n.is_named}


def _errors(root) -> list[str]:
    out = []
    for n in _walk(root):
        if n.type == "ERROR" or n.is_missing:
            out.append(f"{n.type} at line {n.start_point[0] + 1}: {n.text[:60]!r}")
    return out


def test_probe_tsx_grammar(probe_log):
    parser, _ = _load_tsx_parser()

    rec = probe_log.record(
        "probe_tsx_grammar",
        "Does `tree-sitter-typescript` (tsx) emit the node types the chunker "
        "assumes for a real Next.js route + component?",
    )
    rec.statement(
        "Language(tree_sitter_typescript.language_tsx())\n"
        f"parse({ROUTE.relative_to(FIXTURE_REPO.parents[2])})\n"
        f"parse({COMPONENT.relative_to(FIXTURE_REPO.parents[2])})"
    )

    seen: set[str] = set()
    for path in (ROUTE, COMPONENT):
        tree = parser.parse(path.read_bytes())
        errs = _errors(tree.root_node)
        rec.observe(f"{path.name} root type", tree.root_node.type)
        rec.observe(f"{path.name} parse errors", errs or "none")
        seen |= _node_types(tree.root_node)

    rec.observe("distinct named node types", len(seen))
    rec.observe("node types emitted", ", ".join(sorted(seen)))

    missing = {
        obligation: sorted(types - seen)
        for obligation, types in ASSUMED.items()
        if types - seen
    }
    for obligation, gone in missing.items():
        rec.observe(f"MISSING for {obligation}", gone)

    rec.conclude(
        "ALL ASSUMED NODE TYPES PRESENT"
        if not missing
        else f"MISSING {sorted(set().union(*missing.values()))} — plan §1 says stop"
    )

    assert not missing, (
        "tsx grammar does not emit node types the chunker assumes: "
        f"{missing}. Plan §1: stop and resolve."
    )


def test_probe_tsx_grammar_overloads_are_distinguishable(probe_log):
    """§3.2's T1 case, in its original habitat.

    Three `parse` declarations share qualified_name *and* arity. The UID's
    source-order ordinal is the fix, and it needs the grammar to surface all
    three as separate top-level nodes with usable byte offsets. If overloads
    collapsed in the parse tree, the ordinal would have nothing to count.
    """
    parser, _ = _load_tsx_parser()
    rec = probe_log.records["probe_tsx_grammar"]

    tree = parser.parse(ROUTE.read_bytes())
    parse_decls = [
        n
        for n in _walk(tree.root_node)
        if n.type in ("function_declaration", "function_signature")
        and (n.child_by_field_name("name") is not None)
        and n.child_by_field_name("name").text == b"parse"
    ]
    rec.observe(
        "overload declarations of `parse`",
        [
            (n.type, n.start_point[0] + 1, n.child_by_field_name("name").text.decode())
            for n in sorted(parse_decls, key=lambda n: n.start_byte)
        ],
    )

    # 'use client' directive detection (§3.5, is_client_component).
    comp_tree = parser.parse(COMPONENT.read_bytes())
    first = next(
        (c for c in comp_tree.root_node.children if c.is_named), None
    )
    rec.observe(
        "first named node of UserCard.tsx",
        f"{first.type}: {first.text[:20].decode()!r}" if first else "none",
    )

    assert len(parse_decls) == 3, (
        "expected the three `parse` overloads as separate declarations; got "
        f"{[(n.type, n.start_point[0] + 1) for n in parse_decls]}"
    )
    starts = sorted(n.start_byte for n in parse_decls)
    assert len(set(starts)) == 3, "overloads must have distinct byte offsets to ordinal"
