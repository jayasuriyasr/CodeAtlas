"""TypeScript / TSX `LanguageAdapter` (S3).

Plan §10 puts this last deliberately: "if S3 is a real seam, adding a language
is additive. **If it is not, that is a finding about the architecture.**"

Three things differ from Python and each one is a defect if missed:

* **Overloads are two node types.** `function_signature` for the bodiless
  declarations, `function_declaration` for the implementation (FINDINGS F-002).
  An adapter matching only the latter finds one of three `parse` declarations,
  and T1's collision becomes invisible rather than fixed.
* **Rest params hide inside `required_parameter`.** `...rest` is not a
  top-level `rest_pattern` node; it is a `required_parameter` whose *pattern* is
  one. Excluding by node type alone counts it, and §3.2's arity rule says not to.
* **Default exports may be anonymous.** T8's `<module>.default`.
"""

from __future__ import annotations

import re

from adapters.base import (
    Edge,
    ParsedFile,
    StackFrame,
    Symbol,
    content_hash,
    normalize_rel_path,
    symbol_uid,
)
from adapters.grammars import tsx_parser, walk

#: Nodes that become symbols. `function_signature` is here for F-002.
_DEF_NODES = (
    "function_declaration",
    "function_signature",
    "class_declaration",
    "abstract_class_declaration",
    "method_definition",
)

#: Compound statements that can contain a definition. Descended into for the
#: same reason as Python's: a conditionally-defined function is still a symbol.
_DESCEND = (
    "statement_block", "if_statement", "else_clause", "try_statement",
    "catch_clause", "finally_clause", "switch_statement", "switch_case",
    "switch_body", "for_statement", "while_statement", "class_body",
    "export_statement", "lexical_declaration", "variable_declaration",
)

#: React's rule, and the only thing that identifies a hook syntactically.
_HOOK_RE = re.compile(r"^use[A-Z_]")

_WS = re.compile(r"\s+")

def module_qualified_name(rel_path: str) -> str:
    """`app/api/users/route.ts` -> `app.api.users.route`.

    `index.ts` is deliberately *not* collapsed the way Python's `__init__.py`
    is. TypeScript module resolution depends on `tsconfig` paths and the
    bundler, so collapsing it would be a guess — and §3.2 already handles the
    resulting name collisions by keeping `rel_path` in the UID.
    """
    path = normalize_rel_path(rel_path)
    for suffix in (".d.ts", ".tsx", ".ts", ".jsx", ".js", ".mjs", ".cjs"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    return ".".join(p for p in path.split("/") if p)


def _text(node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _squash(text: str) -> str:
    return _WS.sub(" ", text).strip()


# --------------------------------------------------------------------------
# Arity (§3.2, TS/JS row)
# --------------------------------------------------------------------------


def compute_arity(node, source: bytes) -> int:
    """Counted: declared params including optional (`b?`). Excluded: rest, `this`.

    The exclusions are checked on the parameter's *pattern*, not its node type.
    tree-sitter-typescript reports `...rest: number[]` as a `required_parameter`
    whose pattern is a `rest_pattern`, so a type-only check counts it — and a
    signature that gains a rest param would churn every UID in the file.
    """
    params = node.child_by_field_name("parameters")
    if params is None:
        return 0

    count = 0
    for child in params.named_children:
        if child.type not in ("required_parameter", "optional_parameter"):
            continue
        pattern = child.child_by_field_name("pattern")
        if pattern is not None and pattern.type in ("rest_pattern", "this"):
            continue
        count += 1
    return count


# --------------------------------------------------------------------------
# 'use client' (§3.5)
# --------------------------------------------------------------------------


def is_client_component(root, source: bytes) -> bool:
    """The `'use client'` directive, which must be the file's first statement.

    Checked positionally rather than by searching the file: a string `'use
    client'` further down is not a directive, and treating it as one would mark
    a server component as client-side — which §3.5 uses to decide what runs
    where.
    """
    for child in root.named_children:
        if child.type != "expression_statement":
            return False
        inner = child.named_children[0] if child.named_children else None
        if inner is None or inner.type != "string":
            return False
        literal = _text(inner, source).strip("\"'`")
        if literal == "use client":
            return True
        if literal in ("use server", "use strict"):
            continue                       # another directive; keep looking
        return False
    return False


# --------------------------------------------------------------------------
# The adapter
# --------------------------------------------------------------------------


class TypeScriptAdapter:
    """S3 for `.ts` / `.tsx`. The tsx grammar parses both (§0.2's probe)."""

    name = "typescript"
    extensions = (".ts", ".tsx", ".mts", ".cts")

    def parse(self, repo_id: str, rel_path: str, source: bytes) -> ParsedFile:
        rel_path = normalize_rel_path(rel_path)
        module_qname = module_qualified_name(rel_path)
        tree = tsx_parser().parse(source)
        root = tree.root_node

        client = is_client_component(root, source)
        imports = _import_table(root, source)

        parsed = ParsedFile(
            rel_path=rel_path,
            content_hash=content_hash(source),
            language=self.name,
            source=source,
        )

        module_symbol = Symbol(
            uid="",
            repo_id=repo_id,
            rel_path=rel_path,
            qualified_name=module_qname,
            name=module_qname.rsplit(".", 1)[-1] if module_qname else rel_path,
            arity=0,
            ordinal=0,
            kind="module",
            signature=f"module {module_qname}",
            docstring=None,
            source_code=source.decode("utf-8", errors="replace"),
            enclosing_signature=None,
            used_imports=sorted(set(imports.values())),
            start_line=1,
            end_line=root.end_point[0] + 1,
            is_client_component=client,
        )

        collected: list[tuple[Symbol, object]] = [(module_symbol, root)]
        self._collect(
            root,
            source=source,
            repo_id=repo_id,
            rel_path=rel_path,
            scope=module_qname,
            enclosing_signature=None,
            imports=imports,
            client=client,
            out=collected,
        )

        body = collected[1:]
        body.sort(key=lambda pair: pair[1].start_byte)
        collected = [collected[0]] + body

        seen: dict[tuple[str, int], int] = {}
        for sym, _node in collected:
            key = sym.identity
            sym.ordinal = seen.get(key, 0)
            seen[key] = sym.ordinal + 1
            sym.uid = symbol_uid(
                repo_id, rel_path, sym.qualified_name, sym.arity, sym.ordinal
            )

        parsed.symbols = [sym for sym, _ in collected]
        parsed.edges = self._edges(collected, source, rel_path, imports, module_qname)
        return parsed

    # ----------------------------------------------------------------------

    def _collect(
        self,
        node,
        *,
        source: bytes,
        repo_id: str,
        rel_path: str,
        scope: str,
        enclosing_signature: str | None,
        imports: dict[str, str],
        client: bool,
        out: list,
    ) -> None:
        for child in node.named_children:
            target, exported, default = child, False, False

            if child.type == "export_statement":
                exported = True
                default = b"default" in child.text[:32]
                inner = child.child_by_field_name("declaration") or child.child_by_field_name("value")
                if inner is None:
                    continue
                target = inner

            if target.type in _DEF_NODES:
                sym = self._build(
                    target, source=source, repo_id=repo_id, rel_path=rel_path,
                    scope=scope, enclosing_signature=enclosing_signature,
                    imports=imports, client=client, exported=exported, default=default,
                )
                out.append((sym, target))
                body = target.child_by_field_name("body")
                if body is not None:
                    self._collect(
                        body, source=source, repo_id=repo_id, rel_path=rel_path,
                        scope=sym.qualified_name,
                        enclosing_signature=(
                            sym.signature
                            if target.type in ("class_declaration", "abstract_class_declaration")
                            else enclosing_signature
                        ),
                        imports=imports, client=client, out=out,
                    )
                continue

            # `const Foo = () => ...` — §3.2: an arrow function assigned to a
            # binding takes the binding's name. Only a *bound* arrow is a
            # symbol; an inline callback is anonymous and is not indexed.
            if target.type in ("lexical_declaration", "variable_declaration"):
                for declarator in target.named_children:
                    if declarator.type != "variable_declarator":
                        continue
                    value = declarator.child_by_field_name("value")
                    if value is None or value.type not in ("arrow_function", "function_expression"):
                        continue
                    name_node = declarator.child_by_field_name("name")
                    if name_node is None:
                        continue
                    sym = self._build(
                        value, source=source, repo_id=repo_id, rel_path=rel_path,
                        scope=scope, enclosing_signature=enclosing_signature,
                        imports=imports, client=client, exported=exported,
                        default=default, name_override=_text(name_node, source),
                        span_override=declarator,
                    )
                    out.append((sym, value))
                continue

            # `export default () => …` — T8: anonymous, so `<module>.default`.
            if exported and default and target.type in ("arrow_function", "function_expression"):
                sym = self._build(
                    target, source=source, repo_id=repo_id, rel_path=rel_path,
                    scope=scope, enclosing_signature=enclosing_signature,
                    imports=imports, client=client, exported=True, default=True,
                    name_override="default",
                )
                out.append((sym, target))
                continue

            if target.type in _DESCEND:
                self._collect(
                    target, source=source, repo_id=repo_id, rel_path=rel_path,
                    scope=scope, enclosing_signature=enclosing_signature,
                    imports=imports, client=client, out=out,
                )

    def _build(
        self,
        node,
        *,
        source: bytes,
        repo_id: str,
        rel_path: str,
        scope: str,
        enclosing_signature: str | None,
        imports: dict[str, str],
        client: bool,
        exported: bool,
        default: bool,
        name_override: str | None = None,
        span_override=None,
    ) -> Symbol:
        if name_override is not None:
            name = name_override
        else:
            name_node = node.child_by_field_name("name")
            name = _text(name_node, source) if name_node is not None else None
            if name is None:
                # T8: "For anonymous default exports, `<module>.default`."
                name = "default" if default else "<anonymous>"

        qualified_name = f"{scope}.{name}" if scope else name
        is_class = node.type in ("class_declaration", "abstract_class_declaration")
        arity = 0 if is_class else compute_arity(node, source)

        body = node.child_by_field_name("body")
        span = span_override if span_override is not None else node
        if body is not None and body.start_byte > node.start_byte:
            signature = _squash(
                source[node.start_byte : body.start_byte].decode("utf-8", errors="replace")
            )
        else:
            signature = _squash(_text(node, source))[:200]

        return Symbol(
            uid="",
            repo_id=repo_id,
            rel_path=rel_path,
            qualified_name=qualified_name,
            name=name,
            arity=arity,
            ordinal=0,
            kind="class" if is_class else "function",
            signature=signature,
            docstring=_jsdoc(span, source),
            source_code=_text(span, source),
            enclosing_signature=enclosing_signature,
            used_imports=_used_imports(node, source, imports),
            decorators=[],
            start_line=span.start_point[0] + 1,
            end_line=span.end_point[0] + 1,
            is_client_component=client,
        )

    def _edges(
        self,
        collected: list,
        source: bytes,
        rel_path: str,
        imports: dict[str, str],
        module_qname: str,
    ) -> list[Edge]:
        """DEFINES, IMPORTS, and the framework layer (§3.5).

        Since v10.1 the allowlist covers all eight types §3.5 names, so a hook
        invocation is an `INVOKES_HOOK` edge rather than a `CALLS` edge wearing
        a naming convention. Under v10.0 it could not be: the type was
        unwritable and §11.1 would not have traversed it (v10.1 §0.0 R2).
        """
        by_qname = {sym.qualified_name: sym for sym, _ in collected}
        module_uid = collected[0][0].uid
        edges: list[Edge] = []

        for sym, _node in collected[1:]:
            parent_qname = sym.qualified_name.rsplit(".", 1)[0]
            parent = by_qname.get(parent_qname)
            if parent is not None:
                edges.append(
                    Edge(source_uid=parent.uid, kind="DEFINES",
                         origin_path=rel_path, target_uid=sym.uid)
                )

        for target in sorted(set(imports.values())):
            edges.append(
                Edge(source_uid=module_uid, kind="IMPORTS",
                     origin_path=rel_path, target_hint=target)
            )

        owner_by_node = {node.id: sym for sym, node in collected}
        for sym, node in collected[1:]:
            for call in walk(node):
                if call.type != "call_expression":
                    continue
                if _owner_of(call, owner_by_node) is not sym:
                    continue
                callee = call.child_by_field_name("function")
                if callee is None:
                    continue
                edges.append(
                    _call_edge(callee, source, sym, by_qname, imports, module_qname, rel_path)
                )

        edges.extend(_app_router_edges(collected, rel_path, by_qname))
        return edges

    def resolve_frame(self, frame: StackFrame) -> StackFrame:
        """§5.1 Tier 2's habitat.

        The path in a TS/TSX frame is a compiled chunk, so normalising it buys
        nothing — which is precisely why §5.1 resolves these by name and refuses
        on ambiguity rather than trusting the path.
        """
        return frame


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _import_table(root, source: bytes) -> dict[str, str]:
    """Local binding -> module specifier, for the whole file."""
    table: dict[str, str] = {}
    for node in walk(root):
        if node.type != "import_statement":
            continue
        source_node = node.child_by_field_name("source")
        if source_node is None:
            continue
        module = _text(source_node, source).strip("\"'`")
        for child in walk(node):
            if child.type == "import_specifier":
                alias = child.child_by_field_name("alias")
                name = child.child_by_field_name("name")
                local = alias if alias is not None else name
                if local is not None and name is not None:
                    table[_text(local, source)] = f"{module}.{_text(name, source)}"
            elif child.type == "namespace_import":
                for grand in child.named_children:
                    table[_text(grand, source)] = module
            elif child.type == "import_clause":
                for grand in child.named_children:
                    if grand.type == "identifier":     # default import
                        table[_text(grand, source)] = f"{module}.default"
    return table


def _used_imports(node, source: bytes, table: dict[str, str]) -> list[str]:
    found: set[str] = set()
    for child in walk(node):
        if child.type != "identifier":
            continue
        parent = child.parent
        if parent is not None and parent.type == "member_expression":
            if parent.child_by_field_name("property") is child:
                continue
        name = _text(child, source)
        if name in table:
            found.add(table[name])
    return sorted(found)


def _owner_of(node, owner_by_node: dict[int, Symbol]) -> Symbol | None:
    cur = node.parent
    while cur is not None:
        owner = owner_by_node.get(cur.id)
        if owner is not None:
            return owner
        cur = cur.parent
    return None


def _call_edge(callee, source, sym, by_qname, imports, module_qname, rel_path) -> Edge:
    """A call, typed `INVOKES_HOOK` when the callee is a React hook (§3.5).

    React's `use[A-Z]` convention is the only syntactic marker a hook has, and
    it is the same marker the compiler's own rules-of-hooks lint relies on. A
    hook invocation is a distinct relationship from an ordinary call — §5.2
    weights it higher — so typing it as `CALLS` would lose the distinction the
    framework layer exists to capture.
    """
    text = _text(callee, source)
    kind = "INVOKES_HOOK" if is_hook_call(text.rsplit(".", 1)[-1]) else "CALLS"

    local = f"{module_qname}.{text}" if module_qname else text
    if local in by_qname:
        return Edge(source_uid=sym.uid, kind=kind, origin_path=rel_path,
                    target_uid=by_qname[local].uid)

    head, _, rest = text.partition(".")
    hint = f"{imports[head]}.{rest}" if head in imports and rest else imports.get(head, text)
    return Edge(source_uid=sym.uid, kind=kind, origin_path=rel_path, target_hint=hint)


def is_hook_call(name: str) -> bool:
    """React's naming rule is the only syntactic marker a hook has."""
    return bool(_HOOK_RE.match(name))


def hooks_invoked(parsed: ParsedFile) -> dict[str, list[str]]:
    """`qualified_name -> [hook names]`, read off the `INVOKES_HOOK` edges.

    A convenience over the edge list, not a substitute for it. Under v10.0 this
    was the *only* way to answer the question, because the edge type could not
    be written (F-010); since v10.1 the graph answers it directly and this is
    just the parse-time view.
    """
    by_uid = {s.uid: s for s in parsed.symbols}
    out: dict[str, list[str]] = {}
    for edge in parsed.edges:
        if edge.kind != "INVOKES_HOOK":
            continue
        target = edge.target_hint or ""
        name = target.rsplit(".", 1)[-1]
        owner = by_uid.get(edge.source_uid)
        if owner is not None and name:
            out.setdefault(owner.qualified_name, []).append(name)
    return {k: sorted(set(v)) for k, v in out.items()}


#: App Router files that render a component, versus those that handle a request.
#: §3.5 gives both relationships: `(Route)-[:RENDERS]->(Component)` for the
#: React side and `(UrlRoute)-[:DISPATCHES_TO]->(View)` for the Django side.
#: `page.tsx` is the first shape; `route.ts` is exactly the second.
_RENDERING_ROUTE_FILES = ("page", "layout", "template", "loading", "error")
_HANDLER_ROUTE_FILES = ("route",)


def _app_router_edges(collected: list, rel_path: str, by_qname: dict) -> list[Edge]:
    """§3.5's App Router conventions, with the type the relationship deserves.

    A `page.tsx` renders its default export; a `route.ts` dispatches a request
    to `GET`/`POST`. Those are different relationships, and v10.1's allowlist
    can express both — under v10.0 neither `RENDERS` nor any distinction was
    writable (§0.0 R2).
    """
    stem = rel_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    if stem in _RENDERING_ROUTE_FILES:
        kind = "RENDERS"
    elif stem in _HANDLER_ROUTE_FILES:
        kind = "DISPATCHES_TO"
    else:
        return []

    module_sym = collected[0][0]
    return [
        Edge(source_uid=module_sym.uid, kind=kind, origin_path=rel_path,
             target_uid=sym.uid)
        for sym, _node in collected[1:]
        if sym.kind == "function"
    ]


_JSDOC_RE = re.compile(r"/\*\*(.*?)\*/", re.S)


def _jsdoc(node, source: bytes) -> str | None:
    """The JSDoc block immediately above a definition, if there is one."""
    prev = node.prev_sibling
    while prev is not None and not prev.is_named and prev.type != "comment":
        prev = prev.prev_sibling
    if prev is None or prev.type != "comment":
        return None
    match = _JSDOC_RE.match(_text(prev, source))
    if match is None:
        return None
    lines = [
        line.strip().lstrip("*").strip()
        for line in match.group(1).strip().splitlines()
    ]
    body = "\n".join(line for line in lines if line)
    return body or None
