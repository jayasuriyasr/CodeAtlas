"""Python `LanguageAdapter` (S3).

Produces `ParsedFile{symbols, edges, content_hash}` from one file's bytes. The
hard part is not extracting functions — it is computing `arity` and `ordinal`
identically on every parse, because §3.2 makes UID stability depend on exactly
that, and a UID that churns silently severs every inbound edge.
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
from adapters.grammars import python_parser, walk

#: Definition nodes that become symbols. `lambda` is deliberately absent:
#: §3.2 says truly anonymous functions are not symbols and appear only inside
#: their enclosing symbol's body. A Python lambda has no name to qualify, even
#: when it is assigned to one.
_DEF_NODES = ("function_definition", "class_definition")

#: Parameter nodes that count toward arity (§3.2, Python row: positional,
#: keyword-only, and params with defaults).
_COUNTED_PARAMS = (
    "identifier",
    "typed_parameter",
    "default_parameter",
    "typed_default_parameter",
)

#: Excluded from arity: `*args` and `**kwargs` (§3.2). `positional_separator`
#: (`/`) and `keyword_separator` (`*`) are punctuation, not parameters — they
#: change how the *counted* ones may be passed, never how many there are.
_EXCLUDED_PARAMS = ("list_splat_pattern", "dictionary_splat_pattern")
_SEPARATORS = ("positional_separator", "keyword_separator")

#: §3.2: excluded from arity when they are the receiver of a method.
_RECEIVER_NAMES = ("self", "cls")

_WS = re.compile(r"\s+")


def module_qualified_name(rel_path: str) -> str:
    """`auth/views.py` -> `auth.views`; `auth/__init__.py` -> `auth`."""
    path = normalize_rel_path(rel_path)
    for suffix in (".py", ".pyi"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    parts = [p for p in path.split("/") if p]
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def package_of(rel_path: str) -> str:
    """The package a module's relative imports are resolved against.

    For `authx/views.py` that is `authx`. For `authx/__init__.py` it is `authx`
    too — the file *is* the package, so `.models` inside it means
    `authx.models`, not `models`.
    """
    path = normalize_rel_path(rel_path)
    if path.rsplit("/", 1)[-1] in ("__init__.py", "__init__.pyi"):
        return module_qualified_name(rel_path)
    module = module_qualified_name(rel_path)
    return module.rsplit(".", 1)[0] if "." in module else ""


def absolutize(dotted: str, package: str) -> str:
    """Resolve a relative import against the importing file's package.

    `used_imports` keeps the relative form, because §3.4's worked example shows
    `.models.User` in the context header and that is what the developer wrote.
    Edge targets cannot: §11.2 3b matches by uid, and a `.services.issue_token`
    hint matches nothing in a graph whose symbols are keyed on absolute
    qualified names. Every cross-file CALLS edge inside a package would be
    dropped, which would leave the graph with only intra-file structure —
    Principle 2's "the graph must earn its keep" quietly false.
    """
    if not dotted.startswith("."):
        return dotted

    level = len(dotted) - len(dotted.lstrip("."))
    remainder = dotted[level:]

    parts = package.split(".") if package else []
    ascend = level - 1                    # one dot means "this package"
    base = parts[: len(parts) - ascend] if ascend else parts
    if ascend and len(parts) - ascend < 0:
        base = []                         # climbed past the root; keep it simple

    return ".".join([*base, remainder]) if remainder else ".".join(base)


def _text(node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _squash(text: str) -> str:
    """Collapse whitespace runs in a signature.

    A signature wrapped across lines by a formatter is the same signature. Since
    §4.2 hashes `signature` into `header_hash`, not collapsing would make
    `black` reformatting invalidate every embedding in the file.
    """
    return _WS.sub(" ", text).strip()


# --------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------


def _import_table(root, source: bytes) -> dict[str, str]:
    """Local binding -> resolved dotted path, for the whole file.

    Relative imports keep their leading dots, matching §3.4's worked example
    (`.models.User`), because resolving them needs the package layout and
    guessing it would be worse than showing the developer what the file says.
    """
    table: dict[str, str] = {}

    for node in walk(root):
        if node.type == "import_statement":
            for child in node.children_by_field_name("name"):
                if child.type == "aliased_import":
                    name = child.child_by_field_name("name")
                    alias = child.child_by_field_name("alias")
                    if name is not None and alias is not None:
                        table[_text(alias, source)] = _text(name, source)
                elif child.type == "dotted_name":
                    dotted = _text(child, source)
                    # `import os.path` binds `os`, not `os.path`.
                    table[dotted.split(".")[0]] = dotted

        elif node.type == "import_from_statement":
            module = node.child_by_field_name("module_name")
            prefix = _text(module, source) if module is not None else ""
            for child in node.children_by_field_name("name"):
                if child.type == "aliased_import":
                    name = child.child_by_field_name("name")
                    alias = child.child_by_field_name("alias")
                    if name is None or alias is None:
                        continue
                    local, target = _text(alias, source), _text(name, source)
                elif child.type == "dotted_name":
                    local = target = _text(child, source)
                else:
                    continue
                joiner = "" if prefix.endswith(".") else "."
                table[local] = f"{prefix}{joiner}{target}" if prefix else target

    return table


def _used_imports(
    node, source: bytes, table: dict[str, str], *, skip_member_bodies: bool = False
) -> list[str]:
    """The imports *this symbol* references (§3.4).

    Built from the symbol's own subtree, never from the file's import block —
    §3.4 is explicit that populating it file-wide makes the context header a
    function of the whole file, so one added import invalidates every cached
    embedding in it.

    `skip_member_bodies` applies the same argument one level down, for classes.
    §3.4.4 defines a class chunk as signature + docstring + method *signature*
    list — the bodies are separate symbols. So an import used only inside a
    method is not part of the class's chunk, and letting it reach the class's
    `header_hash` would make editing any method body invalidate the class's
    cached embedding for no change to what the class chunk actually shows.

    Two identifier positions are skipped because they are never a reference to
    an imported binding: the attribute half of `obj.attr`, and parameter names.
    Both would otherwise produce false hits against a same-named import, and a
    false hit means a spurious cache miss.
    """
    found: set[str] = set()

    if skip_member_bodies:
        subtree = _walk_skipping_member_bodies(node)
    else:
        subtree = walk(node)

    for child in subtree:
        if child.type != "identifier":
            continue
        parent = child.parent
        if parent is None:
            continue
        if parent.type == "attribute" and parent.child_by_field_name("attribute") is child:
            continue
        if parent.type == "keyword_argument" and parent.child_by_field_name("name") is child:
            continue
        if parent.type in ("parameters", "typed_parameter", "default_parameter",
                           "typed_default_parameter"):
            continue
        name = _text(child, source)
        if name in table:
            found.add(table[name])

    return sorted(found)


def _walk_skipping_member_bodies(root):
    """Depth-first over a class, stopping at each nested definition's body.

    Signatures, decorators, base classes and class-level assignments are
    visited; method bodies are not.
    """
    stack = [root]
    while stack:
        cur = stack.pop()
        yield cur

        nested_def = cur.type in _DEF_NODES and cur.id != root.id
        body = cur.child_by_field_name("body") if nested_def else None
        for child in reversed(cur.children):
            if body is not None and child.id == body.id:
                continue                  # a member's body is its own symbol
            stack.append(child)


# --------------------------------------------------------------------------
# Arity (§3.2)
# --------------------------------------------------------------------------


def _param_name(node, source: bytes) -> str | None:
    if node.type == "identifier":
        return _text(node, source)
    if node.type == "typed_parameter":
        for child in node.children:
            if child.type == "identifier":
                return _text(child, source)
        return None
    name = node.child_by_field_name("name")
    return _text(name, source) if name is not None else None


def compute_arity(func_node, source: bytes, *, is_method: bool) -> int:
    """§3.2, Python row.

    Counted: positional, keyword-only, and params with defaults.
    Excluded: `self`, `cls`, `*args`, `**kwargs`.

    Default *values* are not counted, which is the point of the rule: changing
    `timeout=30` to `timeout=60` must not churn the UID.
    """
    params = func_node.child_by_field_name("parameters")
    if params is None:
        return 0

    count = 0
    first = True
    for child in params.children:
        if not child.is_named or child.type in _SEPARATORS:
            continue
        if child.type in _EXCLUDED_PARAMS:
            first = False
            continue
        if child.type not in _COUNTED_PARAMS:
            continue
        name = _param_name(child, source)
        if first and is_method and name in _RECEIVER_NAMES:
            first = False
            continue                      # the receiver, excluded by §3.2
        first = False
        count += 1

    return count


# --------------------------------------------------------------------------
# The adapter
# --------------------------------------------------------------------------


class PythonAdapter:
    """S3 implementation for Python. Stateless; safe to share."""

    name = "python"
    extensions = (".py", ".pyi")

    def parse(self, repo_id: str, rel_path: str, source: bytes) -> ParsedFile:
        rel_path = normalize_rel_path(rel_path)
        module_qname = module_qualified_name(rel_path)
        tree = python_parser().parse(source)
        table = _import_table(tree.root_node, source)

        parsed = ParsedFile(
            rel_path=rel_path,
            content_hash=content_hash(source),
            language=self.name,
            source=source,
        )

        # One :Module symbol per file (§3.1), so DEFINES has a parent to hang
        # top-level symbols from and §3.4.5's file chunk has a home.
        module_symbol = Symbol(
            uid="",                       # assigned below, with the ordinals
            repo_id=repo_id,
            rel_path=rel_path,
            qualified_name=module_qname,
            name=module_qname.rsplit(".", 1)[-1] if module_qname else rel_path,
            arity=0,
            ordinal=0,
            kind="module",
            signature=f"module {module_qname}",
            docstring=_docstring(tree.root_node, source),
            source_code=source.decode("utf-8", errors="replace"),
            enclosing_signature=None,
            used_imports=sorted(set(table.values())),
            start_line=1,
            end_line=tree.root_node.end_point[0] + 1,
        )

        collected: list[tuple[Symbol, object]] = [(module_symbol, tree.root_node)]
        self._collect(
            tree.root_node,
            source=source,
            repo_id=repo_id,
            rel_path=rel_path,
            scope=module_qname,
            enclosing_signature=None,
            in_class_body=False,
            table=table,
            out=collected,
        )

        # Source order, then ordinals within each (qualified_name, arity) group.
        body_symbols = collected[1:]
        body_symbols.sort(key=lambda pair: pair[1].start_byte)
        collected = [collected[0]] + body_symbols

        seen: dict[tuple[str, int], int] = {}
        for sym, _node in collected:
            key = sym.identity
            sym.ordinal = seen.get(key, 0)
            seen[key] = sym.ordinal + 1
            sym.uid = symbol_uid(
                repo_id, rel_path, sym.qualified_name, sym.arity, sym.ordinal
            )

        parsed.symbols = [sym for sym, _ in collected]
        parsed.edges = self._edges(collected, source, rel_path, table, module_qname)
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
        in_class_body: bool,
        table: dict[str, str],
        out: list,
    ) -> None:
        """Walk one scope's statements, descending into definitions and blocks.

        Compound statements are descended into because that is where the T1
        collision lives: `if TYPE_CHECKING:` and platform guards define the same
        function twice, at the same arity, in one file. A collector that only
        looked at direct children of `module` would find one of them, and the
        ordinal would have nothing to disambiguate.
        """
        for child in node.named_children:
            target = child
            decorators: list[str] = []

            if child.type == "decorated_definition":
                decorators = [
                    _text(d, source) for d in child.children if d.type == "decorator"
                ]
                inner = child.child_by_field_name("definition")
                if inner is None:
                    continue
                target = inner

            if target.type in _DEF_NODES:
                sym = self._build_symbol(
                    target,
                    outer=child,          # decorated_definition, when present
                    source=source,
                    repo_id=repo_id,
                    rel_path=rel_path,
                    scope=scope,
                    enclosing_signature=enclosing_signature,
                    in_class_body=in_class_body,
                    table=table,
                    decorators=decorators,
                )
                out.append((sym, target))

                body = target.child_by_field_name("body")
                if body is not None:
                    self._collect(
                        body,
                        source=source,
                        repo_id=repo_id,
                        rel_path=rel_path,
                        scope=sym.qualified_name,
                        enclosing_signature=(
                            sym.signature
                            if target.type == "class_definition"
                            else enclosing_signature
                        ),
                        in_class_body=target.type == "class_definition",
                        table=table,
                        out=out,
                    )
                continue

            # Not a definition: descend anyway if it can contain one.
            if target.type in (
                "if_statement", "else_clause", "elif_clause", "try_statement",
                "except_clause", "finally_clause", "with_statement", "for_statement",
                "while_statement", "block", "match_statement", "case_clause",
                "decorated_definition",
            ):
                self._collect(
                    target,
                    source=source,
                    repo_id=repo_id,
                    rel_path=rel_path,
                    scope=scope,
                    enclosing_signature=enclosing_signature,
                    in_class_body=in_class_body,
                    table=table,
                    out=out,
                )

    def _build_symbol(
        self,
        node,
        *,
        outer,
        source: bytes,
        repo_id: str,
        rel_path: str,
        scope: str,
        enclosing_signature: str | None,
        in_class_body: bool,
        table: dict[str, str],
        decorators: list[str],
    ) -> Symbol:
        name_node = node.child_by_field_name("name")
        name = _text(name_node, source) if name_node is not None else "<anonymous>"
        qualified_name = f"{scope}.{name}" if scope else name

        is_class = node.type == "class_definition"
        arity = 0 if is_class else compute_arity(node, source, is_method=in_class_body)

        body = node.child_by_field_name("body")
        signature = _squash(
            source[node.start_byte : body.start_byte].decode("utf-8", errors="replace")
            if body is not None
            else _text(node, source)
        )

        # Decorators are part of the symbol: a changed decorator changes what the
        # symbol does, so it must reach body_hash (§4.2) and invalidate the cache.
        span = outer if outer.type == "decorated_definition" else node

        return Symbol(
            uid="",
            repo_id=repo_id,
            rel_path=rel_path,
            qualified_name=qualified_name,
            name=name,
            arity=arity,
            ordinal=0,                    # assigned after the full source-order sort
            kind="class" if is_class else "function",
            signature=signature,
            docstring=_docstring(body, source) if body is not None else None,
            source_code=_text(span, source),
            enclosing_signature=enclosing_signature,
            used_imports=_used_imports(node, source, table, skip_member_bodies=is_class),
            decorators=decorators,
            start_line=span.start_point[0] + 1,
            end_line=span.end_point[0] + 1,
        )

    def _edges(
        self,
        collected: list,
        source: bytes,
        rel_path: str,
        table: dict[str, str],
        module_qname: str,
    ) -> list[Edge]:
        """DEFINES, CALLS and IMPORTS for one file.

        Only intra-file targets get a `target_uid`. Cross-file calls carry a
        `target_hint` and are resolved against the graph later — §11.2 3b
        matches both endpoints by uid, so an unresolved edge is simply never
        written, which is the right answer for a call into a third-party
        library.
        """
        by_qname = {sym.qualified_name: sym for sym, _ in collected}
        edges: list[Edge] = []
        module_uid = collected[0][0].uid

        # DEFINES: parent scope -> symbol. Always intra-file, always resolvable.
        for sym, _node in collected[1:]:
            parent_qname = sym.qualified_name.rsplit(".", 1)[0]
            parent = by_qname.get(parent_qname)
            if parent is not None:
                edges.append(
                    Edge(
                        source_uid=parent.uid,
                        kind="DEFINES",
                        origin_path=rel_path,
                        target_uid=sym.uid,
                    )
                )

        # IMPORTS: the module symbol -> each import, absolutised for matching.
        package = package_of(rel_path)
        for target in sorted({absolutize(t, package) for t in table.values()}):
            edges.append(
                Edge(
                    source_uid=module_uid,
                    kind="IMPORTS",
                    origin_path=rel_path,
                    target_hint=target,
                )
            )

        # CALLS: attributed to the innermost enclosing symbol.
        #
        # Keyed on `node.id`, never Python's `id()`. py-tree-sitter builds a
        # fresh wrapper object on every `.parent` / `.children` access, so
        # `id()` compares wrapper addresses — which CPython happily reuses after
        # a temporary is freed, making it look correct in a two-line test and
        # silently match nothing here. `node.id` is the underlying node pointer.
        owner_by_node = {node.id: sym for sym, node in collected}
        for sym, node in collected[1:]:
            for call in walk(node):
                if call.type != "call":
                    continue
                if _owner_of(call, owner_by_node) is not sym:
                    continue              # belongs to a nested symbol
                callee = call.child_by_field_name("function")
                if callee is None:
                    continue
                edges.append(
                    _call_edge(
                        callee, source, sym, by_qname, table, module_qname,
                        rel_path, package,
                    )
                )

        return edges

    def resolve_frame(self, frame: StackFrame) -> StackFrame:
        """Tier 0 is a graph lookup on `(repo_id, rel_path, start_line)` (§5.1).

        Python tracebacks already carry an exact file and line, so the adapter's
        only job is normalising the path; the resolution itself belongs to the
        frame resolver in step 7.
        """
        if frame.file_path is None:
            return frame
        return StackFrame(
            raw=frame.raw,
            file_path=normalize_rel_path(frame.file_path),
            line_no=frame.line_no,
            fn_name=frame.fn_name,
        )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _owner_of(node, owner_by_node: dict[int, Symbol]) -> Symbol | None:
    """The innermost symbol whose definition node encloses `node`."""
    cur = node.parent
    while cur is not None:
        owner = owner_by_node.get(cur.id)
        if owner is not None:
            return owner
        cur = cur.parent
    return None


def _call_edge(
    callee,
    source: bytes,
    sym: Symbol,
    by_qname: dict[str, Symbol],
    table: dict[str, str],
    module_qname: str,
    rel_path: str,
    package: str,
) -> Edge:
    text = _text(callee, source)
    local = f"{module_qname}.{text}" if module_qname else text

    if local in by_qname:                 # same-file call, resolvable now
        return Edge(
            source_uid=sym.uid,
            kind="CALLS",
            origin_path=rel_path,
            target_uid=by_qname[local].uid,
        )

    head, _, rest = text.partition(".")
    if head in table:
        resolved = absolutize(table[head], package)
        hint = f"{resolved}.{rest}" if rest else resolved
    else:
        hint = text
    return Edge(source_uid=sym.uid, kind="CALLS", origin_path=rel_path, target_hint=hint)


def _docstring(body, source: bytes) -> str | None:
    """The first string literal in a body, per PEP 257."""
    if body is None:
        return None
    for child in body.named_children:
        if child.type != "expression_statement":
            return None
        inner = child.named_children[0] if child.named_children else None
        if inner is None or inner.type != "string":
            return None
        for part in inner.named_children:
            if part.type == "string_content":
                return _text(part, source).strip()
        return _text(inner, source).strip("\"'").strip()
    return None
