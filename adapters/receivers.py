"""What a Python receiver is, when the code in front of it says so for certain.

`x.method()` is a call to a method of whatever `x` holds. Without types that is
unknowable, so the call used to be left unresolved. This module recovers the
type in the cases where the source states it outright, and in no others:

* a local name bound exactly once in its function, by `x = Callee(...)` or by
  an annotated assignment `x: Cls = ...`;
* a parameter with a plain annotation, `def f(x: Cls)`, never rebound;
* a class attribute bound once in the class body by `name = Callee(...)`
  (Django's `objects = TagManager()`);
* an instance attribute bound exactly once in the class, in `__init__`, by
  `self.name = Callee(...)`.

"Plain" annotation means a name or dotted name: `Optional[Cls]`, `Cls | None`
and string annotations are not trusted. Any other binding of the name — a loop
target, `with … as`, `+=`, a walrus, `global`, unpacking, a second assignment —
makes it unknown. The result is a *candidate* type: whether `Callee` is a class
at all (rather than a factory function) is decided by the graph writer, which
resolves a member lookup only on a class it knows.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable

from adapters.grammars import walk

Resolve = Callable[[str], str]

#: Node types that open a new scope: their bindings are not the function's.
_SCOPES = ("function_definition", "class_definition", "lambda")


def _text(node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _scope_nodes(node):
    """Every node of a function or class body, not descending into nested scopes."""
    stack = list(reversed(node.named_children))
    while stack:
        current = stack.pop()
        yield current
        if current.type in _SCOPES:
            continue
        stack.extend(reversed(current.named_children))


def plain_annotation(type_node, source: bytes) -> str | None:
    """The annotation's text when it is a bare (dotted) name, else None."""
    if type_node is None or len(type_node.named_children) != 1:
        return None
    inner = type_node.named_children[0]
    return _text(inner, source) if inner.type in ("identifier", "attribute") else None


def constructed_type(value, source: bytes) -> str | None:
    """`Callee` for a value of the form `Callee(...)`, else None.

    `self.make()` and chained calls are not constructors of anything nameable.
    """
    if value is None or value.type != "call":
        return None
    callee = value.child_by_field_name("function")
    if callee is None or callee.type not in ("identifier", "attribute"):
        return None
    text = _text(callee, source)
    if text.split(".", 1)[0] in ("self", "cls", "super"):
        return None
    return text


def _poison_targets(node, source: bytes, poisoned: set[str]) -> None:
    for child in walk(node):
        if child.type == "identifier":
            poisoned.add(_text(child, source))


def local_types(function_node, source: bytes, resolve: Resolve) -> dict[str, str]:
    """`name -> type` for the locals and parameters of one function, when certain."""
    counts: Counter[str] = Counter()
    typed: dict[str, str] = {}
    poisoned: set[str] = set()

    params = function_node.child_by_field_name("parameters")
    for param in params.named_children if params is not None else []:
        if param.type == "identifier":
            counts[_text(param, source)] += 1
        elif param.type in ("typed_parameter", "typed_default_parameter"):
            name_node = param.child_by_field_name("name") or param.named_children[0]
            name = _text(name_node, source)
            counts[name] += 1
            annotation = plain_annotation(param.child_by_field_name("type"), source)
            if annotation:
                typed[name] = resolve(annotation)
        elif param.type == "default_parameter":
            name_node = param.child_by_field_name("name")
            if name_node is not None:
                counts[_text(name_node, source)] += 1
        else:                                      # *args, **kwargs, keyword separators
            _poison_targets(param, source, poisoned)

    body = function_node.child_by_field_name("body")
    for node in _scope_nodes(body) if body is not None else []:
        if node.type == "assignment":
            left = node.child_by_field_name("left")
            if left is None:
                continue
            if left.type != "identifier":
                if left.type in ("pattern_list", "tuple_pattern", "list_pattern"):
                    _poison_targets(left, source, poisoned)
                continue                          # self.x = …, a[i] = … bind no local
            name = _text(left, source)
            counts[name] += 1
            annotation = plain_annotation(node.child_by_field_name("type"), source)
            built = constructed_type(node.child_by_field_name("right"), source)
            if annotation:
                typed[name] = resolve(annotation)
            elif built:
                typed[name] = resolve(built)
            else:
                poisoned.add(name)
        elif node.type == "augmented_assignment":
            left = node.child_by_field_name("left")
            if left is not None and left.type == "identifier":
                poisoned.add(_text(left, source))
        elif node.type in ("for_statement", "for_in_clause"):
            left = node.child_by_field_name("left")
            if left is not None:
                _poison_targets(left, source, poisoned)
        elif node.type in ("as_pattern_target", "global_statement", "nonlocal_statement",
                           "delete_statement"):
            _poison_targets(node, source, poisoned)
        elif node.type == "named_expression":
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                poisoned.add(_text(name_node, source))

    return {name: kind for name, kind in typed.items()
            if counts[name] == 1 and name not in poisoned}


def attribute_types(class_node, source: bytes, resolve: Resolve) -> dict[str, str]:
    """`attribute -> type` for one class: class-body and `__init__` bindings."""
    body = class_node.child_by_field_name("body")
    if body is None:
        return {}
    class_counts: Counter[str] = Counter()
    class_typed: dict[str, str] = {}
    instance_counts: Counter[str] = Counter()
    instance_typed: dict[str, str] = {}

    for statement in body.named_children:
        if statement.type == "expression_statement" and statement.named_children:
            assign = statement.named_children[0]
            if assign.type == "assignment":
                left = assign.child_by_field_name("left")
                if left is not None and left.type == "identifier":
                    name = _text(left, source)
                    class_counts[name] += 1
                    annotation = plain_annotation(assign.child_by_field_name("type"), source)
                    built = constructed_type(assign.child_by_field_name("right"), source)
                    if annotation or built:
                        class_typed[name] = resolve(annotation or built)

    for method in body.named_children:
        definition = method
        if method.type == "decorated_definition":
            definition = method.child_by_field_name("definition") or method
        if definition.type != "function_definition":
            continue
        name_node = definition.child_by_field_name("name")
        in_init = name_node is not None and _text(name_node, source) == "__init__"
        method_body = definition.child_by_field_name("body")
        for node in _scope_nodes(method_body) if method_body is not None else []:
            if node.type not in ("assignment", "augmented_assignment"):
                continue
            left = node.child_by_field_name("left")
            if left is None or left.type != "attribute":
                continue
            obj = left.child_by_field_name("object")
            attr = left.child_by_field_name("attribute")
            if obj is None or attr is None or _text(obj, source) != "self":
                continue
            name = _text(attr, source)
            instance_counts[name] += 1
            built = constructed_type(node.child_by_field_name("right"), source) \
                if node.type == "assignment" else None
            if in_init and built:
                instance_typed[name] = resolve(built)

    out: dict[str, str] = {}
    for name, kind in class_typed.items():
        if class_counts[name] == 1 and name not in instance_counts:
            out[name] = kind
    for name, kind in instance_typed.items():
        if instance_counts[name] == 1 and name not in class_counts:
            out[name] = kind
    return out
