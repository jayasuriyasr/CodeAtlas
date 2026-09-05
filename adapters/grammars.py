"""tree-sitter grammar loading, in one place.

Two things are centralised here. First, the 0.22 -> 0.23 `Parser` constructor
change, which would otherwise be duplicated in every adapter. Second, parsers
are cached per language: building a `Language` is not free, and §4's cost table
budgets ~5ms per file for the whole parse.
"""

from __future__ import annotations

from functools import lru_cache

import tree_sitter


def _build(language: tree_sitter.Language) -> tree_sitter.Parser:
    try:                                        # tree-sitter >= 0.23
        return tree_sitter.Parser(language)
    except TypeError:                           # tree-sitter <= 0.22
        parser = tree_sitter.Parser()
        parser.set_language(language)
        return parser


@lru_cache(maxsize=None)
def python_parser() -> tree_sitter.Parser:
    import tree_sitter_python

    return _build(tree_sitter.Language(tree_sitter_python.language()))


@lru_cache(maxsize=None)
def typescript_parser() -> tree_sitter.Parser:
    import tree_sitter_typescript

    return _build(tree_sitter.Language(tree_sitter_typescript.language_typescript()))


@lru_cache(maxsize=None)
def tsx_parser() -> tree_sitter.Parser:
    import tree_sitter_typescript

    return _build(tree_sitter.Language(tree_sitter_typescript.language_tsx()))


def walk(node):
    """Depth-first pre-order over a subtree, children in source order."""
    stack = [node]
    while stack:
        cur = stack.pop()
        yield cur
        stack.extend(reversed(cur.children))
