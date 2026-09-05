"""Step 6 — §5.3's identifier tokenizer, both sides of it.

No database. These are the pure functions §5.3 specifies, and they are worth
isolating because the failure they prevent is invisible from the outside: the
index still returns something, just never the thing that was asked for.
"""

from __future__ import annotations

import pytest

from adapters.base import Symbol
from retrieve.tokenize import (
    build_fulltext_query,
    build_search_text,
    split_identifier,
    tokenize_query,
)


def symbol(name: str, *, qualified_name: str = "", docstring: str | None = None) -> Symbol:
    return Symbol(
        uid="u", repo_id="r", rel_path="m.py",
        qualified_name=qualified_name or f"m.{name}", name=name,
        arity=0, ordinal=0, kind="function", signature=f"def {name}():",
        docstring=docstring, source_code="", enclosing_signature=None,
    )


# --------------------------------------------------------------------------
# split_identifier — §5.3's worked table
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("get_user_by_id", ["get", "user", "by", "id"]),
        ("getUserById", ["get", "User", "By", "Id"]),
        ("HTTPResponseHandler", ["HTTP", "Response", "Handler"]),
        ("MAX_RETRY_COUNT", ["MAX", "RETRY", "COUNT"]),
        ("oauth2Client", ["oauth", "2", "Client"]),
        ("simple", ["simple"]),
        ("a.b/c:d-e", ["a", "b", "c", "d", "e"]),
        ("", []),
    ],
)
def test_split_identifier(name, expected):
    """Every row of §5.3's table, plus the separators the regex names."""
    assert split_identifier(name) == expected


def test_snake_and_camel_cross_match():
    """`get_user_by_id` and `getUserById` must normalize identically.

    This is the whole reason §5.3 exists. Under UAX#29 an underscore joins
    rather than separates, so both forms stay one opaque token and the query
    "get user by id" scores zero against either.
    """
    snake = build_search_text(symbol("get_user_by_id"))
    camel = build_search_text(symbol("getUserById"))

    assert "get user by id" in snake
    assert "get user by id" in camel
    assert "getuserbyid" in snake
    assert "getuserbyid" in camel

    # Each contains the *other's* exact spelling, via the rejoined forms.
    assert "get_user_by_id" in camel
    assert "getuserbyid" in snake


def test_exact_identifier_still_matches():
    """The split forms must not replace the exact one.

    A developer's most likely query is the identifier itself, and dropping the
    exact spelling to save index space would break precisely that.
    """
    text = build_search_text(symbol("getUserById", qualified_name="common.utils.getUserById"))
    assert "getUserById" in text
    assert "common.utils.getUserById" in text


def test_search_text_includes_the_docstring():
    text = build_search_text(symbol("mask_email", docstring="Reduce an email for logs."))
    assert "Reduce an email for logs." in text


def test_search_text_is_deterministic():
    """§8.2 runs eval at temperature 0; retrieval inputs must not drift."""
    sym = symbol("get_user_by_id", docstring="Fetch one user.")
    assert build_search_text(sym) == build_search_text(sym)


# --------------------------------------------------------------------------
# The query side — "the invisible half-failure"
# --------------------------------------------------------------------------


def test_query_side_tokenized():
    """§5.3: "Tokenizing one side only is a common and invisible failure."

    Invisible because nothing errors: the index answers, the ranking looks
    plausible, and the right symbol is simply never in it.
    """
    assert tokenize_query("getUserById") == ["getuserbyid", "get", "user", "by", "id"]
    assert tokenize_query("get_user_by_id") == ["get_user_by_id", "get", "user", "by", "id"]


def test_query_and_index_sides_agree_on_the_split():
    """The two sides must produce the same parts, or they cannot meet.

    Checked as a property rather than on one example, because the failure mode
    is the two sides drifting apart later.
    """
    for name in ("get_user_by_id", "getUserById", "HTTPResponseHandler", "oauth2Client"):
        index_side = [t.lower() for t in split_identifier(name)]
        query_side = tokenize_query(name)
        assert set(index_side) <= set(query_side), name


def test_no_stopword_loss():
    """`if`, `not`, `for`, `in` are meaningful in code and must survive.

    §11.3 chooses `standard-no-stop-words` for the index half. This is the
    query half: a tokenizer that dropped them here would undo that choice.
    """
    terms = tokenize_query("if not found for user in cache")
    for word in ("if", "not", "for", "in"):
        assert word in terms, f"{word} was dropped"


def test_query_terms_are_deduplicated_but_ordered():
    assert tokenize_query("user user_id user") == ["user", "user_id", "id"]


def test_empty_query():
    assert tokenize_query("") == []
    assert tokenize_query("   ") == []
    assert build_fulltext_query("") == ""


# --------------------------------------------------------------------------
# Lucene query construction
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query, metachar",
    [
        ("arr[i]", "["),
        ("a+b", "+"),
        ("what calls charge_card?", "?"),
        ('say "hello"', '"'),
        ("path/to/thing", "/"),
        ("a:b", ":"),
    ],
)
def test_lucene_metacharacters_are_escaped(query, metachar):
    """A raw `[` or `?` in a query is Lucene syntax, not a character.

    Unescaped, the query either raises or silently matches something else — and
    `arr[i]`, a trailing `?` and a path with slashes are all things a developer
    types without thinking about it.
    """
    built = build_fulltext_query(query)
    for index, char in enumerate(built):
        if char == metachar:
            assert index > 0 and built[index - 1] == "\\", (
                f"unescaped {metachar!r} at {index} in {built!r}"
            )


def test_underscore_is_not_escaped():
    """`_` is not a Lucene metacharacter, and escaping it would be wrong.

    Over-escaping is the quieter half of this bug: the query stays legal, so
    nothing raises, and `__init__` simply stops matching itself.
    """
    built = build_fulltext_query("__init__")
    assert "\\" not in built
    assert "__init__" in built
    assert "init" in built


def test_fulltext_query_keeps_both_forms():
    built = build_fulltext_query("getUserById")
    assert "getuserbyid" in built
    assert "user" in built
