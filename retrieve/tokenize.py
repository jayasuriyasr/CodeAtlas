"""§5.3 — identifier tokenization for the fulltext arm.

Lucene's `StandardAnalyzer` is tokenizer + lowercase + stop-word filter. It does
not stem, and two failures matter:

1. **No identifier splitting.** Under UAX#29 an underscore joins rather than
   separates, so `get_user_by_id` and `getUserById` each stay one opaque token
   and "get user by id" scores zero against both.
2. **Stop-word removal**, which deletes `in`, `for`, `if`, `not` — all meaningful
   in code. That half is handled by the `standard-no-stop-words` analyzer in
   §11.3; the splitting half is not, which is why it happens here in Python.

**The same tokenizer runs on the user's query.** §5.3 calls tokenizing one side
only "a common and invisible failure" — invisible because the index still
returns *something*, just never the thing that was asked for.
"""

from __future__ import annotations

import re

from adapters.base import Symbol

# Spec §5.3, verbatim.
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_DIGIT = re.compile(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")
_SEPS = re.compile(r"[_\-./:]+")

#: Lucene query syntax metacharacters. Escaped rather than stripped: a query for
#: `__init__` or `a[i]` must reach the index as those characters, not as a
#: malformed query that raises or silently matches nothing.
_LUCENE_SPECIAL = re.compile(r'([+\-&|!(){}\[\]^"~*?:\\/])')


def split_identifier(name: str) -> list[str]:
    """§5.3, verbatim.

    `get_user_by_id` -> [get, user, by, id]
    `getUserById`    -> [get, User, By, Id]
    `oauth2Client`   -> [oauth, 2, Client]
    """
    parts: list[str] = []
    for seg in _SEPS.split(name):              # get_user_by_id -> [get,user,by,id]
        if not seg:
            continue
        for cam in _CAMEL.split(seg):          # getUserById    -> [get,User,ById]
            parts.extend(p for p in _DIGIT.split(cam) if p)
    return parts


def build_search_text(sym: Symbol) -> str:
    """§5.3, verbatim.

    Both the exact forms and the split forms are kept, which is what makes
    `get_user_by_id` and `getUserById` reachable from each other *and* from
    themselves. Dropping the exact form to save index space would break the
    case a developer is most likely to type: the identifier itself.
    """
    lowered = [t.lower() for t in split_identifier(sym.name)]
    return " ".join(
        filter(
            None,
            [
                sym.name,
                sym.qualified_name,
                sym.name.lower(),          # exact forms preserved
                *lowered,
                "".join(lowered),          # split + rejoined
                "_".join(lowered),
                sym.docstring or "",
            ],
        )
    )


def tokenize_query(query: str) -> list[str]:
    """The query side of §5.3 — the half whose absence is invisible.

    Every whitespace-separated term is split the same way an identifier is, and
    both the original term and its parts are kept. A user who types
    `getUserById` must match a symbol indexed as `get_user_by_id`, and a user
    who types `get user by id` must match both.
    """
    terms: list[str] = []
    for raw in query.split():
        term = raw.strip().lower()
        if not term:
            continue
        if term not in terms:
            terms.append(term)
        for part in split_identifier(raw):
            lowered = part.lower()
            if lowered and lowered not in terms:
                terms.append(lowered)
    return terms


def build_fulltext_query(query: str) -> str:
    """A Lucene query string for the `symbol_search` index.

    Terms are OR-ed, which is the default, and each is escaped. No stop-word
    filtering happens here or in the index (§11.3 uses
    `standard-no-stop-words`), so `if`, `not` and `for` survive to the index —
    which is the point of choosing that analyzer.
    """
    terms = [_LUCENE_SPECIAL.sub(r"\\\1", t) for t in tokenize_query(query)]
    return " ".join(t for t in terms if t)
