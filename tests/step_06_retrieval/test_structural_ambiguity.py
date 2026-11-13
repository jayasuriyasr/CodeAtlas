"""A name that means several symbols is a question back, not a merged answer.

`chosen = exact or found` took every symbol whose name matched — up to eight —
and returned all their callers together. "what calls `post`" in a project with
four views answered with the callers of all four, as if they were one function.

The rule now matches the rest of the system (Tier 2 frames, edge resolution):
more than one candidate is a refusal, and the candidates are returned so the
reader can say which one was meant. One function under one qualified name with
several overloads is still one function.
"""

from __future__ import annotations

import metrics
from retrieve.structural import RESOLVE_CYPHER, StructuralResolver


class Runner:
    """Answers the entity lookup from a fixed list; every traversal finds `related`."""

    def __init__(self, matches: list[tuple[str, str, bool]], related: list[dict] | None = None):
        self.matches = matches
        self.related = related or [{"uid": "caller-1", "rel_path": "x.py", "start_line": 1, "rel": "CALLS"}]
        self.traversed_with: list[list[str]] = []

    def run(self, query: str, **params):
        if query == RESOLVE_CYPHER:
            term = params["terms"][0]["term"]
            return [{"term": term, "uid": uid, "qualified_name": qn, "exact": exact}
                    for uid, qn, exact in self.matches]
        self.traversed_with.append(list(params["uids"]))
        return self.related


def setup_function():
    metrics.reset()


def test_several_symbols_with_the_name_are_returned_as_candidates():
    runner = Runner([
        ("u1", "authx.views.LoginView.post", False),
        ("u2", "authx.views.TokenRevokeView.post", False),
        ("u3", "billing.views.SubscriptionView.post", False),
    ])
    answer = StructuralResolver(runner=runner).answer("r", "what calls `post`", "callers")

    assert answer.related_uids == [] and runner.traversed_with == [], "nothing merged"
    assert answer.candidates == [
        "authx.views.LoginView.post",
        "authx.views.TokenRevokeView.post",
        "billing.views.SubscriptionView.post",
    ]
    assert "say which one" in answer.fallback and "refusing" in answer.fallback
    assert answer.note()["candidates"] == answer.candidates
    assert metrics.get("structural.ambiguous") == 1


def test_a_unique_name_is_still_answered():
    runner = Runner([("u1", "billing.tasks.charge_card", False)])
    answer = StructuralResolver(runner=runner).answer("r", "what calls `charge_card`", "callers")
    assert answer.fallback is None
    assert runner.traversed_with == [["u1"]] and answer.related_uids == ["caller-1"]


def test_an_exact_qualified_name_picks_its_symbol_among_namesakes():
    runner = Runner([
        ("u1", "authx.views.LoginView.post", False),
        ("u3", "billing.views.SubscriptionView.post", True),
    ])
    answer = StructuralResolver(runner=runner).answer(
        "r", "what calls `billing.views.SubscriptionView.post`", "callers")
    assert answer.fallback is None and runner.traversed_with == [["u3"]]


def test_overloads_of_one_function_are_one_entity():
    """Three TypeScript overload signatures share one qualified name."""
    runner = Runner([
        ("o1", "app.api.users.route.parse", False),
        ("o2", "app.api.users.route.parse", False),
        ("o3", "app.api.users.route.parse", False),
    ])
    answer = StructuralResolver(runner=runner).answer("r", "what calls `parse`", "callers")
    assert answer.fallback is None
    assert runner.traversed_with == [["o1", "o2", "o3"]]
