"""Which way a "what … call" question points does not depend on the name's length.

The callers rule was tried first and allowed up to 20 characters between the
question word and "call". "what does charge_card call" fits in 20, so it was
answered as *callers*; "what does LoginView.post call" does not, so it was
answered as *callees*. Same question, opposite answers, decided by a name.
"""

from __future__ import annotations

import pytest

from retrieve.router import QueryClass, QueryRouter


@pytest.fixture(scope="module")
def router() -> QueryRouter:
    return QueryRouter()


@pytest.mark.parametrize(
    "query",
    [
        "what does foo call",
        "what does run call",
        "what does charge_card call",
        "what does LoginView.post call",
        "what does run_monthly_billing call",
        "what does `foo` call?",
        "which functions does run call",
        "who does foo call",
        "what do the views call",
    ],
)
def test_what_does_x_call_is_callees(router, query):
    routed = router.route(query)
    assert routed.query_class is QueryClass.STRUCTURAL
    assert routed.rule == "callees", query


@pytest.mark.parametrize(
    "query",
    [
        "what calls foo",
        "what calls run",
        "who calls run",
        "which functions call foo",
        "what calls `charge_card`?",
        "callers of run",
        "what function calls foo",
    ],
)
def test_what_calls_x_is_still_callers(router, query):
    routed = router.route(query)
    assert routed.query_class is QueryClass.STRUCTURAL
    assert routed.rule == "callers", query
