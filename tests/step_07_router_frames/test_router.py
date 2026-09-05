"""Step 7 — §5.1's query router.

The failure this guards is not an error, it is a wasted path: "what calls
`charge_card`" sent to the semantic arm runs an ANN search over a question
whose answer is one exact traversal away.
"""

from __future__ import annotations

import pytest

import metrics
from retrieve.router import QueryClass, QueryRouter


@pytest.fixture(autouse=True)
def reset_metrics():
    metrics.reset()
    yield
    metrics.reset()


@pytest.fixture
def router() -> QueryRouter:
    return QueryRouter()


PY_TRACE = '''Traceback (most recent call last):
  File "/app/authx/views.py", line 42, in post
    user = verify_credentials(email, password)
  File "/app/authx/services.py", line 17, in verify_credentials
    return user
AttributeError: 'NoneType' object has no attribute 'password'
'''

JS_TRACE = """TypeError: Cannot read properties of undefined (reading 'id')
    at GET (/app/.next/server/chunks/4821.js:1:5678)
    at async handler (/app/node_modules/next/dist/server/route.js:12:9)
"""


# --------------------------------------------------------------------------
# STRUCTURAL
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "what calls charge_card",
        "who calls verify_credentials?",
        "which functions call audit_event",
        "callers of issue_token",
        "what does LoginView.post call",
        "where is mask_email used",
        "usages of TimestampMixin",
        "references to ApiToken",
        "where is charge_card defined",
        "subclasses of APIView",
        "what imports common.utils",
        "which view handles the login route",
    ],
)
def test_router_classifies_structural(router, query):
    """"what calls X" does not go to ANN.

    Each of these names a relationship the graph stores exactly. Sending them
    to hybrid retrieval trades an exact answer for a ranked guess.
    """
    routed = router.route(query)
    assert routed.query_class is QueryClass.STRUCTURAL, query
    assert routed.decided_by == "regex"
    assert routed.rule, "the deciding rule must be recorded for misroute triage"


def test_structural_rule_is_named(router):
    assert router.route("callers of issue_token").rule == "callers"
    assert router.route("where is charge_card defined").rule == "definition"


# --------------------------------------------------------------------------
# SEMANTIC
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "how does auth work",
        "how do tokens get revoked",
        "why is billing split into tasks",
        "explain the subscription lifecycle",
        "walk me through the login flow",
        "what is the purpose of the mixins module",
    ],
)
def test_router_classifies_semantic(router, query):
    routed = router.route(query)
    assert routed.query_class is QueryClass.SEMANTIC, query
    assert routed.decided_by == "regex"


def test_unmatched_query_defaults_to_semantic_without_an_llm(router):
    """§9.3's shape: degrade to the general path, never fail to route."""
    routed = router.route("billing subscription totals")
    assert routed.query_class is QueryClass.SEMANTIC
    assert routed.decided_by == "default"
    assert metrics.get("router.default_semantic") == 1


# --------------------------------------------------------------------------
# TRACE
# --------------------------------------------------------------------------


def test_router_classifies_a_python_traceback(router):
    routed = router.route(PY_TRACE)
    assert routed.query_class is QueryClass.TRACE
    assert routed.rule == "stack_trace"
    assert [f.fn_name for f in routed.frames] == ["post", "verify_credentials"]
    assert routed.frames[0].line_no == 42


def test_router_classifies_a_js_stack(router):
    routed = router.route(JS_TRACE)
    assert routed.query_class is QueryClass.TRACE
    assert "GET" in [f.fn_name for f in routed.frames]


def test_trace_wins_over_structural_wording(router):
    """A pasted trace that happens to contain "calls" is still a trace.

    Order matters: the trace check runs first because a trace carries an exact
    location and the structural path has nothing to resolve against.
    """
    routed = router.route("what calls this?\n" + PY_TRACE)
    assert routed.query_class is QueryClass.TRACE


def test_prose_about_a_traceback_is_not_a_trace(router):
    """The router keys on trace *shape*, not on the word.

    Matching the word would send "why do I get a traceback here" down the frame
    resolver with no frames to resolve.
    """
    routed = router.route("why do I keep getting a traceback in the login view")
    assert routed.query_class is QueryClass.SEMANTIC


# --------------------------------------------------------------------------
# The LLM fallback
# --------------------------------------------------------------------------


def test_llm_fallback_is_consulted_only_when_the_regexes_are_silent():
    """§5.1: "Regex heuristics first, LLM fallback for ambiguity."

    Consulting it on every query would add a model call to the p50 of a system
    whose TTFT gate is 1.2s — for questions the regexes already answered.
    """
    seen: list[str] = []

    def classify(query: str):
        seen.append(query)
        return QueryClass.STRUCTURAL

    router = QueryRouter(llm_classify=classify)

    router.route("what calls charge_card")          # structural by regex
    router.route("how does auth work")              # semantic by regex
    router.route(PY_TRACE)                          # trace by shape
    assert seen == [], "the LLM was consulted for a query the regexes decided"

    routed = router.route("billing totals subscription")
    assert seen == ["billing totals subscription"]
    assert routed.query_class is QueryClass.STRUCTURAL
    assert routed.decided_by == "llm"
    assert metrics.get("router.llm_fallback") == 1


def test_llm_declining_falls_back_to_semantic():
    """A classifier that cannot decide must not block the query."""
    router = QueryRouter(llm_classify=lambda q: None)
    routed = router.route("billing totals subscription")

    assert routed.query_class is QueryClass.SEMANTIC
    assert routed.decided_by == "default"
    assert metrics.get("router.llm_declined") == 1


def test_router_is_deterministic(router):
    """§8.2 runs eval at temperature 0; the route must not vary between runs."""
    for query in ("what calls x", "how does auth work", PY_TRACE, "unclassifiable"):
        first = router.route(query)
        for _ in range(3):
            again = router.route(query)
            assert (again.query_class, again.rule) == (first.query_class, first.rule)
