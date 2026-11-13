"""A citation must point at code that contains what the sentence names.

T1 checks that a cited handle exists; T2 that each claim sentence has one.
Nothing checked that the cited block has anything to do with the sentence: "The
`charge_card` function retries three times [C1]", with C1 the body of
`issue_token`, scored as fully cited.

The check is deliberately cheap and one-sided. Every code name a cited claim
mentions — backticked, or code-shaped like `snake_case`, `CamelCase`,
`dotted.names` — must appear in at least one of the blocks it cites. A sentence
that fails is "cited but unsupported", reported beside Coverage rather than
folded into it. Passing proves little (the names are there; the claim may still
be wrong); failing is a strong sign the citation is decoration.
"""

from __future__ import annotations

from api.app import InvestigationService
from ground.classify import classify
from ground.validate import unsupported_claims
from index.providers import ApproxCodeTokenizer, ScriptedLLM

from .test_pipeline_hardening import block

ISSUE_TOKEN = (
    "# authx/services.py :: def issue_token(user, *, length: int = TOKEN_BYTES) -> ApiToken\n"
    "def issue_token(user, *, length=TOKEN_BYTES):\n"
    "    key = secrets.token_hex(length)\n"
    "    token = ApiToken.objects.create(user=user, key=key)\n"
    "    audit_event(\"token_issue\", user_id=user.pk)\n"
    "    return token\n"
)
LOGIN_POST = (
    "# authx/views.py :: class LoginView(APIView) :: def post(self, request)\n"
    "def post(self, request):\n"
    "    token = issue_token(user)\n"
)
BLOCKS = {"C1": ISSUE_TOKEN, "C2": LOGIN_POST}


def unsupported(text: str) -> list[tuple[str, tuple[str, ...]]]:
    return [(s.text, missing) for s, missing in unsupported_claims(classify(text), BLOCKS)]


def test_a_citation_of_unrelated_code_is_unsupported():
    (found,) = unsupported("The `charge_card` function retries three times [C1].")
    assert found[1] == ("charge_card",)


def test_names_the_cited_block_contains_are_supported():
    assert unsupported("`issue_token` creates the key with `secrets.token_hex` [C1].") == []


def test_unbackticked_code_names_count_too():
    assert unsupported("issue_token logs via audit_event [C1].") == []
    (found,) = unsupported("issue_token then calls refund_card [C1].")
    assert found[1] == ("refund_card",)


def test_a_dotted_name_needs_each_of_its_parts():
    """The block for `LoginView.post` says `class LoginView` and `def post`;
    it never spells `LoginView.post`."""
    assert unsupported("`LoginView.post` calls `issue_token` [C2].") == []
    (found,) = unsupported("`LoginView.patch` calls `issue_token` [C2].")
    assert found[1] == ("LoginView.patch",)


def test_several_citations_are_pooled():
    assert unsupported("`LoginView.post` calls `issue_token`, which uses `audit_event` [C1][C2].") == []


def test_a_sentence_with_no_code_names_is_not_judged():
    assert unsupported("The token is then returned [C1].") == []


def test_language_keywords_are_not_code_names():
    assert unsupported("It returns `None` when nothing matches, else `True` [C1].") == []


def test_uncited_and_invented_citations_are_left_to_t1_and_t2():
    assert unsupported("The `charge_card` function retries.") == []
    assert unsupported("The `charge_card` function retries [C9].") == []


def test_the_service_reports_it_beside_coverage():
    service = InvestigationService(
        retriever=lambda q, r: ([block("charge_card", 1.0)], []),
        llm=ScriptedLLM("The `refund_card` function reverses a charge [C1]."),
        tokenizer=ApproxCodeTokenizer(),
    )
    events = list(service.stream("how are refunds made"))
    done, result = events[-1]
    assert done["type"] == "done"
    assert done["coverage"] == 1.0, "Coverage keeps its definition"
    assert done["unsupported_claims"] == 1
    assert result.report.unsupported[0][1] == ("refund_card",)
