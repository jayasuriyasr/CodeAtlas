"""T3 reads relations the way the model actually writes them.

T3 only extracted "`A` calls `B`" with nothing between the names. Over 45 real
answers from the local model that form never occurred, so Relation Precision
had nothing to measure. These are sentences the model wrote, verbatim or
nearly, taken from the recorded run. One of them is wrong — `soft_delete` is
not called by a `delete` method — which is exactly what T3 exists to catch.
"""

from __future__ import annotations

import pytest

from ground.classify import classify
from ground.validate import extract_relation_claims


def triples(text: str) -> set[tuple[str, str, str]]:
    return {(c.subject, c.relation, c.object)
            for c in extract_relation_claims(classify(text))}


@pytest.mark.parametrize("sentence, expected", [
    ("The `verify_credentials` function in `authx/services.py` calls `constant_time_equals` "
     "to compare the passwords [C2].",
     ("verify_credentials", "CALLS", "constant_time_equals")),
    ("The `soft_delete` method is called by the `delete` method in the `SoftDeleteMixin` class [C1].",
     ("SoftDeleteMixin.delete", "CALLS", "soft_delete")),
    ("The `post` method in `authx/views.py` calls `mask_email` when login fails [C2].",
     ("post", "CALLS", "mask_email")),
    ("The `issue_token` function is called in the `post` method of the `LoginView` class [C2].",
     ("LoginView.post", "CALLS", "issue_token")),
    ("`run_monthly_billing` calls the `charge_card` function for each subscription [C1].",
     ("run_monthly_billing", "CALLS", "charge_card")),
    ("The `Subscription` model inherits from `TimestampMixin` [C3].",
     ("Subscription", "INHERITS", "TimestampMixin")),
    ("`ApiToken` is a subclass of `TimestampMixin` [C1].",
     ("ApiToken", "INHERITS", "TimestampMixin")),
])
def test_the_models_phrasings_are_extracted(sentence, expected):
    assert expected in triples(sentence)


@pytest.mark.parametrize("sentence", [
    "The `charge_card` function is called in `billing/views.py` [C2].",   # a file, not a symbol
    "The `revoke` method calls `self.save(update_fields=['revoked_at'])` [C1].",
    "`revoke_all` might call `revoke` [C1].",                               # hedged
    "Does `charge_card` call `audit_event`?",
    "The `issue_token` function uses `secrets.token_hex` [C1].",           # "uses" is not a call
])
def test_what_is_not_a_relation_claim_stays_out(sentence):
    assert triples(sentence) == set()


def test_the_original_forms_still_work():
    assert triples("`SubscriptionView.post` calls `charge_card` [C1].") == {
        ("SubscriptionView.post", "CALLS", "charge_card")}
    assert triples("`charge_card` is called by `SubscriptionView.post` [C1].") == {
        ("SubscriptionView.post", "CALLS", "charge_card")}
