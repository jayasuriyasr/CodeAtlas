"""Step 9 — §6.1's claim-sentence classifier and §6.4's handle parsing.

This module owns T2's denominator. §6.1 states the cost of getting it wrong:
"an over-inclusive denominator manufactures low Coverage and fires retries on
answers that were never wrong". So the bias is fixed — when ambiguous, exclude —
and §12.9 is honest that the errors propagate into Coverage regardless, which is
why §8.2 measures this classifier's own precision and recall.
"""

from __future__ import annotations

import pytest

from ground.classify import (
    ClassifiedAnswer,
    classify,
    fenced_spans,
    find_handles,
    split_sentences,
)


# --------------------------------------------------------------------------
# v9.1 #9 — handles inside fenced code
# --------------------------------------------------------------------------


def test_handle_in_code_fence_ignored():
    """v9.1 #9. `arr[C1]` in a snippet is not a citation.

    §6.4: "Recognized outside fenced code blocks only — answers about code
    routinely contain bracket-index expressions." Counting it credits a
    sentence that cited nothing, and if the block does not exist, reports the
    model for hallucinating a handle inside its own example code.
    """
    text = (
        "The handler reads the first element [C1].\n"
        "\n"
        "```python\n"
        "value = arr[C1]\n"
        "other = matrix[C7]\n"
        "```\n"
    )
    assert [h for h, _pos in find_handles(text)] == [1]


def test_tilde_fences_are_also_excluded():
    text = "See below [C2].\n\n~~~js\nconst x = arr[C9];\n~~~\n"
    assert [h for h, _pos in find_handles(text)] == [2]


def test_unclosed_fence_still_excludes_to_end_of_text():
    """A truncated stream can end mid-fence.

    Treating the remainder as prose would resurrect exactly the bracket-index
    false positives the rule exists to remove, and a truncated answer is when
    the model is most likely to be mid-snippet.
    """
    text = "Look at this [C1].\n\n```python\nvalue = arr[C5]\n"
    assert [h for h, _pos in find_handles(text)] == [1]


def test_fenced_spans_cover_the_fence_markers():
    text = "a\n```\nb\n```\nc\n"
    spans = fenced_spans(text)
    assert len(spans) == 1
    assert text[spans[0].start : spans[0].end].startswith("```")


def test_inline_code_is_not_a_fence():
    """A backticked identifier is *evidence of* a claim, not an exclusion.

    §6.1 lists "a backticked identifier" as one of the four things that make a
    sentence reference a code entity. Treating inline spans like fenced blocks
    would exclude most real claims.
    """
    answer = classify("The `charge_card` function bills the user [C1].")
    assert len(answer.claims) == 1
    assert answer.claims[0].handles == (1,)


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------


def test_sentences_split_on_terminators():
    sentences = split_sentences("First one. Second one! Third one?")
    assert [s.text for s in sentences] == ["First one.", "Second one!", "Third one?"]


def test_trailing_handle_stays_with_its_sentence():
    """`... here. [C3]` cites the sentence it follows.

    Splitting before the handle would leave the claim uncited and the handle
    stranded in a fragment that is not a claim — one lost citation, counted
    twice against Coverage.
    """
    sentences = split_sentences("The view calls it here. [C3] Then it returns.")
    assert sentences[0].text.endswith("[C3]")


def test_abbreviations_do_not_split():
    """Over-splitting turns one claim into two and changes T2's denominator."""
    sentences = split_sentences("It handles auth, e.g. login and logout, in one view.")
    assert len(sentences) == 1


def test_dotted_paths_do_not_split():
    sentences = split_sentences("The logic lives in authx.services.issue_token today.")
    assert len(sentences) == 1


def test_terminators_inside_fences_do_not_split():
    text = "Here is the code.\n\n```python\nx = 1. + 2.\ny = f()\n```\n\nThat is all."
    sentences = split_sentences(text)
    assert any("```" in s.text for s in sentences)
    assert sentences[-1].text.strip() == "That is all."


# --------------------------------------------------------------------------
# The predicate — inclusions
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sentence, why",
    [
        ("The handler validates the payload [C1].", "handle"),
        ("`charge_card` bills the stored card.", "backticked identifier"),
        ("The billing module runs nightly.", "code-entity noun"),
        ("That endpoint returns 201 on success.", "code-entity noun"),
        ("The component subscribes on mount.", "code-entity noun"),
    ],
)
def test_declarative_sentences_referencing_code_are_claims(sentence, why):
    """§6.1: declarative **and** references a code entity."""
    answer = classify(sentence)
    assert answer.claims, f"{sentence!r} should be a claim ({why})"
    assert answer.claims[0].reason == why


def test_unbackticked_packed_name_counts():
    """The one form the other three tests miss.

    §6.1 lists "a `qualified_name` present in packed context" separately for a
    reason: models routinely name a symbol in prose without backticks, and
    without this the sentence reads as ordinary English.
    """
    answer = classify(
        "Tokens are minted in authx.services.issue_token.",
        known_names=frozenset({"authx.services.issue_token"}),
    )
    assert len(answer.claims) == 1
    assert answer.claims[0].reason == "packed qualified_name"

    # Without the packed context, the same sentence is not recognised — which is
    # the conservative direction, and why known_names is passed in.
    assert classify("Tokens are minted in authx.services.issue_token.").claims == []


# --------------------------------------------------------------------------
# The predicate — exclusions
# --------------------------------------------------------------------------


def test_interrogative_not_a_claim():
    """§6.1 excludes interrogatives. A question asserts nothing to cite."""
    answer = classify("Which function handles the refund path?")
    assert answer.claims == []
    assert answer.sentences[0].reason == "interrogative"


@pytest.mark.parametrize(
    "sentence",
    [
        "See the `charge_card` function for details.",
        "Check the billing module first.",
        "Note that the endpoint is authenticated.",
        "Consider the `Subscription` model here.",
        "Please review the auth module.",
    ],
)
def test_imperative_not_a_claim(sentence):
    """An instruction asserts nothing about the codebase.

    Requiring a citation for "see the billing module" manufactures a coverage
    failure on a sentence that was never a claim.
    """
    answer = classify(sentence)
    assert answer.claims == [], sentence
    assert answer.sentences[0].reason == "imperative"


@pytest.mark.parametrize(
    "sentence",
    [
        "The `charge_card` function might retry on failure.",
        "It appears to call the billing module.",
        "This probably handles the webhook endpoint.",
        "I am not sure which function owns that route.",
        "It is unclear whether the component re-renders.",
        "The module seems to import the mixin.",
        "I cannot tell which endpoint is used.",
    ],
)
def test_hedged_sentence_not_a_claim(sentence):
    """§6.1: do not punish declared uncertainty.

    "Penalizing declared uncertainty trains the overconfidence this system
    exists to prevent." A hedged sentence is the behaviour we want; counting it
    against Coverage would teach the model to drop the hedge.
    """
    answer = classify(sentence)
    assert answer.claims == [], sentence
    assert answer.sentences[0].reason == "hedged"


def test_prose_with_no_code_reference_is_not_a_claim():
    answer = classify("This took a while to work out.")
    assert answer.claims == []
    assert answer.sentences[0].reason == "no code reference"


def test_sentence_that_is_only_a_code_fence_is_not_a_claim():
    answer = classify("```python\nx = 1\n```")
    assert answer.claims == []


def test_when_ambiguous_exclude():
    """§6.1's stated bias, as a property.

    Each of these has *some* code-adjacent flavour and none of them asserts a
    checkable fact. A missed claim understates a problem; a false claim invents
    one and burns a retry.
    """
    for sentence in (
        "That is worth checking.",
        "Here is what I found.",
        "The rest follows the same shape.",
    ):
        assert classify(sentence).claims == [], sentence


# --------------------------------------------------------------------------
# Hand-annotated fixtures — §8.2's precision/recall
# --------------------------------------------------------------------------

#: (sentence, is_claim). Annotated by hand against §6.1's rule. §8.2 gates the
#: classifier at precision >= 0.90 / recall >= 0.85, and §12.9 is explicit that
#: the number "will not be 1.0" — Coverage reads as "coverage as this
#: classifier sees it".
ANNOTATED: tuple[tuple[str, bool], ...] = (
    ("The `LoginView.post` method validates credentials [C1].", True),
    ("`issue_token` mints a new API token [C2].", True),
    ("The billing module charges cards nightly [C3].", True),
    ("That endpoint returns 401 when credentials fail [C1].", True),
    ("`Subscription.monthly_total` returns zero when inactive [C4].", True),
    ("The `TimestampMixin` class adds created and updated columns.", True),
    ("Token revocation is handled by the `ApiToken.revoke` method [C5].", True),
    ("The route dispatches to `SubscriptionView` [C6].", True),
    ("`mask_email` is called before logging [C7].", True),
    ("The `charge_card` function is a no-op for zero amounts [C3].", True),
    ("Which module owns the retry logic?", False),
    ("See the `billing` package for the full flow.", False),
    ("It might also handle refunds.", False),
    ("I am not sure where that is configured.", False),
    ("This is a fairly conventional layout.", False),
    ("Let me know if you want more detail.", False),
    ("That should be everything.", False),
    ("The code appears to follow the same pattern.", False),
    ("Consider narrowing the question.", False),
    ("Here is what the retrieved context shows.", False),
)


def test_classifier_precision_recall():
    """§8.2's gate: precision >= 0.90, recall >= 0.85.

    Measured against the hand annotations above rather than asserted. §12.9:
    "the number will not be 1.0" — what matters is that it is a number, and
    that it is this number the Coverage figure should be read against.
    """
    known = frozenset()
    tp = fp = fn = tn = 0

    for sentence, expected in ANNOTATED:
        predicted = bool(classify(sentence, known).claims)
        if predicted and expected:
            tp += 1
        elif predicted and not expected:
            fp += 1
        elif not predicted and expected:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0

    assert precision >= 0.90, (
        f"precision {precision:.2f} (tp={tp} fp={fp}) — an over-inclusive "
        f"denominator manufactures low Coverage (§6.1)"
    )
    assert recall >= 0.85, f"recall {recall:.2f} (tp={tp} fn={fn})"


def test_annotated_fixture_is_balanced():
    """A fixture that is all positives cannot measure precision."""
    positives = sum(1 for _s, expected in ANNOTATED if expected)
    assert 5 <= positives <= len(ANNOTATED) - 5


# --------------------------------------------------------------------------
# Whole answers
# --------------------------------------------------------------------------


ANSWER = """\
The `LoginView.post` method validates credentials and issues a token [C1].

It delegates to `verify_credentials`, which compares the stored hash [C2].

```python
user = verify_credentials(email, password)
token = tokens[C1]
```

Which caller handles the failure path?

The token is minted by `issue_token` [C3].
It might also be revoked elsewhere.
"""


def test_whole_answer_classification():
    answer = classify(ANSWER)

    claims = [s.text for s in answer.claims]
    assert len(claims) == 3, claims
    assert all(s.cited for s in answer.claims)
    assert answer.all_handles == [1, 2, 3], "the fenced tokens[C1] leaked in"

    reasons = {s.reason for s in answer.sentences if not s.is_claim}
    assert "interrogative" in reasons
    assert "hedged" in reasons


def test_classified_answer_partitions_cleanly():
    """cited + uncited must be exactly the claims, with no overlap.

    Compared by identity rather than value: `Sentence` is mutable (the
    classifier writes `is_claim` and `reason` onto it), so it is deliberately
    unhashable and two equal-looking sentences are not the same sentence.
    """
    answer = classify(ANSWER)
    cited = [id(s) for s in answer.cited_claims]
    uncited = [id(s) for s in answer.uncited_claims]

    assert sorted(cited + uncited) == sorted(id(s) for s in answer.claims)
    assert not set(cited) & set(uncited)


def test_empty_answer():
    answer = classify("")
    assert isinstance(answer, ClassifiedAnswer)
    assert answer.sentences == [] and answer.claims == []
