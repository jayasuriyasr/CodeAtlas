"""The citation-first prompt (§6.1).

The three instructions are quoted from §6.1 rather than paraphrased, because
each one is load-bearing for a specific validator:

* "must end with one or more context handles" — T1 and T2 both read handles;
  without the instruction the model has no reason to emit them.
* "state the uncertainty and do not cite" — this is what makes §6.1's hedge
  exclusion honest. Penalising a hedged sentence would train exactly the
  overconfidence the system exists to prevent, so the prompt has to invite the
  hedge in the first place.
* "Content marked ⟨N lines elided⟩ is not available to you" — pairs with
  §5.4's elision marker. Silent truncation manufactures the confident-uncited
  claim T2 catches; a marked elision the model has been told about does not.
"""

from __future__ import annotations

from pack.packer import PackedContext

SYSTEM = """\
You are a code investigation assistant answering questions about one repository.

Every factual claim about the codebase must end with one or more context \
handles, e.g. [C3]. If retrieved context does not support a claim, state the \
uncertainty and do not cite. Content marked ⟨N lines elided⟩ is not available \
to you — do not make claims about it.

Answer only from the context below. If the context does not contain the answer, \
say so plainly rather than inferring from what a codebase like this usually \
does. A handle must name a block that appears in the context; never invent one.

Write for a developer who will act on the answer. Prefer naming the symbol and \
the file over describing them.\
"""


def build_prompt(question: str, context: PackedContext) -> str:
    """Question plus packed context, with the packing report attached.

    The report is in the prompt as well as the UI on purpose. §5.4 gives the
    user-facing reason — "a developer who sees '2 omitted' knows to narrow the
    question" — and the model needs the same fact for the same reason: it is
    what makes "the context does not contain the answer" a conclusion it can
    reach rather than a gap it fills.
    """
    blocks = context.render()
    parts = [
        f"# Question\n{question.strip()}",
        f"# Context\n{blocks}" if blocks else "# Context\n(no context retrieved)",
        f"# Retrieval report\n{context.report()}",
    ]
    return "\n\n".join(parts)


RETRY_PREFIX = """\
A previous answer to this question was not sufficiently grounded in the \
retrieved context. Additional context has been retrieved below. Answer again, \
citing a handle for every factual claim about the codebase.\
"""


def build_retry_prompt(question: str, context: PackedContext, reason: str) -> str:
    """§6.3's single bounded retry.

    The reason is included so the second attempt is not simply the first one
    re-rolled at the same temperature — §8.2 runs at temperature 0, so an
    identical prompt would produce an identical answer and the retry would cost
    a full generation to change nothing.
    """
    return "\n\n".join([RETRY_PREFIX, f"# Why\n{reason}", build_prompt(question, context)])
