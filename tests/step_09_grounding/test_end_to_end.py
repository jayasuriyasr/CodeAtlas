"""Step 9's gate — a real question about the Django fixture, answered.

Plan §9: "**A real question about the Django fixture returns a cited, streamed
answer.** This is the first moment the system exists."

Two things are needed for that sentence to be true, and only one of them is
Neo4j. The other is a configured LLM, and a test must not supply one: calling it
sends full function bodies to a third party. So this module goes as far as it
honestly can — the whole pipeline over a really-indexed corpus, with the model
scripted — and marks the remaining half explicitly.

What *is* established here: the corpus indexes, retrieval returns real symbols,
the packer builds real context with real handles, T3 verifies real edges against
the real graph, and the citation the answer makes resolves to a symbol that
exists. What is not: whether the model, unscripted, cites correctly.
"""

from __future__ import annotations

import asyncio
import time

import pytest

import metrics
from api.app import InvestigationService
from api.stream import SentenceEvent, TextEvent
from config import GATE_COST_P50_USD, GATE_TTFT_P50_S
from graph.reader import GraphReader
from graph.writer import GraphWriter
from index.cache import EmbeddingCache
from index.indexer import Indexer
from index.providers import ApproxCodeTokenizer, HashEmbeddingProvider, ScriptedLLM
from pack.render import Block
from retrieve.expand import NeighborhoodExpander

REPO = "repo_step9"


@pytest.fixture(autouse=True)
def reset_metrics():
    metrics.reset()
    yield
    metrics.reset()


class Runner:
    def __init__(self, graph_db):
        self._db = graph_db
        self.queries: list[str] = []

    def run(self, query: str, **params):
        self.queries.append(query)
        return [dict(row) for row in self._db.run(query, **params)]


@pytest.fixture(scope="module")
def indexed(graph_db, fixtures_dir, tmp_path_factory):
    """The Django fixture, indexed through the real `Indexer`.

    Not hand-written symbols: the point of an end-to-end test is that the parse,
    chunk, embed and write path all agree, and any of them substituted is a
    place the disagreement could hide.
    """
    graph_db.run("MATCH (n) DETACH DELETE n")

    root = fixtures_dir / "repos" / "django_min"
    provider = HashEmbeddingProvider()
    cache = EmbeddingCache(
        tmp_path_factory.mktemp("e") / "cache.sqlite", model=provider.name
    )
    indexer = Indexer(
        writer=GraphWriter(graph_db.driver, graph_db.name),
        reader=GraphReader(graph_db.driver, graph_db.name),
        cache=cache,
        provider=provider,
        tokenizer=ApproxCodeTokenizer(),
        roots={REPO: root},
    )
    asyncio.run(indexer.full_reconcile(REPO))
    cache.close()
    graph_db.run("CALL db.awaitIndexes(180)")
    return indexer


@pytest.fixture
def reader(graph_db) -> GraphReader:
    return GraphReader(graph_db.driver, graph_db.name)


def load(reader: GraphReader, uid: str):
    from adapters.base import Symbol

    props = reader.symbol(REPO, uid)
    assert props is not None, uid
    return Symbol(
        uid=props["uid"], repo_id=REPO, rel_path=props.get("rel_path", ""),
        qualified_name=props.get("qualified_name", ""), name=props.get("name", ""),
        arity=props.get("arity", 0), ordinal=props.get("ordinal", 0),
        kind=props.get("kind", "function"), signature=props.get("signature", ""),
        docstring=props.get("docstring"), source_code=props.get("source_code", ""),
        enclosing_signature=props.get("enclosing_signature"),
        used_imports=list(props.get("used_imports") or []),
        start_line=props.get("start_line", 0), end_line=props.get("end_line", 0),
    )


def uid_of(reader: GraphReader, qualified_name: str) -> str:
    candidates = reader.symbols_named(REPO, qualified_name.rsplit(".", 1)[-1])
    matching = [c for c in candidates if c.qualified_name == qualified_name]
    assert len(matching) == 1, f"{qualified_name}: {[c.qualified_name for c in candidates]}"
    return matching[0].uid


@pytest.fixture
def service(indexed, graph_db, reader):
    """The full path, with only the model scripted."""
    expander = NeighborhoodExpander(runner=Runner(graph_db))

    def build(answers):
        def retrieve(question, routed):
            seed_uid = uid_of(reader, "billing.tasks.charge_card")
            seeds = [Block(symbol=load(reader, seed_uid), score=1.0)]
            neighbors = [
                Block(symbol=load(reader, n.uid), score=n.score)
                for n in expander.expand(REPO, [seed_uid])
            ]
            return seeds, neighbors

        return InvestigationService(
            retriever=retrieve,
            llm=ScriptedLLM(*answers),
            tokenizer=ApproxCodeTokenizer(),
            graph_runner=Runner(graph_db),
            repo_id=REPO,
        )

    return build


# --------------------------------------------------------------------------
# The corpus is really there
# --------------------------------------------------------------------------


def test_the_fixture_indexed(indexed, graph_db):
    symbols = graph_db.run(
        "MATCH (s:Symbol {repo_id: $r}) RETURN count(s) AS n", r=REPO
    )[0]["n"]
    assert symbols > 60, f"only {symbols} symbols indexed"

    missing_vectors = graph_db.run(
        "MATCH (s:Symbol {repo_id: $r}) WHERE s.code_vec IS NULL RETURN count(s) AS n",
        r=REPO,
    )[0]["n"]
    assert missing_vectors == 0, "T5: symbols indexed with null embeddings"


# --------------------------------------------------------------------------
# The gate, as far as it goes without a model
# --------------------------------------------------------------------------


def test_a_real_question_returns_a_cited_streamed_answer(service, reader):
    """Plan §9's gate, minus the model.

    The citation is checked all the way through: the handle appears in the
    answer, T1 resolves it to a uid, and that uid is a symbol that exists in the
    graph with the qualified name the answer claims.
    """
    answer = (
        "The `charge_card` function bills a user's stored card [C1]. "
        "It returns early when the amount is not positive [C1]."
    )
    svc = service([answer])

    events = []
    result = None
    for event, result in svc.stream("what charges a customer's card"):
        events.append(event)

    assert any(isinstance(e, TextEvent) for e in events), "nothing streamed"
    sentences = [e for e in events if isinstance(e, SentenceEvent)]
    assert sentences and all(e.badge == "✓" for e in sentences if e.is_claim)

    assert result.report.coverage == 1.0
    assert result.report.handle_errors == []
    assert not result.retried

    cited_uid = result.context.handles["C1"]
    symbol = reader.symbol(REPO, cited_uid)
    assert symbol is not None, "the citation resolves to nothing in the graph"
    assert symbol["qualified_name"] == "billing.tasks.charge_card"


def test_expansion_puts_the_real_caller_in_context(service, reader):
    """§5.2 step 3, reaching the prompt.

    `SubscriptionView.post` calls `charge_card` from another file and shares no
    vocabulary with it. If it is in the packed context, the graph earned its
    keep on this question.
    """
    svc = service(["The `charge_card` function bills the card [C1]."])
    result = svc.answer("what charges a customer's card")

    names = {p.block.symbol.qualified_name for p in result.context.blocks}
    assert "billing.views.SubscriptionView.post" in names, sorted(names)


def test_t3_verifies_a_real_edge(service, graph_db):
    """T3 against the real graph — one batched query, real relationships."""
    svc = service(
        ["`SubscriptionView.post` calls `charge_card` [C1]."]
    )
    result = svc.answer("who charges the card")

    assert result.report.relation_verdicts, "no relation claim was extracted"
    assert result.report.relation_precision == 1.0, (
        "the edge exists in the fixture; T3 did not find it"
    )
    assert result.report.conflict_flags == 0
    assert metrics.get("ground.t3.queries") == 1


def test_t3_flags_a_relation_the_graph_contradicts(service):
    """The other direction: a plausible sentence the code does not support.

    Both symbols exist, so this is a contradiction rather than an unknown —
    §12.3's distinction, and the one that decides whether §6.3 retries.
    """
    svc = service(
        [
            "`mask_email` calls `charge_card` [C1].",
            "`mask_email` calls `charge_card` [C1].",
        ]
    )
    result = svc.answer("does masking charge the card")

    assert result.report.conflict_flags == 1
    assert result.retried, "§6.3 retries on ConflictFlags > 0"


def test_hallucinated_handle_is_caught_end_to_end(service):
    svc = service(
        [
            "The `charge_card` function bills the card [C99].",
            "The `charge_card` function bills the card [C1].",
        ]
    )
    result = svc.answer("what charges the card")

    first_errors = [e for e in result.events if isinstance(e, SentenceEvent)]
    assert any(e.badge == "⚠ unknown citation" for e in first_errors)


# --------------------------------------------------------------------------
# §8.2's recorded numbers
# --------------------------------------------------------------------------


def test_ttft_and_cost_are_recorded(service):
    """§8.2 records TTFT p50/p95 and `$/query`, plus p50 packed tokens.

    Recorded here, not gated: TTFT against `ScriptedLLM` measures this machine,
    and `$/query` uses a packed-token count from the stand-in tokenizer (F-006).
    Both become real numbers at the same moment — when the provider is wired.
    """
    result = service(["The `charge_card` function bills the card [C1]."]).answer(
        "what charges the card"
    )

    assert result.ttft_seconds is not None
    assert result.cost.packed_tokens > 0
    assert result.cost.usd >= 0

    # Sanity only. A scripted provider cannot fail a latency gate meaningfully.
    assert result.ttft_seconds < GATE_TTFT_P50_S
    assert result.cost.usd < GATE_COST_P50_USD


def test_packed_tokens_stay_inside_the_budget(service):
    result = service(["The `charge_card` function bills the card [C1]."]).answer(
        "what charges the card"
    )
    assert result.context.packed_tokens <= result.context.budget
