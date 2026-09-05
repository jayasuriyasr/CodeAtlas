"""Step 6 — expansion and recall, against a real graph.

§5.2 on why step 3 exists at all:

> Step 3 is the justification for running a graph database. Without it, Neo4j
> is an expensive Postgres.

So the tests that matter here are the ones only a graph can answer: does
expansion return the inbound callers, and does the hub cutoff stop a
400-caller utility from flooding every result.
"""

from __future__ import annotations

import time

import pytest

import metrics
from adapters.base import Edge, Symbol, symbol_uid
from adapters.python import PythonAdapter
from config import GATE_RECALL_AT_10, HUB_CUTOFF
from eval.harness import EvalHarness, load_questions
from graph.reader import GraphReader
from graph.writer import GraphWriter
from index.cache import EmbeddingCache, embed_batch
from index.chunker import chunk_file
from index.providers import ApproxCodeTokenizer, HashEmbeddingProvider
from retrieve.expand import NeighborhoodExpander
from retrieve.hybrid import HybridRetriever, Neo4jSearchBackend
from retrieve.tokenize import build_search_text

REPO = "repo_step6"
#: The hub fixture gets its own tenant. It used to share `REPO` and wipe the
#: whole graph, which destroyed the module-scoped corpus underneath the
#: recall baseline — every question then missed, and the report recorded a
#: Recall@10 of 0.000 that measured nothing but fixture ordering.
HUB_REPO = "repo_step6_hub"
REPORT = "tests/step_06_retrieval/RECALL_BASELINE.md"


@pytest.fixture(autouse=True)
def reset_metrics():
    metrics.reset()
    yield
    metrics.reset()


@pytest.fixture
def writer(graph_db) -> GraphWriter:
    return GraphWriter(graph_db.driver, graph_db.name)


@pytest.fixture
def reader(graph_db) -> GraphReader:
    return GraphReader(graph_db.driver, graph_db.name)


@pytest.fixture(scope="module")
def provider() -> HashEmbeddingProvider:
    return HashEmbeddingProvider()


@pytest.fixture(scope="module")
def indexed_corpus(graph_db, fixtures_dir, provider, tmp_path_factory):
    """The Django fixture, fully indexed: parsed, chunked, embedded, written.

    Module-scoped: the corpus is read-only for every test here, and rebuilding
    it per test would spend most of the module's runtime on setup.
    """
    graph_db.run("MATCH (n) DETACH DELETE n")

    repo = fixtures_dir / "repos" / "django_min"
    adapter, tokenizer = PythonAdapter(), ApproxCodeTokenizer()
    cache = EmbeddingCache(
        tmp_path_factory.mktemp("cache") / "e.sqlite", model=provider.name
    )

    batch = {}
    for path in sorted(repo.rglob("*.py")):
        rel = path.relative_to(repo).as_posix()
        parsed = chunk_file(adapter.parse(REPO, rel, path.read_bytes()), tokenizer)
        for sym in parsed.symbols:
            sym.search_text = build_search_text(sym)
        batch[rel] = parsed

    vectors = embed_batch(REPO, batch, cache=cache, provider=provider)
    GraphWriter(graph_db.driver, graph_db.name).apply(
        REPO, batch, batch.keys(), vectors, time.time_ns()
    )
    cache.close()
    graph_db.run("CALL db.awaitIndexes(180)")
    return batch


def uids_for(batch, rel_path: str, qualified_name: str) -> list[str]:
    parsed = batch.get(rel_path)
    if parsed is None:
        return []
    return [s.uid for s in parsed.symbols if s.qualified_name == qualified_name]


# --------------------------------------------------------------------------
# search_text actually reached the graph
# --------------------------------------------------------------------------


def test_search_text_is_written_to_every_symbol(indexed_corpus, graph_db):
    """The fulltext arm's only input.

    A symbol written without `search_text` is invisible to fulltext search while
    looking entirely healthy in the graph — T5's failure shape, on the other arm.
    """
    missing = graph_db.run(
        "MATCH (s:Symbol {repo_id: $r}) WHERE s.search_text IS NULL "
        "RETURN s.qualified_name AS qn LIMIT 5",
        r=REPO,
    )
    assert missing == [], f"symbols with no search_text: {[m['qn'] for m in missing]}"


def test_fulltext_index_matches_a_cross_form_query(indexed_corpus, graph_db):
    """§5.3 end to end: query `getUserById`, match `get_user_by_id`.

    The tokenizer tests prove the two forms normalize alike. This proves the
    normalized form survived into Lucene and comes back out.
    """
    backend = Neo4jSearchBackend(graph_db.driver, graph_db.name)
    from retrieve.tokenize import build_fulltext_query

    rows = backend.fulltext_search(REPO, build_fulltext_query("getUserById"), 10)
    gold = set(uids_for(indexed_corpus, "common/utils.py", "common.utils.get_user_by_id"))
    assert gold & {uid for uid, _ in rows}, (
        "the snake_case symbol was not reachable from the camelCase query"
    )


# --------------------------------------------------------------------------
# Expansion — §11.1
# --------------------------------------------------------------------------


def test_expansion_returns_inbound_callers(indexed_corpus, graph_db):
    """The reason the graph exists.

    "What calls this" is answerable only by traversal. A seed on `charge_card`
    must surface `SubscriptionView.post`, which calls it from another file and
    shares no vocabulary with it.
    """
    expander = NeighborhoodExpander(runner=graph_db)
    seed = uids_for(indexed_corpus, "billing/tasks.py", "billing.tasks.charge_card")
    assert seed, "seed symbol missing from the corpus"

    names = {n.qualified_name for n in expander.expand(REPO, seed)}
    assert "billing.views.SubscriptionView.post" in names, (
        f"inbound caller not returned; got {sorted(names)}"
    )


def test_expansion_is_bidirectional(indexed_corpus, graph_db):
    """§5.2 step 3 lists "outbound CALLS · inbound CALLS" — both directions.

    §11.1's pattern is undirected for exactly this reason. Outbound alone
    answers "what does this use"; the question users actually ask is the
    inbound one.
    """
    expander = NeighborhoodExpander(runner=graph_db)
    seed = uids_for(indexed_corpus, "billing/views.py", "billing.views.SubscriptionView.post")
    names = {n.qualified_name for n in expander.expand(REPO, seed)}
    assert "billing.tasks.charge_card" in names, "outbound call missing"


def test_expansion_excludes_the_seed_itself(indexed_corpus, graph_db):
    expander = NeighborhoodExpander(runner=graph_db)
    seed = uids_for(indexed_corpus, "common/utils.py", "common.utils.audit_event")
    assert seed[0] not in {n.uid for n in expander.expand(REPO, seed)}


def test_expansion_of_no_seeds_is_empty(graph_db):
    assert NeighborhoodExpander(runner=graph_db).expand(REPO, []) == []


def test_expansion_is_scoped_to_the_repo(indexed_corpus, graph_db):
    expander = NeighborhoodExpander(runner=graph_db)
    seed = uids_for(indexed_corpus, "billing/tasks.py", "billing.tasks.charge_card")
    assert expander.expand("some_other_repo", seed) == []


# --------------------------------------------------------------------------
# The hub cutoff
# --------------------------------------------------------------------------


@pytest.fixture
def hub_graph(graph_db, writer):
    """A `logger.warning`-shaped utility with 400 callers, plus one specific
    neighbour. §12.10 names utility-heavy codebases as the reason `hub_cutoff`
    will move; 400 is comfortably past the 100† default.

    Written under its own `repo_id` and cleaned by `repo_id`, not by wiping the
    graph. A function-scoped fixture that deletes everything destroys the
    module-scoped corpus other tests in this file depend on — which is exactly
    how the recall baseline came to report 0.000 for every question.
    """
    def sym(rel_path, qualified_name):
        name = qualified_name.rsplit(".", 1)[-1]
        return Symbol(
            uid=symbol_uid(HUB_REPO, rel_path, qualified_name, 0, 0),
            repo_id=HUB_REPO, rel_path=rel_path, qualified_name=qualified_name,
            name=name, arity=0, ordinal=0, kind="function",
            signature=f"def {name}():", docstring=None,
            source_code=f"def {name}():\n    pass\n", enclosing_signature=None,
        )

    from adapters.base import ParsedFile

    hub = sym("common/log.py", "common.log.warning")
    specific = sym("billing/tasks.py", "billing.tasks.charge_card")
    seed = sym("billing/views.py", "billing.views.post")

    callers = [sym("app/mod.py", f"app.mod.caller{i}") for i in range(400)]
    edges = [
        Edge(source_uid=c.uid, kind="CALLS", origin_path="app/mod.py", target_uid=hub.uid)
        for c in callers
    ]
    edges.append(
        Edge(source_uid=seed.uid, kind="CALLS", origin_path="billing/views.py",
             target_uid=hub.uid)
    )
    edges.append(
        Edge(source_uid=seed.uid, kind="CALLS", origin_path="billing/views.py",
             target_uid=specific.uid)
    )

    batch = {
        "common/log.py": ParsedFile("common/log.py", "h1", [hub], []),
        "billing/tasks.py": ParsedFile("billing/tasks.py", "h2", [specific], []),
        "billing/views.py": ParsedFile("billing/views.py", "h3", [seed], edges[-2:]),
        "app/mod.py": ParsedFile("app/mod.py", "h4", callers, edges[:-2]),
    }
    writer.apply(HUB_REPO, batch, batch.keys(), {}, time.time_ns())
    yield seed, hub, specific
    graph_db.run("MATCH (n:Symbol {repo_id: $r}) DETACH DELETE n", r=HUB_REPO)
    graph_db.run("MATCH (f:File {repo_id: $r}) DETACH DELETE f", r=HUB_REPO)


def test_hub_cutoff_excludes_logger(hub_graph, graph_db):
    """A 400-caller utility must not flood expansion.

    Without the cutoff, `common.log.warning` is one hop from nearly everything
    and appears in every result set — crowding out the specific neighbours that
    actually answer the question, and spending the packer's budget on a symbol
    the developer already knows.
    """
    seed, hub, specific = hub_graph
    degree = graph_db.run(
        "MATCH (s:Symbol {uid: $uid}) RETURN s.degree AS degree", uid=hub.uid
    )[0]["degree"]
    assert degree > HUB_CUTOFF, f"fixture hub has degree {degree}; cutoff is {HUB_CUTOFF}"

    names = {
        n.qualified_name for n in NeighborhoodExpander(runner=graph_db).expand(HUB_REPO, [seed.uid])
    }
    assert "common.log.warning" not in names, "the hub crossed the cutoff"
    assert "billing.tasks.charge_card" in names, "the specific neighbour was lost too"


def test_raising_the_cutoff_admits_the_hub(hub_graph, graph_db):
    """The converse: the exclusion is the cutoff doing its job, not a bug.

    A test that only ever sees the hub excluded cannot tell "filtered by degree"
    from "never traversed at all".
    """
    seed, _hub, _specific = hub_graph
    expander = NeighborhoodExpander(runner=graph_db, hub_cutoff=10_000)
    names = {n.qualified_name for n in expander.expand(HUB_REPO, [seed.uid])}
    assert "common.log.warning" in names


def test_hub_penalty_ranks_specific_above_generic(hub_graph, graph_db):
    """§11.1's `raw / log(2 + degree)`.

    Even with the cutoff raised so both survive, the 400-degree hub must rank
    below the 1-degree neighbour. §12.8 is honest that this is a proxy rather
    than centrality — but the proxy has to at least order these two correctly.
    """
    seed, _hub, _specific = hub_graph
    expander = NeighborhoodExpander(runner=graph_db, hub_cutoff=10_000)
    ranked = expander.expand(HUB_REPO, [seed.uid])
    order = [n.qualified_name for n in ranked]

    assert order.index("billing.tasks.charge_card") < order.index("common.log.warning")


def test_hub_cutoff_matches_the_spec_constant():
    """† §5.2 step 3 / §11.1's `$hub_cutoff`."""
    assert HUB_CUTOFF == 100


# --------------------------------------------------------------------------
# Recall@10 — recorded, not gated (plan §6)
# --------------------------------------------------------------------------


def test_recall_at_10_baseline(indexed_corpus, graph_db, provider, request):
    """Plan §6: "Records, does not gate yet."

    Two numbers are recorded, and only one of them means anything today.

    The **fulltext-only** figure is real: BM25 over `search_text` is a genuine
    retrieval system and this is its recall. The **blended** figure is not, and
    the reason is F-006's neighbour: `HashEmbeddingProvider` maps text to a
    deterministic hash, so a query vector bears no relation to a symbol vector.
    Its arm contributes noise, and fusing noise with signal can only push gold
    documents down. Reporting the blended number as "Recall@10" would be
    reporting the stand-in provider's absence of semantics as a retrieval
    result.

    §8.2's gate is >= 0.80 against **packed** context (§8.1), which is a
    strictly harder measurement than this one and arrives at step 8.
    """
    questions, meta = load_questions()
    backend = Neo4jSearchBackend(graph_db.driver, graph_db.name)

    def resolve(rel_path: str, qualified_name: str) -> list[str]:
        return uids_for(indexed_corpus, rel_path, qualified_name)

    fulltext_only = HybridRetriever(backend=backend)
    blended = HybridRetriever(backend=backend)

    ft_report = EvalHarness(
        resolve,
        lambda q: [c.uid for c in fulltext_only.search(REPO, q.query)],
        stage="retrieved (fulltext arm only)",
        embedding_model="n/a",
    ).run(questions, corpus=meta["corpus"])

    blended_report = EvalHarness(
        resolve,
        lambda q: [
            c.uid
            for c in blended.search(
                REPO, q.query, query_vec=provider.embed([q.query])[0]
            )
        ],
        stage="retrieved (blended, stand-in embeddings)",
        embedding_model=provider.name,
    ).run(questions, corpus=meta["corpus"])

    blended_report.notes.append(
        "The blended number is NOT a retrieval result. HashEmbeddingProvider has "
        "no semantic structure (see index/providers.py and FINDINGS F-006), so "
        "its arm contributes noise. Re-run with the real provider before "
        "recording anything against the §8.2 gate."
    )

    text = "\n\n".join(
        [
            "# Step 6 — Recall baseline",
            "Written by `tests/step_06_retrieval/test_retrieval_graph.py`. "
            "Do not hand-edit; re-run the test.",
            f"- Questions: {len(questions)} (plan §6's seed set)",
            f"- Gate for reference: Recall@10 >= {GATE_RECALL_AT_10} against "
            f"**packed** context (§8.1/§8.2) — a harder measurement than either "
            f"figure below, and not available until step 8.",
            "## Fulltext arm only (meaningful)",
            "```\n" + ft_report.render() + "\n```",
            "## Blended with stand-in embeddings (NOT meaningful)",
            "```\n" + blended_report.render() + "\n```",
        ]
    )
    (request.config.rootpath / REPORT).write_text(text, encoding="utf-8")

    # Assert only that the harness produced a real measurement. The plan is
    # explicit that step 6 records rather than gates, and a threshold asserted
    # here would be asserting against the stand-in provider.
    assert ft_report.scoreable, "no question had resolvable gold — the golden set is broken"
    assert len(ft_report.scoreable) == len(questions), (
        f"unresolved gold: {[r.unresolved_gold for r in ft_report.results if r.unresolved_gold]}"
    )
    assert 0.0 <= ft_report.recall_at(10) <= 1.0


def test_golden_set_gold_all_resolves(indexed_corpus):
    """A typo in the golden file reads as a retrieval regression.

    Checked separately from the baseline so the failure names the real cause.
    """
    questions, _meta = load_questions()
    unresolved = [
        f"{q.id}: {rel}::{qn}"
        for q in questions
        for rel, qn in q.gold
        if not uids_for(indexed_corpus, rel, qn)
    ]
    assert unresolved == [], unresolved
