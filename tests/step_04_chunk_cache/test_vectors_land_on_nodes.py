"""Step 4 — T5, end to end.

`test_vectors_merged_into_props_before_upsert` (step 3) pins the writer's merge.
This is the version the plan's §12 table names: parse a real file, chunk it,
embed it, write it, then ask the database whether every symbol actually has a
vector.

The distinction matters because T5's original form was not a wrong value but a
missing statement. `embed_batch` returned vectors, step 3a wrote `sym.props`,
and nothing joined the two. Symbols indexed with null embeddings, were invisible
to vector search, and looked entirely healthy in the graph — so the only test
that could have caught it is one that queries the graph.
"""

from __future__ import annotations

import time

import pytest

from adapters.python import PythonAdapter
from config import EMBED_DIM
from graph.writer import GraphWriter
from index.cache import EmbeddingCache, embed_batch
from index.chunker import chunk_file
from index.providers import ApproxCodeTokenizer, HashEmbeddingProvider

REPO = "repo_step4_graph"


@pytest.fixture(scope="module")
def adapter() -> PythonAdapter:
    return PythonAdapter()


@pytest.fixture
def writer(graph_db) -> GraphWriter:
    return GraphWriter(graph_db.driver, graph_db.name)


@pytest.fixture(autouse=True)
def clean(graph_db):
    graph_db.run("MATCH (n) DETACH DELETE n")
    yield
    graph_db.run("MATCH (n) DETACH DELETE n")


@pytest.fixture
def indexed(adapter, writer, graph_db, tmp_path, fixtures_dir):
    """The full step 1-4 pipeline over the Django fixture: parse, chunk, embed, write."""
    repo = fixtures_dir / "repos" / "django_min"
    tokenizer = ApproxCodeTokenizer()
    provider = HashEmbeddingProvider()
    cache = EmbeddingCache(tmp_path / "e.sqlite", model=provider.name)

    batch = {}
    for path in sorted(repo.rglob("*.py")):
        rel = path.relative_to(repo).as_posix()
        batch[rel] = chunk_file(adapter.parse(REPO, rel, path.read_bytes()), tokenizer)

    vectors = embed_batch(REPO, batch, cache=cache, provider=provider)
    report = writer.apply(REPO, batch, batch.keys(), vectors, time.time_ns())
    cache.close()
    return batch, vectors, report


def test_vectors_land_on_nodes(indexed, graph_db):
    """T5. Every symbol written must have a non-null `code_vec`."""
    batch, vectors, report = indexed

    expected = sum(len(f.symbols) for f in batch.values())
    assert report.symbols_written == expected
    assert len(vectors) == expected

    missing = graph_db.run(
        """
        MATCH (s:Symbol {repo_id: $r})
        WHERE s.code_vec IS NULL
        RETURN s.qualified_name AS qn, s.rel_path AS rel ORDER BY rel, qn
        """,
        r=REPO,
    )
    assert missing == [], (
        f"{len(missing)} symbols indexed with a null embedding: "
        f"{[(m['rel'], m['qn']) for m in missing[:5]]}"
    )

    dims = graph_db.run(
        "MATCH (s:Symbol {repo_id: $r}) RETURN DISTINCT size(s.code_vec) AS n", r=REPO
    )
    assert [row["n"] for row in dims] == [EMBED_DIM]


def test_vector_search_actually_returns_the_symbol(indexed, graph_db):
    """A stored vector that the index never returns is the same failure.

    §12.1 names this state exactly — "present in traversal and invisible to
    hybrid retrieval" — so checking the property is non-null is not enough;
    the index has to answer.
    """
    batch, vectors, _report = indexed
    target = next(
        s for f in batch.values() for s in f.symbols
        if s.qualified_name == "billing.tasks.charge_card"
    )

    graph_db.run("CALL db.awaitIndexes(120)")
    rows = graph_db.run(
        """
        CALL db.index.vector.queryNodes('symbol_code_vec', 5, $vec)
        YIELD node, score
        WHERE node.repo_id = $r
        RETURN node.qualified_name AS qn, score ORDER BY score DESC
        """,
        vec=list(vectors[target.uid]),
        r=REPO,
    )
    assert rows, "the vector index returned nothing for a vector it stores"
    assert rows[0]["qn"] == "billing.tasks.charge_card"


def test_hashes_are_persisted_for_the_header_only_metric(indexed, graph_db):
    """§4.2's header-only miss share reads `body_hash` back off the node.

    If the writer dropped it, §8.3's decision metric would silently read zero
    and the warm-reuse trigger could never fire.
    """
    rows = graph_db.run(
        """
        MATCH (s:Symbol {repo_id: $r})
        WHERE s.body_hash IS NULL OR s.header_hash IS NULL
        RETURN count(s) AS n
        """,
        r=REPO,
    )
    assert rows[0]["n"] == 0


def test_scrubbed_source_is_what_reaches_the_graph(adapter, writer, graph_db, tmp_path):
    """§9.2: "the unredacted form is never stored".

    Not "is redacted before display" — never stored. So the assertion belongs
    against the database, not against the in-memory symbol.
    """
    source = (
        b'def connect():\n'
        b'    key = "AKIAIOSFODNN7EXAMPLE"\n'
        b'    return key\n'
    )
    tokenizer = ApproxCodeTokenizer()
    provider = HashEmbeddingProvider()
    cache = EmbeddingCache(tmp_path / "e.sqlite", model=provider.name)

    batch = {"authx/secrets.py": chunk_file(
        adapter.parse(REPO, "authx/secrets.py", source), tokenizer
    )}
    vectors = embed_batch(REPO, batch, cache=cache, provider=provider)
    writer.apply(REPO, batch, batch.keys(), vectors, time.time_ns())
    cache.close()

    stored = graph_db.run(
        "MATCH (s:Symbol {repo_id: $r}) "
        "RETURN s.source_code AS src, s.docstring AS doc, keys(s) AS keys",
        r=REPO,
    )
    assert stored
    for row in stored:
        assert "AKIAIOSFODNN7EXAMPLE" not in (row["src"] or "")
        assert "AKIAIOSFODNN7EXAMPLE" not in (row["doc"] or "")
        # `chunk_text` is deliberately not persisted: §3.3's property list does
        # not include it, the packer renders from `source_code`, and storing it
        # would duplicate the body plus its header on every node.
        assert "chunk_text" not in row["keys"]

    # And the in-memory symbol is redacted too — the scrub happens at chunk
    # construction, so there is no window in which the raw form exists downstream.
    assert all(
        "AKIAIOSFODNN7EXAMPLE" not in (s.chunk_text or "")
        for f in batch.values()
        for s in f.symbols
    )
