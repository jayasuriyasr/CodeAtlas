"""S5 — the embedding cache — and §4.2's `embed_batch`.

§4.2 calls this "the structural cost control": the saving comes from a
composite key that survives ordinary edits, not from developer discipline about
when to sync. §8.2 gates the hit rate at >=0.90 for a specific reason — "the
failure is silent and cannot be noticed, only measured".
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import metrics
from adapters.base import ParsedFile, Symbol
from config import EMBED_DIM, EMBEDDING_CACHE_PATH
from index.chunker import cache_key

Vector = Sequence[float]


# --------------------------------------------------------------------------
# S2 — the provider seam
# --------------------------------------------------------------------------


class EmbeddingProvider(Protocol):
    """S2's `.embed()` half.

    Swapping the *LLM* is one file. Swapping this is a full re-index (§12.11):
    new dimensionality means dropping and recreating the vector index,
    re-embedding every symbol, and invalidating the entire cache. `name` and
    `dimensions` are on the interface so a mismatch is caught at startup rather
    than discovered as a silently empty vector search.
    """

    name: str
    dimensions: int

    def embed(self, texts: list[str]) -> list[Vector]: ...


# --------------------------------------------------------------------------
# S5 — SQLite
# --------------------------------------------------------------------------


class EmbeddingCache:
    """`cache_key -> vector`, keyed by §4.2's composite.

    The model name is part of the primary key, not just a column. Two models
    produce different vectors for identical text, and a cache that returned one
    model's vector for another's query would put a wrong answer into a vector
    index that has no way to notice.
    """

    def __init__(self, path: Path | str = EMBEDDING_CACHE_PATH, *, model: str = "",
                 dimensions: int = EMBED_DIM) -> None:
        self.path = Path(path)
        self.model = model
        self.dimensions = dimensions
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                cache_key  TEXT NOT NULL,
                model      TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                vector     TEXT NOT NULL,
                PRIMARY KEY (cache_key, model)
            )
            """
        )
        self._conn.commit()

    def get_many(self, keys: list[str]) -> dict[str, Vector]:
        if not keys:
            return {}
        out: dict[str, Vector] = {}
        # SQLite caps host parameters (999 on older builds); chunk rather than
        # discover the limit in production.
        for start in range(0, len(keys), 500):
            window = keys[start : start + 500]
            placeholders = ",".join("?" * len(window))
            rows = self._conn.execute(
                f"SELECT cache_key, vector, dimensions FROM embeddings "
                f"WHERE model = ? AND cache_key IN ({placeholders})",
                [self.model, *window],
            ).fetchall()
            for key, blob, dims in rows:
                if dims != self.dimensions:
                    continue          # a stale row from a different model version
                out[key] = json.loads(blob)
        return out

    def put_many(self, pairs: Iterable[tuple[str, Vector]]) -> int:
        rows = [
            (key, self.model, self.dimensions, json.dumps(list(vec)))
            for key, vec in pairs
        ]
        if not rows:
            return 0
        self._conn.executemany(
            "INSERT OR REPLACE INTO embeddings "
            "(cache_key, model, dimensions, vector) VALUES (?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        return len(rows)

    def __len__(self) -> int:
        return self._conn.execute(
            "SELECT count(*) FROM embeddings WHERE model = ?", (self.model,)
        ).fetchone()[0]

    def close(self) -> None:
        self._conn.close()


# --------------------------------------------------------------------------
# §4.2 — embed_batch
# --------------------------------------------------------------------------


class BodyHashSource(Protocol):
    """Just the `GraphReader` method §4.2 needs, so tests need no database."""

    def body_hashes(self, repo_id: str, uids: list[str]) -> dict[str, str]: ...


def embed_batch(
    repo_id: str,
    batch: dict[str, ParsedFile],
    *,
    cache: EmbeddingCache,
    provider: EmbeddingProvider,
    reader: BodyHashSource | None = None,
) -> dict[str, Vector]:
    """§4.2, with the metrics §8.3 needs to decide about warm reuse.

    The returned map is `uid -> vector`. `GraphWriter.apply` merges it into
    `sym.props` before step 3a — that merge is T5, and without it symbols index
    with null embeddings, are invisible to vector search, and look entirely
    healthy in the graph.
    """
    syms: list[Symbol] = [s for f in batch.values() for s in f.symbols]
    if not syms:
        return {}

    if provider.dimensions != cache.dimensions:
        raise ValueError(
            f"provider {provider.name} emits {provider.dimensions} dims but the "
            f"cache holds {cache.dimensions}. §12.11: changing the embedding "
            f"model is a full re-index, not a config change."
        )

    keys = {s.uid: cache_key(s) for s in syms}
    hits = dict(cache.get_many(sorted(set(keys.values()))))

    cold = [s for s in syms if keys[s.uid] not in hits]
    metrics.incr("cache.lookups", len(syms))
    metrics.incr("cache.hits", len(syms) - len(cold))

    if cold:
        # §8.3's header-only miss share: a miss whose stored body_hash still
        # matches was a header change, and that ratio decides whether warm reuse
        # is worth building. An absent uid is a new symbol and counts as a body
        # change, which is what `prev.get` returning None gives us.
        if reader is not None:
            prev = reader.body_hashes(repo_id, [s.uid for s in cold])
            metrics.incr(
                "cache.miss.header_only",
                sum(1 for s in cold if prev.get(s.uid) == s.body_hash),
            )
        metrics.incr("cache.miss.total", len(cold))

        fresh = provider.embed([s.chunk_text or "" for s in cold])
        if len(fresh) != len(cold):
            raise ValueError(
                f"provider {provider.name} returned {len(fresh)} vectors for "
                f"{len(cold)} inputs"
            )
        cache.put_many(zip((keys[s.uid] for s in cold), fresh))
        hits.update(dict(zip((keys[s.uid] for s in cold), fresh)))

    return {s.uid: hits[keys[s.uid]] for s in syms}
