"""S6 — the only thing in the system that writes to Neo4j.

Implements spec §11.2 steps 2a-6b. Two properties matter more than the code:

**The sequence is not atomic.** It spans several statements and, at scale,
several transactions. What it provides instead (§4.3):

  *Superset-on-prefix* — any prefix leaves the graph a superset of the truth:
  stale rows, never dangling references, because every write precedes every
  delete.
  *Idempotent replay* — re-running from step 1 with the same epoch converges.
  Nothing depends on observing an intermediate state.

Describing this as atomic, or as one transaction, would be worse than saying
nothing: §12.1's split-store analysis is built on the weaker claim being the
true one.

**One chokepoint.** §9.1's tenancy model is `repo_id` on every node enforced
through this class and `GraphReader`. No raw Cypher anywhere else, so there is
one place to audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from neo4j import Driver

from adapters.base import Edge, ParsedFile, Symbol
from config import EDGE_TYPE_ALLOWLIST, WRITE_CHUNK_SIZE
from index.moves import Remap

Vector = Sequence[float]

#: The boundaries §4.3's guarantee is stated over. `apply` can be stopped after
#: any of them, which is how `test_superset_on_prefix` injects a crash without
#: monkeypatching the statements it is trying to verify.
STEP_BOUNDARIES: tuple[str, ...] = ("3a", "3b", "4", "5", "6a", "6b")


class InjectedCrash(RuntimeError):
    """Raised by `apply(stop_after=...)`. Test scaffolding, never production."""


@dataclass
class WriteReport:
    """What one `apply` actually did. Feeds §9.4's spans."""

    symbols_written: int = 0
    edges_written: int = 0
    edges_unresolved: int = 0
    edges_ambiguous: int = 0
    stale_edges_deleted: int = 0
    orphans_deleted: int = 0
    stopped_after: str | None = None


# --------------------------------------------------------------------------
# Property serialisation
# --------------------------------------------------------------------------

#: §3.3's property list, minus the ones the writer sets itself (`epoch`,
#: `origin_path`, `repo_id`) and minus `vec_epoch`, which T9 marks Phase 1 and
#: nothing writes at MVP.
_PROP_FIELDS = (
    "uid", "rel_path", "qualified_name", "name", "arity", "ordinal", "kind",
    "signature", "docstring", "source_code", "enclosing_signature",
    "search_text", "body_hash", "header_hash", "used_imports", "code_vec",
    "start_line", "end_line", "is_client_component", "decorators",
)


def symbol_props(sym: Symbol) -> dict:
    """§11.2 3a's `sym.props`.

    `None` values are dropped rather than sent. `SET s += props` treats a null
    as "remove this property", so sending `code_vec: None` for a symbol the
    embedder happened to skip would erase a vector that was already correct —
    a T5-shaped failure arriving by a different route.
    """
    props = {}
    for field in _PROP_FIELDS:
        value = getattr(sym, field, None)
        if value is None:
            continue
        props[field] = value
    return props


def merge_vectors(batch: dict[str, ParsedFile], vectors: dict[str, Vector]) -> int:
    """T5. Merge `embed_batch`'s return into each symbol before the node upsert.

    v9.2 returned vectors that no statement consumed: symbols indexed with null
    embeddings, were invisible to vector search, and looked entirely healthy in
    the graph. This is the line that was missing.
    """
    merged = 0
    for parsed in batch.values():
        for sym in parsed.symbols:
            vec = vectors.get(sym.uid)
            if vec is not None:
                sym.code_vec = list(vec)
                merged += 1
    return merged


def _chunked(items: Sequence, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


# --------------------------------------------------------------------------
# The writer
# --------------------------------------------------------------------------


class GraphWriter:
    def __init__(self, driver: Driver, database: str) -> None:
        self._driver = driver
        self._database = database

    def _run(self, cypher: str, **params):
        with self._driver.session(database=self._database) as session:
            return list(session.run(cypher, params))

    # -- step 2: remaps ----------------------------------------------------

    def remap_uids(self, repo_id: str, remaps: list[Remap], epoch: int) -> int:
        """§11.2 steps 2a and 2b — the in-place UID rewrite.

        Applied *before* the upsert so `MERGE` lands on corrected UIDs rather
        than creating duplicates that step 5 would delete, taking their inbound
        edges with them (§4.3).

        Chains are collapsed in the driver (§4.1 `_collapse_chains`), so no row
        here depends on another row's `SET` having landed first — which also
        removes any dependence on Cypher's row ordering.
        """
        if not remaps:
            return 0

        payload = [
            {
                "old_uid": r.old_uid,
                "new_uid": r.new_uid,
                "old_path": r.old_path,
                "new_path": r.new_path,
            }
            for r in remaps
        ]

        # 2a. Rewrite symbol UIDs in place. Inbound edges reference uid and are
        # preserved without being touched.
        rewritten = self._run(
            """
            UNWIND $remaps AS m
            MATCH (s:Symbol {uid: m.old_uid, repo_id: $repo_id})
            WHERE NOT EXISTS { MATCH (c:Symbol {uid: m.new_uid}) }
            SET s.uid = m.new_uid, s.rel_path = m.new_path,
                s.origin_path = m.new_path, s.epoch = $epoch
            RETURN count(s) AS n
            """,
            remaps=payload,
            repo_id=repo_id,
            epoch=epoch,
        )

        # 2b. Repoint origin_path on edges the moved file originated, so step 4
        # scopes correctly. Anchored on the indexed :Symbol node, not an
        # unanchored relationship scan.
        self._run(
            """
            UNWIND $remaps AS m
            MATCH (s:Symbol {uid: m.new_uid})-[r]->()
            WHERE r.origin_path = m.old_path
            SET r.origin_path = m.new_path, r.epoch = $epoch
            """,
            remaps=payload,
            epoch=epoch,
        )

        # The :File node moves too. §11.2 omits it; see FINDINGS.md F-004.
        self._run(
            """
            UNWIND $remaps AS m
            MATCH (f:File {repo_id: $repo_id, rel_path: m.old_path})
            WHERE NOT EXISTS {
                MATCH (g:File {repo_id: $repo_id, rel_path: m.new_path})
            }
            SET f.rel_path = m.new_path, f.epoch = $epoch
            """,
            remaps=payload,
            repo_id=repo_id,
            epoch=epoch,
        )

        return rewritten[0]["n"] if rewritten else 0

    # -- steps 3-6: the batch ---------------------------------------------

    def apply(
        self,
        repo_id: str,
        batch: dict[str, ParsedFile],
        batch_paths: Iterable[str],
        vectors: dict[str, Vector],
        epoch: int,
        *,
        stop_after: str | None = None,
    ) -> WriteReport:
        """Run §11.2 steps 3a-6b for one batch, under one epoch.

        `batch_paths` is `batch.keys() | {move.old_path}` (§4.3). The moved-from
        paths must stay in scope or a remap collision leaves an orphan node and
        un-repointed edges uncollected (v9.1 #16).

        `stop_after` names a boundary from `STEP_BOUNDARIES` and raises
        `InjectedCrash` once it is reached. It exists so crash-safety can be
        tested against the real statement sequence rather than a mock of it.
        """
        if stop_after is not None and stop_after not in STEP_BOUNDARIES:
            raise ValueError(f"unknown boundary {stop_after!r}; expected one of "
                             f"{STEP_BOUNDARIES}")

        paths = sorted(set(batch_paths))
        report = WriteReport(stopped_after=stop_after)

        # T5: vectors must reach sym.props before the node upsert, or step 3a
        # persists null embeddings.
        merge_vectors(batch, vectors)

        symbols = [s for parsed in batch.values() for s in parsed.symbols]
        edges = [e for parsed in batch.values() for e in parsed.edges]

        def boundary(name: str) -> None:
            if stop_after == name:
                raise InjectedCrash(f"stopped after §11.2 step {name}")

        # -- 3a. Nodes -----------------------------------------------------
        self._upsert_files(repo_id, batch, epoch)
        report.symbols_written = self._upsert_symbols(repo_id, symbols, epoch)
        boundary("3a")

        # -- 3b. Edges, one statement per relationship type -----------------
        resolved = self._resolve_edge_targets(repo_id, edges, report)
        report.edges_written = self._upsert_edges(resolved, epoch)
        boundary("3b")

        # -- 4. Stale edges, scoped to this batch's paths --------------------
        report.stale_edges_deleted = self._delete_stale_edges(repo_id, paths, epoch)
        boundary("4")

        # -- 5. Orphan symbols ----------------------------------------------
        report.orphans_deleted = self._delete_orphans(repo_id, paths, epoch)
        boundary("5")

        # -- 6a / 6b. Degrees -----------------------------------------------
        self._refresh_own_degrees(repo_id, paths, epoch)
        boundary("6a")
        self._refresh_neighbor_degrees(repo_id, paths, epoch)
        boundary("6b")

        return report

    # -- individual statements --------------------------------------------

    def _upsert_files(self, repo_id: str, batch: dict[str, ParsedFile], epoch: int) -> None:
        """§3.3's `:File` node.

        Not in §11.2's numbered sequence — see FINDINGS.md F-004. It has to
        exist: §4.5's reconcile diff reads `reader.file_hashes(repo)`, and
        `Indexer.startup` seeds `_indexed` from `reader.known_paths(repo)`.
        Grouped with 3a because it is a write, and every write precedes every
        delete.
        """
        rows = [
            {"rel_path": rel, "content_hash": parsed.content_hash}
            for rel, parsed in batch.items()
        ]
        if not rows:
            return
        for chunk in _chunked(rows, WRITE_CHUNK_SIZE):
            self._run(
                """
                UNWIND $files AS f
                MERGE (file:File {repo_id: $repo_id, rel_path: f.rel_path})
                SET file.content_hash = f.content_hash, file.epoch = $epoch
                """,
                files=chunk,
                repo_id=repo_id,
                epoch=epoch,
            )

    def _upsert_symbols(self, repo_id: str, symbols: list[Symbol], epoch: int) -> int:
        """§11.2 step 3a."""
        if not symbols:
            return 0
        payload = [{"uid": s.uid, "props": symbol_props(s), "rel_path": s.rel_path}
                   for s in symbols]
        written = 0
        for chunk in _chunked(payload, WRITE_CHUNK_SIZE):
            rows = self._run(
                """
                UNWIND $symbols AS sym
                MERGE (s:Symbol {uid: sym.uid})
                  ON CREATE SET s.created_epoch = $epoch
                SET s += sym.props,
                    s.epoch = $epoch, s.origin_path = sym.rel_path,
                    s.repo_id = $repo_id
                RETURN count(s) AS n
                """,
                symbols=chunk,
                repo_id=repo_id,
                epoch=epoch,
            )
            written += rows[0]["n"] if rows else 0
        return written

    def _resolve_edge_targets(
        self, repo_id: str, edges: list[Edge], report: WriteReport
    ) -> list[Edge]:
        """Turn `target_hint` into `target_uid` where the graph can say so.

        The adapter resolves same-file targets; everything else arrives as an
        absolute qualified name. Resolution is deliberately strict:

        * exactly one candidate -> resolve;
        * more than one -> refuse and count it. §5.1 establishes the house rule
          that ambiguity is refusal, and an edge pointing at an arbitrary one of
          two same-named symbols is a wrong answer that looks like a right one;
        * none -> leave unresolved. §11.2 3b matches both endpoints by uid, so
          the edge is simply never written — correct for a call into a
          third-party library.

        A hint like `authx.models.User.objects.filter` is retried against
        successively shorter prefixes, because attribute access on a symbol is
        still a reference to that symbol. The trimming is deterministic and
        stops at the first prefix with exactly one candidate.
        """
        pending = [e for e in edges if e.target_uid is None and e.target_hint]
        already = [e for e in edges if e.target_uid is not None]
        if not pending:
            report.edges_unresolved = len(edges) - len(already)
            return already

        candidates: set[str] = set()
        for edge in pending:
            parts = edge.target_hint.split(".")
            for cut in range(len(parts), 0, -1):
                candidates.add(".".join(parts[:cut]))

        counts: dict[str, list[str]] = {}
        names = sorted(candidates)
        for chunk in _chunked(names, WRITE_CHUNK_SIZE):
            for row in self._run(
                """
                UNWIND $names AS n
                MATCH (s:Symbol {repo_id: $repo_id, qualified_name: n})
                RETURN n AS name, collect(s.uid) AS uids
                """,
                names=chunk,
                repo_id=repo_id,
            ):
                counts[row["name"]] = row["uids"]

        out = list(already)
        for edge in pending:
            parts = edge.target_hint.split(".")
            for cut in range(len(parts), 0, -1):
                uids = counts.get(".".join(parts[:cut]))
                if not uids:
                    continue
                if len(uids) > 1:
                    report.edges_ambiguous += 1
                    break                 # refuse, do not guess
                out.append(
                    Edge(
                        source_uid=edge.source_uid,
                        kind=edge.kind,
                        origin_path=edge.origin_path,
                        target_uid=uids[0],
                    )
                )
                break
            else:
                report.edges_unresolved += 1
        return out

    def _upsert_edges(self, edges: list[Edge], epoch: int) -> int:
        """§11.2 step 3b — ONE STATEMENT PER RELATIONSHIP TYPE.

        `MERGE` accepts exactly one type; `MERGE (a)-[:A|B]->(b)` is a syntax
        error (confirmed by `probe_merge_multitype`). The type is substituted
        from `EDGE_TYPE_ALLOWLIST`, never from parsed input — the allowlist is
        what keeps a hostile or merely surprising identifier in a source file
        from becoming a relationship type.

        Transaction subqueries cannot follow an updating clause, so batching is
        the driver's job here, not `CALL { } IN TRANSACTIONS`.
        """
        by_kind: dict[str, list[dict]] = {}
        for edge in edges:
            if edge.target_uid is None:
                continue
            if edge.kind not in EDGE_TYPE_ALLOWLIST:
                raise ValueError(
                    f"edge kind {edge.kind!r} is not in the §11.2 3b allowlist "
                    f"{EDGE_TYPE_ALLOWLIST}"
                )
            by_kind.setdefault(edge.kind, []).append(
                {
                    "source_uid": edge.source_uid,
                    "target_uid": edge.target_uid,
                    "origin_path": edge.origin_path,
                }
            )

        written = 0
        for kind in EDGE_TYPE_ALLOWLIST:          # fixed order: replay stability
            rows = by_kind.get(kind)
            if not rows:
                continue
            for chunk in _chunked(rows, WRITE_CHUNK_SIZE):
                result = self._run(
                    f"""
                    UNWIND $edges AS e
                    MATCH (src:Symbol {{uid: e.source_uid}})
                    MATCH (tgt:Symbol {{uid: e.target_uid}})
                    MERGE (src)-[r:{kind} {{origin_path: e.origin_path}}]->(tgt)
                    SET r.epoch = $epoch
                    RETURN count(r) AS n
                    """,
                    edges=chunk,
                    epoch=epoch,
                )
                written += result[0]["n"] if result else 0
        return written

    def _delete_stale_edges(self, repo_id: str, paths: list[str], epoch: int) -> int:
        """§11.2 step 4. Only edges *originated by* paths in this batch.

        `s.epoch <= $epoch` is not a filter — it is what makes the
        `symbol_origin` index usable, and §11.2 carries it as of v10.1. Neo4j
        will not seek a composite index on a prefix of its properties (measured:
        hinting `(repo_id, origin_path)` reports "the hinted index does not
        exist"), so v10.0's statement, constraining only two of the three, fell
        back to `NodeByLabelScan` — the exact defect B8 names. (v10.1 §0.0 R3;
        FINDINGS F-011 carries the plans.)

        It is a tautology here: step 3a has just stamped every symbol in the
        batch with `$epoch`, and anything else at these paths is older. A symbol
        with a *newer* epoch would belong to a later batch, and excluding it is
        correct rather than merely harmless.
        """
        if not paths:
            return 0
        rows = self._run(
            """
            MATCH (s:Symbol)
            WHERE s.repo_id = $repo_id AND s.origin_path IN $batch_paths
              AND s.epoch <= $epoch
            MATCH (s)-[r]->()
            WHERE r.epoch < $epoch
            DELETE r
            RETURN count(r) AS n
            """,
            repo_id=repo_id,
            batch_paths=paths,
            epoch=epoch,
        )
        return rows[0]["n"] if rows else 0

    def _delete_orphans(self, repo_id: str, paths: list[str], epoch: int) -> int:
        """§11.2 step 5, plus the `:File` rows the same paths no longer back."""
        if not paths:
            return 0
        rows = self._run(
            """
            MATCH (s:Symbol)
            WHERE s.repo_id = $repo_id AND s.origin_path IN $batch_paths
              AND s.epoch < $epoch
            DETACH DELETE s
            RETURN count(s) AS n
            """,
            repo_id=repo_id,
            batch_paths=paths,
            epoch=epoch,
        )
        self._run(
            """
            MATCH (f:File)
            WHERE f.repo_id = $repo_id AND f.rel_path IN $batch_paths
              AND f.epoch < $epoch
            DETACH DELETE f
            """,
            repo_id=repo_id,
            batch_paths=paths,
            epoch=epoch,
        )
        return rows[0]["n"] if rows else 0

    def _refresh_own_degrees(self, repo_id: str, paths: list[str], epoch: int) -> None:
        """§11.2 step 6a.

        Omitting this leaves new symbols with a null `degree`, which §11.1's
        score denominator then treats as 1 — systematic over-scoring of exactly
        the symbols just written (v9.1 #13).
        """
        if not paths:
            return
        self._run(
            """
            MATCH (s:Symbol)
            WHERE s.repo_id = $repo_id AND s.origin_path IN $batch_paths
              AND s.epoch <= $epoch
            SET s.degree = COUNT { (s)--() }
            """,
            repo_id=repo_id,
            batch_paths=paths,
            epoch=epoch,
        )

    def _refresh_neighbor_degrees(self, repo_id: str, paths: list[str], epoch: int) -> None:
        """§11.2 step 6b. Idempotent overlap with 6a is harmless."""
        if not paths:
            return
        self._run(
            """
            MATCH (s:Symbol)
            WHERE s.repo_id = $repo_id AND s.origin_path IN $batch_paths
              AND s.epoch <= $epoch
            MATCH (s)-[]-(n:Symbol)
            WITH DISTINCT n
            SET n.degree = COUNT { (n)--() }
            """,
            repo_id=repo_id,
            batch_paths=paths,
            epoch=epoch,
        )

    def refresh_all_degrees(self, repo_id: str) -> None:
        """§4.5's repo-wide pass. One sweep beats N local ones after a bulk."""
        self._run(
            """
            MATCH (s:Symbol {repo_id: $repo_id})
            SET s.degree = COUNT { (s)--() }
            """,
            repo_id=repo_id,
        )
