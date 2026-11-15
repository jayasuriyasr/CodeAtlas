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

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from neo4j import Driver

import metrics
from adapters.base import Edge, ParsedFile, Symbol
from config import EDGE_TYPE_ALLOWLIST, MRO_MAX_DEPTH, WRITE_CHUNK_SIZE
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
    #: Paths outside this batch whose symbols had edges into a symbol this
    #: batch orphaned — a function whose arity changed gets a new UID, and
    #: step 5's DETACH DELETE takes its incoming edges with it. The indexer
    #: re-reads these files so their edges resolve to the new node.
    relink_paths: list[str] = field(default_factory=list)
    #: Unresolved call names per path, recorded on its `:File` node so a later
    #: write that creates one of those names can re-link the caller (A3).
    waiting: dict[str, set[str]] = field(default_factory=dict)


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
    "start_line", "end_line", "is_client_component", "decorators", "bases", "attributes",
)


def symbol_props(sym: Symbol) -> dict:
    """§11.2 3a's `sym.props`.

    `None` values are dropped rather than sent. `SET s += props` treats a null
    as "remove this property", so sending `code_vec: None` for a symbol the
    embedder happened to skip would erase a vector that was already correct —
    a T5-shaped failure arriving by a different route.
    """
    props = {}
    for name in _PROP_FIELDS:
        value = getattr(sym, name, None)
        if value is None:
            continue
        props[name] = value
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


#: Sentinel for "a candidate set was found and refused" — distinct from None,
#: which means nothing was found at all. The two feed different counters.
_REFUSED = object()


def _c3(
    cls: str, classes: dict[str, list[str] | None], memo: dict, _stack: tuple = ()
) -> list[str] | None:
    """The C3 linearisation of `cls` — Python's method resolution order.

    A name with no known bases (`classes[name]` is None: a library class, or
    one the graph has twice) is linearised as itself alone, an opaque leaf the
    caller will refuse to walk past. Returns None for a hierarchy C3 rejects
    (Python raises `TypeError` for the same classes) and for a cycle.
    """
    if cls in memo:
        return memo[cls]
    bases = classes.get(cls)
    if bases is None:
        memo[cls] = [cls]
        return memo[cls]
    if cls in _stack:
        return None                                   # a cycle: no order exists
    sequences = []
    for base in bases:
        order = _c3(base, classes, memo, (*_stack, cls))
        if order is None:
            memo[cls] = None
            return None
        sequences.append(list(order))
    sequences.append(list(bases))

    result = [cls]
    while True:
        sequences = [seq for seq in sequences if seq]
        if not sequences:
            break
        for seq in sequences:
            head = seq[0]
            if not any(head in other[1:] for other in sequences):
                break
        else:
            memo[cls] = None                          # inconsistent hierarchy
            return None
        result.append(head)
        for seq in sequences:
            if seq[0] == head:
                seq.pop(0)
    memo[cls] = result
    return result


def _walk_mro(
    order: list[str] | None,
    classes: dict[str, list[str] | None],
    method: str,
    defined: dict[str, list[str]],
) -> tuple[str | None, str]:
    """The uid `method` resolves to along `order`, and why, as `(uid, outcome)`.

    Outcomes: "resolved"; "ambiguous" (two symbols of that name on one class);
    "uncertain" (an opaque base came first, or no order exists); "absent"
    (no class on the order defines it — a library method, or a typo).
    """
    if order is None:
        return None, "uncertain"
    for owner in order:
        if classes.get(owner) is None:
            return None, "uncertain"
        uids = defined.get(f"{owner}.{method}", [])
        if len(uids) > 1:
            return None, "ambiguous"
        if uids:
            return uids[0], "resolved"
    return None, "absent"


def _hint_prefixes(hint: str) -> list[str]:
    """`a.b.c` -> `[a.b.c, a.b, a]`, longest first.

    A prefix with an empty segment is skipped: no qualified name contains one,
    so a hint like `..lib.x` (an unresolved relative path) can only produce
    look-ups that are guaranteed to miss.
    """
    parts = hint.split(".")
    return [
        ".".join(parts[:cut])
        for cut in range(len(parts), 0, -1)
        if all(parts[:cut])
    ]


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

        # The :File node moves too. §11.2 omits it; see F-004 in DESIGN.md.
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
        resolved = self._resolve_edge_targets(repo_id, edges, report, paths, epoch)
        resolved += self._override_edges(repo_id, symbols, paths, epoch)
        report.edges_written = self._upsert_edges(resolved, epoch)
        boundary("3b")

        # -- 4. Stale edges, scoped to this batch's paths --------------------
        report.stale_edges_deleted = self._delete_stale_edges(repo_id, paths, epoch)
        boundary("4")

        # -- 5. Orphan symbols ----------------------------------------------
        # Who points at an orphan from outside the batch is read first:
        # DETACH DELETE removes those edges, and nothing else would put
        # them back until the callers' own files were written.
        report.relink_paths = self._outside_callers_of_orphans(repo_id, paths, epoch)
        report.orphans_deleted = self._delete_orphans(repo_id, paths, epoch)
        # Files elsewhere that were waiting for a name this batch created.
        self._record_waiting(repo_id, paths, report.waiting)
        report.relink_paths = sorted(
            set(report.relink_paths) | set(self._files_waiting_for_new_names(repo_id, paths, epoch))
        )
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

        Not in §11.2's numbered sequence — see F-004 in DESIGN.md. It has to
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
        self,
        repo_id: str,
        edges: list[Edge],
        report: WriteReport,
        batch_paths: list[str],
        epoch: int,
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
        stops at the first prefix with exactly one candidate — **but never at a
        module reached by trimming.** `pkg.z_callee.helper` with `z_callee` not
        yet indexed would otherwise trim to the package `pkg` and write
        `run -CALLS-> pkg`: a confident edge to the wrong node, while `helper`
        reports zero callers (FINDINGS F-015). A module is a valid target only
        when the hint names it exactly, as an `import` does.

        Two more rules keep the candidate set honest:

        * **Symbols this batch is about to delete are not candidates.** A
          symbol at a batch path with an older epoch is step 5's orphan — the
          previous version of something 3a just rewrote. Counting it makes a
          signature change and its caller, saved together, look ambiguous, and
          the edge is refused for a reason that is gone one statement later.
        * **The lookup goes through `symbol_name`.** `qualified_name` has no
          index (§11.3), so matching on it alone is a label scan per name. Every
          symbol's `name` is the last segment of its `qualified_name`, so the
          pair is an exact, index-served restatement of the same predicate.
        """
        pending = [e for e in edges if e.target_uid is None and e.target_hint]
        already = [e for e in edges if e.target_uid is not None]
        report.edges_unresolved = len(edges) - len(already) - len(pending)
        members = [e for e in pending if e.through_bases]
        pending = [e for e in pending if not e.through_bases]
        if not pending:
            return already + self._resolve_members(repo_id, members, report, batch_paths, epoch)

        candidates: set[str] = set()
        for edge in pending:
            candidates.update(_hint_prefixes(edge.target_hint))

        found: dict[str, list[dict]] = {}
        rows = [{"qn": qn, "name": qn.rsplit(".", 1)[-1]} for qn in sorted(candidates)]
        for chunk in _chunked(rows, WRITE_CHUNK_SIZE):
            for row in self._run(
                """
                UNWIND $names AS n
                MATCH (s:Symbol {repo_id: $repo_id, name: n.name})
                WHERE s.qualified_name = n.qn
                  AND NOT (s.origin_path IN $batch_paths AND s.epoch < $epoch)
                RETURN n.qn AS name, collect({uid: s.uid, kind: s.kind}) AS candidates
                """,
                names=chunk,
                repo_id=repo_id,
                batch_paths=batch_paths,
                epoch=epoch,
            ):
                found[row["name"]] = row["candidates"]

        out = list(already)
        for edge in pending:
            full = edge.target_hint
            target = None
            for prefix in _hint_prefixes(full):
                if edge.kind == "INHERITS" and prefix != full:
                    # A base class is named exactly or not at all: trimming
                    # `Config.Nested` to `Config` would record an inheritance
                    # the code never states.
                    break
                hits = found.get(prefix)
                if not hits:
                    continue
                if len(hits) > 1:
                    report.edges_ambiguous += 1
                    target = _REFUSED
                elif prefix != full and hits[0]["kind"] == "module":
                    target = None             # trimmed onto a module: a guess
                else:
                    target = hits[0]["uid"]
                break
            if target is None:
                report.edges_unresolved += 1
                report.waiting.setdefault(edge.origin_path, set()).add(full)
            elif target is not _REFUSED:
                out.append(
                    Edge(
                        source_uid=edge.source_uid,
                        kind=edge.kind,
                        origin_path=edge.origin_path,
                        target_uid=target,
                    )
                )
        out.extend(self._resolve_members(repo_id, members, report, batch_paths, epoch))
        return out

    # -- member lookups: `self.method()` through the class hierarchy --------------

    def _resolve_members(
        self,
        repo_id: str,
        edges: list[Edge],
        report: WriteReport,
        batch_paths: list[str],
        epoch: int,
    ) -> list[Edge]:
        """Resolve `Class.method` member lookups by the class's MRO.

        The method resolution order is computed with C3 — what Python uses —
        from each class's recorded `bases`. A base the graph cannot place (a
        library class, or a name defined twice) is opaque: walking the order
        and reaching it before any repository class defines the method is a
        refusal, because that base may define the method itself. TypeScript's
        single `extends` chain is the one-base case of the same walk.

        Step 3a has already written this batch's symbols, so the graph holds
        their new `bases`; symbols the batch is about to orphan are excluded
        exactly as in `_resolve_edge_targets`.
        """
        if not edges:
            return []
        edges, wanted = self._through_attributes(repo_id, edges, report, batch_paths, epoch)
        if not edges:
            return []
        classes = self._class_bases(repo_id, {cls for cls, _ in wanted}, batch_paths, epoch)
        memo: dict[str, list[str] | None] = {}
        orders = {cls: _c3(cls, classes, memo) for cls, _ in wanted}

        lookups = {
            f"{owner}.{method}"
            for (cls, method) in wanted
            for owner in (orders[cls] or [])
            if classes.get(owner) is not None
        }
        defined = self._exact_symbols(repo_id, lookups, batch_paths, epoch)

        out: list[Edge] = []
        for edge, (cls, method) in zip(edges, wanted, strict=True):
            if edge.if_class and classes.get(cls) is None:
                # `curl.post()`, `timezone.now()`: not a repository class, so
                # nothing to look up; the ordinary edge for the call stands.
                metrics.incr("writer.member_not_a_class")
                continue
            order = orders[cls]
            if order is not None and edge.after_owner:
                order = order[1:]                 # super(): start after the class
            target, outcome = _walk_mro(order, classes, method, defined)
            if outcome == "ambiguous":
                report.edges_ambiguous += 1
                metrics.incr("writer.member_ambiguous")
            elif target is None:
                report.edges_unresolved += 1
                metrics.incr(f"writer.member_{outcome}")
            else:
                metrics.incr("writer.member_resolved")
                out.append(Edge(source_uid=edge.source_uid, kind=edge.kind,
                                origin_path=edge.origin_path, target_uid=target))
        return out

    def _override_edges(
        self, repo_id: str, symbols: list[Symbol], batch_paths: list[str], epoch: int
    ) -> list[Edge]:
        """OVERRIDES from each method in the batch to the method it overrides.

        The overridden method is the nearest definition of the same name along
        the class's resolution order, after the class itself — what `super()`
        would reach. A library base reached first makes it unknown, so no edge.
        Computed when the subclass's file is written; a base gaining a method
        later is picked up when the subclass's file is next written.
        """
        classes_here = {s.qualified_name for s in symbols if s.kind == "class"}
        methods = [s for s in symbols if s.kind == "function"
                   and s.qualified_name.rsplit(".", 1)[0] in classes_here]
        if not methods:
            return []
        classes = self._class_bases(repo_id, {m.qualified_name.rsplit(".", 1)[0] for m in methods},
                                    batch_paths, epoch)
        memo: dict[str, list[str] | None] = {}
        orders = {cls: _c3(cls, classes, memo) for cls in classes}
        lookups = {f"{owner}.{m.name}" for m in methods
                   for owner in (orders.get(m.qualified_name.rsplit(".", 1)[0]) or [])[1:]
                   if classes.get(owner) is not None}
        defined = self._exact_symbols(repo_id, lookups, batch_paths, epoch)
        out = []
        for method in methods:
            order = orders.get(method.qualified_name.rsplit(".", 1)[0])
            target, outcome = _walk_mro(order[1:] if order else None, classes, method.name, defined)
            if target is not None and outcome == "resolved":
                out.append(Edge(source_uid=method.uid, kind="OVERRIDES",
                                origin_path=method.rel_path, target_uid=target))
        return out

    def _through_attributes(
        self, repo_id: str, edges: list[Edge], report: WriteReport,
        batch_paths: list[str], epoch: int,
    ) -> tuple[list[Edge], list[tuple[str, str]]]:
        """`(edges, [(class, method)])` with `Owner.attr.method` lookups turned
        into `Type.method` lookups, or dropped as unresolved.

        The attribute is looked up along Owner's resolution order: the first
        class that declares it (`objects = TagManager()`) decides its type. An
        opaque class reached first — a library base such as `models.Model`,
        which supplies Django's default manager — means the type is unknown.
        """
        plain = [(e, *e.target_hint.rsplit(".", 1)) for e in edges if not e.via_attribute]
        hops = [e for e in edges if e.via_attribute]
        if hops:
            owners = {e.target_hint.rsplit(".", 2)[0] for e in hops}
            classes = self._class_bases(repo_id, owners, batch_paths, epoch)
            attributes = self._class_attributes(repo_id, set(classes), batch_paths, epoch)
            memo: dict[str, list[str] | None] = {}
            for edge in hops:
                owner, attr, method = edge.target_hint.rsplit(".", 2)
                kind = None
                for cls in _c3(owner, classes, memo) or []:
                    if classes.get(cls) is None:
                        break                        # opaque: the type is unknown
                    kind = attributes.get(cls, {}).get(attr)
                    if kind:
                        break
                if kind:
                    plain.append((edge, kind, method))
                else:
                    report.edges_unresolved += 1
                    metrics.incr("writer.member_attribute_unknown")
        return [e for e, _c, _m in plain], [(c, m) for _e, c, m in plain]

    def _class_attributes(
        self, repo_id: str, qualified_names: set[str], batch_paths: list[str], epoch: int
    ) -> dict[str, dict[str, str]]:
        """`class -> {attribute: type}` from the classes' recorded `attributes`."""
        if not qualified_names:
            return {}
        rows = self._run(
            """
            UNWIND $names AS n
            MATCH (s:Symbol {repo_id: $repo_id, name: n.name})
            WHERE s.qualified_name = n.qn AND s.kind = 'class'
              AND NOT (s.origin_path IN $batch_paths AND s.epoch < $epoch)
            RETURN n.qn AS qn, collect(coalesce(s.attributes, [])) AS attributes
            """,
            names=[{"qn": qn, "name": qn.rsplit(".", 1)[-1]} for qn in sorted(qualified_names)],
            repo_id=repo_id,
            batch_paths=batch_paths,
            epoch=epoch,
        )
        out: dict[str, dict[str, str]] = {}
        for row in rows:
            if len(row["attributes"]) == 1:          # two definitions: no answer
                out[row["qn"]] = dict(entry.split("=", 1) for entry in row["attributes"][0]
                                      if "=" in entry)
        return out

    def _class_bases(
        self, repo_id: str, start: set[str], batch_paths: list[str], epoch: int
    ) -> dict[str, list[str] | None]:
        """`qualified_name -> bases` for every class reachable from `start`.

        `None` marks a name that is not exactly one class in this repository —
        a library base, a name with two definitions, or a hierarchy deeper
        than `MRO_MAX_DEPTH`. The walk treats all three the same way: it stops.
        """
        classes: dict[str, list[str] | None] = {}
        frontier = set(start)
        for _ in range(MRO_MAX_DEPTH):
            if not frontier:
                break
            rows = self._run(
                """
                UNWIND $names AS n
                MATCH (s:Symbol {repo_id: $repo_id, name: n.name})
                WHERE s.qualified_name = n.qn AND s.kind = 'class'
                  AND NOT (s.origin_path IN $batch_paths AND s.epoch < $epoch)
                RETURN n.qn AS qn, collect(coalesce(s.bases, [])) AS bases
                """,
                names=[{"qn": qn, "name": qn.rsplit(".", 1)[-1]} for qn in sorted(frontier)],
                repo_id=repo_id,
                batch_paths=batch_paths,
                epoch=epoch,
            )
            found = {row["qn"]: row["bases"] for row in rows}
            nxt: set[str] = set()
            for qn in frontier:
                hits = found.get(qn, [])
                classes[qn] = list(hits[0]) if len(hits) == 1 else None
                if classes[qn]:
                    nxt.update(base for base in classes[qn] if base and base not in classes)
            frontier = nxt
        for qn in frontier:                       # deeper than the bound: opaque
            classes.setdefault(qn, None)
        return classes

    def _exact_symbols(
        self, repo_id: str, qualified_names: set[str], batch_paths: list[str], epoch: int
    ) -> dict[str, list[str]]:
        """`qualified_name -> [uid, …]` for the names that exist, orphans excluded."""
        if not qualified_names:
            return {}
        rows = self._run(
            """
            UNWIND $names AS n
            MATCH (s:Symbol {repo_id: $repo_id, name: n.name})
            WHERE s.qualified_name = n.qn
              AND NOT (s.origin_path IN $batch_paths AND s.epoch < $epoch)
            RETURN n.qn AS qn, collect(s.uid) AS uids
            """,
            names=[{"qn": qn, "name": qn.rsplit(".", 1)[-1]} for qn in sorted(qualified_names)],
            repo_id=repo_id,
            batch_paths=batch_paths,
            epoch=epoch,
        )
        return {row["qn"]: row["uids"] for row in rows}

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

    def _outside_callers_of_orphans(
        self, repo_id: str, paths: list[str], epoch: int
    ) -> list[str]:
        """Origin paths, outside the batch, of symbols with edges into an orphan."""
        if not paths:
            return []
        rows = self._run(
            """
            MATCH (s:Symbol)
            WHERE s.repo_id = $repo_id AND s.origin_path IN $batch_paths
              AND s.epoch < $epoch
            MATCH (c:Symbol {repo_id: $repo_id})-[]->(s)
            WHERE NOT c.origin_path IN $batch_paths
            RETURN DISTINCT c.origin_path AS path ORDER BY path
            """,
            repo_id=repo_id,
            batch_paths=paths,
            epoch=epoch,
        )
        return [row["path"] for row in rows if row["path"]]

    def _record_waiting(self, repo_id: str, paths: list[str], waiting: dict[str, set[str]]) -> None:
        """Replace each batch file's list of call names that resolved to nothing.

        Kept on the `:File` node because that is the unit that gets re-written:
        when the name appears, the whole file is re-parsed and its edges
        resolved again. Replaced on every write, so a call that was removed
        stops waiting.
        """
        rows = [{"path": path, "hints": sorted(waiting.get(path, ()))} for path in paths]
        for chunk in _chunked(rows, WRITE_CHUNK_SIZE):
            self._run(
                """
                UNWIND $rows AS row
                MATCH (f:File {repo_id: $repo_id, rel_path: row.path})
                SET f.waiting = row.hints
                """,
                rows=chunk,
                repo_id=repo_id,
            )

    def _files_waiting_for_new_names(self, repo_id: str, paths: list[str], epoch: int) -> list[str]:
        """Files outside the batch waiting for a name this batch just created.

        A waiting hint matches a new symbol exactly, or extends it by
        attribute access (`pkg.helpers.helper.cache` waits on `helper`) — the
        same prefixes edge resolution tries. A module only matches exactly:
        resolution never trims a call onto a module (F-015), so re-writing a
        file for that would change nothing.
        """
        if not paths:
            return []
        rows = self._run(
            """
            MATCH (s:Symbol)
            WHERE s.repo_id = $repo_id AND s.origin_path IN $batch_paths
              AND s.epoch <= $epoch AND s.created_epoch = $epoch
            WITH collect({qn: s.qualified_name, module: s.kind = 'module'}) AS created
            WHERE size(created) > 0
            MATCH (f:File {repo_id: $repo_id})
            WHERE NOT f.rel_path IN $batch_paths AND size(coalesce(f.waiting, [])) > 0
              AND any(c IN created WHERE any(w IN f.waiting
                    WHERE w = c.qn OR (NOT c.module AND w STARTS WITH c.qn + '.')))
            RETURN DISTINCT f.rel_path AS path ORDER BY path
            """,
            repo_id=repo_id,
            batch_paths=paths,
            epoch=epoch,
        )
        return [row["path"] for row in rows]

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
