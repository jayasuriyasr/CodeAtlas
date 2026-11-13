"""S6's read half. The other end of §9.1's single chokepoint.

Four queries, each named by the caller that needs it:

| Method | Called by | For |
|---|---|---|
| `known_paths` | `Indexer.startup` (§4.1) | seeding `_indexed`, which decides whether Signal B runs |
| `file_hashes` | `full_reconcile` (§4.5) | the added/modified/deleted diff |
| `body_hashes` | `embed_batch` (§4.2) | the §8.3 header-only miss share |
| `symbols_named` | frame resolution Tier 2 (§5.1) | name match, with ambiguity as refusal |

Every one leads with `repo_id`, so every one can be served by an index from
§11.3 rather than a label scan.
"""

from __future__ import annotations

from dataclasses import dataclass

from neo4j import Driver

from adapters.base import Symbol


def symbol_from_props(repo_id: str, props: dict) -> Symbol:
    """Rebuild a `Symbol` from the properties §11.2 3a stored.

    Every field the packer renders is restored — the context header needs
    `enclosing_signature` and `used_imports` (§3.4.2), which a neighbour row from
    §11.1 does not carry. Missing properties fall back to empty values rather
    than raising: a node written by an older build is still packable.
    """
    return Symbol(
        uid=props["uid"],
        repo_id=repo_id,
        rel_path=props.get("rel_path") or "",
        qualified_name=props.get("qualified_name") or "",
        name=props.get("name") or "",
        arity=props.get("arity") or 0,
        ordinal=props.get("ordinal") or 0,
        kind=props.get("kind") or "function",
        signature=props.get("signature") or "",
        docstring=props.get("docstring"),
        source_code=props.get("source_code") or "",
        enclosing_signature=props.get("enclosing_signature"),
        used_imports=list(props.get("used_imports") or []),
        decorators=list(props.get("decorators") or []),
        start_line=props.get("start_line") or 0,
        end_line=props.get("end_line") or 0,
        is_client_component=bool(props.get("is_client_component")),
        header_hash=props.get("header_hash"),
        body_hash=props.get("body_hash"),
        search_text=props.get("search_text"),
    )


@dataclass(frozen=True)
class NamedSymbol:
    """The subset of a symbol Tier 2 needs to decide, and no more."""

    uid: str
    qualified_name: str
    rel_path: str
    start_line: int
    end_line: int
    #: "function" | "class" | "module". Tier 2 needs it: a stack frame names a
    #: function, and `UserCard.tsx` yields a module *and* a component both
    #: called `UserCard` (FINDINGS F-027).
    kind: str = ""


class GraphReader:
    def __init__(self, driver: Driver, database: str) -> None:
        self._driver = driver
        self._database = database

    def _run(self, cypher: str, **params):
        with self._driver.session(database=self._database) as session:
            return list(session.run(cypher, params))

    def known_paths(self, repo_id: str) -> set[str]:
        """Every path the graph has seen (§4.1 `startup`).

        `collect_moves` runs the git fallback only when the batch holds a path
        this set does not contain, so an under-reporting version of this method
        silently disables Signal B.
        """
        return {
            row["rel_path"]
            for row in self._run(
                "MATCH (f:File {repo_id: $repo_id}) RETURN f.rel_path AS rel_path",
                repo_id=repo_id,
            )
        }

    def file_hashes(self, repo_id: str) -> dict[str, str]:
        """`rel_path -> content_hash` for §4.5's reconcile diff."""
        return {
            row["rel_path"]: row["content_hash"]
            for row in self._run(
                """
                MATCH (f:File {repo_id: $repo_id})
                RETURN f.rel_path AS rel_path, f.content_hash AS content_hash
                """,
                repo_id=repo_id,
            )
        }

    def body_hashes(self, repo_id: str, uids: list[str]) -> dict[str, str]:
        """Stored `body_hash` for the given uids, in one batched read (§4.2).

        Feeds the header-only miss share: a cache miss whose stored body_hash
        still matches was a header change, and §8.3 uses that ratio to decide
        whether warm reuse is worth building. An absent uid is a new symbol and
        counts as a body change, so missing keys are simply not returned.
        """
        if not uids:
            return {}
        return {
            row["uid"]: row["body_hash"]
            for row in self._run(
                """
                MATCH (s:Symbol {repo_id: $repo_id})
                WHERE s.uid IN $uids AND s.body_hash IS NOT NULL
                RETURN s.uid AS uid, s.body_hash AS body_hash
                """,
                repo_id=repo_id,
                uids=uids,
            )
        }

    def symbol_at_line(self, repo_id: str, rel_path: str, line: int) -> NamedSymbol | None:
        """§5.1 Tier 0. Served by the `symbol_lines` index.

        `ORDER BY start_line DESC` picks the *innermost* enclosing symbol: a
        traceback line inside `LoginView.post` is inside `LoginView` too, and
        the method is the answer. Taking the first row in any other order would
        return the class about half the time.
        """
        rows = self._run(
            """
            MATCH (s:Symbol {repo_id: $repo_id, rel_path: $rel_path})
            WHERE s.start_line <= $line AND $line <= s.end_line
            RETURN s.uid AS uid, s.qualified_name AS qualified_name,
                   s.rel_path AS rel_path, s.start_line AS start_line,
                   s.end_line AS end_line
            ORDER BY s.start_line DESC
            LIMIT 1
            """,
            repo_id=repo_id,
            rel_path=rel_path,
            line=line,
        )
        if not rows:
            return None
        row = rows[0]
        return NamedSymbol(
            uid=row["uid"],
            qualified_name=row["qualified_name"],
            rel_path=row["rel_path"],
            start_line=row["start_line"] or 0,
            end_line=row["end_line"] or 0,
        )

    def symbol_identities(
        self, repo_id: str, paths: list[str]
    ) -> dict[str, set[tuple[str, str]]]:
        """`rel_path -> {(name, body_hash)}` for §4.5's reconcile diff.

        Feeds the undetected-move count and nothing else. It is a measurement,
        never an action: §4.4 refuses content-similarity move detection outright
        because "a wrong merge is silent and corrupts the graph". This says how
        often that case occurred, so §10.2 can decide whether import-path repair
        is worth building.

        **The unqualified `name`, not `qualified_name`** — which is what §4.5
        specifies as of v10.1. v10.0 named the latter, but §3.2 derives
        qualified names from the module path, so a moved symbol's changes with
        the file: a pure move never matched, and this counter read zero forever,
        deferring §10.2's import-path repair on evidence never collected. Same
        root cause as R1, second site (FINDINGS F-012).

        Scoped by `origin_path` rather than `rel_path` so it reads the same
        column steps 4-6 scope on.
        """
        if not paths:
            return {}
        out: dict[str, set[tuple[str, str]]] = {}
        for row in self._run(
            """
            MATCH (s:Symbol)
            WHERE s.repo_id = $repo_id AND s.origin_path IN $paths
              AND s.body_hash IS NOT NULL
            RETURN s.origin_path AS path, s.name AS name,
                   s.body_hash AS body_hash
            """,
            repo_id=repo_id,
            paths=paths,
        ):
            out.setdefault(row["path"], set()).add((row["name"], row["body_hash"]))
        return out

    def symbols_named(self, repo_id: str, name: str) -> list[NamedSymbol]:
        """§5.1 Tier 2's candidate set. Served by the `symbol_name` index.

        Returns *all* candidates rather than one. The caller refuses when there
        is more than one — a repo of Next.js route files carries dozens of
        symbols named `GET`, and a `~ name match` badge on a 1-in-40 guess is
        worse than no answer.
        """
        return [
            NamedSymbol(
                uid=row["uid"],
                qualified_name=row["qualified_name"],
                rel_path=row["rel_path"],
                start_line=row["start_line"] or 0,
                end_line=row["end_line"] or 0,
                kind=row["kind"] or "",
            )
            for row in self._run(
                """
                MATCH (s:Symbol {repo_id: $repo_id, name: $name})
                RETURN s.uid AS uid, s.qualified_name AS qualified_name,
                       s.rel_path AS rel_path, s.start_line AS start_line,
                       s.end_line AS end_line, s.kind AS kind
                ORDER BY s.rel_path, s.start_line
                """,
                repo_id=repo_id,
                name=name,
            )
        ]

    def max_epoch(self, repo_id: str) -> int:
        """The newest batch epoch written for a repository, or 0.

        Every batch stamps its `:File` rows and symbols with its epoch, so this
        is where the indexer resumes counting after a restart.
        """
        rows = self._run(
            """
            CALL () {
                MATCH (f:File {repo_id: $repo_id}) RETURN max(f.epoch) AS e
                UNION ALL
                MATCH (s:Symbol {repo_id: $repo_id}) RETURN max(s.epoch) AS e
            }
            RETURN max(e) AS epoch
            """,
            repo_id=repo_id,
        )
        return int(rows[0]["epoch"] or 0) if rows else 0

    def stats(self, repo_id: str) -> dict[str, int]:
        """Files, symbols, relationships and embedded symbols for one repository.

        For `/status`: "is anything indexed, and does it have vectors" is the
        first question when an answer looks wrong. `embedded` below `symbols` is
        normal — it is how §9.3's degraded indexing shows.
        """
        row = self._run(
            """
            OPTIONAL MATCH (f:File {repo_id: $repo_id})
            WITH count(f) AS files
            OPTIONAL MATCH (s:Symbol {repo_id: $repo_id})
            WITH files, count(s) AS symbols, count(s.code_vec) AS embedded
            OPTIONAL MATCH (:Symbol {repo_id: $repo_id})-[r]->()
            RETURN files, symbols, embedded, count(r) AS relationships
            """,
            repo_id=repo_id,
        )[0]
        return {key: row[key] for key in ("files", "symbols", "embedded", "relationships")}

    # -- helpers the tests and the packer share ---------------------------

    def symbols(self, repo_id: str, uids: list[str]) -> dict[str, Symbol]:
        """`uid -> Symbol` for the packer, in one read.

        `code_vec` is projected out: 1,536 floats per symbol are what the
        vector index needs and nothing the packer reads. Missing uids are
        absent from the result rather than an error — a symbol deleted between
        retrieval and packing is simply not packed.
        """
        if not uids:
            return {}
        rows = self._run(
            """
            MATCH (s:Symbol {repo_id: $repo_id})
            WHERE s.uid IN $uids
            RETURN s {.*, code_vec: null} AS s
            """,
            repo_id=repo_id,
            uids=list(dict.fromkeys(uids)),
        )
        out = {}
        for row in rows:
            props = dict(row["s"])
            out[props["uid"]] = symbol_from_props(repo_id, props)
        return out

    def symbol(self, repo_id: str, uid: str) -> dict | None:
        rows = self._run(
            "MATCH (s:Symbol {uid: $uid, repo_id: $repo_id}) RETURN s AS s",
            uid=uid,
            repo_id=repo_id,
        )
        return dict(rows[0]["s"]) if rows else None

    def inbound_callers(self, repo_id: str, uid: str) -> list[str]:
        """Who calls this symbol. The question the graph exists to answer."""
        return [
            row["uid"]
            for row in self._run(
                """
                MATCH (caller:Symbol {repo_id: $repo_id})-[:CALLS]->
                      (target:Symbol {uid: $uid})
                RETURN DISTINCT caller.uid AS uid ORDER BY uid
                """,
                repo_id=repo_id,
                uid=uid,
            )
        ]
