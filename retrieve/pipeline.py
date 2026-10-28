"""§7's retrieval half: the router's decision, acted on.

The router classifies; until this module nothing used the classification, so
every question — a pasted stack trace, "what calls `charge_card`" — took the
SEMANTIC path. This is the retriever `InvestigationService` is injected with,
and it follows §5.1's table:

| Class | Path |
|---|---|
| STRUCTURAL | entity resolve -> Cypher traversal *is* the answer |
| SEMANTIC   | hybrid RRF -> 1-hop expand |
| TRACE      | frame resolution -> UID lookup -> expand; Tier 3 on what did not resolve |

Every path degrades rather than fails: an entity that does not resolve, or a
relationship the graph has no edge for, falls back to SEMANTIC and says so in
the notes; an unreachable embedding API runs fulltext plus expansion and says
"degraded" (§9.3). The notes reach the reader as a `retrieval` event — per-frame
badges included, which §5.1 requires to be surfaced.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

import metrics
from config import RERANK_TOP_K
from index.cache import EmbeddingUnavailable
from pack.render import Block
from retrieve.expand import NeighborhoodExpander
from retrieve.frames import FrameResolver, build_tier3_query, is_library_path
from retrieve.hybrid import HybridRetriever
from retrieve.router import QueryClass, Routed
from retrieve.structural import StructuralResolver

Vector = Sequence[float]


class SymbolReader(Protocol):
    """The `GraphReader` methods retrieval needs."""

    def symbols(self, repo_id: str, uids: list[str]) -> dict: ...

    def known_paths(self, repo_id: str) -> set[str]: ...

    def symbol_at_line(self, repo_id: str, rel_path: str, line: int): ...

    def symbols_named(self, repo_id: str, name: str) -> list: ...


@dataclass
class QueryRetriever:
    """`(question, routed) -> (seeds, neighbors, notes)` for one repository."""

    repo_id: str
    reader: SymbolReader
    hybrid: HybridRetriever
    expander: NeighborhoodExpander
    structural: StructuralResolver
    #: Query embedding. None turns the vector arm off — `--offline`, or a
    #: stand-in provider whose vectors carry no meaning (F-009).
    embed_query: Callable[[str], Vector] | None = None
    repo_root: Path | str | None = None
    seed_count: int = RERANK_TOP_K

    def __call__(self, question: str, routed: Routed):
        notes: dict = {"route": routed.query_class.value}
        if routed.query_class is QueryClass.TRACE:
            seeds, neighbors = self._trace(question, routed, notes)
        elif routed.query_class is QueryClass.STRUCTURAL:
            seeds, neighbors = self._structural(question, routed, notes)
        else:
            seeds, neighbors = self._semantic(question, notes)
        notes["seeds"] = len(seeds)
        notes["neighbors"] = len(neighbors)
        return seeds, neighbors, notes

    def expand(self, uids: Sequence[str]) -> list[Block]:
        """§6.3's bounded expansion, as `InvestigationService.expander`."""
        return self._neighbors(list(uids))

    # -- SEMANTIC ------------------------------------------------------------

    def _semantic(
        self, question: str, notes: dict, *, lead: Sequence[str] = ()
    ) -> tuple[list[Block], list[Block]]:
        """Hybrid RRF -> top seeds -> 1-hop expand. `lead` uids rank first."""
        vec = self._query_vector(question, notes)
        candidates = self.hybrid.search(self.repo_id, question, query_vec=vec)
        top = max((c.score for c in candidates), default=1.0)
        ranked = [(uid, top + 1.0 + (len(lead) - i)) for i, uid in enumerate(lead)]
        ranked += [
            (c.uid, c.score)
            for c in candidates
            if c.uid not in lead
        ][: max(0, self.seed_count - len(lead))]
        seeds = self._blocks(ranked)
        return seeds, self._neighbors([b.uid for b in seeds])

    def _query_vector(self, text: str, notes: dict) -> Vector | None:
        if self.embed_query is None:
            notes.setdefault("vector_search", "off")
            return None
        try:
            vec = self.embed_query(text)
        except EmbeddingUnavailable:
            # §9.3: "Embedding API -> Fulltext + graph expansion", and the user
            # is told "Semantic search degraded".
            metrics.incr("retrieve.embedding_unavailable")
            notes["vector_search"] = "degraded"
            return None
        notes.setdefault("vector_search", "on")
        return vec

    # -- STRUCTURAL ----------------------------------------------------------

    def _structural(
        self, question: str, routed: Routed, notes: dict
    ) -> tuple[list[Block], list[Block]]:
        answer = self.structural.answer(self.repo_id, question, routed.rule)
        notes["structural"] = answer.note()
        if answer.fallback is not None:
            notes["route_taken"] = "SEMANTIC"
            return self._semantic(question, notes, lead=answer.entity_uids)

        notes["route_taken"] = "STRUCTURAL"
        # The entity and the first related symbols render in full — their bodies
        # show *how* they relate. The rest render as signatures, so a 50-caller
        # answer lists all 50 rather than the first few that fit at full size.
        full_slots = max(0, self.seed_count - len(answer.entity_uids))
        related = answer.related_uids
        n = len(related)
        ranked = [(uid, 3.0) for uid in answer.entity_uids]
        ranked += [(uid, 2.0 - i / (n + 1)) for i, uid in enumerate(related[:full_slots])]
        seeds = self._blocks(ranked)
        stubs = self._blocks(
            [(uid, 1.0 - i / (n + 1)) for i, uid in enumerate(related[full_slots:])]
        )
        return seeds, stubs

    # -- TRACE ---------------------------------------------------------------

    def _trace(
        self, question: str, routed: Routed, notes: dict
    ) -> tuple[list[Block], list[Block]]:
        resolver = FrameResolver(
            reader=self.reader,
            repo_root=self.repo_root,
            known_paths=self.reader.known_paths(self.repo_id),
        )
        resolved = resolver.resolve_all(self.repo_id, list(routed.frames))
        notes["frames"] = [
            {"frame": r.frame.raw, "tier": r.tier, "badge": r.badge, "uid": r.uid}
            for r in resolved
        ]

        # Innermost frame first: Python prints it last ("most recent call
        # last"), JavaScript first. The innermost frame is where it broke.
        python = [r for r in resolved if r.frame.language == "python"]
        others = [r for r in resolved if r.frame.language != "python"]
        frame_uids: list[str] = []
        for r in [*reversed(python), *others]:
            if r.uid is not None and r.uid not in frame_uids:
                frame_uids.append(r.uid)

        ranked = [(uid, 3.0 - i / (len(frame_uids) + 1)) for i, uid in enumerate(frame_uids)]

        # Tier 3 (§5.1): "re-route on the error message plus any resolvable
        # neighbouring frame" — when repository code was involved and did not
        # resolve, or nothing resolved at all. Library frames alone do not
        # trigger it; they were never going to resolve.
        unresolved_repo_frames = [
            r for r in resolved if r.uid is None and not is_library_path(r.frame.file_path)
        ]
        if unresolved_repo_frames or not frame_uids:
            tier3 = build_tier3_query(question, resolved)
            notes["tier3"] = tier3.text
            if tier3.text:
                vec = self._query_vector(tier3.text, notes)
                candidates = self.hybrid.search(self.repo_id, tier3.text, query_vec=vec)
                room = max(0, self.seed_count - len(ranked))
                ranked += [
                    (c.uid, c.score) for c in candidates if c.uid not in frame_uids
                ][:room]

        seeds = self._blocks(ranked)
        return seeds, self._neighbors([b.uid for b in seeds])

    # -- loading -------------------------------------------------------------

    def _blocks(self, ranked: Sequence[tuple[str, float]]) -> list[Block]:
        if not ranked:
            return []
        symbols = self.reader.symbols(self.repo_id, [uid for uid, _ in ranked])
        seen: set[str] = set()
        blocks = []
        for uid, score in ranked:
            if uid in symbols and uid not in seen:
                seen.add(uid)
                blocks.append(Block(symbol=symbols[uid], score=score))
        return blocks

    def _neighbors(self, seed_uids: list[str]) -> list[Block]:
        """1-hop neighbours of the seeds, excluding the seeds themselves.

        §11.1 excludes each seed from *its own* neighbourhood only, so with two
        seeds that call each other each comes back as the other's neighbour.
        The packer would skip the duplicate anyway; dropping it here keeps the
        neighbour count in the notes honest.
        """
        if not seed_uids:
            return []
        held = set(seed_uids)
        found = self.expander.expand(self.repo_id, seed_uids)
        return self._blocks([(n.uid, n.score) for n in found if n.uid not in held])
