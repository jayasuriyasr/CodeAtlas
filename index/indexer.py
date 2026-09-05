"""The indexing spine — §4.1's debounce contract and §4.5's full reconcile.

Three properties are load-bearing and easy to lose in a refactor:

1. **`collect_moves` runs before `batch` is drained**, because it mutates
   `_pending` (§4.1). Draining first leaves the moved-from entry in the batch,
   and the upsert recreates the retired UID.
2. **`batch_paths` is `batch.keys() | {move.old_path}`** (§4.3). The plan's §11
   names this the single most coupled interface in the system, and step 3's
   tests are what notice when it is wrong.
3. **Remaps apply before the bulk branch** (§4.1). After the rewrite the graph
   and filesystem agree, so reconcile finds less to do — and, more importantly,
   a bulk rename does not lose its moves to the reconcile path.

The writer and reader are synchronous (the Neo4j driver is), so every call into
them goes through a thread — S4's `JobRunner`, whose MVP §2 names as a
`ThreadPoolExecutor`. That is what `asyncio.to_thread` uses.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

import metrics
from adapters.base import ParsedFile, content_hash
from adapters.python import PythonAdapter
from config import BULK_THRESHOLD, DEBOUNCE_SECONDS
from index.cache import EmbeddingCache, EmbeddingProvider, embed_batch
from index.chunker import Tokenizer, chunk_file
from index.moves import Move, Remap, collapse_chains, parse_git_renames, resolve_moves
from retrieve.tokenize import build_search_text

Vector = Sequence[float]


# --------------------------------------------------------------------------
# The narrow slices of S6 this module needs
# --------------------------------------------------------------------------


class WriterLike(Protocol):
    def apply(self, repo_id: str, batch: dict[str, ParsedFile],
              batch_paths: Iterable[str], vectors: dict[str, Vector],
              epoch: int) -> Any: ...

    def remap_uids(self, repo_id: str, remaps: list[Remap], epoch: int) -> int: ...

    def refresh_all_degrees(self, repo_id: str) -> None: ...


class ReaderLike(Protocol):
    def known_paths(self, repo_id: str) -> set[str]: ...

    def file_hashes(self, repo_id: str) -> dict[str, str]: ...

    def body_hashes(self, repo_id: str, uids: list[str]) -> dict[str, str]: ...

    def symbol_identities(
        self, repo_id: str, paths: list[str]
    ) -> dict[str, set[tuple[str, str]]]: ...


@dataclass(frozen=True)
class RenameEvent:
    """The inbound editor event (§4.4 Signal A).

    §2: "Editor integration is not a seam: it is an inbound event boundary with
    one implementation per editor." This is that boundary, transport-agnostic —
    the HTTP binding `POST /index/rename` lands with FastAPI at step 9, and the
    VS Code extension that posts to it is a separate package (see PROGRESS.md).
    """

    repo: str
    old_path: str
    new_path: str


# --------------------------------------------------------------------------
# Working tree
# --------------------------------------------------------------------------


def _run_git(root: Path, *args: str) -> str:
    """Run git in `root`, returning stdout. Failure degrades to empty output.

    Signal B is a fallback (§4.4); a repository with no git, a git that is not
    installed, or a corrupt index must degrade to "no moves detected", which is
    an existing row in §9.3's table, not an exception.
    """
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        metrics.incr("git.unavailable")
        return ""
    if completed.returncode != 0:
        metrics.incr("git.failed")
        return ""
    return completed.stdout


def _is_git_repo(root: Path) -> bool:
    return bool(_run_git(root, "rev-parse", "--is-inside-work-tree").strip())


#: Directories never worth indexing even when git is unavailable to say so.
_DEFAULT_IGNORES = frozenset(
    {".git", ".venv", "venv", "__pycache__", "node_modules", ".pytest_cache",
     ".mypy_cache", ".sei", "dist", "build", ".next"}
)


def walk_working_tree(root: Path, extensions: Iterable[str]) -> dict[str, str]:
    """`rel_path -> content_hash` for every indexable file (§4.5).

    Respects `.gitignore` by asking git, which is the only way to get it right —
    `.gitignore` semantics include negation, directory scoping and nested files.
    Outside a git repository it falls back to a fixed ignore list, and that
    difference is deliberate: reimplementing gitignore would be a source of
    quiet disagreement between what is indexed and what a developer sees.
    """
    root = Path(root)
    suffixes = tuple(extensions)
    tree: dict[str, str] = {}

    if _is_git_repo(root):
        listing = _run_git(
            root, "ls-files", "--cached", "--others", "--exclude-standard", "-z"
        )
        candidates = [p for p in listing.split("\0") if p]
    else:
        candidates = [
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
            and not (_DEFAULT_IGNORES & set(path.relative_to(root).parts))
        ]

    for rel in candidates:
        if not rel.endswith(suffixes):
            continue
        try:
            tree[rel] = content_hash((root / rel).read_bytes())
        except OSError:
            metrics.incr("reconcile.unreadable")     # deleted mid-walk
    return tree


def _chunked(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


# --------------------------------------------------------------------------
# The indexer
# --------------------------------------------------------------------------


@dataclass
class Indexer:
    """§4.1's debounce contract.

    Per-repo mutex, not global: §4.1 says so explicitly, and
    `test_concurrent_repos_dont_block` is what keeps it that way. A global lock
    would make one large repository's flush stall every other tenant's saves,
    which is invisible until it is someone's outage.
    """

    writer: WriterLike
    reader: ReaderLike
    cache: EmbeddingCache
    provider: EmbeddingProvider
    tokenizer: Tokenizer
    roots: dict[str, Path]
    adapters: dict[str, Any] = field(default_factory=dict)
    debounce: float = DEBOUNCE_SECONDS
    bulk_threshold: int = BULK_THRESHOLD

    _pending: dict[str, dict[str, ParsedFile]] = field(
        default_factory=lambda: defaultdict(dict), init=False
    )
    _renames: dict[str, list[Move]] = field(
        default_factory=lambda: defaultdict(list), init=False
    )
    _indexed: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set), init=False
    )
    _timers: dict[str, asyncio.TimerHandle] = field(default_factory=dict, init=False)
    _locks: dict[str, asyncio.Lock] = field(
        default_factory=lambda: defaultdict(asyncio.Lock), init=False
    )
    _tasks: set[asyncio.Task] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        if not self.adapters:
            python = PythonAdapter()
            self.adapters = {ext: python for ext in python.extensions}

    # -- lifecycle ---------------------------------------------------------

    async def startup(self, repo: str) -> None:
        """§4.1. Seed `_indexed` from the graph.

        This is what gates the Signal B subprocess. An empty seed makes the
        gate true on every flush, so `git status` runs every time — the exact
        cost §4.1's comment says the gate exists to avoid. See FINDINGS F-004
        for why the `:File` node had to exist for this to work at all.
        """
        self._indexed[repo] = await asyncio.to_thread(self.reader.known_paths, repo)

    async def aclose(self) -> None:
        """Cancel outstanding timers and await in-flight flushes."""
        for handle in list(self._timers.values()):
            handle.cancel()
        self._timers.clear()
        await self.join()

    async def join(self) -> None:
        """Await every flush already scheduled. Shutdown and tests."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    # -- inbound events ----------------------------------------------------

    def on_save(self, repo: str, path: str, content: bytes) -> None:
        """§4.1. Parse eagerly (local, ~5ms), defer the network and the lock."""
        parsed = self._parse(repo, path, content)
        if parsed is None:
            return                                   # not a language we index
        self._pending[repo][parsed.rel_path] = parsed
        self._reset_debounce(repo)

    def on_rename(self, repo: str, old: str, new: str) -> None:
        """§4.1 Signal A. Recorded now, parsed at flush — a pure move fires no save."""
        self._renames[repo].append(Move(old_path=old, new_path=new))
        self._reset_debounce(repo)

    def on_rename_event(self, event: RenameEvent) -> None:
        self.on_rename(event.repo, event.old_path, event.new_path)

    def _reset_debounce(self, repo: str, delay: float | None = None) -> None:
        loop = asyncio.get_running_loop()
        handle = self._timers.pop(repo, None)
        if handle is not None:
            handle.cancel()
        self._timers[repo] = loop.call_later(
            self.debounce if delay is None else delay, self._fire, repo
        )

    def _fire(self, repo: str) -> None:
        self._timers.pop(repo, None)
        task = asyncio.create_task(self._flush(repo))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # -- parsing -----------------------------------------------------------

    def _adapter_for(self, rel_path: str):
        for ext, adapter in self.adapters.items():
            if rel_path.endswith(ext):
                return adapter
        return None

    def _parse(self, repo: str, rel_path: str, content: bytes) -> ParsedFile | None:
        adapter = self._adapter_for(rel_path)
        if adapter is None:
            return None
        return adapter.parse(repo, rel_path, content)

    def _read(self, repo: str, rel_path: str) -> bytes:
        return (self.roots[repo] / rel_path).read_bytes()

    # -- moves (§4.1) ------------------------------------------------------

    async def collect_moves(self, repo: str) -> list[Move]:
        """Single owner of move bookkeeping: drain, collapse, evict, parse.

        §4.1, followed in order. The eviction and the parse are the two halves
        that are easy to drop:

        * `pending.pop(mv.old_path)` — or the upsert recreates `old_uid`
          (v9.1 #14), producing a duplicate node the remap has already retired.
        * parsing `mv.new_path` when Signal B found the move — Signal B reads
          git, not the editor, so nothing has parsed the new path (v9.1 #17).
          Without this, `resolve_moves` finds no `ParsedFile` and increments
          `move.unresolvable`, which §9.4 alarms on.

        **Async, unlike §4.1's `def`** — see FINDINGS.md F-008. The git call is
        a blocking subprocess (measured at 67ms on a 400-file repository), and
        §4.1 runs it inside `_flush`, on the event loop thread. The per-repo
        mutex stops two flushes of *one* repo from overlapping; it does nothing
        about the loop itself, so every other repo's flush — and, from step 9,
        every HTTP request in the process — stalls for the duration. Only the
        subprocess moves to a thread; every mutation of `_pending` stays on the
        loop thread, so there is no race with `on_save`.
        """
        pending = self._pending[repo]
        moves, self._renames[repo] = self._renames[repo], []          # Signal A

        # Signal B costs a subprocess spawn (~50-200ms on a large repo), so run
        # it only when Signal A found nothing and the batch holds a path the
        # graph has never seen.
        if not moves and any(p not in self._indexed[repo] for p in pending):
            root = self.roots.get(repo)
            if root is not None:
                porcelain = await asyncio.to_thread(
                    _run_git, root, "status", "--porcelain", "-M"
                )
                moves = moves + parse_git_renames(porcelain)

        moves = collapse_chains(moves)

        resolved: list[Move] = []
        for mv in moves:
            pending.pop(mv.old_path, None)          # or the upsert recreates old_uid
            if mv.new_path not in pending:          # Signal B never parsed it
                try:
                    parsed = self._parse(repo, mv.new_path, self._read(repo, mv.new_path))
                except OSError:
                    metrics.incr("move.new_path_unreadable")
                    continue                        # degrades to undetected move
                if parsed is None:
                    continue                        # not an indexed language
                pending[mv.new_path] = parsed
            resolved.append(mv)
        return resolved

    # -- flush (§4.1) ------------------------------------------------------

    async def _flush(self, repo: str) -> None:
        async with self._locks[repo]:               # per-repo, not global
            moves = await self.collect_moves(repo)  # mutates _pending; call first
            batch, self._pending[repo] = self._pending[repo], {}
            epoch = time.time_ns()                  # one epoch per batch

            # Moved-from paths stay in scope, or a remap collision leaves an
            # orphan node and un-repointed edges uncollected (v9.1 #16).
            batch_paths = set(batch) | {m.old_path for m in moves}

            if moves:                               # before the bulk branch
                remaps = resolve_moves(
                    repo, moves, batch, adapter_for=self._adapter_for
                )
                await asyncio.to_thread(self.writer.remap_uids, repo, remaps, epoch)
                self._indexed[repo] -= {m.old_path for m in moves}

            if len(batch) > self.bulk_threshold:    # 200†
                return await self.full_reconcile(repo, epoch=epoch)

            if not batch_paths:
                return                              # nothing to do; do not touch the graph

            vectors = await self._embed(repo, batch)
            await asyncio.to_thread(
                self.writer.apply, repo, batch, batch_paths, vectors, epoch
            )
            self._indexed[repo] |= set(batch)

    async def _embed(self, repo: str, batch: dict[str, ParsedFile]) -> dict[str, Vector]:
        """Chunk, then embed. §4.2's `embed_batch` reads `chunk_text`.

        Chunking happens here rather than at parse time because it is where
        §9.2's scrub runs, and a symbol that sat in `_pending` across several
        saves must be scrubbed exactly once against the current rule set.

        `search_text` (§5.3) is built here too. It is the fulltext arm's only
        input, and nothing else in the write path computes it — a symbol that
        reached the graph without it would be invisible to fulltext search while
        looking entirely healthy, which is T5's failure shape on the other arm.
        """
        for parsed in batch.values():
            chunk_file(parsed, self.tokenizer)
            for sym in parsed.symbols:
                sym.search_text = build_search_text(sym)
        return await asyncio.to_thread(
            embed_batch, repo, batch,
            cache=self.cache, provider=self.provider, reader=self.reader,
        )

    # -- reconcile (§4.5) --------------------------------------------------

    async def full_reconcile(self, repo: str, epoch: int | None = None) -> None:
        """§4.5. Referenced by the bulk branch, `post-checkout`, Sync and cold start.

        Resumable: each chunk is independently replayable under the same epoch
        (§4.3), so an interrupted reconcile resumes by re-running. Partial state
        is queryable throughout.
        """
        epoch = epoch or time.time_ns()
        root = self.roots[repo]
        tree = await asyncio.to_thread(
            walk_working_tree, root, tuple(self.adapters)
        )
        known = await asyncio.to_thread(self.reader.file_hashes, repo)

        added = [p for p in tree if p not in known]
        modified = [p for p in tree if p in known and tree[p] != known[p]]
        deleted = [p for p in known if p not in tree]

        # Undetected-move rate (§8.3) falls out of this diff: a symbol that left
        # a deleted path and appeared at an added path with the same body_hash
        # and qualified_name was a move no live signal caught. No stored
        # previous-reconcile state is required.
        if added and deleted:
            metrics.incr(
                "move.undetected",
                await self._count_body_hash_matches(repo, added, deleted),
            )

        for chunk in _chunked(sorted(added + modified), self.bulk_threshold):
            parsed = {}
            for rel in chunk:
                try:
                    one = self._parse(repo, rel, self._read(repo, rel))
                except OSError:
                    metrics.incr("reconcile.unreadable")
                    continue
                if one is not None:
                    parsed[rel] = one
            if not parsed:
                continue
            vectors = await self._embed(repo, parsed)
            await asyncio.to_thread(
                self.writer.apply, repo, parsed, set(parsed), vectors, epoch
            )

        if deleted:                                  # steps 4-5 remove them
            await asyncio.to_thread(
                self.writer.apply, repo, {}, set(deleted), {}, epoch
            )

        # One repo-wide pass beats N local ones after a bulk change.
        await asyncio.to_thread(self.writer.refresh_all_degrees, repo)
        self._indexed[repo] = set(tree)

    async def _count_body_hash_matches(
        self, repo: str, added: list[str], deleted: list[str]
    ) -> int:
        """How many deleted paths reappeared, unchanged, at an added path.

        **Measurement only.** §4.4 forbids acting on content similarity — "a
        wrong merge is silent and corrupts the graph" — so this never produces a
        `Remap`. It produces one number, which §10.2 uses to decide whether an
        import-path repair pass is worth building.

        Counted in files, not symbols, because §8.3's denominator is total
        moves. Matching is one-to-one and deterministic (sorted order), so a
        file split into three does not count as three moves.
        """
        before = await asyncio.to_thread(
            self.reader.symbol_identities, repo, sorted(deleted)
        )
        if not before:
            return 0

        after: dict[str, set[tuple[str, str]]] = {}
        for rel in sorted(added):
            try:
                parsed = self._parse(repo, rel, self._read(repo, rel))
            except OSError:
                continue
            if parsed is None:
                continue
            chunk_file(parsed, self.tokenizer)
            # `name`, not `qualified_name` (§4.5 as of v10.1). A module
            # symbol's name changes with the filename too, so it never matches;
            # the functions inside it do, and one match pairs the files.
            # See GraphReader.symbol_identities.
            after[rel] = {
                (s.name, s.body_hash) for s in parsed.symbols if s.body_hash
            }

        matched_new: set[str] = set()
        count = 0
        for old_path in sorted(before):
            identities = before[old_path]
            if not identities:
                continue
            for new_path in sorted(after):
                if new_path in matched_new:
                    continue
                if identities & after[new_path]:
                    matched_new.add(new_path)
                    count += 1
                    break
        return count

    # -- manual triggers (§4.1) -------------------------------------------

    async def sync_project(self, repo: str) -> None:
        """The Sync Project button, and the `post-checkout` hook.

        The recovery path for every row in §4.4's table that says "Undetected".
        """
        await self.full_reconcile(repo)
