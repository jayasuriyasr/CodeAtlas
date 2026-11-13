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
import logging
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

import metrics
from adapters.base import ParsedFile, content_hash
from adapters.python import PythonAdapter
from config import BULK_THRESHOLD, DEBOUNCE_SECONDS, FLUSH_MAX_RETRIES
from index.cache import (
    EmbeddingCache,
    EmbeddingDisabled,
    EmbeddingProvider,
    EmbeddingUnavailable,
    cached_vectors,
    embed_batch,
    embedder_key,
)
from index.chunker import Tokenizer, chunk_file
from index.moves import Move, Remap, collapse_chains, parse_git_renames, resolve_moves
from retrieve.tokenize import build_search_text

Vector = Sequence[float]

_log = logging.getLogger("sei.indexer")


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


#: The wall clock, read through one name so a test can step it back.
_wall_clock_ns = time.time_ns

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
    tree: dict[str, str] = {}
    for rel in list_indexable(root, extensions):
        try:
            tree[rel] = content_hash((root / rel).read_bytes())
        except OSError:
            metrics.incr("reconcile.unreadable")     # deleted mid-walk
    return tree


def list_indexable(root: Path, extensions: Iterable[str]) -> list[str]:
    """Every indexable path under `root`, POSIX and repo-relative.

    The one definition of "what belongs in the index", shared by the reconcile
    walk and the file watcher — two answers to that question would have the
    watcher indexing a file the next reconcile deletes.
    """
    root = Path(root)
    suffixes = tuple(extensions)
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
    return [rel for rel in candidates if rel.endswith(suffixes)]


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
    #: Consecutive failed flushes per repo, for `_requeue`'s backoff.
    _failures: dict[str, int] = field(default_factory=dict, init=False)
    #: Repos whose last embedding call fell back to cached vectors only.
    _degraded: set[str] = field(default_factory=set, init=False)
    #: The last epoch handed out per repo. Seeded from the graph on first use,
    #: so it survives a restart as well as a clock that steps back.
    _last_epoch: dict[str, int] = field(default_factory=dict, init=False)

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
        """Await every flush already scheduled. Shutdown and tests.

        Finished tasks are removed here, not left to their done-callback. On
        Python 3.12 `gather` over tasks that are already done completes
        eagerly, without yielding to the loop, so the callback that discards
        them never gets to run: `while self._tasks: await gather(...)` then
        spins forever. Observed as a hang whenever a flush finished in the
        instant before `aclose`.
        """
        while self._tasks:
            batch = list(self._tasks)
            await asyncio.gather(*batch, return_exceptions=True)
            self._tasks.difference_update(task for task in batch if task.done())

    async def settle(self) -> None:
        """Wait until nothing is armed and nothing is flushing.

        `join` waits for flushes that have *started*; a save inside its
        debounce window has only a timer. A caller that must act after the
        graph reflects every event so far — the watcher, before it reconciles
        deletions — needs both.
        """
        loop = asyncio.get_running_loop()
        while self._timers or self._tasks:
            if self._timers:
                due = min(handle.when() for handle in self._timers.values())
                await asyncio.sleep(max(0.0, due - loop.time()) + 0.01)
            await self.join()

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
            try:
                await self._write_batch(repo, batch, moves)
            except Exception:                       # noqa: BLE001 - re-queued, not lost
                self._requeue(repo, batch, moves)
                return
            self._failures.pop(repo, None)

    async def _write_batch(
        self, repo: str, batch: dict[str, ParsedFile], moves: list[Move]
    ) -> None:
        epoch = await self._next_epoch(repo)         # one epoch per batch

        # Moved-from paths stay in scope, or a remap collision leaves an
        # orphan node and un-repointed edges uncollected (v9.1 #16).
        batch_paths = set(batch) | {m.old_path for m in moves}

        if moves:                                   # before the bulk branch
            remaps = resolve_moves(
                repo, moves, batch, adapter_for=self._adapter_for
            )
            await asyncio.to_thread(self.writer.remap_uids, repo, remaps, epoch)
            self._indexed[repo] -= {m.old_path for m in moves}

        if len(batch) > self.bulk_threshold:        # 200†
            return await self.full_reconcile(repo, epoch=epoch)

        if not batch_paths:
            return                                  # nothing to do; do not touch the graph

        vectors = await self._embed(repo, batch)
        report = await asyncio.to_thread(
            self.writer.apply, repo, batch, batch_paths, vectors, epoch
        )
        self._indexed[repo] |= set(batch)
        await self._relink(repo, getattr(report, "relink_paths", None) or [], batch_paths)

    async def _relink(self, repo: str, paths: list[str], written: set[str]) -> None:
        """Re-write the files whose edges pointed at a symbol the batch replaced.

        A changed parameter count is a new UID (`symbol_uid`), and step 5's
        `DETACH DELETE` of the old node takes its incoming edges with it. The
        callers live in files nobody saved, so nothing else would re-resolve
        them until Sync Project. Their content is unchanged, so this costs a
        parse and cache hits — and their symbols keep their UIDs, so the
        follow-up orphans nothing and the re-link stops after one round.
        """
        todo = sorted(set(paths) - written)
        if not todo:
            return
        followup: dict[str, ParsedFile] = {}
        for rel in todo:
            try:
                parsed = self._parse(repo, rel, self._read(repo, rel))
            except OSError:
                metrics.incr("index.relink_unreadable")   # gone; the reconcile will see it
                continue
            if parsed is not None:
                followup[rel] = parsed
        if not followup:
            return
        metrics.incr("index.relinked_files", len(followup))
        try:
            vectors = await self._embed(repo, followup)
            await asyncio.to_thread(
                self.writer.apply, repo, followup, set(followup), vectors,
                await self._next_epoch(repo),
            )
        except Exception:
            # The batch itself is written; only the re-link failed. Re-writing
            # the batch would orphan nothing now, so it could never ask for this
            # again. Queue the callers' files as if they had been saved instead.
            for rel, parsed in followup.items():
                self._pending[repo].setdefault(rel, parsed)
            raise

    async def _next_epoch(self, repo: str) -> int:
        """A batch epoch strictly greater than every one before it.

        Steps 4 and 5 delete what is older than the batch (`epoch < $epoch`),
        so the order is all that matters, and the wall clock alone cannot give
        it: when the clock steps back (NTP, a resumed laptop, a restored VM),
        every new batch would be "older" than the data it replaces and nothing
        stale would be deleted. Nanoseconds are kept so epochs stay readable as
        times while the clock is right.
        """
        if repo not in self._last_epoch:
            read = getattr(self.reader, "max_epoch", None)
            last = await asyncio.to_thread(read, repo) if read is not None else 0
            self._last_epoch[repo] = int(last or 0)
        epoch = max(_wall_clock_ns(), self._last_epoch[repo] + 1)
        self._last_epoch[repo] = epoch
        return epoch

    def _requeue(self, repo: str, batch: dict[str, ParsedFile], moves: list[Move]) -> None:
        """§9.3: "Flush crashes mid-sequence — next flush converges."

        That row is true only if the next flush *contains* the failed batch. The
        batch was drained from `_pending` before the write, so a failure that is
        merely logged loses it: those files stay stale until someone saves them
        again, while the graph reports nothing wrong. So the batch goes back,
        under anything saved since (newer content wins), with its moves, and the
        flush is re-armed with backoff. Replay is safe because the sequence is
        (§4.3): remaps already applied match nothing the second time.

        Retries stop after `FLUSH_MAX_RETRIES`; the batch stays pending, so the
        next save re-arms it. Nothing is ever dropped.
        """
        _log.exception("flush failed for repo %r; %d file(s) re-queued", repo, len(batch))
        metrics.incr("index.flush_failed")
        pending = self._pending[repo]
        for path, parsed in batch.items():
            pending.setdefault(path, parsed)
        self._renames[repo] = list(moves) + self._renames[repo]

        attempt = self._failures.get(repo, 0) + 1
        self._failures[repo] = attempt
        if attempt > FLUSH_MAX_RETRIES:
            metrics.incr("index.flush_retries_exhausted")
            return
        try:
            self._reset_debounce(repo, delay=self.debounce * (2 ** attempt))
        except RuntimeError:
            pass                                    # no running loop: next save re-arms

    async def _embed(
        self, repo: str, batch: dict[str, ParsedFile], *, rewrite: bool = False
    ) -> dict[str, Vector]:
        """Chunk, then embed. §4.2's `embed_batch` reads `chunk_text`.

        Chunking happens here rather than at parse time because it is where
        §9.2's scrub runs, and a symbol that sat in `_pending` across several
        saves must be scrubbed exactly once against the current rule set.

        `search_text` (§5.3) is built here too. It is the fulltext arm's only
        input, and nothing else in the write path computes it — a symbol that
        reached the graph without it would be invisible to fulltext search while
        looking entirely healthy, which is T5's failure shape on the other arm.

        **An unreachable embedding API does not stop indexing** (§9.3's first
        two rows). The batch is written with whatever vectors the cache holds;
        the rest index without one and are found by fulltext and expansion until
        their next write — the next save of the file, or Sync Project — embeds
        them. A symbol that already had a vector keeps it (`symbol_props` never
        sends a null), which is §9.3's "stale", not "missing".

        **One embedder per repository.** If this repository's vectors were
        written by a different model (`embedder_key`), an incremental write
        adds none: a few vectors from a second model would sit in the first
        model's index and be compared as if they belonged. Only a `rewrite` —
        a reconcile that re-embeds every file — may change the embedder.
        """
        for parsed in batch.values():
            chunk_file(parsed, self.tokenizer)
            for sym in parsed.symbols:
                sym.search_text = build_search_text(sym)
        if not rewrite and not self._embedder_matches(repo):
            metrics.incr("index.embedder_mismatch")
            return {}
        try:
            return await asyncio.to_thread(
                embed_batch, repo, batch,
                cache=self.cache, provider=self.provider, reader=self.reader,
            )
        except EmbeddingUnavailable as exc:
            self._degraded.add(repo)
            if not isinstance(exc, EmbeddingDisabled):
                _log.warning(
                    "embedding provider unavailable; indexing %r with cached vectors only",
                    repo,
                )
                metrics.incr("index.embedding_unavailable")
            return await asyncio.to_thread(cached_vectors, batch, self.cache)

    def _embedder_matches(self, repo: str) -> bool:
        recorded = self.cache.get_meta(embedder_key(repo))
        return recorded is None or recorded == self.provider.name

    # -- reconcile (§4.5) --------------------------------------------------

    async def full_reconcile(
        self, repo: str, epoch: int | None = None, *, reparse_all: bool = False
    ) -> None:
        """§4.5. Referenced by the bulk branch, `post-checkout`, Sync and cold start.

        Resumable: each chunk is independently replayable under the same epoch
        (§4.3), so an interrupted reconcile resumes by re-running. Partial state
        is queryable throughout.

        **Earlier chunks are replayed once every chunk is written** (v10.2 §4.5,
        FINDINGS F-015). A chunk resolves its cross-file edges against the graph
        as it stands when that chunk is written, so a call from chunk 1 into a
        file chunk 3 adds — or into a symbol chunk 3 re-identifies — finds no
        target and is dropped. Only the last chunk saw a complete graph. Replay
        is §4.3's own guarantee put to work: same epoch, same statements, and
        now every target exists. It costs a second local parse of each replayed
        file and no embedding calls, since every key is already in the cache.

        `reparse_all` treats every file as modified. A cross-file edge is
        resolved only when its *source* file is written, so an unchanged caller
        keeps whatever its last write could resolve — a callee added after it,
        or one whose UID changed with its signature, stays unlinked however
        often the changed side is re-indexed. Sync Project is §4.4's recovery
        button, so it is the trigger that pays to re-resolve everything (F-016).
        """
        epoch = epoch or await self._next_epoch(repo)
        root = self.roots[repo]
        tree = await asyncio.to_thread(
            walk_working_tree, root, tuple(self.adapters)
        )
        known = await asyncio.to_thread(self.reader.file_hashes, repo)
        # Every file is about to be embedded afresh: the one moment the
        # repository's embedder may change, and the moment to record it.
        rewrite = reparse_all or not known
        self._degraded.discard(repo)

        added = [p for p in tree if p not in known]
        if reparse_all:
            modified = [p for p in tree if p in known]
        else:
            modified = [p for p in tree if p in known and tree[p] != known[p]]
        deleted = [p for p in known if p not in tree]

        # Undetected-move rate (§8.3) falls out of this diff: a symbol that left
        # a deleted path and appeared at an added path with the same body_hash
        # and unqualified `name` was a move no live signal caught. No stored
        # previous-reconcile state is required.
        if added and deleted:
            metrics.incr(
                "move.undetected",
                await self._count_body_hash_matches(repo, added, deleted),
            )

        chunks = list(_chunked(sorted(added + modified), self.bulk_threshold))
        for chunk in chunks:
            await self._reconcile_chunk(repo, chunk, epoch, rewrite)

        if deleted:                                  # steps 4-5 remove them
            await asyncio.to_thread(
                self.writer.apply, repo, {}, set(deleted), {}, epoch
            )

        if len(chunks) > 1:
            metrics.incr("reconcile.chunks_replayed", len(chunks) - 1)
            for chunk in chunks[:-1]:
                await self._reconcile_chunk(repo, chunk, epoch, rewrite)

        # One repo-wide pass beats N local ones after a bulk change.
        await asyncio.to_thread(self.writer.refresh_all_degrees, repo)
        self._indexed[repo] = set(tree)
        if rewrite and repo not in self._degraded:
            self.cache.set_meta(embedder_key(repo), self.provider.name)

    async def _reconcile_chunk(
        self, repo: str, chunk: list[str], epoch: int, rewrite: bool = False
    ) -> None:
        """Parse, embed and write one reconcile chunk under `epoch`."""
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
            return
        vectors = await self._embed(repo, parsed, rewrite=rewrite)
        await asyncio.to_thread(
            self.writer.apply, repo, parsed, set(parsed), vectors, epoch
        )

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

    async def reconcile(self, repo: str) -> None:
        """§4.5 on demand, outside a flush: `post-checkout`, and the watcher.

        Diff-only, under the repo's lock — the reconcile the bulk branch runs,
        reachable by a caller that is not already holding the lock.
        """
        async with self._locks[repo]:
            await self.full_reconcile(repo)

    async def sync_project(self, repo: str) -> None:
        """The Sync Project button.

        The recovery path for every row in §4.4's table that says "Undetected",
        and the one trigger that re-resolves every edge rather than only those
        of changed files (`full_reconcile(reparse_all=True)`, F-016). The
        `post-checkout` hook stays on the cheaper diff-only reconcile.
        """
        async with self._locks[repo]:
            await self.full_reconcile(repo, reparse_all=True)
