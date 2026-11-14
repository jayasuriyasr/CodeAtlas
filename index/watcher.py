"""Keeps a repository's graph current without an editor plugin.

§4.1's triggers begin with "save (eager, local)", and §4.4's Signal A is an
editor's rename event. Both assume an editor integration, which the cut ladder
dropped. §2 calls that integration "an inbound event boundary with one
implementation per editor"; this is the implementation for no editor at all. It
polls the working tree and turns what it sees into events the indexer already
takes — it adds no second way to write the graph:

* **A new or modified file** becomes `Indexer.on_save`: §4.1's eager path, so
  the debounce, the per-repo lock and Signal B (git) apply unchanged.
* **A file that vanished** becomes `Indexer.reconcile`, once every pending
  flush has landed. §4.5 already knows how to remove a path's symbols. The
  wait is the point: a `git mv` appears here as one path vanishing and another
  appearing, and the flush's Signal B must rewrite the UIDs in place *before* a
  reconcile sees the old path as deleted — the other order severs every
  inbound edge the move mechanism exists to keep (§4.4).

What it deliberately does not do is pair a vanished file with a new one by
content. §4.4 forbids similarity heuristics: a missed move is recoverable by
one button, a wrong merge is silent. An unstaged `mv` is therefore still the
"Undetected" row of §4.4's table — delete and create — exactly as with an
editor that has no plugin.

Polling, not OS file notifications: it needs no dependency, behaves the same on
every platform and network drive, and a scan reads no file that did not change.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Protocol

import metrics
from adapters.base import content_hash
from config import WATCH_INTERVAL_SECONDS
from index.indexer import _DEFAULT_IGNORES, list_indexable

_log = logging.getLogger("sei.watcher")

#: `(st_mtime_ns, st_size)` — enough to notice a save without reading the file.
Stat = tuple[int, int]

#: Seconds before a failed start-up index is tried again (Neo4j still booting).
_BOOTSTRAP_RETRY_SECONDS = 5.0


class WatchedIndexer(Protocol):
    """The slice of `Indexer` the watcher drives."""

    def on_save(self, repo: str, path: str, content: bytes) -> None: ...

    async def settle(self) -> None: ...

    async def reconcile(self, repo: str) -> None: ...


@dataclass
class Delta:
    """What one poll found. Returned for tests and for the status page."""

    saved: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.saved or self.removed)


@dataclass
class TreeWatcher:
    indexer: WatchedIndexer
    repo: str
    root: Path
    extensions: tuple[str, ...]
    interval: float = WATCH_INTERVAL_SECONDS

    #: starting | indexing | watching | error | stopped — for `/status`.
    state: str = "starting"
    last_error: str | None = None
    last_change_at: float | None = None
    files_watched: int = 0

    _stats: dict[str, Stat] = field(default_factory=dict, init=False)
    _allowed: set[str] = field(default_factory=set, init=False)
    _hashes: dict[str, str] = field(default_factory=dict, init=False)
    _stopping: bool = field(default=False, init=False)

    # -- scanning ----------------------------------------------------------

    def _stat_tree(self) -> dict[str, Stat]:
        """`rel_path -> (mtime, size)` for every file with an indexed suffix.

        Prunes the directories that are never source (`.git`, `node_modules`,
        virtualenvs), which are never listed. Whether a file
        is *indexable* — `.gitignore` and all — is `list_indexable`'s answer,
        asked only when this cheap pass saw something change.

        One `scandir` pass. On Windows the directory listing already carries
        each file's mtime and size, so `DirEntry.stat()` costs nothing; a
        separate `os.stat` per file — what `os.walk` plus `os.stat` did — was
        two thirds of a poll on a 7,363-file repository (2.8 s of 1.0 s
        listing plus 1.8 s of stats), with the poll interval at 1 s.
        """
        root = str(self.root)
        prefix = len(os.path.join(root, ""))
        out: dict[str, Stat] = {}
        pending = [root]
        while pending:
            try:
                listing = os.scandir(pending.pop())
            except OSError:
                continue                        # removed or unreadable mid-walk
            with listing:
                for entry in listing:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if entry.name not in _DEFAULT_IGNORES:
                                pending.append(entry.path)
                            continue
                        if not entry.name.endswith(self.extensions):
                            continue
                        info = entry.stat()
                    except OSError:
                        continue                # deleted between listing and stat
                    rel = entry.path[prefix:].replace(os.sep, "/")
                    out[rel] = (info.st_mtime_ns, info.st_size)
        return out

    def _indexable(self, stats: dict[str, Stat]) -> set[str]:
        # git lists a tracked file until its deletion is staged; only files
        # that are actually on disk count.
        return {p for p in list_indexable(self.root, self.extensions) if p in stats}

    def prime(self) -> None:
        """Take the first snapshot without reporting it as changes.

        Whatever differs between the tree and the graph at start-up is the
        start-up reconcile's job; the watcher reports what changes *after*.
        """
        self._stats = self._stat_tree()
        self._allowed = self._indexable(self._stats)
        self.files_watched = len(self._allowed)

    # -- one poll ----------------------------------------------------------

    async def poll_once(self) -> Delta:
        stats = await asyncio.to_thread(self._stat_tree)
        if stats == self._stats:
            return Delta()

        changed = {p for p, s in stats.items() if self._stats.get(p) != s}
        allowed = await asyncio.to_thread(self._indexable, stats)
        # A file can become indexable without changing: `.gitignore` was edited.
        candidates = (changed & allowed) | (allowed - self._allowed)
        removed = sorted(self._allowed - allowed)

        # The snapshot advances only when everything it implies was delivered.
        # If a save or the reconcile fails, the next poll must see the same
        # difference again — otherwise one transient error loses the change for
        # good, since the file's mtime will never move a second time. Files
        # already delivered are not re-sent: their hashes are remembered.
        delta = Delta(removed=removed)
        for rel in sorted(candidates):
            try:
                content = await asyncio.to_thread((self.root / rel).read_bytes)
            except OSError:
                continue                        # gone again; the next poll says so
            digest = content_hash(content)
            if self._hashes.get(rel) == digest:
                continue                        # touched, not changed
            self.indexer.on_save(self.repo, rel, content)
            self._hashes[rel] = digest
            delta.saved.append(rel)

        if removed:
            # Flushes first (Signal B may turn this "deletion" into a remap),
            # then the reconcile that removes whatever is really gone.
            await self.indexer.settle()
            await self.indexer.reconcile(self.repo)
        for rel in removed:
            self._hashes.pop(rel, None)

        self._stats = stats
        self._allowed = allowed
        self.files_watched = len(allowed)

        if delta:
            self.last_change_at = time.time()
            metrics.incr("watch.files_saved", len(delta.saved))
            metrics.incr("watch.files_removed", len(delta.removed))
        return delta

    # -- the loop ----------------------------------------------------------

    async def run(self, bootstrap: Callable[[], Awaitable[None]] | None = None) -> None:
        """Index what start-up found, then watch until cancelled.

        `bootstrap` brings the graph level with the tree before watching
        starts. It is retried rather than fatal: the usual reason it fails is a
        database that is still starting, and a watcher that gave up then would
        leave the server answering from an index nothing is maintaining.
        """
        self._stopping = False
        try:
            await asyncio.to_thread(self.prime)
            while bootstrap is not None and not self._stopping:
                self.state = "indexing"
                try:
                    await bootstrap()
                    self.last_error = None
                    break
                except Exception as exc:        # noqa: BLE001 - retried, reported
                    self._report(exc, "start-up index failed; retrying")
                    await asyncio.sleep(_BOOTSTRAP_RETRY_SECONDS)

            self.state = "watching"
            while not self._stopping:
                await asyncio.sleep(self.interval)
                try:
                    await self.poll_once()
                    if self.state == "error":
                        self.state, self.last_error = "watching", None
                except Exception as exc:        # noqa: BLE001 - one bad poll is not the end
                    self._report(exc, "poll failed")
        finally:
            self.state = "stopped"

    def _report(self, exc: Exception, what: str) -> None:
        self.state = "error"
        self.last_error = f"{type(exc).__name__}: {exc}"
        metrics.incr("watch.errors")
        _log.warning("%s for %r: %s", what, self.repo, self.last_error)

    def stop(self) -> None:
        self._stopping = True
