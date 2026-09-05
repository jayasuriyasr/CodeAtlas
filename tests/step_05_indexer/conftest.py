"""Scaffolding for the step 5 indexer tests.

Two kinds of double are used here and they are **not** interchangeable:

* `RecordingWriter` / `StubReader` — record what the indexer *decided*: which
  paths were in the batch, what `batch_paths` was, which remaps were issued.
  They answer questions about §4.1's control flow and nothing else.
* the real `GraphWriter` / `GraphReader` against a throwaway database — the
  only thing that can answer "did the inbound edges survive", which is the
  actual claim in §8.2's move-survival fixture.

A test that asserts on caller counts must use the second. Asserting a caller
count against a recording double would be measuring the double.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import pytest

from adapters.base import ParsedFile
from index.cache import EmbeddingCache
from index.indexer import Indexer
from index.moves import Remap
from index.providers import ApproxCodeTokenizer, HashEmbeddingProvider

REPO = "repo_step5"
OTHER_REPO = "repo_step5_other"

Vector = Sequence[float]


# --------------------------------------------------------------------------
# Doubles
# --------------------------------------------------------------------------


@dataclass
class AppliedBatch:
    repo_id: str
    paths: set[str]
    batch_paths: set[str]
    epoch: int
    vector_uids: set[str]


@dataclass
class RecordingWriter:
    """Records §11.2's inputs. Makes no claim about what the graph would do."""

    applied: list[AppliedBatch] = field(default_factory=list)
    remaps: list[list[Remap]] = field(default_factory=list)
    degree_refreshes: list[str] = field(default_factory=list)
    #: Set to block inside `apply`, for the per-repo-lock test. Scoped to one
    #: repo — a shared gate would block both and prove nothing about the lock.
    gate: Any = None
    gate_repo: str | None = None
    #: Raise on the Nth apply call (1-based), for the resumability test.
    fail_on_apply: int | None = None
    #: Set once `apply` runs for `signal_repo`. Lets the concurrency test wait
    #: on an event instead of on a sleep, which is what made it flaky.
    signal: Any = None
    signal_repo: str | None = None

    def apply(self, repo_id: str, batch: dict[str, ParsedFile],
              batch_paths: Iterable[str], vectors: dict[str, Vector],
              epoch: int) -> None:
        if self.gate is not None and repo_id == self.gate_repo:
            self.gate.wait(timeout=10)
        self.applied.append(
            AppliedBatch(
                repo_id=repo_id,
                paths=set(batch),
                batch_paths=set(batch_paths),
                epoch=epoch,
                vector_uids=set(vectors),
            )
        )
        if self.signal is not None and repo_id == self.signal_repo:
            self.signal.set()
        if self.fail_on_apply is not None and len(self.applied) == self.fail_on_apply:
            raise RuntimeError("injected writer failure")

    def remap_uids(self, repo_id: str, remaps: list[Remap], epoch: int) -> int:
        self.remaps.append(list(remaps))
        return len(remaps)

    def refresh_all_degrees(self, repo_id: str) -> None:
        self.degree_refreshes.append(repo_id)

    # -- convenience for assertions ---------------------------------------

    @property
    def epochs(self) -> list[int]:
        return [a.epoch for a in self.applied]

    @property
    def last(self) -> AppliedBatch:
        assert self.applied, "no batch was applied"
        return self.applied[-1]

    @property
    def all_remaps(self) -> list[Remap]:
        return [r for batch in self.remaps for r in batch]


@dataclass
class StubReader:
    """A seedable stand-in for `GraphReader`."""

    paths: dict[str, dict[str, str]] = field(default_factory=dict)
    identities: dict[str, dict[str, set[tuple[str, str]]]] = field(default_factory=dict)
    body: dict[str, dict[str, str]] = field(default_factory=dict)

    def known_paths(self, repo_id: str) -> set[str]:
        return set(self.paths.get(repo_id, {}))

    def file_hashes(self, repo_id: str) -> dict[str, str]:
        return dict(self.paths.get(repo_id, {}))

    def body_hashes(self, repo_id: str, uids: list[str]) -> dict[str, str]:
        stored = self.body.get(repo_id, {})
        return {uid: stored[uid] for uid in uids if uid in stored}

    def symbol_identities(
        self, repo_id: str, paths: list[str]
    ) -> dict[str, set[tuple[str, str]]]:
        stored = self.identities.get(repo_id, {})
        return {p: stored[p] for p in paths if p in stored}


# --------------------------------------------------------------------------
# A real git repository on disk
# --------------------------------------------------------------------------


@dataclass
class GitRepo:
    root: Path

    def git(self, *args: str) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=str(self.root), capture_output=True, text=True,
            check=True,
        )
        return completed.stdout

    def write(self, rel: str, text: str) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")

    def commit(self, message: str = "wip") -> None:
        self.git("add", "-A")
        self.git("commit", "-qm", message)

    def mv(self, old: str, new: str) -> None:
        (self.root / new).parent.mkdir(parents=True, exist_ok=True)
        self.git("mv", old, new)

    def status(self) -> str:
        return self.git("status", "--porcelain", "-M")


@pytest.fixture
def git_repo(tmp_path) -> GitRepo:
    root = tmp_path / "work"
    root.mkdir()
    repo = GitRepo(root)
    repo.git("init", "-q", ".")
    # autocrlf off: a CRLF rewrite on checkout would change content_hash and
    # make every reconcile diff report the whole tree as modified.
    repo.git("config", "core.autocrlf", "false")
    repo.git("config", "user.email", "test@example.invalid")
    repo.git("config", "user.name", "test")
    return repo


# --------------------------------------------------------------------------
# Source fixtures — one callee, five callers (§8.2's move-survival shape)
# --------------------------------------------------------------------------

CALLEE = '''\
"""The moved module."""


def helper(value):
    """Called from five places."""
    return value * 2
'''

CALLERS = '''\
"""Five callers, in a file that never moves."""
from pkg.callee import helper


def caller0(x):
    return helper(x)


def caller1(x):
    return helper(x) + 1


def caller2(x):
    return helper(x) + 2


def caller3(x):
    return helper(x) + 3


def caller4(x):
    return helper(x) + 4
'''


def seed_move_fixture(repo: GitRepo) -> None:
    """§8.2 fixture (a): a file with >=5 known inbound callers."""
    repo.write("pkg/__init__.py", "")
    repo.write("pkg/callee.py", CALLEE)
    repo.write("pkg/callers.py", CALLERS)
    repo.commit("seed")


# --------------------------------------------------------------------------
# Indexer wiring
# --------------------------------------------------------------------------


@pytest.fixture
def tokenizer() -> ApproxCodeTokenizer:
    return ApproxCodeTokenizer()


@pytest.fixture
def provider() -> HashEmbeddingProvider:
    return HashEmbeddingProvider()


@pytest.fixture
def cache(tmp_path, provider) -> EmbeddingCache:
    cache = EmbeddingCache(tmp_path / "embeddings.sqlite", model=provider.name)
    yield cache
    cache.close()


@pytest.fixture
def recording_writer() -> RecordingWriter:
    return RecordingWriter()


@pytest.fixture
def stub_reader() -> StubReader:
    return StubReader()


@pytest.fixture
def make_indexer(cache, provider, tokenizer):
    """Build an Indexer over whichever writer/reader a test wants.

    The debounce defaults to 50ms rather than §4.1's 2.0s: the behaviour under
    test is coalescing, and `test_debounce_matches_the_spec_constant` is what
    holds the constant itself to the spec.
    """

    def build(writer, reader, roots: dict[str, Path], **kw) -> Indexer:
        kw.setdefault("debounce", 0.05)
        return Indexer(
            writer=writer, reader=reader, cache=cache, provider=provider,
            tokenizer=tokenizer, roots=roots, **kw,
        )

    return build


# --------------------------------------------------------------------------
# The step 5 gate: `move.unresolvable` is zero across the entire suite
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def unresolvable_moves_stay_at_zero(request):
    """§9.4 alarms on this counter — "must be identically zero".

    A gate phrased as "across the entire suite" is not something one assertion
    in one test can establish, so it is checked after every test in this
    directory. The single test that exercises the counter deliberately opts out
    with `@pytest.mark.expects_unresolvable`; without that escape the guard
    would either be wrong or would force that test to be deleted, and the
    counter would then be unreachable code behind a live alarm.
    """
    import metrics

    yield

    if request.node.get_closest_marker("expects_unresolvable"):
        return
    unresolvable = metrics.get("move.unresolvable")
    assert unresolvable == 0, (
        f"move.unresolvable reached {unresolvable} in {request.node.name}; "
        f"§9.4 requires it to be identically zero"
    )
