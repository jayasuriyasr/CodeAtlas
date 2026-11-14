"""SEI on one large repository: index time, restart, save-to-searchable, memory, latency.

    python -m eval.scale --root <checkout> [--repo scale] [--ai local]

Runs the same objects `sei serve` runs, in four phases, and prints one JSON line
per phase. The warm restart is a second process, so nothing is still in memory.

1. **first index** — empty graph, empty embedding cache: `Runtime.bootstrap()`.
2. **warm restart** — a new process, graph and cache kept: `bootstrap()` again,
   which diffs the tree against the graph and finds nothing to do.
3. **save to searchable** — the file watcher running, a new function appended
   to a file, timed until the graph returns it by name. Five saves.
4. **query latency** — retrieval only (no answer model): 15 "what calls X"
   questions on functions drawn from the graph with a fixed seed, and 15
   semantic questions written for the repository's domain.

Memory is this process's peak working set. Neo4j runs in its own process and is
measured from outside. Nothing here is a test: it needs the app's Neo4j and, for
`--ai local`, Ollama with the embedding model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import config

SEMANTIC = [
    "how is a booking cancelled", "where are webhooks sent when a booking is created",
    "how are available time slots computed", "how is an API key verified",
    "where is the confirmation email sent", "how is a recurring event expanded",
    "how are calendar credentials refreshed", "where is a payment taken for a booking",
    "how does a team round-robin pick a host", "how is a booking rescheduled",
    "where are user time zones converted", "how is an event type's duration validated",
    "how are workflow reminders scheduled", "how does the app check for busy times",
    "where is a meeting link generated",
]


def peak_memory_mb() -> float:
    """This process's peak working set (Windows) or max RSS (elsewhere), in MiB."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE      # a 64-bit pseudo-handle
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters),
                                               wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(),
                                          ctypes.byref(counters), counters.cb):
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        return counters.PeakWorkingSetSize / 2**20
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def _runtime(args):
    from api.cli import build_runtime

    return build_runtime(args.repo, args.root, ai=args.ai, cache_path=args.cache)


def _counts(runtime) -> dict:
    rows = runtime.service.graph_runner.run(
        "MATCH (s:Symbol {repo_id: $r}) RETURN count(s) AS symbols, "
        "count(s.code_vec) AS embedded", r=runtime.repo)
    files = runtime.service.graph_runner.run(
        "MATCH (f:File {repo_id: $r}) RETURN count(f) AS files", r=runtime.repo)
    return {**rows[0], **files[0]}


def phase_first(args) -> dict:
    from api.cli import _driver

    driver = _driver()
    with driver.session(database=config.NEO4J_DEFAULT_DATABASE) as session:
        session.run("MATCH (n {repo_id: $r}) DETACH DELETE n", r=args.repo).consume()
    driver.close()
    if args.cache.exists():
        args.cache.unlink()
    runtime = _runtime(args)
    seconds = asyncio.run(_timed_bootstrap(runtime))
    out = {"phase": "first index", "seconds": round(seconds, 1), **_counts(runtime),
           "peak_mb": round(peak_memory_mb())}
    runtime.indexer.cache.close()
    return out


async def _timed_bootstrap(runtime) -> float:
    """Seconds until the index is current — what `sei serve` waits for.

    Timed inside the loop on purpose: `bootstrap()` also starts loading the
    answer model in a thread and does not wait for it (the server answers
    while it loads), but `asyncio.run` would, at exit. The first measurement
    made that mistake: 129 s for a warm restart, nearly all of it a cold
    model load right after Ollama had restarted.
    """
    started = time.perf_counter()
    await runtime.bootstrap()
    return time.perf_counter() - started


def phase_warm(args) -> dict:
    started = time.perf_counter()
    runtime = _runtime(args)
    built = time.perf_counter() - started

    async def run() -> dict:
        seconds = built + await _timed_bootstrap(runtime)
        out = {"phase": "warm restart", "seconds": round(seconds, 1),
               "of_which_startup": round(built, 1), **_counts(runtime),
               "peak_mb": round(peak_memory_mb())}
        out.update(await _save_and_query(args, runtime))
        return out

    out = asyncio.run(run())
    out["peak_mb_after_queries"] = round(peak_memory_mb())
    runtime.indexer.cache.close()
    return out


async def _save_and_query(args, runtime) -> dict:
    from index.watcher import TreeWatcher
    from retrieve.router import QueryRouter

    watcher = TreeWatcher(indexer=runtime.indexer, repo=runtime.repo, root=runtime.root,
                          extensions=tuple(runtime.indexer.adapters))
    task = asyncio.create_task(watcher.run(None))
    while watcher.state != "watching":
        await asyncio.sleep(0.05)
    polls = []
    for _ in range(3):
        started = time.perf_counter()
        await watcher.poll_once()
        polls.append(time.perf_counter() - started)

    target = args.root / args.probe_file
    original = target.read_bytes()
    saves = []
    try:
        for i in range(5):
            name = f"seiScaleProbe{i}{random.Random(i).randrange(10**6)}"
            body = original + f"\n\nexport function {name}() {{\n  return {i}\n}}\n".encode()
            target.write_bytes(body)
            written = time.perf_counter()
            while not runtime.reader.symbols_named(runtime.repo, name):
                if time.perf_counter() - written > 120:
                    raise RuntimeError(f"{name} never became searchable")
                await asyncio.sleep(0.05)
            saves.append(time.perf_counter() - written)
    finally:
        target.write_bytes(original)
        await asyncio.sleep(watcher.interval + 0.5)
        watcher.stop()
        await runtime.indexer.settle()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    names = runtime.service.graph_runner.run(
        "MATCH (s:Symbol {repo_id: $r}) WHERE s.kind IN ['function', 'method'] "
        "AND size(s.name) > 6 RETURN s.name AS name ORDER BY s.uid", r=runtime.repo)
    pool = [row["name"] for row in names]
    picked = random.Random(0).sample(pool, min(15, len(pool)))
    questions = [f"what calls `{name}`" for name in picked] + SEMANTIC
    router = QueryRouter()
    latencies: dict[str, list[float]] = {"structural": [], "semantic": []}
    for question in questions:
        routed = router.route(question)
        started = time.perf_counter()
        await asyncio.to_thread(runtime.retriever, question, routed)
        kind = "structural" if question.startswith("what calls") else "semantic"
        latencies[kind].append(time.perf_counter() - started)

    def pct(values: list[float], q: float) -> float:
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]

    every = latencies["structural"] + latencies["semantic"]
    return {
        "idle_poll_ms": round(statistics.median(polls) * 1000),
        "save_to_searchable_s": {"p50": round(statistics.median(saves), 2),
                                 "max": round(max(saves), 2)},
        "query_ms": {kind: {"p50": round(pct(v, 0.5) * 1000), "p95": round(pct(v, 0.95) * 1000)}
                     for kind, v in (*latencies.items(), ("all", every))},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.scale")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", default="scale")
    parser.add_argument("--ai", default="local")
    parser.add_argument("--cache", type=Path, default=None,
                        help="embedding cache for this run (default: next to the checkout)")
    parser.add_argument("--probe-file", default=None,
                        help="the file the save phase appends to (default: the first indexed)")
    parser.add_argument("--phase", choices=["all", "first", "warm"], default="all")
    args = parser.parse_args(argv)
    args.root = args.root.resolve()
    args.cache = args.cache or args.root.parent / f"{args.repo}-cache.sqlite"
    if args.probe_file is None:
        from index.indexer import list_indexable

        args.probe_file = sorted(list_indexable(args.root, (".ts", ".tsx", ".py")))[0]

    if args.phase == "first":
        print(json.dumps(phase_first(args)), flush=True)
    elif args.phase == "warm":
        print(json.dumps(phase_warm(args)), flush=True)
    else:
        for phase in ("first", "warm"):
            subprocess.run([sys.executable, "-m", "eval.scale", "--root", str(args.root),
                            "--repo", args.repo, "--ai", args.ai, "--cache", str(args.cache),
                            "--probe-file", args.probe_file, "--phase", phase],
                           check=True, env={**os.environ})
    return 0


if __name__ == "__main__":                         # pragma: no cover
    raise SystemExit(main())
