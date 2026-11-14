"""SEI's callers against a language server's, on real open-source repositories.

The golden set's two fixtures were written for this project. This measures
"what calls X" on code nobody wrote for SEI, with the answer taken from jedi —
a Python language server's reference search — rather than written by hand:

    python -m eval.benchmark                    # every function of every repo
    python -m eval.benchmark --repo django-taggit --sample 100 --seed 7

For each repository, pinned to a release commit: clone it once (into the tools
folder, `SEI_BENCH_DIR`, or `.sei/benchmarks`), index it with SEI, enumerate its
functions and methods with Python's `ast`, and for each one — every one, or a
seeded random sample when `--sample` is given — ask jedi for every place that
*calls* it, and the graph for the callers SEI has (CALLS edges into it). Scored
by qualified name, micro-averaged, two ways:

- precision over the functions jedi finds a caller for (the original measure);
- precision over every function examined, where a caller SEI reports for a
  function jedi finds *no* caller for counts as wrong.

Recall is over jedi's callers either way. The report goes to
`eval/BENCHMARK_REPORT.md` and says whether it covers every function or a sample.

What this does not cover: TypeScript. A TS gold set needs tsserver, which
means the npm `typescript` package and a project configuration per repository;
that is not built. jedi is a dev-only dependency (requirements-dev.txt): it
infers more than SEI does — `obj.method()` with a known `obj` type is a call to
that method — which is exactly the gap this is meant to measure.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import random
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import config

REPOS_FILE = config.REPO_ROOT / "eval" / "benchmark_repos.json"
REPORT_PATH = config.REPO_ROOT / "eval" / "BENCHMARK_REPORT.md"


@dataclass(frozen=True)
class Function:
    rel_path: str
    qualified_name: str
    line: int               # 1-based line of the `def`
    name_column: int        # 0-based column of the name, where jedi is asked


@dataclass
class Scored:
    function: Function
    gold: set[str]
    found: set[str]

    @property
    def hits(self) -> set[str]:
        return self.gold & self.found


@dataclass
class RepoResult:
    name: str
    commit: str
    files: int = 0
    functions: int = 0
    #: How many functions were looked at: all of them, or the sample.
    examined: int = 0
    sample: int | None = None
    seed: int | None = None
    #: Functions jedi finds a caller for.
    scored: list[Scored] = field(default_factory=list)
    #: Functions jedi finds no caller for, but SEI does.
    unconfirmed: list[Scored] = field(default_factory=list)

    @property
    def precision(self) -> float | None:
        found = sum(len(s.found) for s in self.scored)
        return sum(len(s.hits) for s in self.scored) / found if found else None

    @property
    def recall(self) -> float | None:
        gold = sum(len(s.gold) for s in self.scored)
        return sum(len(s.hits) for s in self.scored) / gold if gold else None

    @property
    def precision_all(self) -> float | None:
        """Precision with SEI's unconfirmed callers counted as wrong."""
        found = sum(len(s.found) for s in (*self.scored, *self.unconfirmed))
        return sum(len(s.hits) for s in self.scored) / found if found else None

    @property
    def scope(self) -> str:
        if self.sample is None:
            return f"every function ({self.functions})"
        return (f"a random sample of {self.examined} of {self.functions} functions, "
                f"seed {self.seed}")


# -- the functions, by Python's own parser -------------------------------------------


def module_name(rel_path: str) -> str:
    parts = rel_path[:-3].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def functions_in(root: Path, rel_path: str) -> list[Function]:
    """Every function and method in a file, named the way SEI names them."""
    source = (root / rel_path).read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines = source.splitlines()
    out: list[Function] = []

    def visit(node, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qualified = f"{scope}.{child.name}" if scope else child.name
                if not isinstance(child, ast.ClassDef):
                    text = lines[child.lineno - 1]
                    column = text.find(child.name, child.col_offset)
                    out.append(Function(rel_path, qualified, child.lineno, column))
                visit(child, qualified)
            elif isinstance(child, (ast.If, ast.Try, ast.With)):
                visit(child, scope)          # definitions under `if`, `try`, `with`

    visit(tree, module_name(rel_path))
    return out


# -- the gold, by jedi ------------------------------------------------------------------


def jedi_callers(project, root: Path, function: Function, indexed: set[str]) -> set[str]:
    """Qualified names of the functions that call `function`, per jedi.

    A reference counts when it is followed by `(` — a call, not an import or
    a mention — and lies in a file SEI also indexed, so both sides look at the
    same code. The caller is the reference's enclosing scope.
    """
    import jedi

    script = jedi.Script(path=str(root / function.rel_path), project=project)
    callers: set[str] = set()
    for ref in script.get_references(function.line, function.name_column, scope="project"):
        if ref.is_definition() or ref.module_path is None:
            continue
        try:
            rel = Path(ref.module_path).resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            continue                          # a reference outside the repository
        if rel not in indexed:
            continue
        text = Path(ref.module_path).read_text(encoding="utf-8", errors="replace").splitlines()
        rest = text[ref.line - 1][ref.column + len(ref.name):].lstrip()
        if not rest.startswith("("):
            continue
        scope = ref.parent()
        if scope is None or scope.type not in ("function", "class") or not scope.full_name:
            continue                          # module-level call: no symbol to attribute it to
        callers.add(scope.full_name)
    return callers


# -- SEI's answer -----------------------------------------------------------------------


def sei_callers(runner, repo_id: str, function: Function) -> set[str]:
    rows = runner.run(
        """
        MATCH (t:Symbol {repo_id: $repo, name: $name})
        WHERE t.qualified_name = $qn AND t.rel_path = $path
        MATCH (c:Symbol {repo_id: $repo})-[r]->(t)
        WHERE type(r) IN ['CALLS', 'INVOKES_HOOK']
        RETURN DISTINCT c.qualified_name AS qn
        """,
        repo=repo_id, name=function.qualified_name.rsplit(".", 1)[-1],
        qn=function.qualified_name, path=function.rel_path,
    )
    return {row["qn"] for row in rows}


def choose_functions(
    functions: list[Function], sample: int | None, seed: int
) -> list[Function]:
    """All of them, or a seeded random sample kept in file order.

    Not the first N: sorted file order puts whole packages (often `tests/`)
    first, so a prefix measures one corner of the repository.
    """
    if sample is None or sample >= len(functions):
        return list(functions)
    chosen = set(random.Random(seed).sample(range(len(functions)), sample))
    return [f for i, f in enumerate(functions) if i in chosen]


def measure_repo(
    root: Path, runtime, *, sample: int | None = None, seed: int = 0,
    limit: int | None = None, name: str = "", commit: str = "",
) -> RepoResult:
    """Index `root` with `runtime`, then score its callers against jedi's.

    `limit` is the older name for `sample`.
    """
    import jedi

    from index.indexer import list_indexable

    sample = sample if sample is not None else limit
    asyncio.run(runtime.indexer.sync_project(runtime.repo))
    indexed = set(list_indexable(root, (".py",)))
    result = RepoResult(name=name or root.name, commit=commit, files=len(indexed))
    project = jedi.Project(str(root))
    functions = [f for rel in sorted(indexed) for f in functions_in(root, rel)]
    chosen = choose_functions(functions, sample, seed)
    result.functions, result.examined = len(functions), len(chosen)
    if len(chosen) < len(functions):
        result.sample, result.seed = len(chosen), seed
    for function in chosen:
        gold = jedi_callers(project, root, function, indexed)
        found = sei_callers(runtime.service.graph_runner, runtime.repo, function)
        if gold:
            result.scored.append(Scored(function, gold, found))
        elif found:
            result.unconfirmed.append(Scored(function, gold, found))
    return result


# -- the report -------------------------------------------------------------------------


def _row(function: str, callers: list[str]) -> str:
    return f"- `{function}` ← " + ", ".join(f"`{c}`" for c in callers)


def render(results: list[RepoResult]) -> str:
    def fmt(value: float | None) -> str:
        return "—" if value is None else f"{value:.3f}"

    lines = [
        "<!-- Written by `python -m eval.benchmark`. -->",
        "",
        "# Callers: SEI against jedi",
        "",
        "Precision, where jedi has callers: over the functions jedi finds a caller",
        "for. Precision, all: over every function examined — a caller SEI reports",
        "for a function jedi finds none for counts as wrong.",
        "",
        "| repository | commit | files | examined | with jedi callers | precision, "
        "where jedi has callers | precision, all | recall |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r.name} | {r.commit[:10]} | {r.files} | {r.scope} | "
                     f"{len(r.scored)} | {fmt(r.precision)} | {fmt(r.precision_all)} | "
                     f"{fmt(r.recall)} |")
    for r in results:
        missed = [(s.function.qualified_name, sorted(s.gold - s.found)) for s in r.scored
                  if s.gold - s.found]
        extra = [(s.function.qualified_name, sorted(s.found - s.gold)) for s in r.scored
                 if s.found - s.gold]
        lines += ["", f"## {r.name}", "", f"Missed callers ({len(missed)} functions):", ""]
        lines += [_row(fn, callers) for fn, callers in missed[:25]]
        lines += ["", f"Callers SEI reports that jedi does not ({len(extra)} functions):", ""]
        lines += [_row(fn, callers) for fn, callers in extra[:25]]
        lines += ["", f"Callers SEI reports for functions jedi finds no caller for "
                      f"({len(r.unconfirmed)} functions):", ""]
        lines += [_row(s.function.qualified_name, sorted(s.found)) for s in r.unconfirmed[:25]]
    return "\n".join(lines) + "\n"


def _bench_dir() -> Path:
    import os

    if os.environ.get("SEI_BENCH_DIR"):
        return Path(os.environ["SEI_BENCH_DIR"])
    local = config.REPO_ROOT / ".local" / "config.json"
    if local.exists():
        tools = json.loads(local.read_text(encoding="utf-8")).get("tools_dir")
        if tools:
            return Path(tools) / "benchmarks"
    return config.REPO_ROOT / ".sei" / "benchmarks"


def checkout(spec: dict, base: Path) -> Path:
    """The pinned checkout, cloned on first use; refuses a HEAD that moved."""
    target = base / spec["name"]
    if not target.exists():
        base.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", "--depth", "1", "--branch", spec["tag"],
                        spec["url"], str(target)], check=True)
    head = subprocess.run(["git", "-C", str(target), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    if head != spec["commit"]:
        raise SystemExit(f"{target} is at {head}, not the pinned {spec['commit']}")
    return target


def main(argv: list[str] | None = None) -> int:
    from api.cli import _driver, build_runtime

    parser = argparse.ArgumentParser(prog="python -m eval.benchmark")
    parser.add_argument("--repo", help="only this repository (by name)")
    parser.add_argument("--sample", type=int, default=None,
                        help="a seeded random sample of this many functions (default: all)")
    parser.add_argument("--seed", type=int, default=0, help="the sample's seed")
    args = parser.parse_args(argv)
    try:
        import jedi  # noqa: F401
    except ImportError:
        print("jedi is not installed: pip install -r requirements-dev.txt", file=sys.stderr)
        return 1

    specs = json.loads(REPOS_FILE.read_text(encoding="utf-8"))["repos"]
    specs = [s for s in specs if not args.repo or s["name"] == args.repo]
    driver = _driver()
    results = []
    try:
        for spec in specs:
            root = checkout(spec, _bench_dir())
            repo_id = f"bench_{spec['name']}"
            with tempfile.TemporaryDirectory() as scratch:
                runtime = build_runtime(repo_id, root, ai="off", driver=driver,
                                        cache_path=Path(scratch) / "c.sqlite")
                try:
                    result = measure_repo(root, runtime, sample=args.sample, seed=args.seed,
                                          name=spec["name"], commit=spec["commit"])
                finally:
                    runtime.indexer.cache.close()
                    with driver.session(database=config.NEO4J_DEFAULT_DATABASE) as session:
                        session.run("MATCH (n {repo_id: $r}) DETACH DELETE n", r=repo_id)
            print(f"# {result.name}: {result.scope}; {len(result.scored)} with jedi callers; "
                  f"precision {result.precision}, all {result.precision_all}, "
                  f"recall {result.recall}", file=sys.stderr)
            results.append(result)
    finally:
        driver.close()
    REPORT_PATH.write_text(render(results), encoding="utf-8")
    print(f"wrote {REPORT_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":                         # pragma: no cover
    raise SystemExit(main())
