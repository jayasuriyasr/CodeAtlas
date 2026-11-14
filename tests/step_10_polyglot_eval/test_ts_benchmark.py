"""The benchmark on TypeScript, with the TypeScript compiler's references as gold.

Before: `eval/benchmark.py` measured Python only — "TypeScript has no
language-server gold". Now a repository marked `"language": "typescript"` is
scored against `eval/ts_gold.cjs`, which asks the TypeScript language service
(tsserver's engine) for the callers of every function. The `typescript`
package is a dev-only tool installed outside the repository; without it, and
without Node, these tests skip.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from api.cli import build_runtime
from eval.benchmark import measure_repo, render, ts_callers, typescript_dir

FILES = {
    "tsconfig.json": json.dumps({"compilerOptions": {"baseUrl": ".", "paths": {"@/*": ["./*"]},
                                                     "jsx": "preserve"}}),
    "lib/math.ts": (
        "export function add(a: number, b: number) {\n  return a + b\n}\n\n"
        "export const twice = (n: number) => add(n, n)\n\n"
        "export class Calc {\n  total(xs: number[]) {\n    return xs.reduce(add, 0)\n  }\n\n"
        "  sum(xs: number[]) {\n    return this.total(xs)\n  }\n}\n"
    ),
    "app/page.tsx": (
        'import { add, Calc } from "@/lib/math"\n\n'
        "export default function Page() {\n  const c = new Calc()\n  add(1, 2)\n"
        "  return <p>{c.sum([1])}</p>\n}\n"
    ),
}


@pytest.fixture
def ts_repo(tmp_path):
    if shutil.which("node") is None:
        pytest.skip("Node.js is not installed")
    if typescript_dir(install=False) is None:
        pytest.skip("the dev-only typescript package is not installed (eval/benchmark.py)")
    for rel, text in FILES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    return tmp_path


def test_tsserver_callers_are_named_the_way_sei_names_them(ts_repo):
    gold = {f.qualified_name: callers
            for f, callers in ts_callers(ts_repo, ["app/page.tsx", "lib/math.ts"]).values()}
    assert gold["lib.math.add"] == {"app.page.Page", "lib.math.twice"}   # not `reduce(add, 0)`
    assert gold["lib.math.Calc.total"] == {"lib.math.Calc.sum"}
    assert gold["lib.math.Calc.sum"] == {"app.page.Page"}
    assert gold["lib.math.twice"] == set()


def test_a_typescript_repository_is_scored_against_tsserver(ts_repo, graph_db, tmp_path_factory):
    graph_db.run("MATCH (n) DETACH DELETE n")
    runtime = build_runtime("repo_ts_bench", ts_repo, ai="off", driver=graph_db.driver,
                            cache_path=tmp_path_factory.mktemp("c") / "c.sqlite")
    try:
        result = measure_repo(ts_repo, runtime, name="calc", language="typescript")
    finally:
        runtime.indexer.cache.close()
        graph_db.run("MATCH (n) DETACH DELETE n")

    by_name = {s.function.qualified_name: s for s in result.scored}
    assert result.gold == "tsserver" and result.examined == result.functions == 5
    assert by_name["lib.math.add"].found == {"app.page.Page", "lib.math.twice"}
    assert by_name["lib.math.Calc.total"].found == {"lib.math.Calc.sum"}   # `this.total`
    assert "| calc |" in render([result]) and "tsserver" in render([result])
