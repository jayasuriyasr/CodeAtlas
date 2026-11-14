"""What the benchmark scores, and what it says it scored.

Two gaps the review found, reproduced here before the fix:

- Only functions jedi finds a caller for were scored, so a caller SEI reports
  for a function jedi finds *no* caller for was never counted against it. The
  precision kept its definition; a second one counts those callers as wrong.
- `--limit 60` stopped after the first 60 scored functions in sorted file
  order — the README called that "every function". Now every function is
  examined unless a seeded random sample is asked for, and the report says
  which it was.
"""

from __future__ import annotations

import pytest

import eval.benchmark as bench
from api.cli import build_runtime
from eval.benchmark import Function, RepoResult, Scored, choose_functions, measure_repo, render

from .test_benchmark import repo  # noqa: F401  (the fixture)

jedi = pytest.importorskip("jedi", reason="dev dependency: pip install -r requirements-dev.txt")


def fn(name: str, line: int = 1) -> Function:
    return Function("pkg/m.py", f"pkg.m.{name}", line, 4)


def test_callers_jedi_does_not_confirm_count_against_the_second_precision():
    result = RepoResult(name="r", commit="c")
    result.scored = [Scored(fn("a"), gold={"x", "y"}, found={"x", "y"})]
    result.unconfirmed = [Scored(fn("b"), gold=set(), found={"z"})]
    assert result.precision == 1.0                      # unchanged definition
    assert result.precision_all == pytest.approx(2 / 3)
    assert result.recall == 1.0


def test_every_function_is_examined_unless_a_sample_is_asked_for():
    functions = [fn(f"f{i}", i) for i in range(100)]
    assert choose_functions(functions, None, 0) == functions


def test_a_sample_is_seeded_and_not_the_first_files():
    functions = [fn(f"f{i}", i) for i in range(100)]
    first = choose_functions(functions, 10, 7)
    assert first == choose_functions(functions, 10, 7)
    assert len(first) == 10 and first != functions[:10]
    assert first == sorted(first, key=functions.index)  # still in file order


def test_the_report_says_what_was_scored():
    whole = RepoResult(name="r", commit="c", functions=3, examined=3)
    sampled = RepoResult(name="s", commit="c", functions=300, examined=60, sample=60, seed=7)
    text = render([whole, sampled])
    assert "every function (3)" in text
    assert "a random sample of 60 of 300 functions, seed 7" in text


def test_measure_repo_examines_all_and_keeps_unconfirmed_callers(
        repo, graph_db, tmp_path_factory, monkeypatch):  # noqa: F811
    real = bench.jedi_callers

    def without_cents(project, root, function, indexed):
        return set() if function.qualified_name == "shop.money.cents" else real(
            project, root, function, indexed)

    monkeypatch.setattr(bench, "jedi_callers", without_cents)
    graph_db.run("MATCH (n) DETACH DELETE n")
    runtime = build_runtime("repo_bench_scope", repo, ai="off", driver=graph_db.driver,
                            cache_path=tmp_path_factory.mktemp("c") / "c.sqlite")
    try:
        result = measure_repo(repo, runtime, name="shop")
    finally:
        runtime.indexer.cache.close()
        graph_db.run("MATCH (n) DETACH DELETE n")

    assert result.examined == result.functions == 4
    unconfirmed = {s.function.qualified_name: s.found for s in result.unconfirmed}
    assert unconfirmed == {"shop.money.cents": {"shop.money.Till.total", "shop.orders.place"}}
    assert result.precision == 1.0 and result.precision_all < 1.0
