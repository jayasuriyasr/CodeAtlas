"""The open-source benchmark's machinery, on a small repository built here.

`python -m eval.benchmark` runs on pinned real repositories and needs the
network once. What a test can hold without it: functions are enumerated the way
SEI names them, jedi's callers are found and attributed correctly, and the two
are scored as micro precision and recall.
"""

from __future__ import annotations

import pytest

from api.cli import build_runtime
from eval.benchmark import functions_in, measure_repo, module_name, render

jedi = pytest.importorskip("jedi", reason="dev dependency: pip install -r requirements-dev.txt")

FILES = {
    "shop/__init__.py": "",
    "shop/money.py": (
        "def cents(value):\n    return int(value * 100)\n\n\n"
        "class Till:\n    def total(self, items):\n        return sum(cents(i) for i in items)\n\n"
        "    def receipt(self, items):\n        return self.total(items)\n"
    ),
    "shop/orders.py": (
        "from shop.money import Till, cents\n\n\n"
        "def place(items):\n    till = Till()\n    print(cents(1))\n    return till.total(items)\n"
    ),
}


@pytest.fixture
def repo(tmp_path):
    for rel, text in FILES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return tmp_path


def test_functions_are_named_the_way_sei_names_them(repo):
    names = [f.qualified_name for f in functions_in(repo, "shop/money.py")]
    assert names == ["shop.money.cents", "shop.money.Till.total", "shop.money.Till.receipt"]
    assert module_name("shop/__init__.py") == "shop"


def test_jedi_and_sei_are_scored_against_each_other(repo, graph_db, tmp_path_factory):
    graph_db.run("MATCH (n) DETACH DELETE n")
    runtime = build_runtime("repo_bench_test", repo, ai="off", driver=graph_db.driver,
                            cache_path=tmp_path_factory.mktemp("c") / "c.sqlite")
    try:
        result = measure_repo(repo, runtime, limit=10, name="shop")
    finally:
        runtime.indexer.cache.close()
        graph_db.run("MATCH (n) DETACH DELETE n")

    by_name = {s.function.qualified_name: s for s in result.scored}
    # jedi sees both calls to `cents`, and `till.total(...)` through the local's type.
    assert by_name["shop.money.cents"].gold == {"shop.money.Till.total", "shop.orders.place"}
    assert by_name["shop.money.Till.total"].gold == {"shop.money.Till.receipt", "shop.orders.place"}
    # `till = Till()` then `till.total(...)`: the local's type is certain, so
    # SEI resolves it too (this pinned the gap before receivers were typed).
    assert by_name["shop.money.Till.total"].found == {"shop.money.Till.receipt", "shop.orders.place"}
    assert result.precision == 1.0 and result.recall == 1.0
    assert "| shop |" in render([result])
