"""Shared pytest fixtures.

The central one is `graph_db`: a throwaway database per **test module**, so a
module never inherits another module's nodes.

The honest version of "throwaway database" depends on the edition:

* **Enterprise** — `CREATE DATABASE` works, so each module gets a real,
  physically separate database, dropped on teardown.
* **Community** — spec §9.1 and §12.1 both state it: one user database, no
  `CREATE DATABASE`. The fixture degrades to wiping the single database and
  re-applying the schema per module. Equivalent isolation between modules,
  *not* equivalent to a separate database, and it means test modules cannot run
  in parallel against one Community instance.

Which mode ran is reported in the fixture and recorded in PROGRESS.md, because
"the tests pass" means something different in each. See FINDINGS.md F-001.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import pytest
from neo4j import Driver, GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from config import (
    NEO4J_DEFAULT_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
)
from graph.bootstrap import apply_schema, drop_schema

_UNREACHABLE = (
    f"NEO4J UNAVAILABLE at {NEO4J_URI} — these tests did NOT run.\n"
    f"Start it with:  docker compose up -d   (then: docker compose ps)\n"
    f"A gate covering a skipped test is NOT met. See PROGRESS.md."
)

#: Set SEI_REQUIRE_NEO4J=1 to turn "no database" into a failure instead of a
#: skip. CI must set it: a suite that goes green because half of it never ran
#: is the exact false green the plan's gates exist to prevent. Locally it
#: defaults to skipping so the database-free steps stay workable.
_REQUIRE_NEO4J = os.environ.get("SEI_REQUIRE_NEO4J", "").strip() not in ("", "0", "false")


# --------------------------------------------------------------------------
# Session-scoped: one driver, one edition probe
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def neo4j_driver() -> Driver:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
    except (ServiceUnavailable, Neo4jError, OSError) as exc:  # pragma: no cover - env
        driver.close()
        message = f"{_UNREACHABLE}\n\nDriver said: {exc}"
        if _REQUIRE_NEO4J:
            pytest.fail(message, pytrace=False)
        pytest.skip(message, allow_module_level=True)
    yield driver
    driver.close()


@dataclass(frozen=True)
class ServerInfo:
    """What the server actually reports. Recorded verbatim in PROGRESS.md."""

    name: str
    version: str
    edition: str

    @property
    def supports_multi_database(self) -> bool:
        return self.edition.lower() == "enterprise"


@pytest.fixture(scope="session")
def server_info(neo4j_driver: Driver) -> ServerInfo:
    with neo4j_driver.session(database="system") as session:
        rec = session.run(
            "CALL dbms.components() YIELD name, versions, edition "
            "RETURN name, versions[0] AS version, edition"
        ).single()
    return ServerInfo(name=rec["name"], version=rec["version"], edition=rec["edition"])


# --------------------------------------------------------------------------
# Module-scoped: the throwaway database itself
# --------------------------------------------------------------------------


@dataclass
class TestDatabase:
    driver: Driver
    name: str
    #: "database" when a real database was created, "scoped-wipe" on Community.
    mode: str

    def session(self, **kw):
        return self.driver.session(database=self.name, **kw)

    def run(self, cypher: str, **params):
        """Run one statement and return the records as a list."""
        with self.session() as s:
            return list(s.run(cypher, params))


def _db_name_for(module_name: str) -> str:
    """A legal Neo4j database name derived from the test module.

    Neo4j: 3-63 chars, starts with an ASCII letter, letters/digits/dots/dashes.
    Module paths contain underscores, so hash rather than sanitise - a hash also
    keeps the name short enough for deeply nested test packages.
    """
    digest = hashlib.sha1(module_name.encode()).hexdigest()[:16]
    return f"seitest{digest}"


@pytest.fixture(scope="module")
def throwaway_db(neo4j_driver: Driver, server_info: ServerInfo, request) -> TestDatabase:
    """A database this module owns outright. Torn down after the module."""
    name = _db_name_for(request.module.__name__)

    if server_info.supports_multi_database:
        with neo4j_driver.session(database="system") as sys_session:
            sys_session.run(f"CREATE OR REPLACE DATABASE {name} WAIT").consume()
        db = TestDatabase(driver=neo4j_driver, name=name, mode="database")
        yield db
        with neo4j_driver.session(database="system") as sys_session:
            sys_session.run(f"DROP DATABASE {name} IF EXISTS DESTROY DATA WAIT").consume()
        return

    # Community: one user database (§9.1). Wipe it instead, both before and
    # after, so a module neither inherits nor leaks state.
    db = TestDatabase(driver=neo4j_driver, name=NEO4J_DEFAULT_DATABASE, mode="scoped-wipe")
    _wipe(db)
    yield db
    _wipe(db)


def _wipe(db: TestDatabase) -> None:
    drop_schema(db.driver, db.name)
    with db.session() as s:
        # Batched: a single DETACH DELETE over a large graph can exhaust the
        # heap, and these tests do build a few thousand nodes.
        while True:
            summary = s.run(
                "MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(n) AS c"
            ).single()
            if summary["c"] == 0:
                break


@pytest.fixture(scope="module")
def graph_db(throwaway_db: TestDatabase) -> TestDatabase:
    """`throwaway_db` with spec §11.3's schema applied."""
    apply_schema(throwaway_db.driver, throwaway_db.name)
    return throwaway_db


@pytest.fixture
def repo_id(request) -> str:
    """A repo_id unique to the test.

    Tenancy is `repo_id` on every node (§9.1), so scoping tests by it is both
    isolation and a standing check that the scoping works.
    """
    return "repo_" + hashlib.sha1(request.node.nodeid.encode()).hexdigest()[:12]


@pytest.fixture(scope="session")
def fixtures_dir():
    from pathlib import Path

    return Path(__file__).parent / "fixtures"
