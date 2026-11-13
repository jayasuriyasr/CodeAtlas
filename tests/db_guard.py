"""Which Neo4j the test suite may use, and whether it may wipe it.

Neo4j Community has one user database, so the suite isolates test modules by
wiping it — data and schema. Pointed at the database a developer actually
indexes into, `pytest` deleted their project without asking. Two rules now:

* The suite talks to its own server: `SEI_TEST_NEO4J_URI`, by default
  `bolt://localhost:7688`, a port the app never uses.
* It wipes a database only if the database is empty or carries the suite's
  marker (an index nothing else creates). A database with data and no marker
  belongs to someone; the suite stops instead. `SEI_TEST_ALLOW_WIPE=1`
  overrides that, for a test database whose marker was lost.

The app refuses the other direction: `sei start` will not index into a
database that carries the marker (`api.cli.refuse_test_database`).
"""

from __future__ import annotations

from typing import Mapping

from config import NEO4J_PASSWORD, NEO4J_USER

TEST_URI_DEFAULT = "bolt://localhost:7688"
#: An index survives `MATCH (n) DETACH DELETE n`, which many test modules run;
#: a marker node would not.
MARKER_INDEX = "sei_test_database_marker"
_MARK = (f"CREATE INDEX {MARKER_INDEX} IF NOT EXISTS "
         f"FOR (m:SeiTestDatabaseMarker) ON (m.id)")


def connection_settings(env: Mapping[str, str]) -> tuple[str, str, str]:
    """`(uri, user, password)` for the test server — never the app's URI."""
    return (
        env.get("SEI_TEST_NEO4J_URI") or TEST_URI_DEFAULT,
        env.get("SEI_TEST_NEO4J_USER") or NEO4J_USER,
        env.get("SEI_TEST_NEO4J_PASSWORD") or NEO4J_PASSWORD,
    )


def allow_wipe(env: Mapping[str, str]) -> bool:
    return env.get("SEI_TEST_ALLOW_WIPE", "").strip() not in ("", "0", "false")


def inspect(driver, database: str) -> tuple[bool, bool, list[str]]:
    """`(has_data, marked, sample_repo_ids)` for `database`."""
    with driver.session(database=database) as session:
        has_data = session.run("MATCH (n) RETURN n LIMIT 1").single() is not None
        marked = session.run(
            "SHOW INDEXES YIELD name WHERE name = $name RETURN count(*) AS n", name=MARKER_INDEX
        ).single()["n"] > 0
        repos = [row["repo"] for row in session.run(
            "MATCH (n) WHERE n.repo_id IS NOT NULL RETURN DISTINCT n.repo_id AS repo LIMIT 5"
        )]
    return has_data, marked, repos


def may_wipe(has_data: bool, marked: bool, allowed: bool) -> bool:
    return marked or not has_data or allowed


def mark(driver, database: str) -> None:
    with driver.session(database=database) as session:
        session.run(_MARK).consume()


def refusal(uri: str, repos: list[str]) -> str:
    shown = ", ".join(repos) if repos else "unlabelled data"
    return (
        f"Refusing to run the graph tests against {uri}: that database holds data "
        f"that is not the test suite's ({shown}). The suite wipes the database it "
        f"uses, so it would delete it.\n"
        f"Point the tests at a server of their own (SEI_TEST_NEO4J_URI, default "
        f"{TEST_URI_DEFAULT}; `docker compose up -d neo4j-test` starts one), or set "
        f"SEI_TEST_ALLOW_WIPE=1 if this really is a test database."
    )
