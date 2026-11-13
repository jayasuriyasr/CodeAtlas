"""The suite never wipes a database it does not own (`tests/db_guard.py`).

Reproduced before the fix: a node of a project indexed with `sei start`, in
the app's database at 7687, was gone after one test module ran — along with
the app's schema.
"""

from __future__ import annotations

import pytest

from config import NEO4J_URI
from tests.db_guard import (
    MARKER_INDEX,
    TEST_URI_DEFAULT,
    allow_wipe,
    connection_settings,
    inspect,
    may_wipe,
    refusal,
)


def test_the_suite_does_not_default_to_the_apps_server():
    uri, _user, _password = connection_settings({})
    assert uri == TEST_URI_DEFAULT
    assert uri != NEO4J_URI


def test_the_test_server_can_be_chosen():
    uri, user, password = connection_settings({
        "SEI_TEST_NEO4J_URI": "bolt://ci:7687", "SEI_TEST_NEO4J_USER": "u",
        "SEI_TEST_NEO4J_PASSWORD": "p"})
    assert (uri, user, password) == ("bolt://ci:7687", "u", "p")


@pytest.mark.parametrize("has_data, marked, allowed, expected", [
    (False, False, False, True),    # empty: nothing to lose
    (True, True, False, True),      # the suite's own database
    (True, False, False, False),    # someone's data: stop
    (True, False, True, True),      # explicitly allowed
])
def test_when_the_suite_may_wipe(has_data, marked, allowed, expected):
    assert may_wipe(has_data, marked, allowed) is expected


def test_the_override_is_explicit():
    assert allow_wipe({}) is False and allow_wipe({"SEI_TEST_ALLOW_WIPE": "0"}) is False
    assert allow_wipe({"SEI_TEST_ALLOW_WIPE": "1"}) is True


def test_the_refusal_says_what_it_found_and_how_to_proceed():
    message = refusal("bolt://localhost:7687", ["my_project"])
    assert "my_project" in message and "SEI_TEST_NEO4J_URI" in message
    assert "SEI_TEST_ALLOW_WIPE" in message


def test_the_database_in_use_is_marked_as_the_suites(graph_db):
    """Every wipe leaves the marker, so the next session recognises it."""
    _has_data, marked, _repos = inspect(graph_db.driver, graph_db.name)
    assert marked, f"{MARKER_INDEX} missing"


def test_unmarked_data_is_detected(graph_db):
    graph_db.run(f"DROP INDEX {MARKER_INDEX} IF EXISTS")
    graph_db.run("CREATE (:Symbol {repo_id: 'someones_project', uid: 'x'})")
    try:
        has_data, marked, repos = inspect(graph_db.driver, graph_db.name)
        assert has_data and not marked and repos == ["someones_project"]
        assert may_wipe(has_data, marked, allowed=False) is False
    finally:
        graph_db.run("MATCH (n) DETACH DELETE n")
        from tests.db_guard import mark
        mark(graph_db.driver, graph_db.name)


def test_the_app_refuses_to_index_into_the_test_database(graph_db):
    from api.cli import refuse_test_database

    with pytest.raises(SystemExit) as stopped:
        refuse_test_database(graph_db.driver, graph_db.name)
    assert "test" in str(stopped.value)
