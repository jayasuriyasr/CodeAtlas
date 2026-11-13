"""The development password stays on this machine.

`config.NEO4J_PASSWORD` defaults to `sei-dev-password`, which is in the
repository for anyone to read. That is fine for a server only this machine can
reach, and it was the only one the default could reach — except that the
compose file published Neo4j's ports on every interface (`"7687:7687"`), and
nothing stopped `SEI_NEO4J_URI` pointing at another host with the same
password.
"""

from __future__ import annotations

import re

import pytest

import config
from api.cli import check_neo4j_credentials


def test_the_default_password_is_fine_for_a_local_server():
    check_neo4j_credentials("bolt://localhost:7687", config.DEFAULT_NEO4J_PASSWORD)
    check_neo4j_credentials("neo4j://127.0.0.1:7687", config.DEFAULT_NEO4J_PASSWORD)


@pytest.mark.parametrize("uri", ["bolt://db.internal:7687", "neo4j+s://example.databases.neo4j.io"])
def test_the_default_password_is_refused_for_any_other_host(uri):
    with pytest.raises(SystemExit) as stopped:
        check_neo4j_credentials(uri, config.DEFAULT_NEO4J_PASSWORD)
    assert "SEI_NEO4J_PASSWORD" in str(stopped.value)


def test_a_chosen_password_may_go_anywhere():
    check_neo4j_credentials("bolt://db.internal:7687", "a-password-someone-chose")


def test_compose_publishes_neo4j_on_localhost_only():
    compose = (config.REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    mappings = re.findall(r'^\s*-\s*"([^"]+:\d+)"', compose, re.M)
    assert mappings, "no port mappings found"
    for mapping in mappings:
        assert mapping.startswith("127.0.0.1:"), mapping


def test_compose_points_at_no_missing_document():
    compose = (config.REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    for path in re.findall(r"\b(docs/[\w./-]+)", compose):
        assert (config.REPO_ROOT / path).exists(), path
