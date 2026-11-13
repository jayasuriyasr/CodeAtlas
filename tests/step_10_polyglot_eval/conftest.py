"""Fixtures shared by the step-10 modules."""

from __future__ import annotations

import pytest

from .pipeline_support import index_fixtures


@pytest.fixture(scope="module")
def indexed(graph_db, fixtures_dir, tmp_path_factory):
    """Both fixture repositories, indexed through the real `Indexer`."""
    yield from index_fixtures(graph_db, fixtures_dir, tmp_path_factory)
