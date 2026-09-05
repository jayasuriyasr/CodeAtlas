"""Step 1 gate — `schema.cypher` applies clean on a fresh DB and is idempotent.

Spec §11.3 is one script of `IF NOT EXISTS` statements, which *should* make a
second run a no-op. "Should" is the reason this test exists: the constraints
are composite, the fulltext index names a non-default analyzer, and the vector
index carries an options map — three places where a re-run can quietly create a
second index under a generated name instead of doing nothing.
"""

from __future__ import annotations

import pytest

from config import EMBED_DIM, SCHEMA_CYPHER
from graph.bootstrap import apply_schema, drop_schema, split_statements

#: The names §11.3 creates. Anything else appearing is a defect, not a bonus.
EXPECTED_CONSTRAINTS = {"symbol_uid", "file_key"}
EXPECTED_INDEXES = {
    "symbol_repo_path",
    "symbol_lines",
    "symbol_name",
    "symbol_origin",
    "symbol_code_vec",
    "symbol_search",
}


def _catalogue(db) -> dict[str, tuple]:
    """A comparable snapshot of the schema catalogue.

    `id` and `lastRead` are excluded deliberately: they legitimately differ
    between runs and comparing them would make idempotency untestable rather
    than tested.
    """
    out: dict[str, tuple] = {}
    for row in db.run(
        "SHOW CONSTRAINTS YIELD name, type, entityType, labelsOrTypes, properties "
        "RETURN name, type, entityType, labelsOrTypes, properties"
    ):
        out[f"constraint:{row['name']}"] = (
            row["type"], row["entityType"], row["labelsOrTypes"], row["properties"]
        )
    for row in db.run(
        "SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties, options "
        "RETURN name, type, entityType, labelsOrTypes, properties, options"
    ):
        out[f"index:{row['name']}"] = (
            row["type"], row["entityType"], row["labelsOrTypes"], row["properties"],
            (row["options"] or {}).get("indexConfig"),
        )
    return out


@pytest.fixture(scope="module")
def fresh(throwaway_db):
    drop_schema(throwaway_db.driver, throwaway_db.name)
    return throwaway_db


def test_schema_splits_into_the_statements_1113_declares():
    """Pure function, no database: the splitter is where a `;` inside a comment
    or a backticked key like `vector.dimensions` would silently truncate the
    script and leave half the schema unapplied."""
    statements = split_statements(SCHEMA_CYPHER.read_text(encoding="utf-8"))
    kinds = [s.split()[1] for s in statements]      # CREATE <kind> ...
    assert kinds.count("CONSTRAINT") == len(EXPECTED_CONSTRAINTS)
    assert kinds.count("INDEX") == 4                # the four plain indexes
    assert kinds.count("VECTOR") == 1
    assert kinds.count("FULLTEXT") == 1
    assert len(statements) == 8


def test_schema_applies_clean_on_fresh_db(fresh, probe_log):
    rec = probe_log.record(
        "schema_apply",
        "Does spec §11.3 apply end to end on a fresh database, and is it "
        "idempotent on a second run?",
    )
    statements = apply_schema(fresh.driver, fresh.name)
    rec.observe("statements applied", len(statements))

    after_first = _catalogue(fresh)
    names = {k.split(":", 1)[1] for k in after_first}
    rec.observe("constraints created", sorted(EXPECTED_CONSTRAINTS & names))
    rec.observe("indexes created", sorted(EXPECTED_INDEXES & names))
    unexpected = names - EXPECTED_CONSTRAINTS - EXPECTED_INDEXES
    # Neo4j maintains built-in token lookup indexes; they are not ours.
    rec.observe("other catalogue entries", sorted(unexpected) or "none")

    assert EXPECTED_CONSTRAINTS <= names
    assert EXPECTED_INDEXES <= names

    # Second run: same statements, catalogue must not move.
    apply_schema(fresh.driver, fresh.name)
    after_second = _catalogue(fresh)
    rec.observe("catalogue identical after second apply", after_first == after_second)
    rec.conclude(
        "APPLIES CLEAN AND IS IDEMPOTENT"
        if after_first == after_second
        else "NOT IDEMPOTENT — second apply changed the catalogue"
    )

    assert after_first == after_second, {
        "only_after_first": {k: v for k, v in after_first.items()
                             if after_second.get(k) != v},
        "only_after_second": {k: v for k, v in after_second.items()
                              if after_first.get(k) != v},
    }


def test_vector_index_config_matches_config_py(fresh, probe_log):
    """A dimension mismatch here is invisible until every embedding is rejected."""
    apply_schema(fresh.driver, fresh.name)
    cfg = _catalogue(fresh)["index:symbol_code_vec"][4]
    probe_log.records["schema_apply"].observe("symbol_code_vec indexConfig", cfg)
    assert cfg["vector.dimensions"] == EMBED_DIM
    assert cfg["vector.similarity_function"].lower() == "cosine"


def test_fulltext_analyzer_is_standard_no_stop_words(fresh, probe_log):
    """§5.3: stop-word removal deletes `in`, `for`, `if`, `not` — meaningful in
    code. The analyzer is half the fix; the other half is Python-side splitting.
    If the server silently fell back to `standard`, the loss would be invisible.
    """
    apply_schema(fresh.driver, fresh.name)
    cfg = _catalogue(fresh)["index:symbol_search"][4]
    probe_log.records["schema_apply"].observe("symbol_search indexConfig", cfg)
    assert cfg["fulltext.analyzer"] == "standard-no-stop-words"


def test_constraints_are_enforced(fresh, repo_id):
    """The uniqueness constraint is what makes `MERGE (s:Symbol {uid})` identity.

    Declared-but-unenforced would let the T1 collision the ordinal exists to
    prevent pass unnoticed.
    """
    from neo4j.exceptions import ConstraintError

    apply_schema(fresh.driver, fresh.name)
    fresh.run("CREATE (s:Symbol {uid: $u, repo_id: $r})", u="dup_probe", r=repo_id)
    with pytest.raises(ConstraintError):
        fresh.run("CREATE (s:Symbol {uid: $u, repo_id: $r})", u="dup_probe", r=repo_id)
    fresh.run("MATCH (s:Symbol {uid: 'dup_probe'}) DETACH DELETE s")
