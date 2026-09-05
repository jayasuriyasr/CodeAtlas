"""Apply graph/schema.cypher (spec §11.3) to a database.

The schema file is kept verbatim from the frozen spec, which means it is a
multi-statement script and Neo4j's Bolt protocol runs one statement per call.
Everything here exists to split that file faithfully and to refuse to apply a
schema whose vector dimensionality disagrees with `config.EMBED_DIM` — a
mismatch that would otherwise surface much later as an index that silently
rejects every vector.
"""

from __future__ import annotations

import re
from pathlib import Path

from neo4j import Driver

from config import EMBED_DIM, SCHEMA_CYPHER


def split_statements(script: str) -> list[str]:
    """Split a Cypher script on top-level `;`.

    Quote- and comment-aware, because a naive ``script.split(";")`` would break
    on the first semicolon inside a string literal or a `//` comment. Neo4j has
    no server-side script runner over Bolt, so this has to be done client side.
    """
    out: list[str] = []
    buf: list[str] = []
    i, n = 0, len(script)
    quote: str | None = None

    while i < n:
        ch = script[i]

        if quote:
            buf.append(ch)
            if ch == "\\" and i + 1 < n:          # escaped char inside a literal
                buf.append(script[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue

        if ch in ("'", '"', "`"):
            quote = ch
            buf.append(ch)
            i += 1
            continue

        if script.startswith("//", i):
            end = script.find("\n", i)
            i = n if end == -1 else end           # drop the comment, keep the \n
            continue

        if script.startswith("/*", i):
            end = script.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue

        if ch == ";":
            stmt = "".join(buf).strip()
            if stmt:
                out.append(stmt)
            buf = []
            i += 1
            continue

        buf.append(ch)
        i += 1

    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out


def schema_vector_dimensions(script: str) -> int | None:
    """Read the `vector.dimensions` literal out of the schema script."""
    m = re.search(r"`vector\.dimensions`\s*:\s*(\d+)", script)
    return int(m.group(1)) if m else None


def apply_schema(driver: Driver, database: str, schema_path: Path = SCHEMA_CYPHER) -> list[str]:
    """Apply every statement in the schema file. Returns the statements run.

    Idempotent: every statement in §11.3 is `IF NOT EXISTS`, so a second run is
    a no-op. That property is a step 1 gate, not an assumption — see
    tests/step_01_ground_truth/test_schema_apply.py.
    """
    script = schema_path.read_text(encoding="utf-8")

    dims = schema_vector_dimensions(script)
    if dims is not None and dims != EMBED_DIM:
        raise ValueError(
            f"schema.cypher declares vector.dimensions={dims} but config.EMBED_DIM="
            f"{EMBED_DIM}. Changing the embedding model is a full re-index (§12.11); "
            f"reconcile these deliberately, do not paper over it."
        )

    statements = split_statements(script)
    with driver.session(database=database) as session:
        for stmt in statements:
            session.run(stmt).consume()
    return statements


def drop_schema(driver: Driver, database: str) -> None:
    """Drop every constraint and index. Used to prove idempotency from clean."""
    with driver.session(database=database) as session:
        for row in list(session.run("SHOW CONSTRAINTS YIELD name RETURN name")):
            session.run(f"DROP CONSTRAINT {row['name']} IF EXISTS").consume()
        for row in list(session.run("SHOW INDEXES YIELD name, type RETURN name, type")):
            if row["type"] == "LOOKUP":           # token lookup indexes are built in
                continue
            session.run(f"DROP INDEX {row['name']} IF EXISTS").consume()
