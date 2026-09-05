"""Step 1 — record the environment, and check §5.4's break-even arithmetic.

These are not probes in the plan's table; they are the other half of the step 1
gate. The Neo4j version and edition are recorded because §11's header says the
constructs shift across 5.x minors, so a probe result without a version is an
anecdote. The break-even check is recorded because §5.4 states plainly that if
the chosen model's input rate exceeds ~$0.94/Mtok, the §8.2 cost gate and the
21,200-token ceiling cannot both hold.
"""

from __future__ import annotations

import pytest

import config
from config import (
    AVAILABLE,
    BREAK_EVEN_INPUT_RATE_USD_PER_MTOK,
    CTX_BUDGET,
    GATE_COST_P50_USD,
    RESERVED_OUT,
    RESERVED_SYS,
)


def test_record_server_version(server_info, probe_log, throwaway_db, request):
    # Recorded on the config so the probe report header can name the build the
    # other probes describe. Set here rather than by an autouse fixture:
    # probe_tsx_grammar needs no database and must not require one.
    request.config._sei_server_line = (
        f"{server_info.name} {server_info.version} ({server_info.edition})"
    )
    rec = probe_log.record(
        "environment",
        "Which Neo4j build are the probes above actually describing?",
    )
    rec.statement("CALL dbms.components() YIELD name, versions, edition")
    rec.observe("name", server_info.name)
    rec.observe("version", server_info.version)
    rec.observe("edition", server_info.edition)
    rec.observe("throwaway-db mode", throwaway_db.mode)
    rec.observe("test database", throwaway_db.name)
    rec.observe(
        "multi-database available",
        server_info.supports_multi_database,
    )
    rec.conclude(
        f"{server_info.name} {server_info.version} {server_info.edition}; "
        f"per-module isolation via {throwaway_db.mode}"
    )

    assert server_info.version.startswith("5."), (
        f"spec §11 targets Neo4j 5.x; server reports {server_info.version}"
    )


def test_throwaway_db_starts_empty(graph_db):
    """The fixture's contract. A module that inherits nodes is a false green."""
    assert graph_db.run("MATCH (n) RETURN count(n) AS c")[0]["c"] == 0


def test_available_budget_matches_spec(probe_log):
    """§5.4's AVAILABLE = 21,200 is arithmetic over three † constants."""
    assert CTX_BUDGET - RESERVED_OUT - RESERVED_SYS == AVAILABLE
    assert AVAILABLE == 21_200


def test_break_even_input_rate(probe_log):
    """§5.4: break-even input rate = $0.02 / 21,200 tok ≈ $0.94 per 1M tokens.

    Checked as arithmetic first, then against the model actually chosen. The
    second half fails until a model is named, and that failure is the point:
    §5.4 says the eval config must name the model and version or neither the
    gate nor the break-even can be checked.
    """
    rec = probe_log.record(
        "break_even",
        "Does the chosen model's input rate clear §5.4's ~$0.94/Mtok "
        "break-even at the 21,200-token ceiling?",
    )
    rec.statement(
        f"break_even = ${GATE_COST_P50_USD} / {AVAILABLE:,} tok "
        f"= ${BREAK_EVEN_INPUT_RATE_USD_PER_MTOK:.4f} per 1M input tokens"
    )
    rec.observe("AVAILABLE (ceiling)", f"{AVAILABLE:,} tokens")
    rec.observe(
        "break-even input rate",
        f"${BREAK_EVEN_INPUT_RATE_USD_PER_MTOK:.4f} / Mtok",
    )
    rec.observe("LLM_MODEL", config.LLM_MODEL)
    rec.observe("EMBED_MODEL", config.EMBED_MODEL)
    rec.observe("EMBED_DIM", config.EMBED_DIM)
    rec.observe("configured input rate", config.LLM_INPUT_RATE_USD_PER_MTOK)
    rec.observe("configured output rate", config.LLM_OUTPUT_RATE_USD_PER_MTOK)

    assert 0.93 < BREAK_EVEN_INPUT_RATE_USD_PER_MTOK < 0.95

    if config.LLM_MODEL == "UNSET" or config.LLM_INPUT_RATE_USD_PER_MTOK is None:
        rec.conclude("BLOCKED — no model named; §5.4 cannot be checked")
        pytest.fail(
            "config.LLM_MODEL / LLM_INPUT_RATE_USD_PER_MTOK are unset. Step 1's "
            "gate requires naming the model and version and checking §5.4's "
            "break-even against the actual rate card. This is a decision, not a "
            "defect — see docs/decisions/0002-model-selection-and-break-even.md.",
            pytrace=False,
        )

    rate = config.LLM_INPUT_RATE_USD_PER_MTOK
    clears = rate <= BREAK_EVEN_INPUT_RATE_USD_PER_MTOK
    rec.observe("clears break-even at the ceiling", clears)
    rec.conclude(
        f"{config.LLM_MODEL} at ${rate}/Mtok "
        + (
            "clears the break-even at the full ceiling"
            if clears
            else "exceeds the break-even at the ceiling; a §5.4 lever is recorded "
            "in docs/decisions/0002 and p50 packed tokens decides actual cost"
        )
    )
