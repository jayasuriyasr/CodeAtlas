"""SEI configuration.

Every constant the spec marks with a dagger (†) lives here and nowhere else.
They are unvalidated starting points (spec §12.10), not measured values; they
will move once step 6 and step 10 produce data. Keeping them in one module is
what makes re-tuning a diff instead of a search.

Anything read from the environment is for connecting to infrastructure, not for
tuning behaviour: a † constant read from the environment would be untraceable
in a run.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# Neo4j connection (infrastructure, not tuning)
# --------------------------------------------------------------------------

NEO4J_URI: str = os.environ.get("SEI_NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER: str = os.environ.get("SEI_NEO4J_USER", "neo4j")
NEO4J_PASSWORD: str = os.environ.get("SEI_NEO4J_PASSWORD", "sei-dev-password")

#: The single user database Community Edition provides (spec §9.1, §12.1).
#: Tests fall back to scoping inside it when CREATE DATABASE is unavailable.
NEO4J_DEFAULT_DATABASE: str = os.environ.get("SEI_NEO4J_DATABASE", "neo4j")

SCHEMA_CYPHER: Path = REPO_ROOT / "graph" / "schema.cypher"


# --------------------------------------------------------------------------
# Models (spec §5.4 requires the eval config to name model and version, or
# neither the $/query gate nor the break-even can be checked)
# --------------------------------------------------------------------------

#: Synthesis model. Recorded here so §8.2's $/query gate has a referent.
#: Chosen in docs/decisions/0002; rates from Anthropic's card, 2026-08-27.
LLM_MODEL: str = os.environ.get("SEI_LLM_MODEL", "claude-haiku-4-5")
LLM_INPUT_RATE_USD_PER_MTOK: float | None = 1.00
LLM_OUTPUT_RATE_USD_PER_MTOK: float | None = 5.00

#: Embedding model. Changing this is a full re-index (§12.11): the vector
#: index is dropped and recreated and the entire cache is invalidated.
#: voyage-code-2 emits 1536 dims, which is what §11.3's index already declares.
EMBED_MODEL: str = os.environ.get("SEI_EMBED_MODEL", "voyage-code-2")

#: Must equal the `vector.dimensions` literal in graph/schema.cypher.
#: bootstrap.apply_schema asserts this rather than letting them drift.
EMBED_DIM: int = 1536


# --------------------------------------------------------------------------
# † constants — spec §12.10. Unvalidated guesses. Expect all of these to move.
# --------------------------------------------------------------------------

# Indexing -----------------------------------------------------------------
DEBOUNCE_SECONDS: float = 2.0          # † §4.1
BULK_THRESHOLD: int = 200              # † §4.1 — monorepo checkouts touch 1000s
CHUNK_OVERFLOW_TOKENS: int = 1200      # † §3.4 overflow split cap

# Retrieval ----------------------------------------------------------------
RRF_TOP_K: int = 30                    # † §5.2 step 1
RERANK_TOP_K: int = 8                  # † §5.2 step 2 (Phase 1; MVP uses raw RRF)
HUB_CUTOFF: int = 100                  # † §5.2 step 3 / §11.1 $hub_cutoff
PER_SEED_LIMIT: int = 25               # † §11.1 $per_seed_limit
GLOBAL_EXPANSION_LIMIT: int = 60       # † §11.1 $global_limit

#: † §5.2. Substituted as a map parameter, so the dynamic-key read
#: `$edge_weights[rel]` in §11.1 must work — probe_dynamic_map_key.
EDGE_WEIGHTS: dict[str, float] = {
    "CALLS": 1.0,
    "DISPATCHES_TO": 0.9,
    "INVOKES_HOOK": 0.9,
    "DEFINES": 0.8,
    "RENDERS": 0.8,
    "USES_SERIALIZER": 0.7,
    "HAS_FIELD": 0.6,
    "IMPORTS": 0.4,
}
EDGE_WEIGHT_DEFAULT: float = 0.5       # † §11.1 coalesce fallback

#: §11.2 step 3b. The driver substitutes the relationship type from this
#: allowlist, never from parsed input. Not a † constant — a safety boundary.
#:
#: Eight, not four. v10.0 listed four while §3.5 named eight, so half the
#: framework layer could not be written and §11.1 would not have traversed it
#: anyway (v10.1 §0.0 R2). A fixed list of eight is exactly as safe as one of
#: four; what matters is that it is fixed, not that it is short.
EDGE_TYPE_ALLOWLIST: tuple[str, ...] = (
    "CALLS",
    "IMPORTS",
    "DEFINES",
    "DISPATCHES_TO",
    "INVOKES_HOOK",
    "RENDERS",
    "HAS_FIELD",
    "USES_SERIALIZER",
)

#: Every allowlisted type must have a weight, or §11.1 scores it at the coalesce
#: default while claiming to weight it. Checked here rather than discovered as a
#: quietly mis-ranked expansion.
assert set(EDGE_TYPE_ALLOWLIST) == set(EDGE_WEIGHTS), (
    f"weights and allowlist disagree: "
    f"{set(EDGE_TYPE_ALLOWLIST) ^ set(EDGE_WEIGHTS)}"
)

# Context budgeting (§5.4) -------------------------------------------------
CTX_BUDGET: int = 24_000               # †
RESERVED_OUT: int = 2_000              # †
RESERVED_SYS: int = 800                # †
AVAILABLE: int = CTX_BUDGET - RESERVED_OUT - RESERVED_SYS   # 21_200 ceiling
SEED_SHARE: float = 0.60               # †
MAX_SEED_FRAC: float = 0.40            # †

#: §5.4 arithmetic, not an estimate: $0.02 / 21_200 tok.
BREAK_EVEN_INPUT_RATE_USD_PER_MTOK: float = 0.02 / (AVAILABLE / 1_000_000)

# Grounding (§6.3) ---------------------------------------------------------
COVERAGE_RETRY_THRESHOLD: float = 0.6  # †
MAX_RETRIES: int = 1                   # never a loop

# Gates (§8.2) — all provisional (§12.10) ----------------------------------
GATE_RECALL_AT_10: float = 0.80        # †
GATE_MRR: float = 0.55                 # †
GATE_CITATION_COVERAGE: float = 0.85   # †
GATE_CLASSIFIER_PRECISION: float = 0.90  # †
GATE_CLASSIFIER_RECALL: float = 0.85   # †
GATE_RELATION_PRECISION: float = 0.90  # †
GATE_CACHE_HIT_RATE: float = 0.90      # †
GATE_CONTEXT_DROP_RATE: float = 0.10   # † (upper bound)
GATE_INBOUND_EDGE_SURVIVAL: float = 0.95  # †
GATE_TTFT_P50_S: float = 1.2           # †
GATE_TTFT_P95_S: float = 2.5           # †
GATE_COST_P50_USD: float = 0.02        # †
GATE_COST_P95_USD: float = 0.06        # †

# Decision metrics (§8.3) --------------------------------------------------
DECIDE_TRACE_RESOLUTION_BELOW: float = 0.70    # †
DECIDE_UNDETECTED_MOVE_ABOVE: float = 0.10     # †
DECIDE_HEADER_ONLY_MISS_ABOVE: float = 0.20    # †
DECIDE_SEED_L1_OVERFLOW_ABOVE: float = 0.05    # †

# Embedding cache (S5) -----------------------------------------------------
EMBEDDING_CACHE_PATH: Path = Path(
    os.environ.get("SEI_CACHE_PATH", str(REPO_ROOT / ".sei" / "embeddings.sqlite"))
)


# --------------------------------------------------------------------------
# Driver mechanics — not † constants, not behaviour. §11.2 3a notes that "the
# driver chunks $symbols for large batches"; this is that chunk. It changes how
# many round trips a write takes, never what the write means.
# --------------------------------------------------------------------------

WRITE_CHUNK_SIZE: int = 1_000


# --------------------------------------------------------------------------
# Retrieval mechanics not named by the spec
# --------------------------------------------------------------------------

#: Reciprocal Rank Fusion's rank offset. §5.2 names RRF but not this constant.
#: 60 is the value from the original RRF paper and the usual default; it damps
#: the influence of the very top ranks so one arm cannot dominate the fusion.
#: Untuned against this corpus — treat it as a † in all but the marking.
RRF_K: int = 60
