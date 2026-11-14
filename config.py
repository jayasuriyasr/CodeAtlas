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
#: The development password, for a server only this machine can reach. It is
#: in the repository, so `api.cli` refuses it for any other host.
DEFAULT_NEO4J_PASSWORD = "sei-dev-password"
NEO4J_PASSWORD: str = os.environ.get("SEI_NEO4J_PASSWORD", DEFAULT_NEO4J_PASSWORD)

#: The single user database Community Edition provides (spec §9.1, §12.1).
#: Tests fall back to scoping inside it when CREATE DATABASE is unavailable.
NEO4J_DEFAULT_DATABASE: str = os.environ.get("SEI_NEO4J_DATABASE", "neo4j")

SCHEMA_CYPHER: Path = REPO_ROOT / "graph" / "schema.cypher"


# --------------------------------------------------------------------------
# Models (spec §5.4 requires the eval config to name model and version, or
# neither the $/query gate nor the break-even can be checked)
# --------------------------------------------------------------------------

#: Synthesis model. Recorded here so §8.2's $/query gate has a referent.
#: Chosen in decision 0002 (DESIGN.md); rates from Anthropic's card, 2026-08-27.
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
# Local models (decision 0003) — an Ollama server on this machine
# --------------------------------------------------------------------------

#: Where the local model server listens. Infrastructure, like the Neo4j URI.
OLLAMA_URL: str = os.environ.get("SEI_OLLAMA_URL", "http://127.0.0.1:11434")

#: Synthesis model when no ANTHROPIC_API_KEY is set. Chosen to fit a 4 GB GPU
#: (1.9 GB of weights) while being trained on code. Free, private, and weaker
#: than the hosted model decision 0002 names — see decision 0003.
LOCAL_LLM_MODEL: str = os.environ.get("SEI_LOCAL_LLM", "qwen2.5-coder:3b")

#: Embedding model when no VOYAGE_API_KEY is set. Emits 768 dimensions, which
#: `OllamaEmbeddingProvider` zero-pads to EMBED_DIM (FINDINGS F-025).
LOCAL_EMBED_MODEL: str = os.environ.get("SEI_LOCAL_EMBED", "nomic-embed-text")

#: Context window the local model is loaded with (`num_ctx`). Unlike
#: CTX_BUDGET — a quality parameter far below the hosted model's window — this
#: is a capacity limit: past it the server silently drops the start of the
#: prompt. 8,192 keeps the model, its cache and the embedder inside 4 GB.
LOCAL_LLM_CONTEXT_TOKENS: int = 8_192

#: Share of the local window the packer may plan to fill. The packer counts
#: with the stand-in tokenizer (F-006). Measured against this model's own over
#: the 93 fixture symbols (F-025), the stand-in *over*-counts: real/stand-in was
#: 0.65 at the median and 0.94 at worst, never above 1. So the whole window is
#: planned, and the plan is already conservative by a third.
LOCAL_CONTEXT_HEADROOM: float = 1.0

#: How long the server keeps a model in memory after a request. Reloading costs
#: several seconds, which would land in every answer's time-to-first-token.
LOCAL_KEEP_ALIVE: str = "30m"

#: Texts per local embedding request.
LOCAL_EMBED_BATCH_SIZE: int = 16

#: Seconds a local request may take. Far longer than the hosted timeout because
#: the first request after a boot loads the model from disk: 46 s was measured
#: for a 274 MB embedder on a cold cache, before a single token was processed.
LOCAL_TIMEOUT_SECONDS: float = 300.0

#: Task prefixes an embedding model was trained with: (document, query). A
#: model missing from this table gets none. Leaving nomic's out costs recall
#: silently — the vectors are still returned, just for a task it was not asked.
LOCAL_EMBED_PREFIXES: dict[str, tuple[str, str]] = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
}

#: Which providers to use: "auto" (hosted when its key is set, else local when
#: the server answers, else none), "local", "cloud", or "off".
AI_MODE: str = os.environ.get("SEI_AI", "auto")


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
    "INHERITS": 0.7,                   # † a guess like the rest; untuned
    "OVERRIDES": 0.6,                  # † a guess like the rest; untuned
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
    "INHERITS",
    "OVERRIDES",
)

#: How many levels of base classes the writer walks to resolve an inherited
#: `self.method()` call. Real hierarchies are a handful deep; the bound exists so
#: a cycle or a pathological graph cannot turn one save into an unbounded walk.
MRO_MAX_DEPTH: int = 16

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

#: A failed flush is re-queued and retried with exponential backoff
#: (DEBOUNCE_SECONDS * 2**attempt). After this many consecutive failures the
#: batch stays pending without a timer; the next save re-arms it. Nothing is
#: dropped either way — this only stops a dead database being polled forever.
FLUSH_MAX_RETRIES: int = 3

#: How often the file watcher looks at the working tree. A scan is a directory
#: walk and a `stat` per source file — no file is read unless it changed — so a
#: second is cheap, and §4.1's debounce still coalesces what the scans find.
WATCH_INTERVAL_SECONDS: float = 1.0

#: Texts per embedding request. Voyage accepts up to 128 inputs per call.
EMBED_BATCH_SIZE: int = 128

#: Seconds before an embedding or LLM HTTP request is abandoned.
PROVIDER_TIMEOUT_SECONDS: float = 60.0

#: Vector arm over-fetch. The vector index is shared across tenants (§9.1), so
#: `queryNodes` is asked for this many times `k` before filtering on repo_id —
#: otherwise a top-k can be entirely another tenant's rows.
VECTOR_OVERFETCH_FACTOR: int = 4

#: The over-fetch above is only a first guess: beside a repository twenty-five
#: times its size, a 73-symbol repository got 0 to 11 of the 60 rows it asked
#: for (FINDINGS F-028). The fetch is therefore widened by the same factor until
#: the repository has its `k`, the index is exhausted, or this many neighbours
#: have been asked for — past which the arm returns what it has and counts
#: `retrieve.vector_starved`.
VECTOR_MAX_FETCH: int = 16_384

#: Most related symbols a STRUCTURAL traversal hands to the packer. The total is
#: always reported; this bounds how many are loaded, not what the answer says
#: exists — a 400-caller hub is "400 callers, 50 shown", never "50 callers".
STRUCTURAL_RESULT_LIMIT: int = 50

#: † Multi-hop questions ("what eventually calls X", "what is affected if I
#: change X"): how many call levels are walked, and how many symbols are
#: returned before the walk stops and says so.
MULTIHOP_MAX_DEPTH: int = 4
MULTIHOP_MAX_RESULTS: int = 50

#: Lucene rejects a query with more than 1,024 clauses (`maxClauseCount`). A
#: pasted file or log as a "question" tokenizes past that and would fail the
#: whole fulltext arm, so the query keeps its first terms and drops the rest.
FULLTEXT_MAX_TERMS: int = 256

#: Sampling temperature for synthesis. §8.2: "Eval runs at temperature 0" — and
#: a production answer that changes between identical requests is a citation
#: the user cannot reproduce. Not a † constant: it is a correctness setting.
#: Sent through `extra_body` (the anthropic 1.x SDK dropped the keyword; the API
#: did not). `claude-haiku-4-5` accepts it; Opus 4.7+ reject any sampling value
#: and Haiku/Sonnet 5.5 reject non-default ones — set None when moving to those.
LLM_TEMPERATURE: float | None = 0.0

#: TypeScript path aliases resolved when turning an import specifier into an
#: edge target. `@/` is create-next-app's default `tsconfig` mapping to the
#: project root. Aliases are configuration, not syntax: a repo that maps `@/`
#: elsewhere, or uses others, is resolved only as far as this table says.
TS_PATH_ALIASES: dict[str, str] = {"@/": ""}


# --------------------------------------------------------------------------
# Retrieval mechanics not named by the spec
# --------------------------------------------------------------------------

#: Reciprocal Rank Fusion's rank offset. §5.2 names RRF but not this constant.
#: 60 is the value from the original RRF paper and the usual default; it damps
#: the influence of the very top ranks so one arm cannot dominate the fusion.
#: Untuned against this corpus — treat it as a † in all but the marking.
RRF_K: int = 60
