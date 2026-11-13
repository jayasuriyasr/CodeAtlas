# SEI — Design

How the engine is built, what it promises, what it trades away, and where it
is known to fall short. This is the one design document in the repository.

**How the code refers to it.** Comments cite sections by number (`§4.1`),
findings by number (`F-015`), decisions by number (`decision 0003`), revisions
of the original specification by letter (`R4`), defect codes from earlier design
rounds (`T5`, `B8`, `v9.1 #14`) and build steps (`plan §9`). Every one of those
is a heading below; `tests/step_10_polyglot_eval/test_design_references.py`
fails if a comment cites one that is not.

Constants marked † in the code (`config.py`) are unvalidated starting points.

---

## §0 Status

### §0.0 Revision history

The design went through several specification versions; the code still names
them where a change was made. **v10.0** was the first complete specification.
**v10.1** fixed three defects found by running it (R1–R3). **v10.2** fixed four
found by a whole-repository audit (R4–R7). Changes since — local models, the
console, re-indexing on save, inheritance edges, the access checks — are
recorded as findings F-025 onward and decision 0003. Each revision is listed
under *Spec revisions* at the end.

### §0.2 Verified behaviour

The Neo4j behaviours the design depends on were probed against Neo4j 5.26.0
Community before anything was built on them (`tests/step_01_ground_truth/`): a
map lookup with a dynamic key, `COUNT {}` and `EXISTS {}` subqueries, the vector
index DDL, and that `MERGE` rejects a multi-type relationship pattern. The
probes write their actual output to `PROBE_RESULTS.md` on every run.

## §1 Design principles

Determinism where it is possible, a model only where it is not. Every number
that gates a release is defined once and measured, never asserted. An
unmeasured gate is reported as unmeasured, never as passing. Ambiguity is a
refusal, not a guess. Anything that degrades says so to the reader.

## §2 Seams

Five interfaces keep the parts replaceable: S1 the search backend (Neo4j
fulltext and vector indexes), S2 the model providers (embeddings and answers —
hosted, local, or none), S3 the language adapters (Python, TypeScript), S5 the
embedding cache, S6 the graph writer and reader, S7 the evaluation harness.
Each is injected, so tests can stand in a stub without a network or a model.

## §3 Data model

### §3.1 Labels

Two node labels: `:Symbol` (a function, class or module) and `:File`. Every node
carries `repo_id`; one database holds many repositories (§9.1).

### §3.2 UID contract

`uid = sha1(repo_id, rel_path, qualified_name, arity, ordinal)[:20]`. Line
numbers and bodies are excluded so an edit inside a function keeps its identity.
`ordinal` separates symbols that share a name and arity in one file (overloads,
platform-conditional definitions); without it `MERGE` silently merges them.
`rel_path` is POSIX and repo-relative. A changed parameter count is a new UID —
the indexer re-links callers in other files when that happens (F-029, A4 in the
review). Anonymous default exports are named `<module>.default`.

### §3.3 Node properties

A symbol stores its identity fields, signature, docstring, source (scrubbed,
§9.2), enclosing class signature, per-symbol resolved imports, decorators,
declared bases (classes), line span, the chunk's cache hashes, search text, its
embedding (`code_vec`) and the epoch that last wrote it. `vec_epoch` is reserved
and unused.

### §3.4 Chunking

One chunk per symbol, embedded with a header that says where the symbol lives.

#### §3.4.1 What a chunk is

A function's chunk is its source. A class's chunk is its signature, docstring and
members' signatures; a module's is its imports and top-level signatures.

#### §3.4.2 The context header

`# path :: enclosing signature :: signature` plus the imports *this symbol*
uses. Per-symbol imports, not the file's import block: one added import must not
invalidate every cached embedding in the file.

#### §3.4.3 Overflow

A chunk over `CHUNK_OVERFLOW_TOKENS`† is split, and only the first part is
embedded — a symbol has one `code_vec` (F-005).

#### §3.4.4 Elision

When the packer must shorten a block it elides lines and says so with a marker;
the prompt tells the model it cannot make claims about elided content.

#### §3.4.5 Determinism

The same bytes always produce the same chunks and hashes, so a re-index of an
unchanged file is all cache hits.

### §3.5 Framework layer

Edges beyond calls and imports: Django URL modules `DISPATCHES_TO` views, views
`USES_SERIALIZER`, relational model fields `HAS_FIELD` the model they point at
(scalar fields have no node, F-024); React components `INVOKES_HOOK` and
`RENDERS`; Next.js route modules dispatch to their handlers. `INHERITS` links a
class to each declared base. The edge types are a fixed allowlist
(`EDGE_TYPE_ALLOWLIST`), never taken from parsed input.

## §4 Indexing

### §4.1 Debounce contract

A save is parsed at once and queued; the write happens after `DEBOUNCE_SECONDS`†
of quiet, under a per-repository lock, so a burst of saves is one batch and one
repository never blocks another. A failed flush re-queues its batch (R6). The
watcher (`index/watcher.py`) delivers saves and deletions where an editor plugin
would.

### §4.2 Embedding cache

Keyed by `header_hash:body_hash` per embedding model. The header hash covers
exactly what the header shows, plus the scrub-rule version, so a rule change
re-embeds. `ParsedFile.symbols` is the only collection (T6).

### §4.3 Epoch reconciliation

Every batch gets an epoch, strictly greater than any before it (monotonic and
resumed from the graph, F-029/A5). Writes stamp it; step 4 deletes edges and step
5 symbols from the batch's paths that are *older*. The sequence is not atomic: it
is crash-safe by ordering — a superset is written before anything is deleted, and
replaying a batch converges (B15).

### §4.4 Move detection

Signal A: the editor reports a rename. Signal B: git reports one (`git status`
renames). A detected move rewrites UIDs in place so every inbound edge survives;
an undetected one is a delete and a create, counted as `move.undetected` and
repaired by Sync Project. Rename chains collapse to one net move (B5).

### §4.5 Full reconcile

The diff between the working tree and the graph (added, modified, deleted,
moved). Sync Project re-parses every file, so callers re-link to callees that
changed or appeared (R5). Large batches are written in chunks that replay
earlier chunks' edges (R4).

## §5 Retrieval

### §5.1 Query router

STRUCTURAL ("what calls X", "subclasses of X", "what imports X"): resolve the
entity and traverse one hop; several symbols under one name is a question back,
not a merge. TRACE (a pasted stack trace): Tier 0 exact file and line, Tier 2
unique name match (refuses on ambiguity, T10), Tier 3 search on the error
message. SEMANTIC: everything else. The LLM fallback hook exists but is not
wired (see `retrieve/router.py`).

### §5.2 Graph-augmented hybrid retrieval

Fulltext and vector search, fused by reciprocal rank (`RRF_K`†), top seeds, then
one hop of weighted expansion (`EDGE_WEIGHTS`†) that skips hubs above
`HUB_CUTOFF`†. The vector fetch widens until a repository has its rows (F-028).

### §5.3 BM25 on code

Identifiers are split for the fulltext index (`snake_case`, `camelCase`, dotted
paths) so a question in words can find `charge_card`.

### §5.4 Context budgeting

Seeds get full bodies first, neighbours signatures, everything within `AVAILABLE`
tokens — or, for a local model, its window less the reserves. Handles are
assigned after packing, so `[C1]…[Cn]` has no gaps. Token counts use a stand-in
that over-counts (F-006, F-025).

## §6 Grounding

### §6.1 Validation tiers

T1: every cited handle names a block that was shown. T2: every claim sentence
(declarative and naming code) cites one. T3: relation claims in a recognised
form are looked up in the graph. Plus a support check: code names in a cited
claim must appear in the blocks it cites (C4 in the review). Hedged and
interrogative sentences are not claims.

### §6.2 Three signals

Coverage (cited claims ÷ claims), Relation Precision (confirmed ÷ asserted
relations), ConflictFlags (relations whose endpoints exist and whose edge does
not). Reported with every answer.

### §6.3 Auto-retry

Coverage under `COVERAGE_RETRY_THRESHOLD`† or any conflict: one retry with
expanded context and the reason stated. Never a loop; an answer still poorly
grounded after it is marked for review.

### §6.4 Handle format

`[C<n>]`, recognised outside fenced code only — `arr[C1]` in a snippet is not a
citation. Verdicts stream sentence by sentence and match the final report.

## §7 System flow

Save → parse → debounce → embed (cache) → write (§11.2). Question → route →
retrieve → pack → stream the answer → validate as it streams → T3 → maybe one
retry → done event with the numbers.

## §8 Evaluation

### §8.1 Corpus and golden set

Two fixtures written for this project (Django: 18 files, 73 symbols; Next.js: 4
files, 20 symbols) and 45 questions, 15 per router class. Recall is measured on
the packed context the model saw. STRUCTURAL questions carry `expected_related`,
the full answer, so the traversal itself is scored (`eval/structural.py`).

### §8.2 Gates

Recall@10 ≥ 0.80, MRR ≥ 0.55, Coverage ≥ 0.85, T2 classifier precision ≥ 0.90 and
recall ≥ 0.85, Relation Precision ≥ 0.90, cache hit rate ≥ 0.90, context drop rate
≤ 0.10, inbound-edge survival across a move ≥ 0.95, TTFT p50 ≤ 1.2 s and p95 ≤ 2.5 s,
`$/query` p50 ≤ $0.02 and p95 ≤ $0.06. All †. One flipped question is 6.7 points of
a class score.

### §8.3 Decision metrics

They choose future work rather than gate a release: TS/TSX trace resolution
(< 0.70 → build source-map support), undetected-move rate, header-only cache-miss
share, seed overflow rate.

## §9 Production concerns

### §9.1 Tenancy

One database (Community has one), `repo_id` on every node and leading every
index. Every query is scoped by it; the vector index cannot be, so its fetch
widens (F-028).

### §9.2 Secret scrubbing

Known key shapes, credentials in URLs and secret-named literals are redacted from
every stored or embedded field before hashing. It will not catch a secret shaped
like ordinary text.

### §9.3 Degradation

No embedder: fulltext and graph only, said in the answer. No model: the ranked
code blocks, no prose. A failed flush is re-queued. Nothing falls back silently.

### §9.4 Observability

Counters (`metrics.py`) for every refusal, fallback and retry, so a degraded path
is a number rather than a mystery.

### §9.5 Caching

The embedding cache (§4.2) is the cost control: an unchanged symbol is never
re-embedded.

## §10 Plan

Built in ten gated steps (*Build plan* below); a step's tests and the full
regression had to pass before the next began.

### §10.1 Cut ladder

What was cut to ship: the editor extension (the watcher stands in), a Next.js
console (one static page stands in), source maps.

### §10.2 Triggers

When a decision metric (§8.3) crosses its threshold, the deferred work it names
is next — e.g. TS/TSX trace resolution at 0.33 says source maps are.

## §11 Cypher

### §11.1 Bounded neighbourhood expansion

One statement: for each seed, its neighbours over the allowlisted edge types,
hubs skipped, a per-seed limit, scored by weighted support over `log(2 + degree)`.
Uses the `CALL (seed) { … }` scope clause (R7).

### §11.2 Remap and reconciliation

The write sequence: 1–2 remap moved UIDs, 3a upsert files and symbols (vectors
merged in first, T5), 3b resolve and write edges one statement per type (`MERGE`
takes one type, B1), 4 delete stale edges, 5 delete orphans (after noting their
callers in other files), 6a–6b refresh degrees. Batched in the driver — no
transaction subqueries after a write (B2).

### §11.3 Schema

Uniqueness on `(repo_id, uid)`, lookups on name, lines and origin (all leading
with `repo_id`, T7), a fulltext index on search text, a 1,536-dimension cosine
vector index (local 768-dimension vectors are zero-padded, F-025).

## §12 Known limitations

### §12.0 Source code leaves the machine

With a hosted provider configured, packed code is sent to it. Local models keep
everything on the machine (decision 0001, 0003).

### §12.1 Neo4j is a correlated failure domain

Graph, fulltext and vector search are one process; if it is down, so is SEI.

### §12.3 String-based framework references are invisible

`include("authx.urls")`, `ForeignKey("app.Model")`, task names in strings: no
edge.

### §12.4 The trust model rests on a badge

The checks are cheap and syntactic; a cited, supported sentence can still be
wrong. The reader is shown the code behind every citation to judge.

### §12.5 Weakest on the code that needs it most

Undocumented, badly named code is the hardest to retrieve by meaning.

### §12.8 Degree is a proxy

`degree` plus seed support approximates importance; it is not centrality.

### §12.9 Coverage depends on a heuristic classifier

T2's claim classifier is measured on the author's own sentences, not model
output.

### §12.10 Every constant is a starting point

All † constants are guesses until measured on real repositories.

### §12.11 Seam cost is not uniform

Swapping the embedding model is a full re-embed (one model per repository,
F-025); swapping the answer model is free.

## §14 Standing

The fixtures are small and written for this project. The numbers in the README
are about them, not about real codebases.

---

## Build plan

### Plan §1 Ground truth

Probe Neo4j's behaviour and record it before building on it.

### Plan §2 UID and the Python adapter

Symbols, arity, ordinals; the UID contract (§3.2).

### Plan §3 Graph writer and epoch reconciliation

The §11.2 sequence, crash injection at every step, idempotent replay.

### Plan §4 Chunker and embedding cache

§3.4 and §4.2; vectors reach the nodes (T5).

### Plan §5 Indexer

Debounce, moves, full reconcile (§4.1, §4.4, §4.5).

### Plan §6 Retrieval

Tokenizer, hybrid search, expansion (§5.2, §5.3, §11.1).

### Plan §7 Router and frame resolution

§5.1.

### Plan §8 Context packer

§5.4.

### Plan §9 Synthesis and grounding

The first end-to-end answer (§6). A CLI was enough for the gate.

### Plan §10 TypeScript adapter, evaluation harness, gates

§3.5 for TypeScript, §8.

---

## Findings

Where running the system contradicted its design. Each was fixed or is listed
as open.

### F-001 One database on Community

The tests could not create a database per module; they wipe the one database
instead — now a separate test server, with a guard (B4 in the review).

### F-002 TSX overloads are `function_signature`

Overload declarations parse as signatures, not declarations; handled.

### F-003 Model cost against the budget

At the time, every hosted model exceeded the cost break-even at the context
ceiling; the lever was to measure p50 packed tokens rather than shrink context.

### F-004 `:File` nodes

The write sequence never wrote `:File` nodes; it does, and moves them.

### F-005 Overflow parts have no vector

A split symbol embeds only its first part (§3.4.3). Open.

### F-006 Stand-in tokenizer

The packer counts with an approximation; measured to over-count the local model
by 0.52–0.94 (F-025). Open for hosted models.

### F-007 Qualified names do not survive a move

Remaps derived old UIDs from the new path's names and matched nothing; fixed by
re-parsing the old bytes (R1).

### F-008 Signal B blocked the event loop

The git call now runs off the loop.

### F-009 Recall needs a real embedder

The hash stand-in has no meaning; measured with the local embedder: 0.933 →
0.978.

### F-010 Framework edges unwritable

Four of the eight §3.5 types could not be written or traversed (R2).

### F-011 `symbol_origin` could not serve steps 4 and 6

Statements constrained a prefix of a composite index, which Neo4j will not use
(R3).

### F-012 Undetected-move criterion

Had F-007's defect; fixed with it.

### F-013 `EXPLAIN` on an empty database

Not evidence of an index plan; the plan tests run on populated data.

### F-014 Deprecated `CALL { WITH … }`

Replaced by `CALL (seed) { … }` (R7).

### F-015 Chunked reconcile lost edges

Edges into later chunks were dropped or trimmed onto a package module (R4).

### F-016 Unchanged callers were never re-linked

Even by Sync Project (R5).

### F-017 A failed flush lost its batch

Now re-queued (R6).

### F-018 Path normalisation stripped characters

`lstrip("./")` turned `.github/x.py` into `github/x.py`.

### F-019 Live badges disagreed with the report

Streaming and batch classification now share one segmenter.

### F-020 TTFT and cost wrong on retried answers

Fixed; both count every generation.

### F-021 The router's decision was ignored

STRUCTURAL and TRACE were unreachable until the router drove retrieval.

### F-022 Repository hygiene

Requirements, generated files and test isolation cleaned up.

### F-023 The SDK rejected `temperature=`

The installed SDK removed the keyword; sent through `extra_body`.

### F-024 Django edges

`DISPATCHES_TO`, `HAS_FIELD`, `USES_SERIALIZER` emitted; field-level nodes do
not exist, so `HAS_FIELD` links model to model.

### F-025 Vector width and one embedder per repository

Neo4j silently leaves a wrong-length vector out of the index; local vectors are
zero-padded to 1,536, and a repository's embedder is recorded and enforced.

### F-026 The small model ignores the citation instruction

A format reminder after the context lifts Coverage from 0.22 to 0.85–0.92.

### F-027 TypeScript frames by name

0 of 9 resolved: components were ambiguous with their own module, `Class.method`
matched nothing, `async`/`new`/Windows frames did not parse. Now 3 of 9; the rest
are correct refusals.

### F-028 Vector search crowded out by another repository

A fixed over-fetch returned 0–11 of 60 rows beside a larger repository; the
fetch now widens.

### F-029 First real runs of the local system

A watcher snapshot that advanced before delivery, a cold-load timeout, a
PowerShell abort, "100% cited" beside an invented handle, a `/status` race.

---

## Decisions

### Decision 0001 — who may see the source

The design must work with no source leaving the machine; a hosted provider is
the user's explicit choice (by setting its key).

### Decision 0002 — the hosted models

`claude-haiku-4-5` for answers and `voyage-code-2` (1,536 dimensions, which the
vector index is sized for) for embeddings. Never called in testing.

### Decision 0003 — local models

`qwen2.5-coder:3b` and `nomic-embed-text` through Ollama, chosen to fit a 4 GB
GPU; the window becomes a hard budget; a format reminder for the small model;
one-command start and stop.

---

## Spec revisions

### R1 Remap UIDs from the old bytes (v10.1)

### R2 Every framework edge type writable and traversed (v10.1)

### R3 Constrain every column of `symbol_origin` (v10.1)

### R4 A chunked reconcile replays earlier chunks' edges (v10.2)

### R5 Sync Project re-parses every file (v10.2)

### R6 A failed flush is re-queued (v10.2)

### R7 The `CALL (seed) { … }` scope clause (v10.2)

---

## Defect codes from earlier design rounds

### T1 UIDs collided within a file — fixed by `ordinal`

### T2 Arity was undefined — defined: declared parameters, excluding `self`/`cls`, `*args`, `**kwargs`, rest

### T3 `full_reconcile` was used and never specified — specified (§4.5)

### T4 The undetected-move rate needed stored state — it falls out of the reconcile diff

### T5 Nothing wrote `code_vec` — vectors are merged into the props before the upsert

### T6 Symbols and chunks were two collections — one

### T7 Two indexes lacked `repo_id` — added

### T8 Anonymous default exports had no name — `<module>.default`; lambdas are not symbols

### T9 `vec_epoch` was written by nothing — reserved

### T10 Tier 2 frame resolution guessed — more than one candidate is a refusal

### T11 A base-class rename did not invalidate the cache — the header hash covers the class signature

### B1 `MERGE` with several relationship types is a syntax error — one statement per type

### B2 Transaction subqueries cannot follow a write — batching happens in the driver

### B5 Rename chains — `A→B→C` in one window is one remap

### B8 Unindexed label scans — every hot statement is index-served, checked with `EXPLAIN`

### B10 A moved file that cannot be read — counted, degraded to an undetected move

### B15 False atomicity — the sequence is crash-safe by ordering, not a transaction

### v9.1 #9 Citation handles inside code fences — not citations, in one shared helper

### v9.1 #12 The packer's budget — every block appended must also add its cost

### v9.1 #13 Degrees — refreshed for the batch's symbols and their neighbours

### v9.1 #14 A stale pending entry — a moved path's pending parse is evicted, or the old UID is recreated

### v9.1 #16 Moved-from paths stay in the batch — or an orphan and its edges are never collected

### v9.1 #17 Signal B's new path — parsed, since nothing else has parsed it
