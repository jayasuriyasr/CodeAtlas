# SEI — Implementation Plan

**Spec:** `SEI_Architecture_v10.0.md` (frozen baseline)
**Method:** 10 steps. Each ends with its own tests plus every prior step's tests re-run. No step starts until the previous step's gate is green.
**Budget:** ~48 engineer-days core (≈9.5 eng-weeks) + slack → the 12 eng-week figure in spec §10.

---

## 0. Working Rules

1. **A step is done when its gate passes, not when its code is written.** The gate is: step tests green **and** full regression green **and** acceptance criteria met.
2. **Never edit the spec in place.** Findings go to `FINDINGS.md`. The spec gets a versioned revision only when a finding forces a design change — and only with the run that produced it cited.
3. **Every † constant stays a variable in config**, never a literal in code. They are guesses (spec §12.10) and will move.
4. **Regression is cumulative.** After step N, `pytest tests/` runs steps 1..N. A red test from step 3 while working on step 7 is a step-7 bug until proven otherwise.
5. **Update `PROGRESS.md` at each gate.** It is the resume point — see §13.

### Repo layout

```
sei/
  adapters/      python.py  typescript.py  base.py        # S3
  graph/         writer.py  reader.py  schema.cypher      # S6
  index/         indexer.py  chunker.py  cache.py  moves.py
  retrieve/      router.py  hybrid.py  expand.py  frames.py
  pack/          packer.py  render.py
  ground/        classify.py  validate.py
  api/           app.py  stream.py
  eval/          harness.py  golden/                      # S7
  config.py                                               # all † constants
tests/
  step_01_ground_truth/ … step_10_polyglot_eval/
  fixtures/      repos/  traces/
FINDINGS.md
PROGRESS.md
```

---

## 1. Ground Truth — resolve the unknowns first *(2 days)*

Spec §0.2 lists four things no amount of reading settles. Nothing else starts until they are facts.

**Build**
- Docker Compose: Neo4j 5.x (pin the exact minor), record the version in `PROGRESS.md`
- Apply `schema.cypher` (spec §11.3) end to end
- A throwaway-DB pytest fixture: fresh database per test module
- Decision record answering spec §12.0 / §12.1 — target market and source egress. **These are one question**; the answer determines whether self-hosted embeddings are Phase 0 scope.
- Name the model + embedding model + version. Check spec §5.4's $0.94/Mtok break-even against the actual rate card.

**Probes** — one statement each, pass or fail recorded verbatim

| Probe | Question |
|---|---|
| `probe_dynamic_map_key` | Does `$edge_weights[rel]` resolve a dynamic key on a map parameter? |
| `probe_count_subquery` | Is `COUNT { (n)--() }` available on this minor? |
| `probe_exists_subquery` | Is `NOT EXISTS { MATCH (c:Symbol {uid: $u}) }` available? |
| `probe_vector_ddl` | Does the vector index take `ON (s.code_vec)` or `ON s.code_vec`? |
| `probe_merge_multitype` | Confirm `MERGE (a)-[:A\|B]->(b)` is rejected — the per-type split in §11.2 3b is required, not defensive |
| `probe_tsx_grammar` | Does `tree-sitter-typescript` (tsx) emit the node types the chunker assumes for a real Next.js route + component? |

**Gate**
- [ ] All six probes recorded with actual output, not expected output
- [ ] `schema.cypher` applies clean on a fresh DB and is idempotent (run twice)
- [ ] Break-even check done; if the chosen model exceeds ~$0.94/Mtok, pick a lever from spec §5.4 **now** and record it
- [ ] Market decision recorded

> If `probe_vector_ddl` or `probe_tsx_grammar` fails, stop and resolve. Everything downstream assumes them.

---

## 2. UID + Python Adapter *(5 days)*

**Build**
- `symbol_uid(repo_id, rel_path, qualified_name, arity, ordinal)` (spec §3.2)
- Arity rule: Python counts positional, keyword-only, and defaulted params; excludes `self`, `cls`, `*args`, `**kwargs`
- Ordinal: source-order index among same-file symbols sharing `(qualified_name, arity)`
- `adapters/python.py` → `ParsedFile{symbols: [Symbol], edges: [Edge], content_hash}`
- Qualified names incl. `<module>.default`; anonymous functions are **not** symbols

**Tests** `tests/step_02_uid_adapter/`

| Test | Guards |
|---|---|
| `test_overload_uids_distinct` | **T1** — conditionally-defined same-name/same-arity functions get distinct UIDs |
| `test_arity_excludes_self_and_varargs` | T2 |
| `test_default_value_change_preserves_uid` | T2 — `timeout=30` → `timeout=60` must not churn |
| `test_parse_is_deterministic` | Parse twice, byte-identical UID set |
| `test_anonymous_lambda_not_a_symbol` | T8 |
| `test_line_move_preserves_uid` | Adding a line above a function must not change its UID |

**Gate**
- [ ] Step 2 tests green
- [ ] Regression: step 1 green
- [ ] Parse a real Django repo (fixture): zero UID collisions across the whole tree

---

## 3. GraphWriter + Epoch Reconciliation *(5 days)*

**Build**
- `graph/writer.py`: spec §11.2 steps 2a–6b, per-type edge MERGE from the fixed allowlist
- `batch_paths = batch.keys() ∪ {move.old_path}`
- `reader.py`: `known_paths`, `file_hashes`, `body_hashes`, `symbols_named`

**Tests** `tests/step_03_writer/`

| Test | Guards |
|---|---|
| `test_edge_merge_per_type` | **B1** — all four types write correctly via separate statements |
| `test_no_transaction_subquery_after_write` | B2 — the statement set actually executes |
| `test_own_degree_refreshed` | **v9.1 #13** — step 6a; a new symbol must not carry null `degree` |
| `test_neighbor_degree_refreshed` | Step 6b |
| `test_superset_on_prefix` | **Crash injection** at each of the 6 boundaries → assert zero dangling references, stale rows allowed |
| `test_idempotent_replay` | Re-run full sequence, same epoch → identical graph |
| `test_stale_edges_deleted_scoped` | Edges from *unchanged* files survive |
| `test_repo_isolation` | Two repos, same paths, zero cross-contamination |

**Gate**
- [ ] Step 3 tests green · regression 1–2 green
- [ ] `EXPLAIN` on §11.2 steps 4/5/6 shows `symbol_origin` index usage, not `NodeByLabelScan`

---

## 4. Chunker + Embedding Cache *(4 days)*

**Build**
- Symbol-scoped context header (spec §3.4) — per-symbol resolved imports, **not** the file import block
- `enclosing_signature` incl. base classes
- `cache_key` = `header_hash:body_hash`; SQLite `EmbeddingCache` (S5)
- `embed_batch` → **`GraphWriter.apply` merges vectors into `sym.props` before step 3a**

**Tests** `tests/step_04_chunk_cache/`

| Test | Guards |
|---|---|
| `test_vectors_land_on_nodes` | **T5** — query `s.code_vec IS NOT NULL` for every symbol after a write |
| `test_added_import_does_not_invalidate` | **The original cache defect** — add `import logging`, assert ≥90% hit rate |
| `test_base_class_rename_invalidates` | **T11 (v9.1)** — `class X(A)` → `class X(B)` must produce a cache miss |
| `test_comment_change_invalidates` | Comments are in `body_hash` by design |
| `test_trailing_whitespace_does_not_invalidate` | `_normalize` |
| `test_scrub_before_hash` | Spec §9.2 ordering |
| `test_overflow_split_at_statement_boundary` | Never mid-block |

**Gate**
- [ ] Step 4 green · regression 1–3 green
- [ ] Full index of the Django fixture: record cache hit rate on a second run (expect ~100%) and on a one-import edit (expect ≥90%)

---

## 5. Indexer — Debounce, Moves, Full Reconcile *(5 days)*

**Build**
- `Indexer` per spec §4.1: per-repo mutex, `_timers`, `_indexed` seeded at `startup`
- `collect_moves` (drain → collapse → evict → parse), `_collapse_chains`, `resolve_moves`
- `full_reconcile` (spec §4.5) incl. the undetected-move counter from the diff
- *(Cut-ladder)* VS Code plugin → `POST /index/rename`. JetBrains is cut step 2.

**Tests** `tests/step_05_indexer/` — the three move-survival fixtures plus the traps

| Test | Guards |
|---|---|
| `test_move_preserves_inbound_edges` | Fixture (a) — `git mv`, ≥5 callers, count unchanged |
| `test_bulk_rename_preserves_edges` | Fixture (b) — **v8.2 flush defect**, >`BULK_THRESHOLD` files |
| `test_rename_chain_single_remap` | Fixture (c) — **B5** A→B→C: one net remap, `move.unresolvable == 0` |
| `test_save_then_rename_no_duplicate` | **v9.1 #14** — stale pending entry must not recreate the retired UID |
| `test_remap_collision_cleans_orphan` | **v9.1 #16** — `batch_paths` includes `old_path` |
| `test_signal_b_move_gets_parsed` | **v9.1 #17** — `git mv` with no IDE event still resolves |
| `test_new_path_unreadable_degrades` | B10 — no crash, counter increments |
| `test_debounce_coalesces` | 8 saves in 2s → one flush, one epoch |
| `test_concurrent_repos_dont_block` | Per-repo lock |
| `test_reconcile_resumable` | Kill mid-reconcile, re-run, converges |

**Gate**
- [ ] Step 5 green · regression 1–4 green
- [ ] `move.unresolvable` is zero across the entire suite
- [ ] Watch step 3: `batch_paths` is where step 5 most often breaks step 3

---

## 6. Retrieval — Tokenizer, Hybrid, Expansion *(5 days)*

**Build**
- `split_identifier` / `build_search_text` (spec §5.3), applied **query-side too**
- Hybrid RRF over vector ∥ fulltext → top-30†
- 1-hop expansion with hub cutoff and `seed_support / log(2+degree)` scoring (§11.1)
- **Start the golden set here:** ~15 retrieval-only questions with gold UIDs, no reference answers. Recall@10 becomes measurable four steps early.

**Tests** `tests/step_06_retrieval/`

| Test | Guards |
|---|---|
| `test_snake_and_camel_cross_match` | `get_user_by_id` ↔ `getUserById` normalize identically |
| `test_exact_identifier_still_matches` | Exact form preserved |
| `test_query_side_tokenized` | The invisible half-failure |
| `test_no_stopword_loss` | `if`, `not`, `for` survive |
| `test_hub_cutoff_excludes_logger` | 400-caller utility does not flood expansion |
| `test_expansion_returns_inbound_callers` | The reason the graph exists |
| `test_rrf_merges_both_arms` | Vector-only and fulltext-only hits both surface |
| `test_recall_at_10_baseline` | Records, does not gate yet |

**Gate**
- [ ] Step 6 green · regression 1–5 green
- [ ] Recall@10 recorded on the 15 seed questions. **If it is far off 0.80†, that is data about the constant, not necessarily a bug** — log to `FINDINGS.md`

---

## 7. Router + Frame Resolution *(3 days)*

**Build**
- Regex router → STRUCTURAL / SEMANTIC / TRACE, LLM fallback
- Tier 0 Python direct; Tier 2 name match **with ambiguity refusal**; Tier 3 SEMANTIC fallback
- Per-frame tier badge in the response payload

**Tests** `tests/step_07_router_frames/`

| Test | Guards |
|---|---|
| `test_tier0_python_traceback` | Exact resolution |
| `test_tier2_ambiguous_refuses` | **T10** — two functions named `GET` → returns `None`, never a guess |
| `test_tier2_single_candidate_resolves` | The valid case still works |
| `test_mangled_name_skipped` | `at t (…)`, `at <anonymous>` |
| `test_tier3_fallback_uses_neighbor_frames` | Degradation path |
| `test_router_classifies_structural` | "what calls X" does not go to ANN |

**Gate**
- [ ] Step 7 green · regression 1–6 green
- [ ] `trace.tier2.ambiguous` fires on the App Router fixture (proves the refusal path is reachable, not dead code)

---

## 8. Context Packer *(3 days)*

**Build**
- Real tokenizer via S2 — **never `len // 4`**
- L0 / L1 (tree-sitter nested-body elision) / L3 stub
- Greedy by score, `MAX_SEED_FRAC` cap, marked truncation for a pathological top seed
- Handles assigned **after** packing; packing report

**Tests** `tests/step_08_packer/`

| Test | Guards |
|---|---|
| `test_budget_never_exceeded` | **v9.1 #12** — including the truncation branch's `used += t.cost` |
| `test_truncation_accounting` | Pathological top seed then 5 more: total stays under cap |
| `test_handles_contiguous_after_drops` | No gaps for the model to cite into |
| `test_elision_marker_present` | Never silent |
| `test_no_mid_line_split` | Statement boundaries only |
| `test_l1_saves_budget` | 4,000-token file elides to fit |
| `test_counters_distinct` | `seed_l1_overflow` vs `seed_truncated` |

**Gate**
- [ ] Step 8 green · regression 1–7 green
- [ ] Fuzz: 500 random seed/neighbor size distributions, budget never breached

---

## 9. Synthesis + Grounding — first end-to-end answer *(6 days)*

**Build**
- FastAPI + SSE streaming
- Citation-first prompt; handle format `[C<n>]`, parsed **outside fenced blocks only**
- Claim-sentence classifier (spec §6.1) — conservative by default
- T1 / T2 inline, T3 batched; Coverage / RelationPrec / ConflictFlags; one bounded retry
- *(Cut-ladder)* Next.js console. CLI is sufficient to gate this step.

**Tests** `tests/step_09_grounding/`

| Test | Guards |
|---|---|
| `test_handle_in_code_fence_ignored` | **v9.1 #9** — `arr[C1]` in a snippet is not a citation |
| `test_hallucinated_handle_caught` | `[C12]` with 9 blocks → T1 flags |
| `test_uncited_claim_flagged` | T2 |
| `test_hedged_sentence_not_a_claim` | Do not punish declared uncertainty |
| `test_interrogative_not_a_claim` | Classifier exclusions |
| `test_relation_verified_batched` | T3, one query |
| `test_retry_promotes_dropped_first` | Spec §6.3 ordering |
| `test_retry_runs_once_only` | Never a loop |
| `test_classifier_precision_recall` | Against hand-annotated fixtures |

**Gate**
- [ ] Step 9 green · regression 1–8 green
- [ ] **A real question about the Django fixture returns a cited, streamed answer.** This is the first moment the system exists.
- [ ] Record TTFT and $/query; compare against the step-1 break-even

---

## 10. TS/TSX Adapter + Eval Harness + Gates *(10 days)*

The largest single unknown (spec §10). Deliberately last: if S3 is a real seam, adding a language is additive. **If it is not, that is a finding about the architecture** — record it, do not paper over it.

**Build**
- `adapters/typescript.py`: tsx parsing, TS arity rule (optional counted, rest excluded)
- Framework layer: `INVOKES_HOOK`, `is_client_component`, App Router `RENDERS`
- Golden set to 45 (15/class) with claim-sentence annotations and the §8.1 composition requirements
- All ten gates in CI + four decision metrics + the ~7 lines of counters

**Tests** `tests/step_10_polyglot_eval/`

| Test | Guards |
|---|---|
| `test_ts_overload_uids_distinct` | **T1 in its original habitat** — three `parse()` overloads |
| `test_ts_arity_optional_and_rest` | T2 for TS |
| `test_default_export_qualified_name` | T8 |
| `test_app_router_get_handlers_collide_by_name` | Confirms step 7's refusal is load-bearing |
| `test_use_client_directive_detected` | Framework layer |
| `test_all_gates_computable` | Every §8.2 metric produces a number |
| `test_gate_resolution_documented` | ~6.7pp per question surfaced in the report |

**Gate**
- [ ] Step 10 green · **full regression 1–9 green**
- [ ] All ten gates produce values; record which pass and which do not
- [ ] Four decision metrics recorded — these choose Phase 1 work (spec §10.2)
- [ ] Re-tune † constants against measured data; revise the spec **once**, citing runs

---

## 11. Regression Watch List

Where later steps most often break earlier ones, and why:

| Working on | Watch | Because |
|---|---|---|
| 3 Writer | 2 UID | Property changes alter `sym.props` and can silently change identity inputs |
| 4 Cache | 2 UID, 3 Writer | Header content feeds `cache_key`; vector merge feeds step 3a |
| 5 Indexer | 3 Writer | `batch_paths` scoping is the single most coupled interface in the system |
| 6 Retrieval | 4 Cache | Null `code_vec` (T5) makes vector search silently return nothing |
| 7 Frames | 3 Writer | Tier 2 needs `(repo_id, name)` index; Tier 0 needs `(repo_id, rel_path, start_line)` |
| 8 Packer | 6 Retrieval | Handle assignment order depends on what expansion returns |
| 9 Grounding | 8 Packer | Coverage denominators shift if packing drops blocks |
| 10 TS adapter | 2 UID, 7 Frames | Different arity rules and heavy name collisions |

**Rule:** a prior-step test that goes red is a current-step bug until proven otherwise. Do not edit the older test to make it pass.

---

## 12. Defect → Test Map

Twenty-two defects were found across v8.0–v10.0 **by reading**. Each now has an owning test. This table is the argument for the whole plan.

| Defect | Test | Step |
|---|---|---|
| T1 UID collision (overloads) | `test_overload_uids_distinct`, `test_ts_overload_uids_distinct` | 2, 10 |
| T2 arity undefined | `test_arity_*`, `test_default_value_change_preserves_uid` | 2 |
| T3/T4 `full_reconcile` unspecified | `test_reconcile_resumable` | 5 |
| T5 `code_vec` never written | `test_vectors_land_on_nodes` | 4 |
| T7 indexes lack `repo_id` | `test_repo_isolation` + `EXPLAIN` check | 3 |
| B1 multi-type MERGE | `probe_merge_multitype`, `test_edge_merge_per_type` | 1, 3 |
| B2 txn subquery after write | `test_no_transaction_subquery_after_write` | 3 |
| B3 vector DDL syntax | `probe_vector_ddl` | 1 |
| B5/B6 rename chains | `test_rename_chain_single_remap` | 5 |
| B8 unindexed scans | `EXPLAIN` gate | 3 |
| B15 false atomicity | `test_superset_on_prefix`, `test_idempotent_replay` | 3 |
| v9.1 #10 Tier 2 ambiguity | `test_tier2_ambiguous_refuses` | 7 |
| v9.1 #11 base-class rename | `test_base_class_rename_invalidates` | 4 |
| v9.1 #12 packer accounting | `test_budget_never_exceeded` | 8 |
| v9.1 #13 own degree | `test_own_degree_refreshed` | 3 |
| v9.1 #14 stale pending | `test_save_then_rename_no_duplicate` | 5 |
| v9.1 #16 `batch_paths` | `test_remap_collision_cleans_orphan` | 5 |
| v9.1 #17 Signal B unparsed | `test_signal_b_move_gets_parsed` | 5 |
| v8.2 bulk flush drops moves | `test_bulk_rename_preserves_edges` | 5 |
| Original cache defect | `test_added_import_does_not_invalidate` | 4 |
| Handle-in-code-fence | `test_handle_in_code_fence_ignored` | 9 |
| BM25 identifier splitting | `test_snake_and_camel_cross_match` | 6 |

---

## 13. PROGRESS.md Template

Keep this current at every gate. It is the resume point for anyone — or any assistant — picking the project up cold.

```markdown
# SEI Progress

Spec: v10.0 (frozen)   Last gate passed: Step __   Date: ____

## Environment
Neo4j version: ____        Python: ____
LLM model + version: ____  Embedding model + dims: ____
Break-even check (§5.4): input rate ____ /Mtok · lever chosen: ____

## Step 1 probe results  (verbatim output, not expectations)
dynamic_map_key: ____   count_subquery: ____   exists_subquery: ____
vector_ddl:      ____   merge_multitype: ____  tsx_grammar: ____

## Decisions
Target market / source egress (§12.0): ____
Self-hosted embeddings in Phase 0? ____
Cut-ladder steps applied: ____

## Gate status
| Step | Tests | Regression | Acceptance | Date |
|------|-------|------------|------------|------|
| 1    |       |            |            |      |
| …    |       |            |            |      |

## Measured constants (replacing † guesses)
hub_cutoff: ____   BULK_THRESHOLD: ____   SEED_SHARE: ____
chunk cap: ____    coverage threshold: ____  MAX_SEED_FRAC: ____

## Latest metrics
Recall@10 ____  Coverage ____  cache hit ____  TTFT p50/p95 ____  $/query p50 ____
Decision metrics: TRACE ____  undetected-move ____  header-only miss ____  L1-overflow ____

## Open findings
(from FINDINGS.md — anything contradicting the spec)

## Next
Step __: ____
```

---

## 14. Milestones

| After | You have |
|---|---|
| Step 1 | Facts instead of assumptions — the four unknowns closed |
| Step 3 | A graph that survives crashes and re-runs |
| Step 5 | An indexer safe to point at a repo people are actively editing |
| Step 6 | Measurable retrieval quality, four steps before the full harness |
| **Step 9** | **A working product** — a real question, a cited streamed answer |
| Step 10 | Two languages, ten gates, and data to choose Phase 1 from |

Steps 1–9 are Python-only and end in something usable. If the schedule collapses, that is the thing to protect — and it is the same conclusion the cut ladder reaches from the other direction.
