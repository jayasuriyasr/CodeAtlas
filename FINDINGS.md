# FINDINGS

Where reality contradicted a document. One entry per finding, each citing the
command that produced it and its actual output.

Two kinds live here, labelled so they are not confused:

- **Contradicts the spec** — `SEI_Architecture_v10.0.md` says X, a run showed
  not-X. These are the ones that can authorise a spec revision, and only with
  my explicit approval and the run cited.
- **Contradicts a working assumption** — the spec is silent or already correct,
  but `CLAUDE.md`, the plan, or an implementation assumption was wrong.

---

## F-001 — "A throwaway database per test module" is not available on Community

**Kind:** contradicts a working assumption (`CLAUDE.md` § Testing).
**Found:** step 1, writing `tests/conftest.py`.
**Status:** accepted, mitigated.

`CLAUDE.md` says "Neo4j tests use a throwaway database per module." Neo4j
Community Edition has no `CREATE DATABASE`: it serves exactly one user
database. The spec already says so, twice, so this is not a spec defect —

> §9.1: "Neo4j Community Edition supports a single user database.
> Database-per-tenant requires Enterprise — unavailability, not expense."
> §12.1: "Community Edition has no clustering and one user database."

— but the testing instruction was written as though the capability existed, and
decision [0001](docs/decisions/0001-target-market-and-source-egress.md) commits
us to Community for the MVP.

**Mitigation.** `throwaway_db` detects the edition through
`CALL dbms.components()` and takes one of two paths:

| Edition | Path | Isolation actually obtained |
|---|---|---|
| Enterprise | `CREATE OR REPLACE DATABASE … WAIT`, dropped on teardown | A separate database |
| Community | Drop all constraints/indexes, batched `DETACH DELETE`, re-apply §11.3 | Equivalent *between modules*; same physical database |

The fixture reports which mode ran, and `PROGRESS.md` records it, because "the
tests pass" means a slightly weaker thing under scoped-wipe.

**What this costs.** Test modules cannot run in parallel against one Community
instance — a second worker would wipe the first worker's graph mid-test. If the
suite ever needs `pytest -n`, that is the moment to either run one Neo4j per
worker or accept the Enterprise licence. Not a problem at step 4's suite size.

---

## F-002 — TSX overload declarations are `function_signature`, not `function_declaration`

**Kind:** contradicts a working assumption (the §3.2 ordinal's implementation).
**Found:** step 1, `probe_tsx_grammar`.
**Status:** recorded; binds step 10.

Spec §3.2 uses three same-arity `parse` declarations as the motivating example
for the UID ordinal, and writes all three as `function …`. The grammar does not
see them as one kind of node:

```
overload declarations of `parse`:
  [('function_signature', 28, 'parse'),
   ('function_signature', 29, 'parse'),
   ('function_declaration', 30, 'parse')]
```

*(`tests/step_01_ground_truth/test_probe_tsx_grammar.py`, run 2026-08-27; full
output in `PROBE_RESULTS.md`.)*

The two bodiless overload declarations parse as `function_signature`; only the
implementation is a `function_declaration`. An adapter that collects symbols by
matching `function_declaration` alone would find **one** `parse`, assign it
ordinal 0, and the T1 collision the ordinal exists to prevent would be invisible
— not because the UID scheme is wrong, but because two of the three symbols were
never presented to it.

**Binds:** step 10, `test_ts_overload_uids_distinct`. The TS adapter's
symbol-node allowlist must contain both `function_signature` and
`function_declaration`. Recorded now rather than at step 10 because the probe is
where the fact appeared.

No spec change: §3.2's rule is correct as written. This is an implementation
note about the parser, which is exactly what the probe was for.

---

## F-003 — Every current Claude model exceeds §5.4's break-even at the ceiling

**Kind:** contradicts the spec's implied headroom (§5.4).
**Found:** step 1, break-even check.
**Status:** accepted with a lever;
see [0002](docs/decisions/0002-model-selection-and-break-even.md).

§5.4 computes a break-even input rate of ~$0.94/Mtok and presents exceeding it
as a contingency: "if the chosen model's input rate exceeds ~$0.94/Mtok, the
§8.2 gate and this ceiling cannot both hold." Against Anthropic's card on
2026-08-27, the contingency is the only case available:

| Model | Input $/Mtok | vs. break-even |
|---|---|---|
| Claude Haiku 4.5 | $1.00 | 1.06× |
| Claude Sonnet 5 | $2.00 | 2.12× |
| Claude Opus 5 | $5.00 | 5.30× |

The spec anticipates this and lists levers, so the design is not contradicted —
what is contradicted is the implicit expectation that a model under $0.94 exists
to be chosen. The break-even is not a filter that narrows the field; it is a
constraint that binds for every option.

§5.4 also observes that actual cost is p50 packed tokens, not the ceiling, and
that at 9k the tension may not exist. That is the lever taken: instrument p50,
do not move `CTX_BUDGET` or the gate until step 8 produces a number. The
arithmetic is in decision 0002.

**Also recorded there:** §5.4's break-even explicitly ignores output tokens, and
at `RESERVED_OUT = 2_000` × $5.00/Mtok the ignored term is $0.010 — half the
entire $0.02 gate. The $0.94 figure is a floor on the input side alone, not a
budget for a query.

---

## F-004 — §11.2's write sequence never writes a `:File` node

**Kind:** contradicts the spec (§11.2 vs §3.3 / §4.1 / §4.5).
**Found:** step 3, implementing `GraphWriter.apply`.
**Status:** implemented around; a spec revision is warranted but not taken.

§3.3 defines the label:

> `:File { repo_id, rel_path, content_hash, epoch }`

and two control-flow paths read it back:

- §4.1 `Indexer.startup` — `self._indexed[repo] = await self.reader.known_paths(repo)`
- §4.5 `full_reconcile` — `known = await self.reader.file_hashes(repo)`, whose
  return type is documented in the same line as `rel_path -> content_hash`

§11.2's numbered sequence writes none of it. Step 3a is
`MERGE (s:Symbol {uid: sym.uid})` and there is no companion statement for
`:File`; steps 4 and 5 scope by `s.origin_path`, which is a Symbol property.

**Consequence if taken literally.** `known_paths` returns the empty set on every
start-up, so §4.1's Signal B condition —

```python
if not moves and any(p not in self._indexed[repo] for p in pending):
```

— is true for every batch, and the `git status --porcelain -M` subprocess spawn
that §4.1 explicitly gates ("costs a subprocess spawn (~50-200ms on a large
repo), so run it only when...") runs on every flush instead. `full_reconcile`
would classify every file as `added` on every pass, re-embedding the whole tree.

**Implemented.** `GraphWriter._upsert_files` MERGEs the `:File` row alongside
step 3a; `_delete_orphans` removes File rows whose `rel_path` is in
`batch_paths` and whose epoch is older; `remap_uids` moves the File row with its
symbols. Grouped with the writes so §4.3's superset-on-prefix ordering is
unchanged — every write still precedes every delete.

**Not treated as licence to revise the spec.** The frozen document stays as it
is; this records the gap and where it was filled.

---

## F-005 — An overflow-split symbol has nowhere to put its extra vectors

**Kind:** contradicts the spec (§3.4.3 vs §3.3 and §3.2).
**Found:** step 4, implementing `split_overflow`.
**Status:** first part embedded; the rest counted, not embedded.

§3.4.3 requires a split:

> **Overflow split** at >1200† tokens, on top-level statement boundaries,
> header repeated.

"Header repeated" says there are several parts. But §3.3 gives a symbol exactly
one `code_vec`, §3.2 makes the symbol the unit of identity, and §4.2's T6 note
is explicit that `ParsedFile.symbols` is the only collection — a symbol carries
its own `chunk_text`, singular. So the second and later parts of a split symbol
have no node to live on and no key to be cached under.

Three resolutions exist and the spec picks none: give the parts their own
synthetic UIDs (which reopens T1), store a vector list per symbol (which changes
the vector index), or embed only the first part.

**Implemented:** the first part becomes `chunk_text`; `chunk.overflow_split` and
`chunk.overflow_parts` count the rest. That makes the loss measurable rather
than silent, and the counter is the evidence a resolution would need.

**Scope of the loss.** A symbol over 1200† tokens is embedded on its opening
statements only, so semantic search can miss its tail. This lands hardest on
exactly the code §12.5 already names as the weak spot — long, poorly factored
functions. `CHUNK_OVERFLOW_TOKENS` is a † constant (§12.10 expects it to move),
so the frequency is not yet knowable.

---

## F-006 — The provider's real tokenizer is not wired, so §5.4 cannot be checked

**Kind:** contradicts a working assumption (plan §4 / §8's `never len // 4`).
**Found:** step 4, implementing the chunk cap.
**Status:** open. Blocks the §5.4 and §8.2 numbers, not the code.

§5.4 requires the provider's real tokenizer via S2 and says `len // 4` is off by
up to 2x on dense code. Neither chosen provider can supply one here:

| Provider | Real tokenizer | Why not now |
|---|---|---|
| `claude-haiku-4-5` | `messages.count_tokens` | network call, needs an API key, and sends source to a third party — which a unit test must not do |
| `voyage-code-2` | downloadable vocabulary | not installed; adding it is a dependency decision |

**Implemented:** the `Tokenizer` seam is real and every count goes through it.
`index/providers.ApproxCodeTokenizer` stands in — a genuine lexer over strings,
comments, identifiers, numbers, whitespace and operators, splitting long lexemes
into fixed-width pieces. It is **not** `len // 4`, and
`test_tokenizer_is_not_a_length_heuristic` holds it to that. Its `name` is
`approx-code-tokenizer(NOT-A-PROVIDER-TOKENIZER)` so a count from it is
traceable wherever it is reported.

**What stays unknown.** The error against the real tokenizer is unmeasured, in
an unknown direction. So `CHUNK_OVERFLOW_TOKENS` splits at the wrong place by an
unknown margin, and every §5.4 figure — p50 packed tokens, `$/query`, the
break-even in decision 0002 — is uncheckable until the real one is wired.

**Resolves at:** step 8, which needs the real tokenizer for the packer's budget,
and step 9, which configures the provider and its credentials. Recorded now
because step 4 is where the gap first bites.

---

## F-007 — `qualified_name` does not survive a move, so every remap is a silent no-op

**Kind:** contradicts the spec (§4.4 vs §3.2).
**Found:** step 5, `test_save_then_rename_no_duplicate`.
**Status:** **RESOLVED** — fixed in `resolve_moves`, and in the spec as
v10.1 §0.0 **R1**. v10.0 is left byte-identical.
**Severity:** this defeats the entire move-survival mechanism, with no symptom.

§4.4's `resolve_moves` computes both UIDs from the new file's parsed symbols:

> Old and new UIDs are both computable from the new file's parsed symbols,
> since `qualified_name`, `arity`, and `ordinal` survive a move.

Arity and ordinal survive. `qualified_name` does not. §3.2 defines it as
"Derived from the module path plus enclosing scopes", and the module path is
derived from `rel_path` — so the instant a file moves, every qualified name in
it changes. The two sentences cannot both hold.

**Actual output** (`scratchpad/f007.py`, 2026-08-27, moving `pkg/callee.py` to
`pkg/moved.py`):

```
stored at old path:
   uid=3d4232fba4933b8db4c6  qname='pkg.callee'
   uid=44a268a7ab97dd61b9ba  qname='pkg.callee.helper'
parsed at new path:
   uid=9229d0ce95e1a9c3e7f3  qname='pkg.moved'
   uid=74c6fda492a325c56de3  qname='pkg.moved.helper'

remaps resolve_moves produced:
   old_uid=af971382c022f87b4ed0 -> ...   MATCHES A STORED NODE: False
   old_uid=5e910bd02a800e4493a7 -> ...   MATCHES A STORED NODE: False

remaps that would match something in the graph: 0/2
```

Every `old_uid` is computed from the *new* qualified name against the *old*
path — a UID that was never written to the graph.

**Why it is worse than an ordinary bug.** §11.2 step 2a is a `MATCH`, so a
remap naming a nonexistent UID updates nothing and raises nothing. The upsert
then creates the symbols fresh at their new UIDs, step 5 detach-deletes the old
ones, and every inbound `CALLS` edge goes with them. That is precisely the
outcome §4.4 says is "worse than an error because it looks like an answer" —
reached by the mechanism built to prevent it. Nothing reports it:
`move.unresolvable` stays zero, because a `ParsedFile` *was* found.

**Fix.** `resolve_moves` re-parses the same bytes at the old path, which
reproduces the qualified names the graph actually stored, and pairs the two
symbol lists positionally (identical bytes through one adapter give identical
order — `test_parse_is_deterministic`). Cost: one extra local parse per moved
file, ~5ms, against §4's own budget.

Two deviations were needed and are flagged rather than buried:

1. `ParsedFile` gains `source: bytes`, so the old-path re-parse has the bytes.
   Negligible next to `Symbol.source_code`, which already holds the text of
   every symbol in the file.
2. `resolve_moves` gains a keyword-only `adapter_for` callable. It does not
   change S3's interface — `LanguageAdapter` still has `parse` and
   `resolve_frame` only.

**Guarded by** `test_the_spec_formula_would_have_matched_nothing`, which
demonstrates the failure rather than asserting about it — the fix is easy to
un-fix, and §4.4's one-liner will keep looking like a simplification.

**This one warrants a spec revision.** Not taken: the spec stays frozen, and a
revision needs explicit approval with the run cited (`CLAUDE.md` rule 1). The
run is above.

---

## F-008 — Signal B's subprocess runs on the event loop, so one repo stalls all of them

**Kind:** contradicts a working assumption (§4.1's per-repo mutex).
**Found:** step 5, chasing a flaky `test_concurrent_repos_dont_block`.
**Status:** fixed by making `collect_moves` async.

§4.1 makes the flush mutex per-repo and comments it as such —
`async with self._locks[repo]:  # per-repo, not global`. The intent is that one
repository's flush cannot stall another's. But the same section calls
`collect_moves` synchronously inside `_flush`, and `collect_moves` spawns
`git status --porcelain -M`, which §4.1 itself prices at 50-200ms.

A blocking subprocess on an asyncio event loop stalls *every* coroutine, not
just the ones holding that lock. The mutex is correct and insufficient: it
prevents lock contention, not loop starvation.

**Measured** (`scratchpad/gitcost.py`, 2026-08-27, 400-file repository):

```
git status --porcelain -M on 400 files: min=65ms median=67ms max=73ms
```

67ms on a repository small enough to be a rounding error. §4.1's own 50-200ms
estimate is for a large one.

**How it surfaced.** `test_concurrent_repos_dont_block` passed in isolation and
failed under full-suite load — the classic shape of a real timing dependency
rather than a bad assertion. The test now waits on an event rather than a
sleep, so it measures the property instead of the machine.

**Fix.** `collect_moves` becomes `async def` and awaits the git call in a
thread. Only the subprocess moves; every mutation of `_pending` stays on the
loop thread, so there is no race with `on_save`. This is a deviation from
§4.1's `def collect_moves`, and the only one.

**Why it matters more later than now.** Step 9 puts FastAPI in this process.
Left unfixed, every flush that consults git would add its full subprocess
latency to the TTFT of any request in flight — against a §8.2 gate of
1.2s p50.

---

## F-009 — Recall@10 is not measurable without a real embedding provider

**Kind:** contradicts a working assumption (plan §6's gate).
**Found:** step 6, building the recall baseline.
**Status:** open. Blocks step 6's gate independently of Neo4j.

Plan §6's gate is "Recall@10 recorded on the 15 seed questions", with the
sensible caveat that a number far off 0.80† is "data about the constant, not
necessarily a bug". Two things have to be true before that reading is available,
and only one of them is about Neo4j.

`HashEmbeddingProvider` maps text to a deterministic hash of itself. It has no
semantic structure at all — that is what makes it safe to use in tests, since it
needs no network and sends no source anywhere. But it means a query vector bears
no relation to the vector of a symbol that answers the query. The vector arm
returns an arbitrary ordering, and RRF fuses that arbitrary ordering with the
fulltext arm's real one.

**So a blended Recall@10 measured today would be lower than fulltext alone**, and
would be reporting the stand-in provider's absence of semantics as a retrieval
result. Fusing noise with signal cannot help; it can only push gold documents
down the list.

**What is recorded instead.** `test_recall_at_10_baseline` computes and writes
two figures to `tests/step_06_retrieval/RECALL_BASELINE.md`:

- **fulltext arm only** — a real number. BM25 over `search_text` is a genuine
  retrieval system, and this is its recall on the seed set.
- **blended** — computed, labelled, and explicitly marked as not a retrieval
  result.

The test asserts neither against a threshold. Plan §6 says step 6 records rather
than gates, and a threshold asserted here would be a threshold asserted against
`HashEmbeddingProvider`.

**Compounding it:** §8.1 measures recall against **packed** context, not
retrieved candidates — "a gold node retrieved and then dropped by the packer
never reached the model". The seed set measures the easier quantity. `stage` is
recorded on every report so the two are never compared; the packed measurement
becomes available now that step 8 exists, and should be wired when the corpus
can actually be indexed.

**Resolves when:** a Voyage API key is configured (decision 0002 names
`voyage-code-2`) and Neo4j is running. Both, not either.

---

## F-010 — §3.5's framework edge types cannot be written or traversed

**Kind:** contradicts the spec (§3.5 vs §11.2 3b and §11.1).
**Found:** step 10, implementing the React/Next framework layer.
**Status:** **RESOLVED** — the allowlist, §11.1's `MATCH` and §5.2's weight
table all carry the eight types §3.5 names, as v10.1 §0.0 **R2**. The
map-onto-`CALLS` workaround described below is gone: hooks are
`INVOKES_HOOK` edges and a `page.tsx` emits `RENDERS`. `HAS_FIELD` and
`USES_SERIALIZER` are writable but not yet produced — they are Django-side,
and step 10's scope was TS/TSX.

§3.5 specifies five framework relationships:

> **Django:** `(UrlRoute)-[:DISPATCHES_TO]->(View)`; `(Model)-[:HAS_FIELD]->(Field)`;
> `(View)-[:USES_SERIALIZER]->(Serializer)`.
> **React/Next:** `(Component)-[:INVOKES_HOOK]->(Hook)`;
> `(Route)-[:RENDERS]->(Component)`.

Three of those five have nowhere to live:

- **§11.2 3b** substitutes the relationship type "from a fixed allowlist
  `{CALLS, IMPORTS, DEFINES, DISPATCHES_TO}`, never from parsed input", so
  `INVOKES_HOOK`, `RENDERS`, `HAS_FIELD` and `USES_SERIALIZER` cannot be
  written at all.
- **§11.1** matches `[r:CALLS|IMPORTS|DEFINES|DISPATCHES_TO]`, so even if one
  were written it would never be traversed.
- **§5.2**'s edge-weight table covers the same four, so an unlisted type would
  score at `EDGE_WEIGHT_DEFAULT` — moot, given the above.

An `INVOKES_HOOK` edge would therefore be **written and never read**. That is
worse than not writing it: the graph would carry rows nothing queries, and
"what hooks does this component use" would return nothing while the data was
sitting there.

**Implemented.** The relationships are emitted under the types the allowlist
permits, chosen so the claim each edge makes stays true:

| §3.5 | Emitted as | Why the claim survives |
|---|---|---|
| `(Component)-[:INVOKES_HOOK]->(Hook)` | `CALLS` | a component genuinely calls a hook |
| `(Route)-[:RENDERS]->(Component)` | `DISPATCHES_TO` | the same relationship §3.5's Django row already calls dispatch |
| `is_client_component` | a node property | §3.3 already defines it as one; no edge needed |

`HAS_FIELD` and `USES_SERIALIZER` are Django-side and not built — step 10's
scope is TS/TSX. They hit the same wall when they are.

**What is lost.** The label. "Which of this component's calls are hooks" is no
longer answerable from the edge type alone, so
`adapters.typescript.hooks_invoked` reconstructs it from the callee's name
using React's `use[A-Z]` convention. That keeps the question answerable and
keeps the loss visible, rather than papering over it.

**The open decision.** Extending `EDGE_TYPE_ALLOWLIST` is a data-model change,
and `CLAUDE.md` rule 6 says ask first. It would also mean adding the new types
to §11.1's `MATCH` and §5.2's weight table — three coordinated edits to a
frozen spec. Recorded here for that decision rather than taken.

---

## F-011 — `symbol_origin` cannot serve §11.2 steps 4, 6a and 6b as written

**Kind:** contradicts the spec (§11.3 vs §11.2).
**Found:** step 3's `EXPLAIN` gate, first run against a live Neo4j 5.26.0.
**Status:** **RESOLVED** — fixed in `graph/writer.py`, and in the spec as
v10.1 §0.0 **R3**.

§11.3 declares the index and §11.2 comments say it drives the reconciliation:

> `CREATE INDEX symbol_origin ... FOR (s:Symbol) ON (s.repo_id, s.origin_path, s.epoch);   -- §11.2 steps 4-6`

Steps 4, 6a and 6b constrain only `repo_id` and `origin_path`. Step 5 also
constrains `s.epoch < $epoch`, and it is the only one of the four the index can
serve. **Neo4j will not seek a composite index on a prefix of its properties.**
Measured, by hinting the two-property prefix:

```
Neo.ClientNotification.Schema.HintedIndexNotFound
The hinted index does not exist, please check the schema
(index is: INDEX FOR (`s`:`Symbol`) ON (`s`.`repo_id`, `s`.`origin_path`))
```

So the plan's step 3 gate — "`EXPLAIN` on §11.2 steps 4/5/6 shows `symbol_origin`
index usage, not `NodeByLabelScan`" — is **unachievable as specified**. Actual
plans for the statements as written, on a 2,000-node / 12,000-edge graph:

| Statement | Operator |
|---|---|
| step 4 | `NodeByLabelScan s:Symbol` |
| step 5 | `NodeIndexSeek RANGE INDEX s:Symbol(repo_id, origin_path, epoch)` |
| step 6a | `NodeByLabelScan s:Symbol` |
| step 6b | `UndirectedAllRelationshipsScan` |

Three of four touch every symbol — or every relationship — in the tenant. B8 is
exactly this defect, and §11.2's own statements trigger it.

**Fix.** Add `AND s.epoch <= $epoch` to steps 4, 6a and 6b. All four then use
the index. The predicate is a tautology at that point in the sequence: step 3a
has just stamped every symbol in the batch with `$epoch`, and anything else at
those paths is older. A symbol with a *newer* epoch belongs to a later batch,
so excluding it is correct rather than merely harmless.

The alternative — adding a `(repo_id, origin_path)` index to §11.3 — is a schema
change and needs approval; this is a query change with no new index.

**Residual, recorded rather than asserted.** With the predicate, step 4 gets a
`NodeIndexScan` over `symbol_origin` rather than a `NodeIndexSeek`. Both satisfy
the gate's wording, and which one the planner picks moves with index statistics
that Neo4j samples asynchronously — the same statement was observed producing
both. A scan reads the whole index; a seek reads the matching range. The test
asserts the gate's actual bar and prints which operator each statement got.

---

## F-012 — §4.5's undetected-move criterion has F-007's defect

**Kind:** contradicts the spec (§4.5 vs §3.2).
**Found:** step 5, `test_undetected_move_counted_from_the_diff`, first live run.
**Status:** **RESOLVED** — fixed, and corrected in the spec alongside R1.
Carried into v10.1 without being separately approved because leaving it
would have made the revision self-contradictory: §4.4 would state the
corrected rule and §4.5 the wrong one, forty lines apart.

§4.5 defines the §8.3 decision metric:

> a symbol that left a deleted path and appeared at an added path with the same
> `body_hash` and `qualified_name` was a move no live signal caught

`qualified_name` is derived from the module path (§3.2), so it changes with the
file — which is F-007's finding. A pure move therefore never matches, and
`move.undetected` reads **zero forever**. Measured: a `mv` of `pkg/callee.py`
to `pkg/unseen.py` with unchanged content produced `move.undetected == 0`.

The consequence is subtler than F-007's. Nothing breaks; a decision metric
simply reports "this never happens", and §10.2's import-path-repair trigger
(`> 0.10`) can never fire. The capability would be deferred forever on evidence
that was never collected.

**Fix.** Match on the unqualified `name` plus `body_hash`. `helper` survives a
move; `pkg.callee.helper` does not. The `:Module` symbol's name changes with the
filename and so never matches — harmless, because the count is over files and
one matching function is enough to pair them.

---

## F-013 — `EXPLAIN` on an empty database is not evidence

**Kind:** contradicts a working assumption (the B8 gate's method).
**Found:** step 3, diagnosing F-011.
**Status:** fixed in the test.

The first version of the `EXPLAIN` gate ran against a freshly-wiped database.
With no statistics the planner's cost estimates are arbitrary, and the same
statement produced two different plans depending only on whether data existed:

| Graph | step 4's access operator |
|---|---|
| empty | `DirectedAllRelationshipsScan` |
| 2,000 nodes / 12,000 edges | `NodeByLabelScan` |

Neither is the plan production would get, and a gate reading either would be
reporting on a planner that had nothing to plan against. The fixture now builds
a graph whose edges outnumber its nodes, as a real call graph's do, before any
plan is read.

**Related, same root:** two bugs in this repository were reachable only against
a live server and passed every offline test. `Neo4jSearchBackend.fulltext_search`
passes a Cypher parameter named `$query`, which collided first with a helper's
positional `query` argument and then with the driver's own
`Session.run(query, parameters=None, **kwargs)`. Parameters are now passed as an
explicit dict everywhere, which makes any parameter name safe.

---

## F-014 — §11.1's `CALL { WITH … }` is deprecated on Neo4j 5.26

**Kind:** contradicts the spec's forward compatibility (§11.1).
**Found:** step 6, live expansion query.
**Status:** recorded, not changed. It works today.

§11's own header flags `CALL { WITH x … }` as one of four constructs to verify.
It executes correctly on 5.26.0 and emits:

```
Neo.ClientNotification.Statement.FeatureDeprecationWarning
CALL subquery without a variable scope clause is now deprecated.
Use CALL (seed) { ... }
```

The modern form is `CALL (seed) { … }`. §11.1's expansion is the only statement
affected, and it is the one §5.2 calls "the justification for running a graph
database" — so it is the worst one to have break on an upgrade.

Not changed: the spec is frozen, the statement works, and rewriting frozen
Cypher to silence a warning is a change that should be made deliberately rather
than in passing. It is a scheduled breakage, not a current defect — the trigger
is a Neo4j major upgrade.
