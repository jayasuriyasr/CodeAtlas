# SEI — Project Rules

Repository-level GraphRAG engine for Python/Django and TypeScript/Next.js codebases.

## Read these first

- `SEI_Architecture_v10.0.md` — the frozen spec. What to build and why.
- `SEI_Implementation_Plan.md` — 10 gated steps. The order to build it in.
- `PROGRESS.md` — where we are. Read at session start, update at every gate.
- `FINDINGS.md` — anything reality contradicted.

## Non-negotiable rules

1. **The spec is frozen.** Never edit `SEI_Architecture_v10.0.md`. If a run contradicts it, append to `FINDINGS.md` with the command and its actual output. A spec revision needs my explicit approval and must cite the run.

2. **Follow the plan in order.** Do not start step N+1 until step N's gate passes: step tests green, full regression green, acceptance criteria met. Do not skip ahead, do not build two steps at once.

3. **Never edit an older test to make it pass.** A red test from an earlier step while working on a later one is a bug in the current step until proven otherwise. Fix the code, not the test.

4. **Record actual output, never expected output.** Especially the step 1 probes. If a command fails, paste the failure verbatim into `PROGRESS.md`. Do not write what the docs say should happen.

5. **Every constant marked † in the spec lives in `config.py`.** Never a literal in code. They are unvalidated guesses and will move.

6. **Ask before adding a dependency**, changing the data model, or introducing an abstraction the spec does not name.

## Known traps — these caused real defects, do not reintroduce

- **`MERGE` accepts exactly one relationship type.** `MERGE (a)-[:A|B]->(b)` is a syntax error. Multi-type patterns are legal in `MATCH` only. Edges are written one statement per type from a fixed allowlist.
- **Transaction subqueries cannot follow an updating clause.** No `CALL { } IN TRANSACTIONS` after `MERGE`/`SET`. Batch in the driver.
- **The write sequence is not atomic.** It is crash-safe by ordering: superset-on-prefix, idempotent replay. Never describe it as atomic or as a single transaction.
- **UIDs need the ordinal.** TS overloads and conditionally-defined Python functions share `(qualified_name, arity)`. Without the ordinal they collide and `MERGE` silently merges distinct symbols.
- **`code_vec` must actually be written.** Vectors returned from `embed_batch` must be merged into `sym.props` before the node upsert, or symbols index with null embeddings and vector search silently returns nothing.
- **Chunk headers use per-symbol resolved imports**, never the file's import block. Otherwise one added import invalidates every cached embedding in the file.
- **Use the provider's real tokenizer.** Never `len // 4` — it is off by up to 2x on dense code.
- **Tier 2 frame resolution refuses on ambiguity.** More than one candidate returns `None` and falls to Tier 3. Never guess.
- **Citation handles are parsed outside fenced code blocks only.** `arr[C1]` in a snippet is not a citation.
- **Every index leads with `repo_id`.** Tenancy depends on it.

## Testing

- `pytest tests/` runs everything built so far. Run it at every gate, not just the current step's directory.
- Each defect in the plan's §12 table has an owning test. Do not delete or weaken those tests.
- Neo4j tests use a throwaway database per module. Never assume a clean graph.

## Style

- Type hints on all public functions. Docstrings explain *why*, not *what*.
- Prefer explicit over clever. This codebase is a spec made executable; readability beats brevity.
- Comments that reference a spec section (`# spec §4.3`) are welcome and should stay accurate.

## At the end of every step

Update `PROGRESS.md`: gate status, measured constants replacing † guesses, latest metrics, open findings, next step. This is the handoff between sessions — treat it as the deliverable, not an afterthought.
