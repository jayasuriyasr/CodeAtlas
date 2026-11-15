# CodeAtlas

*CodeAtlas is the repository; SEI is the engine inside it, and the name its
command (`sei`), environment variables (`SEI_*`) and package use.*

**SEI — Software Engineering Investigation Engine.** Repository-level GraphRAG
over Python/Django and TypeScript/Next.js codebases: ask a question about a
repository and get a streamed answer whose claims cite the code they came
from, with each citation checked as described below.

- **Indexes a codebase into a graph.** Tree-sitter parses Python and TypeScript
  into symbols (functions, classes, modules) and edges (calls, imports,
  definitions, inheritance, Django URL dispatch, React hooks). They are written
  to Neo4j with a vector embedding per symbol. Calls through `self`, `cls`,
  `this` and `super()` are resolved along the class's method resolution order; calls on other
  objects (`order.ship()`) are not, because that needs type inference.
- **Routes each question to the retrieval it needs.** "What calls `X`" is
  answered by traversing the graph one hop; a pasted stack trace is resolved
  frame by frame to the symbols it names; anything else goes to hybrid fulltext +
  vector search fused with reciprocal rank fusion, then one hop of graph
  expansion. A name that matches several symbols is answered with the list of
  candidates, not with all of them merged.
- **Packs the evidence into a token budget** — full bodies for the best
  matches, signatures for the rest — and hands it to the model with numbered
  handles (`[C1]`, `[C2]`, …).
- **Checks the answer.** Each sentence is classified as a claim or not. A claim
  must cite a block that was actually shown, and the code names it mentions must
  appear in the block it cites (a cheap check: it catches citations of unrelated
  code, not wrong statements about the right code). Relation claims written in a
  recognised form ("`a` calls `b`", "`X` is called by the `Y` method",
  "inherits from") are looked up in the graph. A poorly grounded answer gets one
  bounded retry, and whatever is still unsupported is marked, not hidden.
- **Keeps the index current.** Saves are debounced and re-indexed
  incrementally; a changed signature re-links the callers in other files; file
  moves keep their inbound edges; embeddings are cached so an unchanged symbol
  is never re-embedded.

How it works — the architecture, the invariants it keeps, the trade-offs and
the known limitations — is in [DESIGN.md](DESIGN.md).

## Running it on one machine (Windows)

Everything runs locally: Neo4j, a local answer model and a local embedding
model. No API key, no account, and no source code leaves the machine.

```bash
sei setup
```

```bash
sei start C:\path\to\your\project
```

`sei start` brings up Neo4j and the model server, indexes the project, keeps the
index current as you save files, and opens the console at
`http://127.0.0.1:8000`. Ask in the page, or from the terminal:

```bash
sei ask "what calls `charge_card`"
```

| Command | What it does |
|---|---|
| `sei setup` | creates `.venv`, installs the Python packages, downloads and checksum-verifies Neo4j 5.26.0, pulls the two models. Safe to re-run |
| `sei start <folder>` | starts everything for that project and opens the console. `-Port 8001`, `-NoBrowser`, `-Repo name` |
| `sei status` | what is running, which models answer, how many files and symbols are indexed |
| `sei ask "…"` | one question, in the terminal, with the claim-by-claim verdicts |
| `sei index` | re-read every file and re-link every edge |
| `sei stop` | stops the server and Neo4j and unloads the models from the GPU |

**What it needs.** Python 3.11+, Java 17 or newer (for Neo4j), git (so
`.gitignore` is respected), and [Ollama](https://ollama.com/download) for the
models. Measured on a laptop with a 4 GB GPU: a 134-file Python project indexed
in about 70 seconds the first time; later starts are ready in about 35 seconds
and re-index nothing; a saved file is searchable about 4 seconds later; an
answer starts in 0.2–2 seconds once the model is loaded. Without Ollama it still
runs, with keyword and graph search and no written answers.

**Where the large files go.** Neo4j, Ollama's models, downloads and logs live in
one *tools folder* outside the repository — `sei setup -ToolsDir D:\sei-tools`
chooses it, and it is remembered in the git-ignored `.local/config.json`. To
keep Ollama itself off `C:`, install it with `OllamaSetup.exe /DIR="<tools
folder>\ollama"` and set the user variable `OLLAMA_MODELS` to `<tools
folder>\ollama-models` before its first start.

**The models, and their licences.** Answers come from `qwen2.5-coder:3b`, search
from `nomic-embed-text`. The 3B model is small enough for a 4 GB GPU and is
published under the *Qwen Research License* — fine for study and evaluation, not
for commercial use. Swap it with `SEI_LOCAL_LLM` (and `SEI_LOCAL_EMBED` for the
embedder), then re-measure with `python -m eval.local_gates`.

**The tests have their own Neo4j.** Neo4j Community has one database and the
suite isolates by wiping it, so it never uses the app's: `sei test` starts a
second server on port 7688 (installed by `sei setup`) and runs the suite there.

## Results

Measured on 2026-11-14 with the local models (`qwen2.5-coder:3b`,
`nomic-embed-text`) on a laptop with a 4 GB GPU. **Read these as numbers about
two small fixtures written for this project** — a Django app of 18 files and
73 symbols, and a Next.js app of 4 files and 20 symbols — not about real
codebases. The golden set is 45 questions: 15 structural, 15 semantic and 15
stack traces (8 Python, 7 TypeScript). At 15 questions a class, one question
moves a class score by 6.7 points.

| Metric | Result | Target |
|---|---|---|
| Recall@10 on the packed context the model saw, vector arm on (off) | 0.978 (0.933) | ≥ 0.80 |
| MRR | 0.828 | ≥ 0.55 |
| Structural traversal against the full expected answer: precision / recall | 0.974 / 0.881 | — |
| Citation coverage (claim sentences citing a shown block), five runs | 0.797 – 0.919 (latest 0.862) | ≥ 0.85 |
| Cited claims whose cited block lacks a code name they mention | 7 of 56 | — |
| Relation precision (relation claims the graph confirms) | 1.000 over 3 claims (earlier: 0.750 over 4) | ≥ 0.90 |
| Answers that needed the one retry | 11 – 13 of 45 | — |
| TS/TSX stack frames resolved to a symbol (golden set: no build output, so no source maps) | 3 of 9 | — |
| Time to first token, p50 / p95, model already loaded | 1.15 s / 2.00 s (earlier: 0.51 s / 1.03 s) | ≤ 1.2 s / 2.5 s |
| Inbound edges kept across a `git mv` | 1.000 | ≥ 0.95 |
| Embedding cache hit rate on an unchanged re-index | 1.000 | ≥ 0.90 |
| T2 claim classifier, precision / recall — on the author's own reference sentences | 0.981 / 0.962 | ≥ 0.90 / 0.85 |

**On code not written for SEI.** `python -m eval.benchmark` indexes pinned
open-source repositories — two Django libraries, a Django application and a
Next.js application — and scores "what calls X" against a language server's
callers: jedi for Python, and for TypeScript the TypeScript language service
(tsserver's engine, `typescript` 5.6.3, a dev-only npm tool installed outside
the repository). Every function in each repository is examined
(`--sample N --seed S` scores a seeded random sample instead, and the report
says so). Precision is given two ways: over the functions the gold finds a
caller for, and over every function — where a caller SEI reports for a function
the gold finds none for counts as wrong:

| Repository (pinned commit) | Gold | Functions examined | With gold callers | Precision, where the gold has callers | Precision, all | Recall |
|---|---|---|---|---|---|---|
| django-taggit 6.1.0 (`4e5f7de6`) | jedi | 273 (all) | 45 | 1.000 | 1.000 | 0.915 |
| django-filter 26.2 (`daddf7b6`) | jedi | 745 (all) | 95 | 0.996 | 0.989 | 0.814 |
| healthchecks v5.0.3 (`51a50353`), a Django application | jedi | 2,760 (all) | 319 | 0.887 | 0.854 | 0.807 |
| taxonomy (`298a8857`), a Next.js App Router application | tsserver | 130 (all) | 28 | 1.000 | 1.000 | 0.990 |

An earlier version of this table said "every function" but scored the first 60
in sorted file order: django-filter recall 0.369 there, against 0.471 for the
same code over every function. Calls on objects whose type the code states
(`till = Till()`, a typed parameter, a class attribute) raised that to 0.748,
and methods called on a class named directly to 0.814. Of the 62 callers still
missed in django-filter, 27 are method calls on locals bound more than once or
by a loop or `with` — SEI types a local only when one assignment fixes it — and
14 are call chains or indexing. Every disagreement
is listed in `eval/BENCHMARK_REPORT.md`, which the run writes.

The gold was checked too ([read by hand](eval/benchmark_hand_check.md)):

- **django-taggit and django-filter, 20 disagreements:** all three callers only
  SEI reports are real calls jedi does not attribute (in a lambda, in a nested
  `def`, through `super()`), so jedi was wrong in 3 of 20. In the 17 sampled
  misses jedi was right. The largest cause of misses was a method called on a
  class named directly (`FilterSet.filter_for_field(…)`, an inherited
  classmethod): 26 of the 92. Resolving it raised django-filter's recall from
  0.748 to 0.814.
- **healthchecks' lower precision is jedi's, not SEI's.** All 201 callers SEI
  reports there and jedi does not have a call by that name in the caller's body
  — 140 of them `super().setUp()` in tests, the rest inherited `self.post()`,
  calls from decorator `wrapper` functions, and module functions called as
  `curl.post(…)` — and a seeded sample of 10 read by hand were all correct.
  "Precision" here measures where jedi stops, so read 0.887 as a floor.

TypeScript recall is measured on one small application (28 functions with
callers); taxonomy's only miss is one caller.

**On a large repository.** `python -m eval.scale` runs the server's own
indexer, watcher and retriever on cal.com v6.2.0 (`1c193cca`): 7,363
TypeScript files, 22,486 symbols. One laptop: 12 threads, 15 GB RAM, an RTX 3050
with 4 GB; local embeddings (`nomic-embed-text`).

| Measurement | Before | After the fix |
|---|---|---|
| First index, empty graph and cache | 1,363 s (22.7 min) | — |
| Warm restart, nothing changed | 8.9 s | 8.4 s |
| Watcher's idle poll (interval 1 s) | 1,639 ms | 325 ms |
| Save to searchable, p50 / max of 5 | 5.27 s / 6.15 s | 3.74 s / 4.94 s |
| "What calls X", p50 / p95 of 15 | 10 ms / 152 ms | 7 ms / 158 ms |
| Semantic retrieval, p50 / p95 of 15 | 106 ms / 117 ms | 106 ms / 125 ms |
| Memory: SEI process peak / Neo4j | 261 MB (first index), 86 MB (serving) / 1.3 GB | same |

Where the time goes, measured on a 286-file slice: the first index is 74% embedding,
15% graph writes and 2% parsing. Embedding is bound by the GPU at about 21 chunks
a second; 64 texts per request instead of 16 saved 7%, so batching is not the
lever. Not fetching vectors before the graph is searchable would be, and is not
built. The bottleneck code could fix is the watcher. On every 1 s tick it re-read
the whole tree, and 1.6 s per pass kept a core about 60% busy while idle and
delayed every save. The pass now takes each file's size and mtime from the directory
listing (`os.scandir`) instead of a second `os.stat` per file. The first index
ran beside a single-threaded benchmark; every other row ran alone. The save
latency left is the 1 s poll plus the 2 s debounce (`DEBOUNCE_SECONDS` †) by
design, plus the flush. Search results are retrieval only: no answer model.

What the table does not say on its own:

- **Recall@10 for structural questions measures entity lookup.** Their `gold`
  is the symbol the question names. Whether the traversal returned the right
  callers is the separate *structural traversal* row, scored against
  `expected_related` in the golden set — the full answer read off the fixture's
  source and cross-checked with Python's `ast`. Its misses are real: a route
  question falls back to search, a `Meta.model` reference is not an edge, and
  `token.revoke()` on a local variable is never resolved.
- **Citation coverage moves between runs of the same model** (0.797 – 0.919
  at temperature 0), so it passes or fails the gate by run. The model is the
  lever, not the wording. Four runs on 2026-11-14, two models by two answer
  formats (the current one, and a tighter one asking for one fact per sentence
  with its own handle and relations written as "`a` calls `b`"):

  | Model, format | Coverage | Relation claims | Relation precision | Retried / still below | TTFT p50 / p95 |
  |---|---|---|---|---|---|
  | 3B, current | 0.862 | 3 | 1.000 | 11 / 3 | 1.15 s / 2.00 s |
  | 3B, tighter | 0.800 | 23 | 0.435 | 19 / 11 | 0.45 s / 0.79 s |
  | 7B, current | 1.000 | 5 | 0.800 | 5 / 1 | 0.92 s / 4.00 s |
  | 7B, tighter | 1.000 | 13 | 0.846 | 4 / 1 | 3.74 s / 8.17 s |

  `qwen2.5-coder:7b` cited every claim sentence in both formats; on this 4 GB
  GPU it spills to the CPU and misses the latency gate (p95 4.0 – 8.2 s). The
  tighter format multiplied checkable relation claims, but on the 3B it left 11
  answers below the bar instead of 3, so it was not adopted. Neither reached 50
  relation claims on 45 questions. Set `SEI_LOCAL_LLM=qwen2.5-coder:7b` where a
  bigger GPU makes its latency acceptable. It is also Apache-2.0 licensed,
  while the 3B's licence is research-only.
- **Relation precision understates the model, and its "contradicted" flag
  fires on true claims.** The 23 relation claims of the 3B/tighter run were
  checked by hand against the fixture: 15 are true, and 8 are wrong — four read
  an audit-event string (`"login"`) as the caller, and three name a class where
  its method calls. The graph confirmed 10. Of the rest of the true ones, two
  name a library function (`secrets.token_hex`), which the graph does not hold,
  and two (`token.revoke()` on a loop variable and on a `.first()` result) are
  calls SEI does not resolve. Those two counted as *contradicted*, and a
  contradiction triggers the retry, so the retry can fire on a correct answer.
- **The classifier row is not a measurement on model output.** It scores the
  classifier on reference sentences the author wrote, with the gold symbols
  passed in as known names.
- **Why 6 of 9 TypeScript frames do not resolve.** Three name `GET` or `POST`,
  which several route files export, so they are refused as ambiguous; one is
  minified (`t`); one is `<anonymous>`, with no position; and one is
  `formatName`, imported from a file the fixture does not contain. The golden
  set has no build output, so the source-map tier cannot run there and the row
  stays 3 of 9. On a chunk TypeScript 5.6.3 built from two route files that both
  export `GET` and `POST` (`tests/step_07_router_frames/test_sourcemap_tier.py`),
  the shared and minified names resolve to the right file and function through
  the map. `<anonymous>` and a symbol the repository lacks cannot resolve by
  any route.
- **The hosted models are unmeasured.** `ANTHROPIC_API_KEY` and
  `VOYAGE_API_KEY` switch the answers and the embeddings to hosted models;
  none of the numbers above apply to them.

`python -m eval.local_gates` re-runs the local measurement and writes the full
report, with every answer, to `eval/GATE_REPORT_local.md`.

## Known limitations

- Call resolution has no type inference. `self.x()`, `super().x()`, imported
  names, a method called on a class named directly
  (`FilterSet.filter_for_field(…)`), and calls on an object whose type the code
  states (`till = Till()` assigned once, a typed parameter, a class attribute,
  `self.a` set once in `__init__`) resolve, through the class hierarchy. A
  local bound twice and a call on a call's result do not. TypeScript resolves
  `this.x()`, same-file names and imports only.
- "What calls X" is one hop, plus the callers of the method X overrides
  (marked "through dispatch"). "What eventually calls X" and "what is affected
  if I change X" walk up to `MULTIHOP_MAX_DEPTH` (4) levels and
  `MULTIHOP_MAX_RESULTS` (50) symbols, and say when they stop at the cap.
- TypeScript stack frames from bundled code (`.next/server/chunks/…`) resolve
  through the build's source map when the build left a `.map` on disk beside
  the chunk; whether it does depends on the build's settings, and only maps
  produced by `tsc` were tested, not a real Next.js build. Without one they are
  matched by name, and a shared name (`GET`) or a minified one (`t`) is refused
  rather than guessed. A map whose embedded source differs from the file on
  disk is not used. Index maps (`sections`) are not read.
- The packer counts tokens with an approximation that over-counts (measured
  against the local model's own tokenizer: 0.52–0.94 real tokens per counted
  one). It errs toward too little context, never a truncated prompt.
- One Neo4j database holds every indexed repository; queries are scoped by
  `repo_id`, and the vector search widens its fetch so a small repository is
  not crowded out by a large one.
- The default local answer model is research-licensed (see above).

## Running it on macOS or Linux

```bash
pip install -e .
```

```bash
docker compose up -d neo4j
```

```bash
sei serve ~/code/your-project
```

`sei serve` indexes the project, follows its saves and prints the console link
(with this run's token). `sei ask <folder> "question"`, `sei index <folder>` and
`sei check` (is Neo4j up, which local models are pulled) work the same way.
Install [Ollama](https://ollama.com/download) and pull `qwen2.5-coder:3b` and
`nomic-embed-text` for local answers and search.

Or run the server in Docker as well: `SEI_PROJECT=<folder> SEI_API_TOKEN=<long
random> SEI_NEO4J_PASSWORD=<password> docker compose --profile app up -d`, then
open `http://127.0.0.1:8000/?token=<token>` (details in `docker-compose.yml`).

### Serving beyond this machine

SEI speaks plain HTTP and has no TLS of its own. Bound to `127.0.0.1` (the
default) that is fine: nothing leaves the machine. Bound to a network address
(`--host 0.0.0.0`, which needs `SEI_API_TOKEN`), the token in every request —
and the source code in every answer — crosses the network readable by anyone
on the path, and the server prints a warning saying so. To use it from another
machine, keep it on `127.0.0.1` and either:

- **tunnel over SSH** (nothing to configure): `ssh -L 8000:127.0.0.1:8000
  <server>`, then open `http://127.0.0.1:8000/?token=<token>` locally; or
- **put a TLS proxy in front** on the same machine (Caddy, nginx). The proxy
  must send `Host: 127.0.0.1:8000` upstream — the server refuses any other host
  name on a local bind (the DNS-rebinding check) — and must not buffer
  responses, because answers stream. With Caddy: `reverse_proxy 127.0.0.1:8000
  { header_up Host 127.0.0.1:8000; flush_interval -1 }`. This route is
  described, not tested here: only the Host check it relies on is.

The console's `?token=` link is taken out of the address bar once the page has
read it, but it can still reach a proxy's access log — prefer the
`Authorization: Bearer` header for scripts.

## Running it by hand

Requires Python 3.11+ and Neo4j 5.26 Community.

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
```

Bring up Neo4j (pinned to an exact minor — see `docker-compose.yml` for why):

```bash
docker compose up -d
```

Without Docker, the official `neo4j-community-5.26.0` distribution run directly
works the same; set the initial password to `sei-dev-password` (or point the
`SEI_NEO4J_*` variables at your own).

### Ask a question

```bash
.venv/Scripts/python -m api.cli --repo demo --root path/to/repo --reindex "how are API tokens issued"
```

The question is routed and answered by the path its class needs:

| Question | Route | What happens |
|---|---|---|
| "what calls `charge_card`", "where is `X` used" | STRUCTURAL | the entity is resolved and the graph traversed — the traversal *is* the answer |
| a pasted stack trace | TRACE | frames resolved to symbols (per-frame badge), unresolved ones fall to a semantic search |
| "how does auth work" | SEMANTIC | hybrid fulltext + vector search, then 1-hop graph expansion |

Useful flags:

| Flag | Effect |
|---|---|
| `--reindex` | re-index the whole tree first (both languages), re-resolving every edge |
| `--sync` | re-read every file — the recovery path after moves no editor reported |
| `--offline` | retrieval only, no model of any kind, hosted or local |
| `--ai auto\|local\|cloud\|off` | which models answer (default `auto`, below) |
| `--serve --port 8000` | run the console and HTTP API: `GET /` (the console), `GET /status`, `POST /ask` (SSE), `POST /index/rename`, `POST /index/sync`, `GET /healthz` |
| `--serve --watch` | also index at start-up and re-index files as they are saved |

Which models answer is decided once, at start, and printed on stderr. `auto`
uses a hosted provider when you have configured its key — setting a key is a
decision to send code to that service — and the local Ollama models otherwise.
The two halves are independent, and without either SEI degrades rather than
fails:

| | Hosted (key set) | Local (Ollama running, model pulled) | Neither |
|---|---|---|---|
| Answers | `ANTHROPIC_API_KEY` | `qwen2.5-coder:3b` | the ranked, cited code blocks, no prose |
| Search | `VOYAGE_API_KEY` | `nomic-embed-text` | fulltext + graph expansion only |

A repository has one embedding model at a time: vectors from two models cannot
be compared. Switching to a local embedder re-embeds the repository at the next
start; switching to a billed one never happens by itself — run `--reindex`
once.

### Run the tests

The suite needs a Neo4j of its own, because it wipes the database it uses:

```bash
docker compose up -d neo4j-test
```

```bash
SEI_REQUIRE_NEO4J=1 .venv/Scripts/python -m pytest tests/
```

(`sei test` does both on Windows without Docker.) The tests connect to
`SEI_TEST_NEO4J_URI`, by default `bolt://localhost:7688` — never the app's
`SEI_NEO4J_URI`. Before the first wipe they check the database: if it holds
data and does not carry the suite's marker, the run stops rather than delete
it (`SEI_TEST_ALLOW_WIPE=1` overrides that). In the other direction, the app
refuses to index into a database that carries the marker.

The tests are grouped by build step, from the Neo4j ground-truth probes up to
the end-to-end evaluation, and `pytest tests/` runs all of them. Set
`SEI_REQUIRE_NEO4J=1` to make a missing database a failure rather than a skip.

**One suite at a time.** Two concurrent runs against the same test server — or
`pytest -n` — delete each other's data and fail in ways that look like real
defects.

No test calls a real LLM or embedding API: that would send source code to a
third party. Providers are tested against stub transports — the local ones
against a stub Ollama server (`tests/ollama_stub.py`), so the suite needs no
model installed.

## Layout

```
adapters/    per-language parsing and symbol extraction: Python, TypeScript/TSX
graph/       schema.cypher, bootstrap, writer, reader
index/       indexer, file watcher, chunker, embedding cache,
             providers: hosted (providers.py) and local (ollama.py)
retrieve/    router, hybrid RRF, expansion, stack frames, structural traversal, pipeline
pack/        context packer
ground/      claim classifier and citation / relation validation
api/         FastAPI + SSE, the CLI, and the console page (static/index.html)
eval/        evaluation harness, golden set, local_gates.py
scripts/     sei.ps1 — setup, start, stop, status (run through sei.cmd)
config.py    every tunable constant, in one place
tests/       one directory per build step, plus shared fixtures
```

## Configuration

Behaviour is tuned in `config.py` and nowhere else. Only infrastructure and
credentials read the environment:

| Variable | Default |
|---|---|
| `SEI_NEO4J_URI` | `bolt://localhost:7687` |
| `SEI_NEO4J_USER` | `neo4j` |
| `SEI_NEO4J_PASSWORD` | `sei-dev-password` |
| `SEI_NEO4J_DATABASE` | `neo4j` |
| `SEI_CACHE_PATH` | `.sei/embeddings.sqlite` |
| `ANTHROPIC_API_KEY` | unset — answers come from the local model, if there is one |
| `VOYAGE_API_KEY` | unset — search uses the local embedder, if there is one |
| `SEI_AI` | `auto` — or `local`, `cloud`, `off` (same as `--ai`) |
| `SEI_OLLAMA_URL` | `http://127.0.0.1:11434` |
| `SEI_LOCAL_LLM` | `qwen2.5-coder:3b` |
| `SEI_LOCAL_EMBED` | `nomic-embed-text` |
| `SEI_TOOLS_DIR` | unset — `sei` reads `.local/config.json`, written by `sei setup` |
| `SEI_TEST_NEO4J_URI` | `bolt://localhost:7688` — the test suite's own server |
| `SEI_TEST_ALLOW_WIPE` | unset — set to `1` to let the suite wipe an unmarked database |
