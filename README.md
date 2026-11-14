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

Measured on 2026-11-13 with the local models (`qwen2.5-coder:3b`,
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
| Citation coverage (claim sentences citing a shown block), four runs | 0.797 – 0.919 | ≥ 0.85 |
| Cited claims whose cited block lacks a code name they mention | 7 of 55 | — |
| Relation precision (relation claims the graph confirms) | 0.750, over 4 claims | ≥ 0.90 |
| Answers that needed the one retry | 11 – 13 of 45 | — |
| TS/TSX stack frames resolved to a symbol (golden set: no build output, so no source maps) | 3 of 9 | — |
| Time to first token, p50 / p95, model already loaded | 0.51 s / 1.03 s | ≤ 1.2 s / 2.5 s |
| Inbound edges kept across a `git mv` | 1.000 | ≥ 0.95 |
| Embedding cache hit rate on an unchanged re-index | 1.000 | ≥ 0.90 |
| T2 claim classifier, precision / recall — on the author's own reference sentences | 0.981 / 0.962 | ≥ 0.90 / 0.85 |

**On code not written for SEI.** `python -m eval.benchmark` indexes pinned
releases of two open-source Django projects and scores "what calls X" against
the callers jedi (a Python language server) finds. Every function in each
repository is examined (`--sample N --seed S` scores a seeded random sample
instead, and the report says so). Precision is given two ways: over the
functions jedi finds a caller for, and over every function — where a caller SEI
reports for a function jedi finds none for counts as wrong:

| Repository (pinned commit) | Functions examined | With jedi callers | Precision, where jedi has callers | Precision, all | Recall |
|---|---|---|---|---|---|
| django-taggit 6.1.0 (`4e5f7de6`) | 273 (all) | 45 | 1.000 | 1.000 | 0.915 |
| django-filter 26.2 (`daddf7b6`) | 745 (all) | 95 | 0.996 | 0.988 | 0.748 |

An earlier version of this table said "every function" but scored the first 60
in sorted file order: django-filter recall 0.369 there, against 0.471 for the
same code over every function. Calls on objects whose type the code states
(`till = Till()`, a typed parameter, a class attribute) raised that to 0.748.
Of the 84 callers still missed in django-filter, 49 are method calls on locals
bound more than once or by a loop or `with` — SEI types a local only when one
assignment fixes it — and 14 are call chains or indexing. Every disagreement
is listed in `eval/BENCHMARK_REPORT.md`, which the run writes.

jedi is the gold here, so it was checked too. [20 disagreements read by
hand](eval/benchmark_hand_check.md): all three callers only SEI reports are
real calls jedi does not attribute (in a lambda, in a nested `def`, through
`super()`), so jedi was wrong in 3 of 20 and "precision, all" understates SEI.
In the 17 sampled misses jedi was right. The largest cause of misses is a
method called on a class named directly (`FilterSet.filter_for_field(…)`, an
inherited classmethod): 26 of the 92, not resolved yet. TypeScript has no
language-server gold yet.

What the table does not say on its own:

- **Recall@10 for structural questions measures entity lookup.** Their `gold`
  is the symbol the question names. Whether the traversal returned the right
  callers is the separate *structural traversal* row, scored against
  `expected_related` in the golden set — the full answer read off the fixture's
  source and cross-checked with Python's `ast`. Its misses are real: a route
  question falls back to search, a `Meta.model` reference is not an edge, and
  `token.revoke()` on a local variable is never resolved.
- **Citation coverage fails the gate on the latest run** (0.797). One answer
  made 8 uncited claims; without it the run is 0.90. The model does not follow
  the citation instruction reliably, and one answer in four needs the retry.
- **Relation precision is over 4 claims** in 45 answers. The local model rarely
  states a relation in a form the checker can read, so this number is too thin
  to mean much, and before this round it was not measured at all.
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

- Call resolution is syntactic: `self.x()` and imported names resolve, calls on
  other objects do not. There is no type inference.
- Structural answers are one hop: direct callers, direct subclasses.
- TypeScript stack frames from bundled code (`.next/server/chunks/…`) resolve
  through the build's source map when the `.map` is on disk beside the chunk
  (Next.js writes server maps only when configured to). Without one they are
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
