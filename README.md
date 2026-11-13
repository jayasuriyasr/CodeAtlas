# CodeAtlas

**SEI — Software Engineering Investigation Engine.** Repository-level GraphRAG
over Python/Django and TypeScript/Next.js codebases: ask a question about a
repository and get a streamed answer in which every claim cites the code it
came from — and every citation is checked.

- **Indexes a codebase into a graph.** Tree-sitter parses Python and TypeScript
  into symbols (functions, classes, modules) and edges (calls, imports,
  definitions, Django URL dispatch, React hooks). They are written to Neo4j with
  a vector embedding per symbol.
- **Routes each question to the retrieval it needs.** "What calls `X`" is
  answered by traversing the graph; a pasted stack trace is resolved frame by
  frame to the symbols it names; anything else goes to hybrid fulltext + vector
  search fused with reciprocal rank fusion, then one hop of graph expansion.
- **Packs the evidence into a token budget** — full bodies for the best
  matches, signatures for the rest — and hands it to the model with numbered
  handles (`[C1]`, `[C2]`, …).
- **Checks the answer as it streams.** Each sentence is classified as a claim
  or not, every citation must name a block that was actually shown, and
  relations the answer asserts ("`a` calls `b`") are verified against the
  graph. A poorly grounded answer gets one bounded retry, and anything still
  unsupported is marked in the output rather than hidden.
- **Keeps the index current.** Saves are debounced and re-indexed
  incrementally; file moves keep their inbound edges; embeddings are cached so
  an unchanged symbol is never re-embedded.

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

**Running the tests deletes the index.** Neo4j Community has one database and
the suite isolates by wiping it. After `pytest`, the next `sei start` rebuilds
the index by itself (embeddings are cached, so that is quick).

## Results

The 45-question golden set (15 structural, 15 semantic, 15 stack-trace
questions over a Django and a Next.js fixture), run through the full pipeline
with the local models:

| Metric | Result | Target |
|---|---|---|
| Recall@10, measured on the packed context the model actually saw | 0.978 | ≥ 0.80 |
| MRR | 0.828 | ≥ 0.55 |
| Citation coverage (claim sentences that cite a shown block) | 0.919 | ≥ 0.85 |
| Time to first token, p50 / p95 | 0.86 s / 1.64 s | ≤ 1.2 s / 2.5 s |
| Inbound edges kept across a `git mv` | 1.000 | ≥ 0.95 |
| Embedding cache hit rate on an unchanged re-index | 1.000 | ≥ 0.90 |

The vector arm with the local embedder lifts Recall@10 from 0.933 (fulltext and
graph only) to 0.978. `python -m eval.local_gates` re-runs the measurement and
writes a full report to `eval/GATE_REPORT_local.md`.

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

```bash
.venv/Scripts/python -m pytest tests/ -v
```

The tests are grouped by build step, from the Neo4j ground-truth probes up to
the end-to-end evaluation, and `pytest tests/` runs all of them.

Set `SEI_REQUIRE_NEO4J=1` to make a missing database a failure rather than a
skip; without it, the graph tests skip when Neo4j is down.

**One suite at a time.** Neo4j Community has a single user database, so the
per-module fixture isolates by wiping it. Two concurrent runs — or `pytest -n`
— delete each other's data and fail in ways that look like real defects.

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
