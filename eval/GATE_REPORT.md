# §8.2 gates

model: claude-haiku-4-5   embedding model: voyage-code-2
p50 packed tokens: 1459
resolution: one flipped question = 6.7pp of a class score (15 questions per class)

Recall@10 (packed context)                0.933  >= 0.8      PASS         measured
MRR                                       0.813  >= 0.55      PASS         measured
Citation Coverage                             —  >= 0.85      NOT MEASURED measured
T2 classifier precision                   0.981  >= 0.9      PASS         measured
T2 classifier recall                      0.962  >= 0.85      PASS         measured
Relation Precision                            —  >= 0.9      NOT MEASURED measured
Embedding cache hit rate                  1.000  >= 0.9      PASS         measured
Context drop rate                         0.000  <= 0.1      PASS         measured
Inbound-edge survival across move         1.000  >= 0.95      PASS         measured
TTFT p50                                      —  <= 1.2s     NOT MEASURED measured
TTFT p95                                      —  <= 2.5s     NOT MEASURED measured
$/query p50 (uncached)                        —  <= 0.02      NOT MEASURED measured
$/query p95 (uncached)                        —  <= 0.06      NOT MEASURED measured

7 passing · 0 failing · 6 not measured, of 13

# §8.3 decision metrics

TS/TSX TRACE resolution             —  < 0.7    —       sourcemap tier
Undetected-move rate                —  > 0.1    —       import-path repair
Header-only miss share              —  > 0.2    —       warm cache reuse
Seed L1-overflow rate           0.000  > 0.05   no      L2 skeleton

# notes

- Neo4j: measured against a live server. T2 classifier over 58 hand-annotated sentences from all 45 golden questions.
- Recall@10 and MRR are against **packed** context (§8.1), the fulltext arm only — the vector arm is HashEmbeddingProvider, which has no semantic structure (F-009). A real embedding provider can only improve these.
- Inbound-edge survival: §8.2's fixture (a) — `git mv` a file with five known callers, flush, recount.
- Token figures use the stand-in tokenizer (F-006) and move when the provider's real one is wired.
- Citation Coverage, Relation Precision, TTFT and $/query need a real LLM. Scripting one would measure the script, so they are NOT MEASURED rather than reported.
- 
- Recall detail:
-     stage=packed (fulltext arm only)  corpus=tests/fixtures/repos/django_min
-     embedding model: n/a — fulltext only (F-009)
-     questions: 15 (15 scoreable)
-     resolution: one flipped question = 6.7pp
-     
-     Recall@1  0.733
-     Recall@5  0.933
-     Recall@10 0.933   (§8.2 gate >= 0.8)
-     MRR       0.813   (§8.2 gate >= 0.55)
-     
-     per class @10: SEMANTIC=0.91, STRUCTURAL=1.00
-     
-     missed at 10:
-       q07  'login endpoint'
