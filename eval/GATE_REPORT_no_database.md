# §8.2 gates

model: claude-haiku-4-5   embedding model: voyage-code-2
p50 packed tokens: 1575
resolution: one flipped question = 6.7pp of a class score (15 questions per class)

Recall@10 (packed context)                    —  >= 0.8      NOT MEASURED measured offline
MRR                                           —  >= 0.55      NOT MEASURED measured offline
Citation Coverage                             —  >= 0.85      NOT MEASURED measured offline
T2 classifier precision                   0.981  >= 0.9      PASS         measured offline
T2 classifier recall                      0.962  >= 0.85      PASS         measured offline
Relation Precision                            —  >= 0.9      NOT MEASURED measured offline
Embedding cache hit rate                  1.000  >= 0.9      PASS         measured offline
Context drop rate                         0.000  <= 0.1      PASS         measured offline
Inbound-edge survival across move             —  >= 0.95      NOT MEASURED measured offline
TTFT p50                                      —  <= 1.2s     NOT MEASURED measured offline
TTFT p95                                      —  <= 2.5s     NOT MEASURED measured offline
$/query p50 (uncached)                        —  <= 0.02      NOT MEASURED measured offline
$/query p95 (uncached)                        —  <= 0.06      NOT MEASURED measured offline

4 passing · 0 failing · 9 not measured, of 13

# §8.3 decision metrics

TS/TSX TRACE resolution             —  < 0.7    —       sourcemap tier
Undetected-move rate                —  > 0.1    —       import-path repair
Header-only miss share              —  > 0.2    —       warm cache reuse
Seed L1-overflow rate           0.000  > 0.05   no      L2 skeleton

# notes

- T2 classifier measured over 58 hand-annotated sentences across all 45 golden questions.
- Embedding cache hit rate measured over a real double index of the Django fixture (73 symbols, 18 files).
- Context drop rate and p50 packed tokens measured with the stand-in tokenizer (FINDINGS F-006) — the figures move when the provider's real tokenizer is wired.
- Recall@10, MRR, Coverage, Relation Precision and inbound-edge survival need a live Neo4j; Recall additionally needs a real embedding provider (F-009). TTFT and $/query need a real LLM.
- NOT MEASURED is not PASS. Nine of the thirteen rows have no run behind them and say so.
