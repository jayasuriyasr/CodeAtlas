# Step 6 — Recall baseline

Written by `tests/step_06_retrieval/test_retrieval_graph.py`. Do not hand-edit; re-run the test.

- Questions: 15 (plan §6's seed set)

- Gate for reference: Recall@10 >= 0.8 against **packed** context (§8.1/§8.2) — a harder measurement than either figure below, and not available until step 8.

## Fulltext arm only (meaningful)

```
stage=retrieved (fulltext arm only)  corpus=tests/fixtures/repos/django_min
embedding model: n/a
questions: 15 (15 scoreable)
resolution: one flipped question = 6.7pp

Recall@1  0.733
Recall@5  0.933
Recall@10 0.933   (§8.2 gate >= 0.8)
MRR       0.813   (§8.2 gate >= 0.55)

per class @10: SEMANTIC=0.91, STRUCTURAL=1.00

missed at 10:
  q07  'login endpoint'
```

## Blended with stand-in embeddings (NOT meaningful)

```
stage=retrieved (blended, stand-in embeddings)  corpus=tests/fixtures/repos/django_min
embedding model: hash-test-provider
questions: 15 (15 scoreable)
resolution: one flipped question = 6.7pp

Recall@1  0.333
Recall@5  0.667
Recall@10 0.867   (§8.2 gate >= 0.8)
MRR       0.517   (§8.2 gate >= 0.55)

per class @10: SEMANTIC=0.91, STRUCTURAL=0.75

missed at 10:
  q07  'login endpoint'
  q09  'revoke a token'

The blended number is NOT a retrieval result. HashEmbeddingProvider has no semantic structure (see index/providers.py and FINDINGS F-006), so its arm contributes noise. Re-run with the real provider before recording anything against the §8.2 gate.
```