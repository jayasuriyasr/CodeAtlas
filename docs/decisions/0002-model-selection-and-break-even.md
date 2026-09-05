# 0002 — Model selection and the §5.4 break-even

**Status:** Accepted · 2026-08-27
**Answers:** plan §1 — "Name the model + embedding model + version. Check spec
§5.4's $0.94/Mtok break-even against the actual rate card."
**Depends on:** [0001](0001-target-market-and-source-egress.md) (hosted path).

## Decision

| Role | Model | Version / ID | Rate (per 1M tokens) |
|---|---|---|---|
| Synthesis (S2 `stream`) | Claude Haiku 4.5 | `claude-haiku-4-5` | $1.00 in · $5.00 out |
| Embeddings (S2 `embed`) | Voyage `voyage-code-2` | 1536 dims | per Voyage's card |

Both are recorded in `config.py` as `LLM_MODEL`, `LLM_INPUT_RATE_USD_PER_MTOK`,
`LLM_OUTPUT_RATE_USD_PER_MTOK`, `EMBED_MODEL`, `EMBED_DIM`. §5.4 requires the
eval config to name model and version "or neither number can be checked", so
these are constants in the repo, not deployment settings.

`voyage-code-2` is chosen over the alternatives for one concrete reason: it
emits **1536** dimensions, which is what §11.3's vector index already declares.
Any other choice would have meant editing a frozen spec's DDL on day one.

## The break-even check

§5.4's arithmetic, unchanged:

```
break-even input_rate = $0.02 / 21,200 tok = $0.9434 per 1M input tokens
```

Anthropic's rate card (2026-08-27): Haiku 4.5 $1.00, Sonnet 5 $2.00, Opus 5
$5.00 per Mtok input. **Every current model exceeds the break-even at the
ceiling.** Haiku 4.5 is the closest, at 1.06×.

So §5.4's tension is real and a lever is required. It is worth being precise
about what the tension actually is, because "1.06×" invites hand-waving:

| Packed tokens | Cost at $1.00/Mtok | Against the $0.02 p50 gate |
|---|---|---|
| 21,200 (the ceiling) | $0.0212 | **breaches** by 6% |
| 12,000 | $0.0120 | clears |
| 9,000 (§5.4's p50 illustration) | $0.0090 | clears with room |

## Lever chosen

**Neither lower `CTX_BUDGET` nor raise the gate — yet. Instrument instead.**

§5.4 says it directly: "What determines actual cost is p50 packed tokens, not
the ceiling." `AVAILABLE` is a ceiling reached only by a query that fills the
entire seed and neighbour budget; the gate in §8.2 is a **p50**, not a max. A
query would have to pack ~19,000 tokens before Haiku 4.5 breached $0.02, and
whether real queries do that is a measurement, not a guess.

So the lever is: **keep `CTX_BUDGET = 24_000`, record p50 packed tokens from
step 8 onward, and revisit at step 10 with data.** If measured p50 packed tokens
lands above ~19,000, drop `CTX_BUDGET` to 14,800 (→ 12,000 available →
$0.012/query), which §5.4 lists as the first lever. That is a one-line change in
`config.py`, which is the entire reason † constants live there.

This is deliberately *not* a decision to raise the §8.2 gate. Raising a gate to
match a measurement you have not taken is how a threshold stops meaning
anything.

**Output tokens are not ignored, they are bounded.** `RESERVED_OUT` is 2,000†,
so worst-case output adds 2,000 × $5.00/Mtok = $0.010 — which on its own would
breach the p50 gate at the ceiling. This is recorded here because §5.4's
arithmetic explicitly says "ignoring output", and the reader deserves to know
the ignored term is the same order of magnitude as the one being computed.
Step 9's gate records actual `$/query`, which includes it.

## Trigger

Re-open this decision when any of:

- measured p50 packed tokens > 19,000 (drop `CTX_BUDGET`)
- measured `$/query` p50 > $0.02 over the 45-question golden set (§8.2)
- answer quality on SEMANTIC questions is the binding constraint rather than
  cost — §5.4's third lever (cheap tier for synthesis, expensive on the §6.3
  retry) is then the move, not a blanket upgrade
