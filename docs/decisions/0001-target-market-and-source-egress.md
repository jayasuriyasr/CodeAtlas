# 0001 — Target market and source egress

**Status:** Accepted · 2026-08-27
**Answers:** spec §12.0 (source leaves the building) and §12.1 (Community Edition
limits). The spec is explicit that these are *one* decision, not two.
**Gate:** plan §1 — "Market decision recorded".

## Decision

**Target teams comfortable with a hosted API.** Full function bodies go to a
third-party provider, protected by the §9.2 regex scrub and nothing stronger.

Consequences, stated as the spec states them rather than softened:

- **Self-hosted embeddings are *not* Phase 0 scope.** They move behind §10.2's
  "Customer forbids source egress" trigger. §12.11 is the reason this matters:
  swapping the embedding model is a full re-index — drop and recreate the vector
  index, re-embed every symbol, invalidate the whole cache — so deferring it
  defers the expensive half of S2, not the cheap half.
- **Neo4j Community Edition is acceptable for the MVP.** Its two binding limits
  (§9.1, §12.1) are single-user-database and no clustering. Both are tolerable
  for a single-tenant MVP; neither is tolerable for the first enterprise
  customer, which is exactly when §10.2's "First enterprise customer" trigger
  fires and tenancy becomes multiple engineer-weeks plus a licensing decision.
- **The correlated failure domain stands.** Graph, vector, and fulltext are one
  process with no fallback (§9.3's last row). Splitting the vector store is
  deferred, and §12.1 explains why that is not a free change.

## What this does not decide

§12.0's honest reading is that regex scrubbing "will not catch a credential
shaped like ordinary text, one assembled from parts, or a proprietary
algorithm". Choosing the hosted path accepts that risk for this market; it does
not eliminate it. Any external claim about SEI carries the phrase "source code
is sent to a hosted API".

## Immediate effect on the build

- `graph/schema.cypher` keeps `vector.dimensions: 1536` (see 0002).
- No vLLM, no self-hosted embedding server, no GPU dependency in steps 1–10.
- `docker-compose.yml` pins the **Community** image, and the throwaway-database
  fixture degrades accordingly — recorded as F-001 in `FINDINGS.md`.
