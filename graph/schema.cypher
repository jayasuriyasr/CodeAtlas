// SEI schema — spec §11.3, verbatim (unchanged by v10.1: the revision
// touched §11.1, §11.2 and §4.4, not the schema).
//
// Statements are separated by `;` and applied one at a time by
// graph/bootstrap.py. Every statement is IF NOT EXISTS, so the whole file is
// idempotent: applying it twice is a no-op (gate: step 1).
//
// Every index leads with repo_id, per §9.1. Tenancy depends on it.

CREATE CONSTRAINT symbol_uid IF NOT EXISTS
  FOR (s:Symbol) REQUIRE s.uid IS UNIQUE;

CREATE CONSTRAINT file_key IF NOT EXISTS
  FOR (f:File) REQUIRE (f.repo_id, f.rel_path) IS UNIQUE;

// Every index leads with repo_id, per §9.1.
CREATE INDEX symbol_repo_path IF NOT EXISTS
  FOR (s:Symbol) ON (s.repo_id, s.rel_path);

CREATE INDEX symbol_lines IF NOT EXISTS
  FOR (s:Symbol) ON (s.repo_id, s.rel_path, s.start_line);   // §5.1 Tier 0

CREATE INDEX symbol_name IF NOT EXISTS
  FOR (s:Symbol) ON (s.repo_id, s.name);                     // §5.1 Tier 2

CREATE INDEX symbol_origin IF NOT EXISTS
  FOR (s:Symbol) ON (s.repo_id, s.origin_path, s.epoch);     // §11.2 steps 4-6

// body_hash stays a property but is not indexed: the index served only
// similarity detection and warm reuse, both deferred.

CREATE VECTOR INDEX symbol_code_vec IF NOT EXISTS
  FOR (s:Symbol) ON (s.code_vec)
  OPTIONS { indexConfig: {
    // Must match the embedding model named in the eval config. Changing models
    // means dropping this index and re-embedding every symbol (§12.11).
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
  }};

CREATE FULLTEXT INDEX symbol_search IF NOT EXISTS
  FOR (s:Symbol) ON EACH [s.search_text, s.qualified_name]
  OPTIONS { indexConfig: { `fulltext.analyzer`: 'standard-no-stop-words' }};
