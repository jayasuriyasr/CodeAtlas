# Step 1 — probe results

Actual output, recorded by `tests/step_01_ground_truth/conftest.py`.
Do not hand-edit: re-run `pytest tests/step_01_ground_truth/ -v`.

- Run at: 2026-08-28T23:38:44+05:30
- Server: Neo4j Kernel 5.26.0 (community)

## `break_even`

**Question.** Does the chosen model's input rate clear §5.4's ~$0.94/Mtok break-even at the 21,200-token ceiling?

Statement(s) run:

```cypher
break_even = $0.02 / 21,200 tok = $0.9434 per 1M input tokens
```

Actual output:

```
AVAILABLE (ceiling): 21,200 tokens
break-even input rate: $0.9434 / Mtok
LLM_MODEL: claude-haiku-4-5
EMBED_MODEL: voyage-code-2
EMBED_DIM: 1536
configured input rate: 1.0
configured output rate: 5.0
clears break-even at the ceiling: False
```

**Verdict.** claude-haiku-4-5 at $1.0/Mtok exceeds the break-even at the ceiling; a §5.4 lever is recorded in docs/decisions/0002 and p50 packed tokens decides actual cost

## `environment`

**Question.** Which Neo4j build are the probes above actually describing?

Statement(s) run:

```cypher
CALL dbms.components() YIELD name, versions, edition
```

Actual output:

```
name: Neo4j Kernel
version: 5.26.0
edition: community
throwaway-db mode: scoped-wipe
test database: neo4j
multi-database available: False
```

**Verdict.** Neo4j Kernel 5.26.0 community; per-module isolation via scoped-wipe

## `probe_count_subquery`

**Question.** Is `COUNT { (n)--() }` available on this minor?

Statement(s) run:

```cypher
MATCH (n:Symbol {uid: 'probe_a'}) RETURN COUNT { (n)--() } AS degree

MATCH (s:Symbol)
WHERE s.repo_id = 'probe'
SET s.degree = COUNT { (s)--() }
RETURN s.uid AS uid, s.degree AS degree ORDER BY uid
```

Actual output:

```
RETURN COUNT { (n)--() }: 1
SET degree on probe_a: 1
SET degree on probe_b: 1
```

**Verdict.** SUPPORTED — legal in a projection and on the right of SET

## `probe_dynamic_map_key`

**Question.** Does `$edge_weights[rel]` resolve a dynamic key on a map parameter?

Statement(s) run:

```cypher
WITH $rel AS rel
RETURN $edge_weights[rel] AS direct,
       coalesce($edge_weights[rel], $default) AS with_default
```

Actual output:

```
rel='CALLS' -> direct: 1.0
rel='CALLS' -> with_default: 1.0
rel='NO_SUCH_REL' -> direct: None
rel='NO_SUCH_REL' -> with_default: 0.5
```

**Verdict.** SUPPORTED — dynamic key resolves; a missing key yields null and coalesce covers it

## `probe_exists_subquery`

**Question.** Is `NOT EXISTS { MATCH (c:Symbol {uid: $u}) }` available?

Statement(s) run:

```cypher
MATCH (s:Symbol {uid: 'probe_a'})
WHERE NOT EXISTS { MATCH (c:Symbol {uid: $u}) }
RETURN s.uid AS uid
```

Actual output:

```
target uid free -> rows: ['probe_a']
target uid taken -> rows: []
```

**Verdict.** SUPPORTED — predicate admits the free case and rejects the collision

## `probe_merge_multitype`

**Question.** Confirm `MERGE (a)-[:A|B]->(b)` is rejected — the per-type split in §11.2 3b is required, not defensive.

Statement(s) run:

```cypher
MATCH (a:Symbol {uid: 'probe_a'}), (b:Symbol {uid: 'probe_b'})
MERGE (a)-[r:CALLS|IMPORTS]->(b)
RETURN type(r) AS t

MATCH (a:Symbol {uid: 'probe_a'})-[r:CALLS|IMPORTS]->(b:Symbol)
RETURN type(r) AS t
```

Actual output:

```
MERGE multi-type: CypherSyntaxError: {code: Neo.ClientError.Statement.SyntaxError} {message: A single relationship type must be specified for MERGE (line 2, column 10 (offset: 72))
"MERGE (a)-[r:CALLS|IMPORTS]->(b)"
          ^}
MATCH multi-type: ['CALLS']
```

**Verdict.** REJECTED as expected — per-type MERGE is required

## `probe_tsx_grammar`

**Question.** Does `tree-sitter-typescript` (tsx) emit the node types the chunker assumes for a real Next.js route + component?

Statement(s) run:

```cypher
Language(tree_sitter_typescript.language_tsx())
parse(fixtures\repos\nextjs_min\app\api\users\route.ts)
parse(fixtures\repos\nextjs_min\components\UserCard.tsx)
```

Actual output:

```
route.ts root type: program
route.ts parse errors: none
UserCard.tsx root type: program
UserCard.tsx parse errors: none
distinct named node types: 76
node types emitted: accessibility_modifier, arguments, array, array_pattern, array_type, arrow_function, as_expression, assignment_expression, await_expression, binary_expression, call_expression, class_body, class_declaration, class_heritage, comment, export_statement, expression_statement, extends_clause, false, formal_parameters, function_declaration, function_signature, function_type, generic_type, identifier, if_statement, import_clause, import_specifier, import_statement, interface_body, interface_declaration, jsx_attribute, jsx_closing_element, jsx_element, jsx_expression, jsx_opening_element, jsx_self_closing_element, jsx_text, lexical_declaration, literal_type, member_expression, method_definition, named_imports, null, number, object, object_assignment_pattern, object_pattern, object_type, optional_parameter, pair, parenthesized_expression, predefined_type, program, property_identifier, property_signature, required_parameter, rest_pattern, return_statement, shorthand_property_identifier, shorthand_property_identifier_pattern, statement_block, string, string_fragment, super, template_string, template_substitution, ternary_expression, this, true, type_annotation, type_arguments, type_identifier, unary_expression, union_type, variable_declarator
overload declarations of `parse`: [('function_signature', 28, 'parse'), ('function_signature', 29, 'parse'), ('function_declaration', 30, 'parse')]
first named node of UserCard.tsx: expression_statement: "'use client';"
```

**Verdict.** ALL ASSUMED NODE TYPES PRESENT

## `probe_vector_ddl`

**Question.** Does the vector index take `ON (s.code_vec)` or `ON s.code_vec`?

Statement(s) run:

```cypher
CREATE VECTOR INDEX probe_vec_parens IF NOT EXISTS
  FOR (s:Symbol) ON (s.code_vec)
  OPTIONS { indexConfig: { `vector.dimensions`: 1536, `vector.similarity_function`: 'cosine' } }

CREATE VECTOR INDEX probe_vec_bare IF NOT EXISTS
  FOR (s:Symbol) ON s.code_vec
  OPTIONS { indexConfig: { `vector.dimensions`: 1536, `vector.similarity_function`: 'cosine' } }

MATCH (s:Symbol {uid: 'probe_a'}) SET s.code_vec = $vec RETURN size(s.code_vec) AS n

CALL db.index.vector.queryNodes('probe_vec_parens', 1, $vec) YIELD node, score RETURN node.uid AS uid, score
```

Actual output:

```
ON (s.code_vec): accepted
ON s.code_vec: accepted
SHOW INDEXES probe_vec_parens: type=VECTOR labels=['Symbol'] properties=['code_vec'] indexConfig={'vector.hnsw.m': 16, 'vector.hnsw.ef_construction': 100, 'vector.dimensions': 1536, 'vector.similarity_function': 'COSINE', 'vector.quantization.enabled': True}
SET s.code_vec -> size(): 1536
queryNodes rows: [('probe_a', 1.0)]
```

**Verdict.** accepted form(s): ON (s.code_vec), ON s.code_vec

## `schema_apply`

**Question.** Does spec §11.3 apply end to end on a fresh database, and is it idempotent on a second run?

Actual output:

```
statements applied: 8
constraints created: ['file_key', 'symbol_uid']
indexes created: ['symbol_code_vec', 'symbol_lines', 'symbol_name', 'symbol_origin', 'symbol_repo_path', 'symbol_search']
other catalogue entries: ['index_343aff4e', 'index_f7700477']
catalogue identical after second apply: True
symbol_code_vec indexConfig: {'vector.hnsw.m': 16, 'vector.hnsw.ef_construction': 100, 'vector.dimensions': 1536, 'vector.similarity_function': 'COSINE', 'vector.quantization.enabled': True}
symbol_search indexConfig: {'fulltext.analyzer': 'standard-no-stop-words', 'fulltext.eventually_consistent': False}
```

**Verdict.** APPLIES CLEAN AND IS IDEMPOTENT

