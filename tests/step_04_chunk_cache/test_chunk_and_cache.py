"""Step 4 — the chunker and the embedding cache.

The headline defect this step exists to prevent is the original cache defect:
a context header built from the file's import block rather than the symbol's
own imports, so one added `import logging` invalidates every cached embedding
in the file. §8.2 gates the hit rate at >=0.90 because "the failure is silent
and cannot be noticed, only measured" — these are the measurements.
"""

from __future__ import annotations

import pytest

import metrics
from adapters.python import PythonAdapter
from config import EMBED_DIM
from index.cache import EmbeddingCache, embed_batch
from index.chunker import (
    REDACTED,
    SCRUB_RULES_VERSION,
    body_hash,
    build_context_header,
    cache_key,
    chunk_file,
    header_hash,
    scrub_secrets,
    split_overflow,
)
from index.providers import ApproxCodeTokenizer, HashEmbeddingProvider

REPO = "repo_step4"


@pytest.fixture(scope="module")
def adapter() -> PythonAdapter:
    return PythonAdapter()


@pytest.fixture(scope="module")
def tokenizer() -> ApproxCodeTokenizer:
    return ApproxCodeTokenizer()


@pytest.fixture
def cache(tmp_path) -> EmbeddingCache:
    cache = EmbeddingCache(tmp_path / "embeddings.sqlite", model="test-model")
    yield cache
    cache.close()


@pytest.fixture
def provider() -> HashEmbeddingProvider:
    return HashEmbeddingProvider()


@pytest.fixture(autouse=True)
def reset_metrics():
    metrics.reset()
    yield
    metrics.reset()


def parse_and_chunk(adapter, tokenizer, source: str, rel_path: str = "authx/views.py"):
    parsed = adapter.parse(REPO, rel_path, source.encode())
    return chunk_file(parsed, tokenizer)


def sym(parsed, qualified_name):
    found = [s for s in parsed.symbols if s.qualified_name == qualified_name]
    assert len(found) == 1, f"{qualified_name}: {[s.qualified_name for s in parsed.symbols]}"
    return found[0]


BASE = '''\
import os
from rest_framework.response import Response
from .models import User


class LoginView(APIView):
    """Login."""

    def post(self, request):
        """Handle a login."""
        user = User.objects.get(pk=1)
        return Response({"id": user.pk})

    def get(self, request):
        return Response({"pid": os.getpid()})
'''


# --------------------------------------------------------------------------
# The original cache defect
# --------------------------------------------------------------------------


def test_added_import_invalidates_only_the_file_chunk(adapter, tokenizer, cache, provider):
    """The defect this whole design exists to avoid, at its smallest scale.

    Adding `import logging` must invalidate exactly one entry: the `:Module`
    symbol, whose chunk *is* the import block (§3.4.5). Every function and class
    in the file keeps its cached embedding.

    Stated as a count rather than a rate on purpose. A four-symbol file cannot
    reach 90% while one symbol legitimately misses, and rounding that away would
    hide the thing being measured — the repo-level rate is the next test.
    """
    before = parse_and_chunk(adapter, tokenizer, BASE)
    embed_batch(REPO, {"authx/views.py": before}, cache=cache, provider=provider)
    metrics.reset()

    after = parse_and_chunk(adapter, tokenizer, "import logging\n" + BASE)
    embed_batch(REPO, {"authx/views.py": after}, cache=cache, provider=provider)

    assert metrics.get("cache.miss.total") == 1, (
        "exactly one miss expected — the module symbol; "
        f"got {metrics.get('cache.miss.total')}"
    )
    assert metrics.get("cache.lookups") == len(after.symbols)


def test_added_import_hit_rate_over_the_whole_repo(
    adapter, tokenizer, cache, provider, fixtures_dir
):
    """Plan §4 acceptance criterion: >=90% hit rate on a one-import edit.

    Measured the way the plan states it — over a full index of the Django
    fixture, not over one file. §8.2 gates this at >=0.90 because the failure is
    silent: a broken cache key costs money and latency and changes no output.
    """
    repo = fixtures_dir / "repos" / "django_min"
    sources = {
        path.relative_to(repo).as_posix(): path.read_bytes()
        for path in sorted(repo.rglob("*.py"))
    }

    def index(files: dict[str, bytes]) -> None:
        batch = {
            rel: chunk_file(adapter.parse(REPO, rel, data), tokenizer)
            for rel, data in files.items()
        }
        embed_batch(REPO, batch, cache=cache, provider=provider)

    index(sources)                                   # cold
    metrics.reset()

    edited = dict(sources)
    edited["authx/views.py"] = b"import logging\n" + sources["authx/views.py"]
    index(edited)

    lookups, hits = metrics.get("cache.lookups"), metrics.get("cache.hits")
    hit_rate = hits / lookups
    assert lookups > 60, f"fixture too small to be evidence: {lookups} symbols"
    assert hit_rate >= 0.90, (
        f"one added import dropped the hit rate to {hit_rate:.2%} "
        f"({hits}/{lookups}); §3.4's per-symbol header is not holding"
    )


def test_second_run_over_the_whole_repo_is_a_total_hit(
    adapter, tokenizer, cache, provider, fixtures_dir
):
    """Plan §4 acceptance: ~100% on an unchanged second run."""
    repo = fixtures_dir / "repos" / "django_min"
    sources = {
        path.relative_to(repo).as_posix(): path.read_bytes()
        for path in sorted(repo.rglob("*.py"))
    }

    def index() -> None:
        batch = {
            rel: chunk_file(adapter.parse(REPO, rel, data), tokenizer)
            for rel, data in sources.items()
        }
        embed_batch(REPO, batch, cache=cache, provider=provider)

    index()
    metrics.reset()
    index()

    assert metrics.get("cache.lookups") > 60
    assert metrics.get("cache.hits") == metrics.get("cache.lookups")
    assert metrics.get("cache.miss.total") == 0


def test_unrelated_symbol_headers_are_byte_identical_after_an_import(adapter, tokenizer):
    """The mechanism behind the previous test, checked directly.

    A hit rate is an aggregate and can stay high for the wrong reason. This
    pins the actual invariant: the header of a symbol that does not use the new
    import does not change at all.
    """
    before = parse_and_chunk(adapter, tokenizer, BASE)
    after = parse_and_chunk(adapter, tokenizer, "import logging\n" + BASE)

    for qualified_name in (
        "authx.views.LoginView",
        "authx.views.LoginView.post",
        "authx.views.LoginView.get",
    ):
        assert build_context_header(sym(before, qualified_name)) == (
            build_context_header(sym(after, qualified_name))
        ), qualified_name
        assert sym(before, qualified_name).header_hash == (
            sym(after, qualified_name).header_hash
        ), qualified_name


def test_module_symbol_does_see_the_new_import(adapter, tokenizer):
    """The file chunk is *supposed* to change — §3.4.5 makes it the import block.

    Without this, "nothing invalidated" could be achieved by a header that
    ignores imports entirely, which would pass the test above for the wrong
    reason.
    """
    before = parse_and_chunk(adapter, tokenizer, BASE)
    after = parse_and_chunk(adapter, tokenizer, "import logging\n" + BASE)
    assert sym(before, "authx.views").header_hash != sym(after, "authx.views").header_hash


# --------------------------------------------------------------------------
# v9.1 T11 — the base-class rename
# --------------------------------------------------------------------------


def test_base_class_rename_invalidates(adapter, tokenizer):
    """T11. `class X(A)` -> `class X(B)` must produce a cache miss.

    §3.3 makes `enclosing_signature` "full, incl. bases" and §4.2 hashes it, so
    a method's header changes when its class's bases change. Stripping the bases
    would leave the header identical and serve an embedding of a method that no
    longer means the same thing.
    """
    before = parse_and_chunk(adapter, tokenizer, BASE)
    after = parse_and_chunk(adapter, tokenizer, BASE.replace("(APIView)", "(GenericAPIView)"))

    post_before, post_after = sym(before, "authx.views.LoginView.post"), sym(
        after, "authx.views.LoginView.post"
    )
    assert post_before.enclosing_signature == "class LoginView(APIView):"
    assert post_after.enclosing_signature == "class LoginView(GenericAPIView):"
    assert post_before.header_hash != post_after.header_hash
    assert cache_key(post_before) != cache_key(post_after)

    # The UID is unchanged: it is the same method (§3.2).
    assert post_before.uid == post_after.uid


def test_class_rename_invalidates_its_methods(adapter, tokenizer):
    """The same argument one level up — the class name is in the header too."""
    before = parse_and_chunk(adapter, tokenizer, BASE)
    after = parse_and_chunk(adapter, tokenizer, BASE.replace("LoginView", "SignInView"))
    assert sym(before, "authx.views.LoginView.post").header_hash != (
        sym(after, "authx.views.SignInView.post").header_hash
    )


# --------------------------------------------------------------------------
# What body_hash does and does not notice
# --------------------------------------------------------------------------


def test_comment_change_invalidates(adapter, tokenizer):
    """Comments are in `body_hash` by design.

    §4.2: "Comments retained: they carry retrieval signal." A comment that
    changes changes the embedded text, so it must change the key — otherwise
    the cache would serve a vector of text nobody will see again.
    """
    plain = parse_and_chunk(adapter, tokenizer, "def f(a):\n    return a\n")
    noted = parse_and_chunk(
        adapter, tokenizer, "def f(a):\n    # the identity function\n    return a\n"
    )
    assert sym(plain, "authx.views.f").body_hash != sym(noted, "authx.views.f").body_hash


def test_trailing_whitespace_does_not_invalidate(adapter, tokenizer):
    """`_normalize` rstrips every line and strips the whole body.

    An editor that trims trailing whitespace on save would otherwise
    re-embed every symbol it touched, for no change a reader could see.
    """
    clean = parse_and_chunk(adapter, tokenizer, "def f(a):\n    return a\n")
    dirty = parse_and_chunk(adapter, tokenizer, "def f(a):   \n    return a    \n\n\n")
    assert sym(clean, "authx.views.f").body_hash == sym(dirty, "authx.views.f").body_hash


def test_line_endings_do_not_invalidate(adapter, tokenizer):
    lf = parse_and_chunk(adapter, tokenizer, "def f(a):\n    return a\n")
    crlf = parse_and_chunk(adapter, tokenizer, "def f(a):\r\n    return a\r\n")
    assert sym(lf, "authx.views.f").body_hash == sym(crlf, "authx.views.f").body_hash


def test_body_change_invalidates(adapter, tokenizer):
    before = parse_and_chunk(adapter, tokenizer, "def f(a):\n    return a\n")
    after = parse_and_chunk(adapter, tokenizer, "def f(a):\n    return a + 1\n")
    assert sym(before, "authx.views.f").body_hash != sym(after, "authx.views.f").body_hash


# --------------------------------------------------------------------------
# §9.2 — scrub before hash
# --------------------------------------------------------------------------


SECRETS = '''\
def connect():
    aws = "AKIAIOSFODNN7EXAMPLE"
    gh = "ghp_abcdefghijklmnopqrstuvwxyz0123456789"
    stripe = "sk_live_abcdefghijklmnopqrst"
    return aws, gh, stripe
'''


def test_scrub_before_hash(adapter, tokenizer):
    """§9.2's ordering: redaction precedes hashing.

    Two things follow, and both are checked. The unredacted form is never
    stored on the symbol, and a *changed secret* still changes nothing —
    because both values redact to the same marker, which is the point.
    """
    parsed = parse_and_chunk(adapter, tokenizer, SECRETS)
    fn = sym(parsed, "authx.views.connect")

    assert "AKIAIOSFODNN7EXAMPLE" not in fn.source_code
    assert "ghp_abcdefghijklmnopqrstuvwxyz0123456789" not in fn.source_code
    assert REDACTED in fn.source_code
    assert "AKIAIOSFODNN7EXAMPLE" not in (fn.chunk_text or "")

    rotated = parse_and_chunk(
        adapter, tokenizer, SECRETS.replace("AKIAIOSFODNN7EXAMPLE", "AKIAJJJJJJJJJJJJJJJJ")
    )
    assert fn.body_hash == sym(rotated, "authx.views.connect").body_hash, (
        "rotating a redacted secret must not churn the cache"
    )


def test_scrub_rule_version_is_in_the_header_hash():
    """§9.2: "rule changes invalidate the cache".

    Redaction-before-hash only delivers that for symbols a *new* rule actually
    matches. Folding the rule-set version into the header hash makes it hold for
    every symbol, which is what the sentence claims.
    """
    from adapters.base import Symbol

    def build() -> Symbol:
        return Symbol(
            uid="u", repo_id=REPO, rel_path="a.py", qualified_name="a.f", name="f",
            arity=0, ordinal=0, kind="function", signature="def f():",
            docstring=None, source_code="def f():\n    pass\n",
            enclosing_signature=None,
        )

    import index.chunker as chunker

    first = header_hash(build())
    original, chunker.SCRUB_RULES_VERSION = chunker.SCRUB_RULES_VERSION, "999"
    try:
        assert header_hash(build()) != first
    finally:
        chunker.SCRUB_RULES_VERSION = original
    assert SCRUB_RULES_VERSION == original


@pytest.mark.parametrize(
    "secret",
    [
        "AKIAIOSFODNN7EXAMPLE",
        "ghp_abcdefghijklmnopqrstuvwxyz0123456789",
        "sk_live_abcdefghijklmnopqrst",
        "xoxb-123456789012-abcdefghijkl",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
    ],
)
def test_scrub_catches_known_key_shapes(secret):
    assert secret not in scrub_secrets(f'token = "{secret}"')


def test_scrub_leaves_ordinary_code_alone():
    """A scrub rule that fires on ordinary code costs retrieval signal."""
    ordinary = 'def get_user_by_id(user_id):\n    return User.objects.get(pk=user_id)\n'
    assert scrub_secrets(ordinary) == ordinary


# --------------------------------------------------------------------------
# §3.4.2 — the header itself
# --------------------------------------------------------------------------


def test_context_header_shape(adapter, tokenizer):
    """§3.4.2's worked example, in the same order."""
    parsed = parse_and_chunk(adapter, tokenizer, BASE)
    header = build_context_header(sym(parsed, "authx.views.LoginView.post"))
    lines = header.split("\n")

    assert lines[0] == (
        "# authx/views.py :: class LoginView(APIView) :: def post(self, request)"
    )
    assert lines[1].startswith("# imports used: ")
    assert ".models.User" in lines[1]
    assert "rest_framework.response.Response" in lines[1]
    assert "os" not in lines[1].replace("rest_framework.response", ""), (
        "post does not reference os"
    )


def test_every_header_component_is_in_the_header_hash(adapter, tokenizer):
    """A field hashed but not shown invalidates a cache entry for nothing.

    A field shown but not hashed serves a stale chunk. Both are silent, so the
    two sets have to be checked against each other rather than assumed equal.
    """
    parsed = parse_and_chunk(adapter, tokenizer, BASE)
    post = sym(parsed, "authx.views.LoginView.post")
    header = build_context_header(post)

    assert post.rel_path in header
    assert post.enclosing_signature.rstrip(":") in header
    assert post.signature.rstrip(":") in header
    for used in post.used_imports:
        assert used in header


def test_class_chunk_lists_method_signatures_not_bodies(adapter, tokenizer):
    """§3.4.4. The bodies are separate symbols; embedding them twice dilutes both."""
    parsed = parse_and_chunk(adapter, tokenizer, BASE)
    text = sym(parsed, "authx.views.LoginView").chunk_text

    assert "def post(self, request):" in text
    assert "def get(self, request):" in text
    assert "User.objects.get" not in text, "a method body leaked into the class chunk"


def test_module_chunk_is_imports_plus_top_level_symbols(adapter, tokenizer):
    """§3.4.5."""
    parsed = parse_and_chunk(adapter, tokenizer, BASE)
    text = sym(parsed, "authx.views").chunk_text
    assert "class LoginView(APIView):" in text
    assert "rest_framework.response.Response" in text


# --------------------------------------------------------------------------
# §3.4.3 — overflow split
# --------------------------------------------------------------------------


def _long_body(statements: int) -> str:
    """A body with unambiguous top-level statements and real nested blocks."""
    out = []
    for i in range(statements):
        out.append(f"    if condition_{i}(payload):")
        out.append(f"        value_{i} = transform_{i}(payload, {i})")
        out.append(f"        results.append(value_{i})")
        out.append(f"    total = total + {i}")
    return "\n".join(out)


def test_overflow_split_at_statement_boundary(tokenizer):
    """Never mid-block.

    A chunk that starts inside an `if` body reads as a different program than
    the one in the file, and the embedding is then of code that does not exist.
    """
    header = "# a.py :: def big(payload)"
    body = "def big(payload):\n" + _long_body(120)

    parts = split_overflow(body, header, tokenizer, cap=300)
    assert len(parts) > 1, "the fixture must actually overflow the cap"

    for part in parts:
        assert part.startswith(header), "the header is repeated on every part"
        first_code_line = part.split("\n")[len(header.split("\n"))]
        assert not first_code_line.startswith("        "), (
            f"split landed inside a nested block: {first_code_line!r}"
        )


def test_no_split_when_under_the_cap(tokenizer):
    header = "# a.py :: def small(a)"
    body = "def small(a):\n    return a\n"
    assert split_overflow(body, header, tokenizer, cap=1200) == [f"{header}\n{body}"]


def test_split_counts_are_recorded(tokenizer):
    """§8.3 needs the numbers; a split that nothing counts cannot inform a gate."""
    metrics.reset()
    split_overflow("def big(p):\n" + _long_body(120), "# h", tokenizer, cap=300)
    assert metrics.get("chunk.overflow_split") == 1
    assert metrics.get("chunk.overflow_parts") >= 1


def test_tokenizer_is_not_a_length_heuristic(tokenizer):
    """§5.4: never `len // 4`.

    Dense code and prose of the same length must not produce the same count, or
    the count is measuring characters with extra steps.
    """
    dense = "x=[a[i]**2for i in r(9)if i%3]"
    prose = "the quick brown fox jumps over"
    assert len(dense) == len(prose)
    assert tokenizer.count(dense) != tokenizer.count(prose)
    assert tokenizer.count("") == 0


# --------------------------------------------------------------------------
# The cache itself
# --------------------------------------------------------------------------


def test_cache_round_trips(cache):
    cache.put_many([("k1", [0.1] * EMBED_DIM), ("k2", [0.2] * EMBED_DIM)])
    got = cache.get_many(["k1", "k2", "missing"])
    assert set(got) == {"k1", "k2"}
    assert got["k1"][0] == pytest.approx(0.1)
    assert cache.get_many([]) == {}


def test_cache_is_keyed_by_model(tmp_path):
    """Two models produce different vectors for identical text.

    Returning one model's vector for the other's query would put a wrong answer
    into a vector index that has no way to notice — §12.11's "full re-index"
    exists for exactly this reason.
    """
    first = EmbeddingCache(tmp_path / "c.sqlite", model="model-a")
    first.put_many([("k", [0.5] * EMBED_DIM)])
    first.close()

    second = EmbeddingCache(tmp_path / "c.sqlite", model="model-b")
    assert second.get_many(["k"]) == {}, "a different model must miss"
    second.close()


def test_cache_survives_reopen(tmp_path):
    """It is a cost control, so it has to outlive the process."""
    first = EmbeddingCache(tmp_path / "c.sqlite", model="m")
    first.put_many([("k", [0.25] * EMBED_DIM)])
    first.close()

    second = EmbeddingCache(tmp_path / "c.sqlite", model="m")
    assert "k" in second.get_many(["k"])
    second.close()


def test_cache_handles_more_keys_than_sqlite_parameters(cache):
    """SQLite caps host parameters; the chunking must not be discovered live."""
    keys = [f"k{i}" for i in range(1500)]
    cache.put_many((k, [0.0] * EMBED_DIM) for k in keys)
    assert len(cache.get_many(keys)) == 1500


# --------------------------------------------------------------------------
# embed_batch
# --------------------------------------------------------------------------


def test_second_run_is_a_full_cache_hit(adapter, tokenizer, cache, provider):
    """Plan §4 acceptance: ~100% on an unchanged second run."""
    parsed = parse_and_chunk(adapter, tokenizer, BASE)
    batch = {"authx/views.py": parsed}

    first = embed_batch(REPO, batch, cache=cache, provider=provider)
    assert len(provider.calls) == 1

    metrics.reset()
    again = parse_and_chunk(adapter, tokenizer, BASE)
    second = embed_batch(REPO, {"authx/views.py": again}, cache=cache, provider=provider)

    assert len(provider.calls) == 1, "no second provider call"
    assert metrics.get("cache.hits") == metrics.get("cache.lookups")
    assert {u: list(v) for u, v in first.items()} == {
        u: list(v) for u, v in second.items()
    }


def test_embed_batch_returns_one_vector_per_symbol(adapter, tokenizer, cache, provider):
    parsed = parse_and_chunk(adapter, tokenizer, BASE)
    vectors = embed_batch(REPO, {"authx/views.py": parsed}, cache=cache, provider=provider)

    assert set(vectors) == {s.uid for s in parsed.symbols}
    assert all(len(v) == EMBED_DIM for v in vectors.values())


def test_dimension_mismatch_is_refused(adapter, tokenizer, cache):
    """§12.11: changing the embedding model is a full re-index, not a config change."""
    wrong = HashEmbeddingProvider(dimensions=768, name="other-model")
    parsed = parse_and_chunk(adapter, tokenizer, BASE)
    with pytest.raises(ValueError, match="full re-index"):
        embed_batch(REPO, {"authx/views.py": parsed}, cache=cache, provider=wrong)


def test_header_only_miss_share_is_counted(adapter, tokenizer, cache, provider):
    """§8.3's decision metric for warm cache reuse.

    A miss whose stored `body_hash` still matches was a header change. §4.2 says
    that path "fires almost only on file moves and class renames" — this test
    creates one deliberately, so the counter has something to count.
    """
    before = parse_and_chunk(adapter, tokenizer, BASE)
    embed_batch(REPO, {"authx/views.py": before}, cache=cache, provider=provider)

    stored = {s.uid: s.body_hash for s in before.symbols}

    class StubReader:
        def body_hashes(self, repo_id, uids):
            return {uid: stored[uid] for uid in uids if uid in stored}

    # A class rename: headers change, bodies do not.
    metrics.reset()
    renamed = parse_and_chunk(adapter, tokenizer, BASE.replace("(APIView)", "(Base)"))
    embed_batch(
        REPO, {"authx/views.py": renamed}, cache=cache, provider=provider,
        reader=StubReader(),
    )

    assert metrics.get("cache.miss.total") > 0
    assert metrics.get("cache.miss.header_only") > 0, (
        "a class rename is the canonical header-only miss"
    )
    assert metrics.get("cache.miss.header_only") <= metrics.get("cache.miss.total")


def test_provider_returning_the_wrong_count_is_refused(adapter, tokenizer, cache):
    class ShortProvider:
        name, dimensions = "short", EMBED_DIM

        def embed(self, texts):
            return [[0.0] * EMBED_DIM]      # one vector regardless of input

    parsed = parse_and_chunk(adapter, tokenizer, BASE)
    with pytest.raises(ValueError, match="vectors for"):
        embed_batch(REPO, {"authx/views.py": parsed}, cache=cache, provider=ShortProvider())


def test_empty_batch_is_a_noop(cache, provider):
    assert embed_batch(REPO, {}, cache=cache, provider=provider) == {}
    assert provider.calls == []
