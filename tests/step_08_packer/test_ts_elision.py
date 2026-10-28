"""§5.4's L1 for TypeScript.

The packer parsed every symbol with the Python grammar. On TypeScript that finds
no definitions at all, so a TS block's "elided" rendering was its full body and
the ladder had one rung for half the supported languages.
"""

from __future__ import annotations

from adapters.base import Symbol
from index.providers import ApproxCodeTokenizer
from pack.packer import pack
from pack.render import Block, elide_nested_bodies, language_of, render, truncate_marked

TS_SOURCE = """\
export async function GET(request: NextRequest): Promise<NextResponse> {
  const users = await listUsers();
  const rows = users.map((user) => {
    const name = user.first + ' ' + user.last;
    const email = user.email.toLowerCase();
    return { name, email };
  });
  const short = users.filter((u) => u.active);
  return NextResponse.json(rows);
}
"""


def ts_symbol(source: str = TS_SOURCE, rel_path: str = "app/api/users/route.ts") -> Symbol:
    return Symbol(
        uid="uid_get", repo_id="r", rel_path=rel_path,
        qualified_name="app.api.users.route.GET", name="GET", arity=1, ordinal=0,
        kind="function", signature="export async function GET(request: NextRequest)",
        docstring=None, source_code=source, enclosing_signature=None,
    )


def test_language_is_chosen_by_extension():
    assert language_of("a/b.ts") == "typescript"
    assert language_of("a/b.tsx") == "typescript"
    assert language_of("a/b.py") == "python"
    assert language_of("a/b.pyi") == "python"


def test_nested_ts_block_bodies_are_elided_with_a_marker():
    text, count = elide_nested_bodies(TS_SOURCE, "typescript")
    assert count == 1
    assert "⟨3 lines elided⟩" in text
    assert "toLowerCase" not in text
    assert "u.active" in text, "an expression-bodied arrow is not a block; it stays"


def test_the_python_grammar_finds_nothing_in_typescript():
    """The defect, stated as a test: the old behaviour was a silent no-op."""
    _text, count = elide_nested_bodies(TS_SOURCE, "python")
    assert count == 0


def test_render_elided_uses_the_symbols_language():
    elided = render(Block(ts_symbol(), 1.0), "elided")
    full = render(Block(ts_symbol(), 1.0), "full")
    assert "⟨3 lines elided⟩" in elided
    tok = ApproxCodeTokenizer()
    assert tok.count(elided) < tok.count(full)


def test_ts_truncation_marker_is_a_ts_comment():
    tok = ApproxCodeTokenizer()
    text, cost = truncate_marked(Block(ts_symbol(), 1.0), 60, tok)
    assert "// ⟨truncated:" in text
    assert "# ⟨truncated:" not in text
    assert cost == tok.count(text)


def test_a_large_ts_seed_fits_at_l1_where_l0_would_not():
    body = "\n".join(f"    const v{i} = compute({i});" for i in range(60))
    source = (
        "export function outer(x: number) {\n"
        "  const helper = () => {\n" + body + "\n  };\n"
        "  return helper();\n}\n"
    )
    block = Block(ts_symbol(source, "lib/outer.ts"), 1.0)
    tok = ApproxCodeTokenizer()
    full_cost = tok.count(render(block, "full"))
    budget = int(full_cost * 0.6 / 0.6 / 0.4) // 2      # per-seed cap below L0
    context = pack([block], [], tok, available=budget)
    assert context.blocks and context.blocks[0].level in ("elided", "truncated")
    assert context.packed_tokens <= budget
