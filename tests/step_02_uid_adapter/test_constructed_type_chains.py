"""Only a plain dotted name is a constructor: `x = a.b(...).first()` types nothing.

Found while checking T3's contradictions: `token = ApiToken.objects.filter(…)
.first()` typed `token` as a "class" named `…filter(user=…, key=key).first`.
The writer refused that lookup (no such class), so no wrong edge was written,
but the junk lookup replaced the ordinary `token.revoke` hint — and with it the
record that the file holds an unresolved call to `revoke`.
"""

from __future__ import annotations

from adapters.grammars import python_parser
from adapters.python import PythonAdapter
from adapters.receivers import constructed_type


def _value(code: bytes):
    tree = python_parser().parse(code)
    assignment = tree.root_node.named_children[0].named_children[0]
    return assignment.child_by_field_name("right")


def test_a_chained_call_is_not_a_constructor():
    code = b"token = ApiToken.objects.filter(user=u).first()\n"
    assert constructed_type(_value(code), code) is None


def test_a_call_on_a_subscript_is_not_a_constructor():
    code = b"w = factories[0]()\n"
    assert constructed_type(_value(code), code) is None


def test_plain_and_dotted_names_still_are():
    for code, expected in ((b"w = Widget()\n", "Widget"), (b"w = shop.Widget(1)\n", "shop.Widget")):
        assert constructed_type(_value(code), code) == expected


def test_the_ordinary_hint_survives_for_an_untyped_receiver():
    source = (b"from pkg.models import ApiToken\n\n\n"
              b"def delete(key):\n    token = ApiToken.objects.filter(key=key).first()\n"
              b"    token.revoke()\n")
    parsed = PythonAdapter().parse("r", "pkg/views.py", source)
    hints = [e.target_hint for e in parsed.edges if e.target_hint and e.target_hint.endswith(".revoke")]
    assert hints == ["token.revoke"]          # the ordinary hint, no junk lookup
