"""TS/TSX frames resolved by name, against the real graph (F-027).

`test_frame_names.py` pins the rules with a stub reader. These are the same
frames through the production `QueryRetriever` over the indexed Next.js
fixture — the path `eval/local_gates.py` measured at 0 of 9.

Every frame here points into a bundled chunk (`/app/.next/server/chunks/…`),
which Tier 0 can never place: that is §8.3's reason for scoping the decision
metric to TS/TSX. What is left is the name.
"""

from __future__ import annotations

import json

import pytest

import config
import metrics
from retrieve.frames import is_library_path

from .test_query_pipeline import TS_REPO, ask, indexed, names  # noqa: F401 - fixture

GOLDEN = config.REPO_ROOT / "eval" / "golden" / "golden_45.json"


@pytest.fixture(autouse=True)
def reset():
    metrics.reset()
    yield
    metrics.reset()


def tiers(notes) -> list[str]:
    return [f["tier"] for f in notes["frames"]]


def test_a_component_frame_resolves_to_the_component(indexed):
    _r, seeds, _n, notes = ask(
        indexed, TS_REPO,
        "TypeError: Cannot read properties of null (reading 'name')\n"
        "    at UserCard (/app/.next/server/chunks/1200.js:1:44)",
    )
    assert tiers(notes) == ["name"], notes["frames"]
    assert notes["frames"][0]["badge"] == "~ name match"
    assert names(seeds)[0] == "components.UserCard.UserCard"
    assert metrics.get("trace.tier2.ambiguous") == 0


def test_a_method_frame_resolves_to_the_method(indexed):
    _r, seeds, _n, notes = ask(
        indexed, TS_REPO,
        "Error: describe failed\n"
        "    at UserCardError.describe (/app/.next/server/chunks/1200.js:1:300)",
    )
    assert tiers(notes) == ["name"], notes["frames"]
    assert names(seeds)[0] == "components.UserCard.UserCardError.describe"


def test_a_constructor_frame_resolves_to_the_class(indexed):
    _r, seeds, _n, notes = ask(
        indexed, TS_REPO,
        "Error: boom\n    at new UserCardError (/app/.next/server/chunks/1200.js:1:9)",
    )
    assert tiers(notes) == ["name"], notes["frames"]
    assert names(seeds)[0] == "components.UserCard.UserCardError"


def test_an_awaited_handler_is_parsed_and_still_refused_when_ambiguous(indexed):
    """Two route files export `GET`. `async` used to hide the frame entirely;
    now it is seen, and refused for the right reason."""
    _r, seeds, _n, notes = ask(
        indexed, TS_REPO,
        "TypeError: users.map is not a function\n"
        "    at async GET (/app/.next/server/chunks/9002.js:1:120)",
    )
    assert tiers(notes) == ["unresolved"]
    assert metrics.get("trace.tier2.ambiguous") == 1
    assert seeds, "Tier 3 still answers"


def test_a_frame_with_a_real_path_on_another_machine_is_direct(indexed):
    """An unbundled (dev-server) trace from a Windows checkout: Tier 0, by
    path suffix. The drive letter's colon used to end the match."""
    _r, seeds, _n, notes = ask(
        indexed, TS_REPO,
        "Error: describe failed\n"
        "    at UserCardError.describe (C:\\work\\shop\\components\\UserCard.tsx:49:11)",
    )
    assert tiers(notes) == ["direct"], notes["frames"]
    assert names(seeds)[0] == "components.UserCard.UserCardError.describe"


def test_the_golden_ts_traces_resolve_at_the_rate_recorded(indexed):
    """§8.3's metric over the golden set's seven TS/TSX traces.

    Three of nine repository frames resolve: the two `UserCard` frames and
    `UserCardError.describe`. The other six are refusals the spec asks for —
    three ambiguous `GET`/`POST` handlers, one `formatName` the fixture does
    not define, a minified `t` and an `<anonymous>`. 0.33 is below §8.3's 0.70,
    so the sourcemap tier's trigger fires, which is what the metric is for.
    """
    from retrieve.router import QueryRouter

    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    total = resolved = 0
    for question in golden["questions"]:
        if question.get("language") != "typescript":
            continue
        routed = QueryRouter().route(question["query"])
        _r, _s, _n, notes = ask(indexed, TS_REPO, question["query"])
        for frame, note in zip(routed.frames, notes["frames"]):
            if is_library_path(frame.file_path):
                continue
            total += 1
            resolved += note["tier"] in ("direct", "name")
        if question.get("expect_tier2_ambiguous"):
            assert all(f["tier"] == "unresolved" for f in notes["frames"]), question["id"]

    assert (resolved, total) == (3, 9)
    assert resolved / total < config.DECIDE_TRACE_RESOLUTION_BELOW
