"""Process-local counters.

The spec calls `metrics.incr(...)` directly in §4.1, §4.2, §4.5, §5.1 and §5.4,
and §9.4 names the handful that need a counter rather than an OTel span. This is
that surface, kept deliberately small: a dict of ints, resettable, with no
exporter. §9.4's spans are step 9's work; nothing before then needs more.

`move.unresolvable` is alarmed in §9.4 — it must be identically zero — so
`snapshot()` exists to let a test assert on the whole set rather than on the one
counter it remembered to check.
"""

from __future__ import annotations

import threading
from collections import defaultdict

_lock = threading.Lock()
_counters: defaultdict[str, int] = defaultdict(int)


def incr(name: str, amount: int = 1) -> None:
    if amount == 0:
        return
    with _lock:
        _counters[name] += amount


def get(name: str) -> int:
    with _lock:
        return _counters[name]


def snapshot() -> dict[str, int]:
    with _lock:
        return {k: v for k, v in _counters.items() if v}


def reset() -> None:
    with _lock:
        _counters.clear()
