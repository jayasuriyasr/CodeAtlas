"""Probe recording.

Plan §1: "one statement each, pass or fail recorded verbatim", and working rule
4: *record actual output, never expected output*. So each probe writes what the
server actually returned - including the exact exception text when a statement
is rejected - into PROBE_RESULTS.md, which is the artefact PROGRESS.md quotes.

The recorder is deliberately dumb: it stores strings produced by the probe and
never interprets them. A probe that fails still records; the assertion runs
after the recording, so a red probe leaves evidence rather than a stack trace.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

RESULTS_MD = Path(__file__).parent / "PROBE_RESULTS.md"
RESULTS_JSON = Path(__file__).parent / "probe_results.json"


@dataclass
class ProbeRecord:
    name: str
    question: str
    statements: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    verdict: str = "not run"

    def statement(self, cypher: str) -> None:
        self.statements.append(cypher.strip())

    def observe(self, label: str, value: object) -> None:
        """Record one raw observation. `value` is stringified, never parsed."""
        self.observations.append(f"{label}: {value!r}" if not isinstance(value, str)
                                 else f"{label}: {value}")

    def conclude(self, verdict: str) -> None:
        self.verdict = verdict


class ProbeLog:
    def __init__(self) -> None:
        self.records: dict[str, ProbeRecord] = {}

    def record(self, name: str, question: str) -> ProbeRecord:
        rec = ProbeRecord(name=name, question=question)
        self.records[name] = rec
        return rec


@pytest.fixture(scope="session")
def probe_log(request) -> ProbeLog:
    log = ProbeLog()
    yield log
    _write(log, request.config)


def _write(log: ProbeLog, config) -> None:
    if not log.records:
        return
    stamp = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    server = getattr(config, "_sei_server_line", "unrecorded")

    lines = [
        "# Step 1 — probe results",
        "",
        "Actual output, recorded by `tests/step_01_ground_truth/conftest.py`.",
        "Do not hand-edit: re-run `pytest tests/step_01_ground_truth/ -v`.",
        "",
        f"- Run at: {stamp}",
        f"- Server: {server}",
        "",
    ]
    for name in sorted(log.records):
        rec = log.records[name]
        lines += [f"## `{rec.name}`", "", f"**Question.** {rec.question}", ""]
        if rec.statements:
            lines += ["Statement(s) run:", "", "```cypher"]
            lines += ["\n\n".join(rec.statements)]
            lines += ["```", ""]
        lines += ["Actual output:", "", "```"]
        lines += rec.observations or ["(nothing recorded — probe did not run)"]
        lines += ["```", "", f"**Verdict.** {rec.verdict}", ""]

    RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    RESULTS_JSON.write_text(
        json.dumps(
            {n: r.__dict__ for n, r in sorted(log.records.items())}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )

