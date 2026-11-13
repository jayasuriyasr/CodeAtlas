"""Every reference a comment makes lands on a real heading of DESIGN.md.

The code cited about 1,185 places in documents that are no longer in the
repository — `§11.2`, `F-015`, `decision 0003`, `R4`, `T5`, `plan §9`. DESIGN.md
keeps their numbering, so each reference now names one of its headings; this
holds the code to that, and keeps the removed documents from being cited again.
"""

from __future__ import annotations

import re
import subprocess

import pytest

import config

PATTERNS = {
    "section": re.compile(r"(?<![\w.])§\d+(?:\.\d+)*"),
    "plan": re.compile(r"\b[Pp]lan §\d+"),
    "finding": re.compile(r"\bF-0\d{2}\b"),
    "decision": re.compile(r"\b[Dd]ecision (000\d)\b"),
    "revision": re.compile(r"\bR[1-7]\b"),
    "defect": re.compile(r"\b(?:T\d{1,2}|B\d{1,2})\b"),
    "v9": re.compile(r"v9\.1 #\d+"),
}
GONE = ("FINDINGS.md", "PROGRESS.md", "CLAUDE.md", "SEI_Architecture",
        "SEI_Implementation_Plan", "docs/decisions")


def normalise(kind: str, match: re.Match[str]) -> str:
    if kind == "decision":
        return f"decision {match.group(1)}"
    if kind == "plan":
        return "plan " + match.group(0).split()[1]
    return match.group(0)


def references(text: str) -> set[str]:
    return {normalise(kind, m) for kind, pattern in PATTERNS.items() for m in pattern.finditer(text)}


def tracked_sources() -> list:
    listed = subprocess.run(["git", "ls-files"], cwd=config.REPO_ROOT, capture_output=True,
                            text=True, check=True).stdout.splitlines()
    keep = (".py", ".yml", ".yaml", ".toml", ".ps1", ".cmd", ".html", ".txt", "Dockerfile")
    return [config.REPO_ROOT / p for p in listed
            if p.endswith(keep) and not p.startswith("tests/fixtures/")]


@pytest.fixture(scope="module")
def headings() -> set[str]:
    design = (config.REPO_ROOT / "DESIGN.md").read_text(encoding="utf-8")
    lines = [line for line in design.splitlines() if line.startswith("#")]
    return {ref.lower() for line in lines for ref in references(line)}


def test_design_md_has_the_headings_the_code_cites(headings):
    missing: dict[str, list[str]] = {}
    for path in tracked_sources():
        text = path.read_text(encoding="utf-8", errors="replace")
        for ref in references(text):
            if ref.lower() not in headings:
                missing.setdefault(ref, []).append(path.relative_to(config.REPO_ROOT).as_posix())
    assert not missing, {ref: files[:3] for ref, files in sorted(missing.items())}


def test_no_file_cites_a_document_the_repository_does_not_have():
    found = []
    for path in tracked_sources():
        if path.name == "test_design_references.py":
            continue                     # it names them in order to look for them
        text = path.read_text(encoding="utf-8", errors="replace")
        found += [(path.name, gone) for gone in GONE if gone in text]
    assert not found, found
