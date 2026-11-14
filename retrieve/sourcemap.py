"""§8.3's sourcemap tier: a compiled position mapped back to its source line.

A Next.js server frame names a chunk (`.next/server/chunks/4821.js:1:5678`)
and a function name that is either shared (`GET`, exported by every route
file) or minified (`t`). Neither can be resolved by name without guessing. The
build's source map can resolve it by position, which is not a guess: it is the
compiler's own record of which source line produced that column.

Source map v3, decoded here directly (base64 VLQ, ~30 lines) rather than through
a dependency. Index maps (`sections`) are not supported and resolve nothing.
"""

from __future__ import annotations

import json
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path

_B64 = {c: i for i, c in enumerate(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/")}


def decode_vlq(segment: str) -> list[int]:
    """One mapping segment's fields. Each is 5 bits per character, low bit the sign."""
    values: list[int] = []
    value = shift = 0
    for char in segment:
        digit = _B64[char]
        value += (digit & 31) << shift
        if digit & 32:
            shift += 5
            continue
        values.append(-(value >> 1) if value & 1 else value >> 1)
        value = shift = 0
    return values


@dataclass(frozen=True)
class Original:
    source: int          # index into the map's `sources`
    line: int            # 1-based
    column: int          # 0-based


def original_position(source_map: dict, line: int, column: int) -> Original | None:
    """The source position of a generated `line:column`, both 1-based as V8 prints them.

    The mapping in force at a column is the last segment that starts at or
    before it. A segment with no source (one field) means generated code that
    came from nowhere — the bundler's own glue — and maps to nothing.
    """
    if "sections" in source_map or line < 1 or column < 1:
        return None
    source = source_line = source_column = 0
    found: Original | None = None
    for generated_line, text in enumerate(source_map.get("mappings", "").split(";")):
        generated_column = 0
        for segment in text.split(",") if text else ():
            fields = decode_vlq(segment)
            generated_column += fields[0]
            if len(fields) >= 4:
                source += fields[1]
                source_line += fields[2]
                source_column += fields[3]
            if generated_line == line - 1 and generated_column <= column - 1:
                found = (Original(source, source_line + 1, source_column)
                         if len(fields) >= 4 else None)
        if generated_line == line - 1:
            return found
    return None


_WEBPACK = re.compile(r"^webpack://[^/]*/")


def source_path(source_map: dict, index: int, map_path: Path, repo_root: Path) -> str | None:
    """`sources[index]` as a repo-relative POSIX path, or None outside the repository.

    webpack (Next.js) writes `webpack://_N_E/./app/api/users/route.ts?abc`,
    relative to the project root; tsc writes a path relative to the map,
    under `sourceRoot`.
    """
    sources = source_map.get("sources") or []
    if not 0 <= index < len(sources) or not sources[index]:
        return None
    raw = sources[index].split("?", 1)[0]
    if _WEBPACK.match(raw):
        rel = posixpath.normpath(_WEBPACK.sub("", raw))
        return None if rel.startswith("..") else rel
    raw = posixpath.join(source_map.get("sourceRoot") or "", raw)
    raw = re.sub(r"^file://", "", raw)
    try:
        resolved = (map_path.parent / raw).resolve()
        return resolved.relative_to(Path(repo_root).resolve()).as_posix()
    except ValueError:
        return None


def map_for(file_path: str, repo_root: Path) -> Path | None:
    """The `.map` beside a frame's compiled file, found under this repository.

    The frame was usually recorded elsewhere (`/app/.next/…` in a container),
    so the part from the build directory onwards is what is looked up here.
    """
    path = re.sub(r"^file://", "", file_path.replace("\\", "/"))
    marker = path.find("/.next/")
    candidates = []
    if marker >= 0:
        candidates.append(Path(repo_root) / path[marker + 1:])
    root = str(Path(repo_root)).replace("\\", "/").rstrip("/")
    if path.lower().startswith(root.lower() + "/"):
        candidates.append(Path(path))
    for compiled in candidates:
        mapped = compiled.with_name(compiled.name + ".map")
        if mapped.is_file():
            return mapped
    return None


def load(map_path: Path) -> dict | None:
    try:
        return json.loads(map_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def is_stale(source_map: dict, index: int, repo_root: Path, rel_path: str) -> bool:
    """The map embeds the source it was built from; a different file today is stale.

    Without `sourcesContent` there is nothing to compare, and the map is used.
    """
    contents = source_map.get("sourcesContent") or []
    if index >= len(contents) or contents[index] is None:
        return False
    try:
        current = (Path(repo_root) / rel_path).read_text(encoding="utf-8")
    except OSError:
        return True
    return current.replace("\r\n", "\n") != contents[index].replace("\r\n", "\n")
