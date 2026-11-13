"""`sei` on any platform: serve, ask and index one project.

The Windows scripts (`sei.cmd`) also install and start Neo4j and Ollama. This
command does not: on macOS and Linux those come from Docker
(`docker compose up -d neo4j`) and the system's Ollama, and `sei check` says
whether they are reachable. Everything else is the same CLI, `api.cli`, with
the five flags a project needs filled in:

    sei serve ~/code/shop            console at http://127.0.0.1:8000, follows saves
    sei ask   ~/code/shop "what calls charge_card"
    sei index ~/code/shop            re-read every file (Sync Project)
    sei check                        is Neo4j up, which local models are pulled
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import PurePath, PureWindowsPath


def repo_id_for(folder: str) -> str:
    """The index name for a project: its folder's name, made safe."""
    name = (PureWindowsPath(folder) if "\\" in folder else PurePath(folder)).name
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name) or "project"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sei", description="Ask questions about a codebase.")
    verbs = parser.add_subparsers(dest="verb", required=True)

    def project_verb(name: str, help_text: str) -> argparse.ArgumentParser:
        verb = verbs.add_parser(name, help=help_text)
        verb.add_argument("project", help="the project folder")
        verb.add_argument("--repo", help="index name (default: the folder's name)")
        verb.add_argument("--ai", choices=("auto", "local", "cloud", "off"),
                          help="which models answer (default: auto)")
        return verb

    serve = project_verb("serve", "run the console for a project and follow its saves")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--no-watch", action="store_true", help="do not re-index on save")
    ask = project_verb("ask", "ask one question about a project")
    ask.add_argument("question")
    project_verb("index", "re-read every file of a project and re-link every edge")
    verbs.add_parser("check", help="report whether Neo4j and the local models are reachable")
    return parser


def build_cli_args(argv: list[str]) -> list[str]:
    """`api.cli` arguments for a launcher command line (not `check`)."""
    args = _parser().parse_args(argv)
    if args.verb == "check":
        return []
    out = ["--repo", args.repo or repo_id_for(args.project), "--root", args.project]
    if args.ai:
        out += ["--ai", args.ai]
    if args.verb == "serve":
        out += ["--serve", "--host", args.host, "--port", str(args.port)]
        if not args.no_watch:
            out.append("--watch")
    elif args.verb == "index":
        out.append("--sync")
    elif args.verb == "ask":
        out.append(args.question)
    return out


def check() -> int:
    """Say what is reachable. Non-zero when Neo4j, which nothing works without, is not."""
    from neo4j import GraphDatabase

    from config import LOCAL_EMBED_MODEL, LOCAL_LLM_MODEL, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
    from index.ollama import has_model, installed_models

    ok = True
    try:
        with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
            driver.verify_connectivity()
        print(f"Neo4j        : reachable at {NEO4J_URI}")
    except Exception as exc:                       # noqa: BLE001 - reported, not raised
        ok = False
        print(f"Neo4j        : NOT reachable at {NEO4J_URI} ({type(exc).__name__}). "
              f"Start one with: docker compose up -d neo4j")
    models = installed_models()
    if models is None:
        print("local models : no Ollama server — answers need a hosted key or Ollama")
    else:
        for name in (LOCAL_LLM_MODEL, LOCAL_EMBED_MODEL):
            state = "pulled" if has_model(models, name) else f"missing — run: ollama pull {name}"
            print(f"{name:<13}: {state}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["check"]:
        return check()
    from api import cli

    return cli.main(build_cli_args(argv))


if __name__ == "__main__":                         # pragma: no cover
    raise SystemExit(main())
