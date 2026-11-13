"""`sei` on any platform (`api/launcher.py`, the console script in pyproject).

Setup was Windows-only: `sei.cmd` runs a PowerShell script. Elsewhere there was
`python -m api.cli` with five flags to remember, and `pip install .` installed
nothing, because pyproject declared no dependencies. The launcher is a thin
verb layer over the same CLI; Neo4j and the models come from Docker or the
system, which `sei check` reports on.
"""

from __future__ import annotations

import re
import tomllib

import pytest

import config
from api.launcher import build_cli_args, repo_id_for


def test_serve_runs_the_console_and_follows_saves():
    args = build_cli_args(["serve", "/work/shop"])
    assert args[:2] == ["--repo", "shop"]
    assert args[2:4] == ["--root", "/work/shop"]
    assert "--serve" in args and "--watch" in args
    assert args[args.index("--port") + 1] == "8000"


def test_serve_options_pass_through():
    args = build_cli_args(["serve", "/work/shop", "--port", "9001", "--repo", "mine",
                           "--host", "0.0.0.0", "--no-watch"])
    assert args[:2] == ["--repo", "mine"]
    assert "--watch" not in args
    assert args[args.index("--port") + 1] == "9001"
    assert args[args.index("--host") + 1] == "0.0.0.0"


def test_ask_passes_the_question():
    args = build_cli_args(["ask", "/work/shop", "what calls charge_card"])
    assert args[-1] == "what calls charge_card" and "--serve" not in args


def test_index_is_sync_project():
    assert "--sync" in build_cli_args(["index", "/work/shop"])


def test_ai_mode_passes_through():
    args = build_cli_args(["ask", "/work/shop", "q", "--ai", "off"])
    assert args[args.index("--ai") + 1] == "off"


@pytest.mark.parametrize("folder, expected", [
    ("/work/shop", "shop"), ("/work/demo shop", "demo_shop"), ("C:\\code\\my-app", "my-app"),
])
def test_the_repo_id_is_the_folder_name_made_safe(folder, expected):
    assert repo_id_for(folder) == expected


def test_an_unknown_command_is_a_usage_error():
    with pytest.raises(SystemExit) as stopped:
        build_cli_args(["deploy", "/work/shop"])
    assert stopped.value.code == 2


def test_pyproject_installs_the_command_and_its_dependencies():
    project = tomllib.loads((config.REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["scripts"]["sei"] == "api.launcher:main"
    declared = {re.split(r"[~=<>!\[ ]", d, maxsplit=1)[0].lower() for d in project["dependencies"]}
    required = {
        re.split(r"[~=<>!\[ ]", line, maxsplit=1)[0].lower()
        for line in (config.REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert required <= declared, required - declared
