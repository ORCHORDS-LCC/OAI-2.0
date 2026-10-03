"""The lint gate must be enforced by the suite, not merely declared.

WHY THIS FILE EXISTS
--------------------
``#264`` found a real ``NameError`` (``FakeCluster.enable_respawn`` read an
undefined ``pgid`` where its parameter was ``pgrp``) sitting in the tree.
The lint rule that can see that class of defect -- ``F821`` -- was configured
in ``.pre-commit-config.yaml``, and nothing ran it: no ``pre-commit`` module,
no binary, no ``.git/hooks/pre-commit``. The declared pin (``v0.7.4``) was not
even the version the project actually lints with (``uv run ruff``, 0.16.9).

A gate that only exists in configuration is not a gate. A gate that is
declared but *skips* when it cannot run is the same failure wearing a
different hat, so this test deliberately does not skip.

WHAT THIS DOES
--------------
Runs the exact command the repo's own ``scripts/verify.py`` declares for the
``ruff`` check, and fails the suite when it reports errors. The command is
read from ``verify.py`` rather than duplicated, so the test and the script
cannot drift apart.

This makes the lint gate part of ``pytest``, which is the repository's
documented acceptance path (``pytest -W error``). No hosted runner, no
pre-commit installation, and no extra tooling are required: ``uv`` is already
mandatory for the suite.
"""

from __future__ import annotations

import ast
import pathlib
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
VERIFY_PY = REPO_ROOT / "scripts" / "verify.py"

# Which key in verify.py's CHECKS tuple this test enforces.
GATE_KEY = "ruff"


def _declared_gate_command() -> list[str]:
    """Extract the gate command from scripts/verify.py's CHECKS.

    Parsing the real source rather than hard-coding the command means that
    if somebody changes what verify.py runs, this test follows instead of
    quietly checking something nobody cares about any more.
    """
    tree = ast.parse(VERIFY_PY.read_text(encoding="utf-8"), filename=str(VERIFY_PY))

    def _names(node: ast.AST) -> list[str]:
        """Collect bound target names from either an Assign or an AnnAssign."""
        if isinstance(node, ast.Assign):
            return [t.id for t in node.targets if isinstance(t, ast.Name)]
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            return [node.target.id]
        return []

    for node in ast.walk(tree):
        if "CHECKS" not in _names(node):
            continue
        value = getattr(node, "value", None)
        if value is None:
            continue
        for element in ast.walk(value):
            if (
                isinstance(element, ast.Tuple)
                and len(element.elts) == 2
                and isinstance(element.elts[0], ast.Constant)
                and element.elts[0].value == GATE_KEY
            ):
                parts = ast.literal_eval(element.elts[1])
                return [str(p) for p in parts]
    pytest.fail(
        f"{VERIFY_PY} no longer declares a {GATE_KEY!r} check; this gate test "
        "must be updated to follow the script, not silently stop running"
    )


def test_verify_py_still_declares_the_gate() -> None:
    """Fail loudly if the script drops the check this test mirrors."""
    command = _declared_gate_command()
    assert "ruff" in command, f"declared gate is not ruff: {command}"
    assert "check" in command, f"declared gate is not a lint run: {command}"


def test_lint_gate_is_clean() -> None:
    """The gate itself.

    Deliberately does not skip. A gate that skips when its tool is missing is
    the exact failure mode #264 documented: a defect class hiding behind a
    check that appears to exist.
    """
    command = _declared_gate_command()
    try:
        proc = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except FileNotFoundError as exc:
        pytest.fail(
            f"cannot run the declared lint gate {command!r}: {exc}. "
            "The repo's documented acceptance path is `uv run`, so the gate "
            "tool must be resolvable. Fix the environment rather than "
            "skipping: a gate that cannot run is not a gate."
        )

    output = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, (
        "lint gate failed; `ruff check` must be clean before pushing\n"
        f"command: {' '.join(command)}\n\n{output}"
    )
