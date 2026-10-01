#!/usr/bin/env python3
"""Runner-free local verification for OAI-2.0.

This script is intentionally local/manual-first. It does not contact GitHub
Actions or require a hosted/self-hosted runner.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence


CHECKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sync", ("uv", "sync", "--extra", "dev")),
    ("ruff", ("uv", "run", "ruff", "check", "oai2", "tests", "scripts")),
    (
        "mypy",
        ("uv", "run", "mypy", "--ignore-missing-imports", "oai2"),
    ),
    ("pytest", ("uv", "run", "pytest", "-W", "error")),
)


def run_check(name: str, command: Sequence[str]) -> bool:
    print(f"\n== {name}: {' '.join(command)} ==", flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        print(f"FAIL {name}: exit {completed.returncode}", flush=True)
        return False
    print(f"PASS {name}", flush=True)
    return True


def main() -> int:
    for name, command in CHECKS:
        if not run_check(name, command):
            return 1
    print("\nALL LOCAL CHECKS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
