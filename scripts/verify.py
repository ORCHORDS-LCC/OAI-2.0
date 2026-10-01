#!/usr/bin/env python3
"""Runner-free local verification for OAI-2.0.

This script is intentionally local/manual-first. It does not contact GitHub
Actions or require a hosted/self-hosted runner.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".md", ".py", ".toml", ".yaml", ".yml", ".json", ".txt"}
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "build",
    "dist",
}
SECRET_PATTERNS = (
    ("private-key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("github-token", re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]+\b")),
    ("provider-token", re.compile(r"\b(?:sk|xoxb)-[A-Za-z0-9_-]{16,}\b")),
    ("aws-access-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    (
        "hard-coded-secret",
        re.compile(
            r'(?i)\b(?:api[_-]?key|secret|password|access[_-]?token)\s*[:=]\s*'
            r'["\'][^"\']{16,}["\']'
        ),
    ),
)
MARKDOWN_LINK = re.compile(r"\[[^]]+\]\(([^)\s]+)(?:\s+['\"][^)]*['\"])?\)")


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
    completed = subprocess.run(command, check=False, cwd=ROOT)
    if completed.returncode:
        print(f"FAIL {name}: exit {completed.returncode}", flush=True)
        return False
    print(f"PASS {name}", flush=True)
    return True


def iter_text_files() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not any(part in SKIP_DIRS for part in path.parts)
        and path.suffix.lower() in TEXT_SUFFIXES
    ]


def run_public_safety_scan() -> bool:
    failures: list[tuple[Path, int, str]] = []
    for path in iter_text_files():
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, start=1):
            for name, pattern in SECRET_PATTERNS:
                if pattern.search(line):
                    failures.append((path.relative_to(ROOT), line_number, name))
    if failures:
        for path, line_number, name in failures:
            print(f"FAIL public-safety: {path}:{line_number} ({name})")
        return False
    print("PASS public-safety")
    return True


def run_markdown_link_scan() -> bool:
    failures: list[tuple[Path, str]] = []
    for path in ROOT.rglob("*.md"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8")
        for target in MARKDOWN_LINK.findall(text):
            if (
                target.startswith(("#", "/", "mailto:"))
                or "://" in target
                or "/blob/" in target
                or "/tree/" in target
            ):
                continue
            target_path = (path.parent / target.split("#", 1)[0]).resolve()
            if not target_path.exists():
                failures.append((path.relative_to(ROOT), target))
    if failures:
        for path, target in failures:
            print(f"FAIL docs-links: {path} -> {target}")
        return False
    print("PASS docs-links")
    return True


def report_platform_check_status() -> None:
    """Report whether Apple-Silicon-specific checks are eligible locally."""

    machine = platform.machine().lower()
    if sys.platform == "darwin" and machine == "arm64":
        print(
            "PASS platform-mlx: macOS arm64; MLX-specific pytest cases are eligible",
            flush=True,
        )
        return
    print(
        "SKIP platform-mlx: MLX-specific checks require macOS arm64 "
        f"(detected platform={sys.platform}, machine={machine or 'unknown'})",
        flush=True,
    )


def run_qpipe_compatibility_check() -> bool:
    """Verify an available q-pipe checkout matches the pinned import contract."""

    from oai2.knowledge import (
        QPIPE_COMPATIBILITY_SOURCE_BLOBS,
        QPIPE_COMPATIBILITY_SOURCE_REVISION,
    )

    configured = os.getenv("OAI2_QPIPE_REPO")
    qpipe_root = (
        Path(configured).expanduser().resolve()
        if configured
        else (ROOT.parent / "q-pipe").resolve()
    )
    if not qpipe_root.exists():
        if configured:
            print(
                f"FAIL qpipe-compat: configured checkout does not exist: {qpipe_root}",
                flush=True,
            )
            return False
        print(
            f"SKIP qpipe-compat: no sibling checkout at {qpipe_root}",
            flush=True,
        )
        return True

    def git_output(*args: str) -> str | None:
        completed = subprocess.run(
            ("git", "-C", str(qpipe_root), *args),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip()
            print(
                "FAIL qpipe-compat: git "
                + " ".join(args)
                + f" exited {completed.returncode}: {detail}",
                flush=True,
            )
            return None
        return completed.stdout.strip()

    head = git_output("rev-parse", "HEAD")
    if head is None:
        return False
    if head != QPIPE_COMPATIBILITY_SOURCE_REVISION:
        print(
            "FAIL qpipe-compat: source revision changed "
            f"(pinned={QPIPE_COMPATIBILITY_SOURCE_REVISION}, current={head})",
            flush=True,
        )
        return False

    for relative_path, expected_blob in QPIPE_COMPATIBILITY_SOURCE_BLOBS.items():
        actual_blob = git_output("hash-object", relative_path)
        if actual_blob is None:
            return False
        if actual_blob != expected_blob:
            print(
                "FAIL qpipe-compat: source blob changed "
                f"({relative_path}: pinned={expected_blob}, current={actual_blob})",
                flush=True,
            )
            return False

    print(
        f"PASS qpipe-compat: {QPIPE_COMPATIBILITY_SOURCE_REVISION}",
        flush=True,
    )
    return True


def main() -> int:
    for name, command in CHECKS:
        if not run_check(name, command):
            return 1
    report_platform_check_status()
    if not run_qpipe_compatibility_check():
        return 1
    if not run_public_safety_scan():
        return 1
    if not run_markdown_link_scan():
        return 1
    print("\nALL LOCAL CHECKS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
