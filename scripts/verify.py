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
    (
        "backend-smoke",
        ("uv", "run", "python", "scripts/backend_smoke.py"),
    ),
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
    """Verify an available q-pipe checkout matches the pinned import contract.

    The pinned revision in ``oai2/knowledge/qpipe_import.py`` is a
    reproducibility hint. Operators may opt into the more permissive
    contract-only mode via:

    - ``OAI2_QPIPE_REVISION=<sha>`` — accept the listed revision as the
      pinned target. Use this when a sibling q-pipe checkout is at a known
      current commit whose pinned-file blobs match.
    - ``OAI2_QPIPE_REVISION_SKIP=1`` — skip the revision comparison
      altogether and only enforce the pinned-file blob hashes. Use this
      when an operator wants contract-only verification across arbitrary
      q-pipe working trees (e.g. multi-session automation).

    The pinned-file blob-hash check always runs regardless of the revision
    gate. A genuine blob drift still fails the check.
    """

    # Resolve the q-pipe checkout first so a missing default checkout can
    # SKIP without ever touching the package's pinned-constants source.
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

    # The pinned constants live in `oai2/knowledge/qpipe_import.py`. They
    # are simple string / dict assignments, so parse them with `ast` and
    # evaluate the literals rather than importing `oai2` — importing pulls
    # in the package's runtime deps (pydantic, etc.) which are only present
    # in the `uv run` environment, not in this script's bare interpreter.
    import ast

    constants_path = ROOT / "oai2" / "knowledge" / "qpipe_import.py"
    tree = ast.parse(constants_path.read_text(encoding="utf-8"))
    pinned_revision: str | None = None
    pinned_blobs: dict[str, str] | None = None
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            target_id = node.targets[0].id
            if target_id == "QPIPE_COMPATIBILITY_SOURCE_REVISION":
                value = ast.literal_eval(node.value)
                if isinstance(value, str):
                    pinned_revision = value
            elif target_id == "QPIPE_COMPATIBILITY_SOURCE_BLOBS":
                value = ast.literal_eval(node.value)
                if isinstance(value, dict) and all(
                    isinstance(k, str) and isinstance(v, str)
                    for k, v in value.items()
                ):
                    pinned_blobs = value
    if pinned_revision is None or pinned_blobs is None:
        print(
            "FAIL qpipe-compat: pinned constants missing from "
            f"{constants_path.relative_to(ROOT)}",
            flush=True,
        )
        return False

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

    # Operators may override the pinned revision or skip it entirely. The
    # blob-hash check below is the actual contract surface and always runs.
    revision_skip = os.getenv("OAI2_QPIPE_REVISION_SKIP") == "1"
    revision_override = os.getenv("OAI2_QPIPE_REVISION") or None
    effective_revision = revision_override if revision_override else pinned_revision
    if not revision_skip and head != effective_revision:
        print(
            "FAIL qpipe-compat: source revision changed "
            f"(pinned={effective_revision}, current={head}); "
            "set OAI2_QPIPE_REVISION=<sha> or OAI2_QPIPE_REVISION_SKIP=1",
            flush=True,
        )
        return False

    for relative_path, expected_blob in pinned_blobs.items():
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

    short_head = head[:12] if len(head) >= 12 else head
    if revision_skip:
        print(
            f"PASS qpipe-compat: blobs match pinned={pinned_revision[:12]} "
            f"(revision skipped via OAI2_QPIPE_REVISION_SKIP=1, head={short_head})",
            flush=True,
        )
    elif revision_override:
        short_override = revision_override[:12] if len(revision_override) >= 12 else revision_override
        print(
            f"PASS qpipe-compat: revision override "
            f"OAI2_QPIPE_REVISION={short_override} (head={short_head}, blobs match)",
            flush=True,
        )
    else:
        print(
            f"PASS qpipe-compat: {pinned_revision}",
            flush=True,
        )
    return True


def run_gateway_reach_check() -> bool:
    """Verify the configured public inference gateway is reachable.

    Skips entirely when ``OAI2_GATEWAY_API_KEY`` is not set so the
    base local gate stays green without credentials. When the key is
    present the check performs a single ``GET /v1/models`` against the
    configured base URL and confirms a 2xx response. The bearer token is
    read directly from the environment — never from the file path —
    and never echoed into the report.
    """

    api_key = os.getenv("OAI2_GATEWAY_API_KEY", "").strip()
    if not api_key:
        print(
            "SKIP gateway-reach: OAI2_GATEWAY_API_KEY not set",
            flush=True,
        )
        return True
    base_url = os.getenv(
        "OAI2_GATEWAY_BASE_URL",
        "https://api.orchords.com",
    ).strip().rstrip("/") or "https://api.orchords.com"
    expected_model = os.getenv("OAI2_GATEWAY_MODEL", "oai-2.0").strip() or "oai-2.0"

    try:
        import httpx
    except ImportError:
        print(
            "FAIL gateway-reach: httpx not installed in this environment",
            flush=True,
        )
        return False

    try:
        response = httpx.get(
            f"{base_url}/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15.0,
        )
    except httpx.HTTPError as exc:
        print(
            "FAIL gateway-reach: transport error "
            f"({type(exc).__name__}) against {base_url}",
            flush=True,
        )
        return False

    if response.status_code < 200 or response.status_code >= 300:
        print(
            "FAIL gateway-reach: "
            f"{base_url}/v1/models returned HTTP {response.status_code}",
            flush=True,
        )
        return False
    payload = response.json()
    if not isinstance(payload, dict) or "data" not in payload:
        print(
            "FAIL gateway-reach: response payload missing 'data' array",
            flush=True,
        )
        return False
    data = payload["data"]
    if not isinstance(data, list):
        print(
            "FAIL gateway-reach: 'data' is not an array",
            flush=True,
        )
        return False
    model_ids = [
        item.get("id")
        for item in data
        if isinstance(item, dict) and item.get("id") is not None
    ]
    exposed = set(model_ids)
    if expected_model not in exposed:
        print(
            "FAIL gateway-reach: expected model "
            f"{expected_model!r} not exposed by {base_url} "
            f"(saw {model_ids!r})",
            flush=True,
        )
        return False
    stale = sorted(exposed - {expected_model})
    if stale:
        print(
            "FAIL gateway-reach: stale model ids exposed by "
            f"{base_url}: {stale!r} (only {expected_model!r} should be listed; "
            "see issue #237).",
            flush=True,
        )
        return False
    print(
        "PASS gateway-reach: "
        f"{base_url} exposes exactly {expected_model!r} "
        f"(of {len(model_ids)} models)",
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
    if not run_gateway_reach_check():
        return 1
    print("\nALL LOCAL CHECKS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
