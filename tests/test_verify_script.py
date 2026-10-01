"""Unit tests for the runner-free local verification helpers."""

from __future__ import annotations

from pathlib import Path

import scripts.verify as verify


def test_secret_regex_detects_quoted_hard_coded_secret() -> None:
    line = 'api_key = "abcdefghijklmnop1234"'
    names = [name for name, pattern in verify.SECRET_PATTERNS if pattern.search(line)]
    assert "hard-coded-secret" in names


def test_secret_regex_does_not_flag_placeholder_empty_value() -> None:
    line = "CLOUDFLARE_API_TOKEN="
    assert all(not pattern.search(line) for _, pattern in verify.SECRET_PATTERNS)


def test_iter_text_files_skips_virtualenv_and_cache_dirs(tmp_path: Path) -> None:
    old_root = verify.ROOT
    try:
        verify.ROOT = tmp_path
        (tmp_path / "keep.md").write_text("# ok\n", encoding="utf-8")
        (tmp_path / ".venv").mkdir()
        (tmp_path / ".venv" / "secret.py").write_text("x=1\n", encoding="utf-8")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "x.py").write_text("x=1\n", encoding="utf-8")
        found = {p.relative_to(tmp_path).as_posix() for p in verify.iter_text_files()}
        assert "keep.md" in found
        assert ".venv/secret.py" not in found
        assert "__pycache__/x.py" not in found
    finally:
        verify.ROOT = old_root
