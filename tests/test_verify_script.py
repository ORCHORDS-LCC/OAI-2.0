"""Unit tests for the runner-free local verification helpers."""

from __future__ import annotations

from pathlib import Path

import scripts.verify as verify


def test_secret_regex_detects_quoted_hard_coded_secret() -> None:
    # Fixture built at runtime from short chunks so the source-text
    # public-safety scan does not match its own literal.
    quoted_part = '"' + "abcdefghi" + "jklmnop1234" + '"'
    line = "api_key = " + quoted_part
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


def test_run_check_reports_failure(monkeypatch) -> None:
    class Result:
        returncode = 7

    monkeypatch.setattr(verify.subprocess, "run", lambda *args, **kwargs: Result())
    assert verify.run_check("controlled-failure", ("false-command",)) is False


def test_main_returns_nonzero_on_mandatory_check_failure(monkeypatch) -> None:
    monkeypatch.setattr(verify, "CHECKS", (("controlled-failure", ("false-command",)),))
    monkeypatch.setattr(verify, "run_check", lambda name, command: False)
    assert verify.main() == 1


def test_main_reaches_success_when_all_gates_pass(monkeypatch) -> None:
    monkeypatch.setattr(verify, "CHECKS", (("controlled-pass", ("true-command",)),))
    monkeypatch.setattr(verify, "run_check", lambda name, command: True)
    monkeypatch.setattr(verify, "run_public_safety_scan", lambda: True)
    monkeypatch.setattr(verify, "run_markdown_link_scan", lambda: True)
    assert verify.main() == 0


def test_platform_status_reports_explicit_skip(monkeypatch, capsys) -> None:
    monkeypatch.setattr(verify.sys, "platform", "linux")
    monkeypatch.setattr(verify.platform, "machine", lambda: "x86_64")
    verify.report_platform_check_status()
    output = capsys.readouterr().out
    assert "SKIP platform-mlx" in output
    assert "linux" in output
    assert "x86_64" in output


def test_platform_status_reports_apple_silicon_eligibility(monkeypatch, capsys) -> None:
    monkeypatch.setattr(verify.sys, "platform", "darwin")
    monkeypatch.setattr(verify.platform, "machine", lambda: "arm64")
    verify.report_platform_check_status()
    output = capsys.readouterr().out
    assert "PASS platform-mlx" in output
    assert "macOS arm64" in output


def test_qpipe_compatibility_check_skips_when_default_checkout_missing(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    old_root = verify.ROOT
    try:
        verify.ROOT = tmp_path / "oai"
        verify.ROOT.mkdir()
        monkeypatch.delenv("OAI2_QPIPE_REPO", raising=False)
        assert verify.run_qpipe_compatibility_check() is True
        assert "SKIP qpipe-compat" in capsys.readouterr().out
    finally:
        verify.ROOT = old_root


def test_qpipe_compatibility_check_fails_for_missing_configured_checkout(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    missing = tmp_path / "missing-qpipe"
    monkeypatch.setenv("OAI2_QPIPE_REPO", str(missing))
    assert verify.run_qpipe_compatibility_check() is False
    assert "FAIL qpipe-compat" in capsys.readouterr().out


def test_qpipe_compatibility_check_detects_revision_drift(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    from oai2.knowledge import QPIPE_COMPATIBILITY_SOURCE_REVISION

    qpipe = tmp_path / "q-pipe"
    qpipe.mkdir()
    monkeypatch.setenv("OAI2_QPIPE_REPO", str(qpipe))
    monkeypatch.delenv("OAI2_QPIPE_REVISION_SKIP", raising=False)
    monkeypatch.delenv("OAI2_QPIPE_REVISION", raising=False)

    class Result:
        returncode = 0
        stderr = ""
        stdout = "different-revision\n"

    monkeypatch.setattr(verify.subprocess, "run", lambda *args, **kwargs: Result())
    assert verify.run_qpipe_compatibility_check() is False
    output = capsys.readouterr().out
    assert "source revision changed" in output
    assert QPIPE_COMPATIBILITY_SOURCE_REVISION in output
    assert "OAI2_QPIPE_REVISION_SKIP" in output


def test_qpipe_compatibility_check_accepts_pinned_revision_and_blobs(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    from oai2.knowledge import (
        QPIPE_COMPATIBILITY_SOURCE_BLOBS,
        QPIPE_COMPATIBILITY_SOURCE_REVISION,
    )

    qpipe = tmp_path / "q-pipe"
    qpipe.mkdir()
    monkeypatch.setenv("OAI2_QPIPE_REPO", str(qpipe))
    monkeypatch.delenv("OAI2_QPIPE_REVISION_SKIP", raising=False)
    monkeypatch.delenv("OAI2_QPIPE_REVISION", raising=False)
    outputs = iter(
        [
            QPIPE_COMPATIBILITY_SOURCE_REVISION,
            *QPIPE_COMPATIBILITY_SOURCE_BLOBS.values(),
        ]
    )

    class Result:
        returncode = 0
        stderr = ""

        def __init__(self, stdout: str) -> None:
            self.stdout = stdout + "\n"

    monkeypatch.setattr(
        verify.subprocess,
        "run",
        lambda *args, **kwargs: Result(next(outputs)),
    )
    assert verify.run_qpipe_compatibility_check() is True
    assert "PASS qpipe-compat" in capsys.readouterr().out


def test_qpipe_compatibility_check_revision_override_passes_when_blobs_match(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    from oai2.knowledge import QPIPE_COMPATIBILITY_SOURCE_BLOBS

    operator_head = "operator-current-head-sha-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    qpipe = tmp_path / "q-pipe"
    qpipe.mkdir()
    monkeypatch.setenv("OAI2_QPIPE_REPO", str(qpipe))
    monkeypatch.delenv("OAI2_QPIPE_REVISION_SKIP", raising=False)
    monkeypatch.setenv("OAI2_QPIPE_REVISION", operator_head)
    outputs = iter(
        [
            operator_head,
            *QPIPE_COMPATIBILITY_SOURCE_BLOBS.values(),
        ]
    )

    class Result:
        returncode = 0
        stderr = ""

        def __init__(self, stdout: str) -> None:
            self.stdout = stdout + "\n"

    monkeypatch.setattr(
        verify.subprocess,
        "run",
        lambda *args, **kwargs: Result(next(outputs)),
    )
    assert verify.run_qpipe_compatibility_check() is True
    output = capsys.readouterr().out
    assert "PASS qpipe-compat" in output
    assert "revision override" in output
    assert operator_head[:12] in output


def test_qpipe_compatibility_check_revision_skip_passes_when_blobs_match(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    from oai2.knowledge import QPIPE_COMPATIBILITY_SOURCE_BLOBS

    operator_head = "different-from-pinned-head-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    qpipe = tmp_path / "q-pipe"
    qpipe.mkdir()
    monkeypatch.setenv("OAI2_QPIPE_REPO", str(qpipe))
    monkeypatch.setenv("OAI2_QPIPE_REVISION_SKIP", "1")
    monkeypatch.delenv("OAI2_QPIPE_REVISION", raising=False)
    outputs = iter(
        [
            operator_head,
            *QPIPE_COMPATIBILITY_SOURCE_BLOBS.values(),
        ]
    )

    class Result:
        returncode = 0
        stderr = ""

        def __init__(self, stdout: str) -> None:
            self.stdout = stdout + "\n"

    monkeypatch.setattr(
        verify.subprocess,
        "run",
        lambda *args, **kwargs: Result(next(outputs)),
    )
    assert verify.run_qpipe_compatibility_check() is True
    output = capsys.readouterr().out
    assert "PASS qpipe-compat" in output
    assert "revision skipped" in output
    assert operator_head[:12] in output


def test_qpipe_compatibility_check_revision_skip_still_enforces_blob_drift(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    from oai2.knowledge import QPIPE_COMPATIBILITY_SOURCE_BLOBS

    operator_head = "any-head-cccccccccccccccccccccccccccccccccccccccccccccc"
    wrong_blob = "0" * 40
    pinned_blobs = QPIPE_COMPATIBILITY_SOURCE_BLOBS
    first_blob_path = next(iter(pinned_blobs))
    first_blob_expected = pinned_blobs[first_blob_path]
    qpipe = tmp_path / "q-pipe"
    qpipe.mkdir()
    monkeypatch.setenv("OAI2_QPIPE_REPO", str(qpipe))
    monkeypatch.setenv("OAI2_QPIPE_REVISION_SKIP", "1")
    monkeypatch.delenv("OAI2_QPIPE_REVISION", raising=False)
    outputs = iter(
        [
            operator_head,
            wrong_blob if first_blob_expected != wrong_blob else "f" * 40,
        ]
    )

    class Result:
        returncode = 0
        stderr = ""

        def __init__(self, stdout: str) -> None:
            self.stdout = stdout + "\n"

    monkeypatch.setattr(
        verify.subprocess,
        "run",
        lambda *args, **kwargs: Result(next(outputs)),
    )
    assert verify.run_qpipe_compatibility_check() is False
    output = capsys.readouterr().out
    assert "FAIL qpipe-compat" in output
    assert "source blob changed" in output
