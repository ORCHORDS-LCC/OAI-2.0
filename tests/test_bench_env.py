"""Unit tests for scripts/bench.py environment diagnostics (Refs #235).

These tests pin the fail-fast guard added for the uv-managed environment
requirement: running bench.py with a bare interpreter must exit with an
actionable message instead of a ModuleNotFoundError traceback.
"""

from __future__ import annotations

import builtins
from typing import Any

import pytest

import scripts.bench as bench


def _run_one_with_broken_import(
    monkeypatch: pytest.MonkeyPatch,
    missing_error: ModuleNotFoundError,
) -> str:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "mlx_lm":
            raise missing_error
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(SystemExit) as excinfo:
        bench.run_one(
            model_id="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            prompt="hello",
            prompt_label="test",
            max_tokens=1,
            warm=False,
        )
    return str(excinfo.value)


def test_run_one_exits_with_actionable_message_when_mlx_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message = _run_one_with_broken_import(
        monkeypatch,
        ModuleNotFoundError("No module named 'mlx_lm'", name="mlx_lm"),
    )
    assert "not installed in this interpreter" in message
    assert "uv run python scripts/bench.py" in message
    assert "'mlx_lm'" in message


def test_run_one_falls_back_to_message_parse_when_name_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A msg-only ModuleNotFoundError leaves exc.name=None; the guard must
    # recover the module name from the message text.
    message = _run_one_with_broken_import(
        monkeypatch,
        ModuleNotFoundError("No module named 'mlx_lm'"),
    )
    assert "'mlx_lm'" in message
    assert "uv run python scripts/bench.py" in message
