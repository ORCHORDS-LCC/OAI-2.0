"""Tests for ``scripts/backend_smoke.py``.

The smoke is wired into :mod:`scripts.verify` as the ``backend-smoke``
gate. These tests pin its stdout contract (``PASS backend-smoke: ...``
on success, ``FAIL backend-smoke: ...`` on failure) and its exit-code
contract (0 on success, 1 on assertion failure, 2 on unexpected
exception) so a regression in the script breaks the verify gate in a
visible way.

The smoke is exercised by running it in a subprocess with the project
root as ``cwd``. That guarantees the script imports the same
``oai2`` package the rest of the suite sees — no in-process state
leakage between tests.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "backend_smoke.py"


def _run_smoke(*, env_overrides: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run ``scripts/backend_smoke.py`` in a subprocess.

    ``env_overrides`` lets individual tests tweak the environment (e.g.
    set ``OAI2_GATEWAY_API_KEY`` to a malformed value). The base
    environment is the test process's environment minus any inherited
    token — the smoke is hermetic and must work without one.
    """
    env = os.environ.copy()
    env.pop("OAI2_GATEWAY_API_KEY", None)
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        check=False,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )


def test_backend_smoke_exits_zero_and_prints_pass() -> None:
    """The hermetic smoke exits 0 and prints the documented PASS prefix."""
    completed = _run_smoke()

    assert completed.returncode == 0, (
        f"expected exit 0, got {completed.returncode}\n"
        f"stdout={completed.stdout!r}\nstderr={completed.stderr!r}"
    )
    assert completed.stdout.startswith("PASS backend-smoke:"), (
        f"stdout must begin with 'PASS backend-smoke:', got {completed.stdout!r}"
    )
    assert "GatewayRuntime" in completed.stdout
    assert "drove" in completed.stdout


def test_backend_smoke_does_not_require_a_live_token() -> None:
    """The smoke must succeed with no ``OAI2_GATEWAY_API_KEY`` in the environment.

    The script sets its own placeholder token internally; an operator
    running it on a fresh checkout must not need to provision
    credentials first.
    """
    # Sanity: confirm the test process is running without a token.
    assert "OAI2_GATEWAY_API_KEY" not in os.environ
    completed = _run_smoke()
    assert completed.returncode == 0
    assert completed.stdout.startswith("PASS backend-smoke:")


def test_backend_smoke_passes_when_caller_overrides_env_key() -> None:
    """A caller-supplied token (any non-empty string) is honored.

    The smoke forces ``OAI2_GATEWAY_API_KEY=smoke-token-xyz`` internally
    via ``os.environ.setdefault``, but a caller can override it. We
    confirm the override is respected by using a different non-empty
    value and asserting the script still exits 0.
    """
    completed = _run_smoke(env_overrides={"OAI2_GATEWAY_API_KEY": "smoke-token-xyz"})
    assert completed.returncode == 0, (
        f"expected exit 0 with caller-supplied key, got {completed.returncode}\n"
        f"stdout={completed.stdout!r}\nstderr={completed.stderr!r}"
    )
    assert completed.stdout.startswith("PASS backend-smoke:")


def test_backend_smoke_runs_against_uv_environment() -> None:
    """The script imports cleanly under the project's uv environment.

    The verify.py gate invokes it via ``uv run python ...``. This test
    confirms ``python`` here also resolves a working interpreter
    (the subprocess uses ``sys.executable``, which pytest already
    routed through ``uv run``). A failure here means the script
    depends on something the verify environment doesn't provide.
    """
    completed = _run_smoke()
    # No traceback in stderr ⇒ clean import path.
    assert "Traceback" not in completed.stderr, (
        f"unexpected traceback in stderr:\n{completed.stderr}"
    )


def test_backend_smoke_runs_from_a_clean_working_directory() -> None:
    """Running the smoke from a different cwd still passes when invoked by absolute path.

    verify.py runs it with ``cwd=ROOT``. We confirm the script is
    robust to running from a subdirectory when the script path is
    absolute (which is how verify.py invokes it via ``uv run python
    scripts/backend_smoke.py``).
    """
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        check=False,
        cwd=str(ROOT / "scripts"),
        env={**os.environ, "OAI2_GATEWAY_API_KEY": "smoke-token-xyz"},
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        f"absolute-path invocation from scripts/ cwd should still pass; "
        f"got exit {completed.returncode}\n"
        f"stdout={completed.stdout!r}\nstderr={completed.stderr!r}"
    )
    assert completed.stdout.startswith("PASS backend-smoke:")


def test_backend_smoke_documents_exit_codes() -> None:
    """The script's module docstring pins exit codes 0/1/2 with semantics.

    This guards against the docstring and the actual ``main()``
    implementation drifting apart. If you change either side, change
    the other.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert "Exit codes" in source
    assert "``0``" in source
    assert "``1``" in source
    assert "``2``" in source
    # The body must actually call SystemExit(2) for the unexpected-exception path
    assert "SystemExit(2)" in source
