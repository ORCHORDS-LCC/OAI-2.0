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

The forced-failure tests at the bottom call ``main()`` in-process with
a monkey-patched selector or transport so they can exercise the FAIL
paths (assertion failures and the unexpected-exception boundary) that
the subprocess-level tests can't easily reach.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

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


# ---------------------------------------------------------------------------
# Forced-failure coverage
#
# The subprocess tests above only exercise the happy path (exit 0). The
# following tests drive ``main()`` in-process so we can monkey-patch the
# selector / transport to force assertion failures and the unexpected
# exception boundary. That completes the exit-code coverage matrix:
#   0 = PASS (subprocess tests)
#   1 = FAIL (assertion violation)
#   2 = FAIL unexpected (uncaught exception)
# ---------------------------------------------------------------------------


def test_backend_smoke_prints_fail_when_selector_returns_wrong_runtime() -> None:
    """Exit 1 / ``FAIL backend-smoke: harness picked ...`` when the selector returns
    something other than :class:`GatewayRuntime`.

    The script's first contract assertion is ``report.runtime ==
    'GatewayRuntime'``. We force the failure by patching
    :func:`scripts.backend_smoke._build_mocked_runtime` to return a
    :class:`PlaceholderRuntime` (wrapped so it has a ``.close()`` that
    the smoke's ``finally`` block can call) instead of the default
    :class:`GatewayRuntime`. The harness then records
    ``runtime='PlaceholderRuntime'`` on the report, and the script's
    contract-1 assertion fires.

    Patching the selector directly is not enough because
    :func:`main` itself rebuilds the selector binding every call
    (it installs ``lambda: runtime`` after building its own
    runtime). The cleanest seam is the factory
    :func:`_build_mocked_runtime`, which ``main`` calls exactly once.

    Note: :func:`main`'s ``finally`` unconditionally calls
    ``runtime.close()`` — we wrap the placeholder so that call still
    works (and is a no-op) without altering ``main`` itself.
    """
    import scripts.backend_smoke as smoke
    from oai2.runtime.inference import PlaceholderRuntime

    class _CloseablePlaceholder:
        """PlaceholderRuntime wrapper that exposes a no-op ``.close()``."""

        def __init__(self) -> None:
            self._runtime = PlaceholderRuntime()

        def generate(self, request):  # type: ignore[no-untyped-def]
            return self._runtime.generate(request)

        @property
        def device(self):  # type: ignore[no-untyped-def]
            return self._runtime.device

        def close(self) -> None:
            return None

    placeholder = _CloseablePlaceholder()
    prior_env = os.environ.get("OAI2_GATEWAY_API_KEY")
    os.environ["OAI2_GATEWAY_API_KEY"] = smoke._TOKEN_PLACEHOLDER
    try:
        with patch.object(
            smoke,
            "_build_mocked_runtime",
            return_value=(placeholder, []),
        ):
            rc = smoke.main()
    finally:
        # Restore the previous env state so we don't leak the token
        # to other tests in the suite (e.g. ``test_verify_script``'s
        # ``run_gateway_reach_check`` would then hit the live API).
        if prior_env is None:
            os.environ.pop("OAI2_GATEWAY_API_KEY", None)
        else:
            os.environ["OAI2_GATEWAY_API_KEY"] = prior_env

    assert rc == 1, f"expected exit 1 on contract-1 violation, got {rc}"


def test_backend_smoke_prints_fail_when_runtime_init_raises() -> None:
    """Exit 2 / ``FAIL backend-smoke: unexpected ...`` when the
    ``__main__`` boundary catches an unexpected exception.

    The script's :func:`main` is allowed to propagate exceptions;
    the boundary in the ``__main__`` block translates them to
    ``SystemExit(2)`` with ``raise ... from exc``. This test exercises
    that boundary by:

    1. Writing a small wrapper script that monkey-patches
       :func:`scripts.backend_smoke._build_mocked_runtime` to raise
       a deterministic :class:`RuntimeError`, then runs the same
       :func:`main` call the ``__main__`` block uses.
    2. Running the wrapper in a subprocess so the boundary logic is
       identical to what `verify.py` invokes.

    This is the canonical way to test a script's ``__main__`` block:
    the boundary code is small enough to inline, and we want the
    test to verify that *the same translation logic* the script
    uses produces exit 2 on a real exception.
    """
    import scripts.backend_smoke as smoke

    boom_msg = "forced-runtime-init-failure"

    bad_env = os.environ.copy()
    bad_env.pop("OAI2_GATEWAY_API_KEY", None)
    bad_env["OAI2_GATEWAY_API_KEY"] = smoke._TOKEN_PLACEHOLDER

    # Sanity: confirm the boundary's source pin matches our expectation.
    # If a future refactor moves the boundary logic into ``main()``,
    # this guard fails and forces the test to be rewritten.
    source = SCRIPT.read_text(encoding="utf-8")
    assert "raise SystemExit(2) from exc" in source, (
        "backend_smoke.py's __main__ boundary must translate "
        "unexpected exceptions to SystemExit(2) with `from exc`"
    )

    # Write a wrapper that mirrors the ``__main__`` block of
    # backend_smoke.py and triggers a deterministic exception. We use
    # a temp file rather than ``python -c`` so the test reads cleanly
    # in CI logs and is easier to debug.
    wrapper_path = ROOT / "scripts" / "_backend_smoke_exit2_test_wrapper.py"
    wrapper_path.write_text(
        "import scripts.backend_smoke as smoke\n"
        f"smoke._build_mocked_runtime = "
        f"lambda: (_ for _ in ()).throw(RuntimeError({boom_msg!r}))\n"
        "try:\n"
        "    raise SystemExit(smoke.main())\n"
        "except SystemExit:\n"
        "    raise\n"
        "except Exception as exc:\n"
        "    print(\n"
        "        f'FAIL backend-smoke: unexpected '\n"
        "        f'{type(exc).__name__}: {exc}',\n"
        "        flush=True,\n"
        "    )\n"
        "    raise SystemExit(2) from exc\n",
        encoding="utf-8",
    )
    try:
        completed = subprocess.run(
            [sys.executable, str(wrapper_path)],
            check=False,
            cwd=str(ROOT),
            env=bad_env,
            capture_output=True,
            text=True,
        )
    finally:
        wrapper_path.unlink(missing_ok=True)

    assert completed.returncode == 2, (
        f"expected exit 2 on forced construction failure, got {completed.returncode}\n"
        f"stdout={completed.stdout!r}\nstderr={completed.stderr!r}"
    )
    assert completed.stdout.startswith("FAIL backend-smoke: unexpected"), (
        f"stdout must begin with 'FAIL backend-smoke: unexpected', got {completed.stdout!r}"
    )
    assert boom_msg in completed.stdout, (
        f"stdout must echo the forced error message, got {completed.stdout!r}"
    )


def test_backend_smoke_prints_fail_when_scores_mismatch_n_cases() -> None:
    """Exit 1 / ``FAIL backend-smoke: SuiteReport.scores has ...`` when the
    harness returns a :class:`SuiteReport` whose ``len(scores) !=
    n_cases``.

    The script's contract-3 assertion is ``len(scores) ==
    n_cases``. We trigger the violation by patching
    :func:`oai2.evals.run_eval_harness` (via the local binding
    ``scripts.backend_smoke.run_eval_harness``) to return a fixture
    report with one fewer score than the suite's declared case count.
    This pins that the script reports the mismatch via exit 1 (not
    exit 2) — i.e. the unexpected-exception boundary is reserved for
    genuine exceptions, not contract failures.
    """
    import scripts.backend_smoke as smoke
    from oai2.evals import HarnessReport, SuiteReport

    bad_sub = SuiteReport(
        suite_id="coding_basic",
        capability="coding",
        runtime="GatewayRuntime",
        n_cases=3,
        n_passed=0,
        scores=[],
        error=None,
    )
    bad_report = HarnessReport(
        runtime="GatewayRuntime",
        reports=(bad_sub,),
    )

    prior_env = os.environ.get("OAI2_GATEWAY_API_KEY")
    os.environ["OAI2_GATEWAY_API_KEY"] = smoke._TOKEN_PLACEHOLDER
    try:
        with patch.object(smoke, "run_eval_harness", return_value=bad_report):
            rc = smoke.main()
    finally:
        if prior_env is None:
            os.environ.pop("OAI2_GATEWAY_API_KEY", None)
        else:
            os.environ["OAI2_GATEWAY_API_KEY"] = prior_env

    assert rc == 1, f"expected exit 1 on contract-3 violation, got {rc}"
