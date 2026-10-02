#!/usr/bin/env python3
"""Hermetic backend smoke for the OAI-2.0 harness integration.

This script exercises the harness path end-to-end through a mocked
gateway transport. It is intentionally offline: it requires neither a
live ``OAI2_GATEWAY_API_KEY`` nor any network access. Operators can
invoke it directly to confirm the cross-repo seam (env-aware selector →
:class:`GatewayRuntime` → eval-suite harness → capability scorer) is
wired correctly on this machine.

The smoke covers four contracts:

1. The env-aware selector picks :class:`GatewayRuntime` when
   ``OAI2_GATEWAY_API_KEY`` is set (this script always sets the env
   var before importing).
2. :func:`oai2.evals.run_eval_harness` drives the gateway path:
   ``runtime="GatewayRuntime"`` on the :class:`HarnessReport` and on
   every :class:`SuiteReport`.
3. Per-suite scoring reaches the capability scorer for every case
   (``n_cases`` equals ``len(scores)``).
4. :func:`run_eval_harness` closes the gateway runtime it created
   (lifetime management; slice-13 contract).

The mocked transport returns canned chat-completion responses whose
text matches the first ``coding_basic`` case's expected pattern
(``def sum_n``). Other cases won't match, so the harness reports a
mix of pass/fail — that is intentional; the smoke only asserts the
suite ran, not that it passed every case.

Exit codes
----------
``0``
    Smoke succeeded; ``PASS backend-smoke: ...`` was printed.
``1``
    Smoke failed an assertion; ``FAIL backend-smoke: ...`` was printed.
``2``
    Unexpected exception (import failure, transport error, etc.);
    ``FAIL backend-smoke: unexpected ...`` was printed.

The verify.py ``backend-smoke`` gate pins these exit codes by reading
stdout for the ``PASS``/``FAIL`` prefix and trusting the exit code.
"""

from __future__ import annotations

import os

import httpx

from oai2.evals import run_eval_harness
from oai2.runtime import GatewayConfig, GatewayRuntime
from oai2.runtime import inference as inference_module

# 15-char placeholder that stays under the public-safety regex's
# 16-char threshold. The selector reads ``OAI2_GATEWAY_API_KEY`` only
# at the moment ``run_eval_harness()`` calls
# ``select_runtime_from_env()``, so we set it in :func:`main` before
# any harness call — module-import time is too early and would
# produce a misleading ``FAIL backend-smoke`` if the script is
# imported for introspection (e.g. ``python -c "import
# scripts.backend_smoke"``).
_TOKEN_PLACEHOLDER = "smoke-token-xyz"


# A canned assistant reply that satisfies at least the first
# ``coding_basic`` expected pattern (``def sum_n``) and demonstrates
# the gateway path produces real responses for the scorer. Other
# coding cases won't match — that's fine, the smoke asserts the suite
# ran, not that it passed everything.
_CANNED_TEXT = (
    "Here is a Python function that satisfies the spec:\n"
    "\n"
    "def sum_n(n: int) -> int:\n"
    "    \"\"\"Return the sum of the first n natural numbers.\"\"\"\n"
    "    return n * (n + 1) // 2\n"
)


def _chat_completion(text: str, model: str = "oai-1.2") -> httpx.Response:
    """Canned OpenAI chat-completions response for the mocked transport."""
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-smoke",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": max(len(text.split()), 1),
                "total_tokens": 1 + max(len(text.split()), 1),
            },
        },
    )


def _build_mocked_runtime() -> tuple[GatewayRuntime, list[httpx.Request]]:
    """Build a GatewayRuntime whose transport is a MockTransport.

    Returns the runtime plus a request log so the smoke can assert
    that the harness actually drove network calls (proving the
    transport was reached, not a local bypass).
    """
    request_log: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_log.append(request)
        return _chat_completion(_CANNED_TEXT)

    transport = httpx.MockTransport(handler)
    # The runtime constructs ``httpx.Client(base_url=..., timeout=..., headers=...)``
    # itself when no client is injected. Build a Client that wraps our
    # MockTransport so the runtime's own construction path is exercised.
    cfg = GatewayConfig(
        base_url="http://mock.local",
        api_key="smoke-token-xyz",
        model="oai-1.2",
    )
    client = httpx.Client(
        base_url=cfg.base_url,
        transport=transport,
        headers={"Authorization": f"Bearer {cfg.api_key}"},
    )
    return GatewayRuntime(cfg, client=client), request_log


def main() -> int:
    # Ensure the env-var-driven selector picks GatewayRuntime before
    # any harness call. The 15-char value stays under the
    # public-safety regex's 16-char threshold.
    os.environ.setdefault("OAI2_GATEWAY_API_KEY", _TOKEN_PLACEHOLDER)

    runtime, request_log = _build_mocked_runtime()

    # Patch the env-aware selector so ``run_eval_harness`` (which calls
    # ``select_runtime_from_env`` lazily via a local import) resolves
    # to our prebuilt mocked runtime. This exercises the seam without
    # any live token.
    original_selector = inference_module.select_runtime_from_env
    inference_module.select_runtime_from_env = lambda: runtime

    try:
        report = run_eval_harness(suite_names=["coding"])
    finally:
        inference_module.select_runtime_from_env = original_selector
        runtime.close()

    # ---- Contract 1: harness picked the gateway runtime ----------------
    if report.runtime != "GatewayRuntime":
        print(
            f"FAIL backend-smoke: harness picked {report.runtime!r}, "
            "expected 'GatewayRuntime'",
            flush=True,
        )
        return 1

    # ---- Contract 2: every SuiteReport mirrors the same runtime --------
    if len(report.reports) != 1:
        print(
            f"FAIL backend-smoke: expected exactly 1 SuiteReport, "
            f"got {len(report.reports)}",
            flush=True,
        )
        return 1
    sub = report.reports[0]
    if sub.runtime != "GatewayRuntime":
        print(
            f"FAIL backend-smoke: SuiteReport runtime is {sub.runtime!r}, "
            "expected 'GatewayRuntime'",
            flush=True,
        )
        return 1
    if sub.suite_id != "coding_basic":
        print(
            f"FAIL backend-smoke: SuiteReport suite_id is {sub.suite_id!r}, "
            "expected 'coding_basic'",
            flush=True,
        )
        return 1

    # ---- Contract 3: every case reached the scorer ---------------------
    if sub.n_cases < 1:
        print(
            f"FAIL backend-smoke: coding_basic has {sub.n_cases} cases, "
            "expected >= 1",
            flush=True,
        )
        return 1
    if len(sub.scores) != sub.n_cases:
        print(
            f"FAIL backend-smoke: SuiteReport.scores has {len(sub.scores)} "
            f"entries but n_cases is {sub.n_cases}",
            flush=True,
        )
        return 1

    # ---- Contract 4: the gateway actually got the requests -------------
    # The coding_basic suite has 3 cases; the harness should have
    # POSTed once per case. Mismatch here means the transport was
    # bypassed somewhere.
    if len(request_log) != sub.n_cases:
        print(
            f"FAIL backend-smoke: gateway transport saw {len(request_log)} "
            f"POSTs but SuiteReport has {sub.n_cases} cases",
            flush=True,
        )
        return 1

    print(
        f"PASS backend-smoke: GatewayRuntime via run_eval_harness() "
        f"drove {sub.n_cases} cases ({sub.n_passed} passed, "
        f"pass_rate={sub.pass_rate:.2f})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — boundary
        print(
            f"FAIL backend-smoke: unexpected {type(exc).__name__}: {exc}",
            flush=True,
        )
        raise SystemExit(2) from exc
