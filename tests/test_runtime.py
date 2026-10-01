"""Runtime tests — exercises the placeholder runtime and the MLX smoke check.

The :func:`smoke_check` test is gated to macOS / Apple Silicon where MLX
actually ships; on other platforms the placeholder runtime still works
but MLX is not installed.
"""

from __future__ import annotations

import sys

import pytest

from oai2.core import Status
from oai2.runtime import (
    InferenceRequest,
    ModelSpec,
    PlaceholderRuntime,
    default_runtime,
    discover_default_device,
    select_runtime_from_env,
    smoke_check,
)


def test_default_runtime_is_placeholder() -> None:
    rt = default_runtime()
    assert isinstance(rt, PlaceholderRuntime)


def test_placeholder_runtime_reports_device() -> None:
    rt = PlaceholderRuntime(ModelSpec(name="probe"))
    resp = rt.generate(InferenceRequest(prompt="hello"))
    assert resp.device == discover_default_device()
    assert resp.status is Status.EXPERIMENTAL
    assert resp.tokens >= 1
    assert "hello" not in resp.text  # placeholder never echoes


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="MLX runtime only ships on Apple Silicon (macOS).",
)
def test_mlx_smoke_check() -> None:
    ok, info = smoke_check()
    assert ok is True
    assert info.startswith("Device(") or info.startswith("gpu") or info.startswith("cpu")


def test_runtime_inference_model_scheduler_exports_from_runtime_package() -> None:
    from oai2.runtime import BatchPlan as ExportedBatchPlan
    from oai2.runtime import (
        InferenceRequest as ExportedInferenceRequest,
    )
    from oai2.runtime import (
        InferenceResponse as ExportedInferenceResponse,
    )
    from oai2.runtime import (
        InferenceRuntime as ExportedInferenceRuntime,
    )
    from oai2.runtime import ModelSpec as ExportedModelSpec
    from oai2.runtime import (
        PlaceholderRuntime as ExportedPlaceholderRuntime,
    )
    from oai2.runtime import (
        SafeBatchScheduler as ExportedSafeBatchScheduler,
    )
    from oai2.runtime import (
        ScheduledRequest as ExportedScheduledRequest,
    )
    from oai2.runtime import (
        SchedulerMetrics as ExportedSchedulerMetrics,
    )
    from oai2.runtime import (
        SessionCompatibilityKey as ExportedSessionCompatibilityKey,
    )
    from oai2.runtime import (
        default_runtime as ExportedDefaultRuntime,
    )
    from oai2.runtime import (
        discover_default_device as ExportedDiscoverDevice,
    )
    from oai2.runtime import smoke_check as ExportedSmokeCheck
    from oai2.runtime.inference import (
        InferenceRequest,
        InferenceResponse,
        InferenceRuntime,
        PlaceholderRuntime,
        default_runtime,
    )
    from oai2.runtime.model import (
        ModelSpec,
        discover_default_device,
        smoke_check,
    )
    from oai2.runtime.scheduler import (
        BatchPlan,
        SafeBatchScheduler,
        ScheduledRequest,
        SchedulerMetrics,
        SessionCompatibilityKey,
    )

    assert ExportedInferenceRequest is InferenceRequest
    assert ExportedInferenceResponse is InferenceResponse
    assert ExportedInferenceRuntime is InferenceRuntime
    assert ExportedPlaceholderRuntime is PlaceholderRuntime
    assert ExportedDefaultRuntime is default_runtime
    assert ExportedModelSpec is ModelSpec
    assert ExportedDiscoverDevice is discover_default_device
    assert ExportedSmokeCheck is smoke_check
    assert ExportedSessionCompatibilityKey is SessionCompatibilityKey
    assert ExportedScheduledRequest is ScheduledRequest
    assert ExportedSchedulerMetrics is SchedulerMetrics
    assert ExportedBatchPlan is BatchPlan
    assert ExportedSafeBatchScheduler is SafeBatchScheduler


# ---------------------------------------------------------------------------
# select_runtime_from_env — env-aware runtime selector
# ---------------------------------------------------------------------------


def test_select_runtime_from_env_returns_placeholder_when_no_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No ``OAI2_GATEWAY_API_KEY`` → deterministic offline runtime.

    Selectors must never raise on a missing key — CI / offline callers
    must keep working with a :class:`PlaceholderRuntime`.
    """
    monkeypatch.delenv("OAI2_GATEWAY_API_KEY", raising=False)
    monkeypatch.delenv("OAI2_GATEWAY_BASE_URL", raising=False)
    monkeypatch.delenv("OAI2_GATEWAY_MODEL", raising=False)
    monkeypatch.delenv("OAI2_GATEWAY_TIMEOUT_SECONDS", raising=False)

    rt = select_runtime_from_env()
    assert isinstance(rt, PlaceholderRuntime)


def test_select_runtime_from_env_returns_gateway_when_key_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``OAI2_GATEWAY_API_KEY`` present → :class:`GatewayRuntime` selected.

    Proves the selector is wired to the actual request path: when the
    operator provisions a key, the public entry point returns the
    :class:`InferenceRuntime` that will drive
    ``POST /v1/chat/completions`` against ``https://api.orchords.com``.
    """
    monkeypatch.setenv("OAI2_GATEWAY_API_KEY", "sel-token-xyz")
    monkeypatch.setenv("OAI2_GATEWAY_BASE_URL", "https://gateway.example.test")
    monkeypatch.setenv("OAI2_GATEWAY_MODEL", "oai-1.2")

    rt = select_runtime_from_env()
    closed = False
    try:
        from oai2.runtime import GatewayRuntime

        assert isinstance(rt, GatewayRuntime)
        assert rt.config.api_key == "sel-token-xyz"
        assert rt.config.base_url == "https://gateway.example.test"
        assert rt.config.model == "oai-1.2"
    finally:
        from oai2.runtime import GatewayRuntime

        if not closed and isinstance(rt, GatewayRuntime):
            rt.close()


def test_select_runtime_from_env_falls_back_on_partial_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Base URL or model without a key must NOT crash the selector.

    Only the API key is required; partial base URL / model env vars are
    documented fallbacks. Without the key the selector must still
    return a :class:`PlaceholderRuntime` and never raise.
    """
    monkeypatch.delenv("OAI2_GATEWAY_API_KEY", raising=False)
    monkeypatch.setenv("OAI2_GATEWAY_BASE_URL", "https://gateway.example.test")
    monkeypatch.setenv("OAI2_GATEWAY_MODEL", "oai-1.2")

    rt = select_runtime_from_env()
    assert isinstance(rt, PlaceholderRuntime)


def test_select_runtime_from_env_is_reexported_from_runtime_package() -> None:
    """The selector must be reachable through ``oai2.runtime``."""
    from oai2.runtime import select_runtime_from_env as PackageExport
    from oai2.runtime.inference import select_runtime_from_env as ModuleExport

    assert PackageExport is ModuleExport


def test_select_runtime_from_env_does_not_change_default_runtime() -> None:
    """``default_runtime`` stays pinned to :class:`PlaceholderRuntime`.

    Selectors are additive — they must never silently flip the static
    default. The existing test at the top of this file pins this
    contract; this test pins it again with the selector in scope to
    guard against accidental coupling.
    """
    from oai2.runtime import default_runtime, select_runtime_from_env

    base = default_runtime()
    selected = select_runtime_from_env()
    # Both are InferenceRuntime but the static default is always
    # PlaceholderRuntime; the selector may be either (depending on env).
    from oai2.runtime import InferenceRuntime

    assert isinstance(base, PlaceholderRuntime)
    assert isinstance(selected, InferenceRuntime)
