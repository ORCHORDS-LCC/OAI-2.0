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
    # Pass 9 — pin identity for every entry of ``oai2.runtime.__all__``.
    # The original 13-entry sibling test covered only the inference /
    # model / scheduler slices; the same coverage-gap pattern that Pass 8
    # closed in ``oai2.model`` (commit ``985d357`` for #216) was open
    # here for admission, admission_scheduler, gateway_runtime and
    # gateway_model_client. This test asserts ``is`` identity for all
    # 36 package-surface entries against their source modules so a
    # future drop or rename in either side trips the test.
    from oai2.runtime import DEFAULT_GATEWAY_BASE_URL as ExportedDEFAULT_GATEWAY_BASE_URL
    from oai2.runtime import DEFAULT_GATEWAY_MODEL as ExportedDEFAULT_GATEWAY_MODEL
    from oai2.runtime import DEFAULT_MAX_TOKENS as ExportedDEFAULT_MAX_TOKENS
    from oai2.runtime import DEFAULT_TEMPERATURE as ExportedDEFAULT_TEMPERATURE
    from oai2.runtime import DEFAULT_TIMEOUT_SECONDS as ExportedDEFAULT_TIMEOUT_SECONDS
    from oai2.runtime import AdmissionAction as ExportedAdmissionAction
    from oai2.runtime import AdmissionBatchController as ExportedAdmissionBatchController
    from oai2.runtime import AdmissionDecision as ExportedAdmissionDecision
    from oai2.runtime import AdmissionPolicy as ExportedAdmissionPolicy
    from oai2.runtime import AdmissionQueue as ExportedAdmissionQueue
    from oai2.runtime import AdmissionReason as ExportedAdmissionReason
    from oai2.runtime import AdmissionRequest as ExportedAdmissionRequest
    from oai2.runtime import AdmissionScheduledRequest as ExportedAdmissionScheduledRequest
    from oai2.runtime import AdmissionSchedulerMetrics as ExportedAdmissionSchedulerMetrics
    from oai2.runtime import BatchPlan as ExportedBatchPlan
    from oai2.runtime import CapacitySnapshot as ExportedCapacitySnapshot
    from oai2.runtime import (
        CapacityTrendPoint as ExportedCapacityTrendPoint,
    )
    from oai2.runtime import ChatReply as ExportedChatReply
    from oai2.runtime import GatewayConfig as ExportedGatewayConfig
    from oai2.runtime import GatewayConfigError as ExportedGatewayConfigError
    from oai2.runtime import GatewayModelClient as ExportedGatewayModelClient
    from oai2.runtime import GatewayRuntime as ExportedGatewayRuntime
    from oai2.runtime import GatewayRuntimeError as ExportedGatewayRuntimeError
    from oai2.runtime import InferenceRequest as ExportedInferenceRequest
    from oai2.runtime import InferenceResponse as ExportedInferenceResponse
    from oai2.runtime import InferenceRuntime as ExportedInferenceRuntime
    from oai2.runtime import ModelSpec as ExportedModelSpec
    from oai2.runtime import PlaceholderRuntime as ExportedPlaceholderRuntime
    from oai2.runtime import (
        ResidencyAccountant as ExportedResidencyAccountant,
    )
    from oai2.runtime import (
        ResidencyOutcome as ExportedResidencyOutcome,
    )
    from oai2.runtime import (
        ResidencyRecord as ExportedResidencyRecord,
    )
    from oai2.runtime import (
        ResidencySummary as ExportedResidencySummary,
    )
    from oai2.runtime import SafeBatchScheduler as ExportedSafeBatchScheduler
    from oai2.runtime import ScheduledRequest as ExportedScheduledRequest
    from oai2.runtime import SchedulerMetrics as ExportedSchedulerMetrics
    from oai2.runtime import SessionCompatibilityKey as ExportedSessionCompatibilityKey
    from oai2.runtime import default_runtime as ExportedDefaultRuntime
    from oai2.runtime import discover_default_device as ExportedDiscoverDevice
    from oai2.runtime import (
        load_gateway_config_from_env as ExportedLoadGatewayConfigFromEnv,
    )
    from oai2.runtime import (
        select_runtime_from_env as ExportedSelectRuntimeFromEnv,
    )
    from oai2.runtime import smoke_check as ExportedSmokeCheck
    from oai2.runtime.admission import (
        AdmissionAction,
        AdmissionDecision,
        AdmissionPolicy,
        AdmissionQueue,
        AdmissionReason,
        AdmissionRequest,
        CapacitySnapshot,
    )
    from oai2.runtime.admission_scheduler import (
        AdmissionBatchController,
        AdmissionScheduledRequest,
        AdmissionSchedulerMetrics,
    )
    from oai2.runtime.gateway_model_client import (
        DEFAULT_MAX_TOKENS,
        DEFAULT_TEMPERATURE,
        ChatReply,
        GatewayModelClient,
    )
    from oai2.runtime.gateway_runtime import (
        DEFAULT_GATEWAY_BASE_URL,
        DEFAULT_GATEWAY_MODEL,
        DEFAULT_TIMEOUT_SECONDS,
        GatewayConfig,
        GatewayConfigError,
        GatewayRuntime,
        GatewayRuntimeError,
        load_gateway_config_from_env,
    )
    from oai2.runtime.inference import (
        InferenceRequest,
        InferenceResponse,
        InferenceRuntime,
        PlaceholderRuntime,
        default_runtime,
        select_runtime_from_env,
    )
    from oai2.runtime.model import (
        ModelSpec,
        discover_default_device,
        smoke_check,
    )
    from oai2.runtime.residency import (
        CapacityTrendPoint,
        ResidencyAccountant,
        ResidencyOutcome,
        ResidencyRecord,
        ResidencySummary,
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
    # admission slice (7)
    assert ExportedAdmissionAction is AdmissionAction
    assert ExportedAdmissionDecision is AdmissionDecision
    assert ExportedAdmissionPolicy is AdmissionPolicy
    assert ExportedAdmissionQueue is AdmissionQueue
    assert ExportedAdmissionReason is AdmissionReason
    assert ExportedAdmissionRequest is AdmissionRequest
    assert ExportedCapacitySnapshot is CapacitySnapshot
    # admission_scheduler slice (3)
    assert ExportedAdmissionBatchController is AdmissionBatchController
    assert ExportedAdmissionScheduledRequest is AdmissionScheduledRequest
    assert ExportedAdmissionSchedulerMetrics is AdmissionSchedulerMetrics
    # gateway_runtime slice (8)
    assert ExportedDEFAULT_GATEWAY_BASE_URL is DEFAULT_GATEWAY_BASE_URL
    assert ExportedDEFAULT_GATEWAY_MODEL is DEFAULT_GATEWAY_MODEL
    assert ExportedDEFAULT_TIMEOUT_SECONDS is DEFAULT_TIMEOUT_SECONDS
    assert ExportedGatewayConfig is GatewayConfig
    assert ExportedGatewayConfigError is GatewayConfigError
    assert ExportedGatewayRuntime is GatewayRuntime
    assert ExportedGatewayRuntimeError is GatewayRuntimeError
    assert ExportedLoadGatewayConfigFromEnv is load_gateway_config_from_env
    # gateway_model_client slice (4)
    assert ExportedChatReply is ChatReply
    assert ExportedDEFAULT_MAX_TOKENS is DEFAULT_MAX_TOKENS
    assert ExportedDEFAULT_TEMPERATURE is DEFAULT_TEMPERATURE
    assert ExportedGatewayModelClient is GatewayModelClient
    # residency/capacity accounting exports added at #232
    assert ExportedCapacityTrendPoint is CapacityTrendPoint
    assert ExportedResidencyAccountant is ResidencyAccountant
    assert ExportedResidencyOutcome is ResidencyOutcome
    assert ExportedResidencyRecord is ResidencyRecord
    assert ExportedResidencySummary is ResidencySummary
    # inference selector (newly exported at #238 slice chain)
    assert ExportedSelectRuntimeFromEnv is select_runtime_from_env


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
    monkeypatch.setenv("OAI2_GATEWAY_MODEL", "oai-2.0")

    rt = select_runtime_from_env()
    closed = False
    try:
        from oai2.runtime import GatewayRuntime

        assert isinstance(rt, GatewayRuntime)
        assert rt.config.api_key == "sel-token-xyz"
        assert rt.config.base_url == "https://gateway.example.test"
        assert rt.config.model == "oai-2.0"
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
    monkeypatch.setenv("OAI2_GATEWAY_MODEL", "oai-2.0")

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
