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
