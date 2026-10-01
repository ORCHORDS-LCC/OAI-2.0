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
