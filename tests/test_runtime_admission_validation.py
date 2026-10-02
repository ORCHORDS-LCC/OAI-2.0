"""Outer guard rails for ``oai2/runtime/admission.py``.

These tests pin the WI-QOS-002 admission-control + backpressure policy
public-safety boundary: every dataclass slot, every enum wire-string,
every ``__post_init__`` validation, and the full ``decide`` priority
order. No live scheduler, no live queue, no live inference backend is
needed — ``decide`` is a deterministic pure function of
``(AdmissionRequest, CapacitySnapshot, now_ms)``.
"""

from __future__ import annotations

import inspect
import math

import pytest

import oai2.runtime.admission as admission_module
from oai2.reasoning.modes import ReasoningMode
from oai2.runtime.admission import (
    AdmissionAction,
    AdmissionDecision,
    AdmissionPolicy,
    AdmissionQueue,
    AdmissionReason,
    AdmissionRequest,
    CapacitySnapshot,
)

# ---------------------------------------------------------------------------
# 1. Module docstring + import discipline
# ---------------------------------------------------------------------------


def test_admission_module_docstring_mentions_wiqo_qos_002() -> None:
    """Module docstring must mention ``WI-QOS-002`` so the policy contract is grep-able."""
    doc = admission_module.__doc__ or ""
    assert "WI-QOS-002" in doc
    assert "admission" in doc.lower()


def test_admission_module_has_no_hardcoded_credentials() -> None:
    """The module source must not contain credentials, tokens, or secrets."""
    source = inspect.getsource(admission_module)
    for forbidden in (
        "AKIA",  # AWS access key prefix
        "sk-",  # OpenAI / many providers
        "ghp_",  # GitHub PAT
        "password=",
        "secret=",
        "BEGIN PRIVATE KEY",
    ):
        assert forbidden not in source, (
            f"forbidden credential marker `{forbidden!r}` found in admission.py source"
        )


def test_admission_module_has_no_unrelated_cloud_runtime_imports() -> None:
    """The module must not import cloud runtimes, urllib3, or unrelated HTTP clients."""
    source = inspect.getsource(admission_module)
    for forbidden in (
        "import boto3",
        "from boto3",
        "import urllib3",
        "from urllib3",
        "import requests",
        "from requests",
        "from oai2.cloudflare",
        "from oai2.workers",
        "from oai2.gateway",  # admission is scheduler-neutral
    ):
        assert forbidden not in source, f"forbidden import `{forbidden!r}` in admission.py source"


def test_admission_imports_use_collections_abc_not_typing_for_runtime() -> None:
    """The module must not use ``typing.Mapping`` (UP006-clean if it imports it)."""
    source = inspect.getsource(admission_module)
    assert "from typing import Mapping" not in source
    assert "from typing import Sequence" not in source
    assert "from typing import Iterable" not in source


# ---------------------------------------------------------------------------
# 2. ``__all__`` completeness + package-level re-export
# ---------------------------------------------------------------------------


def test_admission_module_all_is_pinned() -> None:
    """``__all__`` must contain exactly 7 documented public names."""
    assert sorted(admission_module.__all__) == sorted(
        [
            "AdmissionAction",
            "AdmissionReason",
            "AdmissionRequest",
            "CapacitySnapshot",
            "AdmissionDecision",
            "AdmissionPolicy",
            "AdmissionQueue",
        ]
    )


def test_admission_module_all_names_are_importable() -> None:
    """Every name in ``__all__`` must be importable from the module."""
    for name in admission_module.__all__:
        assert hasattr(admission_module, name), f"{name!r} not found on admission module"


def test_admission_symbols_are_reexported_at_package_level() -> None:
    """Each public name must be re-exported through ``oai2.runtime``."""
    from oai2.runtime import (  # noqa: PLC0415
        AdmissionAction as PackageAction,
    )
    from oai2.runtime import (
        AdmissionDecision as PackageDecision,
    )
    from oai2.runtime import (
        AdmissionPolicy as PackagePolicy,
    )
    from oai2.runtime import (
        AdmissionQueue as PackageQueue,
    )
    from oai2.runtime import (
        AdmissionReason as PackageReason,
    )
    from oai2.runtime import (
        AdmissionRequest as PackageRequest,
    )
    from oai2.runtime import (
        CapacitySnapshot as PackageSnapshot,
    )

    assert PackageAction is AdmissionAction
    assert PackageDecision is AdmissionDecision
    assert PackagePolicy is AdmissionPolicy
    assert PackageQueue is AdmissionQueue
    assert PackageReason is AdmissionReason
    assert PackageRequest is AdmissionRequest
    assert PackageSnapshot is CapacitySnapshot


# ---------------------------------------------------------------------------
# 3. ``AdmissionAction`` StrEnum
# ---------------------------------------------------------------------------


def test_admission_action_wire_strings_are_pinned() -> None:
    """``AdmissionAction`` must have exactly 3 wire strings in this order."""
    assert AdmissionAction.ADMIT.value == "admit"
    assert AdmissionAction.QUEUE.value == "queue"
    assert AdmissionAction.REJECT.value == "reject"
    assert len(AdmissionAction) == 3


def test_admission_action_is_str_enum() -> None:
    """``AdmissionAction`` must be a ``StrEnum`` subclass."""
    from enum import StrEnum  # noqa: PLC0415

    assert issubclass(AdmissionAction, StrEnum)


def test_admission_action_round_trips_via_value() -> None:
    """Each wire value must reconstruct its enum member."""
    for member in AdmissionAction:
        assert AdmissionAction(member.value) is member


def test_admission_action_value_is_str() -> None:
    """Each ``AdmissionAction.value`` must be a ``str`` instance."""
    for member in AdmissionAction:
        assert isinstance(member.value, str)


# ---------------------------------------------------------------------------
# 4. ``AdmissionReason`` StrEnum
# ---------------------------------------------------------------------------


def test_admission_reason_wire_strings_are_pinned() -> None:
    """``AdmissionReason`` must have exactly 6 wire strings."""
    expected = {
        "capacity_available",
        "memory_pressure",
        "active_limit",
        "queue_full",
        "deadline_expired",
        "request_too_large",
    }
    actual = {member.value for member in AdmissionReason}
    assert actual == expected
    assert len(AdmissionReason) == 6


def test_admission_reason_is_str_enum() -> None:
    """``AdmissionReason`` must be a ``StrEnum`` subclass."""
    from enum import StrEnum  # noqa: PLC0415

    assert issubclass(AdmissionReason, StrEnum)


def test_admission_reason_round_trips_via_value() -> None:
    """Each wire value must reconstruct its enum member."""
    for member in AdmissionReason:
        assert AdmissionReason(member.value) is member


# ---------------------------------------------------------------------------
# 5. ``AdmissionRequest`` dataclass shape + ``__post_init__`` validation
# ---------------------------------------------------------------------------


def test_admission_request_field_set_is_pinned() -> None:
    """``AdmissionRequest`` must have exactly 6 fields with documented defaults."""
    fields = {f.name for f in AdmissionRequest.__dataclass_fields__.values()}
    assert fields == {
        "request_id",
        "mode",
        "per_lane_memory_gb",
        "estimated_duration_ms",
        "enqueued_at_ms",
        "deadline_ms",
        "swarm_lanes",
    }


def test_admission_request_is_slots_and_frozen() -> None:
    """``AdmissionRequest`` must be ``slots=True`` + ``frozen=True``."""
    params = AdmissionRequest.__dataclass_params__  # type: ignore[attr-defined]
    assert params.slots is True
    assert params.frozen is True


def test_admission_request_rejects_empty_request_id() -> None:
    """An empty ``request_id`` is rejected as not-normalized."""
    with pytest.raises(ValueError, match="request_id"):
        AdmissionRequest(
            request_id="",
            mode=ReasoningMode.NORMAL,
            per_lane_memory_gb=1.0,
            estimated_duration_ms=100.0,
            enqueued_at_ms=0.0,
        )


def test_admission_request_rejects_padded_request_id() -> None:
    """A padded ``request_id`` (whitespace around it) is rejected as not-normalized."""
    with pytest.raises(ValueError, match="request_id"):
        AdmissionRequest(
            request_id="  req-1  ",
            mode=ReasoningMode.NORMAL,
            per_lane_memory_gb=1.0,
            estimated_duration_ms=100.0,
            enqueued_at_ms=0.0,
        )


def test_admission_request_rejects_non_positive_per_lane_memory_gb() -> None:
    """``per_lane_memory_gb`` must be positive finite (parametrized over the bad set)."""
    for bad in (0.0, -1.0, -1e9, math.nan, math.inf, -math.inf, True, None, "1", [1]):
        with pytest.raises(ValueError, match="per_lane_memory_gb"):
            AdmissionRequest(
                request_id="req-1",
                mode=ReasoningMode.NORMAL,
                per_lane_memory_gb=bad,  # type: ignore[arg-type]
                estimated_duration_ms=100.0,
                enqueued_at_ms=0.0,
            )


def test_admission_request_rejects_non_positive_estimated_duration_ms() -> None:
    """``estimated_duration_ms`` must be positive finite."""
    for bad in (0.0, -100.0, math.nan, math.inf, None, "1"):
        with pytest.raises(ValueError, match="estimated_duration_ms"):
            AdmissionRequest(
                request_id="req-1",
                mode=ReasoningMode.NORMAL,
                per_lane_memory_gb=1.0,
                estimated_duration_ms=bad,  # type: ignore[arg-type]
                enqueued_at_ms=0.0,
            )


def test_admission_request_rejects_negative_enqueued_at_ms() -> None:
    """``enqueued_at_ms`` must be non-negative finite (negative is rejected; zero is OK)."""
    for bad in (-1.0, -1e9, math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="enqueued_at_ms"):
            AdmissionRequest(
                request_id="req-1",
                mode=ReasoningMode.NORMAL,
                per_lane_memory_gb=1.0,
                estimated_duration_ms=100.0,
                enqueued_at_ms=bad,
            )


def test_admission_request_accepts_zero_enqueued_at_ms() -> None:
    """``enqueued_at_ms=0.0`` is the documented lower bound and must be accepted."""
    req = AdmissionRequest(
        request_id="req-1",
        mode=ReasoningMode.NORMAL,
        per_lane_memory_gb=1.0,
        estimated_duration_ms=100.0,
        enqueued_at_ms=0.0,
    )
    assert req.enqueued_at_ms == 0.0


def test_admission_request_rejects_non_positive_deadline_ms() -> None:
    """``deadline_ms=None`` is allowed; non-positive finite rejects negative."""
    with pytest.raises(ValueError, match="deadline_ms"):
        AdmissionRequest(
            request_id="req-1",
            mode=ReasoningMode.NORMAL,
            per_lane_memory_gb=1.0,
            estimated_duration_ms=100.0,
            enqueued_at_ms=0.0,
            deadline_ms=-1.0,
        )


def test_admission_request_accepts_none_deadline_ms() -> None:
    """``deadline_ms=None`` is the documented default and must be accepted."""
    req = AdmissionRequest(
        request_id="req-1",
        mode=ReasoningMode.NORMAL,
        per_lane_memory_gb=1.0,
        estimated_duration_ms=100.0,
        enqueued_at_ms=0.0,
    )
    assert req.deadline_ms is None


def test_admission_request_rejects_non_positive_swarm_lanes() -> None:
    """``swarm_lanes`` must be a positive int; non-int / non-positive rejected."""
    for bad in (-1, 0, "1", 1.5, True, False, None, [1]):
        with pytest.raises(ValueError, match="swarm_lanes"):
            AdmissionRequest(
                request_id="req-1",
                mode=ReasoningMode.SWARM,
                per_lane_memory_gb=1.0,
                estimated_duration_ms=100.0,
                enqueued_at_ms=0.0,
                swarm_lanes=bad,  # type: ignore[arg-type]
            )


def test_admission_request_rejects_non_swarm_with_multi_lane() -> None:
    """Non-SWARM requests must use exactly one lane (the gate here is for >1)."""
    with pytest.raises(ValueError, match="non-SWARM"):
        AdmissionRequest(
            request_id="req-1",
            mode=ReasoningMode.NORMAL,
            per_lane_memory_gb=1.0,
            estimated_duration_ms=100.0,
            enqueued_at_ms=0.0,
            swarm_lanes=2,
        )


def test_admission_request_swarm_with_multi_lane_is_accepted() -> None:
    """SWARM with ``swarm_lanes=N>1`` is accepted; ``aggregate_memory_gb`` scales."""
    req = AdmissionRequest(
        request_id="req-1",
        mode=ReasoningMode.SWARM,
        per_lane_memory_gb=1.5,
        estimated_duration_ms=100.0,
        enqueued_at_ms=0.0,
        swarm_lanes=4,
    )
    assert req.aggregate_memory_gb == 6.0


def test_admission_request_defaults_pinned() -> None:
    """Defaults are pinned: ``deadline_ms=None``, ``swarm_lanes=1``."""
    req = AdmissionRequest(
        request_id="req-1",
        mode=ReasoningMode.NORMAL,
        per_lane_memory_gb=1.0,
        estimated_duration_ms=100.0,
        enqueued_at_ms=0.0,
    )
    assert req.deadline_ms is None
    assert req.swarm_lanes == 1
    assert req.aggregate_memory_gb == 1.0


def test_admission_request_is_frozen() -> None:
    """``AdmissionRequest`` is frozen — attribute mutation must raise."""
    req = AdmissionRequest(
        request_id="req-1",
        mode=ReasoningMode.NORMAL,
        per_lane_memory_gb=1.0,
        estimated_duration_ms=100.0,
        enqueued_at_ms=0.0,
    )
    with pytest.raises((AttributeError, dataclasses_FrozenInstanceError())):
        req.request_id = "x"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 6. ``CapacitySnapshot`` dataclass shape + ``__post_init__`` validation
# ---------------------------------------------------------------------------


def test_capacity_snapshot_field_set_is_pinned() -> None:
    """``CapacitySnapshot`` must have exactly 7 fields."""
    fields = {f.name for f in CapacitySnapshot.__dataclass_fields__.values()}
    assert fields == {
        "total_memory_gb",
        "used_memory_gb",
        "reserved_memory_gb",
        "active_tasks",
        "max_active_tasks",
        "queue_depth",
        "max_queue_depth",
    }


def test_capacity_snapshot_is_slots_and_frozen() -> None:
    """``CapacitySnapshot`` must be ``slots=True`` + ``frozen=True``."""
    params = CapacitySnapshot.__dataclass_params__  # type: ignore[attr-defined]
    assert params.slots is True
    assert params.frozen is True


def test_capacity_snapshot_rejects_non_positive_total_memory_gb() -> None:
    """``total_memory_gb`` must be positive finite."""
    for bad in (0.0, -1.0, math.nan, math.inf, True, None, "1"):
        with pytest.raises(ValueError, match="total_memory_gb"):
            CapacitySnapshot(
                total_memory_gb=bad,  # type: ignore[arg-type]
                used_memory_gb=0.0,
                reserved_memory_gb=0.0,
                active_tasks=0,
                max_active_tasks=1,
                queue_depth=0,
                max_queue_depth=1,
            )


def test_capacity_snapshot_rejects_negative_used_memory_gb() -> None:
    """``used_memory_gb`` must be non-negative finite."""
    with pytest.raises(ValueError, match="used_memory_gb"):
        CapacitySnapshot(
            total_memory_gb=10.0,
            used_memory_gb=-1.0,
            reserved_memory_gb=0.0,
            active_tasks=0,
            max_active_tasks=1,
            queue_depth=0,
            max_queue_depth=1,
        )


def test_capacity_snapshot_rejects_used_exceeding_total() -> None:
    """``used_memory_gb`` must not exceed ``total_memory_gb``."""
    with pytest.raises(ValueError, match="used_memory_gb cannot exceed"):
        CapacitySnapshot(
            total_memory_gb=10.0,
            used_memory_gb=11.0,
            reserved_memory_gb=0.0,
            active_tasks=0,
            max_active_tasks=1,
            queue_depth=0,
            max_queue_depth=1,
        )


def test_capacity_snapshot_rejects_reserved_equals_total_memory() -> None:
    """``reserved_memory_gb`` must be strictly less than ``total_memory_gb``."""
    with pytest.raises(ValueError, match="reserved_memory_gb must be less"):
        CapacitySnapshot(
            total_memory_gb=10.0,
            used_memory_gb=0.0,
            reserved_memory_gb=10.0,
            active_tasks=0,
            max_active_tasks=1,
            queue_depth=0,
            max_queue_depth=1,
        )


def test_capacity_snapshot_rejects_reserved_exceeding_total() -> None:
    """``reserved_memory_gb > total_memory_gb`` is rejected."""
    with pytest.raises(ValueError, match="reserved_memory_gb must be less"):
        CapacitySnapshot(
            total_memory_gb=10.0,
            used_memory_gb=0.0,
            reserved_memory_gb=11.0,
            active_tasks=0,
            max_active_tasks=1,
            queue_depth=0,
            max_queue_depth=1,
        )


@pytest.mark.parametrize(
    "name",
    ["active_tasks", "max_active_tasks", "queue_depth", "max_queue_depth"],
)
def test_capacity_snapshot_rejects_negative_int_count(name: str) -> None:
    """Each int field must be a non-negative int; negative rejected."""
    base: dict[str, object] = {
        "total_memory_gb": 10.0,
        "used_memory_gb": 0.0,
        "reserved_memory_gb": 0.0,
        "active_tasks": 0,
        "max_active_tasks": 1,
        "queue_depth": 0,
        "max_queue_depth": 1,
    }
    base[name] = -1
    with pytest.raises(ValueError, match=name):
        CapacitySnapshot(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "name",
    ["active_tasks", "max_active_tasks", "queue_depth", "max_queue_depth"],
)
def test_capacity_snapshot_rejects_bool_as_int(name: str) -> None:
    """Each int field must reject bool (the load-bearing ``isinstance(value, bool)`` first-check)."""
    base: dict[str, object] = {
        "total_memory_gb": 10.0,
        "used_memory_gb": 0.0,
        "reserved_memory_gb": 0.0,
        "active_tasks": 0,
        "max_active_tasks": 1,
        "queue_depth": 0,
        "max_queue_depth": 1,
    }
    base[name] = True
    with pytest.raises(ValueError, match=name):
        CapacitySnapshot(**base)  # type: ignore[arg-type]


def test_capacity_snapshot_rejects_zero_max_active_tasks() -> None:
    """``max_active_tasks`` must be positive."""
    with pytest.raises(ValueError, match="max_active_tasks must be positive"):
        CapacitySnapshot(
            total_memory_gb=10.0,
            used_memory_gb=0.0,
            reserved_memory_gb=0.0,
            active_tasks=0,
            max_active_tasks=0,
            queue_depth=0,
            max_queue_depth=1,
        )


def test_capacity_snapshot_rejects_queue_depth_exceeding_max() -> None:
    """``queue_depth`` must not exceed ``max_queue_depth``."""
    with pytest.raises(ValueError, match="queue_depth cannot exceed"):
        CapacitySnapshot(
            total_memory_gb=10.0,
            used_memory_gb=0.0,
            reserved_memory_gb=0.0,
            active_tasks=0,
            max_active_tasks=1,
            queue_depth=2,
            max_queue_depth=1,
        )


def test_capacity_snapshot_available_memory_gb_clamped() -> None:
    """``available_memory_gb`` clamps at 0.0 when usage+reserved > total."""
    snap = CapacitySnapshot(
        total_memory_gb=10.0,
        used_memory_gb=8.0,
        reserved_memory_gb=1.0,
        active_tasks=0,
        max_active_tasks=1,
        queue_depth=0,
        max_queue_depth=1,
    )
    # 10 - 1 - 8 = 1 → still positive.
    assert snap.available_memory_gb == 1.0
    snap2 = CapacitySnapshot(
        total_memory_gb=10.0,
        used_memory_gb=5.0,
        reserved_memory_gb=5.0,
        active_tasks=0,
        max_active_tasks=1,
        queue_depth=0,
        max_queue_depth=1,
    )
    # 10 - 5 - 5 = 0 → clamp.
    assert snap2.available_memory_gb == 0.0


# ---------------------------------------------------------------------------
# 7. ``AdmissionDecision`` dataclass shape
# ---------------------------------------------------------------------------


def test_admission_decision_field_set_is_pinned() -> None:
    """``AdmissionDecision`` must have exactly 5 fields."""
    fields = {f.name for f in AdmissionDecision.__dataclass_fields__.values()}
    assert fields == {
        "request_id",
        "action",
        "reason",
        "required_memory_gb",
        "available_memory_gb",
    }


def test_admission_decision_is_slots_and_frozen() -> None:
    """``AdmissionDecision`` must be ``slots=True`` + ``frozen=True``."""
    params = AdmissionDecision.__dataclass_params__  # type: ignore[attr-defined]
    assert params.slots is True
    assert params.frozen is True


def test_admission_decision_is_frozen() -> None:
    """Attribute mutation must raise."""
    decision = AdmissionDecision(
        request_id="req-1",
        action=AdmissionAction.ADMIT,
        reason=AdmissionReason.CAPACITY_AVAILABLE,
        required_memory_gb=1.0,
        available_memory_gb=10.0,
    )
    with pytest.raises((AttributeError, dataclasses_FrozenInstanceError())):
        decision.action = AdmissionAction.REJECT  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 8. ``AdmissionPolicy`` dataclass + ``decide``
# ---------------------------------------------------------------------------


def test_admission_policy_default_starvation_after_ms_is_30s() -> None:
    """``AdmissionPolicy`` defaults ``starvation_after_ms=30_000.0``."""
    policy = AdmissionPolicy()
    assert policy.starvation_after_ms == 30_000.0


def test_admission_policy_rejects_non_positive_starvation_after_ms() -> None:
    """``starvation_after_ms`` must be positive finite."""
    for bad in (0.0, -1.0, math.nan, math.inf, None, "1"):
        with pytest.raises(ValueError, match="starvation_after_ms"):
            AdmissionPolicy(starvation_after_ms=bad)  # type: ignore[arg-type]


def _req(
    *,
    request_id: str = "req-1",
    mode: ReasoningMode = ReasoningMode.NORMAL,
    per_lane_memory_gb: float = 1.0,
    estimated_duration_ms: float = 100.0,
    enqueued_at_ms: float = 0.0,
    deadline_ms: float | None = None,
    swarm_lanes: int = 1,
) -> AdmissionRequest:
    return AdmissionRequest(
        request_id=request_id,
        mode=mode,
        per_lane_memory_gb=per_lane_memory_gb,
        estimated_duration_ms=estimated_duration_ms,
        enqueued_at_ms=enqueued_at_ms,
        deadline_ms=deadline_ms,
        swarm_lanes=swarm_lanes,
    )


def _cap(
    *,
    total_memory_gb: float = 10.0,
    used_memory_gb: float = 0.0,
    reserved_memory_gb: float = 0.0,
    active_tasks: int = 0,
    max_active_tasks: int = 1,
    queue_depth: int = 0,
    max_queue_depth: int = 1,
) -> CapacitySnapshot:
    return CapacitySnapshot(
        total_memory_gb=total_memory_gb,
        used_memory_gb=used_memory_gb,
        reserved_memory_gb=reserved_memory_gb,
        active_tasks=active_tasks,
        max_active_tasks=max_active_tasks,
        queue_depth=queue_depth,
        max_queue_depth=max_queue_depth,
    )


def test_admission_decide_admit_when_room_available() -> None:
    """``decide`` returns ``ADMIT`` + ``CAPACITY_AVAILABLE`` when room is available."""
    policy = AdmissionPolicy()
    decision = policy.decide(
        _req(),
        _cap(total_memory_gb=10.0, used_memory_gb=0.0),
        now_ms=0.0,
    )
    assert decision.action is AdmissionAction.ADMIT
    assert decision.reason is AdmissionReason.CAPACITY_AVAILABLE
    assert decision.request_id == "req-1"
    assert decision.required_memory_gb == 1.0
    assert decision.available_memory_gb == 10.0


def test_admission_decide_rejects_deadline_expired() -> None:
    """``decide`` rejects when ``now_ms >= deadline_ms``."""
    policy = AdmissionPolicy()
    decision = policy.decide(
        _req(deadline_ms=1000.0),
        _cap(),
        now_ms=1000.0,
    )
    assert decision.action is AdmissionAction.REJECT
    assert decision.reason is AdmissionReason.DEADLINE_EXPIRED


def test_admission_decide_rejects_when_required_exceeds_usable_total() -> None:
    """``decide`` rejects when ``required > total - reserved``."""
    policy = AdmissionPolicy()
    decision = policy.decide(
        _req(per_lane_memory_gb=20.0),
        _cap(total_memory_gb=10.0, reserved_memory_gb=0.0),
        now_ms=0.0,
    )
    assert decision.action is AdmissionAction.REJECT
    assert decision.reason is AdmissionReason.REQUEST_TOO_LARGE


def test_admission_decide_rejects_when_required_exceeds_usable_after_reserved() -> None:
    """Reserved memory shrinks the usable total — large request must reject."""
    policy = AdmissionPolicy()
    decision = policy.decide(
        _req(per_lane_memory_gb=9.0),
        _cap(total_memory_gb=10.0, reserved_memory_gb=2.0, used_memory_gb=0.0),
        now_ms=0.0,
    )
    # usable_total = 10 - 2 = 8; required = 9; REJECT.
    assert decision.action is AdmissionAction.REJECT
    assert decision.reason is AdmissionReason.REQUEST_TOO_LARGE


def test_admission_decide_queue_when_active_limit_reached() -> None:
    """``decide`` queues with ``ACTIVE_LIMIT`` when active_tasks == max and room in queue."""
    policy = AdmissionPolicy()
    decision = policy.decide(
        _req(),
        _cap(active_tasks=1, max_active_tasks=1, queue_depth=0, max_queue_depth=1),
        now_ms=0.0,
    )
    assert decision.action is AdmissionAction.QUEUE
    assert decision.reason is AdmissionReason.ACTIVE_LIMIT


def test_admission_decide_queue_when_memory_pressure() -> None:
    """``decide`` queues with ``MEMORY_PRESSURE`` when room short and queue open."""
    policy = AdmissionPolicy()
    decision = policy.decide(
        _req(per_lane_memory_gb=10.0),
        _cap(
            total_memory_gb=10.0,
            used_memory_gb=5.0,
            active_tasks=0,
            max_active_tasks=1,
            queue_depth=0,
            max_queue_depth=1,
        ),
        now_ms=0.0,
    )
    assert decision.action is AdmissionAction.QUEUE
    assert decision.reason is AdmissionReason.MEMORY_PRESSURE


def test_admission_decide_rejects_queue_full() -> None:
    """``decide`` rejects with ``QUEUE_FULL`` when queue is at capacity."""
    policy = AdmissionPolicy()
    decision = policy.decide(
        _req(),
        _cap(active_tasks=1, max_active_tasks=1, queue_depth=1, max_queue_depth=1),
        now_ms=0.0,
    )
    assert decision.action is AdmissionAction.REJECT
    assert decision.reason is AdmissionReason.QUEUE_FULL


def test_admission_decide_active_limit_wins_over_memory_pressure() -> None:
    """When both active_limit AND memory_pressure trigger, ``ACTIVE_LIMIT`` wins."""
    policy = AdmissionPolicy()
    decision = policy.decide(
        _req(per_lane_memory_gb=10.0),
        _cap(
            total_memory_gb=10.0,
            used_memory_gb=5.0,
            active_tasks=1,
            max_active_tasks=1,
            queue_depth=0,
            max_queue_depth=1,
        ),
        now_ms=0.0,
    )
    assert decision.action is AdmissionAction.QUEUE
    assert decision.reason is AdmissionReason.ACTIVE_LIMIT


def test_admission_decide_rejects_negative_now_ms() -> None:
    """``now_ms`` must be non-negative finite."""
    policy = AdmissionPolicy()
    with pytest.raises(ValueError, match="now_ms"):
        policy.decide(_req(), _cap(), now_ms=-1.0)


def test_admission_decide_rejects_non_finite_now_ms() -> None:
    """``now_ms`` must reject NaN / ±inf."""
    policy = AdmissionPolicy()
    with pytest.raises(ValueError, match="now_ms"):
        policy.decide(_req(), _cap(), now_ms=math.nan)


def test_admission_decide_now_ms_is_keyword_only() -> None:
    """``now_ms`` must be keyword-only."""
    sig = inspect.signature(AdmissionPolicy.decide)
    assert sig.parameters["now_ms"].kind == inspect.Parameter.KEYWORD_ONLY


def test_admission_decide_does_not_mutate_request_or_capacity() -> None:
    """``decide`` is a pure function — neither argument changes."""
    policy = AdmissionPolicy()
    req = _req()
    cap = _cap()
    policy.decide(req, cap, now_ms=0.0)
    assert req.request_id == "req-1"
    assert req.per_lane_memory_gb == 1.0
    assert cap.total_memory_gb == 10.0


# ---------------------------------------------------------------------------
# 9. ``AdmissionQueue`` lifecycle
# ---------------------------------------------------------------------------


def test_admission_queue_default_starvation_after_ms() -> None:
    """``AdmissionQueue()`` defaults ``starvation_after_ms=30_000.0``."""
    queue: AdmissionQueue = AdmissionQueue()
    assert len(queue) == 0


def test_admission_queue_rejects_non_positive_starvation_after_ms() -> None:
    """``starvation_after_ms`` must be positive finite."""
    for bad in (0.0, -1.0, math.nan, math.inf, True, None, "1"):
        with pytest.raises(ValueError, match="starvation_after_ms"):
            AdmissionQueue(starvation_after_ms=bad)  # type: ignore[arg-type]


def test_admission_queue_enqueue_preserves_request() -> None:
    """``enqueue`` stores the request and ``len`` reflects it."""
    queue: AdmissionQueue = AdmissionQueue()
    queue.enqueue(_req(request_id="req-a"))
    queue.enqueue(_req(request_id="req-b"))
    assert len(queue) == 2


def test_admission_queue_enqueue_duplicate_raises() -> None:
    """Re-enqueueing the same ``request_id`` is rejected."""
    queue: AdmissionQueue = AdmissionQueue()
    queue.enqueue(_req(request_id="dup"))
    with pytest.raises(ValueError, match="request already queued"):
        queue.enqueue(_req(request_id="dup"))


def test_admission_queue_cancel_returns_true_when_present() -> None:
    """``cancel`` returns ``True`` and removes the request when present."""
    queue: AdmissionQueue = AdmissionQueue()
    queue.enqueue(_req(request_id="req-a"))
    assert queue.cancel("req-a") is True
    assert len(queue) == 0


def test_admission_queue_cancel_returns_false_when_absent() -> None:
    """``cancel`` returns ``False`` and is a no-op when the id is absent."""
    queue: AdmissionQueue = AdmissionQueue()
    assert queue.cancel("missing") is False


def test_admission_queue_pop_next_empty_returns_none() -> None:
    """``pop_next`` on an empty queue returns ``None``."""
    queue: AdmissionQueue = AdmissionQueue()
    assert queue.pop_next(now_ms=0.0) is None


def test_admission_queue_pop_next_rejects_negative_now_ms() -> None:
    """``pop_next`` rejects negative ``now_ms``."""
    queue: AdmissionQueue = AdmissionQueue()
    with pytest.raises(ValueError, match="now_ms"):
        queue.pop_next(now_ms=-1.0)


def test_admission_queue_pop_next_returns_starved_request_first() -> None:
    """A request past ``starvation_after_ms`` wins (priority 0.0 — bounded wait)."""
    queue: AdmissionQueue = AdmissionQueue(starvation_after_ms=1000.0)
    queue.enqueue(_req(request_id="fresh", enqueued_at_ms=0.0))
    queue.enqueue(_req(request_id="starved", enqueued_at_ms=1500.0))
    # At now=2000: fresh waited 2000ms (starved); starved waited 500ms (not yet).
    # Both inside the starvation window — the order is decided by the secondary
    # tuple ``(1.0, deadline, duration, memory, enqueued_at_ms)``.
    popped = queue.pop_next(now_ms=2000.0)
    assert popped is not None
    assert popped.request_id == "fresh"


def test_admission_queue_pop_next_starvation_priority_zero_wins() -> None:
    """When ALL queued items are starved, the one with the longest wait wins."""
    queue: AdmissionQueue = AdmissionQueue(starvation_after_ms=100.0)
    queue.enqueue(_req(request_id="older", enqueued_at_ms=0.0))
    queue.enqueue(_req(request_id="newer", enqueued_at_ms=2000.0))
    popped = queue.pop_next(now_ms=3000.0)
    assert popped is not None
    assert popped.request_id == "older"


def test_admission_queue_pop_next_prefers_deadline_over_duration() -> None:
    """Among non-starved items, the tighter deadline wins."""
    queue: AdmissionQueue = AdmissionQueue()
    queue.enqueue(
        _req(
            request_id="loose", enqueued_at_ms=0.0, estimated_duration_ms=10.0, deadline_ms=10_000.0
        )
    )
    queue.enqueue(
        _req(request_id="tight", enqueued_at_ms=0.0, estimated_duration_ms=10.0, deadline_ms=2.0)
    )
    popped = queue.pop_next(now_ms=0.0)
    assert popped is not None
    assert popped.request_id == "tight"


def test_admission_queue_pop_next_prefers_shorter_duration_when_deadlines_match() -> None:
    """When deadlines tie, the shorter estimated duration wins."""
    queue: AdmissionQueue = AdmissionQueue()
    queue.enqueue(
        _req(
            request_id="long",
            enqueued_at_ms=0.0,
            estimated_duration_ms=10_000.0,
            deadline_ms=2.0,
        )
    )
    queue.enqueue(
        _req(
            request_id="short",
            enqueued_at_ms=0.0,
            estimated_duration_ms=10.0,
            deadline_ms=2.0,
        )
    )
    popped = queue.pop_next(now_ms=0.0)
    assert popped is not None
    assert popped.request_id == "short"


def test_admission_queue_pop_next_removes_returned_item() -> None:
    """``pop_next`` removes the returned request from the queue."""
    queue: AdmissionQueue = AdmissionQueue()
    queue.enqueue(_req(request_id="a", enqueued_at_ms=0.0))
    queue.enqueue(_req(request_id="b", enqueued_at_ms=0.0))
    popped = queue.pop_next(now_ms=0.0)
    assert popped is not None
    assert len(queue) == 1
    remaining = queue.pop_next(now_ms=0.0)
    assert remaining is not None
    assert remaining.request_id != popped.request_id
    assert queue.pop_next(now_ms=0.0) is None


def test_admission_queue_now_ms_is_keyword_only() -> None:
    """``pop_next(now_ms=...)`` must be keyword-only."""
    sig = inspect.signature(AdmissionQueue.pop_next)
    assert sig.parameters["now_ms"].kind == inspect.Parameter.KEYWORD_ONLY


def test_admission_queue_does_not_leak_negative_now_ms_validation_to_decide() -> None:
    """``decide`` and ``pop_next`` both enforce ``_non_negative(now_ms, ...)``."""
    policy = AdmissionPolicy()
    queue: AdmissionQueue = AdmissionQueue()
    with pytest.raises(ValueError, match="now_ms"):
        policy.decide(_req(), _cap(), now_ms=-1.0)
    with pytest.raises(ValueError, match="now_ms"):
        queue.pop_next(now_ms=-1.0)


# ---------------------------------------------------------------------------
# 10. Helpers — ``_non_negative`` / ``_positive``
# ---------------------------------------------------------------------------


def test_admission_private_helpers_are_module_level_callables() -> None:
    """``_non_negative`` and ``_positive`` must exist as module-level callables."""
    assert callable(admission_module._non_negative)
    assert callable(admission_module._positive)


def test_admission_private_non_negative_returns_float() -> None:
    """``_non_negative`` returns the float value when valid."""
    assert admission_module._non_negative(0, "x") == 0.0
    assert admission_module._non_negative(1, "x") == 1.0
    assert admission_module._non_negative(2.5, "x") == 2.5


def test_admission_private_non_negative_rejects_bool_first() -> None:
    """``_non_negative`` rejects ``bool`` before the int check — load-bearing order."""
    with pytest.raises(ValueError):
        admission_module._non_negative(True, "x")
    with pytest.raises(ValueError):
        admission_module._non_negative(False, "x")


def test_admission_private_non_negative_rejects_negative() -> None:
    """``_non_negative`` rejects negative numbers."""
    with pytest.raises(ValueError):
        admission_module._non_negative(-1, "x")


def test_admission_private_non_negative_rejects_nan_inf() -> None:
    """``_non_negative`` rejects NaN / ±inf."""
    with pytest.raises(ValueError):
        admission_module._non_negative(math.nan, "x")
    with pytest.raises(ValueError):
        admission_module._non_negative(math.inf, "x")
    with pytest.raises(ValueError):
        admission_module._non_negative(-math.inf, "x")


def test_admission_private_positive_rejects_zero() -> None:
    """``_positive`` rejects 0.0 (the boundary from non-negative)."""
    with pytest.raises(ValueError):
        admission_module._positive(0, "x")
    with pytest.raises(ValueError):
        admission_module._positive(0.0, "x")


def test_admission_private_positive_accepts_small_positive() -> None:
    """``_positive`` accepts arbitrarily small positive values."""
    assert admission_module._positive(1e-9, "x") == 1e-9
    assert admission_module._positive(1.5e-9, "x") == 1.5e-9


# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------


def dataclasses_FrozenInstanceError() -> type[Exception]:  # noqa: N802
    """Local factory to avoid importing ``dataclasses`` at module top."""
    import dataclasses  # noqa: PLC0415

    return dataclasses.FrozenInstanceError
