from __future__ import annotations

import pytest

from oai2.reasoning import ReasoningMode
from oai2.runtime.admission import (
    AdmissionAction,
    AdmissionPolicy,
    AdmissionQueue,
    AdmissionReason,
    AdmissionRequest,
    CapacitySnapshot,
)


def _request(
    request_id: str,
    *,
    mode: ReasoningMode = ReasoningMode.NORMAL,
    memory: float = 2.0,
    duration: float = 1_000.0,
    enqueued: float = 0.0,
    deadline: float | None = None,
    lanes: int = 1,
) -> AdmissionRequest:
    return AdmissionRequest(
        request_id=request_id,
        mode=mode,
        per_lane_memory_gb=memory,
        estimated_duration_ms=duration,
        enqueued_at_ms=enqueued,
        deadline_ms=deadline,
        swarm_lanes=lanes,
    )


def _capacity(
    *,
    used: float = 4.0,
    active: int = 1,
    queue: int = 0,
) -> CapacitySnapshot:
    return CapacitySnapshot(
        total_memory_gb=64.0,
        used_memory_gb=used,
        reserved_memory_gb=8.0,
        active_tasks=active,
        max_active_tasks=4,
        queue_depth=queue,
        max_queue_depth=8,
    )


def test_admits_when_memory_and_active_slot_are_available() -> None:
    decision = AdmissionPolicy().decide(
        _request("r1"),
        _capacity(),
        now_ms=100.0,
    )
    assert decision.action is AdmissionAction.ADMIT
    assert decision.reason is AdmissionReason.CAPACITY_AVAILABLE
    assert decision.required_memory_gb == 2.0
    assert decision.available_memory_gb == 52.0


def test_memory_pressure_queues_instead_of_overcommitting() -> None:
    decision = AdmissionPolicy().decide(
        _request("r1", memory=8.0),
        _capacity(used=52.0),
        now_ms=100.0,
    )
    assert decision.action is AdmissionAction.QUEUE
    assert decision.reason is AdmissionReason.MEMORY_PRESSURE


def test_active_limit_queues_when_memory_still_fits() -> None:
    decision = AdmissionPolicy().decide(
        _request("r1"),
        _capacity(active=4),
        now_ms=100.0,
    )
    assert decision.action is AdmissionAction.QUEUE
    assert decision.reason is AdmissionReason.ACTIVE_LIMIT


def test_queue_full_rejects_explicitly() -> None:
    decision = AdmissionPolicy().decide(
        _request("r1", memory=8.0),
        _capacity(used=52.0, queue=8),
        now_ms=100.0,
    )
    assert decision.action is AdmissionAction.REJECT
    assert decision.reason is AdmissionReason.QUEUE_FULL


def test_request_larger_than_usable_memory_is_rejected_not_queued() -> None:
    decision = AdmissionPolicy().decide(
        _request("r1", memory=57.0),
        _capacity(),
        now_ms=100.0,
    )
    assert decision.action is AdmissionAction.REJECT
    assert decision.reason is AdmissionReason.REQUEST_TOO_LARGE


def test_expired_deadline_is_rejected_before_capacity_check() -> None:
    decision = AdmissionPolicy().decide(
        _request("r1", deadline=100.0),
        _capacity(),
        now_ms=100.0,
    )
    assert decision.action is AdmissionAction.REJECT
    assert decision.reason is AdmissionReason.DEADLINE_EXPIRED


def test_swarm_uses_aggregate_lane_memory() -> None:
    request = _request(
        "swarm",
        mode=ReasoningMode.SWARM,
        memory=6.0,
        lanes=4,
    )
    assert request.aggregate_memory_gb == 24.0

    decision = AdmissionPolicy().decide(
        request,
        _capacity(used=40.0),
        now_ms=100.0,
    )
    assert decision.action is AdmissionAction.QUEUE
    assert decision.required_memory_gb == 24.0


def test_non_swarm_request_cannot_claim_multiple_lanes() -> None:
    with pytest.raises(ValueError, match="non-SWARM"):
        _request("bad", mode=ReasoningMode.NORMAL, lanes=2)


def test_queue_prefers_small_short_work_over_large_work_before_starvation() -> None:
    queue = AdmissionQueue(starvation_after_ms=30_000.0)
    queue.enqueue(
        _request(
            "deep",
            mode=ReasoningMode.DEEP,
            memory=16.0,
            duration=20_000.0,
            enqueued=0.0,
        )
    )
    queue.enqueue(
        _request(
            "small",
            mode=ReasoningMode.FURIOUS,
            memory=1.0,
            duration=100.0,
            enqueued=1_000.0,
        )
    )

    assert queue.pop_next(now_ms=5_000.0).request_id == "small"


def test_starvation_bound_eventually_promotes_long_waiting_work() -> None:
    queue = AdmissionQueue(starvation_after_ms=10_000.0)
    queue.enqueue(
        _request(
            "old-deep",
            mode=ReasoningMode.DEEP,
            memory=16.0,
            duration=20_000.0,
            enqueued=0.0,
        )
    )
    queue.enqueue(
        _request(
            "new-small",
            mode=ReasoningMode.FURIOUS,
            memory=1.0,
            duration=100.0,
            enqueued=9_500.0,
        )
    )

    assert queue.pop_next(now_ms=10_001.0).request_id == "old-deep"


def test_cancellation_removes_queued_work_immediately() -> None:
    queue = AdmissionQueue()
    queue.enqueue(_request("r1"))
    queue.enqueue(_request("r2"))

    assert len(queue) == 2
    assert queue.cancel("r1") is True
    assert len(queue) == 1
    assert queue.cancel("missing") is False
    assert queue.pop_next(now_ms=100.0).request_id == "r2"



def test_admission_symbols_are_exported_from_runtime_package() -> None:
    """Every name in ``oai2/runtime/admission.py`` ``__all__`` is importable
    from ``oai2.runtime`` and aliases its source-of-truth.

    Closes a coverage gap where ``oai2.runtime`` re-exported 7
    ``oai2/runtime/admission.py`` ``__all__`` entries via its own
    ``__all__`` but ``tests/test_admission.py`` only pinned 2 of them
    (``AdmissionPolicy`` and ``AdmissionQueue``) at the package
    surface. The remaining 5 — ``AdmissionAction``, ``AdmissionDecision``,
    ``AdmissionReason``, ``AdmissionRequest``, ``CapacitySnapshot`` —
    were not pinned at the package surface, leaving a 71 percent
    coverage gap on the admission policy's public contract.

    The 11 pre-existing tests in ``tests/test_admission.py`` are
    preserved byte-for-byte. The test now imports all 7 names from
    ``oai2.runtime`` (the package-level re-export) and asserts each is
    identical to the canonical binding in ``oai2.runtime.admission``
    (the source-of-truth module).
    """
    from oai2.runtime import (
        AdmissionAction as ExportedAction,
        AdmissionDecision as ExportedDecision,
        AdmissionPolicy as ExportedPolicy,
        AdmissionQueue as ExportedQueue,
        AdmissionReason as ExportedReason,
        AdmissionRequest as ExportedRequest,
        CapacitySnapshot as ExportedCapacitySnapshot,
    )
    from oai2.runtime.admission import (
        AdmissionAction,
        AdmissionDecision,
        AdmissionPolicy,
        AdmissionQueue,
        AdmissionReason,
        AdmissionRequest,
        CapacitySnapshot,
    )

    assert ExportedAction is AdmissionAction
    assert ExportedDecision is AdmissionDecision
    assert ExportedPolicy is AdmissionPolicy
    assert ExportedQueue is AdmissionQueue
    assert ExportedReason is AdmissionReason
    assert ExportedRequest is AdmissionRequest
    assert ExportedCapacitySnapshot is CapacitySnapshot