from __future__ import annotations

from oai2.reasoning import ReasoningMode
from oai2.runtime.admission import (
    AdmissionAction,
    AdmissionReason,
    AdmissionRequest,
    CapacitySnapshot,
)
from oai2.runtime.admission_scheduler import (
    AdmissionBatchController,
    AdmissionScheduledRequest,
)
from oai2.runtime.scheduler import ScheduledRequest, SessionCompatibilityKey


def _key(*, model: str = "model-a") -> SessionCompatibilityKey:
    return SessionCompatibilityKey(
        model_id=model,
        tokenizer_version="tok-v1",
        prefix_digest="prefix-v1",
        tool_schema_version="tools-v1",
        world_state_version="world-v1",
        security_context="user-a",
    )


def _request(
    request_id: str,
    session_id: str,
    *,
    memory: float = 2.0,
    model: str = "model-a",
    at: float = 0.0,
) -> AdmissionScheduledRequest:
    return AdmissionScheduledRequest(
        admission=AdmissionRequest(
            request_id=request_id,
            mode=ReasoningMode.NORMAL,
            per_lane_memory_gb=memory,
            estimated_duration_ms=1000.0,
            enqueued_at_ms=at,
        ),
        scheduled=ScheduledRequest(
            request_id=request_id,
            session_id=session_id,
            compatibility=_key(model=model),
            enqueued_at_ms=at,
        ),
    )


def _capacity(
    *,
    used: float = 4.0,
    active: int = 0,
    queue_depth: int = 0,
    max_queue: int = 8,
) -> CapacitySnapshot:
    return CapacitySnapshot(
        total_memory_gb=64.0,
        used_memory_gb=used,
        reserved_memory_gb=8.0,
        active_tasks=active,
        max_active_tasks=4,
        queue_depth=queue_depth,
        max_queue_depth=max_queue,
    )


def test_admitted_request_flows_into_safe_batch_scheduler() -> None:
    controller = AdmissionBatchController()

    decision = controller.submit(
        _request("r1", "s1"),
        _capacity(),
        now_ms=10.0,
    )

    assert decision.action is AdmissionAction.ADMIT
    batch = controller.pop_batch(max_batch_size=8)
    assert batch is not None
    assert [item.request_id for item in batch.requests] == ["r1"]


def test_memory_pressure_queues_then_promotes_when_capacity_frees() -> None:
    controller = AdmissionBatchController()

    decision = controller.submit(
        _request("r1", "s1", memory=8.0),
        _capacity(used=52.0),
        now_ms=10.0,
    )

    assert decision.action is AdmissionAction.QUEUE
    assert decision.reason is AdmissionReason.MEMORY_PRESSURE
    assert controller.metrics.pending_admission == 1
    assert controller.pop_batch(max_batch_size=8) is None

    promoted = controller.promote_one(
        _capacity(used=10.0),
        now_ms=20.0,
    )
    assert promoted is not None
    assert promoted.action is AdmissionAction.ADMIT
    assert controller.metrics.pending_admission == 0
    assert controller.metrics.ready_queue_depth == 1


def test_still_over_capacity_request_remains_pending_after_promotion_attempt() -> None:
    controller = AdmissionBatchController()
    controller.submit(
        _request("r1", "s1", memory=8.0),
        _capacity(used=52.0),
        now_ms=10.0,
    )

    decision = controller.promote_one(
        _capacity(used=52.0),
        now_ms=20.0,
    )

    assert decision is not None
    assert decision.action is AdmissionAction.QUEUE
    assert controller.metrics.pending_admission == 1


def test_cancellation_removes_pending_or_ready_work() -> None:
    controller = AdmissionBatchController()
    controller.submit(
        _request("pending", "s1", memory=8.0),
        _capacity(used=52.0),
        now_ms=10.0,
    )
    controller.submit(
        _request("ready", "s2"),
        _capacity(),
        now_ms=10.0,
    )

    assert controller.cancel("pending") is True
    assert controller.cancel("ready") is True
    assert controller.cancel("missing") is False
    assert controller.metrics.pending_admission == 0
    assert controller.metrics.ready_queue_depth == 0


def test_queue_full_rejection_does_not_create_pending_work() -> None:
    controller = AdmissionBatchController()
    controller.submit(
        _request("first", "s1", memory=8.0),
        _capacity(used=52.0, max_queue=1),
        now_ms=10.0,
    )

    decision = controller.submit(
        _request("second", "s2", memory=8.0),
        _capacity(used=52.0, max_queue=1),
        now_ms=11.0,
    )

    assert decision.action is AdmissionAction.REJECT
    assert decision.reason is AdmissionReason.QUEUE_FULL
    assert controller.metrics.pending_admission == 1


def test_admission_does_not_bypass_exact_batch_compatibility() -> None:
    controller = AdmissionBatchController()
    controller.submit(
        _request("a", "s1", model="model-a"),
        _capacity(),
        now_ms=10.0,
    )
    controller.submit(
        _request("b", "s2", model="model-b"),
        _capacity(),
        now_ms=11.0,
    )

    first = controller.pop_batch(max_batch_size=8)
    second = controller.pop_batch(max_batch_size=8)

    assert first is not None and len(first.requests) == 1
    assert second is not None and len(second.requests) == 1
    assert {first.requests[0].request_id, second.requests[0].request_id} == {
        "a",
        "b",
    }


def test_admission_scheduler_symbols_are_exported_from_runtime_package() -> None:
    from oai2.runtime import AdmissionBatchController as ExportedController
    from oai2.runtime import AdmissionScheduledRequest as ExportedScheduledRequest
    from oai2.runtime import AdmissionSchedulerMetrics as ExportedMetrics
    from oai2.runtime.admission_scheduler import (
        AdmissionBatchController,
        AdmissionScheduledRequest,
        AdmissionSchedulerMetrics,
    )

    assert ExportedController is AdmissionBatchController
    assert ExportedScheduledRequest is AdmissionScheduledRequest
    assert ExportedMetrics is AdmissionSchedulerMetrics