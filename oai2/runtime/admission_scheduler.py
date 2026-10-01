"""Bridge admission/backpressure decisions into the safe batching scheduler.

This is a deterministic policy/scheduler integration layer. It does not execute
MLX work; callers still own live resource sampling and execution lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .admission import (
    AdmissionAction,
    AdmissionDecision,
    AdmissionPolicy,
    AdmissionQueue,
    AdmissionRequest,
    CapacitySnapshot,
)
from .scheduler import BatchPlan, SafeBatchScheduler, ScheduledRequest, SchedulerMetrics


@dataclass(slots=True, frozen=True)
class AdmissionScheduledRequest:
    admission: AdmissionRequest
    scheduled: ScheduledRequest

    def __post_init__(self) -> None:
        if self.admission.request_id != self.scheduled.request_id:
            raise ValueError("admission and scheduled request_id must match")


@dataclass(slots=True, frozen=True)
class AdmissionSchedulerMetrics:
    pending_admission: int
    ready_queue_depth: int
    ready_sessions: int
    cancelled_ready_requests: int
    batches_emitted: int
    requests_emitted: int


class AdmissionBatchController:
    """Apply admission/backpressure before exact-compatibility batching."""

    def __init__(
        self,
        *,
        policy: AdmissionPolicy | None = None,
        scheduler: SafeBatchScheduler | None = None,
    ) -> None:
        self._policy = policy or AdmissionPolicy()
        self._scheduler = scheduler or SafeBatchScheduler()
        self._pending_queue = AdmissionQueue(
            starvation_after_ms=self._policy.starvation_after_ms
        )
        self._pending: dict[str, AdmissionScheduledRequest] = {}

    def submit(
        self,
        request: AdmissionScheduledRequest,
        capacity: CapacitySnapshot,
        *,
        now_ms: float,
    ) -> AdmissionDecision:
        if request.admission.request_id in self._pending:
            raise ValueError(f"request already pending: {request.admission.request_id}")
        effective = replace(capacity, queue_depth=len(self._pending))
        decision = self._policy.decide(request.admission, effective, now_ms=now_ms)

        if decision.action is AdmissionAction.ADMIT:
            self._scheduler.enqueue(request.scheduled)
        elif decision.action is AdmissionAction.QUEUE:
            self._pending[request.admission.request_id] = request
            self._pending_queue.enqueue(request.admission)
        return decision

    def promote_one(
        self,
        capacity: CapacitySnapshot,
        *,
        now_ms: float,
    ) -> AdmissionDecision | None:
        admission = self._pending_queue.pop_next(now_ms=now_ms)
        if admission is None:
            return None

        request = self._pending.pop(admission.request_id)
        effective = replace(capacity, queue_depth=len(self._pending))
        decision = self._policy.decide(admission, effective, now_ms=now_ms)

        if decision.action is AdmissionAction.ADMIT:
            self._scheduler.enqueue(request.scheduled)
        elif decision.action is AdmissionAction.QUEUE:
            self._pending[admission.request_id] = request
            self._pending_queue.enqueue(admission)
        return decision

    def cancel(self, request_id: str) -> bool:
        pending = self._pending.pop(request_id, None)
        if pending is not None:
            self._pending_queue.cancel(request_id)
            return True
        return self._scheduler.cancel_request(request_id)

    def pop_batch(self, *, max_batch_size: int) -> BatchPlan | None:
        return self._scheduler.pop_batch(max_batch_size=max_batch_size)

    @property
    def metrics(self) -> AdmissionSchedulerMetrics:
        scheduler_metrics: SchedulerMetrics = self._scheduler.metrics
        return AdmissionSchedulerMetrics(
            pending_admission=len(self._pending),
            ready_queue_depth=scheduler_metrics.queue_depth,
            ready_sessions=scheduler_metrics.queued_sessions,
            cancelled_ready_requests=scheduler_metrics.cancelled_requests,
            batches_emitted=scheduler_metrics.batches_emitted,
            requests_emitted=scheduler_metrics.requests_emitted,
        )


__all__ = [
    "AdmissionScheduledRequest",
    "AdmissionSchedulerMetrics",
    "AdmissionBatchController",
]
