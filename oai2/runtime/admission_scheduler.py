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
    AdmissionReason,
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


@dataclass(slots=True, frozen=True)
class AdmissionDecisionTrace:
    """Public-safe metadata explaining the most recent admission decision.

    Request/session identifiers and prompt/model payloads are intentionally
    excluded so callers can expose this shape in diagnostics without leaking
    user or workload identity.
    """

    action: AdmissionAction
    reason: AdmissionReason
    mode: str
    swarm_lanes: int
    required_memory_gb: float
    available_memory_gb: float
    active_tasks: int
    max_active_tasks: int
    pending_admission: int
    max_queue_depth: int
    ready_queue_depth: int
    deadline_present: bool


@dataclass(slots=True, frozen=True)
class AdmissionTelemetry:
    decisions_total: int
    admitted: int
    queued: int
    rejected: int
    cancelled: int
    reason_counts: tuple[tuple[str, int], ...]
    last_decision: AdmissionDecisionTrace | None


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
        self._decision_counts: dict[AdmissionAction, int] = {
            action: 0 for action in AdmissionAction
        }
        self._reason_counts: dict[AdmissionReason, int] = {}
        self._cancelled = 0
        self._last_decision: AdmissionDecisionTrace | None = None

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
        self._record_decision(request.admission, decision, effective)
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
        self._record_decision(admission, decision, effective)
        return decision

    def cancel(self, request_id: str) -> bool:
        pending = self._pending.pop(request_id, None)
        if pending is not None:
            self._pending_queue.cancel(request_id)
            self._cancelled += 1
            return True
        cancelled = self._scheduler.cancel_request(request_id)
        if cancelled:
            self._cancelled += 1
        return cancelled

    def pop_batch(self, *, max_batch_size: int) -> BatchPlan | None:
        return self._scheduler.pop_batch(max_batch_size=max_batch_size)

    def _record_decision(
        self,
        request: AdmissionRequest,
        decision: AdmissionDecision,
        capacity: CapacitySnapshot,
    ) -> None:
        self._decision_counts[decision.action] += 1
        self._reason_counts[decision.reason] = self._reason_counts.get(decision.reason, 0) + 1
        scheduler_metrics = self._scheduler.metrics
        self._last_decision = AdmissionDecisionTrace(
            action=decision.action,
            reason=decision.reason,
            mode=request.mode.value,
            swarm_lanes=request.swarm_lanes,
            required_memory_gb=decision.required_memory_gb,
            available_memory_gb=decision.available_memory_gb,
            active_tasks=capacity.active_tasks,
            max_active_tasks=capacity.max_active_tasks,
            pending_admission=len(self._pending),
            max_queue_depth=capacity.max_queue_depth,
            ready_queue_depth=scheduler_metrics.queue_depth,
            deadline_present=request.deadline_ms is not None,
        )

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

    @property
    def telemetry(self) -> AdmissionTelemetry:
        return AdmissionTelemetry(
            decisions_total=sum(self._decision_counts.values()),
            admitted=self._decision_counts[AdmissionAction.ADMIT],
            queued=self._decision_counts[AdmissionAction.QUEUE],
            rejected=self._decision_counts[AdmissionAction.REJECT],
            cancelled=self._cancelled,
            reason_counts=tuple(
                sorted((reason.value, count) for reason, count in self._reason_counts.items())
            ),
            last_decision=self._last_decision,
        )


__all__ = [
    "AdmissionScheduledRequest",
    "AdmissionSchedulerMetrics",
    "AdmissionDecisionTrace",
    "AdmissionTelemetry",
    "AdmissionBatchController",
]
