"""Deterministic admission control and backpressure policy for local OAI runtime.

This module implements the policy core required by WI-QOS-002. It is deliberately
scheduler-neutral: callers provide the current capacity snapshot and enqueue or
execute according to the returned decision. Integration with the live service and
inference scheduler remains separate work.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from ..reasoning.modes import ReasoningMode


class AdmissionAction(StrEnum):
    ADMIT = "admit"
    QUEUE = "queue"
    REJECT = "reject"


class AdmissionReason(StrEnum):
    CAPACITY_AVAILABLE = "capacity_available"
    MEMORY_PRESSURE = "memory_pressure"
    ACTIVE_LIMIT = "active_limit"
    QUEUE_FULL = "queue_full"
    DEADLINE_EXPIRED = "deadline_expired"
    REQUEST_TOO_LARGE = "request_too_large"


@dataclass(slots=True, frozen=True)
class AdmissionRequest:
    request_id: str
    mode: ReasoningMode
    per_lane_memory_gb: float
    estimated_duration_ms: float
    enqueued_at_ms: float
    deadline_ms: float | None = None
    swarm_lanes: int = 1

    def __post_init__(self) -> None:
        if not self.request_id or self.request_id != self.request_id.strip():
            raise ValueError("request_id must be a non-empty normalized string")
        _positive(self.per_lane_memory_gb, "per_lane_memory_gb")
        _positive(self.estimated_duration_ms, "estimated_duration_ms")
        _non_negative(self.enqueued_at_ms, "enqueued_at_ms")
        if self.deadline_ms is not None:
            _non_negative(self.deadline_ms, "deadline_ms")
        if isinstance(self.swarm_lanes, bool) or not isinstance(self.swarm_lanes, int):
            raise ValueError("swarm_lanes must be a positive integer")
        if self.swarm_lanes <= 0:
            raise ValueError("swarm_lanes must be a positive integer")
        if self.mode is not ReasoningMode.SWARM and self.swarm_lanes != 1:
            raise ValueError("non-SWARM requests must use exactly one lane")

    @property
    def aggregate_memory_gb(self) -> float:
        return self.per_lane_memory_gb * self.swarm_lanes


@dataclass(slots=True, frozen=True)
class CapacitySnapshot:
    total_memory_gb: float
    used_memory_gb: float
    reserved_memory_gb: float
    active_tasks: int
    max_active_tasks: int
    queue_depth: int
    max_queue_depth: int

    def __post_init__(self) -> None:
        _positive(self.total_memory_gb, "total_memory_gb")
        _non_negative(self.used_memory_gb, "used_memory_gb")
        _non_negative(self.reserved_memory_gb, "reserved_memory_gb")
        if self.used_memory_gb > self.total_memory_gb:
            raise ValueError("used_memory_gb cannot exceed total_memory_gb")
        if self.reserved_memory_gb >= self.total_memory_gb:
            raise ValueError("reserved_memory_gb must be less than total_memory_gb")
        for name in ("active_tasks", "max_active_tasks", "queue_depth", "max_queue_depth"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.max_active_tasks <= 0:
            raise ValueError("max_active_tasks must be positive")
        if self.queue_depth > self.max_queue_depth:
            raise ValueError("queue_depth cannot exceed max_queue_depth")

    @property
    def available_memory_gb(self) -> float:
        return max(self.total_memory_gb - self.reserved_memory_gb - self.used_memory_gb, 0.0)


@dataclass(slots=True, frozen=True)
class AdmissionDecision:
    request_id: str
    action: AdmissionAction
    reason: AdmissionReason
    required_memory_gb: float
    available_memory_gb: float


@dataclass(slots=True, frozen=True)
class AdmissionPolicy:
    starvation_after_ms: float = 30_000.0

    def __post_init__(self) -> None:
        _positive(self.starvation_after_ms, "starvation_after_ms")

    def decide(
        self,
        request: AdmissionRequest,
        capacity: CapacitySnapshot,
        *,
        now_ms: float,
    ) -> AdmissionDecision:
        now = _non_negative(now_ms, "now_ms")
        required = request.aggregate_memory_gb
        available = capacity.available_memory_gb

        if request.deadline_ms is not None and now >= request.deadline_ms:
            return AdmissionDecision(
                request_id=request.request_id,
                action=AdmissionAction.REJECT,
                reason=AdmissionReason.DEADLINE_EXPIRED,
                required_memory_gb=required,
                available_memory_gb=available,
            )

        usable_total = capacity.total_memory_gb - capacity.reserved_memory_gb
        if required > usable_total:
            return AdmissionDecision(
                request_id=request.request_id,
                action=AdmissionAction.REJECT,
                reason=AdmissionReason.REQUEST_TOO_LARGE,
                required_memory_gb=required,
                available_memory_gb=available,
            )

        if capacity.active_tasks < capacity.max_active_tasks and required <= available:
            return AdmissionDecision(
                request_id=request.request_id,
                action=AdmissionAction.ADMIT,
                reason=AdmissionReason.CAPACITY_AVAILABLE,
                required_memory_gb=required,
                available_memory_gb=available,
            )

        reason = (
            AdmissionReason.ACTIVE_LIMIT
            if capacity.active_tasks >= capacity.max_active_tasks
            else AdmissionReason.MEMORY_PRESSURE
        )
        if capacity.queue_depth < capacity.max_queue_depth:
            return AdmissionDecision(
                request_id=request.request_id,
                action=AdmissionAction.QUEUE,
                reason=reason,
                required_memory_gb=required,
                available_memory_gb=available,
            )
        return AdmissionDecision(
            request_id=request.request_id,
            action=AdmissionAction.REJECT,
            reason=AdmissionReason.QUEUE_FULL,
            required_memory_gb=required,
            available_memory_gb=available,
        )


class AdmissionQueue:
    """Small deterministic queue with bounded-wait anti-starvation selection."""

    def __init__(self, *, starvation_after_ms: float = 30_000.0) -> None:
        self._starvation_after_ms = _positive(starvation_after_ms, "starvation_after_ms")
        self._items: dict[str, AdmissionRequest] = {}

    def __len__(self) -> int:
        return len(self._items)

    def enqueue(self, request: AdmissionRequest) -> None:
        if request.request_id in self._items:
            raise ValueError(f"request already queued: {request.request_id}")
        self._items[request.request_id] = request

    def cancel(self, request_id: str) -> bool:
        return self._items.pop(request_id, None) is not None

    def pop_next(self, *, now_ms: float) -> AdmissionRequest | None:
        now = _non_negative(now_ms, "now_ms")
        if not self._items:
            return None
        request = min(self._items.values(), key=lambda item: self._priority(item, now))
        del self._items[request.request_id]
        return request

    def _priority(self, request: AdmissionRequest, now_ms: float) -> tuple[float, ...]:
        waited = max(now_ms - request.enqueued_at_ms, 0.0)
        starved = waited >= self._starvation_after_ms
        deadline = request.deadline_ms if request.deadline_ms is not None else math.inf

        # Bounded wait wins first. Otherwise deadline-sensitive and short/small
        # work moves ahead of large DEEP/SWARM work, preventing head-of-line
        # blocking of small requests.
        if starved:
            return (0.0, request.enqueued_at_ms, deadline)
        return (
            1.0,
            deadline,
            request.estimated_duration_ms,
            request.aggregate_memory_gb,
            request.enqueued_at_ms,
        )


def _non_negative(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


def _positive(value: object, name: str) -> float:
    result = _non_negative(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


__all__ = [
    "AdmissionAction",
    "AdmissionReason",
    "AdmissionRequest",
    "CapacitySnapshot",
    "AdmissionDecision",
    "AdmissionPolicy",
    "AdmissionQueue",
]
