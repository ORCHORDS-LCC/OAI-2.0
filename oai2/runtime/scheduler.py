"""Deterministic session-isolation and safe-batching scheduler core.

This module establishes the correctness boundary for WI-INF-001 without
pretending the live MLX batching/runtime exists yet. Requests may batch only
when every context identity/version that affects model output is identical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class SessionCompatibilityKey:
    model_id: str
    tokenizer_version: str
    prefix_digest: str
    tool_schema_version: str
    world_state_version: str
    security_context: str

    def __post_init__(self) -> None:
        for name in (
            "model_id",
            "tokenizer_version",
            "prefix_digest",
            "tool_schema_version",
            "world_state_version",
            "security_context",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{name} must be a non-empty normalized string")


@dataclass(slots=True, frozen=True)
class ScheduledRequest:
    request_id: str
    session_id: str
    compatibility: SessionCompatibilityKey
    enqueued_at_ms: float

    def __post_init__(self) -> None:
        for name in ("request_id", "session_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{name} must be a non-empty normalized string")
        if (
            isinstance(self.enqueued_at_ms, bool)
            or not isinstance(self.enqueued_at_ms, (int, float))
            or not math.isfinite(float(self.enqueued_at_ms))
            or float(self.enqueued_at_ms) < 0.0
        ):
            raise ValueError("enqueued_at_ms must be a finite non-negative number")


@dataclass(slots=True, frozen=True)
class SchedulerMetrics:
    queue_depth: int
    queued_sessions: int
    cancelled_requests: int
    batches_emitted: int
    requests_emitted: int


@dataclass(slots=True, frozen=True)
class BatchPlan:
    compatibility: SessionCompatibilityKey
    requests: tuple[ScheduledRequest, ...]

    def __post_init__(self) -> None:
        if not self.requests:
            raise ValueError("batch must contain at least one request")
        sessions: set[str] = set()
        for request in self.requests:
            if request.compatibility != self.compatibility:
                raise ValueError("batch contains an incompatible request")
            if request.session_id in sessions:
                raise ValueError("batch cannot contain multiple requests from one session")
            sessions.add(request.session_id)


@dataclass(slots=True)
class SafeBatchScheduler:
    """Queue requests and emit only exact-compatibility batches."""

    _requests: dict[str, ScheduledRequest] = field(default_factory=dict)
    _session_to_request: dict[str, str] = field(default_factory=dict)
    _cancelled_requests: int = 0
    _batches_emitted: int = 0
    _requests_emitted: int = 0

    def enqueue(self, request: ScheduledRequest) -> None:
        if request.request_id in self._requests:
            raise ValueError(f"request already queued: {request.request_id}")
        if request.session_id in self._session_to_request:
            raise ValueError(
                f"session already has queued work: {request.session_id}"
            )
        self._requests[request.request_id] = request
        self._session_to_request[request.session_id] = request.request_id

    def cancel_request(self, request_id: str) -> bool:
        request = self._requests.pop(request_id, None)
        if request is None:
            return False
        self._session_to_request.pop(request.session_id, None)
        self._cancelled_requests += 1
        return True

    def cancel_session(self, session_id: str) -> bool:
        request_id = self._session_to_request.get(session_id)
        if request_id is None:
            return False
        return self.cancel_request(request_id)

    def pop_batch(self, *, max_batch_size: int) -> BatchPlan | None:
        if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int):
            raise ValueError("max_batch_size must be a positive integer")
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer")
        if not self._requests:
            return None

        first = min(
            self._requests.values(),
            key=lambda item: (item.enqueued_at_ms, item.request_id),
        )
        compatible = sorted(
            (
                item
                for item in self._requests.values()
                if item.compatibility == first.compatibility
            ),
            key=lambda item: (item.enqueued_at_ms, item.request_id),
        )
        selected = tuple(compatible[:max_batch_size])

        for request in selected:
            del self._requests[request.request_id]
            self._session_to_request.pop(request.session_id, None)

        self._batches_emitted += 1
        self._requests_emitted += len(selected)
        return BatchPlan(compatibility=first.compatibility, requests=selected)

    @property
    def metrics(self) -> SchedulerMetrics:
        return SchedulerMetrics(
            queue_depth=len(self._requests),
            queued_sessions=len(self._session_to_request),
            cancelled_requests=self._cancelled_requests,
            batches_emitted=self._batches_emitted,
            requests_emitted=self._requests_emitted,
        )


__all__ = [
    "SessionCompatibilityKey",
    "ScheduledRequest",
    "SchedulerMetrics",
    "BatchPlan",
    "SafeBatchScheduler",
]
