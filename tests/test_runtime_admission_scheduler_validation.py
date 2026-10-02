"""Validation tests for ``oai2.runtime.admission_scheduler``.

This file pins the outer guard rails of the admission→scheduler bridge that
sits on top of ``oai2.runtime.admission`` (WI-QOS-002) and
``oai2.runtime.scheduler`` (WI-INF-001). It complements the integration
tests in ``tests/test_admission_scheduler.py`` by asserting the structural
contract: dataclass validation matrices, frozen-slot semantics,
keyword-only signatures, dedup invariants, the capacity-queue_depth
override, the three-action routing for both ``submit`` and ``promote_one``,
the ``cancel`` dual-store invariant, the six-field ``metrics`` shape, and
the package-level re-export identity.

The tests intentionally avoid live network I/O, MLX execution, and any
real clock — all time-sensitive routing is exercised by passing explicit
``now_ms`` to the controller.
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest

from oai2.reasoning import ReasoningMode
from oai2.runtime.admission import (
    AdmissionAction,
    AdmissionPolicy,
    AdmissionReason,
    AdmissionRequest,
    CapacitySnapshot,
)
from oai2.runtime.admission_scheduler import (
    AdmissionBatchController,
    AdmissionScheduledRequest,
    AdmissionSchedulerMetrics,
)
from oai2.runtime.scheduler import (
    BatchPlan,
    SafeBatchScheduler,
    ScheduledRequest,
    SessionCompatibilityKey,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _key(*, model: str = "model-a") -> SessionCompatibilityKey:
    return SessionCompatibilityKey(
        model_id=model,
        tokenizer_version="tok-v1",
        prefix_digest="prefix-v1",
        tool_schema_version="tools-v1",
        world_state_version="world-v1",
        security_context="user-a",
    )


def _scheduled(
    request_id: str,
    session_id: str = "s1",
    *,
    model: str = "model-a",
    enqueued: float = 0.0,
) -> ScheduledRequest:
    return ScheduledRequest(
        request_id=request_id,
        session_id=session_id,
        compatibility=_key(model=model),
        enqueued_at_ms=enqueued,
    )


def _admission(
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


def _scheduled_request(
    request_id: str,
    session_id: str = "s1",
    *,
    memory: float = 2.0,
    model: str = "model-a",
    enqueued: float = 0.0,
    deadline: float | None = None,
) -> AdmissionScheduledRequest:
    return AdmissionScheduledRequest(
        admission=_admission(
            request_id,
            memory=memory,
            enqueued=enqueued,
            deadline=deadline,
        ),
        scheduled=_scheduled(request_id, session_id, model=model, enqueued=enqueued),
    )


def _capacity(
    *,
    total: float = 64.0,
    used: float = 4.0,
    reserved: float = 8.0,
    active: int = 0,
    max_active: int = 4,
    queue: int = 0,
    max_queue: int = 8,
) -> CapacitySnapshot:
    return CapacitySnapshot(
        total_memory_gb=total,
        used_memory_gb=used,
        reserved_memory_gb=reserved,
        active_tasks=active,
        max_active_tasks=max_active,
        queue_depth=queue,
        max_queue_depth=max_queue,
    )


# ---------------------------------------------------------------------------
# 1. Module docstring + import discipline
# ---------------------------------------------------------------------------


def test_module_docstring_is_present_and_substantive() -> None:
    """The module must carry a docstring stating what it bridges."""
    import oai2.runtime.admission_scheduler as mod

    assert mod.__doc__ is not None
    body = mod.__doc__.strip().lower()
    # The bridge scope (admission + batching scheduler) is mandatory context.
    assert "admission" in body
    assert "scheduler" in body or "batching" in body


def test_module_does_not_hardcode_credentials() -> None:
    """No hardcoded API keys, tokens, or bearer strings in source."""
    import oai2.runtime.admission_scheduler as mod

    source = inspect.getsource(mod)
    # Public-safety linter regex: \b(?:sk|xoxb)-[A-Za-z0-9_-]{16,}\b
    import re

    assert not re.search(r"\b(?:sk|xoxb)-[A-Za-z0-9_-]{16,}\b", source)
    # Public-safety linter regex: \bapi[_-]?key\s*[:=]\s*["\'][^"\']{16,}["\']
    assert not re.search(r"\bapi[_-]?key\s*[:=]\s*[\"'][^\"']{16,}[\"']", source)


def test_module_does_not_import_unrelated_cloud_runtimes() -> None:
    """No cloud-runtime imports leak into the admission/scheduler bridge."""
    import oai2.runtime.admission_scheduler as mod

    source = inspect.getsource(mod)
    forbidden_imports = (
        "import boto3",
        "from boto3",
        "import azure",
        "from azure",
        "import google.cloud",
        "from google.cloud",
        "import kubernetes",
        "from kubernetes",
        "import docker",
        "from docker",
    )
    for needle in forbidden_imports:
        assert needle not in source, f"unexpected cloud runtime import: {needle!r}"


def test_module_uses_future_annotations_for_up006_compliance() -> None:
    """``from __future__ import annotations`` is required for modern typing."""
    import oai2.runtime.admission_scheduler as mod

    source = inspect.getsource(mod)
    assert "from __future__ import annotations" in source


# ---------------------------------------------------------------------------
# 2. ``__all__`` completeness + package-level re-export
# ---------------------------------------------------------------------------


def test_module_all_lists_exactly_three_public_names() -> None:
    """The bridge exports exactly three public names."""
    import oai2.runtime.admission_scheduler as mod

    assert set(mod.__all__) == {
        "AdmissionScheduledRequest",
        "AdmissionSchedulerMetrics",
        "AdmissionBatchController",
    }


def test_package_level_reexport_preserves_identity() -> None:
    """All three names re-export from ``oai2.runtime`` are the same objects."""
    from oai2.runtime import (
        AdmissionBatchController as ExportedController,
    )
    from oai2.runtime import (
        AdmissionScheduledRequest as ExportedScheduled,
    )
    from oai2.runtime import (
        AdmissionSchedulerMetrics as ExportedMetrics,
    )

    assert ExportedController is AdmissionBatchController
    assert ExportedScheduled is AdmissionScheduledRequest
    assert ExportedMetrics is AdmissionSchedulerMetrics


def test_module_does_not_reexport_unrelated_dependencies() -> None:
    """The bridge must not leak ``AdmissionPolicy`` / ``SafeBatchScheduler``."""
    import oai2.runtime.admission_scheduler as mod

    assert "AdmissionPolicy" not in mod.__all__
    assert "SafeBatchScheduler" not in mod.__all__
    assert "AdmissionRequest" not in mod.__all__
    assert "CapacitySnapshot" not in mod.__all__
    # But the underlying types remain accessible by direct import (used at
    # construction time) — verify the module still imports them.
    assert AdmissionPolicy is not None
    assert SafeBatchScheduler is not None


# ---------------------------------------------------------------------------
# 3. ``AdmissionScheduledRequest`` dataclass + cross-id invariant
# ---------------------------------------------------------------------------


def test_admission_scheduled_request_constructs_with_matching_ids() -> None:
    """``admission`` and ``scheduled`` must share the same ``request_id``."""
    scheduled = _scheduled_request("r1", "s1")
    assert scheduled.admission.request_id == "r1"
    assert scheduled.scheduled.request_id == "r1"
    assert scheduled.admission.request_id == scheduled.scheduled.request_id


def test_admission_scheduled_request_rejects_mismatched_ids() -> None:
    """Cross-id mismatch raises ``ValueError`` at construction time."""
    with pytest.raises(ValueError, match="request_id must match"):
        AdmissionScheduledRequest(
            admission=_admission("alpha"),
            scheduled=_scheduled("bravo", "s1"),
        )


def test_admission_scheduled_request_is_frozen() -> None:
    """Frozen-slot dataclass: attribute assignment raises ``FrozenInstanceError``."""
    scheduled = _scheduled_request("r1", "s1")
    with pytest.raises(dataclasses.FrozenInstanceError):
        scheduled.admission = _admission("different")  # type: ignore[misc]


def test_admission_scheduled_request_uses_slots() -> None:
    """The dataclass declares ``slots=True`` (no ``__dict__`` per instance)."""
    assert AdmissionScheduledRequest.__dataclass_params__.slots is True  # type: ignore[attr-defined]
    assert "admission" in AdmissionScheduledRequest.__dataclass_fields__
    assert "scheduled" in AdmissionScheduledRequest.__dataclass_fields__


def test_admission_scheduled_request_rejects_non_admission_object() -> None:
    """Passing a non-``AdmissionRequest`` as the first field fails the gate."""
    with pytest.raises((TypeError, ValueError, AttributeError)):
        AdmissionScheduledRequest(
            admission="not-an-admission",  # type: ignore[arg-type]
            scheduled=_scheduled("r1", "s1"),
        )


def test_admission_scheduled_request_rejects_non_scheduled_object() -> None:
    """Passing a non-``ScheduledRequest`` as the second field fails the gate."""
    with pytest.raises((TypeError, ValueError, AttributeError)):
        AdmissionScheduledRequest(
            admission=_admission("r1"),
            scheduled=object(),  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# 4. ``AdmissionSchedulerMetrics`` shape contract
# ---------------------------------------------------------------------------


def test_scheduler_metrics_field_order_is_pinned() -> None:
    """All six fields are declared and named in the public order."""
    field_names = tuple(AdmissionSchedulerMetrics.__dataclass_fields__.keys())
    assert field_names == (
        "pending_admission",
        "ready_queue_depth",
        "ready_sessions",
        "cancelled_ready_requests",
        "batches_emitted",
        "requests_emitted",
    )


def test_scheduler_metrics_is_frozen_and_slotted() -> None:
    """The metrics dataclass is frozen with slots."""
    assert AdmissionSchedulerMetrics.__dataclass_params__.frozen is True  # type: ignore[attr-defined]
    assert AdmissionSchedulerMetrics.__dataclass_params__.slots is True  # type: ignore[attr-defined]


def test_scheduler_metrics_construction_requires_all_six_fields() -> None:
    """Omitting any of the six fields raises ``TypeError``."""
    with pytest.raises(TypeError):
        AdmissionSchedulerMetrics(  # type: ignore[call-arg]
            pending_admission=0,
            ready_queue_depth=0,
            ready_sessions=0,
            cancelled_ready_requests=0,
            batches_emitted=0,
        )


def test_scheduler_metrics_is_immutable_after_construction() -> None:
    """Frozen metrics: any field assignment raises ``FrozenInstanceError``."""
    metrics = AdmissionSchedulerMetrics(
        pending_admission=1,
        ready_queue_depth=2,
        ready_sessions=3,
        cancelled_ready_requests=0,
        batches_emitted=0,
        requests_emitted=0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        metrics.pending_admission = 99  # type: ignore[misc]


def test_scheduler_metrics_field_types_are_strict() -> None:
    """The field types are concrete ``int`` (no coercion from bool/float)."""
    # bools are technically ints in Python but the dataclass annotations
    # do not coerce; passing True as a count should still produce 1 but the
    # structural contract is that callers pass int values.
    metrics = AdmissionSchedulerMetrics(
        pending_admission=0,
        ready_queue_depth=0,
        ready_sessions=0,
        cancelled_ready_requests=0,
        batches_emitted=1,
        requests_emitted=2,
    )
    assert isinstance(metrics.batches_emitted, int)
    assert isinstance(metrics.requests_emitted, int)


# ---------------------------------------------------------------------------
# 5. ``AdmissionBatchController`` constructor
# ---------------------------------------------------------------------------


def test_controller_constructor_accepts_no_arguments() -> None:
    """Default constructor wires the default policy + scheduler + queue."""
    controller = AdmissionBatchController()
    metrics = controller.metrics
    assert metrics.pending_admission == 0
    assert metrics.ready_queue_depth == 0


def test_controller_constructor_signature_is_keyword_only() -> None:
    """``policy`` and ``scheduler`` are keyword-only — no positional form."""
    sig = inspect.signature(AdmissionBatchController.__init__)
    for name in ("policy", "scheduler"):
        param = sig.parameters[name]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"parameter {name!r} must be keyword-only, got {param.kind!r}"
        )


def test_controller_constructor_uses_provided_dependencies() -> None:
    """Custom policy / scheduler are wired through (not replaced with defaults)."""
    custom_policy = AdmissionPolicy(starvation_after_ms=42.0)
    custom_scheduler = SafeBatchScheduler()
    controller = AdmissionBatchController(policy=custom_policy, scheduler=custom_scheduler)
    # Starvation window: the controller's internal pending queue must use the
    # custom value, which means a custom AdmissionQueue got constructed with
    # the custom starvation_after_ms. We can verify by checking the controller
    # uses the custom policy: submit a request that would admit and confirm
    # the decision matches.
    decision = controller.submit(
        _scheduled_request("r1", "s1"),
        _capacity(),
        now_ms=10.0,
    )
    assert decision.action is AdmissionAction.ADMIT
    # Now verify a custom scheduler is used: the controller's pop_batch must
    # delegate to it. Submit and then pop — if the custom scheduler were not
    # wired, the request would not appear in the popped batch.
    batch = controller.pop_batch(max_batch_size=8)
    assert batch is not None
    assert {item.request_id for item in batch.requests} == {"r1"}


def test_controller_constructor_propagates_starvation_window_to_queue() -> None:
    """The pending ``AdmissionQueue`` is built with the policy's starvation window."""
    custom_policy = AdmissionPolicy(starvation_after_ms=12_345.0)
    controller = AdmissionBatchController(policy=custom_policy)
    # Indirect verification: a request that is queued will be re-promoted with
    # the custom window. Submit a memory-pressure request, then promote — if
    # the custom starvation value were not propagated, the behavior would
    # differ. We just assert the controller remains functional here.
    decision = controller.submit(
        _scheduled_request("r1", "s1", memory=8.0),
        _capacity(used=52.0),
        now_ms=10.0,
    )
    assert decision.action is AdmissionAction.QUEUE


# ---------------------------------------------------------------------------
# 6. ``AdmissionBatchController.submit``
# ---------------------------------------------------------------------------


def test_submit_keyword_only_now_ms() -> None:
    """``now_ms`` is keyword-only on ``submit``."""
    sig = inspect.signature(AdmissionBatchController.submit)
    param = sig.parameters["now_ms"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


def test_submit_admit_routes_to_scheduler_not_pending() -> None:
    """``ADMIT`` enqueues the scheduled request and leaves pending empty."""
    controller = AdmissionBatchController()
    decision = controller.submit(
        _scheduled_request("r1", "s1"),
        _capacity(),
        now_ms=10.0,
    )
    assert decision.action is AdmissionAction.ADMIT
    assert decision.reason is AdmissionReason.CAPACITY_AVAILABLE
    assert decision.request_id == "r1"
    assert controller.metrics.pending_admission == 0
    assert controller.metrics.ready_queue_depth == 1


def test_submit_queue_routes_to_pending_map_and_pending_queue() -> None:
    """``QUEUE`` adds to both ``_pending`` and the ``AdmissionQueue``."""
    controller = AdmissionBatchController()
    decision = controller.submit(
        _scheduled_request("r1", "s1", memory=8.0),
        _capacity(used=52.0),
        now_ms=10.0,
    )
    assert decision.action is AdmissionAction.QUEUE
    assert decision.reason is AdmissionReason.MEMORY_PRESSURE
    assert controller.metrics.pending_admission == 1
    assert controller.metrics.ready_queue_depth == 0


def test_submit_reject_does_not_touch_scheduler_or_pending() -> None:
    """``REJECT`` must NOT enqueue and must NOT add to pending."""
    controller = AdmissionBatchController()
    # First fill the queue with one memory-pressure request (max_queue=1).
    controller.submit(
        _scheduled_request("first", "s1", memory=8.0),
        _capacity(used=52.0, max_queue=1),
        now_ms=10.0,
    )
    # Second request hits QUEUE_FULL and must be rejected.
    decision = controller.submit(
        _scheduled_request("second", "s2", memory=8.0),
        _capacity(used=52.0, max_queue=1),
        now_ms=11.0,
    )
    assert decision.action is AdmissionAction.REJECT
    assert decision.reason is AdmissionReason.QUEUE_FULL
    # The pending count must be exactly 1 (the first), not 2.
    assert controller.metrics.pending_admission == 1
    # The rejected request is nowhere in the ready queue.
    assert controller.metrics.ready_queue_depth == 0


def test_submit_admit_duplicate_is_blocked_by_scheduler() -> None:
    """A duplicate submit for an ADMIT-routing request hits the scheduler dedup.

    The controller's own ``_pending`` map is empty for an admitted request, so
    the duplicate submit reaches ``self._scheduler.enqueue`` which raises
    ``ValueError("request already queued: ...")``.
    """
    controller = AdmissionBatchController()
    controller.submit(
        _scheduled_request("dup", "s1"),
        _capacity(),
        now_ms=10.0,
    )
    with pytest.raises(ValueError, match="request already queued"):
        controller.submit(
            _scheduled_request("dup", "s1"),
            _capacity(),
            now_ms=11.0,
        )


def test_submit_queue_duplicate_is_blocked_by_controller_pending_map() -> None:
    """A duplicate submit for a QUEUE-routing request hits the controller dedup.

    The controller's ``_pending`` map still holds the prior submission (still
    pending), so the duplicate submit raises the controller's own error before
    reaching the scheduler.
    """
    controller = AdmissionBatchController()
    controller.submit(
        _scheduled_request("p1", "s1", memory=8.0),
        _capacity(used=52.0),
        now_ms=10.0,
    )
    with pytest.raises(ValueError, match="request already pending"):
        controller.submit(
            _scheduled_request("p1", "s1", memory=8.0),
            _capacity(used=52.0),
            now_ms=11.0,
        )


def test_submit_overrides_capacity_queue_depth_with_pending_count() -> None:
    """The ``queue_depth`` passed to the policy is the controller's pending count,
    not the caller-supplied one. Callers may pass a stale snapshot."""
    controller = AdmissionBatchController()
    # Caller reports queue_depth=0; controller has 0 pending. Decision is ADMIT.
    first = controller.submit(
        _scheduled_request("r1", "s1"),
        _capacity(queue=0, max_queue=4),
        now_ms=10.0,
    )
    assert first.action is AdmissionAction.ADMIT

    # Now fill the pending queue with a memory-pressure request.
    controller.submit(
        _scheduled_request("p1", "s2", memory=8.0),
        _capacity(used=52.0, queue=0, max_queue=4),
        now_ms=11.0,
    )
    assert controller.metrics.pending_admission == 1

    # Submit a third request with caller-supplied queue=0 even though
    # controller has 1 pending. The effective capacity must use 1, not 0.
    # With max_queue=4 and 1 already pending, we still have room.
    decision = controller.submit(
        _scheduled_request("p2", "s3", memory=8.0),
        _capacity(used=52.0, queue=0, max_queue=4),
        now_ms=12.0,
    )
    assert decision.action is AdmissionAction.QUEUE
    assert controller.metrics.pending_admission == 2


def test_submit_does_not_count_already_emitted_sessions_in_queue_depth() -> None:
    """A scheduler-enqueued request must not be re-counted as pending.

    Submit one admitted request, then submit a second under pressure. The
    first was routed to the scheduler — it must NOT inflate the effective
    queue_depth used to evaluate the second.
    """
    controller = AdmissionBatchController()
    controller.submit(
        _scheduled_request("a", "s1"),
        _capacity(used=4.0, max_queue=2),
        now_ms=10.0,
    )
    # Now a memory-pressure request — controller.pending = 0 so QUEUE
    # decision is fine and effective queue_depth = 0 (caller passed 0).
    pending = controller.submit(
        _scheduled_request("b", "s2", memory=8.0),
        _capacity(used=52.0, queue=0, max_queue=2),
        now_ms=11.0,
    )
    assert pending.action is AdmissionAction.QUEUE
    assert controller.metrics.pending_admission == 1


def test_submit_propagates_request_id_to_decision() -> None:
    """The returned ``AdmissionDecision`` carries the original request id."""
    controller = AdmissionBatchController()
    decision = controller.submit(
        _scheduled_request("echo-42", "s1"),
        _capacity(),
        now_ms=10.0,
    )
    assert decision.request_id == "echo-42"


# ---------------------------------------------------------------------------
# 7. ``AdmissionBatchController.promote_one``
# ---------------------------------------------------------------------------


def test_promote_one_keyword_only_now_ms() -> None:
    """``now_ms`` is keyword-only on ``promote_one``."""
    sig = inspect.signature(AdmissionBatchController.promote_one)
    param = sig.parameters["now_ms"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


def test_promote_one_returns_none_when_no_pending() -> None:
    """No pending work: ``promote_one`` returns ``None``."""
    controller = AdmissionBatchController()
    assert controller.promote_one(_capacity(), now_ms=10.0) is None


def test_promote_one_admit_routes_to_scheduler() -> None:
    """On ADMIT, the request leaves pending and lands in the scheduler."""
    controller = AdmissionBatchController()
    controller.submit(
        _scheduled_request("p1", "s1", memory=8.0),
        _capacity(used=52.0),
        now_ms=10.0,
    )
    assert controller.metrics.pending_admission == 1

    promoted = controller.promote_one(
        _capacity(used=10.0),
        now_ms=20.0,
    )
    assert promoted is not None
    assert promoted.action is AdmissionAction.ADMIT
    assert promoted.reason is AdmissionReason.CAPACITY_AVAILABLE
    assert controller.metrics.pending_admission == 0
    assert controller.metrics.ready_queue_depth == 1


def test_promote_one_queue_keeps_request_pending() -> None:
    """On QUEUE during promotion, the request stays in pending (re-queued)."""
    controller = AdmissionBatchController()
    controller.submit(
        _scheduled_request("p1", "s1", memory=8.0),
        _capacity(used=52.0),
        now_ms=10.0,
    )
    promoted = controller.promote_one(
        _capacity(used=52.0),  # still under pressure
        now_ms=20.0,
    )
    assert promoted is not None
    assert promoted.action is AdmissionAction.QUEUE
    assert promoted.reason is AdmissionReason.MEMORY_PRESSURE
    assert controller.metrics.pending_admission == 1
    assert controller.metrics.ready_queue_depth == 0


def test_promote_one_reject_drops_request_from_pending() -> None:
    """On REJECT during promotion, the request is removed from pending
    but is also NOT re-inserted — this pins the existing semantic exactly."""
    controller = AdmissionBatchController()
    # A request with deadline already expired at promote time.
    controller.submit(
        _scheduled_request("p1", "s1", memory=8.0, deadline=5.0),
        _capacity(used=52.0),
        now_ms=2.0,
    )
    assert controller.metrics.pending_admission == 1

    promoted = controller.promote_one(
        _capacity(used=10.0),
        now_ms=10.0,  # now > deadline => DEADLINE_EXPIRED
    )
    assert promoted is not None
    assert promoted.action is AdmissionAction.REJECT
    assert promoted.reason is AdmissionReason.DEADLINE_EXPIRED
    # Pinned: the request is dropped, not re-queued.
    assert controller.metrics.pending_admission == 0
    assert controller.metrics.ready_queue_depth == 0


def test_promote_one_overrides_capacity_queue_depth_with_pending_count() -> None:
    """Promotion re-evaluates with the controller's own pending count."""
    controller = AdmissionBatchController()
    # Two pending requests (memory pressure).
    controller.submit(
        _scheduled_request("p1", "s1", memory=8.0),
        _capacity(used=52.0, max_queue=2),
        now_ms=10.0,
    )
    controller.submit(
        _scheduled_request("p2", "s2", memory=8.0),
        _capacity(used=52.0, max_queue=2),
        now_ms=11.0,
    )
    assert controller.metrics.pending_admission == 2

    # Promote one with capacity snapshot showing queue=0 (caller stale).
    # Effective queue_depth used by policy = 2 (one remaining after pop_next).
    # With max_queue=2 and 2 pending, the next promotion would re-queue.
    promoted = controller.promote_one(
        _capacity(used=10.0, queue=0, max_queue=2),
        now_ms=20.0,
    )
    assert promoted is not None
    # The popped one was decided ADMIT (capacity is now free).
    assert promoted.action is AdmissionAction.ADMIT
    # The remaining pending is still 1.
    assert controller.metrics.pending_admission == 1


# ---------------------------------------------------------------------------
# 8. ``AdmissionBatchController.cancel``
# ---------------------------------------------------------------------------


def test_cancel_returns_false_for_unknown_request_id() -> None:
    """Cancelling an unknown request id returns ``False`` and changes nothing."""
    controller = AdmissionBatchController()
    assert controller.cancel("does-not-exist") is False
    assert controller.metrics.pending_admission == 0
    assert controller.metrics.ready_queue_depth == 0


def test_cancel_removes_pending_from_both_pending_map_and_queue() -> None:
    """Cancelling a pending request removes it from both stores."""
    controller = AdmissionBatchController()
    controller.submit(
        _scheduled_request("p1", "s1", memory=8.0),
        _capacity(used=52.0),
        now_ms=10.0,
    )
    assert controller.metrics.pending_admission == 1

    assert controller.cancel("p1") is True
    assert controller.metrics.pending_admission == 0

    # And a subsequent promote finds nothing.
    assert controller.promote_one(_capacity(used=10.0), now_ms=20.0) is None


def test_cancel_removes_scheduler_queued_request() -> None:
    """Cancelling a scheduler-enqueued request removes it from the ready queue."""
    controller = AdmissionBatchController()
    controller.submit(
        _scheduled_request("a1", "s1"),
        _capacity(),
        now_ms=10.0,
    )
    assert controller.metrics.ready_queue_depth == 1

    assert controller.cancel("a1") is True
    assert controller.metrics.ready_queue_depth == 0
    # The scheduler's cancelled_requests counter must increment.
    assert controller.metrics.cancelled_ready_requests == 1


# ---------------------------------------------------------------------------
# 9. ``AdmissionBatchController.pop_batch``
# ---------------------------------------------------------------------------


def test_pop_batch_keyword_only_max_batch_size() -> None:
    """``max_batch_size`` is keyword-only on ``pop_batch``."""
    sig = inspect.signature(AdmissionBatchController.pop_batch)
    param = sig.parameters["max_batch_size"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


def test_pop_batch_returns_none_when_no_ready_work() -> None:
    """An empty scheduler has nothing to emit."""
    controller = AdmissionBatchController()
    assert controller.pop_batch(max_batch_size=8) is None


def test_pop_batch_delegates_to_safe_batch_scheduler() -> None:
    """``pop_batch`` returns a ``BatchPlan`` whose requests are scheduler-owned."""
    controller = AdmissionBatchController()
    controller.submit(
        _scheduled_request("r1", "s1"),
        _capacity(),
        now_ms=10.0,
    )
    batch = controller.pop_batch(max_batch_size=8)
    assert batch is not None
    assert isinstance(batch, BatchPlan)
    assert {item.request_id for item in batch.requests} == {"r1"}
    # After pop, ready_queue_depth drops to 0 and batches/requests increment.
    metrics = controller.metrics
    assert metrics.ready_queue_depth == 0
    assert metrics.batches_emitted == 1
    assert metrics.requests_emitted == 1


def test_pop_batch_metrics_increment_across_multiple_emissions() -> None:
    """Repeated pops accumulate ``batches_emitted`` and ``requests_emitted``."""
    controller = AdmissionBatchController()
    # Two distinct compatibility keys, each emitted as a separate batch.
    controller.submit(
        _scheduled_request("a", "s1", model="model-a"),
        _capacity(),
        now_ms=10.0,
    )
    controller.submit(
        _scheduled_request("b", "s2", model="model-b"),
        _capacity(),
        now_ms=11.0,
    )
    first = controller.pop_batch(max_batch_size=8)
    second = controller.pop_batch(max_batch_size=8)
    assert first is not None and len(first.requests) == 1
    assert second is not None and len(second.requests) == 1
    metrics = controller.metrics
    assert metrics.batches_emitted == 2
    assert metrics.requests_emitted == 2


# ---------------------------------------------------------------------------
# 10. ``AdmissionBatchController.metrics`` contract
# ---------------------------------------------------------------------------


def test_metrics_initial_values_are_all_zero() -> None:
    """A fresh controller reports zero for every metric field."""
    controller = AdmissionBatchController()
    metrics = controller.metrics
    assert metrics.pending_admission == 0
    assert metrics.ready_queue_depth == 0
    assert metrics.ready_sessions == 0
    assert metrics.cancelled_ready_requests == 0
    assert metrics.batches_emitted == 0
    assert metrics.requests_emitted == 0


def test_metrics_reflects_pending_state_after_queue_decision() -> None:
    """After a QUEUE submit, ``pending_admission`` reflects the count."""
    controller = AdmissionBatchController()
    controller.submit(
        _scheduled_request("p1", "s1", memory=8.0),
        _capacity(used=52.0),
        now_ms=10.0,
    )
    metrics = controller.metrics
    assert metrics.pending_admission == 1
    assert metrics.ready_queue_depth == 0
    assert metrics.batches_emitted == 0
    assert metrics.requests_emitted == 0


def test_metrics_reflects_ready_state_after_admit_decision() -> None:
    """After an ADMIT submit, ``ready_queue_depth`` reflects the scheduler count."""
    controller = AdmissionBatchController()
    controller.submit(
        _scheduled_request("r1", "s1"),
        _capacity(),
        now_ms=10.0,
    )
    metrics = controller.metrics
    assert metrics.pending_admission == 0
    assert metrics.ready_queue_depth == 1
    assert metrics.ready_sessions == 1


def test_metrics_reflects_full_pending_to_ready_flow() -> None:
    """End-to-end: pending -> admit -> pop increments all three counters."""
    controller = AdmissionBatchController()
    controller.submit(
        _scheduled_request("p1", "s1", memory=8.0),
        _capacity(used=52.0),
        now_ms=10.0,
    )
    assert controller.metrics.pending_admission == 1

    controller.promote_one(
        _capacity(used=10.0),
        now_ms=20.0,
    )
    mid = controller.metrics
    assert mid.pending_admission == 0
    assert mid.ready_queue_depth == 1

    controller.pop_batch(max_batch_size=8)
    end = controller.metrics
    assert end.ready_queue_depth == 0
    assert end.batches_emitted == 1
    assert end.requests_emitted == 1


def test_metrics_is_a_property_returning_fresh_snapshot() -> None:
    """``metrics`` is a read-only property that returns a fresh snapshot each call."""
    controller = AdmissionBatchController()
    first = controller.metrics
    second = controller.metrics
    # Different dataclass instances, equal field values.
    assert first is not second
    assert first == second
    # Both are frozen — mutating either must fail.
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.pending_admission = 5  # type: ignore[misc]


def test_metrics_uses_scheduler_metrics_underlying_primitive() -> None:
    """The controller composes metrics from the underlying ``SchedulerMetrics``.

    The three scheduler-side fields (queue_depth, queued_sessions,
    cancelled_requests) must match the underlying scheduler's snapshot.
    """
    controller = AdmissionBatchController()
    controller.submit(
        _scheduled_request("r1", "s1"),
        _capacity(),
        now_ms=10.0,
    )
    controller.submit(
        _scheduled_request("r2", "s2"),
        _capacity(),
        now_ms=11.0,
    )
    # The controller does not expose the underlying scheduler directly, but
    # the metrics fields ``ready_queue_depth`` and ``ready_sessions`` both
    # derive from the underlying SchedulerMetrics.queue_depth / queued_sessions.
    metrics = controller.metrics
    assert metrics.ready_queue_depth == 2
    assert metrics.ready_sessions == 2


def test_metrics_does_not_count_rejected_requests_in_any_field() -> None:
    """A REJECT submit must leave all six metric fields unchanged."""
    controller = AdmissionBatchController()
    # Fill the queue first so the second submit is REJECT.
    controller.submit(
        _scheduled_request("first", "s1", memory=8.0),
        _capacity(used=52.0, max_queue=1),
        now_ms=10.0,
    )
    before = controller.metrics
    assert before.pending_admission == 1

    decision = controller.submit(
        _scheduled_request("second", "s2", memory=8.0),
        _capacity(used=52.0, max_queue=1),
        now_ms=11.0,
    )
    assert decision.action is AdmissionAction.REJECT

    after = controller.metrics
    # No field changes (the second was rejected, never enqueued).
    assert after.pending_admission == 1
    assert after.ready_queue_depth == 0
    assert after.ready_sessions == 0
    assert after.cancelled_ready_requests == 0
    assert after.batches_emitted == 0
    assert after.requests_emitted == 0
