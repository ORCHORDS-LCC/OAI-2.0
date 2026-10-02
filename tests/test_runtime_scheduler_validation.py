"""Validation tests for ``oai2.runtime.scheduler``.

This file pins the outer guard rails of the WI-INF-001 deterministic
session-isolation and safe-batching scheduler. It complements the
integration tests in ``tests/test_scheduler.py`` by asserting the
structural contract: dataclass validation matrices, frozen-slot
semantics, keyword-only signatures, dedup invariants, the per-field
validation order in ``__post_init__`` hooks, the empty-batches
boundary, and the precise selection ordering (enqueued_at_ms → request_id
tiebreak, session-isolation, exact-compatibility grouping).

All time-sensitive selection is exercised via explicit ``enqueued_at_ms``
values rather than wall-clock calls.
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest

from oai2.runtime.scheduler import (
    BatchPlan,
    SafeBatchScheduler,
    ScheduledRequest,
    SchedulerMetrics,
    SessionCompatibilityKey,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _key(
    *,
    model: str = "model-a",
    tokenizer: str = "tok-v1",
    prefix: str = "prefix-v1",
    tools: str = "tools-v1",
    world: str = "world-v1",
    security: str = "user-a",
) -> SessionCompatibilityKey:
    return SessionCompatibilityKey(
        model_id=model,
        tokenizer_version=tokenizer,
        prefix_digest=prefix,
        tool_schema_version=tools,
        world_state_version=world,
        security_context=security,
    )


def _req(
    request_id: str,
    session_id: str = "s1",
    *,
    key: SessionCompatibilityKey | None = None,
    enqueued: float = 0.0,
) -> ScheduledRequest:
    return ScheduledRequest(
        request_id=request_id,
        session_id=session_id,
        compatibility=key or _key(),
        enqueued_at_ms=enqueued,
    )


def _plan(
    request_ids: tuple[str, ...],
    *,
    session_prefix: str = "s",
    enqueued_start: float = 0.0,
    key: SessionCompatibilityKey | None = None,
) -> BatchPlan:
    """Build a BatchPlan from request_ids, each with a distinct session id."""
    requests = tuple(
        _req(
            rid,
            f"{session_prefix}{i}",
            key=key,
            enqueued=enqueued_start + float(i),
        )
        for i, rid in enumerate(request_ids)
    )
    return BatchPlan(compatibility=key or _key(), requests=requests)


# ---------------------------------------------------------------------------
# 1. Module docstring + import discipline
# ---------------------------------------------------------------------------


def test_module_docstring_is_present_and_substantive() -> None:
    """The module must carry a docstring stating the correctness boundary."""
    import oai2.runtime.scheduler as mod

    assert mod.__doc__ is not None
    body = mod.__doc__.strip().lower()
    # Must reference the WI-INF-001 boundary (or at least the
    # session-isolation / safe-batching scope).
    assert "wi-inf" in body or "session" in body
    assert "batch" in body or "batching" in body or "scheduler" in body


def test_module_does_not_hardcode_credentials() -> None:
    """No hardcoded API keys, tokens, or bearer strings in source."""
    import oai2.runtime.scheduler as mod

    source = inspect.getsource(mod)
    import re

    # Public-safety linter regex: \b(?:sk|xoxb)-[A-Za-z0-9_-]{16,}\b
    assert not re.search(r"\b(?:sk|xoxb)-[A-Za-z0-9_-]{16,}\b", source)
    # Public-safety linter regex: \bapi[_-]?key\s*[:=]\s*["\'][^"\']{16,}["\']
    assert not re.search(r"\bapi[_-]?key\s*[:=]\s*[\"'][^\"']{16,}[\"']", source)


def test_module_does_not_import_unrelated_cloud_runtimes() -> None:
    """No cloud-runtime imports leak into the deterministic scheduler."""
    import oai2.runtime.scheduler as mod

    source = inspect.getsource(mod)
    forbidden = (
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
        "import httpx",  # scheduler is HTTP-free
        "from httpx",
    )
    for needle in forbidden:
        assert needle not in source, f"unexpected import: {needle!r}"


def test_module_uses_future_annotations_for_up006_compliance() -> None:
    """``from __future__ import annotations`` is required for modern typing."""
    import oai2.runtime.scheduler as mod

    source = inspect.getsource(mod)
    assert "from __future__ import annotations" in source


# ---------------------------------------------------------------------------
# 2. ``__all__`` completeness + package-level re-export
# ---------------------------------------------------------------------------


def test_module_all_lists_exactly_five_public_names() -> None:
    """The scheduler exports exactly five public names."""
    import oai2.runtime.scheduler as mod

    assert set(mod.__all__) == {
        "SessionCompatibilityKey",
        "ScheduledRequest",
        "SchedulerMetrics",
        "BatchPlan",
        "SafeBatchScheduler",
    }


def test_package_level_reexport_preserves_identity() -> None:
    """All five names re-export from ``oai2.runtime`` are the same objects."""
    from oai2.runtime import BatchPlan as ExportedBatchPlan
    from oai2.runtime import SafeBatchScheduler as ExportedScheduler
    from oai2.runtime import ScheduledRequest as ExportedRequest
    from oai2.runtime import SchedulerMetrics as ExportedMetrics
    from oai2.runtime import SessionCompatibilityKey as ExportedKey

    assert ExportedBatchPlan is BatchPlan
    assert ExportedScheduler is SafeBatchScheduler
    assert ExportedRequest is ScheduledRequest
    assert ExportedMetrics is SchedulerMetrics
    assert ExportedKey is SessionCompatibilityKey


# ---------------------------------------------------------------------------
# 3. ``SessionCompatibilityKey`` validation matrix
# ---------------------------------------------------------------------------


def test_session_compatibility_key_field_order_is_pinned() -> None:
    """The six fields are declared and named in the public order."""
    field_names = tuple(SessionCompatibilityKey.__dataclass_fields__.keys())
    assert field_names == (
        "model_id",
        "tokenizer_version",
        "prefix_digest",
        "tool_schema_version",
        "world_state_version",
        "security_context",
    )


def test_session_compatibility_key_is_frozen() -> None:
    """Frozen dataclass: attribute assignment raises ``FrozenInstanceError``."""
    key = _key()
    with pytest.raises(dataclasses.FrozenInstanceError):
        key.model_id = "different"  # type: ignore[misc]


def test_session_compatibility_key_rejects_empty_string_per_field() -> None:
    """Each of the six fields must be non-empty; ``""`` is rejected."""
    for field in (
        "model_id",
        "tokenizer_version",
        "prefix_digest",
        "tool_schema_version",
        "world_state_version",
        "security_context",
    ):
        with pytest.raises(ValueError, match=field):
            dataclasses.replace(_key(), **{field: ""})


def test_session_compatibility_key_rejects_whitespace_padded_string_per_field() -> None:
    """Each field must be already-normalized; padded values are rejected."""
    for field in (
        "model_id",
        "tokenizer_version",
        "prefix_digest",
        "tool_schema_version",
        "world_state_version",
        "security_context",
    ):
        with pytest.raises(ValueError, match=field):
            dataclasses.replace(_key(), **{field: f"  padded-{field}  "})


def test_session_compatibility_key_rejects_non_string_per_field() -> None:
    """Each field must be a string; non-strings are rejected."""
    for field in (
        "model_id",
        "tokenizer_version",
        "prefix_digest",
        "tool_schema_version",
        "world_state_version",
        "security_context",
    ):
        with pytest.raises((TypeError, ValueError)):
            dataclasses.replace(_key(), **{field: 42})  # type: ignore[arg-type]


def test_session_compatibility_key_value_equality_and_hash() -> None:
    """Two equivalent keys are equal and hash equal (frozen dataclass)."""
    a = _key()
    b = _key()
    assert a == b
    assert hash(a) == hash(b)


def test_session_compatibility_key_distinguishes_every_field() -> None:
    """Changing any single field makes the keys unequal."""
    field_changes = {
        "model_id": "model-b",
        "tokenizer_version": "tok-v2",
        "prefix_digest": "prefix-b",
        "tool_schema_version": "tools-v2",
        "world_state_version": "world-v2",
        "security_context": "user-b",
    }
    for field, new_value in field_changes.items():
        other = dataclasses.replace(_key(), **{field: new_value})
        assert _key() != other, f"changing {field!r} must change equality"


# ---------------------------------------------------------------------------
# 4. ``ScheduledRequest`` validation matrix
# ---------------------------------------------------------------------------


def test_scheduled_request_field_order_is_pinned() -> None:
    """The four fields are declared and named in the public order."""
    field_names = tuple(ScheduledRequest.__dataclass_fields__.keys())
    assert field_names == (
        "request_id",
        "session_id",
        "compatibility",
        "enqueued_at_ms",
    )


def test_scheduled_request_is_frozen() -> None:
    """Frozen dataclass: attribute assignment raises ``FrozenInstanceError``."""
    req = _req("r1", "s1")
    with pytest.raises(dataclasses.FrozenInstanceError):
        req.request_id = "different"  # type: ignore[misc]


def test_scheduled_request_rejects_non_string_request_id() -> None:
    """``request_id`` must be a non-empty normalized string."""
    with pytest.raises(ValueError, match="request_id"):
        ScheduledRequest(
            request_id=42,  # type: ignore[arg-type]
            session_id="s1",
            compatibility=_key(),
            enqueued_at_ms=0.0,
        )


def test_scheduled_request_rejects_non_string_session_id() -> None:
    """``session_id`` must be a non-empty normalized string."""
    with pytest.raises(ValueError, match="session_id"):
        ScheduledRequest(
            request_id="r1",
            session_id="   ",  # whitespace-only
            compatibility=_key(),
            enqueued_at_ms=0.0,
        )


def test_scheduled_request_rejects_non_finite_enqueued_at_ms() -> None:
    """``enqueued_at_ms`` must be a finite non-negative number."""
    for bad in (
        -1.0,
        float("nan"),
        float("inf"),
        -float("inf"),
    ):
        with pytest.raises(ValueError, match="finite non-negative"):
            ScheduledRequest(
                request_id=f"r-{bad}",
                session_id="s1",
                compatibility=_key(),
                enqueued_at_ms=bad,
            )


def test_scheduled_request_rejects_bool_enqueued_at_ms() -> None:
    """``enqueued_at_ms`` must reject ``True`` (bool-first guard)."""
    with pytest.raises(ValueError, match="finite non-negative"):
        ScheduledRequest(
            request_id="r1",
            session_id="s1",
            compatibility=_key(),
            enqueued_at_ms=True,
        )


def test_scheduled_request_does_not_strict_type_check_compatibility() -> None:
    """``compatibility`` is duck-typed at construction; only string fields and
    ``enqueued_at_ms`` are validated. Downstream ``BatchPlan.__post_init__``
    is the gate that enforces the actual compatibility contract.
    """
    # An object that is NOT a SessionCompatibilityKey still constructs.
    sentinel = object()
    request = ScheduledRequest(
        request_id="r1",
        session_id="s1",
        compatibility=sentinel,  # type: ignore[arg-type]
        enqueued_at_ms=0.0,
    )
    assert request.compatibility is sentinel


# ---------------------------------------------------------------------------
# 5. ``SchedulerMetrics`` shape contract
# ---------------------------------------------------------------------------


def test_scheduler_metrics_field_order_is_pinned() -> None:
    """All five fields are declared and named in the public order."""
    field_names = tuple(SchedulerMetrics.__dataclass_fields__.keys())
    assert field_names == (
        "queue_depth",
        "queued_sessions",
        "cancelled_requests",
        "batches_emitted",
        "requests_emitted",
    )


def test_scheduler_metrics_is_frozen() -> None:
    """Frozen metrics: any field assignment raises ``FrozenInstanceError``."""
    metrics = SchedulerMetrics(0, 0, 0, 0, 0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        metrics.queue_depth = 99  # type: ignore[misc]


def test_scheduler_metrics_construction_requires_all_five_fields() -> None:
    """Omitting any of the five fields raises ``TypeError``."""
    with pytest.raises(TypeError):
        SchedulerMetrics(  # type: ignore[call-arg]
            queue_depth=0,
            queued_sessions=0,
            cancelled_requests=0,
            batches_emitted=0,
        )


# ---------------------------------------------------------------------------
# 6. ``BatchPlan`` validation matrix
# ---------------------------------------------------------------------------


def test_batch_plan_field_order_is_pinned() -> None:
    """The two fields are declared and named in the public order."""
    field_names = tuple(BatchPlan.__dataclass_fields__.keys())
    assert field_names == ("compatibility", "requests")


def test_batch_plan_is_frozen() -> None:
    """Frozen BatchPlan: attribute assignment raises ``FrozenInstanceError``."""
    plan = _plan(("r1",))
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.compatibility = _key()  # type: ignore[misc]


def test_batch_plan_rejects_empty_requests_tuple() -> None:
    """A batch must contain at least one request."""
    with pytest.raises(ValueError, match="at least one request"):
        BatchPlan(compatibility=_key(), requests=())


def test_batch_plan_rejects_request_with_mismatched_compatibility() -> None:
    """Every request's compatibility must match the batch's compatibility."""
    plan_compat = _key(model="batch")
    request_compat = _key(model="request")
    bad_request = ScheduledRequest(
        request_id="r1",
        session_id="s1",
        compatibility=request_compat,
        enqueued_at_ms=0.0,
    )
    with pytest.raises(ValueError, match="incompatible"):
        BatchPlan(compatibility=plan_compat, requests=(bad_request,))


def test_batch_plan_rejects_duplicate_session_across_requests() -> None:
    """No two requests in a batch may share a session id."""
    key = _key()
    with pytest.raises(ValueError, match="multiple requests from one session"):
        BatchPlan(
            compatibility=key,
            requests=(
                _req("r1", "same", key=key),
                _req("r2", "same", key=key),
            ),
        )


# ---------------------------------------------------------------------------
# 7. ``SafeBatchScheduler.enqueue``
# ---------------------------------------------------------------------------


def test_enqueue_rejects_duplicate_request_id() -> None:
    """A second enqueue of the same request_id raises ``ValueError``."""
    scheduler = SafeBatchScheduler()
    scheduler.enqueue(_req("r1", "s1"))
    with pytest.raises(ValueError, match="request already queued"):
        scheduler.enqueue(_req("r1", "s2"))


def test_enqueue_rejects_duplicate_session_id() -> None:
    """A second enqueue from the same session raises ``ValueError``.

    Even if the request_id is different, the session cannot have two
    queued requests concurrently.
    """
    scheduler = SafeBatchScheduler()
    scheduler.enqueue(_req("r1", "same"))
    with pytest.raises(ValueError, match="session already has queued work"):
        scheduler.enqueue(_req("r2", "same"))


def test_enqueue_initial_metrics_are_zero() -> None:
    """A fresh scheduler reports zero for all metric fields."""
    scheduler = SafeBatchScheduler()
    metrics = scheduler.metrics
    assert metrics.queue_depth == 0
    assert metrics.queued_sessions == 0
    assert metrics.cancelled_requests == 0
    assert metrics.batches_emitted == 0
    assert metrics.requests_emitted == 0


def test_enqueue_increments_queue_depth_and_queued_sessions() -> None:
    """After enqueue, ``queue_depth`` and ``queued_sessions`` reflect the count."""
    scheduler = SafeBatchScheduler()
    scheduler.enqueue(_req("r1", "s1"))
    scheduler.enqueue(_req("r2", "s2"))
    metrics = scheduler.metrics
    assert metrics.queue_depth == 2
    assert metrics.queued_sessions == 2
    assert metrics.cancelled_requests == 0
    assert metrics.batches_emitted == 0
    assert metrics.requests_emitted == 0


# ---------------------------------------------------------------------------
# 8. ``SafeBatchScheduler.cancel_request``
# ---------------------------------------------------------------------------


def test_cancel_request_returns_false_for_unknown_request_id() -> None:
    """Cancelling an unknown request id returns ``False`` and changes nothing."""
    scheduler = SafeBatchScheduler()
    assert scheduler.cancel_request("does-not-exist") is False
    assert scheduler.metrics.cancelled_requests == 0


def test_cancel_request_removes_known_request_and_increments_counter() -> None:
    """Cancelling a queued request returns ``True`` and increments the counter."""
    scheduler = SafeBatchScheduler()
    scheduler.enqueue(_req("r1", "s1"))
    assert scheduler.cancel_request("r1") is True
    assert scheduler.metrics.cancelled_requests == 1
    assert scheduler.metrics.queue_depth == 0
    assert scheduler.metrics.queued_sessions == 0


# ---------------------------------------------------------------------------
# 9. ``SafeBatchScheduler.cancel_session``
# ---------------------------------------------------------------------------


def test_cancel_session_returns_false_for_unknown_session() -> None:
    """Cancelling an unknown session returns ``False`` and changes nothing."""
    scheduler = SafeBatchScheduler()
    assert scheduler.cancel_session("does-not-exist") is False
    assert scheduler.metrics.cancelled_requests == 0


def test_cancel_session_chains_to_cancel_request() -> None:
    """Cancelling a session removes the associated request and increments."""
    scheduler = SafeBatchScheduler()
    scheduler.enqueue(_req("r1", "s1"))
    assert scheduler.cancel_session("s1") is True
    assert scheduler.metrics.cancelled_requests == 1
    assert scheduler.metrics.queue_depth == 0


# ---------------------------------------------------------------------------
# 10. ``SafeBatchScheduler.pop_batch``
# ---------------------------------------------------------------------------


def test_pop_batch_keyword_only_max_batch_size() -> None:
    """``max_batch_size`` is keyword-only on ``pop_batch``."""
    sig = inspect.signature(SafeBatchScheduler.pop_batch)
    param = sig.parameters["max_batch_size"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


def test_pop_batch_rejects_non_int_max_batch_size() -> None:
    """``max_batch_size`` must be an int (not float, str, bool, None)."""
    scheduler = SafeBatchScheduler()
    for bad in (1.5, "8", None):
        with pytest.raises(ValueError, match="positive integer"):
            scheduler.pop_batch(max_batch_size=bad)  # type: ignore[arg-type]


def test_pop_batch_rejects_bool_max_batch_size() -> None:
    """``max_batch_size`` must reject ``True`` (bool-first guard)."""
    scheduler = SafeBatchScheduler()
    with pytest.raises(ValueError, match="positive integer"):
        scheduler.pop_batch(max_batch_size=True)


def test_pop_batch_rejects_non_positive_max_batch_size() -> None:
    """``max_batch_size`` must be strictly positive (0 and negatives rejected)."""
    scheduler = SafeBatchScheduler()
    for bad in (0, -1, -100):
        with pytest.raises(ValueError, match="positive integer"):
            scheduler.pop_batch(max_batch_size=bad)


def test_pop_batch_returns_none_when_empty() -> None:
    """An empty scheduler has nothing to emit."""
    scheduler = SafeBatchScheduler()
    assert scheduler.pop_batch(max_batch_size=8) is None
    assert scheduler.metrics.batches_emitted == 0
    assert scheduler.metrics.requests_emitted == 0


def test_pop_batch_selects_oldest_compatible_group_first() -> None:
    """Among the oldest enqueued request's compatibility key, group everything
    sharing that key in the emitted batch."""
    scheduler = SafeBatchScheduler()
    older = _key(model="older")
    newer = _key(model="newer")
    # Newer key enqueued FIRST; older key enqueued SECOND. Selection should
    # still pick the older (earliest enqueued) request and group all matching.
    scheduler.enqueue(_req("new", "s-new", key=newer, enqueued=20.0))
    scheduler.enqueue(_req("old1", "s-old1", key=older, enqueued=10.0))
    scheduler.enqueue(_req("old2", "s-old2", key=older, enqueued=11.0))

    batch = scheduler.pop_batch(max_batch_size=8)

    assert batch is not None
    assert batch.compatibility == older
    assert [r.request_id for r in batch.requests] == ["old1", "old2"]
    # The newer-key request remains in the scheduler.
    assert scheduler.metrics.queue_depth == 1


def test_pop_batch_orders_within_batch_by_enqueued_at_ms_then_request_id() -> None:
    """Within a compatibility group, ordering is enqueued_at_ms asc, then request_id asc."""
    scheduler = SafeBatchScheduler()
    scheduler.enqueue(_req("zeta", "s-zeta", enqueued=5.0))
    scheduler.enqueue(_req("alpha", "s-alpha", enqueued=2.0))
    scheduler.enqueue(_req("beta", "s-beta", enqueued=2.0))  # ties on time with alpha

    batch = scheduler.pop_batch(max_batch_size=8)

    assert batch is not None
    # alpha and beta both enqueued at 2.0; request_id breaks the tie (alpha < beta).
    assert [r.request_id for r in batch.requests] == ["alpha", "beta", "zeta"]


def test_pop_batch_bounds_emit_size_by_max_batch_size() -> None:
    """A single pop_batch respects the max_batch_size cap."""
    scheduler = SafeBatchScheduler()
    for i in range(4):
        scheduler.enqueue(_req(f"r{i}", f"s{i}", enqueued=float(i)))

    batch = scheduler.pop_batch(max_batch_size=2)

    assert batch is not None
    assert len(batch.requests) == 2
    assert scheduler.metrics.queue_depth == 2
    assert scheduler.metrics.queued_sessions == 2
    assert scheduler.metrics.batches_emitted == 1
    assert scheduler.metrics.requests_emitted == 2


def test_pop_batch_drains_scheduler_after_repeated_pops() -> None:
    """Repeated pops with sufficient max_batch_size empty the scheduler."""
    scheduler = SafeBatchScheduler()
    scheduler.enqueue(_req("r1", "s1", enqueued=1.0))
    scheduler.enqueue(_req("r2", "s2", enqueued=2.0))

    first = scheduler.pop_batch(max_batch_size=8)
    second = scheduler.pop_batch(max_batch_size=8)

    assert first is not None and len(first.requests) == 2
    assert second is None
    assert scheduler.metrics.batches_emitted == 1
    assert scheduler.metrics.requests_emitted == 2
    assert scheduler.metrics.queue_depth == 0
    assert scheduler.metrics.queued_sessions == 0


def test_pop_batch_emits_compatible_sessions_together() -> None:
    """Two requests from different sessions with the same compat key batch
    together in a single pop_batch emission. (The session-isolation rule
    prevents the SAME session from appearing twice in the queue, so this
    is the only multi-request batch case to verify here.)"""
    scheduler = SafeBatchScheduler()
    key = _key()
    scheduler.enqueue(_req("a", "s-a", key=key, enqueued=1.0))
    scheduler.enqueue(_req("b", "s-b", key=key, enqueued=2.0))

    batch = scheduler.pop_batch(max_batch_size=8)

    assert batch is not None
    assert batch.compatibility == key
    assert [r.request_id for r in batch.requests] == ["a", "b"]


def test_enqueue_rejects_same_session_with_different_compat_keys() -> None:
    """The session-uniqueness invariant is enforced at enqueue time, not at
    pop_batch time. Two requests with the same session but different
    compat keys cannot coexist in the queue."""
    scheduler = SafeBatchScheduler()
    k1 = _key(model="m1")
    k2 = _key(model="m2")
    scheduler.enqueue(_req("a", "shared", key=k1, enqueued=1.0))
    with pytest.raises(ValueError, match="session already has queued work"):
        scheduler.enqueue(_req("b", "shared", key=k2, enqueued=2.0))


# ---------------------------------------------------------------------------
# 11. ``SafeBatchScheduler.metrics`` contract
# ---------------------------------------------------------------------------


def test_metrics_is_a_property_returning_fresh_snapshot() -> None:
    """``metrics`` is a read-only property that returns a fresh snapshot."""
    scheduler = SafeBatchScheduler()
    first = scheduler.metrics
    second = scheduler.metrics
    # Different dataclass instances with equal field values.
    assert first is not second
    assert first == second
    # The underlying instance is frozen.
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.queue_depth = 99  # type: ignore[misc]


def test_metrics_reflects_cancellation() -> None:
    """``cancelled_requests`` is monotonic across ``cancel_request`` calls."""
    scheduler = SafeBatchScheduler()
    scheduler.enqueue(_req("r1", "s1"))
    scheduler.enqueue(_req("r2", "s2"))
    assert scheduler.cancel_request("r1") is True
    assert scheduler.cancel_session("s2") is True
    assert scheduler.metrics.cancelled_requests == 2
    assert scheduler.metrics.queue_depth == 0
    assert scheduler.metrics.queued_sessions == 0


def test_metrics_accumulates_batches_and_requests_across_pops() -> None:
    """``batches_emitted`` and ``requests_emitted`` are monotonic across pops."""
    scheduler = SafeBatchScheduler()
    scheduler.enqueue(_req("a1", "sa1", enqueued=1.0))
    scheduler.enqueue(_req("a2", "sa2", enqueued=2.0))
    scheduler.enqueue(_req("b1", "sb1", enqueued=3.0))

    first = scheduler.pop_batch(max_batch_size=2)
    second = scheduler.pop_batch(max_batch_size=2)

    assert first is not None and len(first.requests) == 2
    assert second is not None and len(second.requests) == 1
    assert scheduler.metrics.batches_emitted == 2
    assert scheduler.metrics.requests_emitted == 3


# ---------------------------------------------------------------------------
# 12. Selection ordering details
# ---------------------------------------------------------------------------


def test_oldest_request_wins_when_compatibility_equals() -> None:
    """With identical compatibility keys, the earliest enqueued request leads."""
    scheduler = SafeBatchScheduler()
    scheduler.enqueue(_req("late", "s-late", enqueued=100.0))
    scheduler.enqueue(_req("early", "s-early", enqueued=1.0))

    batch = scheduler.pop_batch(max_batch_size=8)

    assert batch is not None
    assert [r.request_id for r in batch.requests] == ["early", "late"]


def test_request_id_breaks_tie_when_enqueued_at_ms_is_equal() -> None:
    """When enqueued_at_ms ties, the lexicographically smaller request_id wins."""
    scheduler = SafeBatchScheduler()
    scheduler.enqueue(_req("zzz", "s-zzz", enqueued=5.0))
    scheduler.enqueue(_req("aaa", "s-aaa", enqueued=5.0))
    scheduler.enqueue(_req("mmm", "s-mmm", enqueued=5.0))

    batch = scheduler.pop_batch(max_batch_size=8)

    assert batch is not None
    assert [r.request_id for r in batch.requests] == ["aaa", "mmm", "zzz"]
