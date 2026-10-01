from __future__ import annotations

import pytest

from oai2.runtime.scheduler import (
    BatchPlan,
    SafeBatchScheduler,
    ScheduledRequest,
    SessionCompatibilityKey,
)


def _key(
    *,
    model: str = "model-a",
    tokenizer: str = "tok-v1",
    prefix: str = "prefix-a",
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
    session_id: str,
    *,
    key: SessionCompatibilityKey | None = None,
    at: float = 0.0,
) -> ScheduledRequest:
    return ScheduledRequest(
        request_id=request_id,
        session_id=session_id,
        compatibility=key or _key(),
        enqueued_at_ms=at,
    )


def test_exactly_compatible_different_sessions_can_batch() -> None:
    scheduler = SafeBatchScheduler()
    scheduler.enqueue(_req("r1", "s1", at=1.0))
    scheduler.enqueue(_req("r2", "s2", at=2.0))

    batch = scheduler.pop_batch(max_batch_size=8)

    assert isinstance(batch, BatchPlan)
    assert [r.request_id for r in batch.requests] == ["r1", "r2"]
    assert scheduler.metrics.queue_depth == 0
    assert scheduler.metrics.batches_emitted == 1
    assert scheduler.metrics.requests_emitted == 2


@pytest.mark.parametrize(
    "other",
    [
        _key(model="model-b"),
        _key(tokenizer="tok-v2"),
        _key(prefix="prefix-b"),
        _key(tools="tools-v2"),
        _key(world="world-v2"),
        _key(security="user-b"),
    ],
)
def test_any_context_mismatch_prevents_batching(
    other: SessionCompatibilityKey,
) -> None:
    scheduler = SafeBatchScheduler()
    scheduler.enqueue(_req("r1", "s1", at=1.0))
    scheduler.enqueue(_req("r2", "s2", key=other, at=2.0))

    first = scheduler.pop_batch(max_batch_size=8)
    second = scheduler.pop_batch(max_batch_size=8)

    assert first is not None and len(first.requests) == 1
    assert second is not None and len(second.requests) == 1
    assert {first.requests[0].request_id, second.requests[0].request_id} == {
        "r1",
        "r2",
    }


def test_one_session_cannot_have_two_queued_requests() -> None:
    scheduler = SafeBatchScheduler()
    scheduler.enqueue(_req("r1", "same"))

    with pytest.raises(ValueError, match="session already has queued work"):
        scheduler.enqueue(_req("r2", "same"))


def test_cancellation_isolated_to_target_request_and_session() -> None:
    scheduler = SafeBatchScheduler()
    scheduler.enqueue(_req("r1", "s1"))
    scheduler.enqueue(_req("r2", "s2"))

    assert scheduler.cancel_session("s1") is True
    assert scheduler.cancel_request("missing") is False
    assert scheduler.metrics.cancelled_requests == 1
    assert scheduler.metrics.queue_depth == 1

    batch = scheduler.pop_batch(max_batch_size=8)
    assert batch is not None
    assert [r.request_id for r in batch.requests] == ["r2"]


def test_batch_size_is_bounded_and_remaining_work_stays_queued() -> None:
    scheduler = SafeBatchScheduler()
    for i in range(4):
        scheduler.enqueue(_req(f"r{i}", f"s{i}", at=float(i)))

    batch = scheduler.pop_batch(max_batch_size=2)

    assert batch is not None
    assert len(batch.requests) == 2
    assert scheduler.metrics.queue_depth == 2
    assert scheduler.metrics.queued_sessions == 2


def test_oldest_compatible_group_is_selected_first() -> None:
    scheduler = SafeBatchScheduler()
    older = _key(model="older")
    newer = _key(model="newer")
    scheduler.enqueue(_req("new", "s-new", key=newer, at=20.0))
    scheduler.enqueue(_req("old1", "s-old1", key=older, at=10.0))
    scheduler.enqueue(_req("old2", "s-old2", key=older, at=11.0))

    batch = scheduler.pop_batch(max_batch_size=8)

    assert batch is not None
    assert batch.compatibility == older
    assert [r.request_id for r in batch.requests] == ["old1", "old2"]


def test_batch_plan_rejects_duplicate_session_or_mismatch() -> None:
    key = _key()
    with pytest.raises(ValueError, match="multiple requests from one session"):
        BatchPlan(
            compatibility=key,
            requests=(
                _req("r1", "same", key=key),
                _req("r2", "same", key=key),
            ),
        )

    with pytest.raises(ValueError, match="incompatible"):
        BatchPlan(
            compatibility=key,
            requests=(_req("r1", "s1", key=_key(model="other")),),
        )



def test_scheduler_symbols_are_exported_from_runtime_package() -> None:
    from oai2.runtime import SafeBatchScheduler as ExportedScheduler
    from oai2.runtime import SessionCompatibilityKey as ExportedKey
    from oai2.runtime.scheduler import SafeBatchScheduler, SessionCompatibilityKey

    assert ExportedScheduler is SafeBatchScheduler
    assert ExportedKey is SessionCompatibilityKey
