"""REQ-CFOPS-013 — saturation is exposed, not silently dropped or queued.

The mechanism is an ISOLATE-LOCAL in-flight bound with a fail-fast, retryable
SATURATED refusal. These tests pin that scope claim as hard as the behaviour,
because the failure mode being guarded against here is not "the limit is wrong"
but "the limit is described as more than it is".
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from oai2.knowledge.admission import (
    IsolateAdmissionGuard,
    KnowledgeSaturatedError,
    is_retryable_saturation,
    is_retryable_saturation_error,
)
from oai2.knowledge.transport import TransportError, TransportErrorCode


def test_the_limit_is_never_exceeded() -> None:
    guard = IsolateAdmissionGuard(limit=2)
    with guard:
        with guard:
            with pytest.raises(KnowledgeSaturatedError):
                guard.try_acquire()
    assert guard.peak_in_flight == 2
    assert guard.in_flight == 0


def test_a_refusal_is_explicit_saturated_and_retryable() -> None:
    guard = IsolateAdmissionGuard(limit=1)
    with guard:
        with pytest.raises(KnowledgeSaturatedError) as caught:
            guard.try_acquire()
    exc = caught.value
    assert exc.code is TransportErrorCode.SATURATED
    assert exc.retryable is True
    assert exc.limit == 1
    assert exc.in_flight == 1
    assert is_retryable_saturation(exc.error)
    assert is_retryable_saturation_error(exc)


def test_saturated_is_distinct_from_unavailable_dependency() -> None:
    """The whole point of a separate code.

    "Full, try again shortly" and "D1 is down" need different client backoff.
    Collapsing them makes a saturated system look like an outage and an outage
    look like routine backpressure.
    """
    assert TransportErrorCode.SATURATED is not TransportErrorCode.UNAVAILABLE_DEPENDENCY
    assert TransportErrorCode.SATURATED.value == "saturated"
    saturated = TransportError(
        code=TransportErrorCode.SATURATED, message="full", retryable=True
    )
    unavailable = TransportError(
        code=TransportErrorCode.UNAVAILABLE_DEPENDENCY,
        message="down",
        retryable=True,
    )
    # Both retryable, but only one is a saturation.
    assert saturated.retryable and unavailable.retryable
    assert is_retryable_saturation(saturated)
    assert not is_retryable_saturation(unavailable)
    assert not is_retryable_saturation(
        TransportError(code=TransportErrorCode.CONFLICT, message="stale",
                       retryable=True)
    )
    assert not is_retryable_saturation(
        TransportError(code=TransportErrorCode.INTEGRITY, message="bad",
                       retryable=False)
    )


def test_capacity_is_released_on_an_exception_path() -> None:
    """A raising operation must not permanently consume a slot."""
    guard = IsolateAdmissionGuard(limit=1)
    with pytest.raises(RuntimeError):
        with guard:
            raise RuntimeError("dependency blew up")
    assert guard.in_flight == 0
    with guard:
        assert guard.in_flight == 1


def test_releasing_without_holding_is_an_error() -> None:
    """Silently tolerating this would make the limit drift upward over time."""
    guard = IsolateAdmissionGuard(limit=1)
    with pytest.raises(RuntimeError, match="without being held"):
        guard.release()


@pytest.mark.parametrize("bad_limit", [0, -1, True, "2", 1.0, None])
def test_a_bad_limit_is_refused_at_construction(bad_limit: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        IsolateAdmissionGuard(limit=bad_limit)  # type: ignore[arg-type]


def test_concurrent_acquires_cannot_race_past_the_limit() -> None:
    """Check and increment must be one critical section.

    If they were not, N threads could all observe a free slot and the bound
    would be advisory rather than real.
    """
    guard = IsolateAdmissionGuard(limit=5)
    admitted: list[int] = []
    refused: list[int] = []
    barrier = threading.Barrier(32)
    # Every thread CONTENDS FOR THE SAME 5 SLOTS and holds them until all have
    # been offered. If the check and the increment were not one critical
    # section, threads would observe the same free slot together and the bound
    # would be advisory.
    release = threading.Event()

    def worker(index: int) -> None:
        barrier.wait()
        try:
            guard.try_acquire()
        except KnowledgeSaturatedError:
            refused.append(index)
            return
        admitted.append(index)
        # Hold the slot so the 5 slots are genuinely contended.
        release.wait(5.0)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(32)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    release.set()

    assert guard.peak_in_flight == 5
    assert len(admitted) == 5, f"admitted {len(admitted)} with a limit of 5"
    assert len(refused) == 27
    assert guard.in_flight == 5
    for _ in admitted:
        guard.release()
    assert guard.in_flight == 0


def test_counters_report_refusals_rather_than_hiding_them() -> None:
    """REQ-CFOPS-013 is about exposure: dropped work must be countable."""
    guard = IsolateAdmissionGuard(limit=1)
    with guard:
        for _ in range(3):
            with pytest.raises(KnowledgeSaturatedError):
                guard.try_acquire()
    assert guard.refused_count == 3
    assert guard.admitted_count == 1
    assert guard.peak_in_flight == 1


def test_no_queue_is_ever_created() -> None:
    """Fail fast, structurally.

    A queue in an isolate that can be evicted at any moment turns a clean
    retryable refusal into a hang that fails for a different reason with the
    caller's deadline already spent.
    """
    guard = IsolateAdmissionGuard(limit=1)
    public = {n for n in dir(guard) if not n.startswith("_")}
    assert public == {
        "limit", "in_flight", "peak_in_flight", "refused_count",
        "admitted_count", "try_acquire", "release",
    }
    for name in public:
        assert "queue" not in name.lower()
        assert "wait" not in name.lower()
        assert "pending" not in name.lower()


def test_the_message_leaks_no_capacity_or_topology_detail() -> None:
    guard = IsolateAdmissionGuard(limit=4)
    # Fill every slot, so the next attempt is the one that is refused.
    held = [guard for _ in range(4)]
    for slot in held:
        slot.try_acquire()
    with pytest.raises(KnowledgeSaturatedError) as caught:
        guard.try_acquire()
    message = caught.value.error.message
    for forbidden in ("account", "database_id", "bucket", "index", "namespace",
                      "token", "secret", "isolate_id", "colo"):
        assert forbidden not in message.lower().replace("this isolate", "")


def test_concurrent_async_operations_are_bounded_and_drain() -> None:
    """Async concurrency is the actual case; the bound must hold there too."""
    guard = IsolateAdmissionGuard(limit=3)
    admitted = 0
    refused = 0
    gate = asyncio.Event()

    async def operation() -> None:
        nonlocal admitted, refused
        try:
            with guard:
                admitted += 1
                await gate.wait()
        except KnowledgeSaturatedError:
            refused += 1

    async def main() -> None:
        tasks = [asyncio.create_task(operation()) for _ in range(10)]
        await asyncio.sleep(0)          # let the first three take their slots
        assert guard.in_flight == 3
        gate.set()
        await asyncio.gather(*tasks)

    asyncio.run(main())
    assert admitted == 3
    assert refused == 7
    assert guard.in_flight == 0


@pytest.mark.asyncio
async def test_the_transport_boundary_surfaces_saturation_as_retryable() -> None:
    """A refusal must reach the caller as ok=false with the SATURATED code.

    Not an exception escaping the transport, and not a successful empty
    response: "we refused to start" and "there was nothing to return" are
    different facts and only one of them invites an immediate retry.
    """
    from dataclasses import dataclass, field

    from oai2.core import KnowledgeId
    from oai2.knowledge.transport import (
        KnowledgeTransportRequest,
        TransportAuthContext,
        TransportErrorCode,
        TransportOperation,
    )
    from oai2.knowledge.worker_transport import KnowledgeWorkerTransport

    @dataclass
    class _Runtime:
        error: Exception | None = None
        puts: list[object] = field(default_factory=list)

        async def put(self, obj: object, *, vector: object = None, now: float = 0.0) -> int:
            if self.error is not None:
                raise self.error
            return 1

        async def get(self, knowledge_id: str) -> object | None:
            if self.error is not None:
                raise self.error
            return None

        async def retrieve(self, request: object, *, query_vector: object = None,
                           now: float = 0.0) -> object:
            if self.error is not None:
                raise self.error
            return None

    runtime = _Runtime(error=KnowledgeSaturatedError(limit=2, in_flight=2))
    transport = KnowledgeWorkerTransport(runtime=runtime)  # type: ignore[arg-type]

    for operation, kwargs in (
        (TransportOperation.GET, {"knowledge_id": KnowledgeId("ko_x")}),
        (TransportOperation.RETRIEVE, {"topic": "semantic"}),
    ):
        response = await transport.handle(
            KnowledgeTransportRequest(
                request_id="req-1",
                version="1",
                auth=TransportAuthContext(subject="s", capabilities=("knowledge.read",)),
                operation=operation,
                **kwargs,
            )
        )
        assert response.ok is False, f"{operation} did not fail"
        assert response.error is not None
        assert response.error.code is TransportErrorCode.SATURATED
        assert response.error.retryable is True
        # No knowledge object may ride along on a failure.
        assert not response.knowledge


def test_a_real_dependency_failure_is_still_not_a_saturation() -> None:
    """The new code must not swallow the distinctions it was added to preserve."""
    from oai2.knowledge.transport import TransportError

    for code in (
        TransportErrorCode.UNAVAILABLE_DEPENDENCY,
        TransportErrorCode.CONFLICT,
        TransportErrorCode.INTEGRITY,
    ):
        err = TransportError(code=code, message="x", retryable=True)
        assert not is_retryable_saturation(err)
