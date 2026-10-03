"""REQ-CFOPS-013 — saturation is exposed, not silently dropped or queued.

The mechanism is an ISOLATE-LOCAL in-flight bound with a fail-fast, retryable
SATURATED refusal. These tests pin that scope claim as hard as the behaviour,
because the failure mode being guarded against here is not "the limit is wrong"
but "the limit is described as more than it is".

Request-path enforcement is NOT proven here. This file covers the primitive;
``test_knowledge_worker_entrypoint_admission.py`` drives the real entrypoint and
is what makes the requirement an integration claim rather than a unit claim.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from pathlib import Path

import pytest

from oai2.knowledge.admission import (
    DEFAULT_ADMISSION_LIMIT,
    PUBLIC_SATURATION_MESSAGE,
    AdmissionState,
    KnowledgeSaturatedError,
    acquire_admission,
    is_retryable_saturation,
    is_retryable_saturation_error,
    validate_limit,
)
from oai2.knowledge.transport import TransportError, TransportErrorCode

# ---------------------------------------------------------------------------
# The bound itself
# ---------------------------------------------------------------------------


def test_the_limit_is_never_exceeded() -> None:
    state = AdmissionState()
    first = state.acquire(2)
    second = state.acquire(2)
    with pytest.raises(KnowledgeSaturatedError):
        state.acquire(2)
    assert state.peak_in_flight == 2
    second.release()
    first.release()
    assert state.in_flight == 0


def test_a_refusal_is_explicit_saturated_and_retryable() -> None:
    state = AdmissionState()
    lease = state.acquire(1)
    with pytest.raises(KnowledgeSaturatedError) as caught:
        state.acquire(1)
    exc = caught.value
    assert exc.code is TransportErrorCode.SATURATED
    assert exc.retryable is True
    # Internal diagnostics, deliberately kept off the public error.
    assert exc.limit == 1
    assert exc.in_flight == 1
    assert is_retryable_saturation(exc.error)
    assert is_retryable_saturation_error(exc)
    lease.release()


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
    state = AdmissionState()
    with pytest.raises(RuntimeError):
        lease = state.acquire(1)
        try:
            raise RuntimeError("dependency blew up")
        finally:
            lease.release()
    assert state.in_flight == 0
    # The slot really is free again, so a new acquire succeeds.
    lease = state.acquire(1)
    assert state.in_flight == 1
    lease.release()
    assert state.in_flight == 0


def test_releasing_without_holding_is_an_error() -> None:
    """Silently tolerating this would make the limit drift upward over time."""
    state = AdmissionState()
    with pytest.raises(RuntimeError, match="without being held"):
        state.release()


def test_a_lease_released_twice_is_an_error() -> None:
    """A double release would inflate the apparent free capacity."""
    state = AdmissionState()
    lease = state.acquire(1)
    lease.release()
    with pytest.raises(RuntimeError, match="already released"):
        lease.release()


@pytest.mark.parametrize("bad_limit", [0, -1, True, "2", 1.0, None])
def test_a_bad_limit_is_refused_at_acquisition(bad_limit: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        validate_limit(bad_limit)  # type: ignore[arg-type]


def test_counters_report_refusals_rather_than_hiding_them() -> None:
    """REQ-CFOPS-013 is about exposure: dropped work must be countable."""
    state = AdmissionState()
    state.acquire(1)
    for _ in range(3):
        with pytest.raises(KnowledgeSaturatedError):
            state.acquire(1)
    assert state.refused_count == 3
    assert state.admitted_count == 1
    assert state.peak_in_flight == 1


def test_no_queue_is_ever_created() -> None:
    """Fail fast, structurally.

    A queue in an isolate that can be evicted at any moment turns a clean
    retryable refusal into a hang that fails for a different reason with the
    caller's deadline already spent.
    """
    state = AdmissionState()
    public = {n for n in dir(state) if not n.startswith("_")}
    assert public == {
        "acquire", "admitted_count", "in_flight", "last_limit",
        "peak_in_flight", "refused_count", "release", "snapshot",
    }
    for name in public:
        assert "queue" not in name.lower()
        assert "wait" not in name.lower()
        assert "pending" not in name.lower()


# ---------------------------------------------------------------------------
# The deployment-runtime constraints, pinned structurally
# ---------------------------------------------------------------------------


def _function_ast(func: object) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(inspect.getsource(func).lstrip())
    assert isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef))
    return tree.body[0]  # type: ignore[return-value]


def test_the_admission_module_imports_no_threading_or_multiprocessing() -> None:
    """The primitive must be honoured by the runtime it ships to.

    Cloudflare's Python Workers run under WebAssembly/Pyodide, where
    ``threading`` is importable but not functional. A lock built on it would
    import cleanly and silently fail to exclude, which is worse than failing
    loudly. So this is a structural assertion, not a behavioural one.

    Parsed from the AST rather than grepped: the module DOCSTRING explains at
    length why threading was removed, and a substring scan would flag that
    explanation as the very thing it forbids.
    """
    import oai2.knowledge.admission as module

    tree = ast.parse(Path(module.__file__).read_text())
    forbidden_modules = {"threading", "multiprocessing", "_thread"}

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert not (imported & forbidden_modules), (
        f"admission.py imports {sorted(imported & forbidden_modules)}; the "
        "Worker runtime does not honour them"
    )

    # And no lock/semaphore primitive reached it under any other name.
    banned_calls = {"Lock", "RLock", "Semaphore", "BoundedSemaphore",
                    "Condition", "Event", "Barrier"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = node.func
            name = getattr(target, "attr", getattr(target, "id", ""))
            assert name not in banned_calls, f"admission.py calls {name}()"


def test_acquire_and_release_contain_no_await() -> None:
    """Atomicity comes from the absence of a suspension point, so assert it.

    If either method ever gained an ``await``, the check-then-increment would no
    longer be one uninterruptible block and the bound would become advisory
    without anything failing. This is the invariant the no-lock design rests
    on, so it is checked rather than described in a comment.
    """
    from oai2.knowledge.admission import AdmissionLease

    for func in (AdmissionState.acquire, AdmissionState.release,
                 AdmissionLease.release, acquire_admission):
        node = _function_ast(func)
        assert not isinstance(node, ast.AsyncFunctionDef), (
            f"{func.__qualname__} is async; the entrypoint's acquire must be "
            "synchronous so the check and the increment cannot interleave"
        )
        for inner in ast.walk(node):
            assert not isinstance(inner, ast.Await), (
                f"{func.__qualname__} awaits; the critical section must not "
                "suspend"
            )


# ---------------------------------------------------------------------------
# Isolation, under the concurrency the runtime actually has
# ---------------------------------------------------------------------------


def test_concurrent_async_operations_are_bounded_and_drain() -> None:
    """Async interleaving is the real case; the bound must hold there too.

    The old version of this test used OS threads. That was evidence about CPython
    and a ``threading.Lock``, neither of which describes the deployment runtime.
    This models the actual model instead: one isolate, one event loop, tasks
    interleaving at await points. The bound still has to hold.
    """
    state = AdmissionState()
    admitted = 0
    refused = 0
    gate = asyncio.Event()

    async def operation() -> None:
        nonlocal admitted, refused
        try:
            lease = state.acquire(3)
        except KnowledgeSaturatedError:
            refused += 1
            return
        admitted += 1
        try:
            await gate.wait()
        finally:
            lease.release()

    async def main() -> None:
        tasks = [asyncio.create_task(operation()) for _ in range(10)]
        await asyncio.sleep(0)          # let the first three take their slots
        assert state.in_flight == 3
        gate.set()
        await asyncio.gather(*tasks)

    asyncio.run(main())
    assert admitted == 3
    assert refused == 7
    assert state.in_flight == 0


def test_check_and_increment_cannot_be_interleaved() -> None:
    """No task may observe a free slot that another has already taken.

    This is the property a lock used to provide. It holds because ``acquire``
    runs to completion with no suspension point, so a second task can only ever
    run either entirely before or entirely after the first.
    """
    state = AdmissionState()
    observed: list[int] = []

    async def attempt() -> None:
        try:
            lease = state.acquire(1)
        except KnowledgeSaturatedError:
            observed.append(-1)
            return
        # Yield here: a scheduler switch lands AFTER the critical section, so
        # by the time this task resumes, the slot is already visible as taken.
        await asyncio.sleep(0)
        observed.append(state.in_flight)
        lease.release()

    async def main() -> None:
        await asyncio.gather(*(asyncio.create_task(attempt()) for _ in range(4)))

    asyncio.run(main())
    assert sorted(observed) == [-1, -1, -1, 1], (
        "a task resumed mid-critical-section; acquire is not atomic"
    )


def test_the_module_singleton_is_shared_and_holds_no_binding_derived_state() -> None:
    from oai2.knowledge.admission import ISOLATE_ADMISSION

    assert isinstance(ISOLATE_ADMISSION, AdmissionState)
    # Counters only. A binding-derived client cached here would go stale when
    # Cloudflare recycles an isolate across a binding-only change.
    for name in dir(ISOLATE_ADMISSION):
        if name.startswith("_"):
            continue
        value = getattr(ISOLATE_ADMISSION, name)
        assert not hasattr(value, "prepare"), (
            f"{name!r} looks like a cached D1 statement"
        )
    ISOLATE_ADMISSION._reset_for_tests()


def test_a_default_limit_is_documented_and_positive() -> None:
    assert isinstance(DEFAULT_ADMISSION_LIMIT, int)
    assert DEFAULT_ADMISSION_LIMIT >= 1
    assert validate_limit(DEFAULT_ADMISSION_LIMIT) == DEFAULT_ADMISSION_LIMIT


# ---------------------------------------------------------------------------
# The public message
# ---------------------------------------------------------------------------


def test_the_message_leaks_no_capacity_or_topology_detail() -> None:
    """§7: the public error must name no figure and no resource.

    The previous revision put ``in_flight/limit`` in the caller-visible string
    while the report claimed capacity detail was not exposed. That claim was
    false. Counts now live on the exception for operator logs; the wire message
    is fixed and generic.
    """
    state = AdmissionState()
    for _ in range(4):
        state.acquire(4)
    with pytest.raises(KnowledgeSaturatedError) as caught:
        state.acquire(4)
    message = caught.value.error.message

    assert message == PUBLIC_SATURATION_MESSAGE
    # No capacity figures.
    assert "4" not in message
    assert "in_flight" not in message
    assert "limit" not in message.lower()
    # No topology or identity.
    for forbidden in ("account", "database_id", "bucket", "index", "namespace",
                      "token", "secret", "isolate_id", "colo", "d1", "r2",
                      "vectorize"):
        assert forbidden not in message.lower()
    # The internal detail is still available, just not published.
    assert caught.value.in_flight == 4
    assert caught.value.limit == 4


def test_a_real_dependency_failure_is_still_not_a_saturation() -> None:
    """The new code must not swallow the distinctions it was added to preserve."""
    for code in (
        TransportErrorCode.UNAVAILABLE_DEPENDENCY,
        TransportErrorCode.CONFLICT,
        TransportErrorCode.INTEGRITY,
    ):
        err = TransportError(code=code, message="x", retryable=True)
        assert not is_retryable_saturation(err)


def test_the_transport_boundary_surfaces_saturation_as_retryable() -> None:
    """A refusal must reach the caller as ok=false with the SATURATED code.

    Not an exception escaping the transport, and not a successful empty
    response: "we refused to start" and "there was nothing to return" are
    different facts and only one of which invites an immediate retry.
    """
    from dataclasses import dataclass, field

    from oai2.core import KnowledgeId
    from oai2.knowledge.transport import (
        KnowledgeTransportRequest,
        TransportAuthContext,
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

    async def main() -> None:
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

    asyncio.run(main())
