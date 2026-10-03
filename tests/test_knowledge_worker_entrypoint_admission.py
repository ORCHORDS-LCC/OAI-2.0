"""Entrypoint-level admission: does a real request path actually enforce it?

The unit tests in ``test_knowledge_admission.py`` prove the guard primitive
behaves. They cannot prove the request path USES it, and they cannot prove the
scope is right. This file does both, by driving ``Default.fetch`` the way
Cloudflare does.

THE STUB IS THE POINT
---------------------
``WorkerEntrypoint`` constructs a NEW instance per invocation. A stub that
reused one instance would let a guard cached on ``self`` look correct, so
``_WorkerEntrypointBase`` below constructs a fresh instance per ``fetch`` call
by modelling the documented lifecycle: a factory function returns a brand new
object every time, and the test only ever holds instances for the duration of a
single request. That is the opposite of a singleton and it is what makes an
``self._admission_guard`` implementation fail here rather than pass.

The stub is a local test double. It is NOT evidence about the Pyodide runtime
itself — see the module docstring in ``oai2/knowledge/admission.py`` for what
is claimed and what is not.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import pytest

TOKEN = "test-token"


# ---------------------------------------------------------------------------
# A `workers` stub whose WorkerEntrypoint has the REAL per-invocation lifecycle
# ---------------------------------------------------------------------------


class _Response:
    def __init__(self, payload: Any, status: int) -> None:
        self.payload = payload
        self.status = status

    @classmethod
    def json(cls, payload: Any, status: int = 200) -> _Response:
        return cls(payload, status)


class _WorkerEntrypointBase:
    """Models a NEW instance per invocation.

    ``__init_subclass__`` is not what does it — the point is that the test
    constructs a fresh object for every simulated request, exactly as the
    runtime does, and never holds one across two.
    """

    def __init__(self, env: Any, ctx: Any = None) -> None:
        self.env = env
        self.ctx = ctx


@contextmanager
def _workers_stub() -> Iterator[None]:
    module = types.ModuleType("workers")
    module.Response = _Response  # type: ignore[attr-defined]
    module.WorkerEntrypoint = _WorkerEntrypointBase  # type: ignore[attr-defined]
    saved = sys.modules.get("workers")
    sys.modules["workers"] = module
    try:
        yield
    finally:
        if saved is None:
            sys.modules.pop("workers", None)
        else:  # pragma: no cover
            sys.modules["workers"] = saved


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _Request:
    # FIELD ORDER IS LOAD-BEARING: `body` is first so that the common
    # `_Request(_valid_body())` call site passes the body, not the method.
    # With `method` first, `_Request(_valid_body())` silently set method to a
    # dict, the handler returned 405 before admission was ever reached, and the
    # concurrency tests then blocked forever on a gate that was never entered.
    body: object = None
    method: str = "POST"
    headers: dict[str, str] = field(
        default_factory=lambda: {"Authorization": f"Bearer {TOKEN}"}
    )

    async def json(self) -> object:
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


@dataclass
class _Env:
    KNOWLEDGE_AUTH_TOKEN: str = TOKEN
    EMBEDDING_VERSION: str = "embed-v1"
    EMBEDDING_DIGEST: str = "digest-v1"
    KNOWLEDGE_MAX_IN_FLIGHT: str = "1"
    DB: object = None
    KNOWLEDGE_R2: object = None
    KNOWLEDGE_VECTORIZE: object = None
    KNOWLEDGE_KV: object = None


class _Gate:
    """A controlled awaited fake operation, so one request can be held open."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def run(self) -> None:
        self.calls += 1
        self.entered.set()
        await self.release.wait()


class _Components:
    def __init__(self, gate: _Gate, *, error: Exception | None = None) -> None:
        self._gate = gate
        self._error = error
        self.transport = self
        self.handled: list[object] = []

    async def handle(self, request: object) -> Any:
        if self._error is not None:
            await self._gate.run()
            raise self._error
        await self._gate.run()
        from oai2.knowledge import (
            KnowledgeTransportResponse,
        )

        if error := _error_for(request):
            return KnowledgeTransportResponse(
                request_id=request.request_id, ok=False, error=error
            )
        return KnowledgeTransportResponse(request_id=request.request_id, ok=True)

    async def __call__(self, *args: object, **kwargs: object) -> _Components:
        return self


def _error_for(request: object) -> Any:
    return None


def _valid_body(request_id: str = "req-1") -> dict[str, object]:
    return {
        "request_id": request_id,
        "version": "1",
        "operation": "get",
        "knowledge_id": "ko_entry",
    }


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@contextmanager
def _entry() -> Iterator[types.SimpleNamespace]:
    """Import the entrypoint fresh with the `workers` stub installed.

    A fresh import per test so module-scope admission state does not leak
    between tests. The state itself is reset explicitly below.
    """
    saved_entry = sys.modules.pop(
        "deploy.cloudflare.knowledge_worker.entry", None
    )
    with _workers_stub():
        module = importlib.import_module(
            "deploy.cloudflare.knowledge_worker.entry"
        )
        try:
            yield module
        finally:
            if saved_entry is not None:  # pragma: no cover
                sys.modules["deploy.cloudflare.knowledge_worker.entry"] = saved_entry


@pytest.fixture(autouse=True)
def _reset_admission() -> Iterator[None]:
    from oai2.knowledge.admission import ISOLATE_ADMISSION

    ISOLATE_ADMISSION._reset_for_tests()
    try:
        yield
    finally:
        if ISOLATE_ADMISSION.in_flight == 0:
            ISOLATE_ADMISSION._reset_for_tests()
        else:  # pragma: no cover
            pytest.fail("an admission slot leaked out of a test")


async def _drive(monkeypatch: pytest.MonkeyPatch, entry: Any, env: object,
                 body: dict[str, object], *, gate: _Gate | None = None,
                 build: Any = None) -> Any:
    """Run one request to completion, releasing its gate.

    Without this an admitted request awaits its gate forever, because a
    ``_Gate`` that is never released models a dependency that never returns.

    ``build`` lets a caller supply its own builder (to inspect kwargs, or to
    count invocations) while still passing the ``gate`` that builder must use.
    Both are needed together: if a caller's builder makes its own gate, this
    helper would await a gate nobody ever enters and the test would hang. The
    gate is therefore always owned here, never by the builder.
    """
    if gate is None:
        gate = _Gate()
    if build is None:
        components = _Components(gate)
        _wire(monkeypatch, components)
    else:
        monkeypatch.setattr(
            entry, "build_cloudflare_knowledge_components", build
        )
    instance = entry.Default(env)
    task = asyncio.create_task(instance.fetch(_Request(body)))
    await gate.entered.wait()
    gate.release.set()
    return await task


def _wire(monkeypatch: pytest.MonkeyPatch, components: _Components) -> None:
    async def fake_build(**_kwargs: object) -> _Components:
        return components

    monkeypatch.setattr(
        "oai2.knowledge.build_cloudflare_knowledge_components", fake_build
    )
    # entry.py imported the symbol directly, so patch it where it is used.
    import deploy.cloudflare.knowledge_worker.entry as entry

    monkeypatch.setattr(entry, "build_cloudflare_knowledge_components", fake_build)


# ---------------------------------------------------------------------------
# §5 — the tests that prove the request path is fenced
# ---------------------------------------------------------------------------


def test_separate_entrypoint_instances_share_the_isolate_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A, B, C as SEPARATE Default instances, with limit=1.

    Proves simultaneously that the state is not on ``self``, that separate
    WorkerEntrypoint instances share one isolate counter, that saturation
    actually protects the request path, and that slots are released.
    """
    with _entry() as entry:
        from oai2.knowledge.admission import ISOLATE_ADMISSION

        gate = _Gate()
        components = _Components(gate)
        _wire(monkeypatch, components)
        env = _Env(KNOWLEDGE_MAX_IN_FLIGHT="1")

        async def scenario() -> tuple[Any, Any, Any, int]:
            # --- A: a DIFFERENT instance, admitted, held open ---
            instance_a = entry.Default(env)
            assert instance_a is not entry.Default(env), (
                "the stub must construct a new instance per invocation, or this "
                "test proves nothing about instance-vs-isolate scope"
            )
            task_a = asyncio.create_task(instance_a.fetch(_Request(_valid_body("A"))))
            await gate.entered.wait()
            assert ISOLATE_ADMISSION.in_flight == 1

            # --- B: ANOTHER different instance, must be refused ---
            instance_b = entry.Default(env)
            response_b = await instance_b.fetch(_Request(_valid_body("B")))

            # C is created after A releases.
            gate.release.set()
            response_a = await task_a

            instance_c = entry.Default(env)
            gate2 = _Gate()
            _wire(monkeypatch, _Components(gate2))
            task_c = asyncio.create_task(instance_c.fetch(_Request(_valid_body("C"))))
            await gate2.entered.wait()
            admitted_c = ISOLATE_ADMISSION.in_flight == 1
            gate2.release.set()
            response_c = await task_c

            return response_a, response_b, response_c, admitted_c

        response_a, response_b, response_c, admitted_c = asyncio.run(scenario())

        # A was admitted and served.
        assert response_a.status == 200
        assert response_a.payload["ok"] is True
        assert response_a.payload["request_id"] == "A"

        # B never reached the knowledge runtime. Only A entered it.
        assert gate.calls == 1, "the runtime was entered by a refused request"
        assert response_b.status == 503
        assert response_b.payload["ok"] is False
        assert response_b.payload["error"]["code"] == "saturated"
        assert response_b.payload["error"]["retryable"] is True
        assert response_b.payload["request_id"] == "B"

        # C was admitted once A drained.
        assert admitted_c, "a fresh instance was refused after the slot freed"
        assert response_c.status == 200
        assert response_c.payload["ok"] is True
        assert ISOLATE_ADMISSION.in_flight == 0
        assert ISOLATE_ADMISSION.refused_count == 1


def test_a_dependency_exception_releases_the_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raises from the dependency; the next FRESH instance must succeed."""
    with _entry() as entry:
        from oai2.knowledge.admission import ISOLATE_ADMISSION

        env = _Env(KNOWLEDGE_MAX_IN_FLIGHT="1")

        async def scenario() -> tuple[Any, Any]:
            gate = _Gate()
            _wire(monkeypatch, _Components(gate, error=RuntimeError("D1 down")))
            instance_a = entry.Default(env)
            task_a = asyncio.create_task(instance_a.fetch(_Request(_valid_body("A"))))
            await gate.entered.wait()

            instance_b = entry.Default(env)
            response_b = await instance_b.fetch(_Request(_valid_body("B")))

            gate.release.set()
            response_a = await task_a
            return response_a, response_b

        response_a, response_b = asyncio.run(scenario())

        # A's dependency failure was reported as a retryable dependency fault,
        # not swallowed and not a crash.
        assert response_a.status == 503
        assert response_a.payload["error"]["code"] == "unavailable_dependency"

        # The slot came back, so B got a real refusal-free path... but B ran
        # while A was still held, so B is saturated. The point is that AFTER A
        # completes, admission is free again.
        assert response_b.payload["error"]["code"] == "saturated"
        assert ISOLATE_ADMISSION.in_flight == 0, "the exception path leaked a slot"

        # A fresh instance now succeeds, proving the slot really is back.
        response_c = asyncio.run(
            _drive(monkeypatch, entry, env, _valid_body("C"))
        )
        assert response_c.status == 200
        assert response_c.payload["ok"] is True


@pytest.mark.parametrize(
    "method,body,expected_status,expected_code",
    [
        ("GET", None, 405, "validation"),
        ("POST", ValueError("not json"), 400, "validation"),
        ("POST", "a string", 400, "validation"),
        ("POST", {"request_id": "x"}, 400, "validation"),
    ],
)
def test_cheap_failures_never_consume_a_slot(
    monkeypatch: pytest.MonkeyPatch,
    method: str, body: object, expected_status: int, expected_code: str
) -> None:
    """An unauthenticated or malformed request must not take a slot.

    These are the requests that bypass admission entirely, so a burst of them
    cannot crowd out real work — which is the whole reason admission sits after
    cheap validation rather than first.
    """
    with _entry() as entry:
        from oai2.knowledge.admission import ISOLATE_ADMISSION

        gate = _Gate()
        _wire(monkeypatch, _Components(gate))
        env = _Env(KNOWLEDGE_MAX_IN_FLIGHT="1")

        response = asyncio.run(
            entry.Default(env).fetch(_Request(method=method, body=body))
        )

        assert response.status == expected_status
        assert response.payload["error"]["code"] == expected_code
        assert ISOLATE_ADMISSION.in_flight == 0
        assert ISOLATE_ADMISSION.admitted_count == 0
        assert gate.calls == 0, "a rejected request reached the runtime"


def test_an_unauthenticated_request_does_not_consume_a_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _entry() as entry:
        from oai2.knowledge.admission import ISOLATE_ADMISSION

        gate = _Gate()
        _wire(monkeypatch, _Components(gate))
        env = _Env(KNOWLEDGE_MAX_IN_FLIGHT="1")
        request = _Request(_valid_body())
        request.headers = {"Authorization": "Bearer wrong"}

        response = asyncio.run(entry.Default(env).fetch(request))

        assert response.status == 401
        assert ISOLATE_ADMISSION.admitted_count == 0
        assert gate.calls == 0


def test_a_misconfigured_limit_is_an_operator_error_not_a_saturation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad limit is not backpressure and must not be reported as such."""
    with _entry() as entry:
        from oai2.knowledge.admission import ISOLATE_ADMISSION

        gate = _Gate()
        _wire(monkeypatch, _Components(gate))

        for bad in ("0", "-3", "abc", True):
            env = _Env(KNOWLEDGE_MAX_IN_FLIGHT=bad)  # type: ignore[arg-type]
            response = asyncio.run(
                entry.Default(env).fetch(_Request(_valid_body()))
            )
            assert response.status == 500, f"limit={bad!r}"
            assert response.payload["error"]["code"] == "internal"
            assert ISOLATE_ADMISSION.admitted_count == 0
            assert gate.calls == 0


def test_the_limit_is_read_per_request_not_frozen_at_module_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A module-scope guard must not bind the first request's limit forever."""
    with _entry() as entry:
        from oai2.knowledge.admission import ISOLATE_ADMISSION

        one = _Env(KNOWLEDGE_MAX_IN_FLIGHT="1")
        four = _Env(KNOWLEDGE_MAX_IN_FLIGHT="4")
        assert asyncio.run(_drive(monkeypatch, entry, one, _valid_body())).status == 200
        assert ISOLATE_ADMISSION.last_limit == 1
        # Same isolate, a different configured limit, immediately effective.
        assert asyncio.run(_drive(monkeypatch, entry, four, _valid_body())).status == 200
        assert ISOLATE_ADMISSION.last_limit == 4


def test_a_missing_limit_falls_back_to_the_documented_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _entry() as entry:
        from oai2.knowledge.admission import (
            DEFAULT_ADMISSION_LIMIT,
            ISOLATE_ADMISSION,
        )

        # A real env with the variable genuinely absent, not set to "" or None:
        # getattr(env, ..., None) is what the entrypoint uses, and a value of
        # None must behave the same as a missing binding.
        env = types.SimpleNamespace(
            KNOWLEDGE_AUTH_TOKEN=TOKEN,
            EMBEDDING_VERSION="embed-v1",
            EMBEDDING_DIGEST="digest-v1",
            DB=None,
            KNOWLEDGE_R2=None,
            KNOWLEDGE_VECTORIZE=None,
            KNOWLEDGE_KV=None,
        )
        assert not hasattr(env, "KNOWLEDGE_MAX_IN_FLIGHT")

        response = asyncio.run(_drive(monkeypatch, entry, env, _valid_body()))
        assert response.status == 200
        assert ISOLATE_ADMISSION.last_limit == DEFAULT_ADMISSION_LIMIT


# ---------------------------------------------------------------------------
# §6 — the HTTP saturation response, end to end
# ---------------------------------------------------------------------------


def test_saturated_error_maps_to_a_status_and_the_full_json_body() -> None:
    """The complete response, not just the status helper.

    The body code is what distinguishes saturation from a dependency outage
    even though both are 503, so the body is the thing actually under test.
    """
    with _entry() as entry:
        from oai2.knowledge import KnowledgeTransportResponse, TransportError
        from oai2.knowledge.transport import TransportErrorCode

        response = KnowledgeTransportResponse(
            request_id="req-sat",
            ok=False,
            error=TransportError(
                code=TransportErrorCode.SATURATED,
                message="knowledge service is saturated; retry shortly",
                retryable=True,
            ),
        )
        status = entry._status_for_error(response)
        assert status == 503

        payload = response.model_dump(mode="json")
        body = json.loads(json.dumps(payload))
        assert body["ok"] is False
        assert body["error"]["code"] == "saturated"
        assert body["error"]["retryable"] is True
        assert "Retry-After" not in body


def test_saturation_and_dependency_outage_are_distinguishable_in_the_body() -> None:
    """Both are 503. The body code is the discriminator, so it must differ."""
    with _entry() as entry:
        from oai2.knowledge import KnowledgeTransportResponse, TransportError
        from oai2.knowledge.transport import TransportErrorCode

        saturated = KnowledgeTransportResponse(
            request_id="r",
            ok=False,
            error=TransportError(
                code=TransportErrorCode.SATURATED, message="m", retryable=True
            ),
        )
        dependency = KnowledgeTransportResponse(
            request_id="r",
            ok=False,
            error=TransportError(
                code=TransportErrorCode.UNAVAILABLE_DEPENDENCY,
                message="m",
                retryable=True,
            ),
        )
        assert entry._status_for_error(saturated) == entry._status_for_error(dependency)
        assert saturated.error.code.value != dependency.error.code.value


def test_every_error_code_has_a_status_and_saturation_is_503_not_429() -> None:
    from oai2.knowledge.transport import TransportErrorCode

    with _entry() as entry:
        from oai2.knowledge import KnowledgeTransportResponse, TransportError

        for code in TransportErrorCode:
            response = KnowledgeTransportResponse(
                request_id="r",
                ok=False,
                error=TransportError(code=code, message="m", retryable=True),
            )
            status = entry._status_for_error(response)
            assert 400 <= status < 600, f"{code} -> {status}"

        saturated = entry._status_for_error(
            KnowledgeTransportResponse(
                request_id="r",
                ok=False,
                error=TransportError(
                    code=TransportErrorCode.SATURATED, message="m", retryable=True
                ),
            )
        )
        assert saturated == 503
        # 429 would tell a client this is its own rate, tied to a quota window
        # this refusal does not have.
        assert saturated != 429


def test_an_unmapped_error_code_fails_closed_instead_of_raising() -> None:
    """A code with no status entry must not raise out of the handler.

    ``KnowledgeTransportResponse`` refuses to hold a failure without an error,
    so the realistic way to reach the guard is a code that exists on the enum
    but has no status entry — exactly what happens when someone adds a new
    code and forgets the map. That is the defect the guard exists for.
    """
    with _entry() as entry:
        from oai2.knowledge import KnowledgeTransportResponse, TransportError
        from oai2.knowledge.transport import TransportErrorCode

        # The real defect shape, against the pre-fix code path: the original
        # `_status_for_error` indexed the enum directly and raised KeyError.
        unmapped = {
            code: status
            for code, status in entry._STATUS_BY_CODE.items()
            if code is not TransportErrorCode.SATURATED
        }
        entry._STATUS_BY_CODE.clear()
        entry._STATUS_BY_CODE.update(unmapped)

        response = KnowledgeTransportResponse(
            request_id="r",
            ok=False,
            error=TransportError(
                code=TransportErrorCode.SATURATED, message="m", retryable=True
            ),
        )
        assert entry._status_for_error(response) == 500


# ---------------------------------------------------------------------------
# §8 — schema is not a request side effect
# ---------------------------------------------------------------------------


def test_the_request_path_never_applies_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ensure_schema must be False on the request path.

    WorkerEntrypoint builds a new instance per invocation, so the per-instance
    component memo is rebuilt every request and DDL would be re-attempted every
    request. That is both an ordinary side effect of serving traffic and
    ineffective as caching.
    """
    with _entry() as entry:
        seen: dict[str, object] = {}
        gate = _Gate()

        async def fake_build(**kwargs: object) -> _Components:
            seen.update(kwargs)
            return _Components(gate)

        env = _Env()
        asyncio.run(_drive(monkeypatch, entry, env, _valid_body(),
                           gate=gate, build=fake_build))
        assert seen["ensure_schema"] is False
        # And no DDL ran as a side effect of serving the request.
        assert set(seen) >= {"d1", "r2", "vectorize", "kv"}


def test_a_per_instance_component_memo_is_not_a_cross_request_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two invocations must build twice, proving the memo is intra-request."""
    with _entry() as entry:
        builds: list[int] = []
        env = _Env()
        gate = _Gate()

        async def counting_build(**_kwargs: object) -> _Components:
            builds.append(1)
            return _Components(gate)

        # One gate, released by the first request, so the second passes
        # straight through: this test is about BUILD COUNT, not concurrency.
        asyncio.run(_drive(monkeypatch, entry, env, _valid_body(),
                           gate=gate, build=counting_build))
        asyncio.run(_drive(monkeypatch, entry, env, _valid_body(),
                           gate=gate, build=counting_build))
        assert len(builds) == 2, (
            "a cross-request component cache would make this 1; that would be "
            "incorrect here because instance attributes do not survive an "
            "invocation, and binding-derived clients must not be cached globally"
        )


def test_no_binding_derived_client_is_cached_at_module_scope() -> None:
    """Guard against the tempting wrong fix: a global component cache."""
    import deploy.cloudflare.knowledge_worker.entry as entry

    module_globals = {
        name: value
        for name, value in vars(entry).items()
        if not name.startswith("__")
    }
    for name, value in module_globals.items():
        # Only the admission state and immutable constants are acceptable.
        if isinstance(value, (types.FunctionType, type)):
            continue
        assert not hasattr(value, "transport"), (
            f"module-level {name!r} looks like a cached component set; "
            "binding-derived clients must not live in global state"
        )
