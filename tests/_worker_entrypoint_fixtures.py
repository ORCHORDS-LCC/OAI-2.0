"""Shared test doubles for driving the Cloudflare Worker entrypoint offline.

Follows the ``tests/_evidence_fixtures.py`` precedent: one canonical set of
builders, imported explicitly, with no ``pytest`` fixtures and no shared
mutable state.

THE STUB IS THE POINT
---------------------
``WorkerEntrypoint`` constructs a NEW instance per invocation. A stub that
reused one instance would let anything cached on ``self`` look correct, so
``_WorkerEntrypointBase`` below is used the way the runtime uses it: the test
constructs a fresh object for every simulated request and never holds one
across two. That is what makes an ``self``-scoped implementation fail here
rather than pass.

The stub is a local test double. It is NOT evidence about the Pyodide runtime
itself — see the module docstring in ``oai2/knowledge/admission.py`` for what
is claimed and what is not.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import pytest

TOKEN = "test-token"


class _Response:
    def __init__(self, payload: Any, status: int) -> None:
        self.payload = payload
        self.status = status

    @classmethod
    def json(cls, payload: Any, status: int = 200) -> _Response:
        return cls(payload, status)


class _WorkerEntrypointBase:
    """Models a NEW instance per invocation."""

    def __init__(self, env: Any, ctx: Any = None) -> None:
        self.env = env
        self.ctx = ctx


@contextmanager
def workers_stub() -> Iterator[None]:
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


@dataclass
class Request:
    # FIELD ORDER IS LOAD-BEARING: `body` is first so the common
    # `Request(_valid_body())` call site passes the body, not the method.
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
class Env:
    KNOWLEDGE_AUTH_TOKEN: str = TOKEN
    EMBEDDING_VERSION: str = "embed-v1"
    EMBEDDING_DIGEST: str = "digest-v1"
    KNOWLEDGE_MAX_IN_FLIGHT: str = "1"
    DB: object = None
    KNOWLEDGE_R2: object = None
    KNOWLEDGE_VECTORIZE: object = None
    KNOWLEDGE_KV: object = None


class Gate:
    """A controlled awaited fake operation, so one request can be held open.

    ``hold`` makes the operation wait for an explicit release. Tests that only
    care about the request completing leave it False, so the gate opens and
    closes on its own and no test has to coordinate a release.
    """

    def __init__(self, *, hold: bool = True) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.hold = hold
        self.calls = 0
        if not hold:
            self.release.set()

    async def run(self) -> None:
        self.calls += 1
        self.entered.set()
        await self.release.wait()


class Components:
    """A stand-in component set whose transport is scriptable.

    ``error`` raises after the gate opens, so a held request can be raced by a
    second one. ``result`` returns a canned response for a scripted outcome.
    """

    def __init__(
        self,
        gate: Gate,
        *,
        error: Exception | None = None,
        result: Any = None,
    ) -> None:
        self._gate = gate
        self._error = error
        self._result = result
        self.transport = self
        self.handled: list[object] = []
        self.traces: list[object] = []

    async def handle(self, request: object, *, trace: object = None) -> Any:
        self.handled.append(request)
        self.traces.append(trace)
        await self._gate.run()
        if self._error is not None:
            raise self._error
        if self._result is not None:
            return self._result
        from oai2.knowledge import KnowledgeTransportResponse

        return KnowledgeTransportResponse(request_id=request.request_id, ok=True)

    async def __call__(self, *args: object, **kwargs: object) -> Components:
        return self


def valid_body(request_id: str = "req-1") -> dict[str, object]:
    return {
        "request_id": request_id,
        "version": "1",
        "operation": "get",
        "knowledge_id": "ko_entry",
    }


@contextmanager
def entrypoint() -> Iterator[types.SimpleNamespace]:
    """Import the entrypoint fresh with the ``workers`` stub installed.

    A fresh import per test so module-scope state does not leak between tests.
    Module-scope admission state is reset explicitly by
    :func:`reset_admission`.
    """
    saved_entry = sys.modules.pop(
        "deploy.cloudflare.knowledge_worker.entry", None
    )
    with workers_stub():
        module = importlib.import_module(
            "deploy.cloudflare.knowledge_worker.entry"
        )
        try:
            yield module
        finally:
            if saved_entry is not None:  # pragma: no cover
                sys.modules["deploy.cloudflare.knowledge_worker.entry"] = saved_entry


def reset_admission() -> None:
    """Clear the module-scope admission counters. Refuses while a slot is held."""
    from oai2.knowledge.admission import ISOLATE_ADMISSION

    ISOLATE_ADMISSION._reset_for_tests()


def wire(monkeypatch: pytest.MonkeyPatch, components: Components) -> None:
    async def fake_build(**_kwargs: object) -> Components:
        return components

    monkeypatch.setattr(
        "oai2.knowledge.build_cloudflare_knowledge_components", fake_build
    )
    # entry.py imported the symbol directly, so patch it where it is used.
    import deploy.cloudflare.knowledge_worker.entry as entry

    monkeypatch.setattr(entry, "build_cloudflare_knowledge_components", fake_build)


async def drive(
    monkeypatch: pytest.MonkeyPatch,
    entry: Any,
    env: object,
    body: dict[str, object],
    *,
    gate: Gate | None = None,
    build: Any = None,
) -> Any:
    """Run one request to completion, releasing its gate.

    ``build`` lets a caller supply its own builder (to inspect kwargs, or to
    count invocations) while still passing the ``gate`` that builder must use.
    The gate is always owned here, never by the builder: a builder that makes
    its own gate would leave this awaiting a gate nobody enters.
    """
    if gate is None:
        gate = Gate(hold=False)
    if build is None:
        wire(monkeypatch, Components(gate))
    else:
        import deploy.cloudflare.knowledge_worker.entry as entry_module

        monkeypatch.setattr(
            entry_module, "build_cloudflare_knowledge_components", build
        )
    instance = entry.Default(env)
    task = asyncio.create_task(instance.fetch(Request(body)))
    if gate.hold:
        await gate.entered.wait()
        gate.release.set()
    return await task


__all__ = [
    "TOKEN",
    "Components",
    "Env",
    "Gate",
    "Request",
    "drive",
    "entrypoint",
    "reset_admission",
    "valid_body",
    "wire",
    "workers_stub",
]
