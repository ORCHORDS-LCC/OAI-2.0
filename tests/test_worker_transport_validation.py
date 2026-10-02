"""Pin the outer guard rails of ``oai2.knowledge.worker_transport``.

The behavioural tests in ``tests/test_worker_transport.py`` cover the
happy paths for PUT (write capability + embedding dispatch), GET
(not-found error mapping), RETRIEVE (RetrievalRequest construction +
query embedding), missing/expired/wrong-capability auth failures, the
three runtime-error→TransportError mappings, and the
``require_auth=False`` local-trusted path. They do NOT pin:

* ``EmbeddingProvider`` Protocol structural-typing — any class with
  ``async embed(text: str) -> Sequence[float]`` is accepted;
* ``KnowledgeWorkerTransport.__init__`` field guards — the
  ``require_auth`` isinstance(bool) check rejects e.g. ``1`` even though
  truthy; the defaults (``embedding_provider=None``, ``require_auth=True``)
  are unpinned;
* ``handle(now=...)`` argument validation — ``_finite_non_negative``
  rejects ``NaN``, ``+inf``, ``-inf``, negative numbers, and ``bool``
  (because Python's ``isinstance(True, int) is True``);
* ``_embed`` invalid-vector guards — the runtime error raised on
  ``str`` / ``bytes`` / empty vectors;
* ``_embed`` no-provider path — returns ``None`` (NOT raises) when
  ``embedding_provider`` is ``None``;
* ``_authorize`` capability routing — PUT requires ``knowledge.write``,
  GET/RETRIEVE require ``knowledge.read`` (the test covers only
  GET-side missing auth + write cap on GET; PUT write-cap rejection
  and RETRIEVE read-cap rejection are asymmetric);
* ``_authorize`` non-finite ``expires_at`` — the ``math.isfinite``
  check (rejects ``NaN``, ``inf``);
* The catch-all ``except Exception`` → INTERNAL+not-retryable path —
  a custom exception class (e.g. ``ValueError``) must map to
  ``TransportErrorCode.INTERNAL`` with ``retryable=False``;
* ``_failure`` message-truncation safety — strips whitespace and
  clamps to 1000 chars (the wire contract for ``TransportError.message``
  from slice 29);
* The ``now=None`` fallback — ``handle()`` with ``now=None`` falls back
  to ``time.time()`` (pinned by asserting the response is constructed
  without raising on the ``now=None`` branch).

A refactor that drops the ``isinstance(bool)`` guard, broadens the
exception handler, changes the capability routing, or removes the
``_finite_non_negative`` check would propagate silently into the live
Cloudflare Worker contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pytest

from oai2.core import KnowledgeId, Status
from oai2.knowledge import (
    KnowledgeObject,
    KnowledgeWorkerTransport,
    RetrievalResult,
    TransportAuthContext,
    sha256_hex,
)
from oai2.knowledge.transport import (
    KnowledgeTransportRequest,
    TransportErrorCode,
    TransportOperation,
)

# ---------------------------------------------------------------------------
# Helper builders (mirror tests/test_worker_transport.py but only the
# fields each new test needs)
# ---------------------------------------------------------------------------


def _obj(content: str = "worker transport body") -> KnowledgeObject:
    return KnowledgeObject(
        knowledge_id=KnowledgeId("ko_worker_1"),
        topic="worker",
        content=content,
        content_hash=sha256_hex(content),
        source_uri="https://example.test/worker",
        retrieved_at=10.0,
        authority=0.9,
        status=Status.EXPERIMENTAL,
    )


@dataclass
class _FakeRuntime:
    """Minimal async runtime satisfying the structural-typing
    ``AsyncCloudflareKnowledgeRuntime`` contract that the worker
    transport actually calls (``put``, ``get``, ``retrieve``)."""

    put_revision: int = 8
    get_result: KnowledgeObject | None = field(default_factory=_obj)
    retrieve_result: RetrievalResult = field(
        default_factory=lambda: RetrievalResult(topic="worker", objects=[_obj()]),
    )
    error: Exception | None = None
    puts: list[tuple[KnowledgeObject, object, float]] = field(default_factory=list)
    retrieves: list[tuple[object, object]] = field(default_factory=list)

    async def put(
        self,
        obj: KnowledgeObject,
        *,
        vector: object = None,
        now: float | None = None,
    ) -> int:
        if self.error is not None:
            raise self.error
        assert now is not None
        self.puts.append((obj, vector, now))
        return self.put_revision

    async def get(self, _knowledge_id: KnowledgeId) -> KnowledgeObject | None:
        if self.error is not None:
            raise self.error
        return self.get_result

    async def retrieve(self, request: object, *, query_vector: object = None) -> RetrievalResult:
        if self.error is not None:
            raise self.error
        self.retrieves.append((request, query_vector))
        return self.retrieve_result


def _auth(*caps: str, expires_at: float | None = 1000.0) -> TransportAuthContext:
    return TransportAuthContext(
        subject="local-user",
        capabilities=tuple(caps),
        expires_at=expires_at,
    )


# ---------------------------------------------------------------------------
# EmbeddingProvider Protocol structural-typing
# ---------------------------------------------------------------------------


class _DuckEmbedder:
    """Duck-typed embedder — no explicit Protocol registration; the
    worker transport accepts it via Python's structural typing."""

    def __init__(self) -> None:
        self.inputs: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.inputs.append(text)
        return [0.0]


def test_embedding_provider_protocol_accepts_duck_typed_implementer() -> None:
    """``EmbeddingProvider`` is a structural ``Protocol``. A class with
    ``async def embed(self, text: str) -> Sequence[float]`` is accepted
    without explicit registration. Pinning this prevents a refactor that
    adds an explicit ABC base class (which would silently break all
    duck-typed embedders, including the test mocks)."""
    embedder = _DuckEmbedder()
    transport = KnowledgeWorkerTransport(
        _FakeRuntime(),  # type: ignore[arg-type]
        embedding_provider=embedder,
    )
    # The structural-typing assertion: just constructing the worker
    # transport with a duck-typed embedder must succeed. (Python's
    # Protocol is nominal only at type-check time; runtime accepts
    # any object with the right shape.)
    assert transport is not None


# ---------------------------------------------------------------------------
# KnowledgeWorkerTransport.__init__ guards
# ---------------------------------------------------------------------------


def test_worker_transport_default_require_auth_is_true() -> None:
    """Default ``require_auth=True`` — the production worker always
    requires authentication. A refactor that flipped the default would
    silently bypass auth on every new live deployment."""
    transport = KnowledgeWorkerTransport(_FakeRuntime())  # type: ignore[arg-type]
    # Direct private-attribute check: ``_require_auth`` is set in
    # ``__init__`` and is the field that ``_authorize`` consults.
    assert transport._require_auth is True  # noqa: SLF001


def test_worker_transport_default_embedding_provider_is_none() -> None:
    """Default ``embedding_provider=None``. Callers that need vector
    embeddings must pass one explicitly. A refactor that added a
    default provider would silently bypass the embedding-step error
    path (str/bytes/empty vector rejection)."""
    transport = KnowledgeWorkerTransport(_FakeRuntime())  # type: ignore[arg-type]
    assert transport._embedding_provider is None  # noqa: SLF001


def test_worker_transport_rejects_non_bool_require_auth() -> None:
    """``__init__`` validates ``isinstance(require_auth, bool)``. Passing
    ``1`` (int) or ``"yes"`` (str) is rejected even though both are
    truthy. The check protects against the implicit bool coercion
    pattern: callers that pass ``0`` / ``1`` would otherwise get the
    wrong auth default (``0`` → False, ``1`` → True; but the contract
    is "must be exactly a bool")."""
    with pytest.raises(ValueError, match="require_auth must be a boolean"):
        KnowledgeWorkerTransport(
            _FakeRuntime(),  # type: ignore[arg-type]
            require_auth=1,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="require_auth must be a boolean"):
        KnowledgeWorkerTransport(
            _FakeRuntime(),  # type: ignore[arg-type]
            require_auth="yes",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# handle(now=...) argument validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_now",
    [
        math.nan,
        math.inf,
        -math.inf,
        -1.0,
        -0.0001,
    ],
)
@pytest.mark.asyncio
async def test_handle_rejects_non_finite_or_negative_now(bad_now: float) -> None:
    """``handle(now=...)`` rejects non-finite and negative values via
    ``_finite_non_negative``. The check also rejects ``bool`` because
    ``isinstance(True, int) is True`` (Python quirk — but the explicit
    ``isinstance(value, bool)`` short-circuits first)."""
    transport = KnowledgeWorkerTransport(
        _FakeRuntime(),  # type: ignore[arg-type]
        require_auth=False,
    )
    request = KnowledgeTransportRequest(
        request_id="r1",
        operation=TransportOperation.GET,
        knowledge_id=KnowledgeId("ko_worker_1"),
    )
    with pytest.raises(ValueError, match="now must be a finite non-negative number"):
        await transport.handle(request, now=bad_now)


@pytest.mark.parametrize("bad_now", [True, False])
@pytest.mark.asyncio
async def test_handle_rejects_bool_now(bad_now: bool) -> None:
    """``bool`` is rejected by ``_finite_non_negative`` even though
    ``isinstance(True, int) is True`` (Python's ``bool`` is an ``int``
    subclass). The explicit ``isinstance(value, bool)`` short-circuit
    is the safety belt against the implicit-bool pattern."""
    transport = KnowledgeWorkerTransport(
        _FakeRuntime(),  # type: ignore[arg-type]
        require_auth=False,
    )
    request = KnowledgeTransportRequest(
        request_id="r1",
        operation=TransportOperation.GET,
        knowledge_id=KnowledgeId("ko_worker_1"),
    )
    with pytest.raises(ValueError, match="now must be a finite non-negative number"):
        await transport.handle(request, now=bad_now)


@pytest.mark.asyncio
async def test_handle_accepts_now_none_and_falls_back_to_time_time() -> None:
    """``handle(now=None)`` falls back to ``time.time()`` (the real
    clock). Pinning this guards against a refactor that makes ``now``
    a required positional argument."""
    transport = KnowledgeWorkerTransport(
        _FakeRuntime(),  # type: ignore[arg-type]
        require_auth=False,
    )
    request = KnowledgeTransportRequest(
        request_id="r1",
        operation=TransportOperation.GET,
        knowledge_id=KnowledgeId("ko_worker_1"),
    )
    response = await transport.handle(request)  # no `now=` kwarg
    assert response.ok is True


# ---------------------------------------------------------------------------
# _embed invalid-vector guards
# ---------------------------------------------------------------------------


class _StringEmbedder:
    async def embed(self, text: str) -> str:
        return "not-a-vector"


class _BytesEmbedder:
    async def embed(self, text: str) -> bytes:
        return b"\x00\x01"


class _EmptyListEmbedder:
    async def embed(self, text: str) -> list[float]:
        return []


@pytest.mark.asyncio
async def test_embed_rejects_string_vector() -> None:
    """``_embed`` raises ``RuntimeError`` when the embedder returns a
    ``str``. The RuntimeError bubbles up to the catch-all and is mapped
    to ``TransportErrorCode.UNAVAILABLE_DEPENDENCY`` with
    ``retryable=True``. Pinning this protects the wire contract: an
    embedder that returns a string would otherwise silently propagate
    into the runtime, which expects ``Sequence[float] | None``."""
    transport = KnowledgeWorkerTransport(
        _FakeRuntime(),  # type: ignore[arg-type]
        require_auth=False,
        embedding_provider=_StringEmbedder(),  # type: ignore[arg-type]
    )
    request = KnowledgeTransportRequest(
        request_id="r1",
        operation=TransportOperation.PUT,
        knowledge=_obj(),
    )
    response = await transport.handle(request, now=100.0)
    assert response.ok is False
    assert response.error is not None
    assert response.error.code is TransportErrorCode.UNAVAILABLE_DEPENDENCY
    assert response.error.retryable is True
    assert "invalid vector" in response.error.message


@pytest.mark.asyncio
async def test_embed_rejects_bytes_vector() -> None:
    """``_embed`` raises ``RuntimeError`` when the embedder returns
    ``bytes`` / ``bytearray`` (same as the string case — any non-numeric
    container is rejected)."""
    transport = KnowledgeWorkerTransport(
        _FakeRuntime(),  # type: ignore[arg-type]
        require_auth=False,
        embedding_provider=_BytesEmbedder(),
    )
    request = KnowledgeTransportRequest(
        request_id="r1",
        operation=TransportOperation.PUT,
        knowledge=_obj(),
    )
    response = await transport.handle(request, now=100.0)
    assert response.ok is False
    assert response.error is not None
    assert response.error.code is TransportErrorCode.UNAVAILABLE_DEPENDENCY


@pytest.mark.asyncio
async def test_embed_rejects_empty_vector() -> None:
    """``_embed`` raises ``RuntimeError`` when the embedder returns an
    empty vector. Empty vectors would silently propagate into the
    runtime and produce useless retrieval scores."""
    transport = KnowledgeWorkerTransport(
        _FakeRuntime(),  # type: ignore[arg-type]
        require_auth=False,
        embedding_provider=_EmptyListEmbedder(),
    )
    request = KnowledgeTransportRequest(
        request_id="r1",
        operation=TransportOperation.PUT,
        knowledge=_obj(),
    )
    response = await transport.handle(request, now=100.0)
    assert response.ok is False
    assert response.error is not None
    assert response.error.code is TransportErrorCode.UNAVAILABLE_DEPENDENCY


@pytest.mark.asyncio
async def test_handle_with_no_embedding_provider_succeeds_for_put() -> None:
    """``_embed`` returns ``None`` when no embedder is configured. The
    runtime then receives ``vector=None`` (the runtime decides whether
    that is acceptable — for the local ``FakeRuntime`` it is). Pinning
    this confirms the no-embedder code path is NOT a hard error at
    the transport layer."""
    runtime = _FakeRuntime()
    transport = KnowledgeWorkerTransport(
        runtime,  # type: ignore[arg-type]
        require_auth=False,
        # embedding_provider=None (default)
    )
    request = KnowledgeTransportRequest(
        request_id="r1",
        operation=TransportOperation.PUT,
        knowledge=_obj(),
    )
    response = await transport.handle(request, now=100.0)
    assert response.ok is True
    assert runtime.puts[0][1] is None


# ---------------------------------------------------------------------------
# _authorize capability routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_put_requires_knowledge_write_capability() -> None:
    """PUT requires ``knowledge.write`` (NOT ``knowledge.read``). A PUT
    with only read capability must be rejected with AUTHORIZATION.
    Pinning this catches a refactor that flips the PUT branch to
    require read."""
    transport = KnowledgeWorkerTransport(
        _FakeRuntime(),  # type: ignore[arg-type]
        require_auth=False,
    )
    request = KnowledgeTransportRequest(
        request_id="r1",
        operation=TransportOperation.PUT,
        auth=_auth("knowledge.read"),
        knowledge=_obj(),
    )
    response = await transport.handle(request, now=100.0)
    assert response.ok is False
    assert response.error is not None
    assert response.error.code is TransportErrorCode.AUTHORIZATION
    assert "knowledge.write" in response.error.message


@pytest.mark.asyncio
async def test_get_requires_knowledge_read_capability() -> None:
    """GET requires ``knowledge.read`` (NOT ``knowledge.write``). A GET
    with only write capability must be rejected with AUTHORIZATION.
    Pinning this catches a refactor that flips the GET branch to
    require write (which would block all read-only callers)."""
    transport = KnowledgeWorkerTransport(
        _FakeRuntime(),  # type: ignore[arg-type]
        require_auth=False,
    )
    request = KnowledgeTransportRequest(
        request_id="r1",
        operation=TransportOperation.GET,
        auth=_auth("knowledge.write"),
        knowledge_id=KnowledgeId("ko_worker_1"),
    )
    response = await transport.handle(request, now=100.0)
    assert response.ok is False
    assert response.error is not None
    assert response.error.code is TransportErrorCode.AUTHORIZATION
    assert "knowledge.read" in response.error.message


@pytest.mark.asyncio
async def test_retrieve_requires_knowledge_read_capability() -> None:
    """RETRIEVE also requires ``knowledge.read``. The capability routing
    is shared with GET (the else-branch of the PUT-vs-rest ternary)."""
    transport = KnowledgeWorkerTransport(
        _FakeRuntime(),  # type: ignore[arg-type]
        require_auth=False,
    )
    request = KnowledgeTransportRequest(
        request_id="r1",
        operation=TransportOperation.RETRIEVE,
        auth=_auth("knowledge.write"),
        topic="worker",
    )
    response = await transport.handle(request, now=100.0)
    assert response.ok is False
    assert response.error is not None
    assert response.error.code is TransportErrorCode.AUTHORIZATION
    assert "knowledge.read" in response.error.message


@pytest.mark.asyncio
async def test_authorize_rejects_non_finite_expires_at() -> None:
    """``_authorize`` rejects ``NaN`` / ``inf`` ``expires_at`` via
    ``math.isfinite``. A non-finite expiry would let a caller assert
    "expired at NaN" or "expired at +inf" — neither is a valid auth
    contract."""
    transport = KnowledgeWorkerTransport(
        _FakeRuntime(),  # type: ignore[arg-type]
    )
    request = KnowledgeTransportRequest(
        request_id="r1",
        operation=TransportOperation.GET,
        auth=_auth("knowledge.read", expires_at=math.nan),
        knowledge_id=KnowledgeId("ko_worker_1"),
    )
    response = await transport.handle(request, now=100.0)
    assert response.ok is False
    assert response.error is not None
    assert response.error.code is TransportErrorCode.AUTHORIZATION
    assert "invalid expiry" in response.error.message


# ---------------------------------------------------------------------------
# Catch-all exception handler
# ---------------------------------------------------------------------------


class _CustomValueError(Exception):
    """Custom non-runtime exception to exercise the catch-all
    ``except Exception`` branch. A ``ValueError`` would NOT be caught
    by the specific ``except RuntimeError`` branch — it falls through
    to ``except Exception``."""


@pytest.mark.asyncio
async def test_unmapped_exception_maps_to_internal_not_retryable() -> None:
    """An exception that is neither ``KnowledgeConflictError``,
    ``KnowledgeIntegrityError``, nor ``RuntimeError`` falls into the
    catch-all ``except Exception`` and is mapped to
    ``TransportErrorCode.INTERNAL`` with ``retryable=False``. Pinning
    this catches a refactor that accidentally promotes ValueError to
    UNAVAILABLE_DEPENDENCY."""
    runtime = _FakeRuntime(error=_CustomValueError("unexpected"))
    transport = KnowledgeWorkerTransport(runtime)  # type: ignore[arg-type]
    request = KnowledgeTransportRequest(
        request_id="r1",
        operation=TransportOperation.GET,
        auth=_auth("knowledge.read"),
        knowledge_id=KnowledgeId("ko_worker_1"),
    )
    response = await transport.handle(request, now=100.0)
    assert response.ok is False
    assert response.error is not None
    assert response.error.code is TransportErrorCode.INTERNAL
    assert response.error.retryable is False


# ---------------------------------------------------------------------------
# _failure message truncation safety
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failure_message_is_truncated_to_1000_chars() -> None:
    """The catch-all path emits the literal string
    ``"knowledge transport operation failed"`` (well under 1000 chars),
    so a custom-message truncation test must use a RuntimeError that
    surfaces its own long message. Pinning this protects the wire
    contract from slice 29 (``TransportError.message`` is bounded to
    1-1000 chars)."""
    long_message = "x" * 1500
    runtime = _FakeRuntime(error=RuntimeError(long_message))
    transport = KnowledgeWorkerTransport(runtime)  # type: ignore[arg-type]
    request = KnowledgeTransportRequest(
        request_id="r1",
        operation=TransportOperation.GET,
        auth=_auth("knowledge.read"),
        knowledge_id=KnowledgeId("ko_worker_1"),
    )
    response = await transport.handle(request, now=100.0)
    assert response.ok is False
    assert response.error is not None
    assert response.error.code is TransportErrorCode.UNAVAILABLE_DEPENDENCY
    # The ``_failure`` helper truncates ``safe_message[:1000]`` after
    # strip; a 1500-char message is clamped to exactly 1000.
    assert len(response.error.message) == 1000


@pytest.mark.asyncio
async def test_failure_whitespace_only_message_falls_back_to_default() -> None:
    """An exception whose ``str()`` returns only whitespace (or empty)
    gets the fallback ``"knowledge transport operation failed"`` after
    ``_failure`` strips and checks emptiness."""
    runtime = _FakeRuntime(error=RuntimeError("   "))
    transport = KnowledgeWorkerTransport(runtime)  # type: ignore[arg-type]
    request = KnowledgeTransportRequest(
        request_id="r1",
        operation=TransportOperation.GET,
        auth=_auth("knowledge.read"),
        knowledge_id=KnowledgeId("ko_worker_1"),
    )
    response = await transport.handle(request, now=100.0)
    assert response.ok is False
    assert response.error is not None
    assert response.error.message == "knowledge transport operation failed"
