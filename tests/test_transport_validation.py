"""Pin the outer guard rails of ``oai2.knowledge.transport``.

The behavioural tests in ``tests/test_knowledge_transport.py`` cover the
happy path for each request envelope, the version-rejection guard, the
operation-specific required-field checks, and the success-vs-failed
response invariants. They do NOT pin:

* enum stability for ``TransportOperation`` (3 values) and
  ``TransportErrorCode`` (7 values) — a future rename or removal of
  either silently breaks the wire contract that any future live
  Cloudflare Worker adapter must implement;
* ``TRANSPORT_VERSION`` literal — the version string the request and
  response envelopes MUST carry (and the validation that rejects any
  other version);
* ``KnowledgeTransportRequest`` negative paths beyond "missing the
  operation-specific required field" and "unsupported version" — the
  PUT-must-not-set-knowledge_id/topic, GET-must-not-set-knowledge/topic,
  and RETRIEVE-must-not-set-knowledge/knowledge_id contracts are not
  pinned; whitespace-only ``knowledge_id`` / ``topic`` slips through;
  field length / range / ``extra="forbid"`` not pinned;
* the request envelope defaults (``limit=8``, ``min_authority=0.0``,
  ``include_status=(IMPLEMENTED, EXPERIMENTAL)``) — a refactor that
  flips the default ``limit`` from 8 to 50 silently changes the
  wire-shaped query budget;
* ``TransportAuthContext`` field bounds (``subject`` 1-256 chars),
  empty-rejection, and ``extra="forbid"`` — the auth context is the
  *only* identity surface the wire contract carries, so a too-loose
  ``subject`` would silently let any string reach the worker;
* ``TransportError`` ``message`` length bound (1-1000) — protects
  workers from attacker-controlled unbounded error messages;
* ``KnowledgeTransportResponse`` ``corpus_revision`` range
  (``>= 0``), ``extra="forbid"``, and the response version-pin;
* ``D1KnowledgeIndexRecord`` field bounds and ``extra="forbid"`` —
  the D1 row shape must reject malformed content_hash / topic /
  authority values BEFORE persistence;
* ``R2BodyDescriptor`` field bounds and ``extra="forbid"`` — protects
  the R2 bucket key derivation from bad inputs;
* ``VectorizeMetadata`` field bounds and ``extra="forbid"`` — protects
  the embedding sidecar metadata from unknown fields;
* ``KnowledgeCacheRef`` length bounds and ``extra="forbid"``;
* ``QueryCacheEnvelope`` field bounds, ``version`` pin, and
  ``refs`` default-to-empty list.

The wire contract is a permanent contract (a future live Worker must
match it byte-for-byte). Pinning all of these ensures a refactor that
introduces a backward-incompatible change fails in CI immediately.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from oai2.core import KnowledgeId, Status
from oai2.knowledge import KnowledgeObject, sha256_hex
from oai2.knowledge.transport import (
    TRANSPORT_VERSION,
    D1KnowledgeIndexRecord,
    KnowledgeCacheRef,
    KnowledgeTransportRequest,
    KnowledgeTransportResponse,
    QueryCacheEnvelope,
    R2BodyDescriptor,
    TransportAuthContext,
    TransportError,
    TransportErrorCode,
    TransportOperation,
    VectorizeMetadata,
)

# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def _obj(content: str = "verified knowledge") -> KnowledgeObject:
    return KnowledgeObject(
        knowledge_id=KnowledgeId("ko_transport_1"),
        topic="transport",
        content=content,
        content_hash=sha256_hex(content),
        source_uri="https://example.test/source",
        retrieved_at=123.0,
        authority=0.9,
        status=Status.EXPERIMENTAL,
    )


def _put_request(**overrides: object) -> KnowledgeTransportRequest:
    base: dict[str, object] = {
        "request_id": "r1",
        "operation": TransportOperation.PUT,
        "knowledge": _obj(),
    }
    base.update(overrides)
    return KnowledgeTransportRequest(**base)  # type: ignore[arg-type]


def _get_request(**overrides: object) -> KnowledgeTransportRequest:
    base: dict[str, object] = {
        "request_id": "r1",
        "operation": TransportOperation.GET,
        "knowledge_id": KnowledgeId("ko_1"),
    }
    base.update(overrides)
    return KnowledgeTransportRequest(**base)  # type: ignore[arg-type]


def _retrieve_request(**overrides: object) -> KnowledgeTransportRequest:
    base: dict[str, object] = {
        "request_id": "r1",
        "operation": TransportOperation.RETRIEVE,
        "topic": "agent-architecture",
    }
    base.update(overrides)
    return KnowledgeTransportRequest(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TransportOperation / TransportErrorCode enum stability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "operation, expected_value",
    [
        (TransportOperation.PUT, "put"),
        (TransportOperation.GET, "get"),
        (TransportOperation.RETRIEVE, "retrieve"),
    ],
)
def test_transport_operation_enum_values_are_stable_strings(
    operation: TransportOperation,
    expected_value: str,
) -> None:
    """``TransportOperation`` is a ``StrEnum``; the string form is the
    wire contract for the ``operation`` field in
    ``KnowledgeTransportRequest``."""
    assert operation == expected_value
    assert str(operation) == expected_value
    assert operation.value == expected_value


def test_transport_operation_has_exactly_three_distinct_values() -> None:
    """Closing the surface: the operation taxonomy is exactly
    PUT / GET / RETRIEVE. Adding a fourth (e.g. ``DELETE``) would
    change the wire contract."""
    assert len(set(TransportOperation)) == 3


@pytest.mark.parametrize(
    "code, expected_value",
    [
        (TransportErrorCode.AUTHORIZATION, "authorization"),
        (TransportErrorCode.VALIDATION, "validation"),
        (TransportErrorCode.NOT_FOUND, "not_found"),
        (TransportErrorCode.CONFLICT, "conflict"),
        (TransportErrorCode.UNAVAILABLE_DEPENDENCY, "unavailable_dependency"),
        (TransportErrorCode.INTEGRITY, "integrity"),
        (TransportErrorCode.INTERNAL, "internal"),
    ],
)
def test_transport_error_code_enum_values_are_stable_strings(
    code: TransportErrorCode,
    expected_value: str,
) -> None:
    """``TransportErrorCode`` is a ``StrEnum``; the string form is the
    wire contract for the ``error.code`` field."""
    assert code == expected_value
    assert str(code) == expected_value
    assert code.value == expected_value


def test_transport_error_code_has_exactly_seven_distinct_values() -> None:
    """Closing the surface: the error-code taxonomy is exactly 7 codes
    (auth, validation, not_found, conflict, unavailable_dependency,
    integrity, internal). A future refactor that collapses these (e.g.
    folding NOT_FOUND and CONFLICT into VALIDATION) breaks the wire
    contract for live workers that branch on ``error.code``."""
    assert len(set(TransportErrorCode)) == 7


# ---------------------------------------------------------------------------
# TRANSPORT_VERSION literal
# ---------------------------------------------------------------------------


def test_transport_version_literal_is_pinned() -> None:
    """The wire contract version is ``"1"``. A bump to ``"2"`` would be
    a coordinated release (request and response envelopes both update
    their ``TRANSPORT_VERSION`` and the ``validate_contract`` guards).
    Pinning the literal prevents silent drift."""
    assert TRANSPORT_VERSION == "1"


# ---------------------------------------------------------------------------
# KnowledgeTransportRequest defaults
# ---------------------------------------------------------------------------


def test_request_defaults_limit_is_8() -> None:
    """Default ``limit=8`` is the wire-shaped query budget. A refactor
    that flips the default to e.g. 50 silently changes the response
    payload size for every RETRIEVE call."""
    req = _retrieve_request()
    assert req.limit == 8


def test_request_defaults_min_authority_is_0() -> None:
    """Default ``min_authority=0.0`` matches the public InMemoryKnowledgeStore
    default. A flip to 0.5 would silently exclude low-authority
    knowledge from RETRIEVE results."""
    req = _retrieve_request()
    assert req.min_authority == 0.0


def test_request_defaults_include_status_is_implemented_and_experimental() -> None:
    """Default ``include_status=(IMPLEMENTED, EXPERIMENTAL)`` matches
    the InMemoryKnowledgeStore default. A flip that adds PROPOSED
    would silently widen the visible status set."""
    req = _retrieve_request()
    assert req.include_status == (Status.IMPLEMENTED, Status.EXPERIMENTAL)


def test_request_defaults_version_is_transport_version() -> None:
    """Every request must carry ``version=TRANSPORT_VERSION`` by
    default. The validate_contract guard rejects any other value."""
    req = _retrieve_request()
    assert req.version == TRANSPORT_VERSION


def test_request_defaults_auth_is_none() -> None:
    """Default ``auth=None`` (the worker side enforces auth, not the
    envelope). A refactor that adds an ``auth`` requirement at the
    envelope level would break tests that construct a request
    without an auth context."""
    req = _retrieve_request()
    assert req.auth is None


# ---------------------------------------------------------------------------
# KnowledgeTransportRequest field ranges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_limit", [0, -1, 201, 1000])
def test_request_rejects_limit_outside_1_to_200_range(bad_limit: int) -> None:
    """``limit`` is bounded to ``[1, 200]``. A RETRIEVE request that
    asks for 0 / negative / > 200 objects is a malformed wire
    contract."""
    with pytest.raises(ValidationError, match="limit"):
        _retrieve_request(limit=bad_limit)


@pytest.mark.parametrize("good_limit", [1, 8, 100, 200])
def test_request_accepts_limit_inside_1_to_200_range(good_limit: int) -> None:
    """Boundary-positive: limits at the inclusive edges (1 and 200)
    are accepted. A future ``Field(le=100)`` change would break
    the 200 test immediately."""
    req = _retrieve_request(limit=good_limit)
    assert req.limit == good_limit


@pytest.mark.parametrize("bad_min_authority", [-0.01, -1.0, 1.01, 2.0])
def test_request_rejects_min_authority_outside_0_to_1_range(
    bad_min_authority: float,
) -> None:
    """``min_authority`` is bounded to ``[0.0, 1.0]``. Out-of-range
    values would let a worker under- or over-filter the corpus."""
    with pytest.raises(ValidationError, match="min_authority"):
        _retrieve_request(min_authority=bad_min_authority)


@pytest.mark.parametrize("good_min_authority", [0.0, 0.5, 1.0])
def test_request_accepts_min_authority_inside_0_to_1_range(
    good_min_authority: float,
) -> None:
    """Boundary-positive: 0.0 and 1.0 are accepted (inclusive)."""
    req = _retrieve_request(min_authority=good_min_authority)
    assert req.min_authority == good_min_authority


def test_request_rejects_empty_request_id() -> None:
    """``request_id`` is required (min_length=1, max_length=128). An
    empty string would let a worker silently accept requests with
    no traceable ID."""
    with pytest.raises(ValidationError, match="request_id"):
        KnowledgeTransportRequest(
            request_id="",
            operation=TransportOperation.GET,
            knowledge_id=KnowledgeId("ko_1"),
        )


def test_request_rejects_oversized_request_id() -> None:
    """``request_id`` is bounded to 128 chars. The wire contract does
    not allow unbounded IDs."""
    with pytest.raises(ValidationError, match="request_id"):
        KnowledgeTransportRequest(
            request_id="x" * 129,
            operation=TransportOperation.GET,
            knowledge_id=KnowledgeId("ko_1"),
        )


def test_request_rejects_oversized_topic() -> None:
    """``topic`` is bounded to 256 chars (matches ``KnowledgeObject.topic``
    upper bound). Out-of-bound topics would let a worker accept data
    that the storage layer rejects."""
    with pytest.raises(ValidationError, match="topic"):
        _retrieve_request(topic="t" * 257)


# ---------------------------------------------------------------------------
# KnowledgeTransportRequest operation-specific negative paths
# ---------------------------------------------------------------------------


def test_put_request_rejects_extra_knowledge_id() -> None:
    """PUT must set ``knowledge`` only — not ``knowledge_id`` or
    ``topic``. A PUT with ``knowledge_id`` is ambiguous (the worker
    cannot decide whether to upsert or insert)."""
    with pytest.raises(ValidationError, match="must not set knowledge_id/topic"):
        _put_request(knowledge_id=KnowledgeId("ko_extra"))


def test_put_request_rejects_extra_topic() -> None:
    """PUT must not set ``topic``. Topics are derived from the
    embedded ``knowledge`` object."""
    with pytest.raises(ValidationError, match="must not set knowledge_id/topic"):
        _put_request(topic="extra-topic")


def test_get_request_rejects_extra_knowledge() -> None:
    """GET must set ``knowledge_id`` only — not ``knowledge`` or
    ``topic``. A GET with ``knowledge`` is redundant (the worker
    looks up by ID)."""
    with pytest.raises(ValidationError, match="must not set knowledge/topic"):
        _get_request(knowledge=_obj())


def test_get_request_rejects_extra_topic() -> None:
    """GET must not set ``topic``. Lookup is by ID, not topic."""
    with pytest.raises(ValidationError, match="must not set knowledge/topic"):
        _get_request(topic="extra-topic")


def test_get_request_rejects_whitespace_only_knowledge_id() -> None:
    """GET's ``knowledge_id`` must be non-empty after strip. A
    whitespace-only ID would slip past ``min_length=1`` if the
    worker does not strip."""
    with pytest.raises(ValidationError, match="get requires knowledge_id"):
        _get_request(knowledge_id=KnowledgeId("   "))


def test_retrieve_request_rejects_extra_knowledge() -> None:
    """RETRIEVE must set ``topic`` only — not ``knowledge`` or
    ``knowledge_id``."""
    with pytest.raises(ValidationError, match="must not set knowledge/knowledge_id"):
        _retrieve_request(knowledge=_obj())


def test_retrieve_request_rejects_extra_knowledge_id() -> None:
    """RETRIEVE must not set ``knowledge_id``. A RETRIEVE with a
    pre-known ID would not need a topic-based search."""
    with pytest.raises(ValidationError, match="must not set knowledge/knowledge_id"):
        _retrieve_request(knowledge_id=KnowledgeId("ko_extra"))


def test_retrieve_request_rejects_whitespace_only_topic() -> None:
    """RETRIEVE's ``topic`` must be non-empty after strip. A
    whitespace-only topic would let a worker return an
    unbounded scan."""
    with pytest.raises(ValidationError, match="retrieve requires topic"):
        _retrieve_request(topic="   ")


def test_request_rejects_unsupported_version_string() -> None:
    """Any ``version`` other than ``TRANSPORT_VERSION`` must be
    rejected. Pinning this prevents a worker from being asked to
    parse a future wire format it does not understand."""
    with pytest.raises(ValidationError, match="unsupported knowledge transport version"):
        _get_request(version="2")


def test_request_extra_forbid_rejects_unknown_field() -> None:
    """``extra="forbid"`` on ``KnowledgeTransportRequest`` rejects any
    unknown field. A typo in a future field name (e.g. ``minAuthoritiy``)
    would otherwise be silently ignored."""
    with pytest.raises(ValidationError, match="unknown_field"):
        _retrieve_request(unknown_field="oops")


# ---------------------------------------------------------------------------
# TransportAuthContext
# ---------------------------------------------------------------------------


def test_auth_context_rejects_empty_subject() -> None:
    """``subject`` must be non-empty (min_length=1). An empty subject
    would let a worker receive an authenticated identity with no
    subject, defeating the auth context's purpose."""
    with pytest.raises(ValidationError, match="subject"):
        TransportAuthContext(subject="")


def test_auth_context_rejects_oversized_subject() -> None:
    """``subject`` is bounded to 256 chars."""
    with pytest.raises(ValidationError, match="subject"):
        TransportAuthContext(subject="s" * 257)


def test_auth_context_default_capabilities_is_empty_tuple() -> None:
    """Default ``capabilities=()`` — an authenticated subject with no
    explicit capabilities is capability-less, NOT capability-everything."""
    auth = TransportAuthContext(subject="local-user")
    assert auth.capabilities == ()


def test_auth_context_default_expires_at_is_none() -> None:
    """Default ``expires_at=None`` — the auth context is unbounded by
    default; expiry is the worker's responsibility."""
    auth = TransportAuthContext(subject="local-user")
    assert auth.expires_at is None


def test_auth_context_extra_forbid_rejects_unknown_field() -> None:
    """``extra="forbid"`` on ``TransportAuthContext`` rejects any
    unknown field. The auth context is the only identity surface on
    the wire, so an unknown field would let callers smuggle data
    past it."""
    with pytest.raises(ValidationError, match="unknown_field"):
        TransportAuthContext(subject="local-user", unknown_field="oops")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# TransportError
# ---------------------------------------------------------------------------


def test_transport_error_rejects_empty_message() -> None:
    """``message`` must be non-empty (min_length=1). An empty error
    message would force workers to surface a contextless failure."""
    with pytest.raises(ValidationError, match="message"):
        TransportError(code=TransportErrorCode.INTERNAL, message="")


def test_transport_error_rejects_oversized_message() -> None:
    """``message`` is bounded to 1000 chars. Unbounded error messages
    would let a caller control unbounded bytes in worker logs."""
    with pytest.raises(ValidationError, match="message"):
        TransportError(code=TransportErrorCode.INTERNAL, message="x" * 1001)


def test_transport_error_default_retryable_is_false() -> None:
    """Default ``retryable=False`` — by default, an error is terminal,
    not transient. A refactor that flips the default would silently
    change the retry semantics for every error code."""
    err = TransportError(code=TransportErrorCode.INTERNAL, message="boom")
    assert err.retryable is False


def test_transport_error_extra_forbid_rejects_unknown_field() -> None:
    """``extra="forbid"`` on ``TransportError`` rejects unknown fields.
    Errors are the wire's diagnostic surface; unknown fields would
    smuggle data past the worker's error log filter."""
    with pytest.raises(ValidationError, match="unknown_field"):
        TransportError(  # type: ignore[call-arg]
            code=TransportErrorCode.INTERNAL,
            message="boom",
            unknown_field="oops",
        )


# ---------------------------------------------------------------------------
# KnowledgeTransportResponse
# ---------------------------------------------------------------------------


def test_response_rejects_unsupported_version_string() -> None:
    """The response envelope pins ``version=TRANSPORT_VERSION`` the
    same way the request does. A response carrying ``"2"`` would be
    silently ignored by a worker expecting version ``"1"``."""
    with pytest.raises(ValidationError, match="unsupported knowledge transport version"):
        KnowledgeTransportResponse(
            version="2",
            request_id="r1",
            ok=True,
        )


def test_response_rejects_negative_corpus_revision() -> None:
    """``corpus_revision`` is bounded to ``>= 0``. A negative corpus
    revision is a malformed wire contract (corpus revisions are
    monotonically non-negative)."""
    with pytest.raises(ValidationError, match="corpus_revision"):
        KnowledgeTransportResponse(
            request_id="r1",
            ok=True,
            corpus_revision=-1,
        )


def test_failed_response_with_knowledge_object_is_rejected() -> None:
    """A failed response (ok=False) carrying a single ``knowledge``
    object is a wire contract violation — failures carry only
    ``error`` and never bodies."""
    with pytest.raises(ValidationError, match="failed response must not contain knowledge"):
        KnowledgeTransportResponse(
            request_id="r1",
            ok=False,
            knowledge=_obj(),
            error=TransportError(
                code=TransportErrorCode.NOT_FOUND,
                message="missing",
            ),
        )


def test_response_extra_forbid_rejects_unknown_field() -> None:
    """``extra="forbid"`` on ``KnowledgeTransportResponse`` rejects
    any unknown field. The response envelope is the canonical
    payload; unknown fields would let callers smuggle data past
    the worker."""
    with pytest.raises(ValidationError, match="unknown_field"):
        KnowledgeTransportResponse(  # type: ignore[call-arg]
            request_id="r1",
            ok=True,
            unknown_field="oops",
        )


# ---------------------------------------------------------------------------
# D1KnowledgeIndexRecord
# ---------------------------------------------------------------------------


def test_d1_record_rejects_empty_topic() -> None:
    """``topic`` must be non-empty (matches the
    ``KnowledgeObject.topic`` min_length=1)."""
    with pytest.raises(ValidationError, match="topic"):
        D1KnowledgeIndexRecord(
            knowledge_id=KnowledgeId("ko_1"),
            topic="",
            content_hash="0123456789abcdef",
            authority=0.5,
            status=Status.EXPERIMENTAL,
            retrieved_at=1.0,
            corpus_revision=0,
        )


def test_d1_record_rejects_oversized_topic() -> None:
    """``topic`` is bounded to 256 chars (matches KnowledgeObject)."""
    with pytest.raises(ValidationError, match="topic"):
        D1KnowledgeIndexRecord(
            knowledge_id=KnowledgeId("ko_1"),
            topic="t" * 257,
            content_hash="0123456789abcdef",
            authority=0.5,
            status=Status.EXPERIMENTAL,
            retrieved_at=1.0,
            corpus_revision=0,
        )


@pytest.mark.parametrize("bad_content_hash_length", [0, 1, 7, 129, 256])
def test_d1_record_rejects_content_hash_outside_8_to_128_range(
    bad_content_hash_length: int,
) -> None:
    """``content_hash`` is bounded to 8-128 chars. Below 8 chars is
    too short to be a content hash; above 128 chars is not a
    canonical hex digest."""
    with pytest.raises(ValidationError, match="content_hash"):
        D1KnowledgeIndexRecord(
            knowledge_id=KnowledgeId("ko_1"),
            topic="t",
            content_hash="x" * bad_content_hash_length,
            authority=0.5,
            status=Status.EXPERIMENTAL,
            retrieved_at=1.0,
            corpus_revision=0,
        )


@pytest.mark.parametrize("good_content_hash_length", [8, 16, 64, 128])
def test_d1_record_accepts_content_hash_inside_8_to_128_range(
    good_content_hash_length: int,
) -> None:
    """Boundary-positive: 8 and 128 are accepted (inclusive)."""
    rec = D1KnowledgeIndexRecord(
        knowledge_id=KnowledgeId("ko_1"),
        topic="t",
        content_hash="x" * good_content_hash_length,
        authority=0.5,
        status=Status.EXPERIMENTAL,
        retrieved_at=1.0,
        corpus_revision=0,
    )
    assert len(rec.content_hash) == good_content_hash_length


@pytest.mark.parametrize("bad_authority", [-0.01, -1.0, 1.01, 2.0])
def test_d1_record_rejects_authority_outside_0_to_1_range(
    bad_authority: float,
) -> None:
    """``authority`` is bounded to ``[0.0, 1.0]``. Out-of-range
    values are malformed wire contracts."""
    with pytest.raises(ValidationError, match="authority"):
        D1KnowledgeIndexRecord(
            knowledge_id=KnowledgeId("ko_1"),
            topic="t",
            content_hash="0123456789abcdef",
            authority=bad_authority,
            status=Status.EXPERIMENTAL,
            retrieved_at=1.0,
            corpus_revision=0,
        )


def test_d1_record_rejects_negative_corpus_revision() -> None:
    """``corpus_revision`` is bounded to ``>= 0``. A negative
    revision is malformed."""
    with pytest.raises(ValidationError, match="corpus_revision"):
        D1KnowledgeIndexRecord(
            knowledge_id=KnowledgeId("ko_1"),
            topic="t",
            content_hash="0123456789abcdef",
            authority=0.5,
            status=Status.EXPERIMENTAL,
            retrieved_at=1.0,
            corpus_revision=-1,
        )


def test_d1_record_optional_fields_default_to_none() -> None:
    """``source_uri``, ``r2_blob_key``, ``vectorize_id`` default to
    ``None``. The D1 index record is sparse: only ``knowledge_id``,
    ``topic``, ``content_hash``, ``authority``, ``status``,
    ``retrieved_at``, ``corpus_revision`` are required."""
    rec = D1KnowledgeIndexRecord(
        knowledge_id=KnowledgeId("ko_1"),
        topic="t",
        content_hash="0123456789abcdef",
        authority=0.5,
        status=Status.EXPERIMENTAL,
        retrieved_at=1.0,
        corpus_revision=0,
    )
    assert rec.source_uri is None
    assert rec.r2_blob_key is None
    assert rec.vectorize_id is None


def test_d1_record_extra_forbid_rejects_unknown_field() -> None:
    """``extra="forbid"`` on ``D1KnowledgeIndexRecord`` rejects
    unknown fields. The D1 row shape is the authoritative D1
    schema; unknown fields would corrupt the index."""
    with pytest.raises(ValidationError, match="unknown_field"):
        D1KnowledgeIndexRecord(  # type: ignore[call-arg]
            knowledge_id=KnowledgeId("ko_1"),
            topic="t",
            content_hash="0123456789abcdef",
            authority=0.5,
            status=Status.EXPERIMENTAL,
            retrieved_at=1.0,
            corpus_revision=0,
            unknown_field="oops",
        )


# ---------------------------------------------------------------------------
# R2BodyDescriptor
# ---------------------------------------------------------------------------


def test_r2_descriptor_rejects_short_content_hash() -> None:
    """``content_hash`` is bounded to 8-128 chars (matches D1)."""
    with pytest.raises(ValidationError, match="content_hash"):
        R2BodyDescriptor(
            knowledge_id=KnowledgeId("ko_1"),
            content_hash="short",
            object_key="oai2-blobs/x",
            size_bytes=0,
        )


def test_r2_descriptor_rejects_empty_object_key() -> None:
    """``object_key`` is bounded to 1-512 chars and must be
    non-empty."""
    with pytest.raises(ValidationError, match="object_key"):
        R2BodyDescriptor(
            knowledge_id=KnowledgeId("ko_1"),
            content_hash="0123456789abcdef",
            object_key="",
            size_bytes=0,
        )


def test_r2_descriptor_rejects_oversized_object_key() -> None:
    """``object_key`` is bounded to 512 chars. Unbounded keys
    would let attackers fill the R2 bucket with pathologically
    long names."""
    with pytest.raises(ValidationError, match="object_key"):
        R2BodyDescriptor(
            knowledge_id=KnowledgeId("ko_1"),
            content_hash="0123456789abcdef",
            object_key="o" * 513,
            size_bytes=0,
        )


def test_r2_descriptor_rejects_negative_size_bytes() -> None:
    """``size_bytes`` is bounded to ``>= 0``. Negative sizes are
    malformed wire contracts."""
    with pytest.raises(ValidationError, match="size_bytes"):
        R2BodyDescriptor(
            knowledge_id=KnowledgeId("ko_1"),
            content_hash="0123456789abcdef",
            object_key="oai2-blobs/x",
            size_bytes=-1,
        )


def test_r2_descriptor_extra_forbid_rejects_unknown_field() -> None:
    """``extra="forbid"`` on ``R2BodyDescriptor`` rejects unknown
    fields. R2 is the content-addressed body store; unknown
    fields would let callers smuggle metadata into the bucket."""
    with pytest.raises(ValidationError, match="unknown_field"):
        R2BodyDescriptor(  # type: ignore[call-arg]
            knowledge_id=KnowledgeId("ko_1"),
            content_hash="0123456789abcdef",
            object_key="oai2-blobs/x",
            size_bytes=0,
            unknown_field="oops",
        )


# ---------------------------------------------------------------------------
# VectorizeMetadata
# ---------------------------------------------------------------------------


def test_vectorize_metadata_rejects_authority_outside_0_to_1_range() -> None:
    """``authority`` is bounded to ``[0.0, 1.0]`` (matches the
    corpus contract)."""
    with pytest.raises(ValidationError, match="authority"):
        VectorizeMetadata(
            knowledge_id=KnowledgeId("ko_1"),
            content_hash="0123456789abcdef",
            status=Status.EXPERIMENTAL,
            authority=1.5,
            embedding_version="embed-v1",
        )


def test_vectorize_metadata_rejects_short_content_hash() -> None:
    """``content_hash`` is bounded to 8-128 chars (matches D1)."""
    with pytest.raises(ValidationError, match="content_hash"):
        VectorizeMetadata(
            knowledge_id=KnowledgeId("ko_1"),
            content_hash="short",
            status=Status.EXPERIMENTAL,
            authority=0.5,
            embedding_version="embed-v1",
        )


def test_vectorize_metadata_rejects_empty_embedding_version() -> None:
    """``embedding_version`` is bounded to 1-128 chars and must be
    non-empty. An empty version would let two different
    embeddings collide on the same sidecar key."""
    with pytest.raises(ValidationError, match="embedding_version"):
        VectorizeMetadata(
            knowledge_id=KnowledgeId("ko_1"),
            content_hash="0123456789abcdef",
            status=Status.EXPERIMENTAL,
            authority=0.5,
            embedding_version="",
        )


def test_vectorize_metadata_default_source_uri_is_none() -> None:
    """``source_uri`` defaults to ``None``. The vectorize sidecar
    is sparse; only provenance fields that are populated are
    present."""
    meta = VectorizeMetadata(
        knowledge_id=KnowledgeId("ko_1"),
        content_hash="0123456789abcdef",
        status=Status.EXPERIMENTAL,
        authority=0.5,
        embedding_version="embed-v1",
    )
    assert meta.source_uri is None


def test_vectorize_metadata_extra_forbid_rejects_unknown_field() -> None:
    """``extra="forbid"`` on ``VectorizeMetadata`` rejects unknown
    fields. Vectorize metadata is the embedding sidecar schema;
    unknown fields would corrupt the embedding index lookup."""
    with pytest.raises(ValidationError, match="unknown_field"):
        VectorizeMetadata(  # type: ignore[call-arg]
            knowledge_id=KnowledgeId("ko_1"),
            content_hash="0123456789abcdef",
            status=Status.EXPERIMENTAL,
            authority=0.5,
            embedding_version="embed-v1",
            unknown_field="oops",
        )


# ---------------------------------------------------------------------------
# KnowledgeCacheRef
# ---------------------------------------------------------------------------


def test_cache_ref_rejects_short_content_hash() -> None:
    """``content_hash`` is bounded to 8-128 chars (matches D1)."""
    with pytest.raises(ValidationError, match="content_hash"):
        KnowledgeCacheRef(
            knowledge_id=KnowledgeId("ko_1"),
            content_hash="short",
        )


def test_cache_ref_extra_forbid_rejects_unknown_field() -> None:
    """``extra="forbid"`` on ``KnowledgeCacheRef`` rejects unknown
    fields. Cache refs are content-addressed; unknown fields
    would invalidate the cache lookup."""
    with pytest.raises(ValidationError, match="unknown_field"):
        KnowledgeCacheRef(  # type: ignore[call-arg]
            knowledge_id=KnowledgeId("ko_1"),
            content_hash="0123456789abcdef",
            unknown_field="oops",
        )


# ---------------------------------------------------------------------------
# QueryCacheEnvelope
# ---------------------------------------------------------------------------


def test_query_cache_envelope_defaults_refs_to_empty_list() -> None:
    """``refs`` defaults to ``[]`` (an envelope with zero cache
    entries is valid; the worker still gets the
    corpus + embedding digest)."""
    env = QueryCacheEnvelope(
        corpus_revision=4,
        embedding_digest="embed-v1-digest",
        request_fingerprint="query-fingerprint",
    )
    assert env.refs == []


def test_query_cache_envelope_rejects_unsupported_version_string() -> None:
    """``QueryCacheEnvelope`` pins ``version=TRANSPORT_VERSION``
    via ``validate_version``. A future bump requires coordinated
    release of both the request envelope and the cache envelope."""
    with pytest.raises(ValidationError, match="unsupported knowledge transport version"):
        QueryCacheEnvelope(
            version="2",
            corpus_revision=4,
            embedding_digest="embed-v1-digest",
            request_fingerprint="query-fingerprint",
        )


def test_query_cache_envelope_rejects_empty_embedding_digest() -> None:
    """``embedding_digest`` is required (min_length=1). An empty
    digest would let two distinct embedding models collide on
    the same cache namespace."""
    with pytest.raises(ValidationError, match="embedding_digest"):
        QueryCacheEnvelope(
            corpus_revision=4,
            embedding_digest="",
            request_fingerprint="query-fingerprint",
        )


def test_query_cache_envelope_rejects_empty_request_fingerprint() -> None:
    """``request_fingerprint`` is required (min_length=1). An empty
    fingerprint would let two distinct queries collide on the
    same cache key."""
    with pytest.raises(ValidationError, match="request_fingerprint"):
        QueryCacheEnvelope(
            corpus_revision=4,
            embedding_digest="embed-v1-digest",
            request_fingerprint="",
        )


def test_query_cache_envelope_rejects_negative_corpus_revision() -> None:
    """``corpus_revision`` is bounded to ``>= 0`` (matches D1 and
    the response envelope)."""
    with pytest.raises(ValidationError, match="corpus_revision"):
        QueryCacheEnvelope(
            corpus_revision=-1,
            embedding_digest="embed-v1-digest",
            request_fingerprint="query-fingerprint",
        )


def test_query_cache_envelope_extra_forbid_rejects_unknown_field() -> None:
    """``extra="forbid"`` on ``QueryCacheEnvelope`` rejects unknown
    fields. The cache envelope is a best-effort lookup; unknown
    fields would let attackers inject stale cache entries."""
    with pytest.raises(ValidationError, match="unknown_field"):
        QueryCacheEnvelope(  # type: ignore[call-arg]
            corpus_revision=4,
            embedding_digest="embed-v1-digest",
            request_fingerprint="query-fingerprint",
            unknown_field="oops",
        )
