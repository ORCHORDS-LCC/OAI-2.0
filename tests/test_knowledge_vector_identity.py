"""Generation-specific vector identity (#19).

The vector id is the mechanism that makes the authoritative D1 row an ATOMIC
switch rather than a description that can drift from the bytes it points at.
These tests pin the properties the rest of the design relies on, rather than
re-testing the runtime paths that consume it (those live in
``test_cloudflare_runtime.py``).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from oai2.core import KnowledgeId
from oai2.knowledge import VECTOR_ID_MAX_BYTES, vector_id_for
from oai2.knowledge.transport import VectorMatch

KID = KnowledgeId("ko_identity")
HASH_A = "a" * 64
HASH_B = "b" * 64
V1 = "embed-v1"
V2 = "embed-v2"


def _vid(
    *,
    knowledge_id: KnowledgeId = KID,
    content_hash: str = HASH_A,
    embedding_version: str = V1,
) -> str:
    return vector_id_for(
        knowledge_id=knowledge_id,
        content_hash=content_hash,
        embedding_version=embedding_version,
    )


def test_vector_id_is_deterministic() -> None:
    # No random and no clock component: a retried put must re-upsert the SAME
    # vector rather than accumulating one orphan per attempt.
    assert _vid() == _vid()
    assert _vid() == vector_id_for(
        knowledge_id=KnowledgeId("ko_identity"),
        content_hash=HASH_A,
        embedding_version="embed-v1",
    )


def test_vector_id_is_never_the_bare_knowledge_id() -> None:
    # The old identity WAS the knowledge_id, which is precisely what let a new
    # generation overwrite the vector a committed row referenced.
    assert _vid() != str(KID)


def test_every_input_participates_in_the_identity() -> None:
    base = _vid()
    assert _vid(knowledge_id=KnowledgeId("ko_other")) != base
    assert _vid(content_hash=HASH_B) != base
    assert _vid(embedding_version=V2) != base


def test_vector_id_fits_the_cloudflare_limit() -> None:
    vector_id = _vid()
    # Measured in BYTES, which is the limit's unit, not characters.
    assert len(vector_id.encode("utf-8")) <= VECTOR_ID_MAX_BYTES
    assert VECTOR_ID_MAX_BYTES == 64


def test_vector_id_is_namespaced_and_ascii() -> None:
    # The prefix is a scheme marker: a future canonicalization change must land
    # in a disjoint id space, not collide with ids written under this scheme.
    vector_id = _vid()
    assert vector_id.startswith("oai2v1-")
    assert vector_id.isascii()


def test_distinct_inputs_give_distinct_ids() -> None:
    # A small collision sweep over the fields that actually vary in practice.
    seen: dict[str, tuple[str, str, str]] = {}
    for i in range(64):
        for version in (V1, V2):
            for content in (HASH_A, HASH_B, f"{i:064x}"):
                key = (f"ko_{i}", content, version)
                vector_id = vector_id_for(
                    knowledge_id=KnowledgeId(key[0]),
                    content_hash=key[1],
                    embedding_version=key[2],
                )
                prior = seen.get(vector_id)
                assert prior is None or prior == key, (
                    f"vector id collision between {prior} and {key}"
                )
                seen[vector_id] = key
    assert len(seen) == 64 * 2 * 3


@pytest.mark.parametrize(
    "label,kwargs",
    [
        ("empty knowledge_id", {"knowledge_id": KnowledgeId(""), "content_hash": HASH_A,
                                 "embedding_version": V1}),
        ("padded knowledge_id", {"knowledge_id": KnowledgeId(" ko_x "),
                                 "content_hash": HASH_A, "embedding_version": V1}),
        ("empty content_hash", {"knowledge_id": KID, "content_hash": "",
                                "embedding_version": V1}),
        ("padded content_hash", {"knowledge_id": KID, "content_hash": f" {HASH_A} ",
                                 "embedding_version": V1}),
        ("empty embedding_version", {"knowledge_id": KID, "content_hash": HASH_A,
                                    "embedding_version": ""}),
        ("padded embedding_version", {"knowledge_id": KID, "content_hash": HASH_A,
                                     "embedding_version": " v1 "}),
        # A line break would make the canonical LF-joined encoding ambiguous,
        # so two different triples could hash to the same byte string.
        ("newline in knowledge_id", {"knowledge_id": KnowledgeId("ko_a\nko_b"),
                                      "content_hash": HASH_A, "embedding_version": V1}),
        ("carriage return in content_hash", {"knowledge_id": KID,
                                             "content_hash": f"{HASH_A}\rX",
                                             "embedding_version": V1}),
        ("newline in embedding_version", {"knowledge_id": KID, "content_hash": HASH_A,
                                          "embedding_version": "v1\nv2"}),
        ("non-string knowledge_id", {"knowledge_id": 5, "content_hash": HASH_A,
                                     "embedding_version": V1}),
        ("non-string embedding_version", {"knowledge_id": KID, "content_hash": HASH_A,
                                          "embedding_version": None}),
    ],
)
def test_vector_id_rejects_unusable_inputs(label: str, kwargs: dict[str, object]) -> None:
    # Fail closed: an id derived from an unnormalized field is not an id.
    with pytest.raises(ValueError):
        vector_id_for(**kwargs)  # type: ignore[arg-type]


def test_the_canonical_encoding_is_unambiguous() -> None:
    # Different triples that a naive concatenation would collapse must not share
    # an id. "ko_a" + "bc" and "ko_ab" + "c" style ambiguity is prevented by the
    # LF separator plus the rejection of embedded line breaks.
    left = _vid(knowledge_id=KnowledgeId("ko_a"), content_hash=HASH_A)
    right = _vid(knowledge_id=KnowledgeId("ko_b"), content_hash=HASH_A)
    assert left != right


def test_vector_match_accepts_a_derived_id_and_rejects_an_over_long_one() -> None:
    good = VectorMatch.from_vectorize_parts(
        _vid(),
        0.5,
        {"knowledge_id": str(KID), "content_hash": HASH_A, "embedding_version": V1},
    )
    assert good.vector_id == _vid()
    assert good.score == 0.5

    with pytest.raises(ValueError, match="length limit"):
        VectorMatch.from_vectorize_parts(
            "oai2v1-" + "z" * 64,
            0.5,
            {"knowledge_id": str(KID), "content_hash": HASH_A, "embedding_version": V1},
        )


def test_vector_match_cannot_be_constructed_with_extra_fields() -> None:
    # A widened binding payload must not smuggle extra state into the read
    # contract that the integrity rules do not inspect.
    with pytest.raises(ValidationError):
        VectorMatch(  # type: ignore[call-arg]
            vector_id=_vid(),
            score=0.5,
            knowledge_id=KID,
            content_hash=HASH_A,
            embedding_version=V1,
            r2_blob_key="oai2-blobs/x",
        )
