from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass

import pytest

from oai2.knowledge.cloudflare_bindings_runtime import (
    CloudflareKvCache,
    CloudflareR2Store,
    CloudflareVectorizeStore,
)


@dataclass
class FakeR2Object:
    value: object

    async def text(self) -> str:
        return self.value  # type: ignore[return-value]


class FakeR2Bucket:
    def __init__(self) -> None:
        self.objects: dict[str, str] = {}
        self.return_none_on_put = False
        self.deleted: list[str] = []

    async def get(self, key: str) -> FakeR2Object | None:
        value = self.objects.get(key)
        return None if value is None else FakeR2Object(value)

    async def head(self, key: str) -> object | None:
        return object() if key in self.objects else None

    async def put(self, key: str, value: str) -> object | None:
        if self.return_none_on_put:
            return None
        self.objects[key] = value
        return object()

    async def delete(self, key: str) -> None:
        self.deleted.append(key)
        self.objects.pop(key, None)


_HASH_A = "a" * 64
_HASH_B = "b" * 64
_META_A = {
    "knowledge_id": "ko_a",
    "content_hash": _HASH_A,
    "embedding_version": "embed-v1",
}
_META_B = {
    "knowledge_id": "ko_b",
    "content_hash": _HASH_B,
    "embedding_version": "embed-v1",
}


class FakeVectorize:
    """Models the documented Vectorize binding, not a convenient one.

    The important detail: a nearest-neighbour query does NOT return metadata
    unless it asks for it. The documented default of ``returnMetadata`` is
    ``"none"``, so a default query match carries only ``id`` and ``score``.

    A fake that always attaches metadata hides the fact that the read contract
    depends on an attribution channel the query never opened. Modelling the
    default honestly is what surfaced that.
    """

    def __init__(self) -> None:
        self.upserts: list[list[dict[str, object]]] = []
        self.queries: list[tuple[list[float], dict[str, object] | None]] = []
        self.get_by_ids_calls: list[list[str]] = []
        self.upsert_result: object = {"mutationId": "m1"}
        # Metadata per vector id, as stored by upsert / as the index would hold it.
        self.stored: dict[str, dict[str, object]] = {
            "oai2v1-" + "a" * 56: dict(_META_A),
            "oai2v1-" + "b" * 56: dict(_META_B),
        }
        self.query_result: object = {
            "matches": [
                {"id": "oai2v1-" + "a" * 56, "score": 0.95},
                {"id": "oai2v1-" + "b" * 56, "score": 0.75},
            ]
        }
        # getByIds results, keyed by requested id. Overridable per test.
        self.get_by_ids_result: object = None
        self.get_by_ids_override: object = "__unset__"

    async def upsert(self, vectors: object) -> object:
        assert isinstance(vectors, list)
        for v in vectors:
            vid = v.get("id")
            meta = v.get("metadata")
            if isinstance(vid, str) and isinstance(meta, dict):
                self.stored[vid] = dict(meta)
        self.upserts.append(vectors)
        return self.upsert_result

    async def query(self, vector: object, options: object = None) -> object:
        assert isinstance(vector, list)
        assert options is None or isinstance(options, dict)
        self.queries.append((vector, options))
        if isinstance(self.query_result, Mapping) and "matches" in self.query_result:
            mode = "none" if not options else options.get("returnMetadata", "none")
            if mode == "all":
                # Attach metadata, as the binding would when asked.
                enriched = []
                for m in self.query_result["matches"]:
                    m = dict(m)
                    if isinstance(m.get("id"), str):
                        m["metadata"] = self.stored.get(m["id"])
                    enriched.append(m)
                return {"matches": enriched}
        return self.query_result

    async def getByIds(self, ids: object) -> object:
        assert isinstance(ids, list)
        self.get_by_ids_calls.append(list(ids))
        if self.get_by_ids_override != "__unset__":
            return self.get_by_ids_override
        out = []
        for vid in ids:
            meta = self.stored.get(vid)
            if meta is not None:
                out.append({"id": vid, "values": [0.1, 0.2], "metadata": dict(meta)})
        return out


class FakeKv:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}
        self.puts: list[tuple[str, str, int | None]] = []

    async def get(self, key: str) -> object | None:
        return self.values.get(key)

    async def put(
        self,
        key: str,
        value: str,
        *,
        expirationTtl: int | None = None,
    ) -> None:
        self.values[key] = value
        self.puts.append((key, value, expirationTtl))


@pytest.mark.asyncio
async def test_r2_text_round_trip_exists_and_delete() -> None:
    bucket = FakeR2Bucket()
    store = CloudflareR2Store(bucket)

    await store.put_text("oai2-blobs/a", "hello")
    assert await store.exists("oai2-blobs/a") is True
    assert await store.get_text("oai2-blobs/a") == "hello"

    await store.delete("oai2-blobs/a")
    assert await store.exists("oai2-blobs/a") is False
    assert await store.get_text("oai2-blobs/a") is None
    assert bucket.deleted == ["oai2-blobs/a"]


@pytest.mark.asyncio
async def test_r2_put_fails_closed_on_unexpected_none_result() -> None:
    bucket = FakeR2Bucket()
    bucket.return_none_on_put = True
    store = CloudflareR2Store(bucket)

    with pytest.raises(RuntimeError, match="R2 put returned no object result"):
        await store.put_text("oai2-blobs/a", "hello")


@pytest.mark.asyncio
async def test_r2_get_fails_closed_on_non_string_text_result() -> None:
    bucket = FakeR2Bucket()
    bucket.objects["oai2-blobs/a"] = "placeholder"
    store = CloudflareR2Store(bucket)

    original_get = bucket.get

    async def bad_get(key: str) -> FakeR2Object | None:
        if key == "oai2-blobs/a":
            return FakeR2Object(123)
        return await original_get(key)

    bucket.get = bad_get  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="non-string"):
        await store.get_text("oai2-blobs/a")


@pytest.mark.asyncio
async def test_kv_text_round_trip_without_ttl() -> None:
    kv = FakeKv()
    cache = CloudflareKvCache(kv)

    await cache.put_text("q:key", "value")
    assert await cache.get_text("q:key") == "value"
    assert kv.puts == [("q:key", "value", None)]


@pytest.mark.asyncio
async def test_kv_put_uses_cloudflare_expiration_ttl_keyword() -> None:
    kv = FakeKv()
    cache = CloudflareKvCache(kv)

    await cache.put_text("q:key", "value", ttl_seconds=300)
    assert kv.puts == [("q:key", "value", 300)]


@pytest.mark.asyncio
async def test_kv_ttl_rejects_values_below_cloudflare_minimum() -> None:
    kv = FakeKv()
    cache = CloudflareKvCache(kv)

    with pytest.raises(ValueError, match=">= 60"):
        await cache.put_text("q:key", "value", ttl_seconds=59)


@pytest.mark.asyncio
async def test_kv_get_fails_closed_on_non_string_value() -> None:
    kv = FakeKv()
    kv.values["q:key"] = 123
    cache = CloudflareKvCache(kv)

    with pytest.raises(RuntimeError, match="non-string"):
        await cache.get_text("q:key")



@pytest.mark.asyncio
async def test_vectorize_upsert_uses_documented_vector_shape() -> None:
    index = FakeVectorize()
    store = CloudflareVectorizeStore(index)

    result = await store.upsert(
        "ko_1",
        [1.0, 2, 3.5],
        metadata={"content_hash": "abc"},
    )

    assert result == {"mutationId": "m1"}
    assert index.upserts == [
        [
            {
                "id": "ko_1",
                "values": [1.0, 2.0, 3.5],
                "metadata": {"content_hash": "abc"},
            }
        ]
    ]


@pytest.mark.asyncio
async def test_vectorize_query_returns_fully_attributed_matches() -> None:
    """The metadata is carried, not discarded.

    Returning bare (id, score) pairs is what made a mixed-generation answer
    expressible: with no knowledge_id, content_hash or embedding_version on the
    match, a caller cannot tell which object a score belongs to. The read
    contract is only enforceable because the attribution arrives here.
    """
    index = FakeVectorize()
    store = CloudflareVectorizeStore(index)

    result = await store.query([1.0, 2.0, 3.0], top_k=2)

    assert [m.vector_id for m in result] == [
        "oai2v1-" + "a" * 56,
        "oai2v1-" + "b" * 56,
    ]
    assert [m.score for m in result] == [0.95, 0.75]
    assert [m.knowledge_id for m in result] == ["ko_a", "ko_b"]
    assert [m.content_hash for m in result] == [_HASH_A, _HASH_B]
    assert [m.embedding_version for m in result] == ["embed-v1", "embed-v1"]
    assert index.queries == [([1.0, 2.0, 3.0], {"topK": 2})]


@pytest.mark.asyncio
async def test_ranking_channel_fails_closed_on_unusable_ids_and_scores() -> None:
    """A match the RANKING query returned unusably is a fault, not a skip.

    These are detected before attribution is requested, so a bad rank never
    costs a second binding call.
    """
    bad_matches = {
        "missing id": {"score": 0.9},
        "empty id": {"id": "", "score": 0.9},
        "padded id": {"id": " oai2v1-x ", "score": 0.9},
        "over-long id": {"id": "oai2v1-" + "z" * 64, "score": 0.9},
        "non-string id": {"id": 7, "score": 0.9},
        "nan score": {"id": "oai2v1-" + "a" * 56, "score": float("nan")},
        "inf score": {"id": "oai2v1-" + "a" * 56, "score": float("inf")},
        "bool score": {"id": "oai2v1-" + "a" * 56, "score": True},
        "string score": {"id": "oai2v1-" + "a" * 56, "score": "0.9"},
        "missing score": {"id": "oai2v1-" + "a" * 56},
    }
    checked: list[str] = []
    for label, match in bad_matches.items():
        index = FakeVectorize()
        index.query_result = {"matches": [match]}
        store = CloudflareVectorizeStore(index)
        with pytest.raises(RuntimeError):
            await store.query([1.0], top_k=1)
        assert index.get_by_ids_calls == [], f"{label}: attribution was requested anyway"
        checked.append(label)
    assert len(checked) == len(bad_matches)


@pytest.mark.asyncio
async def test_attribution_channel_fails_closed_on_unusable_metadata() -> None:
    """A record whose metadata cannot be attributed fails the whole read.

    Skipping it would shorten the result list, and a short list is
    indistinguishable from a genuine low-recall answer.
    """
    bad_metadata: dict[str, object] = {
        "no metadata key": "__absent__",
        "null metadata": None,
        "metadata not a mapping": "nope",
        "empty knowledge_id": {"knowledge_id": "", "content_hash": _HASH_A,
                               "embedding_version": "embed-v1"},
        "padded knowledge_id": {"knowledge_id": " ko_a ", "content_hash": _HASH_A,
                                "embedding_version": "embed-v1"},
        "padded content_hash": {"knowledge_id": "ko_a", "content_hash": f" {_HASH_A} ",
                                "embedding_version": "embed-v1"},
        "padded embedding_version": {"knowledge_id": "ko_a", "content_hash": _HASH_A,
                                     "embedding_version": " embed-v1 "},
        "null knowledge_id": {"knowledge_id": None, "content_hash": _HASH_A,
                              "embedding_version": "embed-v1"},
        "non-string knowledge_id": {"knowledge_id": 5, "content_hash": _HASH_A,
                                    "embedding_version": "embed-v1"},
        "empty embedding_version": {"knowledge_id": "ko_a", "content_hash": _HASH_A,
                                    "embedding_version": ""},
        "short content_hash": {"knowledge_id": "ko_a", "content_hash": "abc",
                               "embedding_version": "embed-v1"},
    }
    checked: list[str] = []
    for label, meta in bad_metadata.items():
        record: dict[str, object] = {"id": "oai2v1-" + "a" * 56}
        if meta != "__absent__":
            record["metadata"] = meta
        index = FakeVectorize()
        index.get_by_ids_override = [record]
        store = CloudflareVectorizeStore(index)
        with pytest.raises(RuntimeError):
            await store.query([1.0], top_k=1)
        checked.append(label)
    assert len(checked) == len(bad_metadata)


@pytest.mark.asyncio
async def test_vector_match_carries_no_private_or_body_material() -> None:
    """The read contract is public-safe by construction."""
    store = CloudflareVectorizeStore(FakeVectorize())
    match = (await store.query([1.0], top_k=1))[0]
    assert set(type(match).model_fields) == {
        "vector_id",
        "score",
        "knowledge_id",
        "content_hash",
        "embedding_version",
    }
    # Values are drawn only from the three attribution fields; nothing that
    # names storage, credentials or topology can appear.
    blob = match.model_dump_json()
    for forbidden in ("r2_blob", "blob_key", "source_uri", "topic",
                      "token", "secret", "account", "bucket", "index",
                      "namespace", "endpoint"):
        assert forbidden not in blob


@pytest.mark.asyncio
async def test_vectorize_rejects_invalid_vectors_topk_and_matches() -> None:
    index = FakeVectorize()
    store = CloudflareVectorizeStore(index)

    with pytest.raises(ValueError, match="non-empty"):
        await store.query([], top_k=2)

    with pytest.raises(ValueError, match="between 1 and 100"):
        await store.query([1.0], top_k=101)

    index.query_result = {
        "matches": [{"id": "oai2v1-" + "a" * 56, "score": float("nan")}]
    }
    with pytest.raises(RuntimeError, match="invalid score"):
        await store.query([1.0], top_k=1)
    assert index.get_by_ids_calls == [], "a bad rank must not spend a second call"


@pytest.mark.asyncio
async def test_vectorize_upsert_requires_mutation_result() -> None:
    index = FakeVectorize()
    index.upsert_result = None
    store = CloudflareVectorizeStore(index)

    with pytest.raises(RuntimeError, match="no mutation result"):
        await store.upsert("ko_1", [1.0])


def test_r2_and_kv_wrappers_export_from_knowledge_package() -> None:
    from oai2.knowledge import CloudflareKvCache as ExportedKv
    from oai2.knowledge import CloudflareR2Store as ExportedR2
    from oai2.knowledge import CloudflareVectorizeStore as ExportedVectorize
    from oai2.knowledge.cloudflare_bindings_runtime import (
        CloudflareKvCache,
        CloudflareR2Store,
        CloudflareVectorizeStore,
    )

    assert ExportedKv is CloudflareKvCache
    assert ExportedR2 is CloudflareR2Store
    assert ExportedVectorize is CloudflareVectorizeStore


def test_r2_kv_binding_protocol_exports_from_knowledge_package() -> None:
    from oai2.knowledge import KvNamespaceBinding as ExportedKvBinding
    from oai2.knowledge import R2BucketBinding as ExportedR2BucketBinding
    from oai2.knowledge import (
        R2ObjectBodyBinding as ExportedR2ObjectBodyBinding,
    )
    from oai2.knowledge.cloudflare_bindings_runtime import (
        KvNamespaceBinding,
        R2BucketBinding,
        R2ObjectBodyBinding,
    )

    assert ExportedR2BucketBinding is R2BucketBinding
    assert ExportedR2ObjectBodyBinding is R2ObjectBodyBinding
    assert ExportedKvBinding is KvNamespaceBinding


def test_vectorize_binding_protocol_exports_from_knowledge_package() -> None:
    from oai2.knowledge import VectorizeIndexBinding as ExportedVectorizeBinding
    from oai2.knowledge.cloudflare_bindings_runtime import (
        VectorizeIndexBinding,
    )

    assert ExportedVectorizeBinding is VectorizeIndexBinding


# ---------------------------------------------------------------------------
# #19 — the adapter must match the DOCUMENTED binding response shape.
#
# The first version of the typed read contract required `match.metadata` on a
# nearest-neighbour result. That is fine for a fake that always attaches
# metadata, and impossible against the real binding, whose query default is
# returnMetadata:"none". The defect was invisible until the fake was made to
# model the documented default.
# ---------------------------------------------------------------------------


def test_a_default_vectorize_query_returns_id_and_score_only() -> None:
    """A. The documented query default does not carry metadata.

    If this ever stops being true the adapter's two-call design can be
    revisited, but the assertion documents WHY it is two calls.
    """
    index = FakeVectorize()
    result = asyncio.run(index.query([1.0], {"topK": 2}))
    for match in result["matches"]:
        assert set(match) == {"id", "score"}, (
            f"default query must not carry metadata, got {sorted(match)}"
        )
    # ...and asking for it does return it, which is the alternative design.
    result = asyncio.run(index.query([1.0], {"topK": 2, "returnMetadata": "all"}))
    for match in result["matches"]:
        assert "metadata" in match


@pytest.mark.asyncio
async def test_b_query_makes_one_query_call_and_one_attribution_call() -> None:
    """B. Exactly one ranking call, then one getByIds attribution call."""
    index = FakeVectorize()
    store = CloudflareVectorizeStore(index)

    result = await store.query([1.0, 2.0, 3.0], top_k=2)

    assert len(result) == 2
    assert len(index.queries) == 1, "ranking must be a single query call"
    assert index.get_by_ids_calls == [
        ["oai2v1-" + "a" * 56, "oai2v1-" + "b" * 56],
    ], "one attribution call carrying exactly the ranked ids"


@pytest.mark.asyncio
async def test_c_reversed_attribution_results_preserve_query_ranking() -> None:
    """C. getByIds order is never assumed."""
    index = FakeVectorize()
    index.get_by_ids_override = [
        {"id": "oai2v1-" + "b" * 56, "metadata": _META_B},
        {"id": "oai2v1-" + "a" * 56, "metadata": _META_A},
    ]
    store = CloudflareVectorizeStore(index)

    result = await store.query([1.0], top_k=2)

    assert [m.vector_id for m in result] == [
        "oai2v1-" + "a" * 56,
        "oai2v1-" + "b" * 56,
    ], "attribution order leaked into ranking"
    assert [m.score for m in result] == [0.95, 0.75]


@pytest.mark.asyncio
async def test_d_explicit_ranking_case_keeps_score_and_order() -> None:
    """D. V1 .99 then V2 .80 ranks; getByIds returns V2 first; order is V1, V2."""
    index = FakeVectorize()
    index.query_result = {
        "matches": [
            {"id": "oai2v1-" + "a" * 56, "score": 0.99},
            {"id": "oai2v1-" + "b" * 56, "score": 0.80},
        ]
    }
    index.get_by_ids_override = [
        {"id": "oai2v1-" + "b" * 56, "metadata": _META_B},
        {"id": "oai2v1-" + "a" * 56, "metadata": _META_A},
    ]
    store = CloudflareVectorizeStore(index)

    result = await store.query([1.0], top_k=2)

    assert [(m.vector_id[-1], m.score) for m in result] == [("a", 0.99), ("b", 0.80)]
    assert [m.knowledge_id for m in result] == ["ko_a", "ko_b"], (
        "attribution was joined positionally instead of by id"
    )


@pytest.mark.asyncio
async def test_e_missing_attribution_fails_closed() -> None:
    """E. A ranked id with no metadata is a dependency failure, not a short list."""
    index = FakeVectorize()
    index.get_by_ids_override = [{"id": "oai2v1-" + "b" * 56, "metadata": _META_B}]
    store = CloudflareVectorizeStore(index)

    with pytest.raises(RuntimeError, match="no metadata for queried vector ids"):
        await store.query([1.0], top_k=2)


@pytest.mark.asyncio
async def test_f_duplicate_attribution_fails_closed() -> None:
    """F. Two records for one ranked id is ambiguous, not a merge."""
    index = FakeVectorize()
    index.get_by_ids_override = [
        {"id": "oai2v1-" + "a" * 56, "metadata": _META_A},
        {"id": "oai2v1-" + "a" * 56, "metadata": _META_B},
        {"id": "oai2v1-" + "b" * 56, "metadata": _META_B},
    ]
    store = CloudflareVectorizeStore(index)

    with pytest.raises(RuntimeError, match="duplicate attribution"):
        await store.query([1.0], top_k=2)


@pytest.mark.asyncio
async def test_g_unrequested_attribution_is_never_promoted() -> None:
    """G. A volunteered record for an id that did not rank stays out."""
    index = FakeVectorize()
    index.get_by_ids_override = [
        {"id": "oai2v1-" + "a" * 56, "metadata": _META_A},
        {"id": "oai2v1-" + "b" * 56, "metadata": _META_B},
        {"id": "oai2v1-" + "c" * 56, "metadata": _META_B},
    ]
    store = CloudflareVectorizeStore(index)

    result = await store.query([1.0], top_k=2)

    assert [m.vector_id for m in result] == [
        "oai2v1-" + "a" * 56,
        "oai2v1-" + "b" * 56,
    ], "an id that never ranked was promoted into the result"


@pytest.mark.asyncio
async def test_h_bad_metadata_fails_the_whole_attribution() -> None:
    """H. One unusable record must not degrade into fewer results."""
    index = FakeVectorize()
    index.get_by_ids_override = [
        {"id": "oai2v1-" + "a" * 56, "metadata": _META_A},
        {"id": "oai2v1-" + "b" * 56, "metadata": {"knowledge_id": "ko_b"}},
    ]
    store = CloudflareVectorizeStore(index)

    with pytest.raises(RuntimeError, match="could not be validated"):
        await store.query([1.0], top_k=2)


@pytest.mark.asyncio
async def test_i_top_k_ceiling_is_preserved_at_100() -> None:
    """I. The ranking query keeps the full documented ceiling."""
    index = FakeVectorize()
    store = CloudflareVectorizeStore(index)

    await store.query([1.0], top_k=100)

    assert index.queries[0][1] == {"topK": 100}
    with pytest.raises(ValueError, match="between 1 and 100"):
        await store.query([1.0], top_k=101)


@pytest.mark.asyncio
async def test_j_ranking_query_never_asks_for_metadata() -> None:
    """J. returnMetadata must not appear: it would halve the topK ceiling.

    Requesting `returnMetadata:"all"` on a nearest-neighbour query lowers the
    documented topK ceiling from 100 to 50. This runtime over-fetches before
    authoritative D1 filtering, so that would cut recall exactly when a stale
    vector is occupying slots. The attribution channel is getByIds.
    """
    index = FakeVectorize()
    store = CloudflareVectorizeStore(index)

    await store.query([1.0], top_k=64)

    options = index.queries[0][1]
    assert "returnMetadata" not in options
    assert "returnValues" not in options
    assert options == {"topK": 64}


@pytest.mark.asyncio
async def test_empty_query_makes_no_attribution_call() -> None:
    """No ranked ids means nothing to attribute, and no binding call to spend."""
    index = FakeVectorize()
    index.query_result = {"matches": []}
    store = CloudflareVectorizeStore(index)

    assert await store.query([1.0], top_k=10) == []
    assert index.get_by_ids_calls == []


@pytest.mark.asyncio
async def test_duplicate_ranked_ids_fail_closed() -> None:
    """One rank per vector: a repeated id would double-attribute one object."""
    index = FakeVectorize()
    index.query_result = {
        "matches": [
            {"id": "oai2v1-" + "a" * 56, "score": 0.9},
            {"id": "oai2v1-" + "a" * 56, "score": 0.8},
        ]
    }
    store = CloudflareVectorizeStore(index)

    with pytest.raises(RuntimeError, match="duplicate id"):
        await store.query([1.0], top_k=2)


@pytest.mark.asyncio
async def test_vector_values_from_get_by_ids_are_never_exposed() -> None:
    """getByIds returns stored vectors; only their metadata is ever read."""
    index = FakeVectorize()
    store = CloudflareVectorizeStore(index)

    result = await store.query([1.0], top_k=2)

    for match in result:
        assert "values" not in type(match).model_fields
        assert "0.1" not in match.model_dump_json()
    # The binding call still happened, and the vectors were simply ignored.
    assert index.get_by_ids_calls == [
        ["oai2v1-" + "a" * 56, "oai2v1-" + "b" * 56],
    ]
