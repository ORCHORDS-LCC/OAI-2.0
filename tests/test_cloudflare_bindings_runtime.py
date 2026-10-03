from __future__ import annotations

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
    def __init__(self) -> None:
        self.upserts: list[list[dict[str, object]]] = []
        self.queries: list[tuple[list[float], dict[str, object] | None]] = []
        self.upsert_result: object = {"mutationId": "m1"}
        self.query_result: object = {
            "matches": [
                {
                    "id": "oai2v1-" + "a" * 56,
                    "score": 0.95,
                    "metadata": _META_A,
                },
                {
                    "id": "oai2v1-" + "b" * 56,
                    "score": 0.75,
                    "metadata": _META_B,
                },
            ]
        }

    async def upsert(self, vectors: object) -> object:
        assert isinstance(vectors, list)
        self.upserts.append(vectors)
        return self.upsert_result

    async def query(self, vector: object, options: object = None) -> object:
        assert isinstance(vector, list)
        assert options is None or isinstance(options, dict)
        self.queries.append((vector, options))
        return self.query_result


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
async def test_vectorize_query_fails_closed_on_unattributable_matches() -> None:
    """A match that cannot be attributed is a dependency fault, not a skip.

    Skipping it would be indistinguishable from "no results", which is how a
    broken index turns into a quiet empty answer instead of an error.
    """
    store = CloudflareVectorizeStore(FakeVectorize())
    bad_cases = {
        "missing id": {"score": 0.9, "metadata": _META_A},
        "empty id": {"id": "", "score": 0.9, "metadata": _META_A},
        "padded id": {"id": " oai2v1-x ", "score": 0.9, "metadata": _META_A},
        "non-string id": {"id": 7, "score": 0.9, "metadata": _META_A},
        "over-long id": {
            "id": "oai2v1-" + "z" * 64,
            "score": 0.9,
            "metadata": _META_A,
        },
        "nan score": {"id": "oai2v1-" + "a" * 56, "score": float("nan"),
                      "metadata": _META_A},
        "inf score": {"id": "oai2v1-" + "a" * 56, "score": float("inf"),
                      "metadata": _META_A},
        "bool score": {"id": "oai2v1-" + "a" * 56, "score": True,
                       "metadata": _META_A},
        "string score": {"id": "oai2v1-" + "a" * 56, "score": "0.9",
                         "metadata": _META_A},
        "no metadata": {"id": "oai2v1-" + "a" * 56, "score": 0.9},
        "null metadata": {"id": "oai2v1-" + "a" * 56, "score": 0.9,
                          "metadata": None},
        "metadata not a mapping": {"id": "oai2v1-" + "a" * 56, "score": 0.9,
                                   "metadata": "nope"},
    }
    checked: list[str] = []
    for label, match in bad_cases.items():
        index = FakeVectorize()
        index.query_result = {"matches": [match]}
        store = CloudflareVectorizeStore(index)
        with pytest.raises(RuntimeError, match="could not be validated"):
            await store.query([1.0], top_k=1)
        checked.append(label)
    # Proves every listed case actually ran rather than being shadowed.
    assert len(checked) == len(bad_cases)


@pytest.mark.asyncio
async def test_vectorize_query_fails_closed_on_incomplete_metadata() -> None:
    """Each attribution field is required; a partial one is not usable."""
    for field in ("knowledge_id", "content_hash", "embedding_version"):
        meta = {k: v for k, v in _META_A.items() if k != field}
        index = FakeVectorize()
        index.query_result = {
            "matches": [{"id": "oai2v1-" + "a" * 56, "score": 0.9, "metadata": meta}]
        }
        store = CloudflareVectorizeStore(index)
        with pytest.raises(RuntimeError, match="could not be validated"):
            await store.query([1.0], top_k=1)


@pytest.mark.asyncio
async def test_vectorize_query_fails_closed_on_invalid_metadata_values() -> None:
    """Present-but-unusable is not better than absent; both are rejected."""
    bad_values = {
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
        "empty embedding_version": {"knowledge_id": "ko_a", "content_hash": _HASH_A,
                                    "embedding_version": ""},
        "short content_hash": {"knowledge_id": "ko_a", "content_hash": "abc",
                               "embedding_version": "embed-v1"},
        "non-string knowledge_id": {"knowledge_id": 5, "content_hash": _HASH_A,
                                    "embedding_version": "embed-v1"},
    }
    checked: list[str] = []
    for label, meta in bad_values.items():
        index = FakeVectorize()
        index.query_result = {
            "matches": [{"id": "oai2v1-" + "a" * 56, "score": 0.9, "metadata": meta}]
        }
        store = CloudflareVectorizeStore(index)
        with pytest.raises(RuntimeError, match="could not be validated"):
            await store.query([1.0], top_k=1)
        checked.append(label)
    assert len(checked) == len(bad_values)


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
        "matches": [
            {
                "id": "oai2v1-" + "a" * 56,
                "score": float("nan"),
                "metadata": _META_A,
            }
        ]
    }
    with pytest.raises(RuntimeError, match="non-finite or non-numeric score"):
        await store.query([1.0], top_k=1)


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
