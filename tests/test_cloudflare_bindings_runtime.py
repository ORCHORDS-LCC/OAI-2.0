from __future__ import annotations

from dataclasses import dataclass

import pytest

from oai2.knowledge.cloudflare_bindings_runtime import (
    CloudflareKvCache,
    CloudflareR2Store,
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



def test_r2_and_kv_wrappers_export_from_knowledge_package() -> None:
    from oai2.knowledge import CloudflareKvCache as ExportedKv
    from oai2.knowledge import CloudflareR2Store as ExportedR2
    from oai2.knowledge.cloudflare_bindings_runtime import (
        CloudflareKvCache,
        CloudflareR2Store,
    )

    assert ExportedKv is CloudflareKvCache
    assert ExportedR2 is CloudflareR2Store
