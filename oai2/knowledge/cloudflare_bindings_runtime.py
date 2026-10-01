"""Async Cloudflare Worker binding wrappers for R2 bodies and KV cache.

The wrappers mirror only documented Python Worker binding operations. They are
deployment-neutral and contain no account IDs, bucket names, namespace IDs,
endpoints, or credentials.
"""

from __future__ import annotations

from typing import Protocol


class R2ObjectBodyBinding(Protocol):
    async def text(self) -> str: ...


class R2BucketBinding(Protocol):
    async def get(self, key: str) -> R2ObjectBodyBinding | None: ...

    async def head(self, key: str) -> object | None: ...

    async def put(self, key: str, value: str) -> object | None: ...

    async def delete(self, key: str) -> None: ...


class KvNamespaceBinding(Protocol):
    async def get(self, key: str) -> object | None: ...

    async def put(
        self,
        key: str,
        value: str,
        *,
        expirationTtl: int | None = None,
    ) -> None: ...


class CloudflareR2Store:
    """UTF-8 text-body adapter over a bound Cloudflare R2 bucket."""

    def __init__(self, bucket: R2BucketBinding) -> None:
        self._bucket = bucket

    async def put_text(self, key: str, value: str) -> None:
        key = _normalized(key, "key")
        if not isinstance(value, str):
            raise ValueError("value must be a string")
        result = await self._bucket.put(key, value)
        if result is None:
            raise RuntimeError("R2 put returned no object result")

    async def get_text(self, key: str) -> str | None:
        key = _normalized(key, "key")
        obj = await self._bucket.get(key)
        if obj is None:
            return None
        value = await obj.text()
        if not isinstance(value, str):
            raise RuntimeError("R2 object text() returned a non-string value")
        return value

    async def exists(self, key: str) -> bool:
        key = _normalized(key, "key")
        return await self._bucket.head(key) is not None

    async def delete(self, key: str) -> None:
        key = _normalized(key, "key")
        await self._bucket.delete(key)


class CloudflareKvCache:
    """Best-effort text cache wrapper over a bound Workers KV namespace."""

    def __init__(self, namespace: KvNamespaceBinding) -> None:
        self._namespace = namespace

    async def get_text(self, key: str) -> str | None:
        key = _normalized(key, "key")
        value = await self._namespace.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise RuntimeError("KV get returned a non-string value")
        return value

    async def put_text(self, key: str, value: str, *, ttl_seconds: int | None = None) -> None:
        key = _normalized(key, "key")
        if not isinstance(value, str):
            raise ValueError("value must be a string")
        if ttl_seconds is None:
            await self._namespace.put(key, value)
            return
        if (
            isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, int)
            or ttl_seconds < 60
        ):
            raise ValueError("ttl_seconds must be an integer >= 60")
        await self._namespace.put(key, value, expirationTtl=ttl_seconds)


def _normalized(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty normalized string")
    return value


__all__ = [
    "R2ObjectBodyBinding",
    "R2BucketBinding",
    "KvNamespaceBinding",
    "CloudflareR2Store",
    "CloudflareKvCache",
]
