"""Async Cloudflare Worker binding wrappers for R2 bodies and KV cache.

The wrappers mirror only documented Python Worker binding operations. They are
deployment-neutral and contain no account IDs, bucket names, namespace IDs,
endpoints, or credentials.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Protocol

from .transport import VectorMatch


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


class VectorizeIndexBinding(Protocol):
    async def upsert(self, vectors: Sequence[Mapping[str, object]]) -> object: ...

    async def query(
        self,
        vector: Sequence[float],
        options: Mapping[str, object] | None = None,
    ) -> object: ...


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


class CloudflareVectorizeStore:
    """Minimal async adapter over a bound Cloudflare Vectorize index.

    Upserts are eventually query-visible; callers must not treat successful
    upsert return as proof that a subsequent query can already observe it.
    """

    def __init__(self, index: VectorizeIndexBinding) -> None:
        self._index = index

    async def upsert(
        self,
        vector_id: str,
        values: Sequence[float],
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> object:
        vid = _normalized(vector_id, "vector_id")
        normalized = _vector(values)
        payload: dict[str, object] = {
            "id": vid,
            "values": normalized,
        }
        if metadata is not None:
            payload["metadata"] = dict(metadata)
        result = await self._index.upsert([payload])
        if result is None:
            raise RuntimeError("Vectorize upsert returned no mutation result")
        return result

    async def query(
        self,
        values: Sequence[float],
        *,
        top_k: int = 5,
    ) -> list[VectorMatch]:
        """Return typed, fully attributed matches.

        The metadata is NOT optional decoration: without the knowledge_id,
        content_hash and embedding_version carried by the same match, a caller
        cannot tell whether the score it is holding belongs to the object it is
        about to return. Discarding it here is what made a mixed-generation
        answer expressible in the first place.

        A match that cannot be parsed is a dependency fault, not a row to skip.
        Raising keeps a broken index from being reported as "no results".
        """
        normalized = _vector(values)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
            raise ValueError("top_k must be an integer between 1 and 100")
        result = await self._index.query(normalized, {"topK": top_k})
        matches = _field(result, "matches")
        if not isinstance(matches, Sequence) or isinstance(
            matches, (str, bytes, bytearray)
        ):
            raise RuntimeError("Vectorize query result does not expose a matches sequence")

        out: list[VectorMatch] = []
        for match in matches:
            try:
                out.append(
                    VectorMatch.from_vectorize_parts(
                        _field(match, "id"),
                        _field(match, "score"),
                        _field(match, "metadata"),
                    )
                )
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Vectorize match could not be validated: {exc}"
                ) from exc
        return out


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


def _field(value: object, name: str) -> object | None:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _vector(values: Sequence[float]) -> list[float]:
    if isinstance(values, (str, bytes, bytearray)) or not values:
        raise ValueError("vector values must be a non-empty numeric sequence")
    out: list[float] = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError("vector values must be finite numbers")
        out.append(float(value))
    return out


def _normalized(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty normalized string")
    return value


__all__ = [
    "R2ObjectBodyBinding",
    "R2BucketBinding",
    "KvNamespaceBinding",
    "VectorizeIndexBinding",
    "CloudflareR2Store",
    "CloudflareVectorizeStore",
    "CloudflareKvCache",
]
