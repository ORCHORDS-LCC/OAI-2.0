"""Async Cloudflare Worker binding wrappers for R2 bodies and KV cache.

The wrappers mirror only documented Python Worker binding operations. They are
deployment-neutral and contain no account IDs, bucket names, namespace IDs,
endpoints, or credentials.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Protocol

from .transport import VECTOR_ID_MAX_BYTES, VectorMatch


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

    async def getByIds(self, ids: Sequence[str]) -> object: ...


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
        """Return typed, fully attributed matches, in query rank order.

        WHY THIS IS TWO BINDING CALLS. A nearest-neighbour query does not
        return metadata unless it asks for it: the documented default of
        ``returnMetadata`` is ``"none"``, and a default-mode match carries only
        ``id`` and ``score``. The read contract genuinely needs the
        knowledge_id, content_hash and embedding_version that sit on the vector
        record, because a bare score cannot be attributed to an object.

        Asking the query for metadata is the obvious alternative and is wrong
        here for a concrete reason: requesting ``returnMetadata:"all"`` lowers
        the documented ``topK`` ceiling from 100 to 50. This runtime deliberately
        over-fetches before authoritative D1 filtering, so halving the ceiling
        would cut recall exactly where a stale or ineligible vector is occupying
        the slots. So the ranking query keeps the full ceiling and the
        attribution arrives through ``getByIds``, which is the documented way to
        retrieve stored vectors including their metadata.

        ORDER AND JOIN. The query's score order is the ranking and is preserved
        exactly. ``getByIds`` result order is never assumed; attribution is
        joined by exact vector id into a map first.

        FAIL-CLOSED. A requested id with no attribution, or with two, is a
        dependency fault and raises. Silently dropping either would turn a
        broken attribution channel into a short result list, and a short result
        list is indistinguishable from a genuine low-recall answer. The vector
        ``values`` returned by ``getByIds`` are never read and never exposed.
        """
        normalized = _vector(values)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
            raise ValueError("top_k must be an integer between 1 and 100")

        # No returnMetadata option is sent. The ranking query must keep the full
        # topK ceiling, and a test asserts this option is never added.
        result = await self._index.query(normalized, {"topK": top_k})
        matches = _field(result, "matches")
        if not isinstance(matches, Sequence) or isinstance(
            matches, (str, bytes, bytearray)
        ):
            raise RuntimeError("Vectorize query result does not expose a matches sequence")

        ranked: list[tuple[str, float]] = []
        seen: set[str] = set()
        for match in matches:
            vector_id = _field(match, "id")
            score = _field(match, "score")
            if (
                not isinstance(vector_id, str)
                or not vector_id
                or vector_id != vector_id.strip()
            ):
                # Validated for normalization HERE, not later: this id is about
                # to be used as the getByIds lookup key, so a padded value would
                # be sent to the binding and come back missing, turning a
                # malformed rank into a misleading "no attribution" error.
                raise RuntimeError("Vectorize match has an invalid id")
            if len(vector_id.encode("utf-8")) > VECTOR_ID_MAX_BYTES:
                raise RuntimeError(
                    "Vectorize match id exceeds the Vectorize id length limit"
                )
            if vector_id in seen:
                # One rank per vector. A repeated id would otherwise be
                # attributed twice and could yield the same object twice.
                raise RuntimeError(
                    f"Vectorize query returned duplicate id {vector_id!r}"
                )
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
            ):
                raise RuntimeError("Vectorize match has an invalid score")
            seen.add(vector_id)
            ranked.append((vector_id, float(score)))

        if not ranked:
            # Nothing ranked, so there is nothing to attribute and no reason to
            # spend a binding call.
            return []

        attribution = await self._metadata_by_ids([vid for vid, _ in ranked])

        out: list[VectorMatch] = []
        for vector_id, score in ranked:
            metadata = attribution[vector_id]
            try:
                out.append(
                    VectorMatch.from_vectorize_parts(vector_id, score, metadata)
                )
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Vectorize match could not be validated: {exc}"
                ) from exc
        return out

    async def _metadata_by_ids(self, ids: Sequence[str]) -> dict[str, object]:
        """Fetch stored attribution for exactly these vector ids.

        Returns a map keyed by the requested ids. Ids the binding returns that
        were NOT requested are ignored: an index that volunteers a record must
        never be able to promote a vector that did not rank.
        """
        requested = set(ids)
        result = await self._index.getByIds(list(ids))
        if not isinstance(result, Sequence) or isinstance(
            result, (str, bytes, bytearray)
        ):
            raise RuntimeError("Vectorize getByIds did not return a sequence")

        attribution: dict[str, object] = {}
        for record in result:
            record_id = _field(record, "id")
            if (
                not isinstance(record_id, str)
                or not record_id
                or record_id != record_id.strip()
            ):
                raise RuntimeError("Vectorize getByIds record has an invalid id")
            if record_id not in requested:
                # Extra, unrequested. Never promoted, never a candidate.
                continue
            if record_id in attribution:
                raise RuntimeError(
                    f"Vectorize getByIds returned duplicate attribution for {record_id!r}"
                )
            attribution[record_id] = _field(record, "metadata")

        missing = requested - attribution.keys()
        if missing:
            raise RuntimeError(
                "Vectorize getByIds returned no metadata for queried vector ids: "
                f"{sorted(missing)!r}"
            )
        return attribution


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
