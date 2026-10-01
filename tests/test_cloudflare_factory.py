from __future__ import annotations

from dataclasses import dataclass

import pytest

from oai2.knowledge.cloudflare_factory import (
    CloudflareKnowledgeComponents,
    build_cloudflare_knowledge_components,
)
from oai2.knowledge.gc_lease_d1 import gc_lease_schema_statements
from oai2.knowledge.knowledge_d1 import knowledge_schema_statements


@dataclass
class FakeStatement:
    query: str
    bound: tuple[object, ...] = ()

    def bind(self, *values: object) -> FakeStatement:
        self.bound = values
        return self

    async def run(self) -> object:
        return {"success": True, "meta": {"changes": 0}, "results": []}

    async def first(self, column_name: str | None = None) -> object | None:
        if column_name == "revision":
            return 0
        return None


class FakeD1:
    def __init__(self) -> None:
        self.batch_calls: list[list[FakeStatement]] = []

    def prepare(self, query: str) -> FakeStatement:
        return FakeStatement(query=query)

    async def batch(self, statements: list[FakeStatement]) -> list[object]:
        copied = list(statements)
        self.batch_calls.append(copied)
        return [{"success": True, "meta": {"changes": 0}} for _ in copied]


class FakeR2:
    async def get(self, key: str) -> object | None:
        return None

    async def head(self, key: str) -> object | None:
        return None

    async def put(self, key: str, value: str) -> object | None:
        return object()

    async def delete(self, key: str) -> None:
        return None


class FakeVectorize:
    async def upsert(self, vectors: object) -> object:
        return {"mutationId": "m1"}

    async def query(self, vector: object, options: object = None) -> object:
        return {"matches": []}


class FakeKv:
    async def get(self, key: str) -> object | None:
        return None

    async def put(
        self,
        key: str,
        value: str,
        *,
        expirationTtl: int | None = None,
    ) -> None:
        return None


@pytest.mark.asyncio
async def test_factory_initializes_knowledge_then_gc_schema() -> None:
    d1 = FakeD1()

    components = await build_cloudflare_knowledge_components(
        d1=d1,  # type: ignore[arg-type]
        r2=FakeR2(),  # type: ignore[arg-type]
        vectorize=FakeVectorize(),  # type: ignore[arg-type]
        kv=FakeKv(),  # type: ignore[arg-type]
        embedding_version="embed-v1",
        embedding_digest="digest-v1",
    )

    assert isinstance(components, CloudflareKnowledgeComponents)
    assert len(d1.batch_calls) == 2
    assert [stmt.query for stmt in d1.batch_calls[0]] == list(
        knowledge_schema_statements()
    )
    assert [stmt.query for stmt in d1.batch_calls[1]] == list(
        gc_lease_schema_statements()
    )


@pytest.mark.asyncio
async def test_factory_can_skip_schema_for_pre_migrated_worker() -> None:
    d1 = FakeD1()

    components = await build_cloudflare_knowledge_components(
        d1=d1,  # type: ignore[arg-type]
        r2=FakeR2(),  # type: ignore[arg-type]
        vectorize=FakeVectorize(),  # type: ignore[arg-type]
        kv=FakeKv(),  # type: ignore[arg-type]
        embedding_version="embed-v1",
        embedding_digest="digest-v1",
        ensure_schema=False,
        require_auth=False,
    )

    assert d1.batch_calls == []
    assert components.transport is not None
    assert components.runtime is not None
    assert components.lease_store is not None


@pytest.mark.asyncio
async def test_factory_rejects_non_boolean_schema_flag() -> None:
    with pytest.raises(ValueError, match="ensure_schema"):
        await build_cloudflare_knowledge_components(
            d1=FakeD1(),  # type: ignore[arg-type]
            r2=FakeR2(),  # type: ignore[arg-type]
            vectorize=FakeVectorize(),  # type: ignore[arg-type]
            kv=FakeKv(),  # type: ignore[arg-type]
            embedding_version="embed-v1",
            embedding_digest="digest-v1",
            ensure_schema=1,  # type: ignore[arg-type]
        )



def test_factory_exports_from_knowledge_package() -> None:
    from oai2.knowledge import (
        CloudflareKnowledgeComponents as ExportedComponents,
    )
    from oai2.knowledge import (
        build_cloudflare_knowledge_components as ExportedBuilder,
    )
    from oai2.knowledge.cloudflare_factory import (
        CloudflareKnowledgeComponents,
        build_cloudflare_knowledge_components,
    )

    assert ExportedComponents is CloudflareKnowledgeComponents
    assert ExportedBuilder is build_cloudflare_knowledge_components
